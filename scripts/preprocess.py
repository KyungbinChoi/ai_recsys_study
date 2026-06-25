"""
전처리 파이프라인: k-core 필터링 → ID 인코딩 → leave-one-out split
"""
import json
from pathlib import Path

import duckdb
import pandas as pd

CATEGORY = "All_Beauty"
K_CORE = 3  # All_Beauty는 92%+ 유저가 리뷰 1~2개 → k=5 시 데이터 과소
DATA_DIR = Path(__file__).parent.parent / "data" / "processed" / CATEGORY


def load_reviews(data_dir: Path) -> pd.DataFrame:
    return duckdb.sql(
        f"SELECT user_id, parent_asin, rating, timestamp FROM '{data_dir}/reviews.parquet'"
    ).df()


def kcore_filter(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    유저와 아이템이 각각 최소 k개의 상호작용을 가질 때까지 반복 제거.
    수렴까지 반복하는 이유: 유저를 제거하면 아이템 수가 줄고,
    아이템을 제거하면 다시 유저 수가 줄 수 있기 때문.
    """
    con = duckdb.connect()
    iteration = 0
    while True:
        n_before = len(df)
        con.register("df", df)
        df = con.execute(f"""
            SELECT * FROM df
            WHERE user_id IN (
                SELECT user_id FROM df GROUP BY user_id HAVING COUNT(*) >= {k}
            )
            AND parent_asin IN (
                SELECT parent_asin FROM df GROUP BY parent_asin HAVING COUNT(*) >= {k}
            )
        """).df()
        iteration += 1
        if len(df) == n_before:
            break
    print(f"  k-core 수렴: {iteration}회 반복")
    return df


def encode_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """
    문자열 ID → 0-indexed 정수.
    모델의 Embedding 레이어는 정수 인덱스를 입력받기 때문에 필요.
    """
    users = sorted(df["user_id"].unique())
    items = sorted(df["parent_asin"].unique())
    user2id = {u: i for i, u in enumerate(users)}
    item2id = {it: i for i, it in enumerate(items)}
    df = df.copy()
    df["user_id"] = df["user_id"].map(user2id)
    df["parent_asin"] = df["parent_asin"].map(item2id)
    return df, user2id, item2id


def leave_one_out_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    각 유저의 상호작용을 timestamp 순 정렬 후:
      - 마지막 1개 → test  (next-item prediction 평가 표준)
      - 나머지    → train
    """
    df = df.sort_values(["user_id", "timestamp"])
    test_idx = df.groupby("user_id").tail(1).index
    test = df.loc[test_idx].reset_index(drop=True)
    train = df.drop(test_idx).reset_index(drop=True)
    return train, test


def sparsity(n_interactions: int, n_users: int, n_items: int) -> float:
    return 1.0 - n_interactions / (n_users * n_items)


def run(k: int = K_CORE, data_dir: Path = DATA_DIR) -> dict:
    print("=" * 50)
    print(f"전처리 파이프라인 시작 (k={k})")
    print("=" * 50)

    # 1. 로드
    print("\n[1/4] 데이터 로드")
    df = load_reviews(data_dir)
    n_users_raw = df["user_id"].nunique()
    n_items_raw = df["parent_asin"].nunique()
    print(f"  interactions : {len(df):,}")
    print(f"  users        : {n_users_raw:,}")
    print(f"  items        : {n_items_raw:,}")
    print(f"  sparsity     : {sparsity(len(df), n_users_raw, n_items_raw):.4%}")

    # 2. k-core
    print(f"\n[2/4] {k}-core 필터링")
    df = kcore_filter(df, k)
    n_users = df["user_id"].nunique()
    n_items = df["parent_asin"].nunique()
    print(f"  interactions : {len(df):,}  (제거: {n_users_raw - n_users:,} users, {n_items_raw - n_items:,} items)")
    print(f"  users        : {n_users:,}")
    print(f"  items        : {n_items:,}")
    print(f"  sparsity     : {sparsity(len(df), n_users, n_items):.4%}")

    # 3. 인코딩
    print("\n[3/4] ID 인코딩")
    df, user2id, item2id = encode_ids(df)
    print(f"  user ID 범위 : 0 ~ {n_users - 1}")
    print(f"  item ID 범위 : 0 ~ {n_items - 1}")

    # 4. 분할
    print("\n[4/4] Leave-one-out split")
    train, test = leave_one_out_split(df)
    print(f"  train : {len(train):,}")
    print(f"  test  : {len(test):,}")

    # 저장
    train.to_parquet(data_dir / "train.parquet", index=False)
    test.to_parquet(data_dir / "test.parquet", index=False)

    meta = {
        "category": CATEGORY,
        "k_core": k,
        "n_users": n_users,
        "n_items": n_items,
        "n_train": len(train),
        "n_test": len(test),
        "sparsity": sparsity(len(df), n_users, n_items),
    }
    (data_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
    (data_dir / "user2id.json").write_text(json.dumps(user2id))
    (data_dir / "item2id.json").write_text(json.dumps(item2id))

    print(f"\n저장 완료 → {data_dir}/")
    print("=" * 50)
    return meta


if __name__ == "__main__":
    run()
