"""
MF 학습 스크립트.
사용: uv run python scripts/train.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent))
from models.evaluate import evaluate
from models.mf import BPRDataset, MatrixFactorization

DATA_DIR = Path("data/processed/All_Beauty")
SAVE_DIR = Path("data/processed/All_Beauty")


def run(
    n_factors: int = 64,
    lr: float = 1e-3,
    batch_size: int = 512,
    n_epochs: int = 50,
    K: int = 10,
):
    # 데이터 로드
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")
    with open(DATA_DIR / "dataset_meta.json") as f:
        meta = json.load(f)

    n_users, n_items = meta["n_users"], meta["n_items"]

    # Dataset / DataLoader
    dataset = BPRDataset(train_df, n_items)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 모델 / 옵티마이저
    model = MatrixFactorization(n_users, n_items, n_factors)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"학습 시작: n_users={n_users}, n_items={n_items}, n_factors={n_factors}")
    print(f"{'Epoch':>6} {'Loss':>8} {'Hit@10':>8} {'NDCG@10':>9}")
    print("-" * 36)

    history = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        for users, pos_items, neg_items in loader:
            optimizer.zero_grad()
            loss = model.bpr_loss(users, pos_items, neg_items)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        if epoch % 10 == 0:
            model.eval()

            def score_fn(user_id, item_ids):
                u = torch.tensor([user_id] * len(item_ids))
                i = torch.tensor(item_ids)
                with torch.no_grad():
                    return model(u, i)

            metrics = evaluate(score_fn, train_df, test_df, n_items, K=K)
            hit = metrics[f"hit@{K}"]
            ndcg = metrics[f"ndcg@{K}"]
            print(f"{epoch:>6} {avg_loss:>8.4f} {hit:>8.4f} {ndcg:>9.4f}")
            history.append({"epoch": epoch, "loss": avg_loss, **metrics})

    # 모델 저장
    torch.save(model.state_dict(), SAVE_DIR / "mf_model.pt")
    print(f"\n모델 저장 완료 → {SAVE_DIR}/mf_model.pt")
    return model, history


if __name__ == "__main__":
    run()
