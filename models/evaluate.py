"""
추천 모델 평가 유틸리티.
NCF, SASRec 등 모든 모델에서 재사용.

평가 프로토콜:
    Leave-one-out + 99 random negatives
    각 유저의 test 아이템을 99개 랜덤 부정 아이템과 함께 순위 매김.
    Hit@K : test 아이템이 상위 K에 있으면 1
    NDCG@K: 순위에 따라 가중치를 준 Hit (1 / log2(rank+1))
"""
import math
import random
from typing import Callable

import numpy as np
import pandas as pd
import torch


def evaluate(
    score_fn: Callable[[int, list[int]], torch.Tensor],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_items: int,
    K: int = 10,
    n_neg: int = 99,
) -> dict[str, float]:
    """
    Args:
        score_fn: (user_id, item_ids) → scores tensor.  모델마다 다르게 전달.
        train_df: 유저별 이미 본 아이템 집합 파악용
        test_df : user_id, parent_asin 컬럼 포함
        n_items : 전체 아이템 수
        K       : Hit@K, NDCG@K 기준
        n_neg   : 랜덤 부정 샘플 수

    Returns:
        {"hit@K": float, "ndcg@K": float}
    """
    user_history = (
        train_df.groupby("user_id")["parent_asin"].apply(set).to_dict()
    )

    hits, ndcgs = [], []
    for row in test_df.itertuples(index=False):
        user = row.user_id
        pos = row.parent_asin
        history = user_history.get(user, set())

        # 99 랜덤 부정 아이템 (유저 이력 + test 아이템 제외)
        neg_pool = list(set(range(n_items)) - history - {pos})
        neg_items = random.sample(neg_pool, min(n_neg, len(neg_pool)))

        candidates = [pos] + neg_items
        scores = score_fn(user, candidates)           # tensor (100,)
        rank = int((scores > scores[0]).sum()) + 1    # 1-indexed

        hits.append(1 if rank <= K else 0)
        ndcgs.append(1 / math.log2(rank + 1) if rank <= K else 0)

    return {f"hit@{K}": float(np.mean(hits)), f"ndcg@{K}": float(np.mean(ndcgs))}
