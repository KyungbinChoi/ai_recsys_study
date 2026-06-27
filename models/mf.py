"""
Matrix Factorization with BPR Loss

핵심 아이디어:
    R ≈ U · Vᵀ
    유저 행렬 U (n_users × k)  ×  아이템 행렬 V (n_items × k)
    예측 점수 = 내적 (dot product)
"""
import random

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class BPRDataset(Dataset):
    """
    BPR 학습용 데이터셋. 각 샘플: (user, pos_item, neg_item)

    BPR을 쓰는 이유:
        리뷰 데이터는 implicit feedback — "좋아함"만 있고 "싫어함"은 없다.
        MSE로 학습하면 관측되지 않은 (user, item) 쌍을 모두 0으로 가정해야 해서 부정확.
        BPR은 "관측된 아이템이 관측되지 않은 아이템보다 선호될 것"이라는 가정만 쓴다.
    """

    def __init__(self, df: pd.DataFrame, n_items: int):
        self.users = df["user_id"].values
        self.items = df["parent_asin"].values
        self.n_items = n_items
        self.user_items = (
            df.groupby("user_id")["parent_asin"].apply(set).to_dict()
        )

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = int(self.users[idx])
        pos_item = int(self.items[idx])
        # 해당 유저 이력에 없는 아이템 샘플링
        while True:
            neg_item = random.randint(0, self.n_items - 1)
            if neg_item not in self.user_items[user]:
                break
        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long),
        )


class MatrixFactorization(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_factors: int = 64):
        super().__init__()
        self.n_factors = n_factors
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_ids)   # (B, k)
        i = self.item_emb(item_ids)   # (B, k)
        return (u * i).sum(dim=-1)    # (B,)  내적

    def bpr_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        """
        -log σ(score_pos - score_neg)
        긍정 아이템 점수가 부정 아이템보다 높도록 순위 학습.
        """
        pos = self.forward(users, pos_items)
        neg = self.forward(users, neg_items)
        return -F.logsigmoid(pos - neg).mean()

    @torch.no_grad()
    def score_all_items(self, user_id: int) -> torch.Tensor:
        """추론용: 유저의 모든 아이템에 대한 점수."""
        u = self.user_emb.weight[user_id]   # (k,)
        return self.item_emb.weight @ u      # (n_items,)
