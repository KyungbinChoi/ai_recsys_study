"""
SASRec: Self-Attentive Sequential Recommendation
논문: Kang & McAuley, "Self-Attentive Sequential Recommendation" (ICDM 2018)

아이템 ID 규칙:
    0           = padding  (빈 위치)
    1 ~ n_items = 실제 아이템  (train_df 기준 0-indexed → +1 shift)
"""
import random

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ── Dataset ──────────────────────────────────────────────────────────────────

class SASRecDataset(Dataset):
    """
    유저별 아이템 시퀀스를 (input_seq, pos_seq, neg_seq) 로 반환.

    훈련 방식 — sliding window:
        items = [a, b, c, d, e]
        input : [a, b, c, d]  ← transformer 입력 (max_len으로 왼쪽 패딩)
        pos   : [b, c, d, e]  ← 각 위치의 정답 next-item
        neg   : [?, ?, ?, ?]  ← 각 위치에 대응하는 랜덤 부정 샘플

    한 번의 forward로 시퀀스 내 모든 위치에서 loss를 계산할 수 있다.
    """

    def __init__(self, df: pd.DataFrame, n_items: int, max_len: int = 50):
        self.max_len = max_len
        self.n_items = n_items
        self.data: list[tuple[list, set]] = []

        for _, group in df.groupby("user_id"):
            # 0-indexed → +1 (0은 padding 예약)
            items = [i + 1 for i in group.sort_values("timestamp")["parent_asin"].tolist()]
            self.data.append((items, set(items)))

    def _pad(self, seq: list, length: int) -> list:
        seq = seq[-length:]
        return [0] * (length - len(seq)) + seq

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        items, seen = self.data[idx]
        input_seq = self._pad(items[:-1], self.max_len)
        pos_seq   = self._pad(items[1:],  self.max_len)

        neg_seq = []
        for pos in pos_seq:
            if pos == 0:
                neg_seq.append(0)
            else:
                while True:
                    neg = random.randint(1, self.n_items)
                    if neg not in seen:
                        break
                neg_seq.append(neg)

        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(pos_seq,   dtype=torch.long),
            torch.tensor(neg_seq,   dtype=torch.long),
        )


# ── Model ─────────────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SASRecBlock(nn.Module):
    """
    Causal Self-Attention + FFN (Pre-LayerNorm 방식).
    Pre-LN: 원 논문은 Post-LN이지만 Pre-LN이 학습 안정성이 높아 실무에서 주로 사용.

    key_padding_mask를 쓰지 않는 이유:
        encode()에서 패딩 위치 임베딩을 0으로 초기화하기 때문에
        패딩 위치를 attention해도 0 * weight = 0 → 결과에 영향 없음.
        key_padding_mask를 쓰면 causal mask와 겹쳐 all-masked → softmax NaN 발생.
    """

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn  = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.ffn   = FeedForward(hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        # Self-Attention (with residual)
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=causal_mask)
        x = x + self.drop(h)

        # FFN (with residual)
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation.

    핵심 — Causal Self-Attention:
        position i 는 position 0 ~ i 까지만 참조 (미래 차단).
        → 위치 i 의 출력으로 i+1 번째 아이템을 예측.

    MF/NCF와의 차이:
        MF/NCF : 유저의 전체 이력을 하나의 벡터로 압축 (순서 무시)
        SASRec : 시퀀스 각 위치의 컨텍스트를 별도로 학습 (순서 보존)
    """

    def __init__(
        self,
        n_items: int,
        hidden_dim: int = 64,
        max_len: int = 50,
        n_heads: int = 2,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_items    = n_items
        self.max_len    = max_len
        self.hidden_dim = hidden_dim

        # padding_idx=0 → 패딩 위치의 item 임베딩은 0 고정 (gradient 없음)
        self.item_emb = nn.Embedding(n_items + 1, hidden_dim, padding_idx=0)
        self.pos_emb  = nn.Embedding(max_len, hidden_dim)
        self.emb_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [SASRecBlock(hidden_dim, n_heads, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.normal_(self.pos_emb.weight,  std=0.01)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq: (B, L)
        returns: (B, L, H) — 각 위치의 컨텍스트 표현
        """
        B, L = seq.shape
        device = seq.device

        pos = torch.arange(L, device=device).unsqueeze(0)  # (1, L)
        x = self.item_emb(seq) + self.pos_emb(pos)

        # 패딩 위치(seq==0)의 임베딩을 0으로 강제.
        # pos_emb는 padding_idx가 없어 패딩 위치에도 비-0 값이 들어가므로 직접 마스킹.
        x = x * (seq != 0).unsqueeze(-1)
        x = self.emb_drop(x)

        # Causal mask: 상삼각 = True → 해당 위치 무시 (미래 차단)
        causal_mask = ~torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))

        for block in self.blocks:
            x = block(x, causal_mask)

        return self.norm(x)  # (B, L, H)

    def forward(
        self,
        seq: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        """훈련용: 유효 위치 전체에서 BPR loss."""
        h = self.encode(seq)                            # (B, L, H)

        pos_emb    = self.item_emb(pos_items)           # (B, L, H)
        neg_emb    = self.item_emb(neg_items)           # (B, L, H)
        pos_scores = (h * pos_emb).sum(-1)             # (B, L)
        neg_scores = (h * neg_emb).sum(-1)             # (B, L)

        # padding 위치 제외
        valid = pos_items != 0
        loss  = -F.logsigmoid(pos_scores - neg_scores)
        return (loss * valid).sum() / valid.sum()

    @torch.no_grad()
    def score_seq(self, seq: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        추론용: 시퀀스 마지막 위치 표현으로 후보 아이템 점수.
        seq:      (L,) or (1, L)
        item_ids: (K,)  — 이미 +1 shift된 ID
        returns:  (K,)
        """
        if seq.dim() == 1:
            seq = seq.unsqueeze(0)
        h   = self.encode(seq)[:, -1, :]         # (1, H)
        emb = self.item_emb(item_ids)            # (K, H)
        return (h @ emb.T).squeeze(0)            # (K,)
