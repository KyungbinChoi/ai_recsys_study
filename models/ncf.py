"""
Neural Collaborative Filtering (NeuMF)
논문: He et al., "Neural Collaborative Filtering" (WWW 2017)

GMF branch  — element-wise product (MF의 선형 상호작용 일반화)
MLP branch  — concat → FC layers  (비선형 상호작용 학습)
NeuMF       — GMF + MLP 결합
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuMF(nn.Module):
    """
    구조:

    user_id ─→ GMF_user_emb ─→ element-wise × ─→          ┐
    item_id ─→ GMF_item_emb ─→                             ├─ concat ─→ Linear(1)
                                                            │
    user_id ─→ MLP_user_emb ─→ concat ─→ FC─ReLU─... ─→  ┘
    item_id ─→ MLP_item_emb ─→

    GMF와 MLP가 각각 독립 임베딩을 사용하는 이유:
        두 브랜치가 서로 다른 방식으로 유저·아이템 관계를 학습하도록
        표현 공간을 분리한다.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int = 32,
        mlp_layers: list[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [64, 32, 16]

        # ── GMF branch ──────────────────────────────────────────
        self.gmf_user_emb = nn.Embedding(n_users, n_factors)
        self.gmf_item_emb = nn.Embedding(n_items, n_factors)

        # ── MLP branch ──────────────────────────────────────────
        # 입력 차원: user_emb(n_factors) + item_emb(n_factors)
        self.mlp_user_emb = nn.Embedding(n_users, n_factors)
        self.mlp_item_emb = nn.Embedding(n_items, n_factors)

        mlp_modules = []
        in_dim = n_factors * 2
        for out_dim in mlp_layers:
            mlp_modules += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim
        self.mlp = nn.Sequential(*mlp_modules)

        # ── 최종 예측 ────────────────────────────────────────────
        # GMF 출력(n_factors) + MLP 출력(mlp_layers[-1]) → 1
        self.predict_layer = nn.Linear(n_factors + mlp_layers[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for emb in [
            self.gmf_user_emb, self.gmf_item_emb,
            self.mlp_user_emb, self.mlp_item_emb,
        ]:
            nn.init.normal_(emb.weight, std=0.01)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        # GMF
        gmf_u = self.gmf_user_emb(user_ids)           # (B, k)
        gmf_i = self.gmf_item_emb(item_ids)           # (B, k)
        gmf_out = gmf_u * gmf_i                       # (B, k) element-wise product

        # MLP
        mlp_u = self.mlp_user_emb(user_ids)           # (B, k)
        mlp_i = self.mlp_item_emb(item_ids)           # (B, k)
        mlp_out = self.mlp(torch.cat([mlp_u, mlp_i], dim=-1))  # (B, mlp_layers[-1])

        # NeuMF: concat → predict
        out = self.predict_layer(torch.cat([gmf_out, mlp_out], dim=-1))  # (B, 1)
        return out.squeeze(-1)                         # (B,)

    def bpr_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        pos = self.forward(users, pos_items)
        neg = self.forward(users, neg_items)
        return -F.logsigmoid(pos - neg).mean()

    @torch.no_grad()
    def score_all_items(self, user_id: int, batch_size: int = 512) -> torch.Tensor:
        """추론용: 배치 단위로 모든 아이템에 대한 점수 계산."""
        n_items = self.gmf_item_emb.num_embeddings
        scores = []
        for start in range(0, n_items, batch_size):
            end = min(start + batch_size, n_items)
            items = torch.arange(start, end)
            users = torch.full((end - start,), user_id, dtype=torch.long)
            scores.append(self.forward(users, items))
        return torch.cat(scores)
