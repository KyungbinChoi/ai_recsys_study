"""
DL 추천 모델 + Gemini 큐레이터 통합 파이프라인.

사용법:
    pipeline = RecommendationPipeline.load(api_key=...)
    result = pipeline.recommend(user_id=42, query="건성 피부용 세럼", top_k=5)
"""

import json
import os
from pathlib import Path

import pandas as pd
import torch

from llm.curator import Curator, CuratedItem, ItemInfo, UserContext

DATA_DIR = Path("data/processed/All_Beauty")
MAX_LEN  = 50


class RecommendationPipeline:
    """
    SASRec(후보 생성) + Gemini(재순위 + 설명)의 통합 인터페이스.
    """

    def __init__(
        self,
        sasrec_model,
        train_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        item2id: dict,
        n_items: int,
        curator: Curator,
    ):
        self.model    = sasrec_model
        self.train_df = train_df
        self.meta_df  = meta_df
        self.item2id  = item2id
        self.id2item  = {v: k for k, v in item2id.items()}
        self.n_items  = n_items
        self.curator  = curator

        # 유저별 시퀀스 캐시 (0-indexed item IDs, +1 shift는 score 시에)
        self._user_seqs: dict[int, list[int]] = {
            uid: grp.sort_values("timestamp")["parent_asin"].tolist()
            for uid, grp in train_df.groupby("user_id")
        }

    # ── 팩토리 메서드 ──────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        api_key: str | None = None,
        data_dir: Path = DATA_DIR,
        model_name: str = "sasrec",
        gemini_model: str = "gemini-2.0-flash",
    ) -> "RecommendationPipeline":
        from models.sasrec import SASRec
        from scripts.train import MAX_LEN as _MAX_LEN

        train_df = pd.read_parquet(data_dir / "train.parquet")
        meta_df  = pd.read_parquet(data_dir / "meta.parquet")
        with open(data_dir / "dataset_meta.json") as f:
            meta     = json.load(f)
        with open(data_dir / "item2id.json") as f:
            item2id  = json.load(f)

        n_items = meta["n_items"]

        sasrec = SASRec(n_items, hidden_dim=64, max_len=_MAX_LEN, n_heads=2, n_layers=2)
        sasrec.load_state_dict(
            torch.load(data_dir / f"{model_name}_model.pt", weights_only=True)
        )
        sasrec.eval()

        curator = Curator(api_key=api_key, model=gemini_model)

        return cls(sasrec, train_df, meta_df, item2id, n_items, curator)

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────────

    def _get_dl_candidates(self, user_id: int, dl_top_k: int = 20) -> list[ItemInfo]:
        """SASRec으로 상위 dl_top_k 후보를 뽑아 ItemInfo 리스트로 반환."""
        seen  = set(self._user_seqs[user_id])
        seq   = self._user_seqs[user_id]

        # +1 shift: 0=padding 예약
        seq_t  = torch.tensor(
            (lambda s: [0] * (MAX_LEN - min(len(s), MAX_LEN)) + [i + 1 for i in s[-MAX_LEN:]])(seq),
            dtype=torch.long,
        )
        # seen 제외한 후보 전체 점수
        cand_ids = [i for i in range(self.n_items) if i not in seen]
        items_t  = torch.tensor([i + 1 for i in cand_ids], dtype=torch.long)

        with torch.no_grad():
            scores = self.model.score_seq(seq_t, items_t).numpy()

        # 점수 내림차순 정렬
        ranked = sorted(zip(cand_ids, scores), key=lambda x: -x[1])[:dl_top_k]

        result = []
        for item_id, _ in ranked:
            asin = self.id2item.get(item_id, "")
            rows = self.meta_df[self.meta_df["parent_asin"] == asin]
            if rows.empty:
                title    = asin
                rating   = None
                n_rating = None
                price    = None
                features = []
            else:
                row      = rows.iloc[0]
                title    = str(row.get("title") or asin)
                rating   = float(row["average_rating"]) if pd.notna(row.get("average_rating")) else None
                n_rating = int(row["rating_number"])   if pd.notna(row.get("rating_number"))   else None
                price_raw = row.get("price")
                price    = str(price_raw) if pd.notna(price_raw) else None
                feats    = row.get("features") or []
                features = list(feats)[:5] if feats else []

            result.append(ItemInfo(
                item_id=item_id,
                title=title,
                avg_rating=rating,
                rating_count=n_rating,
                price=price,
                features=features,
            ))
        return result

    # ── 메인 인터페이스 ────────────────────────────────────────────────────────

    def recommend(
        self,
        user_id: int,
        query: str,
        top_k: int = 5,
        dl_top_k: int = 20,
    ) -> tuple[UserContext, list[CuratedItem]]:
        """
        유저 ID + 자연어 쿼리 → 큐레이션된 추천 결과.

        Args:
            user_id:  train_df 기준 유저 ID
            query:    자연어 추천 요청
            top_k:    최종 반환 아이템 수
            dl_top_k: DL 모델에서 뽑을 후보 수 (Gemini로 넘길 풀)

        Returns:
            (user_context, curated_items)
        """
        # 1. 쿼리 이해
        user_context = self.curator.understand_query(query)

        # 2. DL 모델로 후보 생성
        candidates = self._get_dl_candidates(user_id, dl_top_k=dl_top_k)

        # 3. Gemini 재순위 + 설명
        curated = self.curator.curate(user_context, candidates, top_k=top_k)

        return user_context, curated
