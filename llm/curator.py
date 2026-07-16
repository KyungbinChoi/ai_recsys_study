"""
Gemini 기반 LLM 큐레이션 모듈.

역할:
  1. 자연어 쿼리 이해 → 구조화된 사용자 컨텍스트 추출 (UserContext)
  2. DL 모델의 Top-K 후보를 사용자 맥락에 맞게 재순위 + 개인화 설명 생성

사용법:
    from llm.curator import Curator, ItemInfo

    curator = Curator(api_key=os.environ["GEMINI_API_KEY"])
    ctx  = curator.understand_query("건성 피부에 좋은 저자극 세럼 추천해줘")
    items = [ItemInfo(item_id=0, title="...", avg_rating=4.5, ...)]
    result = curator.curate(ctx, items, top_k=5)
"""

import json
import os
from typing import Generator

from pydantic import BaseModel, Field

from llm.prompts import (
    QUERY_UNDERSTANDING_SYSTEM,
    QUERY_UNDERSTANDING_USER,
    RERANK_SYSTEM,
    RERANK_USER,
    format_candidates,
)


# ── Pydantic 모델 ──────────────────────────────────────────────────────────────

class UserContext(BaseModel):
    intent: str = Field(description="유저 의도 요약")
    preferences: list[str] = Field(default_factory=list, description="선호 조건")
    constraints: list[str] = Field(default_factory=list, description="제외 조건")
    keywords: list[str] = Field(default_factory=list, description="상품 키워드")
    language: str = Field(default="ko", description="쿼리 언어 (ISO 639-1)")


class ItemInfo(BaseModel):
    item_id: int = Field(description="0-indexed item ID (DL 모델 기준)")
    title: str
    avg_rating: float | None = None
    rating_count: int | None = None
    price: str | None = None
    features: list[str] = Field(default_factory=list)


class CuratedItem(BaseModel):
    item_id: int
    title: str
    reason: str = Field(description="개인화 추천 이유")
    avg_rating: float | None = None
    rating_count: int | None = None
    price: str | None = None
    dl_rank: int = Field(description="DL 모델 원래 순위 (1-indexed)")


# ── Curator ────────────────────────────────────────────────────────────────────

class Curator:
    """
    Gemini 기반 추천 큐레이터.

    Args:
        api_key: Gemini API 키 (없으면 GEMINI_API_KEY 환경 변수 사용)
        model:   사용할 Gemini 모델 ID
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.0-flash",
    ):
        try:
            from google import genai
        except ImportError as e:
            raise ImportError("google-genai 패키지를 설치해주세요: uv add google-genai") from e

        self._genai = genai
        self.model = model
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다.")
        self.client = genai.Client(api_key=key)

    # ── 쿼리 이해 ──────────────────────────────────────────────────────────────

    def understand_query(self, query: str) -> UserContext:
        """
        자연어 쿼리를 구조화된 UserContext로 변환.
        Gemini JSON mode를 사용해 파싱 없이 바로 dict를 받음.
        """
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model,
            contents=QUERY_UNDERSTANDING_USER.format(query=query),
            config=types.GenerateContentConfig(
                system_instruction=QUERY_UNDERSTANDING_SYSTEM,
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        raw = json.loads(response.text)
        return UserContext(**raw)

    # ── 재순위 + 설명 ──────────────────────────────────────────────────────────

    def curate(
        self,
        user_context: UserContext,
        candidates: list[ItemInfo],
        top_k: int = 5,
    ) -> list[CuratedItem]:
        """
        DL 모델의 후보 목록을 재순위하고 개인화 설명을 생성.

        Args:
            user_context: understand_query()의 결과
            candidates:   DL 모델이 생성한 Top-K 후보 (순서: DL 모델 rank 순)
            top_k:        최종 반환할 아이템 수

        Returns:
            CuratedItem 리스트 (Gemini 추천 순서)
        """
        from google.genai import types

        # 후보를 dict로 변환 (프롬프트 포매팅용)
        candidates_dicts = [c.model_dump() for c in candidates]
        candidates_text  = format_candidates(candidates_dicts)

        prompt = RERANK_USER.format(
            intent=user_context.intent,
            preferences=", ".join(user_context.preferences) or "없음",
            constraints=", ".join(user_context.constraints) or "없음",
            candidates_text=candidates_text,
        )

        system = RERANK_SYSTEM.format(
            top_k=top_k,
            language=user_context.language,
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                temperature=0.3,
            ),
        )

        ranked_raw: list[dict] = json.loads(response.text)

        # item_id로 원본 ItemInfo 매핑
        id2item = {c.item_id: c for c in candidates}
        result = []
        for llm_rank, raw in enumerate(ranked_raw[:top_k], start=1):
            iid = raw["item_id"]
            item = id2item.get(iid)
            if item is None:
                continue
            result.append(
                CuratedItem(
                    item_id=iid,
                    title=item.title,
                    reason=raw["reason"],
                    avg_rating=item.avg_rating,
                    rating_count=item.rating_count,
                    price=item.price,
                    dl_rank=candidates_dicts.index(item.model_dump()) + 1,
                )
            )
        return result

    # ── 스트리밍 설명 (UI 용) ──────────────────────────────────────────────────

    def explain_item_stream(
        self,
        query: str,
        item: ItemInfo,
        user_context: UserContext,
    ) -> Generator[str, None, None]:
        """
        단일 아이템에 대해 스트리밍으로 개인화 설명 생성.
        Streamlit의 st.write_stream()에 직접 전달 가능.
        """
        from google.genai import types

        features_text = "; ".join(item.features[:5]) if item.features else "정보 없음"
        prompt = (
            f"User query: {query}\n"
            f"User intent: {user_context.intent}\n"
            f"Preferences: {', '.join(user_context.preferences)}\n\n"
            f"Product: {item.title}\n"
            f"Rating: {item.avg_rating}/5 ({item.rating_count} reviews)\n"
            f"Price: {item.price or '정보 없음'}\n"
            f"Features: {features_text}\n\n"
            f"Write 2-3 sentences explaining why this product fits the user's needs. "
            f"Write in {user_context.language}."
        )

        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.5),
        ):
            if chunk.text:
                yield chunk.text
