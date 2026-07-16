"""
AI 쇼핑 추천 챗봇 — Streamlit UI

실행:
    uv run streamlit run ui/app.py
"""

import json
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# 프로젝트 루트를 경로에 추가 (ui/ 하위에서 실행 시)
sys.path.append(str(Path(__file__).parent.parent))
load_dotenv()

# ── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI 뷰티 쇼핑 어시스턴트",
    page_icon="✨",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.product-card {
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    background: #fafafa;
}
.product-title {
    font-weight: 600;
    font-size: 15px;
    margin-bottom: 4px;
}
.product-meta {
    font-size: 13px;
    color: #666;
    margin-bottom: 8px;
}
.reason-box {
    background: #f0f7ff;
    border-left: 3px solid #1a73e8;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 13px;
    color: #333;
}
.rank-badge {
    display: inline-block;
    background: #1a73e8;
    color: white;
    border-radius: 50%;
    width: 24px;
    height: 24px;
    text-align: center;
    line-height: 24px;
    font-size: 12px;
    font-weight: bold;
    margin-right: 8px;
}
</style>
""", unsafe_allow_html=True)


# ── 파이프라인 로딩 (캐시) ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="모델 로딩 중...")
def load_pipeline(api_key: str):
    from llm.pipeline import RecommendationPipeline
    return RecommendationPipeline.load(api_key=api_key)


def get_image_url(images_data) -> str | None:
    """meta.parquet의 images 컬럼에서 썸네일 URL 추출."""
    try:
        if images_data is None:
            return None
        if isinstance(images_data, dict):
            large = images_data.get("large") or images_data.get("thumb")
            if large is not None and len(large) > 0 and large[0] is not None:
                return str(large[0])
        return None
    except Exception:
        return None


class _DummyCuratedItem:
    """세션 상태 복원용 (Pydantic 직렬화 없이 dict → 객체)."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def render_product_cards(curated_items, meta_df, id2item):
    """큐레이션 결과를 카드 형태로 렌더링."""
    for rank, item in enumerate(curated_items, start=1):
        asin = id2item.get(item.item_id, "")
        rows = meta_df[meta_df["parent_asin"] == asin]
        img_url = get_image_url(rows.iloc[0]["images"]) if not rows.empty else None

        col_img, col_info = st.columns([1, 4])

        with col_img:
            if img_url:
                st.image(img_url, width=100)
            else:
                st.markdown("🧴", unsafe_allow_html=False)

        with col_info:
            rating_str = f"{'⭐' * round(item.avg_rating)} {item.avg_rating:.1f}" if item.avg_rating else ""
            review_str = f"({item.rating_count:,}개 리뷰)" if item.rating_count else ""
            price_str  = f"💰 {item.price}" if item.price else ""
            dl_str     = f"DL 추천 순위: {item.dl_rank}위"

            st.markdown(
                f'<div class="product-card">'
                f'<div class="product-title"><span class="rank-badge">{rank}</span>{item.title}</div>'
                f'<div class="product-meta">{rating_str} {review_str} &nbsp; {price_str} &nbsp; <span style="color:#999">{dl_str}</span></div>'
                f'<div class="reason-box">✨ {item.reason}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── 사이드바 ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ 설정")

    # API 키
    api_key_input = st.text_input(
        "Gemini API Key",
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
        help=".env 파일에 GEMINI_API_KEY를 설정하거나 여기에 직접 입력하세요.",
    )

    st.divider()

    # 유저 선택
    st.subheader("👤 테스트 유저")
    user_id = st.number_input(
        "User ID",
        min_value=0,
        max_value=2086,
        value=0,
        step=1,
        help="0 ~ 2086 범위의 유저 ID (All_Beauty 데이터셋 기준)",
    )

    # 추천 설정
    st.divider()
    st.subheader("🎛️ 추천 설정")
    top_k   = st.slider("최종 추천 수 (Top-K)", min_value=1, max_value=10, value=5)
    dl_top_k = st.slider("DL 후보 수", min_value=10, max_value=50, value=20,
                         help="Gemini에 넘기기 전 DL 모델이 생성하는 후보 수")

    st.divider()
    st.caption("**아키텍처**")
    st.caption("SASRec → 후보 생성")
    st.caption("Gemini → 재순위 + 설명")

    if st.button("💬 대화 초기화"):
        st.session_state.messages = []
        st.rerun()


# ── 메인 영역 ─────────────────────────────────────────────────────────────────

st.title("✨ AI 뷰티 쇼핑 어시스턴트")
st.caption("SASRec(후보 생성) + Gemini(큐레이션)가 결합된 개인화 추천 챗봇")

# API 키 체크
if not api_key_input:
    st.warning("사이드바에 Gemini API Key를 입력해 주세요.")
    st.stop()

# 파이프라인 로드
try:
    pipeline = load_pipeline(api_key_input)
except Exception as e:
    st.error(f"파이프라인 로딩 실패: {e}")
    st.stop()

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 히스토리 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
        else:
            # 어시스턴트 메시지: content에 따라 다른 렌더링
            if msg.get("type") == "context":
                ctx = msg["content"]
                with st.expander("🔍 쿼리 이해 결과", expanded=False):
                    st.markdown(f"**의도:** {ctx['intent']}")
                    if ctx["preferences"]:
                        st.markdown(f"**선호:** {', '.join(ctx['preferences'])}")
                    if ctx["constraints"]:
                        st.markdown(f"**제외:** {', '.join(ctx['constraints'])}")
            elif msg.get("type") == "cards":
                items_data = msg["content"]
                render_product_cards(
                    [_DummyCuratedItem(**d) for d in items_data],
                    pipeline.meta_df,
                    pipeline.id2item,
                )
            else:
                st.write(msg["content"])


# ── 채팅 입력 ─────────────────────────────────────────────────────────────────

if prompt := st.chat_input("추천받고 싶은 상품을 자유롭게 입력하세요 (예: 건성 피부에 좋은 무향 세럼)"):
    # 유저 메시지 기록
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 어시스턴트 응답
    with st.chat_message("assistant"):
        with st.spinner("🤔 분석 중..."):
            try:
                user_context, curated = pipeline.recommend(
                    user_id=user_id,
                    query=prompt,
                    top_k=top_k,
                    dl_top_k=dl_top_k,
                )
            except Exception as e:
                st.error(f"추천 생성 실패: {e}")
                st.stop()

        # ① 쿼리 이해 결과
        ctx_dict = user_context.model_dump()
        with st.expander("🔍 쿼리 이해 결과", expanded=True):
            st.markdown(f"**의도:** {user_context.intent}")
            if user_context.preferences:
                st.markdown(f"**선호:** {', '.join(user_context.preferences)}")
            if user_context.constraints:
                st.markdown(f"**제외:** {', '.join(user_context.constraints)}")

        st.session_state.messages.append({
            "role": "assistant",
            "type": "context",
            "content": ctx_dict,
        })

        # ② 추천 카드
        st.markdown(f"**{user_id}번 유저**를 위한 맞춤 추천 Top-{top_k}")
        render_product_cards(curated, pipeline.meta_df, pipeline.id2item)

        items_serializable = [
            {
                "item_id":      c.item_id,
                "title":        c.title,
                "reason":       c.reason,
                "avg_rating":   c.avg_rating,
                "rating_count": c.rating_count,
                "price":        c.price,
                "dl_rank":      c.dl_rank,
            }
            for c in curated
        ]
        st.session_state.messages.append({
            "role": "assistant",
            "type": "cards",
            "content": items_serializable,
        })
