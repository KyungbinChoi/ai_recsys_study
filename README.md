# AI 추천 시스템 스터디

Amazon Reviews 2023 데이터셋을 기반으로 **딥러닝 추천 모델**과 **LLM 큐레이션**을 결합한 쇼핑 챗봇 구현 프로젝트.

> "먼저 구현하고, 원리는 역설계한다" — 각 구현 단계 이후 노트북에서 개념을 정리합니다.

---

## 아키텍처 개요

```
사용자 입력 (자연어)
        │
        ▼
┌───────────────────┐
│   Gemini (LLM)    │  ① 쿼리 이해: 의도 / 선호 / 제약 조건 추출
└────────┬──────────┘
         │ 구조화된 사용자 컨텍스트
         ▼
┌───────────────────┐
│  DL 추천 모델     │  ② 후보 생성: 유저-아이템 임베딩 기반 Top-K
│  (NCF / SASRec)   │
└────────┬──────────┘
         │ Top-K 후보 아이템
         ▼
┌───────────────────┐
│   Gemini (LLM)    │  ③ 재순위 + 개인화 설명 생성
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Streamlit UI     │  ④ 쇼핑 챗봇 화면 출력
└───────────────────┘
```

**핵심 설계 원칙:**
- DL 모델은 방대한 아이템 공간에서 *관련성 높은 후보*를 빠르게 추립니다 (Recall 최적화).
- LLM은 후보를 사용자 맥락에 맞게 *정렬하고 설명*합니다 (Precision + UX 최적화).

---

## 구성 요소

### 1. 데이터 파이프라인

| 단계 | 내용 |
|------|------|
| 수집 | HuggingFace `datasets` — `McAuley-Lab/Amazon-Reviews-2023` (All_Beauty) |
| 전처리 | User/Item ID 인코딩, k-core 필터링(cold-start 제거), train/test split |
| 저장 | `data/processed/All_Beauty/*.parquet` |

**k-core 필터링:** 최소 k개 이상의 상호작용이 있는 유저·아이템만 남깁니다.
실제 추천 시스템에서 cold-start 문제를 줄이는 표준 전처리 방식입니다.

---

### 2. 딥러닝 추천 모델 (단계적 구현)

#### Stage 1 — Matrix Factorization (MF)
```
R ≈ U · Vᵀ
유저 행렬 U (n_users × k)  ×  아이템 행렬 V (n_items × k)
```
- 가장 기본적인 협업 필터링. 유저와 아이템을 k차원 잠재 공간에 임베딩.
- **학습 목표:** 잠재 요인(Latent Factor)의 의미 이해.

#### Stage 2 — Neural Collaborative Filtering (NCF)
```
GMF branch:  user_emb ⊙ item_emb  (element-wise product)
MLP branch:  concat → Linear → ReLU → ... → Linear
Output:      sigmoid(GMF output + MLP output)
```
- MF의 선형 한계를 신경망으로 극복.
- 논문: *He et al., "Neural Collaborative Filtering" (WWW 2017)*
- **학습 목표:** Implicit Feedback (구매=1, 미구매=0) 학습 방식 이해.

#### Stage 3 — SASRec (Self-Attentive Sequential Recommendation)
```
유저 A의 리뷰 이력 (timestamp 정렬):
  [item₁, item₂, ..., itemₙ]  →  다음 구매 아이템 예측
              ↓
  Transformer Encoder (Self-Attention)
```
- Amazon Reviews 데이터는 `user_id` + `timestamp`가 있어 Sequential 추천에 적합.
  (세션 기반과 다름 — 세션은 익명 단기 클릭스트림, Sequential은 장기 유저 이력)
- NCF/MF가 정적인 선호를 학습한다면, SASRec은 취향의 **변화 방향성**을 포착.
  (예: 기초 크림 → 수분 토너 → 고보습 세럼 → 다음은?)
- 논문: *Kang & McAuley, "Self-Attentive Sequential Recommendation" (ICDM 2018)*
  — 이 논문 자체가 Amazon Reviews 데이터를 벤치마크로 사용.
- **학습 목표:** 정적 협업 필터링의 한계와 시퀀셜 모델링의 필요성, Attention 작동 원리 이해.

**평가 지표:** `Hit@10`, `NDCG@10`

---

### 3. LLM 큐레이션 (Gemini)

| 기능 | 설명 |
|------|------|
| 쿼리 이해 | 자연어 입력 → 카테고리/가격/효능 등 구조화된 선호 추출 |
| 후보 설명 | DL 모델 추천 아이템에 대해 "왜 이 상품인지" 개인화 설명 생성 |
| 대화형 필터링 | "더 저렴한 걸로", "향이 없는 걸로" 등 후속 조건 반영 |
| 리뷰 요약 | 아이템의 리뷰 텍스트를 장단점 중심으로 요약 |

---

### 4. Streamlit 쇼핑 챗봇 UI

- 사용자가 자연어로 상품 추천 요청
- 추천 결과를 카드 형태로 표시 (상품명, 가격, 평점, 추천 이유)
- 대화 히스토리 유지하며 추가 조건 반영
- 실행: `uv run streamlit run ui/app.py`

---

## 디렉토리 구조

```
ai_recsys_study/
├── data/
│   ├── raw/                        # HuggingFace 캐시 (gitignore)
│   └── processed/
│       └── All_Beauty/
│           ├── reviews.parquet
│           └── meta.parquet
│
├── models/
│   ├── base.py                     # 추상 베이스 클래스
│   ├── mf.py                       # Matrix Factorization
│   ├── ncf.py                      # Neural Collaborative Filtering
│   └── sasrec.py                   # Self-Attentive Sequential Rec
│
├── llm/
│   ├── curator.py                  # Gemini 큐레이션 로직
│   └── prompts.py                  # 프롬프트 템플릿
│
├── ui/
│   └── app.py                      # Streamlit 챗봇
│
├── scripts/
│   ├── preprocess.py               # 전처리 파이프라인
│   └── train.py                    # 모델 학습 스크립트
│
└── notebooks/
    ├── 01_data_fetch.ipynb         # ✅ 데이터 수집 & EDA
    ├── 02_preprocessing.ipynb      # 전처리 & 인코딩
    ├── 03_mf_baseline.ipynb        # Matrix Factorization
    ├── 04_ncf.ipynb                # NCF 모델
    ├── 05_sasrec.ipynb             # SASRec 모델
    └── 06_llm_curation.ipynb       # LLM 통합
```

---

## 구현 로드맵

```
[✅] 1. 환경 설정 (uv + 의존성)
[✅] 2. 데이터 수집 & EDA (01_data_fetch.ipynb)
[ ]  3. 전처리 파이프라인 (k-core, 인코딩, split)
[ ]  4. MF 베이스라인 구현 & 평가
[ ]  5. NCF 구현 & MF와 성능 비교
[ ]  6. SASRec 구현 & 시퀀셜 패턴 분석
[ ]  7. Gemini 큐레이션 모듈 구현
[ ]  8. Streamlit 챗봇 UI 구현
[ ]  9. 전체 파이프라인 통합
```

---

## 시작하기

```bash
# 의존성 설치
uv sync --dev

# 환경 변수 설정
cp .env.example .env
# .env 에 GEMINI_API_KEY 입력

# 데이터 수집 (노트북)
uv run jupyter lab notebooks/01_data_fetch.ipynb

# 챗봇 실행 (구현 완료 후)
uv run streamlit run ui/app.py
```

---

## 기술 스택

| 역할 | 라이브러리 |
|------|-----------|
| 딥러닝 | PyTorch |
| 데이터 | HuggingFace datasets, pandas |
| LLM | Google Gemini (`google-genai`) |
| UI | Streamlit |
| ML 유틸 | scikit-learn |
| 개발 환경 | uv, Jupyter |
