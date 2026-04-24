# 🏦 FinSight AI — Bank Marketing Intelligence Suite

> Production-grade ML pipeline + RAG chatbot predicting term deposit subscriptions from bank telemarketing campaign data — deployed live on Hugging Face Spaces and Streamlit Cloud.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.5-brightgreen)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikitlearn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.14-0194E2?logo=mlflow&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?logo=mongodb&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-multi--stage-2496ED?logo=docker&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-27%20tests-success?logo=pytest&logoColor=white)

---

## 🌐 Live Demo

| Service | URL |
|---|---|
| **Streamlit Dashboard** | https://finsight-ai-k8qemrdroxhexpnqumnmrw.streamlit.app |
| **FastAPI Backend (HF Spaces)** | https://nikollass-finsight-ai-backend.hf.space/docs |

> HF Spaces free tier — first request may take ~20s if the Space is waking up.

---

## 📖 About This Project

### The Problem
Banks run telemarketing campaigns to sell term deposits (savings products with fixed interest rates and lock-in periods). Campaign conversion rates are low — around 11% in historical Portuguese bank data — and every call costs money. Calling every customer is wasteful; the business needs to prioritize leads most likely to subscribe.

### The Solution
FinSight AI is a production-grade machine learning system that scores each customer by their likelihood of subscribing to a term deposit, enabling marketing teams to focus their calls on high-probability leads. The system was built end-to-end: from exploratory data analysis and statistical testing through feature engineering, model training, threshold tuning, REST API deployment, an interactive dashboard, and a RAG-powered strategy chatbot.

### The Dataset
Built on the UCI Bank Marketing Dataset — 41,188 real telemarketing contacts from a Portuguese retail bank between May 2008 and November 2010. Features include customer demographics (age, job, education, marital status), financial attributes (credit default, loans), campaign history (contacts, days since last contact, previous outcomes), and macro-economic indicators (Euribor rate, employment variation, consumer confidence index).

### Methodology
The project applies rigorous ML engineering practices at every step:

1. **Exploratory & Statistical Analysis** — Hypothesis tests (chi-square, t-tests), multicollinearity diagnostics (VIF), logistic regression summary statistics via statsmodels, distribution and correlation visualizations via Plotly.
2. **Feature Engineering** — Saved `ColumnTransformer` (StandardScaler for numerics, OrdinalEncoder for education's natural progression, OneHotEncoder for nominal categories) fit only on training data to prevent **training/serving skew**. Recoded `pdays=999` sentinel to `-1`. Dropped the `duration` feature because it's only known after the call (**data leakage**).
3. **Stratified Splits** — Train (72%), validation (8%), test (20%), all **stratified splits** on the target to preserve the 11% positive class rate across splits.
4. **Class Imbalance Handling** — SMOTE applied only to the training set (never before splitting — that would leak test-set neighborhoods into training).
5. **Multi-Model Training** — Four models trained with hyperparameters config-driven from YAML: Logistic Regression, Random Forest, XGBoost, and LightGBM. XGBoost and LightGBM use early stopping on the validation set.
6. **Threshold Tuning** — Decision **threshold tuning** per model on the validation set to maximize F1 (optimal range: 0.23 to 0.84). The test set is used exactly once for final reporting.
7. **Model Selection** — Best model selected by validation **PR-AUC** (appropriate for imbalanced classes, not ROC-AUC).
8. **Experiment Tracking** — All runs logged to MLflow with parameters, metrics, and model artifacts for reproducibility.
9. **Explainability** — SHAP values computed in the evaluation notebook to show per-feature contribution to predictions.

### Deliverables
- **REST API** (FastAPI): `/predict`, `/batch-predict`, `/model-info`, `/analytics` endpoints with Pydantic validation, fire-and-forget MongoDB logging.
- **Interactive Dashboard** (Streamlit, 3 tabs):
  - **Tab 1 — Lead Scorer**: Form with human-readable labels and probability tier classification (High / Medium / Low).
  - **Tab 2 — Strategy Copilot**: RAG-powered chatbot with suggested questions, copy-to-clipboard, timestamps, and clear-chat.
  - **Tab 3 — Analytics**: Live KPI cards, lead tier donut chart, probability trend chart, and recent strategy questions — all pulled from MongoDB Atlas in real time.
- **RAG Knowledge Base** (LlamaIndex + OpenAI): Indexed financial documents covering campaign strategy, GDPR/MiFID II compliance, best practices, and dataset insights.
- **MongoDB Atlas Logging**: Every prediction and research query logged with structured input/output and human-readable `logged_at` timestamp.
- **Docker Compose Orchestration**: One-command deployment of API + Streamlit + MLflow as separate containers.
- **Testing & CI/CD**: 27 pytest tests covering data pipeline, models, and API. GitHub Actions runs artifact-free tests on every push.

### Business Value
- **Cost savings**: Prioritizing high-probability leads reduces call center expenses (each call costs ~€3 in agent time).
- **Revenue impact**: Better conversion rates on a smaller targeted outreach — the model achieves a 4.3× lift over random targeting.
- **Regulatory alignment**: The RAG chatbot surfaces compliance considerations (GDPR consent, MiFID II disclosures, Banco de Portugal contact rules) alongside predictions.
- **Explainability**: SHAP values and documented feature engineering decisions support fair-lending reviews and customer explanation requirements.

---

## 🎯 Results — Model Comparison

All models trained on the UCI Bank Marketing dataset (41,188 rows, 11% positive class). Evaluation on a held-out test set (8,238 rows). Decision threshold tuned on a separate validation set to maximise F1 — **not** the test set.

| Model | Val PR-AUC | Test PR-AUC | Test F1 | Test ROC-AUC | Threshold |
|---|---|---|---|---|---|
| Logistic Regression | 0.4485 | 0.4594 | 0.5050 | 0.8010 | 0.69 |
| Random Forest | 0.4640 | 0.4720 | 0.5228 | 0.8078 | 0.41 |
| XGBoost | 0.4650 | 0.4747 | 0.5267 | 0.8035 | 0.84 |
| **LightGBM (Best)** ✅ | **0.4736** | **0.4763** | **0.5242** | **0.8056** | **0.23** |

**Why LightGBM won:** LightGBM's histogram-based gradient boosting handles the mix of ordinal, nominal, and continuous features efficiently and benefits from the explicit ordinal encoding of `education`. It achieved the highest PR-AUC on the validation set — the correct selection criterion for an imbalanced dataset.

**Why threshold tuning mattered:** The optimal thresholds range from `0.23` to `0.84` across models — far from the naive default of `0.5`. Using `0.5` on LightGBM would misclassify the majority of the minority class. Threshold selection is part of the model, not an afterthought.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  OFFLINE PIPELINE                                                   │
│                                                                     │
│  bank-additional-full.csv                                           │
│         │                                                           │
│         ▼                                                           │
│  src/data_processing.py  ──►  data/processed/  ──►  models/        │
│  • drop duration (leakage)     train.parquet        preprocessor.pkl│
│  • recode pdays 999→-1         val.parquet          feature_names.json│
│  • ColumnTransformer (fit      test.parquet                         │
│    on train only)                │                                  │
│  • SMOTE on train only           ▼                                  │
│                           src/train.py                              │
│                           • 4 models from config.yaml               │
│                           • threshold tuning on val                 │
│                           • MLflow experiment tracking              │
│                                  │                                  │
│                                  ▼                                  │
│                           models/best_model.pkl                     │
│                           models/best_model_metadata.json           │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PRODUCTION DEPLOYMENT                                               │
│                                                                      │
│  ┌────────────────────────────┐       ┌──────────────────────────┐  │
│  │  FastAPI  (HF Spaces)      │◄──────│  Streamlit  (Cloud)      │  │
│  │  POST /predict             │       │  Tab 1: Lead Scorer      │  │
│  │  POST /batch-predict       │       │  Tab 2: Strategy Copilot │  │
│  │  POST /research (RAG)      │       │  Tab 3: Analytics        │  │
│  │  GET  /analytics           │       └──────────────────────────┘  │
│  │  GET  /model-info          │                                      │
│  │  GET  /  (health)          │       ┌──────────────────────────┐  │
│  │                            │──────►│  MongoDB Atlas           │  │
│  │  LightGBM model            │       │  prediction_logs         │  │
│  │  LlamaIndex RAG engine     │       │  research_logs           │  │
│  └────────────────────────────┘       └──────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘

  LOCAL / DOCKER-COMPOSE (development)
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │ FastAPI:8000 │  │Streamlit:8501│  │ MLflow:5000  │
  └──────────────┘  └──────────────┘  └──────────────┘
```

Full technical design: [docs/architecture.md](docs/architecture.md)

---

## 🔧 Key Engineering Decisions

- **Saved `ColumnTransformer`** — `preprocessor.pkl` is fit on training data only and used at inference time. Eliminates training/serving skew: the API's `preprocessor.transform(df)` is byte-for-byte identical to what the model was trained on.

- **`pdays 999 → -1` recode** — `999` is a sentinel for "never previously contacted". Left as-is, StandardScaler would treat it as an extreme outlier (~70× the real max), corrupting the scaled feature for all models.

- **`duration` column dropped** — Call duration is only known *after* the call ends, at which point the outcome is already known. Including it produces benchmark inflation, not a useful predictor.

- **PR-AUC as model selection metric** — With only 11% positive class rate, ROC-AUC is misleading (inflated by true negatives). PR-AUC focuses purely on precision/recall trade-offs for the minority class — the commercially important one.

- **Threshold tuning on held-out val set** — The test set is used exactly once for final reporting. All tuning decisions (including threshold) are made on the validation split.

- **SMOTE only on training data** — Synthetic oversampling before splitting would generate test samples from training neighbourhoods, making evaluation metrics falsely optimistic.

- **`OrdinalEncoder` for `education`** — Education has a natural progression (`illiterate → university.degree`). OHE would discard this semantic order; ordinal encoding preserves it for meaningful model splits.

- **Fire-and-forget MongoDB logging** — `asyncio.create_task()` is used for all DB writes so MongoDB latency never blocks API response time. Logging failures are caught and printed — they never propagate to the client.

- **Two-branch deployment** — `main` branch serves GitHub CI and Streamlit Cloud (`app.py`). `deploy-branch` is force-pushed to the `hf` remote for the HF Spaces Docker backend. Keeps CI clean and HF config separate.

---

## 📁 Project Structure

```
FinSight_AI/
├── README.md
├── requirements.txt
├── Dockerfile                    # Multi-stage: builder + slim runtime
├── docker-compose.yml            # api + streamlit + mlflow
├── app.py                        # Streamlit Cloud entry point (calls HF backend)
├── config/
│   └── config.yaml               # Single source of truth for all parameters
├── data/
│   ├── raw/                      # bank-additional-full.csv (gitignored)
│   └── processed/                # Parquet splits (gitignored)
├── notebooks/
│   ├── 01_statistical_analysis.ipynb
│   ├── 02_eda_plotly.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_interpretation.ipynb
├── src/
│   ├── data_processing.py        # Full feature engineering pipeline
│   ├── statistical_analysis.py   # Hypothesis tests, VIF, logit summary
│   ├── train.py                  # Multi-model training + MLflow tracking
│   ├── evaluate.py               # Metrics, ROC/PR/confusion plots
│   ├── predict.py                # Standalone inference (no server needed)
│   └── rag/
│       ├── indexer.py            # LlamaIndex document indexing
│       └── query_engine.py       # RAG query interface
├── api/
│   ├── main.py                   # FastAPI app factory + lifespan loader
│   ├── routes.py                 # All endpoint definitions
│   ├── schemas.py                # Pydantic input/output models
│   └── database.py               # Motor async MongoDB client + logging helpers
├── app/
│   ├── labels.py                 # UI display mappings for Streamlit
│   └── streamlit_app.py          # Standalone local demo (loads model directly)
├── docs/
│   ├── architecture.md           # System design + decision rationale
│   └── financial_reports/        # PDFs/TXTs indexed by RAG
├── models/                       # Trained artifacts (git LFS)
├── reports/figures/              # Generated plots (gitignored)
└── tests/
    ├── test_data_processing.py   # 8 tests (no artifacts needed)
    ├── test_models.py            # 8 tests (4 skip without artifacts)
    └── test_api.py               # 11 tests (skip without artifacts)
```

---

## 🚀 Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/nikaergemlidze1/finsight-ai.git
cd finsight-ai
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

The Strategy Copilot RAG chatbot uses OpenAI. Create `.env` from the template:

```bash
cp .env.example .env
```

Open `.env` and fill in:

```
OPENAI_API_KEY=sk-proj-your-actual-key-here
MONGO_URL=mongodb+srv://...   # optional — prediction/research logging
```

> Without `OPENAI_API_KEY`, Tab 1 (Lead Scorer) works normally. Tab 2 (Strategy Copilot) shows an auth error.
> Without `MONGO_URL`, all API endpoints work — MongoDB logging silently no-ops.

---

### 3. Download the dataset

Download `bank-additional-full.csv` from the [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) and place it at:

```
data/raw/bank-additional-full.csv
```

### 4. Run the training pipeline

```bash
# Feature engineering, splits, SMOTE
python -m src.data_processing

# Train 4 models, tune thresholds, log to MLflow, save best
python -m src.train

# (optional) Evaluate and generate plots
python -m src.evaluate
```

### 5. Launch the app

**Option A — Standalone demo (no Docker, no API server)**

```bash
streamlit run app/streamlit_app.py
```

Opens **http://localhost:8501**. Model loaded directly from `models/`.

---

**Option B — Production-like (Docker Compose)**

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Streamlit dashboard | http://localhost:8501 |
| FastAPI REST API | http://localhost:8000/docs |
| MLflow tracking | http://localhost:5000 |

---

### 6. Optional extras

```bash
# MLflow UI locally (without Docker)
mlflow ui --port 5001

# CLI prediction (no server needed)
python -m src.predict --json '{"age": 35, ...}'
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

**27 tests** across 3 files:
- `test_data_processing.py` — 8 tests, always run (no artifacts needed)
- `test_models.py` — 8 tests (4 always run; 4 require trained artifacts — auto-skip in CI)
- `test_api.py` — 11 tests (require trained artifacts — auto-skip in CI)

CI runs the 10 artifact-free tests on every push to `main`. LFS pointer detection prevents false positives on CI runners without real model binaries.

---

## 🛠️ Tech Stack

| Category | Technology | Version |
|---|---|---|
| Language | Python | 3.11 |
| ML Framework | scikit-learn | 1.5.1 |
| Gradient Boosting | XGBoost / LightGBM | 2.1.1 / 4.5.0 |
| Imbalance Handling | imbalanced-learn (SMOTE) | 0.12.3 |
| Model Interpretation | SHAP | 0.46.0 |
| Experiment Tracking | MLflow | 2.14.3 |
| Statistical Analysis | SciPy / statsmodels | 1.13.1 / 0.14.2 |
| Visualisation | Plotly / Matplotlib | 5.22.0 / 3.9.1 |
| API | FastAPI + Uvicorn | 0.111.1 / 0.30.3 |
| Data Validation | Pydantic v2 | 2.8.2 |
| Dashboard | Streamlit | 1.36.0 |
| RAG | LlamaIndex + OpenAI | 0.11.0 |
| Database | MongoDB Atlas + Motor | async |
| Containerisation | Docker + Compose | multi-stage |
| CI | GitHub Actions | — |
| Testing | pytest + httpx | 8.3.2 / 0.27.0 |
| Backend Hosting | Hugging Face Spaces | Docker SDK |
| Frontend Hosting | Streamlit Cloud | — |

---

## 📊 Dataset

**UCI Bank Marketing Dataset (Social/Economic Context)**
- **Source:** [Moro et al., 2014](https://doi.org/10.1016/j.dss.2014.03.001) — Portuguese retail bank, May 2008 – November 2010
- **Size:** 41,188 rows × 21 columns
- **Target:** `y` — did the client subscribe to a term deposit? (`yes` / `no`)
- **Class imbalance:** ~11% positive rate
- **Features:** 7 client demographics, 7 campaign-history fields, 5 macro-economic indicators (Euribor, employment rate, consumer confidence, etc.)
- **Notable:** `duration` column is a post-call data leak and is excluded from all production models

Full feature dictionary: [data/README.md](data/README.md)

---

## ⚠️ Limitations

- **Temporal scope:** Data is from 2008–2010 during the European financial crisis. Macro-economic indicators (Euribor, employment variation rate) reflect that specific period and may not generalise to current market conditions.
- **Geographic scope:** Portuguese retail banking market only. Customer behaviour and regulatory context differ across markets.
- **Task scope:** Binary classification (subscribe / not subscribe). Does not support multi-tier lead scoring, lifetime value estimation, or churn prediction.
- **Portfolio project:** This system is built for educational and portfolio demonstration purposes. It has not undergone production validation, A/B testing, or regulatory review.

---

## 📬 Contact

**Nika Ergemlidze** — Data Scientist / ML Engineer
- 🐙 [github.com/nikaergemlidze1](https://github.com/nikaergemlidze1)
- 💼 [linkedin.com/in/nika-ergemlidze](https://linkedin.com/in/nika-ergemlidze)

---

<div align="center">
  <sub>Built with Python 3.11 · UCI Bank Marketing Dataset · Moro et al., 2014</sub>
</div>
