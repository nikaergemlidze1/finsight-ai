# 🏦 FinSight AI — Bank Marketing Predictor

> Production-grade ML pipeline predicting term deposit subscriptions from bank telemarketing campaign data.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.5-brightgreen)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikitlearn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.14-0194E2?logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-multi--stage-2496ED?logo=docker&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-27%20tests-success?logo=pytest&logoColor=white)

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
- **REST API** (FastAPI): `/predict`, `/batch-predict`, `/model-info`, `/health` endpoints with Pydantic validation.
- **Interactive Dashboard** (Streamlit): Lead Scorer form with human-readable labels + RAG-powered Strategy Copilot chatbot.
- **RAG Knowledge Base** (LlamaIndex + OpenAI): Indexed financial documents covering campaign strategy, GDPR/MiFID II compliance, best practices, and dataset insights. Returns context-grounded answers with source attribution.
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
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
┌─────────────────────────────┐   ┌───────────────────────────────┐
│  FastAPI  :8000             │   │  Streamlit  :8501             │
│  POST /predict              │◄──│  Tab 1: Lead Scorer           │
│  POST /batch-predict        │   │  Tab 2: Strategy Copilot      │
│  GET  /model-info           │   │  (LlamaIndex RAG over         │
│                             │   │   financial documents)        │
│  preprocessor.pkl           │   └───────────────────────────────┘
│  best_model.pkl             │
└─────────────────────────────┘
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

---

## 📁 Project Structure

```
FinSight_AI/
├── README.md
├── requirements.txt
├── Dockerfile                    # Multi-stage: builder + slim runtime
├── docker-compose.yml            # api + streamlit + mlflow
├── config/
│   └── config.yaml               # Single source of truth for all parameters
├── data/
│   ├── raw/                      # bank-additional-full.csv (gitignored)
│   └── processed/                # Parquet splits + preprocessor (gitignored)
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
│   ├── predict.py                # CLI inference without running the server
│   └── rag/
│       ├── indexer.py            # LlamaIndex document indexing
│       └── query_engine.py       # RAG query interface
├── api/
│   ├── main.py                   # FastAPI app factory + lifespan loader
│   ├── routes.py                 # All endpoint definitions
│   └── schemas.py                # Pydantic input/output models
├── app/
│   ├── __init__.py               # Package initialization
│   ├── labels.py                 # UI display mappings for Streamlit
│   └── streamlit_app.py          # Multi-tab interactive dashboard
├── docs/
│   ├── architecture.md           # System design + decision rationale
│   └── financial_reports/        # PDFs/TXTs for RAG component
├── models/                       # Trained artifacts (gitignored)
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
git clone https://github.com/nikaergemlidze/FinSight_AI.git
cd FinSight_AI
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 1b. Set up API keys (for RAG chatbot)

The Strategy Copilot RAG chatbot uses OpenAI for answer generation. Create a `.env` file from the template:

```bash
cp .env.example .env
```

Then open `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-proj-your-actual-key-here
```

Get a key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys) if you don't have one.

> **Note:** Without this, Tab 1 (Lead Scorer) will work normally, but Tab 2 (Strategy Copilot) will show an authentication error.

---

### 2. Download the dataset

Download `bank-additional-full.csv` from the [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) and place it at:

```
data/raw/bank-additional-full.csv
```

### 3. Run the training pipeline

```bash
# Step 1 — Feature engineering, splits, SMOTE (saves to data/processed/ and models/)
python -m src.data_processing

# Step 2 — Train 4 models, tune thresholds, log to MLflow, save best model
python -m src.train

# Step 3 (optional) — Evaluate and generate plots to reports/figures/
python -m src.evaluate
```

### 4. Launch the demo

There are two ways to run FinSight AI. Choose the one that fits your goal:

---

#### ⚡ Option A — Simple demo (one command, no Docker required)

Streamlit loads the trained model directly from `models/` — no API server needed.

```bash
streamlit run app/streamlit_app.py
```

Open **http://localhost:8501** to use the Lead Scorer and Strategy Copilot.

---

#### 🐳 Option B — Production-like deployment (Docker Compose)

Runs FastAPI, Streamlit, and MLflow as separate containers on a shared network.

```bash
docker compose up --build
```

| Service | URL | Notes |
|---|---|---|
| Streamlit dashboard | http://localhost:8501 | Full UI — waits for API healthcheck |
| FastAPI REST API | http://localhost:8000 | Swagger UI at `/docs` |
| MLflow tracking | http://localhost:5000 | Experiment history |

All three services come up automatically. Streamlit loads predictions locally from `models/`; the FastAPI service is available separately for programmatic access.

---

### 5. Optional extras

```bash
# MLflow UI locally (without Docker)
mlflow ui --port 5001
# -> Then open http://localhost:5001

# CLI prediction (no server needed)
python -m src.predict --json '...'
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

**27 tests** across 3 files:
- `test_data_processing.py` — 8 tests, all run without trained artifacts
- `test_models.py` — 8 tests (4 run always; 4 require trained artifacts)
- `test_api.py` — 11 tests (require trained artifacts)

CI runs 10 artifact-free tests on every push. All 27 pass after the full pipeline has been run.

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
| RAG | LlamaIndex | 0.11.0 |
| Containerisation | Docker + Compose | multi-stage |
| CI | GitHub Actions + ruff | — |
| Testing | pytest + httpx | 8.3.2 |

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

**Nika Ergemlidze** — Data Scientist/Analyst/Data Engineer & AI/ML Engineer
- 🐙 [github.com/nikaergemlidze1](https://github.com/nikaergemlidze1)
- 💼 [linkedin.com/in/nika-ergemlidze](https://linkedin.com/in/nika-ergemlidze)

---

<div align="center">
  <sub>Built with Python 3.11 · UCI Bank Marketing Dataset · Moro et al., 2014</sub>
</div>
