"""
FinSight AI — Streamlit demo application.

Two tabs:
    1. Lead Scorer — form-based ML prediction (loads model directly)
    2. Strategy Copilot — RAG chatbot over financial documents

Runs standalone (no API server required). Artifacts are loaded via
src.predict.load_artifacts() and cached with @st.cache_resource.

Launch: streamlit run app/streamlit_app.py
"""

import streamlit as st
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
try:
    from src.rag.query_engine import get_query_engine
    from src.predict import load_artifacts, predict_one
    from app.labels import (
        EDUCATION_LABELS, JOB_LABELS, MARITAL_LABELS, YES_NO_UNKNOWN,
        CONTACT_LABELS, MONTH_LABELS, DAY_LABELS, POUTCOME_LABELS
    )
except ModuleNotFoundError:
    # Fallback if the app is run directly from inside the app/ directory instead of the project root
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.rag.query_engine import get_query_engine
    from src.predict import load_artifacts, predict_one
    from app.labels import (
        EDUCATION_LABELS, JOB_LABELS, MARITAL_LABELS, YES_NO_UNKNOWN,
        CONTACT_LABELS, MONTH_LABELS, DAY_LABELS, POUTCOME_LABELS
    )


# ── Cached artifact loader (runs once per Streamlit session) ─────────────────

@st.cache_resource(show_spinner="Loading prediction model…")
def _get_artifacts():
    """Load model + preprocessor + metadata from models/. Cached for the session."""
    return load_artifacts()   # (model, preprocessor, metadata, cfg)


# --- Page Config ---
st.set_page_config(page_title="FinSight AI", layout="wide", page_icon="🏦")

# Sidebar — explain standalone mode
with st.sidebar:
    st.markdown("## ℹ️ About")
    st.caption(
        "Streamlit runs predictions **locally** by loading `models/best_model.pkl` "
        "directly — no separate API server required for demo use.\n\n"
        "See `api/` for the deployable FastAPI version used in production."
    )
    try:
        _, _, meta, _ = _get_artifacts()
        st.success(f"Model loaded: **{meta['model_name']}**")
        st.caption(f"Val PR-AUC: {meta['val_pr_auc']} · Threshold: {meta['tuned_threshold']}")
    except Exception:
        st.warning("Model artifacts not found. Run the training pipeline first.")

# Create Tabs
tab1, tab2 = st.tabs(["📊 Lead Scoring", "🤖 Strategy Copilot"])

# --- TAB 1: MACHINE LEARNING LEAD SCORING ---
with tab1:
    st.title("🏦 FinSight AI: Telemarketing Lead Scorer")
    st.markdown("Enter a customer's profile to predict their likelihood of subscribing.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("👤 Demographics")
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        job = st.selectbox("Job", list(JOB_LABELS.keys()),
                           format_func=lambda x: JOB_LABELS[x])
        marital = st.selectbox("Marital Status", list(MARITAL_LABELS.keys()),
                               format_func=lambda x: MARITAL_LABELS[x])
        education = st.selectbox("Education", list(EDUCATION_LABELS.keys()),
                                 format_func=lambda x: EDUCATION_LABELS[x],
                                 index=list(EDUCATION_LABELS.keys()).index("university.degree"))

    with col2:
        st.header("💰 Financials")
        default = st.selectbox("Has Credit in Default?", list(YES_NO_UNKNOWN.keys()),
                               format_func=lambda x: YES_NO_UNKNOWN[x])
        housing = st.selectbox("Has Housing Loan?", list(YES_NO_UNKNOWN.keys()),
                               format_func=lambda x: YES_NO_UNKNOWN[x])
        loan = st.selectbox("Has Personal Loan?", list(YES_NO_UNKNOWN.keys()),
                            format_func=lambda x: YES_NO_UNKNOWN[x])

        st.header("📞 Campaign Data")
        contact = st.selectbox("Contact Method", list(CONTACT_LABELS.keys()),
                               format_func=lambda x: CONTACT_LABELS[x])
        month = st.selectbox("Last Contact Month", list(MONTH_LABELS.keys()),
                             format_func=lambda x: MONTH_LABELS[x],
                             index=list(MONTH_LABELS.keys()).index("may"))
        day_of_week = st.selectbox("Last Contact Day", list(DAY_LABELS.keys()),
                                   format_func=lambda x: DAY_LABELS[x])

    with col3:
        st.header("📈 Macro-Economics")
        emp_var_rate = st.number_input("Employment Variation Rate", value=-1.8)
        cons_price_idx = st.number_input("Consumer Price Index", value=92.893)
        cons_conf_idx = st.number_input("Consumer Confidence Index", value=-46.2)
        euribor3m = st.number_input("Euribor 3-Month Rate", value=1.299)
        nr_employed = st.number_input("Number of Employees", value=5099.1)
        
        st.markdown("---")
        campaign = st.number_input("Contacts this campaign", min_value=1, max_value=50, value=1)
        previous = st.number_input("Contacts in prior campaigns", min_value=0, max_value=20, value=0)
        poutcome = st.selectbox("Previous campaign outcome", list(POUTCOME_LABELS.keys()),
                                format_func=lambda x: POUTCOME_LABELS[x])
        
        # pdays=999 is the UCI dataset's sentinel for "never previously contacted".
        # The data pipeline (src/data_processing.py) recodes this to -1 for model input.
        # We keep 999 here so the API layer performs the recode consistently.
        pdays = 999  

    if st.button("Predict Subscription Probability", type="primary", use_container_width=True):
        payload = {
            "age": age, "job": job, "marital": marital, "education": education,
            "default": default, "housing": housing, "loan": loan,
            "contact": contact, "month": month, "day_of_week": day_of_week,
            "campaign": campaign, "pdays": pdays, "previous": previous, "poutcome": poutcome,
            "emp.var.rate": emp_var_rate, "cons.price.idx": cons_price_idx,
            "cons.conf.idx": cons_conf_idx, "euribor3m": euribor3m, "nr.employed": nr_employed,
        }
        try:
            model, prep, meta, cfg = _get_artifacts()
            result = predict_one(payload, model, prep, meta, cfg)
            prob = result["probability_of_subscription"] * 100
            if result["prediction_class"] == 1:
                st.success(f"### 📞 {result['recommendation']}")
            else:
                st.warning(f"### 🛑 {result['recommendation']}")
            col_a, col_b = st.columns(2)
            col_a.metric("Subscription Probability", f"{prob:.1f}%")
            col_b.metric("Decision Threshold", f"{result['threshold_used']:.2f}")
        except FileNotFoundError as e:
            st.error(
                f"**Model artifacts not found.**\n\n{e}\n\n"
                "Run the training pipeline first:\n"
                "```\npython -m src.data_processing\npython -m src.train\n```"
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# --- TAB 2: GEN AI STRATEGY COPILOT ---
with tab2:
    st.title("🤖 FinSight Strategy Copilot")
    st.info(
        "Ask about campaign strategies, regulatory compliance, best practices, "
        "or insights from our Portuguese bank marketing dataset."
    )

    # ── Example question chips ────────────────────────────────────────────────
    EXAMPLE_QUESTIONS = [
        "What are the GDPR penalties for non-compliance?",
        "What are the best times to call customers?",
        "Why was the duration feature removed from the model?",
        "How should we handle objections during calls?",
    ]

    st.caption("💡 Try an example question:")
    chip_cols = st.columns(len(EXAMPLE_QUESTIONS))
    clicked_question = None
    for col, question in zip(chip_cols, EXAMPLE_QUESTIONS):
        if col.button(question, use_container_width=True):
            clicked_question = question

    st.divider()

    # ── Initialize RAG engine once per session ────────────────────────────────
    if "query_engine" not in st.session_state:
        with st.spinner("Initializing AI Copilot…"):
            try:
                st.session_state.query_engine = get_query_engine()
                st.session_state.rag_ready = True
            except FileNotFoundError as e:
                st.session_state.query_engine = None
                st.session_state.rag_ready = False
                st.error(
                    f"**RAG index not found.**\n\n{e}\n\n"
                    "Run `python -m src.rag.indexer` to build the knowledge base."
                )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Chat history display ──────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # ── Resolve prompt: typed input OR chip click ─────────────────────────────
    typed_prompt = st.chat_input("Ask anything about the campaign or dataset…")
    prompt = clicked_question or typed_prompt

    if prompt and st.session_state.get("rag_ready"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("AI is thinking…"):
                response = st.session_state.query_engine.query(prompt)
                full_response = str(response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

            with st.chat_message("assistant"):
                st.markdown(full_response)

        st.rerun()