"""
FinSight AI — Production Streamlit Frontend.

Connects to the FastAPI backend hosted on Hugging Face for:
    1. Lead Scorer — ML predictions via POST /predict
    2. Strategy Copilot — RAG chatbot via POST /chat

Launch: streamlit run app/streamlit_app.py
"""

import streamlit as st
import requests
import os
import sys
from dotenv import load_dotenv

# Load environment variables (BACKEND_URL)
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "https://nikollass-finsight-ai-backend.hf.space")

# Ensure app can find labels even if run from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.labels import (
    EDUCATION_LABELS, JOB_LABELS, MARITAL_LABELS, YES_NO_UNKNOWN,
    CONTACT_LABELS, MONTH_LABELS, DAY_LABELS, POUTCOME_LABELS
)

# --- Page Config ---
st.set_page_config(page_title="FinSight AI", layout="wide", page_icon="🏦")

# ── Sidebar: API Health & Info ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ℹ️ System Status")
    
    try:
        # Ping the /health endpoint we verified in your logs
        health_resp = requests.get(f"{BACKEND_URL}/", timeout=5)
        health_data = health_resp.json()
        
        if health_data.get("status") == "active":
            st.success("● Backend Online")
            st.caption(f"**Model:** {health_data.get('model_name')} (v2.1.1)")
            st.caption(f"**Threshold:** {health_data.get('tuned_threshold'):.2f}")
            st.caption(f"**PR-AUC:** {health_data.get('val_pr_auc'):.4f}")
        else:
            st.warning("○ Backend Starting...")
    except Exception:
        st.error("○ Backend Offline")
        st.info("Check if BACKEND_URL is correct in .env")

    st.divider()
    st.caption("Architecture: Streamlit (Frontend) ↔ FastAPI (Backend)")

# Create Tabs
tab1, tab2 = st.tabs(["📊 Lead Scoring", "🤖 Strategy Copilot"])

# --- TAB 1: MACHINE LEARNING LEAD SCORING ---
with tab1:
    st.title("🏦 Lead Scorer")
    st.markdown("Predict customer subscription likelihood using the live LightGBM pipeline.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("👤 Demographics")
        age = st.number_input("Age", 18, 100, 35)
        job = st.selectbox("Job", list(JOB_LABELS.keys()), format_func=lambda x: JOB_LABELS[x])
        marital = st.selectbox("Marital", list(MARITAL_LABELS.keys()), format_func=lambda x: MARITAL_LABELS[x])
        education = st.selectbox("Education", list(EDUCATION_LABELS.keys()), 
                                 format_func=lambda x: EDUCATION_LABELS[x], index=6)

    with col2:
        st.header("💰 Financials")
        default = st.selectbox("Default?", list(YES_NO_UNKNOWN.keys()), format_func=lambda x: YES_NO_UNKNOWN[x])
        housing = st.selectbox("Housing Loan?", list(YES_NO_UNKNOWN.keys()), format_func=lambda x: YES_NO_UNKNOWN[x])
        loan = st.selectbox("Personal Loan?", list(YES_NO_UNKNOWN.keys()), format_func=lambda x: YES_NO_UNKNOWN[x])

        st.header("📞 Campaign")
        contact = st.selectbox("Method", list(CONTACT_LABELS.keys()), format_func=lambda x: CONTACT_LABELS[x])
        month = st.selectbox("Month", list(MONTH_LABELS.keys()), format_func=lambda x: MONTH_LABELS[x], index=4)
        day_of_week = st.selectbox("Day", list(DAY_LABELS.keys()), format_func=lambda x: DAY_LABELS[x])

    with col3:
        st.header("📈 Macro-Economics")
        emp_var_rate = st.number_input("Emp. Var Rate", value=-1.8)
        cons_price_idx = st.number_input("CPI", value=92.893)
        cons_conf_idx = st.number_input("Confidence Index", value=-46.2)
        euribor3m = st.number_input("Euribor 3M", value=1.299)
        nr_employed = st.number_input("Nr. Employed", value=5099.1)
        
        campaign = st.number_input("Contacts (current)", 1, 50, 1)
        previous = st.number_input("Contacts (prior)", 0, 20, 0)
        poutcome = st.selectbox("Prior Outcome", list(POUTCOME_LABELS.keys()), format_func=lambda x: POUTCOME_LABELS[x])

    if st.button("Calculate Probability", type="primary", use_container_width=True):
        payload = {
            "age": age, "job": job, "marital": marital, "education": education,
            "default": default, "housing": housing, "loan": loan,
            "contact": contact, "month": month, "day_of_week": day_of_week,
            "campaign": campaign, "pdays": 999, "previous": previous, "poutcome": poutcome,
            "emp.var.rate": emp_var_rate, "cons.price.idx": cons_price_idx,
            "cons.conf.idx": cons_conf_idx, "euribor3m": euribor3m, "nr.employed": nr_employed,
        }
        
        try:
            # POST to the FastAPI backend
            response = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=10)
            result = response.json()
            
            prob = result["probability_of_subscription"] * 100
            
            if result["prediction_class"] == 1:
                st.success(f"### 📞 {result['recommendation']}")
            else:
                st.warning(f"### 🛑 {result['recommendation']}")
            
            c_a, c_b = st.columns(2)
            c_a.metric("Sub Probability", f"{prob:.1f}%")
            c_b.metric("Model Threshold", f"{result['threshold_used']:.2f}")
            
        except Exception as e:
            st.error(f"Connection Error: Ensure your API at {BACKEND_URL} is running.")

# --- TAB 2: GEN AI STRATEGY COPILOT ---
with tab2:
    st.title("🤖 Strategy Copilot")
    st.info("Ask about campaign strategies, compliance, or insights from the dataset.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("How can we improve conversion for retirees?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Querying FinSight RAG Engine..."):
                try:
                    # POST to the FastAPI chat endpoint
                    resp = requests.post(f"{BACKEND_URL}/research", json={"query": prompt}, timeout=30)
                    full_response = resp.json().get("answer", "No response from AI.")
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception:
                    st.error("Chat Error: Could not connect to the RAG engine on the backend.")