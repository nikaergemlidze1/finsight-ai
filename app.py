import streamlit as st
import requests

# --- Page Configuration (must be first Streamlit call) ---
st.set_page_config(page_title="FinSight AI", layout="wide", page_icon="🏦")

# --- Backend URL: secret with safe fallback, no st.stop() ---
BACKEND_URL = st.secrets.get(
    "BACKEND_URL",
    "https://nikollass-finsight-ai-backend.hf.space"
)

# --- Initialize ALL session state before any tabs or widgets ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- UI Header ---
st.title("🏦 FinSight AI: Financial Intelligence Suite")
st.markdown("Advanced analytics and RAG-powered strategy for modern banking.")

tab1, tab2 = st.tabs(["📊 Lead Scoring", "🤖 Strategy Copilot"])

# --- TAB 1: MACHINE LEARNING LEAD SCORING ---
with tab1:
    st.header("Telemarketing Lead Scorer")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("👤 Demographics")
        age = st.number_input("Age", 18, 100, 35)
        job = st.selectbox("Job", ["admin.", "blue-collar", "entrepreneur", "housemaid",
                                   "management", "retired", "self-employed", "services",
                                   "student", "technician", "unemployed", "unknown"])
        marital = st.selectbox("Marital Status", ["divorced", "married", "single", "unknown"])
        education = st.selectbox("Education", ["basic.4y", "basic.6y", "basic.9y",
                                               "high.school", "illiterate",
                                               "professional.course", "university.degree",
                                               "unknown"])

    with col2:
        st.subheader("💰 Financials")
        default = st.selectbox("Has Credit in Default?", ["no", "yes", "unknown"])
        housing = st.selectbox("Has Housing Loan?", ["no", "yes", "unknown"])
        loan = st.selectbox("Has Personal Loan?", ["no", "yes", "unknown"])

        st.subheader("📞 Campaign")
        contact = st.selectbox("Contact Method", ["cellular", "telephone"])
        month = st.selectbox("Month", ["jan", "feb", "mar", "apr", "may", "jun",
                                       "jul", "aug", "sep", "oct", "nov", "dec"])

    with col3:
        st.subheader("📈 Macro-Economics")
        emp_var_rate = st.number_input("Emp. Var. Rate", value=-1.8)
        cons_price_idx = st.number_input("Cons. Price Index", value=92.893)
        euribor3m = st.number_input("Euribor 3-Month", value=1.299)
        nr_employed = st.number_input("Nr. Employed", value=5099.1)

    if st.button("Predict Subscription Probability", type="primary"):
        payload = {
            "age": age, "job": job, "marital": marital, "education": education,
            "default": default, "housing": housing, "loan": loan,
            "contact": contact, "month": month, "day_of_week": "mon",
            "campaign": 1, "pdays": 999, "previous": 0, "poutcome": "nonexistent",
            "emp.var.rate": emp_var_rate,
            "cons.price.idx": cons_price_idx,
            "cons.conf.idx": -46.2,
            "euribor3m": euribor3m,
            "nr.employed": nr_employed,
        }
        with st.spinner("Analyzing Lead..."):
            try:
                response = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=30)
                if response.status_code == 200:
                    res = response.json()
                    prob = res.get("probability_of_subscription", 0) * 100
                    st.metric("Subscription Probability", f"{prob:.1f}%")
                    st.success(f"Recommendation: {res.get('recommendation', 'N/A')}")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

# --- TAB 2: GEN AI STRATEGY COPILOT ---
with tab2:
    st.header("🤖 Strategy Copilot")

    # Render chat history (session state already initialized above)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask about campaign strategies...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consulting knowledge base..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/research",
                        json={"query": prompt},
                        timeout=90,
                    )
                    if resp.status_code == 200:
                        answer = resp.json().get("answer", "No response.")
                        st.markdown(answer)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer}
                        )
                    else:
                        st.error("Backend error. It might be waking up.")
                except Exception as e:
                    st.error(f"Error: {e}")
