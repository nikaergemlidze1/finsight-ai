import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Page config must be first Streamlit call ---
st.set_page_config(page_title="FinSight AI", layout="wide", page_icon="🏦")

# --- Backend URL: safe fallback, no st.stop() ---
BACKEND_URL = st.secrets.get(
    "BACKEND_URL",
    "https://nikollass-finsight-ai-backend.hf.space",
)

# --- Initialize session state up front ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- HTTP session with retries + connection reuse ---
@st.cache_resource
def _http_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=2, backoff_factor=1.5,
        status_forcelist=[502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

# --- Warm the HF Space on first load ---
@st.cache_resource(show_spinner=False)
def _warm_backend() -> bool:
    try:
        _http_session().get(f"{BACKEND_URL}/", timeout=5)
        return True
    except requests.RequestException:
        return False

BACKEND_WARM = _warm_backend()

# ═══════════════════════════════════════════════════════════════════════════
# HUMAN-READABLE LABEL MAPS — UI shows the value, backend receives the key
# ═══════════════════════════════════════════════════════════════════════════

JOB_LABELS = {
    "admin.": "Administrative",
    "blue-collar": "Blue Collar",
    "entrepreneur": "Entrepreneur",
    "housemaid": "Housemaid",
    "management": "Management",
    "retired": "Retired",
    "self-employed": "Self-Employed",
    "services": "Services",
    "student": "Student",
    "technician": "Technician",
    "unemployed": "Unemployed",
    "unknown": "Unknown",
}

MARITAL_LABELS = {
    "divorced": "Divorced",
    "married": "Married",
    "single": "Single",
    "unknown": "Unknown",
}

EDUCATION_LABELS = {
    "basic.4y": "Primary (4 years)",
    "basic.6y": "Primary (6 years)",
    "basic.9y": "Lower Secondary (9 years)",
    "high.school": "High School",
    "illiterate": "Illiterate",
    "professional.course": "Professional Course",
    "university.degree": "University Degree",
    "unknown": "Unknown",
}

YES_NO_LABELS = {
    "no": "No",
    "yes": "Yes",
    "unknown": "Unknown",
}

CONTACT_LABELS = {
    "cellular": "Cellular",
    "telephone": "Telephone",
}

MONTH_LABELS = {
    "jan": "January", "feb": "February", "mar": "March",
    "apr": "April", "may": "May", "jun": "June",
    "jul": "July", "aug": "August", "sep": "September",
    "oct": "October", "nov": "November", "dec": "December",
}

# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — better chat message distinction
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <style>
    /* User messages: right-aligned, blue tint */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: rgba(59, 130, 246, 0.08);
        border-left: 3px solid #3B82F6;
        margin-left: 15%;
    }
    /* Assistant messages: left-aligned, warm tint */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background: rgba(251, 191, 36, 0.06);
        border-left: 3px solid #FBBF24;
        margin-right: 15%;
    }
    /* Tighter spacing between messages */
    [data-testid="stChatMessage"] {
        padding: 0.6rem 1rem;
        margin-bottom: 0.4rem;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════

st.title("🏦 FinSight AI: Financial Intelligence Suite")
st.caption(
    "Advanced analytics and RAG-powered strategy for modern banking. "
    "*Backend on HF Spaces free tier — first request may take ~20s if the Space is waking up.*"
)

tab1, tab2 = st.tabs(["📊 Lead Scoring", "🤖 Strategy Copilot"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: LEAD SCORING (with human-readable labels)
# ═══════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("Telemarketing Lead Scorer")

    with st.form("lead_scorer", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("👤 Demographics")
            age = st.number_input("Age", 18, 100, 35)
            job = st.selectbox(
                "Job",
                options=list(JOB_LABELS.keys()),
                format_func=lambda k: JOB_LABELS[k],
            )
            marital = st.selectbox(
                "Marital Status",
                options=list(MARITAL_LABELS.keys()),
                format_func=lambda k: MARITAL_LABELS[k],
            )
            education = st.selectbox(
                "Education",
                options=list(EDUCATION_LABELS.keys()),
                format_func=lambda k: EDUCATION_LABELS[k],
            )

        with col2:
            st.subheader("💰 Financials")
            default = st.selectbox(
                "Has Credit in Default?",
                options=list(YES_NO_LABELS.keys()),
                format_func=lambda k: YES_NO_LABELS[k],
            )
            housing = st.selectbox(
                "Has Housing Loan?",
                options=list(YES_NO_LABELS.keys()),
                format_func=lambda k: YES_NO_LABELS[k],
            )
            loan = st.selectbox(
                "Has Personal Loan?",
                options=list(YES_NO_LABELS.keys()),
                format_func=lambda k: YES_NO_LABELS[k],
            )

            st.subheader("📞 Campaign")
            contact = st.selectbox(
                "Contact Method",
                options=list(CONTACT_LABELS.keys()),
                format_func=lambda k: CONTACT_LABELS[k],
            )
            month = st.selectbox(
                "Month",
                options=list(MONTH_LABELS.keys()),
                format_func=lambda k: MONTH_LABELS[k],
            )

        with col3:
            st.subheader("📈 Macro-Economics")
            emp_var_rate = st.number_input("Emp. Var. Rate", value=-1.8)
            cons_price_idx = st.number_input("Cons. Price Index", value=92.893)
            euribor3m = st.number_input("Euribor 3-Month", value=1.299)
            nr_employed = st.number_input("Nr. Employed", value=5099.1)

        submitted = st.form_submit_button("Predict Subscription Probability", type="primary")

    if submitted:
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
        with st.spinner("Analyzing lead..."):
            try:
                response = _http_session().post(
                    f"{BACKEND_URL}/predict", json=payload, timeout=25
                )
                if response.status_code == 200:
                    res = response.json()
                    prob = res.get("probability_of_subscription", 0) * 100
                    st.metric("Subscription Probability", f"{prob:.1f}%")
                    st.success(f"Recommendation: {res.get('recommendation', 'N/A')}")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except requests.exceptions.Timeout:
                st.warning("Backend is waking up (HF Space cold start). Please click Predict again in a few seconds.")
            except Exception as e:
                st.error(f"Connection failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: STRATEGY COPILOT (compact empty state, adaptive history)
# ═══════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("🤖 Strategy Copilot")
    st.caption("Ask about campaign strategies, GDPR/MiFID II compliance, or bank marketing best practices.")

    # Compact empty state — only visible when there's no history
    if not st.session_state.messages:
        st.info("👋 Start the conversation by typing your first question below.")
    else:
        # Adaptive-height container — grows with content up to a sensible cap
        history_height = min(400, max(200, len(st.session_state.messages) * 85))
        history_container = st.container(height=history_height)
        with history_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

    prompt = st.chat_input("Ask about campaign strategies...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Consulting knowledge base..."):
            try:
                resp = _http_session().post(
                    f"{BACKEND_URL}/research",
                    json={"query": prompt},
                    timeout=60,
                )
                if resp.status_code == 200:
                    answer = resp.json().get("answer", "No response.")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                else:
                    st.session_state.messages.append(
                        {"role": "assistant",
                         "content": "⚠️ Backend error. It may be waking up — try again in 20s."}
                    )
            except requests.exceptions.Timeout:
                st.session_state.messages.append(
                    {"role": "assistant",
                     "content": "⚠️ Backend timeout. The RAG engine is still loading — try again shortly."}
                )
            except Exception as e:
                st.session_state.messages.append(
                    {"role": "assistant",
                     "content": f"⚠️ Error: {e}"}
                )
        # Force a rerun so the new messages appear immediately
        st.rerun()