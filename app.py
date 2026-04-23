import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Page config must be first Streamlit call ---
st.set_page_config(page_title="FinSight AI", layout="wide", page_icon="🏦")
# --- Hide cosmetic "Bad message format" popup (Streamlit upstream bug #9767) ---
st.markdown(
    """
    <style>
    div[data-testid="stStatusWidget"] div[role="alert"]:has(> div:first-child:contains("Bad message format")) {
        display: none !important;
    }
    div[role="dialog"]:has(*:contains("SessionInfo")) {
        display: none !important;
    }
    </style>
    <script>
    (function() {
        const observer = new MutationObserver(() => {
            document.querySelectorAll('div').forEach(el => {
                if (el.textContent && el.textContent.includes('Tried to use SessionInfo before it was initialized')) {
                    let parent = el;
                    for (let i = 0; i < 6 && parent; i++) {
                        if (parent.getAttribute('role') === 'dialog' ||
                            parent.getAttribute('data-testid')?.includes('Dialog') ||
                            parent.getAttribute('data-testid')?.includes('Toast')) {
                            parent.style.display = 'none';
                            break;
                        }
                        parent = parent.parentElement;
                    }
                }
            });
        });
        observer.observe(document.body, { childList: true, subtree: true });
    })();
    </script>
    """,
    unsafe_allow_html=True,
)

# --- Backend URL: safe fallback, no st.stop() ---
BACKEND_URL = st.secrets.get(
    "BACKEND_URL",
    "https://nikollass-finsight-ai-backend.hf.space",
)
# --- Backend URL: safe fallback, no st.stop() ---
BACKEND_URL = st.secrets.get(
    "BACKEND_URL",
    "https://nikollass-finsight-ai-backend.hf.space",
)

# --- Initialize ALL session state up front ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Reusable HTTP session with retries + connection reuse ---
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

# --- Wake the HF Space on first load (cached per session) ---
@st.cache_resource(show_spinner=False)
def _warm_backend() -> bool:
    try:
        _http_session().get(f"{BACKEND_URL}/", timeout=5)
        return True
    except requests.RequestException:
        return False

BACKEND_WARM = _warm_backend()

# --- Header ---
st.title("🏦 FinSight AI: Financial Intelligence Suite")
st.caption(
    "Advanced analytics and RAG-powered strategy for modern banking. "
    "*Backend hosted on HF Spaces free tier — first request may take ~20s if the Space is waking up.*"
)

tab1, tab2 = st.tabs(["📊 Lead Scoring", "🤖 Strategy Copilot"])

# --- TAB 1: LEAD SCORING ---
with tab1:
    st.header("Telemarketing Lead Scorer")

    with st.form("lead_scorer", clear_on_submit=False):
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

# --- TAB 2: STRATEGY COPILOT (scrollable history + pinned input) ---
with tab2:
    st.header("🤖 Strategy Copilot")
    st.caption("Ask questions about campaign strategies, GDPR/MiFID II compliance, or bank marketing best practices.")

    history_container = st.container(height=500)

    with history_container:
        if not st.session_state.messages:
            st.info("👋 Ask your first question below to get started.")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    prompt = st.chat_input("Ask about campaign strategies...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with history_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Consulting knowledge base..."):
                    try:
                        resp = _http_session().post(
                            f"{BACKEND_URL}/research",
                            json={"query": prompt},
                            timeout=60,
                        )
                        if resp.status_code == 200:
                            answer = resp.json().get("answer", "No response.")
                            st.markdown(answer)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": answer}
                            )
                        else:
                            st.error("Backend error. It may be waking up — try again in 20s.")
                    except requests.exceptions.Timeout:
                        st.warning("Backend timeout. The RAG engine is still loading — try again shortly.")
                    except Exception as e:
                        st.error(f"Error: {e}")