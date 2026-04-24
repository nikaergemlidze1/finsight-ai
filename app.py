import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Page config must be first Streamlit call ---
st.set_page_config(
    page_title="FinSight AI",
    layout="wide",
    page_icon="🏦",
    menu_items={
        "Get Help": "https://github.com/nikaergemlidze1/finsight-ai",
        "Report a bug": "https://github.com/nikaergemlidze1/finsight-ai/issues",
        "About": "**FinSight AI** — Financial Intelligence Suite for bank marketing campaigns. Built with FastAPI + Streamlit + LightGBM + LlamaIndex RAG.",
    },
)

BACKEND_URL = st.secrets.get(
    "BACKEND_URL",
    "https://nikollass-finsight-ai-backend.hf.space",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

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

@st.cache_resource(show_spinner=False)
def _warm_backend() -> bool:
    try:
        _http_session().get(f"{BACKEND_URL}/", timeout=5)
        return True
    except requests.RequestException:
        return False

BACKEND_WARM = _warm_backend()

# ═══════════════════════════════════════════════════════════════════════════
# LABEL MAPS
# ═══════════════════════════════════════════════════════════════════════════

JOB_LABELS = {
    "admin.": "Administrative", "blue-collar": "Blue Collar",
    "entrepreneur": "Entrepreneur", "housemaid": "Housemaid",
    "management": "Management", "retired": "Retired",
    "self-employed": "Self-Employed", "services": "Services",
    "student": "Student", "technician": "Technician",
    "unemployed": "Unemployed", "unknown": "Unknown",
}
MARITAL_LABELS = {"divorced": "Divorced", "married": "Married", "single": "Single", "unknown": "Unknown"}
EDUCATION_LABELS = {
    "basic.4y": "Primary (4 years)", "basic.6y": "Primary (6 years)",
    "basic.9y": "Lower Secondary (9 years)", "high.school": "High School",
    "illiterate": "Illiterate", "professional.course": "Professional Course",
    "university.degree": "University Degree", "unknown": "Unknown",
}
YES_NO_LABELS = {"no": "No", "yes": "Yes", "unknown": "Unknown"}
CONTACT_LABELS = {"cellular": "Cellular", "telephone": "Telephone"}
MONTH_LABELS = {
    "jan": "January", "feb": "February", "mar": "March", "apr": "April",
    "may": "May", "jun": "June", "jul": "July", "aug": "August",
    "sep": "September", "oct": "October", "nov": "November", "dec": "December",
}

SUGGESTED_QUESTIONS = [
    "What's the optimal time to call customers?",
    "Which customer segments convert best?",
    "How does GDPR affect telemarketing consent?",
    "What factors drive term deposit subscriptions?",
]

# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <style>
    /* Hide Streamlit Cloud's auto-generated GitHub icon (links to source file, not repo) */
    [data-testid="stToolbarActions"] a[href*="github.com"]:not([data-custom="true"]) {
        display: none !important;
    }
    /* Hide header anchor link icons (chain symbols) */
    [data-testid="stHeaderActionElements"] { display: none !important; }
    h1 a, h2 a, h3 a, h4 a, h5 a, h6 a { display: none !important; }

    /* Chat message styling */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: rgba(59, 130, 246, 0.08);
        border-left: 3px solid #3B82F6;
        margin-left: 10%;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background: rgba(251, 191, 36, 0.06);
        border-left: 3px solid #FBBF24;
        margin-right: 10%;
    }
    [data-testid="stChatMessage"] {
        padding: 0.6rem 1rem;
        margin-bottom: 0.4rem;
        border-radius: 8px;
    }

    /* Suggested-question buttons — subtle, pill-shaped */
    .stButton > button[kind="secondary"] {
        border-radius: 999px;
        font-size: 0.85rem;
        padding: 0.25rem 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════

header_left, header_right = st.columns([6, 1])
with header_left:
    st.title("🏦 FinSight AI: Financial Intelligence Suite", anchor=False)
with header_right:
    st.markdown(
        "<div style='padding-top: 1.5rem; text-align: right;'>"
        "<a href='https://github.com/nikaergemlidze1/finsight-ai' target='_blank' "
        "style='text-decoration:none; color:inherit; font-size:0.9rem;'>"
        "⭐ View on GitHub</a></div>",
        unsafe_allow_html=True,
    )

st.caption(
    "Advanced analytics and RAG-powered strategy for modern banking. "
    "*Backend on HF Spaces free tier — first request may take ~20s if the Space is waking up.*"
)

tab1, tab2 = st.tabs(["📊 Lead Scoring", "🤖 Strategy Copilot"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: LEAD SCORING
# ═══════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("Telemarketing Lead Scorer", anchor=False)

    with st.form("lead_scorer", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("👤 Demographics", anchor=False)
            age = st.number_input("Age", 18, 100, 35)
            job = st.selectbox("Job", list(JOB_LABELS.keys()), format_func=lambda k: JOB_LABELS[k])
            marital = st.selectbox("Marital Status", list(MARITAL_LABELS.keys()), format_func=lambda k: MARITAL_LABELS[k])
            education = st.selectbox("Education", list(EDUCATION_LABELS.keys()), format_func=lambda k: EDUCATION_LABELS[k])

        with col2:
            st.subheader("💰 Financials", anchor=False)
            default = st.selectbox("Has Credit in Default?", list(YES_NO_LABELS.keys()), format_func=lambda k: YES_NO_LABELS[k])
            housing = st.selectbox("Has Housing Loan?", list(YES_NO_LABELS.keys()), format_func=lambda k: YES_NO_LABELS[k])
            loan = st.selectbox("Has Personal Loan?", list(YES_NO_LABELS.keys()), format_func=lambda k: YES_NO_LABELS[k])

            st.subheader("📞 Campaign", anchor=False)
            contact = st.selectbox("Contact Method", list(CONTACT_LABELS.keys()), format_func=lambda k: CONTACT_LABELS[k])
            month = st.selectbox("Month", list(MONTH_LABELS.keys()), format_func=lambda k: MONTH_LABELS[k])

        with col3:
            st.subheader("📈 Macro-Economics", anchor=False)
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
            "emp.var.rate": emp_var_rate, "cons.price.idx": cons_price_idx,
            "cons.conf.idx": -46.2, "euribor3m": euribor3m, "nr.employed": nr_employed,
        }
        with st.spinner("Analyzing lead..."):
            try:
                response = _http_session().post(f"{BACKEND_URL}/predict", json=payload, timeout=25)
                if response.status_code == 200:
                    res = response.json()
                    prob = res.get("probability_of_subscription", 0) * 100
                    recommendation = res.get("recommendation", "N/A")

                    # Color-coded result panel
                    if prob >= 60:
                        tier, color, icon = "High Priority", "#10B981", "🔥"
                    elif prob >= 30:
                        tier, color, icon = "Medium Priority", "#F59E0B", "⚡"
                    else:
                        tier, color, icon = "Low Priority", "#EF4444", "❄️"

                    mcol1, mcol2, mcol3 = st.columns(3)
                    with mcol1:
                        st.metric("Subscription Probability", f"{prob:.1f}%")
                    with mcol2:
                        st.metric("Lead Tier", f"{icon} {tier}")
                    with mcol3:
                        st.metric("Recommendation", recommendation)

                    # Visual progress bar
                    st.markdown(
                        f"<div style='background:#1f2937;border-radius:8px;height:12px;overflow:hidden;margin-top:1rem;'>"
                        f"<div style='background:{color};width:{prob}%;height:100%;transition:width 0.5s;'></div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except requests.exceptions.Timeout:
                st.warning("Backend is waking up (HF Space cold start). Please click Predict again in a few seconds.")
            except Exception as e:
                st.error(f"Connection failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: STRATEGY COPILOT
# ═══════════════════════════════════════════════════════════════════════════

def _send_to_copilot(prompt: str):
    """Shared handler for typed prompts and suggested-question clicks."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        resp = _http_session().post(
            f"{BACKEND_URL}/research", json={"query": prompt}, timeout=60,
        )
        if resp.status_code == 200:
            answer = resp.json().get("answer", "No response.")
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.session_state.messages.append(
                {"role": "assistant", "content": "⚠️ Backend error. It may be waking up — try again in 20s."}
            )
    except requests.exceptions.Timeout:
        st.session_state.messages.append(
            {"role": "assistant", "content": "⚠️ Backend timeout. The RAG engine is still loading — try again shortly."}
        )
    except Exception as e:
        st.session_state.messages.append(
            {"role": "assistant", "content": f"⚠️ Error: {e}"}
        )

with tab2:
    # Header row with clear-chat button on the right
    hcol1, hcol2 = st.columns([5, 1])
    with hcol1:
        st.header("🤖 Strategy Copilot", anchor=False)
    with hcol2:
        if st.session_state.messages:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    st.caption("Ask about campaign strategies, GDPR/MiFID II compliance, or bank marketing best practices.")

    # Empty state OR chat history
    if not st.session_state.messages:
        st.info("👋 Start the conversation by typing your question below, or pick one of the suggestions.")

        # Suggested questions grid
        st.markdown("**💭 Try these:**")
        sugg_cols = st.columns(2)
        for i, q in enumerate(SUGGESTED_QUESTIONS):
            with sugg_cols[i % 2]:
                if st.button(q, key=f"sugg_{i}", use_container_width=True):
                    st.session_state.pending_prompt = q
                    st.rerun()
    else:
        history_height = min(500, max(250, len(st.session_state.messages) * 90))
        history_container = st.container(height=history_height)
        with history_container:
            for msg in st.session_state.messages:
                avatar = "🧑‍💼" if msg["role"] == "user" else "💡"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["content"])

    # Input
    prompt = st.chat_input("Ask about campaign strategies...")

    # Handle typed prompt OR pending suggested question
    active_prompt = prompt or st.session_state.pending_prompt
    if active_prompt:
        st.session_state.pending_prompt = None  # consume it
        with st.spinner("Consulting knowledge base..."):
            _send_to_copilot(active_prompt)
        st.rerun()