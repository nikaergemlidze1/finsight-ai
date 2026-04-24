import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timezone

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ═══════════════════════════════════════════════════════════════════════════
# HTTP HELPERS
# ═══════════════════════════════════════════════════════════════════════════

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

def _relative_time(iso_or_dt) -> str:
    """Turn a timestamp into 'just now' / '2m ago' / '1h ago' / etc."""
    if isinstance(iso_or_dt, str):
        try:
            dt = datetime.fromisoformat(iso_or_dt.replace("Z", "+00:00"))
        except ValueError:
            return ""
    else:
        dt = iso_or_dt
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = now - dt
    seconds = int(delta.total_seconds())
    if seconds < 10:
        return "just now"
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"

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
    /* Hide Streamlit Cloud's auto GitHub icon (links to source file, not repo) */
    [data-testid="stToolbarActions"] a[href*="github.com"]:not([data-custom="true"]) {
        display: none !important;
    }
    /* Hide header anchor icons (chain symbols) */
    [data-testid="stHeaderActionElements"] { display: none !important; }

    /* Chat messages */
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

    /* Timestamp under chat messages */
    .chat-timestamp {
        color: #6B7280;
        font-size: 0.75rem;
        margin-top: 0.3rem;
        font-style: italic;
    }

    /* Suggested-question pill buttons */
    .stButton > button[kind="secondary"] {
        border-radius: 999px;
        font-size: 0.85rem;
        padding: 0.25rem 0.85rem;
    }

    /* Loading skeleton shimmer */
    .skeleton-card {
        background: linear-gradient(90deg, #1f2937 25%, #374151 50%, #1f2937 75%);
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }
    @keyframes shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    .skeleton-line {
        background: rgba(107, 114, 128, 0.3);
        height: 1rem;
        border-radius: 4px;
        margin: 0.4rem 0;
    }
    .skeleton-line.short { width: 40%; }
    .skeleton-line.medium { width: 70%; }

    /* Analytics KPI cards */
    .kpi-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(71, 85, 105, 0.4);
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #FBBF24;
    }
    .kpi-label {
        color: #9CA3AF;
        font-size: 0.85rem;
        margin-top: 0.3rem;
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

tab1, tab2, tab3 = st.tabs(["📊 Lead Scoring", "🕴 Strategy Copilot", "📈 Analytics"])

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

        # Skeleton loader while request is pending
        skeleton_slot = st.empty()
        skeleton_slot.markdown(
            """
            <div class='skeleton-card'>
                <div class='skeleton-line short'></div>
                <div class='skeleton-line medium'></div>
                <div class='skeleton-line'></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        try:
            response = _http_session().post(f"{BACKEND_URL}/predict", json=payload, timeout=25)
            skeleton_slot.empty()

            if response.status_code == 200:
                res = response.json()
                prob = res.get("probability_of_subscription", 0) * 100
                recommendation = res.get("recommendation", "N/A")

                if prob >= 60:
                    tier, color, icon = "High Priority", "#10B981", "🔥"
                elif prob >= 30:
                    tier, color, icon = "Medium Priority", "#F59E0B", "⚡"
                else:
                    tier, color, icon = "Low Priority", "#EF4444", "❄️"

                # Two-column metric layout — gives each metric room
                mcol1, mcol2 = st.columns(2)
                with mcol1:
                    st.metric("Subscription Probability", f"{prob:.1f}%")
                with mcol2:
                    st.metric("Lead Tier", f"{icon} {tier}")

                # Visual probability bar
                st.markdown(
                    f"<div style='background:#1f2937;border-radius:8px;height:14px;overflow:hidden;margin:0.8rem 0 1rem 0;'>"
                    f"<div style='background:{color};width:{prob}%;height:100%;transition:width 0.8s;'></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Recommendation — full-width so it's never truncated
                st.info(f"**💡 Recommendation:** {recommendation}")
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except requests.exceptions.Timeout:
            skeleton_slot.empty()
            st.warning("Backend is waking up (HF Space cold start). Please click Predict again in a few seconds.")
        except Exception as e:
            skeleton_slot.empty()
            st.error(f"Connection failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: STRATEGY COPILOT
# ═══════════════════════════════════════════════════════════════════════════

def _send_to_copilot(prompt: str):
    user_msg = {
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    st.session_state.messages.append(user_msg)
    try:
        resp = _http_session().post(
            f"{BACKEND_URL}/research", json={"query": prompt}, timeout=60,
        )
        if resp.status_code == 200:
            answer = resp.json().get("answer", "No response.")
            st.session_state.messages.append({
                "role": "assistant", "content": answer,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "⚠️ Backend error. It may be waking up — try again in 20s.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
    except requests.exceptions.Timeout:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⚠️ Backend timeout. The RAG engine is still loading — try again shortly.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant", "content": f"⚠️ Error: {e}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

def _render_message(msg, idx):
    """Render a chat message with timestamp; assistant messages get a copy button via popover."""
    avatar = "🧑‍💼" if msg["role"] == "user" else "🎯"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

        ts_text = _relative_time(msg.get("timestamp", "")) if msg.get("timestamp") else ""
        st.markdown(
            f"<span class='chat-timestamp'>🕐 {ts_text}</span>",
            unsafe_allow_html=True,
        )

        if msg["role"] == "assistant":
            with st.popover("📋 Copy response", use_container_width=False):
                st.code(msg["content"], language=None, wrap_lines=True)
                st.caption("Click the copy icon in the top-right of the box above.")

with tab2:
    hcol1, hcol2 = st.columns([5, 1])
    with hcol1:
        st.header("🕴 Strategy Copilot", anchor=False)
    with hcol2:
        if st.session_state.messages:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    st.caption("Ask about campaign strategies, GDPR/MiFID II compliance, or bank marketing best practices.")

    # Chat history
    if not st.session_state.messages:
        st.info("👋 Start the conversation by typing your question below, or pick one of the suggestions.")
    else:
        history_height = min(500, max(250, len(st.session_state.messages) * 90))
        history_container = st.container(height=history_height)
        with history_container:
            for i, msg in enumerate(st.session_state.messages):
                _render_message(msg, i)

    # Suggested questions — visible on empty state, collapsible once chat starts
    if not st.session_state.messages:
        st.markdown("**💭 Try these:**")
        sugg_cols = st.columns(2)
        for i, q in enumerate(SUGGESTED_QUESTIONS):
            with sugg_cols[i % 2]:
                if st.button(q, key=f"sugg_empty_{i}", use_container_width=True):
                    st.session_state.pending_prompt = q
                    st.rerun()
    else:
        with st.expander("💭 Suggested questions", expanded=False):
            sugg_cols = st.columns(2)
            for i, q in enumerate(SUGGESTED_QUESTIONS):
                with sugg_cols[i % 2]:
                    if st.button(q, key=f"sugg_chat_{i}", use_container_width=True):
                        st.session_state.pending_prompt = q
                        st.rerun()

    # Input
    prompt = st.chat_input("Ask about campaign strategies...")

    active_prompt = prompt or st.session_state.pending_prompt
    if active_prompt:
        st.session_state.pending_prompt = None
        with st.spinner("Consulting knowledge base..."):
            _send_to_copilot(active_prompt)
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("📈 Usage Analytics", anchor=False)
    st.caption("Live aggregates from MongoDB — updated in real time as the app is used.")

    if st.button("🔄 Refresh", key="refresh_analytics"):
        st.rerun()

    try:
        analytics_resp = _http_session().get(f"{BACKEND_URL}/analytics", timeout=10)
        data = analytics_resp.json() if analytics_resp.status_code == 200 else {"available": False}
    except Exception as e:
        data = {"available": False, "reason": str(e)}

    if not data.get("available", False):
        st.warning(f"Analytics unavailable: {data.get('reason', 'backend offline')}")
    else:
        # KPI row
        kcol1, kcol2, kcol3, kcol4 = st.columns(4)
        with kcol1:
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-value'>{data['total_predictions']}</div>"
                f"<div class='kpi-label'>Total Predictions</div></div>",
                unsafe_allow_html=True,
            )
        with kcol2:
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-value'>{data['total_questions']}</div>"
                f"<div class='kpi-label'>Strategy Questions</div></div>",
                unsafe_allow_html=True,
            )
        with kcol3:
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-value'>{data['avg_probability']}%</div>"
                f"<div class='kpi-label'>Avg. Probability</div></div>",
                unsafe_allow_html=True,
            )
        with kcol4:
            total = data['total_predictions'] + data['total_questions']
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-value'>{total}</div>"
                f"<div class='kpi-label'>Total Activity</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        ccol1, ccol2 = st.columns(2)

        with ccol1:
            st.subheader("Lead Tier Distribution", anchor=False)
            tiers = data.get("tier_distribution", {})
            if sum(tiers.values()) > 0:
                import plotly.graph_objects as go
                fig = go.Figure(data=[go.Pie(
                    labels=["🔥 High Priority", "⚡ Medium Priority", "❄️ Low Priority"],
                    values=[tiers.get("high", 0), tiers.get("medium", 0), tiers.get("low", 0)],
                    hole=0.5,
                    marker=dict(colors=["#10B981", "#F59E0B", "#EF4444"]),
                )])
                fig.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=20, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E5E7EB"),
                    showlegend=True,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No predictions logged yet.")

        with ccol2:
            st.subheader("Recent Prediction Probabilities", anchor=False)
            activity = data.get("recent_activity", [])
            if activity:
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(activity) + 1)),
                    y=[a["probability"] for a in activity],
                    mode="lines+markers",
                    line=dict(color="#FBBF24", width=2),
                    marker=dict(size=8, color="#FBBF24"),
                    fill="tozeroy",
                    fillcolor="rgba(251, 191, 36, 0.12)",
                ))
                fig.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=20, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E5E7EB"),
                    xaxis=dict(title="Prediction #", gridcolor="#374151"),
                    yaxis=dict(title="Probability (%)", gridcolor="#374151", range=[0, 100]),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No predictions logged yet.")

        st.markdown("---")
        st.subheader("Recent Strategy Questions", anchor=False)
        recent_q = data.get("recent_questions", [])
        if recent_q:
            for q in recent_q:
                ts = _relative_time(q.get("timestamp", ""))
                st.markdown(
                    f"<div style='padding:0.6rem 1rem;background:rgba(59,130,246,0.06);"
                    f"border-left:3px solid #3B82F6;border-radius:6px;margin-bottom:0.4rem;'>"
                    f"{q.get('query', '')}<br>"
                    f"<span class='chat-timestamp'>🕐 {ts}</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No strategy questions logged yet.")