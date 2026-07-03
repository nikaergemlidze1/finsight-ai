import io
from datetime import datetime, timezone

import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.labels import (
    CONTACT_LABELS,
    DAY_LABELS,
    EDUCATION_LABELS,
    JOB_LABELS,
    MARITAL_LABELS,
    MONTH_LABELS,
    POUTCOME_LABELS,
    YES_NO_UNKNOWN,
)

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

# Tier cut-offs mirror api/tiers.py; the API's lead_tier field is authoritative
# and these are only used as a fallback for older backend responses.
TIER_HIGH, TIER_MEDIUM = 60.0, 30.0

TIER_STYLE = {
    "high": ("High Priority", "#10B981", "🔥"),
    "medium": ("Medium Priority", "#F59E0B", "⚡"),
    "low": ("Low Priority", "#EF4444", "❄️"),
}

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

def _headers() -> dict:
    """Send X-API-Key when configured in Streamlit secrets (backend auth optional)."""
    api_key = st.secrets.get("API_KEY", "")
    return {"X-API-Key": api_key} if api_key else {}


@st.cache_resource
def _http_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=2, backoff_factor=1.5,
        status_forcelist=[502, 503, 504],
        allowed_methods=["GET"],  # POSTs are not idempotent — never auto-retry them
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


def _error_detail(response: requests.Response, limit: int = 300) -> str:
    """Extract a safe, short error message from an API response."""
    try:
        detail = response.json().get("detail", "")
        if isinstance(detail, list):  # FastAPI validation errors
            parts = []
            for err in detail[:3]:
                loc = ".".join(str(p) for p in err.get("loc", []) if p != "body")
                parts.append(f"{loc}: {err.get('msg', '')}")
            detail = "; ".join(parts)
        text = str(detail)
    except Exception:
        text = "Unexpected backend error."
    return text[:limit] or "Unexpected backend error."


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
    seconds = int((now - dt).total_seconds())
    if seconds < 10:
        return "just now"
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


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

    /* SHAP driver rows */
    .driver-row {
        padding: 0.45rem 0.9rem;
        border-radius: 6px;
        margin-bottom: 0.35rem;
        background: rgba(30, 41, 59, 0.5);
        border-left: 3px solid #6B7280;
        font-size: 0.92rem;
    }
    .driver-row.up { border-left-color: #10B981; }
    .driver-row.down { border-left-color: #EF4444; }
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
if not BACKEND_WARM:
    st.caption("⏳ Backend appears to be waking up — the first prediction may be slow.")

tab1, tab2, tab3 = st.tabs(["📊 Lead Scoring", "🕴 Strategy Copilot", "📈 Analytics"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: LEAD SCORING
# ═══════════════════════════════════════════════════════════════════════════

BATCH_COLUMNS = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "campaign", "pdays", "previous",
    "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed",
]

BATCH_TEMPLATE = (
    ",".join(BATCH_COLUMNS) + "\n"
    "35,admin.,married,university.degree,no,yes,no,cellular,may,mon,"
    "1,999,0,nonexistent,-1.8,92.893,-46.2,1.299,5099.1\n"
    "58,retired,married,high.school,no,no,no,cellular,oct,thu,"
    "2,5,3,success,-3.4,92.431,-26.9,0.742,5017.5\n"
)


def _render_prediction_result(res: dict) -> None:
    """Shared renderer for a single /predict response."""
    prob = res.get("probability_of_subscription", 0) * 100
    recommendation = res.get("recommendation", "N/A")

    tier_key = res.get("lead_tier")
    if tier_key not in TIER_STYLE:  # fallback for older backends
        tier_key = "high" if prob >= TIER_HIGH else "medium" if prob >= TIER_MEDIUM else "low"
    tier, color, icon = TIER_STYLE[tier_key]

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

    st.info(f"**💡 Recommendation:** {recommendation}")

    # SHAP top drivers — present once the backend explainer has warmed up
    drivers = res.get("top_drivers")
    if drivers:
        st.markdown("**🔍 Why this score** *(top model drivers, SHAP)*")
        for d in drivers:
            up = d.get("direction") == "increases"
            arrow, cls = ("▲", "up") if up else ("▼", "down")
            verb = "raises" if up else "lowers"
            st.markdown(
                f"<div class='driver-row {cls}'>{arrow} <b>{d.get('feature', '?')}</b> "
                f"{verb} the subscription probability "
                f"<span style='color:#9CA3AF'>(impact {d.get('impact', 0):+.3f})</span></div>",
                unsafe_allow_html=True,
            )


with tab1:
    st.header("Telemarketing Lead Scorer", anchor=False)

    with st.form("lead_scorer", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("👤 Demographics", anchor=False)
            age = st.number_input("Age", 18, 100, 35)
            job = st.selectbox("Job", list(JOB_LABELS.keys()), format_func=lambda k: JOB_LABELS[k])
            marital = st.selectbox("Marital Status", list(MARITAL_LABELS.keys()), format_func=lambda k: MARITAL_LABELS[k])
            education = st.selectbox("Education", list(EDUCATION_LABELS.keys()), format_func=lambda k: EDUCATION_LABELS[k],
                                     index=list(EDUCATION_LABELS.keys()).index("university.degree"))
            st.subheader("💰 Financials", anchor=False)
            default = st.selectbox("Has Credit in Default?", list(YES_NO_UNKNOWN.keys()), format_func=lambda k: YES_NO_UNKNOWN[k])
            housing = st.selectbox("Has Housing Loan?", list(YES_NO_UNKNOWN.keys()), format_func=lambda k: YES_NO_UNKNOWN[k])
            loan = st.selectbox("Has Personal Loan?", list(YES_NO_UNKNOWN.keys()), format_func=lambda k: YES_NO_UNKNOWN[k])
        with col2:
            st.subheader("📞 Campaign", anchor=False)
            contact = st.selectbox("Contact Method", list(CONTACT_LABELS.keys()), format_func=lambda k: CONTACT_LABELS[k])
            month = st.selectbox("Last Contact Month", list(MONTH_LABELS.keys()), format_func=lambda k: MONTH_LABELS[k],
                                 index=list(MONTH_LABELS.keys()).index("may"))
            day_of_week = st.selectbox("Last Contact Day", list(DAY_LABELS.keys()), format_func=lambda k: DAY_LABELS[k])
            campaign = st.number_input("Contacts this campaign", 1, 50, 1)
            previous = st.number_input("Contacts in prior campaigns", 0, 20, 0)
            poutcome = st.selectbox("Previous campaign outcome", list(POUTCOME_LABELS.keys()), format_func=lambda k: POUTCOME_LABELS[k])
            pdays = st.number_input(
                "Days since last contact", 0, 999, 999,
                help="999 means the customer was never contacted in a previous campaign.",
            )
        with col3:
            st.subheader("📈 Macro-Economics", anchor=False)
            emp_var_rate = st.number_input("Emp. Var. Rate", value=-1.8)
            cons_price_idx = st.number_input("Cons. Price Index", value=92.893)
            cons_conf_idx = st.number_input("Cons. Confidence Index", value=-46.2)
            euribor3m = st.number_input("Euribor 3-Month", value=1.299)
            nr_employed = st.number_input("Nr. Employed", value=5099.1)

        submitted = st.form_submit_button("Predict Subscription Probability", type="primary")

    if submitted:
        payload = {
            "age": age, "job": job, "marital": marital, "education": education,
            "default": default, "housing": housing, "loan": loan,
            "contact": contact, "month": month, "day_of_week": day_of_week,
            "campaign": campaign, "pdays": pdays, "previous": previous, "poutcome": poutcome,
            "emp.var.rate": emp_var_rate, "cons.price.idx": cons_price_idx,
            "cons.conf.idx": cons_conf_idx, "euribor3m": euribor3m, "nr.employed": nr_employed,
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
            response = _http_session().post(
                f"{BACKEND_URL}/predict", json=payload, headers=_headers(), timeout=25)
            skeleton_slot.empty()

            if response.status_code == 200:
                _render_prediction_result(response.json())
            elif response.status_code == 429:
                st.warning("⏳ Rate limit reached — please wait a moment and try again.")
            else:
                st.error(f"Prediction failed ({response.status_code}): {_error_detail(response)}")
        except requests.exceptions.Timeout:
            skeleton_slot.empty()
            st.warning("Backend is waking up (HF Space cold start). Please click Predict again in a few seconds.")
        except Exception:
            skeleton_slot.empty()
            st.error("Connection failed — backend unreachable. Try again shortly.")

    # ── Batch scoring via CSV upload ──────────────────────────────────────────
    with st.expander("📁 Batch scoring — upload a CSV of leads"):
        st.caption(
            "Score up to **500 leads** at once. The file must contain exactly these "
            f"columns: `{', '.join(BATCH_COLUMNS)}`"
        )
        st.download_button(
            "⬇️ Download CSV template", BATCH_TEMPLATE,
            file_name="finsight_batch_template.csv", mime="text/csv",
        )
        uploaded = st.file_uploader("Upload leads CSV", type=["csv"], key="batch_csv")

        if uploaded is not None:
            import pandas as pd
            try:
                batch_df = pd.read_csv(uploaded)
            except Exception:
                batch_df = None
                st.error("Could not parse the file as CSV.")

            if batch_df is not None:
                missing = [c for c in BATCH_COLUMNS if c not in batch_df.columns]
                if missing:
                    st.error(f"Missing required columns: {', '.join(missing)}")
                elif len(batch_df) == 0:
                    st.error("The file contains no rows.")
                elif len(batch_df) > 500:
                    st.error(f"Too many rows: {len(batch_df)} (maximum 500 per upload).")
                else:
                    st.dataframe(batch_df.head(5), use_container_width=True)
                    if st.button(f"Score {len(batch_df)} leads", type="primary", key="run_batch"):
                        records = batch_df[BATCH_COLUMNS].to_dict(orient="records")
                        try:
                            with st.spinner("Scoring batch..."):
                                resp = _http_session().post(
                                    f"{BACKEND_URL}/batch-predict", json=records,
                                    headers=_headers(), timeout=60,
                                )
                            if resp.status_code == 200:
                                results = resp.json()
                                out = batch_df.copy()
                                out["probability_%"] = [round(r["probability_of_subscription"] * 100, 1) for r in results]
                                out["lead_tier"] = [r.get("lead_tier", "") for r in results]
                                out["recommendation"] = [r["recommendation"] for r in results]
                                out = out.sort_values("probability_%", ascending=False)
                                st.success(f"Scored {len(out)} leads.")
                                st.dataframe(out, use_container_width=True)
                                csv_buf = io.StringIO()
                                out.to_csv(csv_buf, index=False)
                                st.download_button(
                                    "⬇️ Download scored leads", csv_buf.getvalue(),
                                    file_name="finsight_scored_leads.csv", mime="text/csv",
                                )
                            elif resp.status_code == 429:
                                st.warning("⏳ Rate limit reached — please wait a minute before the next batch.")
                            else:
                                st.error(f"Batch scoring failed ({resp.status_code}): {_error_detail(resp)}")
                        except requests.exceptions.Timeout:
                            st.warning("Batch request timed out — the backend may be waking up. Try again.")
                        except Exception:
                            st.error("Connection failed — backend unreachable. Try again shortly.")

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

    def _assistant(content: str, sources: list | None = None):
        st.session_state.messages.append({
            "role": "assistant", "content": content,
            "sources": sources or [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    try:
        resp = _http_session().post(
            f"{BACKEND_URL}/research", json={"query": prompt},
            headers=_headers(), timeout=60,
        )
        if resp.status_code == 200:
            body = resp.json()
            _assistant(body.get("answer", "No response."), body.get("sources", []))
        elif resp.status_code == 503:
            _assistant("⚠️ The knowledge engine is still loading — try again in ~20 seconds.")
        elif resp.status_code == 429:
            _assistant("⏳ Rate limit reached — please wait a moment before asking again.")
        elif resp.status_code == 401:
            _assistant("🔒 This backend requires an API key. Configure `API_KEY` in the app secrets.")
        else:
            _assistant(f"⚠️ Backend error ({resp.status_code}): {_error_detail(resp)}")
    except requests.exceptions.Timeout:
        _assistant("⚠️ Backend timeout. The RAG engine is still loading — try again shortly.")
    except Exception:
        _assistant("⚠️ Connection failed — backend unreachable. Try again shortly.")


def _render_message(msg):
    """Render a chat message with timestamp, sources, and a copy popover."""
    avatar = "🧑‍💼" if msg["role"] == "user" else "🎯"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

        sources = msg.get("sources") or []
        if sources:
            files = " · ".join(f"`{s.get('file', '?')}`" for s in sources)
            st.caption(f"📚 Sources: {files}")

        ts_text = _relative_time(msg.get("timestamp", "")) if msg.get("timestamp") else ""
        if ts_text:
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
            for msg in st.session_state.messages:
                _render_message(msg)

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
    st.caption(
        "Live aggregates from MongoDB — updated in real time as the app is used. "
        "Only anonymous aggregates are shown; individual inputs and questions stay private."
    )

    if st.button("🔄 Refresh", key="refresh_analytics"):
        st.rerun()

    try:
        analytics_resp = _http_session().get(
            f"{BACKEND_URL}/analytics", headers=_headers(), timeout=10)
        data = analytics_resp.json() if analytics_resp.status_code == 200 else {"available": False}
    except Exception:
        data = {"available": False, "reason": "backend unreachable"}

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
        st.subheader("🌡️ Feature Drift Monitor", anchor=False)
        st.caption(
            "Compares recent prediction inputs against the training distribution "
            "(|mean z-score| per numeric feature). Green = stable, amber = drifting, "
            "red = significant shift — consider retraining."
        )

        try:
            drift_resp = _http_session().get(
                f"{BACKEND_URL}/drift", headers=_headers(), timeout=10)
            drift_data = drift_resp.json() if drift_resp.status_code == 200 else {"available": False}
        except Exception:
            drift_data = {"available": False, "reason": "backend unreachable"}

        if not drift_data.get("available", False):
            st.info(f"Drift monitor unavailable: {drift_data.get('reason', 'backend offline')}")
        else:
            feats = drift_data.get("features", [])
            if not feats:
                st.info("Not enough recent prediction data yet.")
            else:
                import plotly.graph_objects as go
                status_color = {"ok": "#10B981", "warn": "#F59E0B", "alert": "#EF4444"}
                thresholds = drift_data.get("thresholds", {"warn": 0.25, "alert": 0.5})
                fig = go.Figure(go.Bar(
                    x=[f["mean_shift"] for f in feats],
                    y=[f["feature"] for f in feats],
                    orientation="h",
                    marker_color=[status_color.get(f["status"], "#6B7280") for f in feats],
                ))
                fig.add_vline(x=thresholds["warn"], line_dash="dot", line_color="#F59E0B")
                fig.add_vline(x=thresholds["alert"], line_dash="dot", line_color="#EF4444")
                fig.update_layout(
                    height=max(260, 34 * len(feats)),
                    margin=dict(l=10, r=10, t=20, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E5E7EB"),
                    xaxis=dict(title="|mean z-score| vs. training", gridcolor="#374151"),
                    yaxis=dict(gridcolor="#374151"),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Based on the last {drift_data.get('n_samples', '?')} logged predictions.")
