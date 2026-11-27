"""AURUM-AI V5 — authenticated Streamlit entrypoint.

Run ONLY: python -m streamlit run frontend/app.py
Existing research dashboards run behind the shared authentication gateway.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
st.set_option("client.toolbarMode", "minimal")
from frontend.branding import apply_aurum_branding, aurum_favicon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.auth.idle_guard import (
    IDLE_TIMEOUT_SECONDS,
    check_local_session,
    end_local_session,
    render_idle_watchdog,
)

st.set_page_config(
    page_title="AURUM-AI V5 | Investment Intelligence",
    page_icon=aurum_favicon(),
    layout="wide",
    initial_sidebar_state="expanded",
)

# Prevent the standalone Login UI and the reused dashboards from setting
# a second, conflicting page configuration inside this V5 application.
st.session_state["aurum_v5_mode"] = True

apply_aurum_branding()

PAGE_DIR = ROOT / "frontend" / "v5_pages"


def page(file: str, title: str, icon: str, url: str, *, default=False):
    return st.Page(
        str(PAGE_DIR / file), title=title, icon=icon,
        url_path=url, default=default,
    )


# st.navigation is called EVEN ON THE LOCKED/LOGGED-OUT SCREEN.
# This is essential: it disables Streamlit's automatic ./pages discovery,
# so an old /NYSE_V3_Backtesting URL cannot bypass the login gateway.
if not check_local_session():
    auth_page = page("auth.py", "Secure Access", "🔐", "login", default=True)
    st.navigation([auth_page], position="hidden").run()
    st.stop()

# Shared inactivity detector on every authenticated page.
render_idle_watchdog()

st.markdown("""
<style>
[data-testid="stSidebar"] {
  background: linear-gradient(165deg,#07192c 0%,#0b2038 65%,#061322 100%);
  border-right: 1px solid rgba(55,177,233,.22);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
  color: #bdd7e9;
}
[data-testid="stSidebar"] h3 { color: #f1f8ff; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ◇ AURUM-AI **V5**")
    st.caption("XOM · Quantitative Intelligence")
    account = st.session_state["aurum_local_user"]
    st.caption(f"👤 {account.get('display_name', 'Utilisateur')}")
    st.caption(f"🔐 Verrouillage : {IDLE_TIMEOUT_SECONDS // 60} minutes" if IDLE_TIMEOUT_SECONDS % 60 == 0 else f"🔐 Verrouillage : {IDLE_TIMEOUT_SECONDS} s")
    if st.button("🚪 Se déconnecter", width="stretch", key="aurum_v5_sign_out"):
        end_local_session()
        st.rerun()
    st.divider()

navigation = {
    "OVERVIEW": [
        page("executive.py", "Executive Dashboard", "🏠", "executive", default=True),
    ],
    "MARKET & AI": [
        page("market_analysis.py", "Market Analysis", "📈", "market-analysis"),
        page("data_quality.py", "Data Quality", "🔍", "data-quality"),
        page("feature_engineering.py", "Feature Engineering", "🧮", "feature-engineering"),
        page("ai_prediction.py", "AI Prediction", "🎯", "ai-prediction"),
        page("ai_model_lab.py", "AI Model Lab", "🧠", "ai-model-lab"),
    ],
    "AUTONOMOUS INTELLIGENCE": [
        page("autonomous_agents.py", "Autonomous Agents", "🤖", "autonomous-agents"),
        page("agent_decision_room.py", "Agent Decision Room", "🗣️", "agent-decision-room"),
        page("digital_twin.py", "Digital Twin", "🔁", "digital-twin"),
    ],
    "PORTFOLIO & RISK": [
        page("portfolio_simulator.py", "Portfolio Simulator", "💼", "portfolio-simulator"),
        page("risk_management.py", "Risk Management", "🛡️", "risk-management"),
        page("backtesting.py", "Backtesting · NYSE V3", "📊", "backtesting"),
        page("stress_testing.py", "Stress Testing", "⚡", "stress-testing"),
        page("what_if_simulator.py", "What-If Simulator", "🎛️", "what-if-simulator"),
    ],
    "BLOCKCHAIN": [
        page("blockchain_ledger.py", "Blockchain Ledger", "🔗", "blockchain-ledger"),
        page("blockchain_explorer.py", "Blockchain Explorer", "🧱", "blockchain-explorer"),
    ],
    "RESEARCH CENTER": [
        page("library.py", "Library", "📁", "library"),
        page("experiments.py", "Experiments", "🧪", "experiments"),
        page("reports.py", "Reports", "📄", "reports"),
    ],
}

selected_page = st.navigation(navigation, position="sidebar")
selected_page.run()
