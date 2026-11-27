
from pathlib import Path
import hashlib
import json
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# AURUM-AI — D.3
# EXECUTIVE DASHBOARD + BLOCKCHAIN EXPLORER
#
# Research prototype — XOM simulated investing.
# Read-only presentation of validation artifacts.
# No real trading.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blockchain.ledger import BlockchainLedger


BACKTEST_DIR = (
    ROOT / "reports" / "backtesting"
    / "combined_holding_30m_60m"
)

BLOCKCHAIN_DIR = (
    ROOT / "reports" / "blockchain"
    / "consensus_trend_60m_validation"
)

SUMMARY_FILE = BACKTEST_DIR / "comparison_summary.json"
TRANSACTIONS_FILE = BACKTEST_DIR / "60m_transactions.csv"
DECISIONS_FILE = BACKTEST_DIR / "60m_decisions.csv"

LEDGER_FILE = BLOCKCHAIN_DIR / "ledger.json"
AUDIT_FILE = BLOCKCHAIN_DIR / "audit_summary.json"


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AURUM-AI | Investment Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. PREMIUM DARK STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #081525;
        color: #F2F6FC;
    }

    [data-testid="stSidebar"] {
        background: #0C1D30;
        border-right: 1px solid #1B3953;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1500px;
    }

    h1, h2, h3 {
        color: #F2F6FC;
    }

    [data-testid="stMetric"] {
        background: #10263D;
        border: 1px solid #21425D;
        border-radius: 14px;
        padding: 18px;
    }

    [data-testid="stMetricLabel"] {
        color: #A5BAD0;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #21425D;
        border-radius: 10px;
    }

    .aurum-header {
        background:
            linear-gradient(110deg, #102B44, #0A1D30);
        border: 1px solid #27506A;
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 22px;
    }

    .aurum-subtitle {
        color: #42D9E8;
        letter-spacing: 2px;
        font-size: 12px;
        font-weight: 600;
    }

    .aurum-description {
        color: #AEC0D2;
        font-size: 14px;
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. DATA LOADING
# ============================================================

@st.cache_data
def load_backtest_report():
    with SUMMARY_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_transactions():
    return pd.read_csv(TRANSACTIONS_FILE)


def calculate_file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_header(title, description):
    st.markdown(
        f"""
        <div class="aurum-header">
            <div class="aurum-subtitle">
                AURUM-AI · XOM INVESTMENT INTELLIGENCE
            </div>
            <h1>{title}</h1>
            <div class="aurum-description">
                {description}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 4. EXECUTIVE DASHBOARD
# ============================================================

def executive_dashboard():

    display_header(
        "Executive Dashboard",
        "Autonomous investment research · "
        "Historical validation · 100% simulated",
    )

    report = load_backtest_report()
    transactions = load_transactions()

    result_30 = report["results"]["30m"]
    result_60 = report["results"]["60m"]

    st.subheader("Portfolio Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Final Capital",
            f"${result_60['final_equity_usd']:,.2f}",
        )

    with c2:
        st.metric(
            "Net Return",
            f"{result_60['net_return_pct']:+.4f}%",
        )

    with c3:
        st.metric(
            "Completed Trades",
            result_60["completed_trades"],
        )

    with c4:
        st.metric(
            "Maximum Drawdown",
            f"{result_60['max_observed_drawdown_pct']:.4f}%",
        )

    st.caption(
        "Research strategy: CNN1D + Logistic Regression "
        "+ Technical Trend | 60m signal horizon | "
        "Validation only."
    )

    st.divider()

    # --------------------------------------------------------
    # Portfolio performance at completed SELL events
    # --------------------------------------------------------

    st.subheader("Portfolio Performance")

    sells = transactions.loc[
        transactions["action"] == "SELL"
    ].copy()

    sells["datetime_utc"] = pd.to_datetime(
        sells["timestamp_ms"],
        unit="ms",
        utc=True,
    )

    sells = sells.sort_values("timestamp_ms")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=sells["datetime_utc"],
            y=sells["cash_after_usd"],
            mode="lines+markers",
            name="Capital after closed trades",
            line=dict(color="#42D9E8", width=3),
            marker=dict(size=4),
            hovertemplate=(
                "%{x}<br>Capital: $%{y:,.2f}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_hline(
        y=10000,
        line_color="#D9B56C",
        line_dash="dash",
        annotation_text="Initial capital",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#081525",
        plot_bgcolor="#10263D",
        font_color="#D6E4F0",
        height=410,
        xaxis_title="Historical date (UTC)",
        yaxis_title="Capital (USD)",
        margin=dict(l=20, r=20, t=35, b=20),
        legend=dict(orientation="h"),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Capital displayed after completed SELL transactions. "
        "This is not the intratrade mark-to-market equity curve."
    )

    st.divider()

    # --------------------------------------------------------
    # Controlled horizon comparison
    # --------------------------------------------------------

    st.subheader("Strategy Comparison")

    c1, c2 = st.columns([2, 1])

    with c1:

        horizons = ["30m", "60m"]

        returns = [
            result_30["net_return_pct"],
            result_60["net_return_pct"],
        ]

        comparison = go.Figure(
            go.Bar(
                x=horizons,
                y=returns,
                marker_color=["#8094AB", "#42D9E8"],
                text=[
                    f"{value:+.4f}%"
                    for value in returns
                ],
                textposition="outside",
            )
        )

        comparison.update_layout(
            template="plotly_dark",
            paper_bgcolor="#081525",
            plot_bgcolor="#10263D",
            font_color="#D6E4F0",
            height=320,
            yaxis_title="Net return (%)",
            yaxis=dict(range=[min(returns) - 0.2, 0.1]),
        )

        st.plotly_chart(
            comparison,
            use_container_width=True,
        )

    with c2:

        st.metric(
            "60m Win Rate",
            f"{result_60['win_rate_pct']:.2f}%",
        )

        st.metric(
            "Execution Fees",
            f"${result_60['total_fees_usd']:.2f}",
        )

        st.metric(
            "Risk System",
            result_60["final_system_state"],
        )

    st.info(
        "The 30m/60m comparison uses the same "
        "132 historically eligible signals. "
        "Selection depends on future candle availability, "
        "so this is an exploratory comparison, not "
        "an independently validated trading policy."
    )


# ============================================================
# 5. BLOCKCHAIN EXPLORER
# ============================================================

def blockchain_explorer():

    display_header(
        "Blockchain Explorer",
        "SHA-256 linked historical decision ledger · "
        "Local and simulated",
    )

    try:
        ledger = BlockchainLedger.load_json(LEDGER_FILE)

        with AUDIT_FILE.open("r", encoding="utf-8") as file:
            audit = json.load(file)

    except (OSError, ValueError, KeyError, TypeError) as error:
        st.error(f"Blockchain loading failed: {error}")
        st.stop()

    blocks = ledger.blocks

    chain_valid = ledger.verify_integrity()

    audit_hash_matches = (
        blocks[-1]["hash"]
        == audit.get("final_block_hash")
    )

    source_paths = {
        "decisions": DECISIONS_FILE,
        "transactions": TRANSACTIONS_FILE,
        "summary": SUMMARY_FILE,
    }

    source_hashes_match = all(
        path.is_file()
        and calculate_file_sha256(path)
        == audit["source_file_sha256"].get(name)
        for name, path in source_paths.items()
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Total Blocks", len(blocks))

    with c2:
        st.metric("Decision Events", len(blocks) - 1)

    with c3:
        st.metric(
            "Chain Integrity",
            "VALID" if chain_valid else "INVALID",
        )

    with c4:
        st.metric(
            "Source Hashes",
            "MATCH" if source_hashes_match else "MISMATCH",
        )

    if (
        not chain_valid
        or not audit_hash_matches
        or not source_hashes_match
    ):
        st.error(
            "Blockchain or source integrity verification failed."
        )
    else:
        st.success(
            "Blockchain, saved audit hash, and historical "
            "source file hashes match."
        )

    st.caption(
        "This is a local tamper-evident audit ledger, "
        "not a decentralized public blockchain."
    )

    st.divider()

    # --------------------------------------------------------
    # Searchable block list
    # --------------------------------------------------------

    st.subheader("Block Explorer")

    records = []

    for block in blocks:

        payload = block["payload"]

        records.append(
            {
                "Block": block["index"],
                "Timestamp (UTC)": block["timestamp_utc"],
                "Event": block["event_type"],
                "Action": payload.get(
                    "strategy", {}
                ).get("action", "-"),
                "Hash": block["hash"][:18] + "...",
            }
        )

    table = pd.DataFrame(records)

    event_options = ["ALL"] + sorted(
        table["Event"].unique().tolist()
    )

    selected_event = st.selectbox(
        "Filter by event type",
        event_options,
    )

    if selected_event != "ALL":
        filtered = table[
            table["Event"] == selected_event
        ]
    else:
        filtered = table

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Block detail
    # --------------------------------------------------------

    st.subheader("Inspect a Block")

    selected_index = st.selectbox(
        "Block number",
        filtered["Block"].tolist()[::-1],
    )

    selected_block = blocks[int(selected_index)]

    st.write("**Current Hash**")
    st.code(selected_block["hash"])

    st.write("**Previous Hash**")
    st.code(selected_block["previous_hash"])

    with st.expander("View complete block data", expanded=True):
        st.json(selected_block)


# ============================================================
# 6. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown("## ⚡ AURUM-AI")

st.sidebar.caption(
    "XOM Autonomous Investment Simulation"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Executive Dashboard",
        "Blockchain Explorer",
    ],
)

st.sidebar.divider()

st.sidebar.caption(
    "RESEARCH MODE · VALIDATION DATA"
)

st.sidebar.caption(
    "No real money · No live orders"
)


# ============================================================
# 7. APPLICATION ROUTING
# ============================================================

required_files = [
    SUMMARY_FILE,
    TRANSACTIONS_FILE,
    DECISIONS_FILE,
    LEDGER_FILE,
    AUDIT_FILE,
]

missing_files = [
    str(path)
    for path in required_files
    if not path.is_file()
]

if missing_files:
    st.error("Required project artifacts are missing.")
    st.code("\n".join(missing_files))
    st.stop()

if page == "Executive Dashboard":
    executive_dashboard()

elif page == "Blockchain Explorer":
    blockchain_explorer()
