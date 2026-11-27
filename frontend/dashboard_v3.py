"""AURUM-AI premium Streamlit dashboard V3 (historical simulation only).

Copy this file to <project>/frontend/dashboard_v3.py and run:
    python -m streamlit run frontend/dashboard_v3.py --server.port 8504

Fixes: Streamlit magic was rendering a conditional expression result
inside Blockchain Audit Trail; portfolio performance zoom now uses
realistic y-axis bounds around the historical simulated capital.
"""
from __future__ import annotations

import hashlib
import html
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blockchain.ledger import BlockchainLedger

BACKTEST_DIR = ROOT / "reports" / "backtesting" / "combined_holding_30m_60m"
LEDGER_DIR = ROOT / "reports" / "blockchain" / "consensus_trend_60m_validation"
SUMMARY_PATH = BACKTEST_DIR / "comparison_summary.json"
DECISIONS_PATH = BACKTEST_DIR / "60m_decisions.csv"
TRANSACTIONS_PATH = BACKTEST_DIR / "60m_transactions.csv"
LEDGER_PATH = LEDGER_DIR / "ledger.json"
AUDIT_PATH = LEDGER_DIR / "audit_summary.json"

st.set_page_config(
    page_title="AURUM-AI | XOM Investment Twin",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

STYLES = """
<style>
:root { color-scheme: dark; }
.stApp { background: #061321; color: #e9f2ff; }
[data-testid="stSidebar"] {
  background: #081a2d !important; border-right: 1px solid #1b3855;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }
.block-container { padding: 1.2rem 1.5rem 2.5rem; max-width: 1800px; }
[data-testid="stAppViewContainer"] { background: #061321; }
[data-testid="stHeader"] { background: rgba(6,19,33,.8); }
#MainMenu, footer { visibility: hidden; }
h1, h2, h3 { color: #edf6ff; }
h1 { font-size: 1.9rem !important; }
h2 { font-size: 1.3rem !important; }
h3 { font-size: 1rem !important; }
[data-testid="stMetric"] {
  background: #0b2339; border: 1px solid #1b4564;
  padding: 13px; border-radius: 12px;
}
[data-testid="stMetricLabel"] { color: #8aaac5; }
[data-testid="stMetricValue"] { color: #edf8ff; font-size: 1.35rem; }
[data-testid="stPlotlyChart"] { border: 1px solid #163a55; border-radius: 12px; }
[data-testid="stDataFrame"] { border: 1px solid #244661; border-radius: 9px; }
[data-testid="stSidebar"] [role="radiogroup"] label { border-radius: 9px; }
.brand { font-weight: 900; font-size: 30px; color: #eff7ff; letter-spacing: -1px; }
.brand-x { color: #ff344b; }
.brand-small { font-size: 10px; letter-spacing: 2px; color: #82a9c7; }
.hero {
  background: radial-gradient(circle at 80% 10%, #164a5b 0, transparent 32%),
              linear-gradient(111deg, #0a2037, #0b2a42 62%, #102f3b);
  border: 1px solid #23546b; border-radius: 15px;
  padding: 20px 23px; margin-bottom: 13px;
}
.eyebrow { color: #3be4ed; font-size: 10px; font-weight: 700; letter-spacing: 2.1px; }
.hero-title { font-size: 27px; font-weight: 850; letter-spacing: -.4px; margin: 5px 0; }
.hero-desc { color: #b6cbde; font-size: 12px; }
.pill { background: #103848; border: 1px solid #227c82; border-radius: 14px;
   color: #50f0c5; padding: 5px 10px; display: inline-block; font-size: 11px; margin-top: 10px; }
.kpi { background: #0b2137; border: 1px solid #1d405b; border-radius: 11px;
       padding: 12px 13px; min-height: 103px; }
.kpi-label { color: #9cb4c9; font-size: 11px; }
.kpi-value { color: #eff9ff; font-size: 24px; font-weight: 750; margin: 5px 0 3px; white-space: nowrap; }
.kpi-note { color: #9cb4c9; font-size: 10px; }
.panel-title { margin-top: 7px; color: #eaf6ff; font-weight: 750; font-size: 16px; }
.micro { color: #91b0c9; font-size: 11px; }
.agent { background: #0b2238; border: 1px solid #24516d; border-radius: 10px;
       min-height: 114px; padding: 12px 11px; }
.agent-name { font-size: 12px; font-weight: 750; color: #e9f5ff; }
.agent-desc { color: #8fafca; font-size: 10px; margin-top: 6px; }
.agent-state { color: #2cdec2; font-weight: 800; margin-top: 12px; font-size: 13px; }
.banner-strip { background: #0a2439; border: 1px solid #1b435b; padding: 9px 13px;
  border-radius: 9px; font-size: 11px; color: #adc5d8; margin: 9px 0; }
[data-testid="stSidebar"] .stMarkdown { color: #d9f0ff; }
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)


def safe(value: object) -> str:
    return html.escape(str(value))


def kpi(label: str, value: str, note: str = "", accent: str = "#eef9ff") -> None:
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{safe(label)}</div>'
        f'<div class="kpi-value" style="color:{accent}">{safe(value)}</div>'
        f'<div class="kpi-note">{safe(note)}</div></div>',
        unsafe_allow_html=True,
    )


def header(title: str, desc: str, label: str = "HISTORICAL RESEARCH · XOM") -> None:
    st.markdown(
        '<div class="hero">'
        f'<div class="eyebrow">{safe(label)}</div>'
        f'<div class="hero-title">{safe(title)}</div>'
        f'<div class="hero-desc">{safe(desc)}</div>'
        '<div class="pill">● SIMULATION · VALIDATION DATA</div>'
        '</div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_reports():
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    decisions = pd.read_csv(DECISIONS_PATH)
    transactions = pd.read_csv(TRANSACTIONS_PATH)
    return summary, decisions, transactions


@st.cache_data(show_spinner="Loading XOM validation candles...")
def load_validation_prices():
    """Load complete Bid OHLC for exactly the validation candles used in backtesting.

    The backtest loader returns a lightweight price table for execution,
    which may not include high, low and close. Candlestick charts require
    these columns, so get them from the existing segmented XOM dataset.
    This is a read-only join by timestamp, not a new train/test split.
    """
    import numpy as np
    from backtesting.simulate_multi_agent_30m import load_inputs

    _signals, execution_prices = load_inputs()
    source = ROOT / "data" / "02_clean" / "xom_segmented.parquet"

    candles = pd.read_parquet(
        source,
        columns=["timestamp", "open", "high", "low", "close", "segment_id"],
    )
    if candles["timestamp"].duplicated().any():
        raise RuntimeError("Duplicate timestamps in segmented XOM candle data.")
    if execution_prices["timestamp"].duplicated().any():
        raise RuntimeError("Duplicate timestamps in validation execution data.")

    # Restrict the chart to the same historical validation candles.
    timestamps = execution_prices["timestamp"].to_numpy(dtype="int64")
    candles = candles.set_index("timestamp", verify_integrity=True).reindex(timestamps)
    if candles[["open", "high", "low", "close", "segment_id"]].isna().any().any():
        raise RuntimeError("Missing OHLC candles for a validation timestamp.")
    if not np.array_equal(
        candles["segment_id"].to_numpy(),
        execution_prices["segment_id"].to_numpy(),
    ):
        raise RuntimeError("Segment IDs differ between market and backtest data.")
    if not np.allclose(
        candles["open"].to_numpy(dtype=float),
        execution_prices["open"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-9,
    ):
        raise RuntimeError("Open prices differ between market and backtest data.")

    return candles.reset_index().sort_values("timestamp").copy()


@st.cache_data(show_spinner=False)
def load_indicators():
    return pd.read_parquet(
        ROOT / "data" / "03_features" / "xom_features_v2.parquet",
        columns=["timestamp", "sma_5", "ema_10"],
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_ledger():
    # Reload each time this page is opened: no stale cached integrity result.
    ledger = BlockchainLedger.load_json(LEDGER_PATH)
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    checks = {
        "Chain SHA-256": ledger.verify_integrity(),
        "Last block hash": ledger.blocks[-1]["hash"] == audit["final_block_hash"],
        "Source decisions": sha256_file(DECISIONS_PATH) == audit["source_file_sha256"]["decisions"],
        "Source transactions": sha256_file(TRANSACTIONS_PATH) == audit["source_file_sha256"]["transactions"],
        "Source summary": sha256_file(SUMMARY_PATH) == audit["source_file_sha256"]["summary"],
    }
    return ledger.blocks, audit, checks


def chart_style(fig, height=315):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#08192a", plot_bgcolor="#0a2135",
        font=dict(color="#d7e8f7", size=11),
        margin=dict(l=12, r=12, t=15, b=12),
        height=height, hovermode="x unified",
        legend=dict(orientation="h", y=1.1, x=0),
        xaxis=dict(gridcolor="#19354a"),
        yaxis=dict(gridcolor="#19354a"),
    )
    return fig


def market_chart(compact: bool = True):
    prices = load_validation_prices().tail(350).copy()
    if prices.empty:
        st.warning("No historical validation candles available.")
        return
    indicators = load_indicators()
    data = prices.merge(indicators, on="timestamp", how="left", validate="one_to_one")
    data["utc"] = pd.to_datetime(data["timestamp"], unit="ms", utc=True)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data["utc"], open=data["open"], high=data["high"],
        low=data["low"], close=data["close"], name="XOM Bid 5m",
        increasing_line_color="#00e2b8", decreasing_line_color="#ff526b",
    ))
    fig.add_trace(go.Scatter(
        x=data["utc"], y=data["sma_5"], mode="lines", name="SMA 5",
        line=dict(color="#13abfa", width=1.4),
    ))
    fig.add_trace(go.Scatter(
        x=data["utc"], y=data["ema_10"], mode="lines", name="EMA 10",
        line=dict(color="#efb341", width=1.3),
    ))
    chart_style(fig, 340 if compact else 515)
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    st.caption("Historical XOMUSUSD Bid · 5-minute candles · Validation sample · Not live market data")


def capital_chart(transactions, height=220):
    sells = transactions.loc[transactions["action"].eq("SELL")].sort_values("timestamp_ms").copy()
    sells["time"] = pd.to_datetime(sells["timestamp_ms"], unit="ms", utc=True)
    fig = go.Figure()
    if sells.empty:
        st.info("No completed simulated trades to chart.")
        return

    capital = sells["cash_after_usd"].astype(float)
    capital_min = min(10000.0, float(capital.min()))
    capital_max = max(10000.0, float(capital.max()))
    # A tight axis shows variations in capital without implying gains.
    padding = max((capital_max - capital_min) * 0.12, 3.0)

    fig.add_trace(go.Scatter(
        x=sells["time"], y=capital,
        mode="lines+markers", line=dict(color="#12e0b8", width=2.5),
        marker=dict(size=3),
        name="Capital after closed trades",
    ))
    fig.add_hline(y=10000, line_dash="dash", line_color="#c7a76c")
    chart_style(fig, height)
    fig.update_layout(
        yaxis_title="Capital (USD)",
        yaxis_range=[capital_min - padding, capital_max + padding],
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def dashboard(summary, decisions, transactions):
    result = summary["results"]["60m"]
    last_decision = decisions.sort_values("timestamp_ms").iloc[-1]
    last_transaction = transactions.sort_values("timestamp_ms").iloc[-1]
    last_price = float(last_transaction["price"])
    header(
        "XOM · Intelligent Investment Twin",
        "AI-powered multi-agent simulation · Historical XOM data · SHA-256 audit trail",
        "AURUM-AI  /  EXXON MOBIL  /  RESEARCH WORKSPACE",
    )
    cards = st.columns(6, gap="small")
    values = [
        ("Last simulated execution", f"${last_price:,.2f}", "Historical Bid price", "#eaf9ff"),
        ("Final portfolio", f"${result['final_equity_usd']:,.2f}", "Starting capital $10,000", "#eaf9ff"),
        ("Net return", f"{result['net_return_pct']:+.3f}%", "After hypothetical costs", "#ff7c89"),
        ("Completed trades", str(result["completed_trades"]), "60m exit horizon", "#eaf9ff"),
        ("Max drawdown", f"{result['max_observed_drawdown_pct']:.3f}%", "Historical simulation", "#eaf9ff"),
        ("Blockchain", f"{len(json.loads(LEDGER_PATH.read_text(encoding='utf-8'))['blocks'])} blocks", "Local audit chain", "#2be1c4"),
    ]
    for column, (label, value, note, color) in zip(cards, values):
        with column:
            kpi(label, value, note, color)

    left, middle, right = st.columns([1.8, 1.05, 1.05], gap="small")
    with left:
        st.markdown('<div class="panel-title">📈 XOM Price Chart</div>', unsafe_allow_html=True)
        market_chart(compact=True)
    with middle:
        st.markdown('<div class="panel-title">🧠 AI Decision Engine</div>', unsafe_allow_html=True)
        st.markdown('<div class="banner-strip">CNN1D + Logistic + SMA5/EMA10 filter<br>Model signals: 30m · Simulated exit: 60m</div>', unsafe_allow_html=True)
        kpi("Last recorded strategy action", str(last_decision["strategy_action"]), "Historical event — not an investment recommendation", "#36e5bd")
        st.markdown(" ")
        kpi("Risk decision", str(last_decision["risk_status"]), "Execution and risk rules", "#eec979")
        st.markdown(" ")
        kpi("Win rate (closed trades)", f"{result['win_rate_pct']:.2f}%", "Not a calibrated model confidence score", "#edf8ff")
    with right:
        st.markdown('<div class="panel-title">🏭 XOM Digital Investment Twin</div>', unsafe_allow_html=True)
        for n, label, sub in [
            ("01", "Market Data", "Historical XOMUSUSD Bid OHLC"),
            ("02", "AI Forecast", "CNN1D · Logistic"),
            ("03", "Strategy & Risk", "Consensus · trend · Kill Switch"),
            ("04", "Portfolio Execution", "Simulated BUY/SELL"),
            ("05", "Blockchain Ledger", "Linked SHA-256 decision records"),
        ]:
            st.markdown(
                f'<div class="banner-strip"><strong style="color:#37d6e9">{n} · {safe(label)}</strong>'
                f'<div class="micro">{safe(sub)}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="panel-title">🤖 Autonomous Investment Agents</div>', unsafe_allow_html=True)
    agents = [
        ("Market / Forecast Agent", "CNN1D & Logistic models", "MODEL OUTPUTS"),
        ("Strategy Agent", "Consensus + SMA/EMA", "RULE-BASED"),
        ("Risk Agent", "Daily loss / drawdown checks", safe(result["final_system_state"])),
        ("Execution Agent", "Delayed virtual orders", f"{result['completed_trades']} COMPLETED TRADES"),
        ("Portfolio Agent", "Capital & positions", f"${result['final_equity_usd']:,.2f}"),
    ]
    agent_cols = st.columns(5, gap="small")
    for column, (name, description, state) in zip(agent_cols, agents):
        with column:
            st.markdown(
                '<div class="agent">'
                f'<div class="agent-name">{safe(name)}</div>'
                f'<div class="agent-desc">{safe(description)}</div>'
                f'<div class="agent-state">{safe(state)}</div>'
                '</div>', unsafe_allow_html=True,
            )

    bottom_left, bottom_mid, bottom_right = st.columns([1.5, 1, 1.15], gap="small")
    with bottom_left:
        st.markdown('<div class="panel-title">💼 Portfolio & Backtest</div>', unsafe_allow_html=True)
        capital_chart(transactions, height=230)
    with bottom_mid:
        st.markdown('<div class="panel-title">🛡️ Risk Management</div>', unsafe_allow_html=True)
        st.metric("System state", result["final_system_state"])
        st.metric("Trading fees", f"${result['total_fees_usd']:.2f}")
        st.metric("Blocked orders", result["blocked_orders"])
    with bottom_right:
        st.markdown('<div class="panel-title">🔗 Blockchain Audit Trail</div>', unsafe_allow_html=True)
        try:
            blocks, _audit, checks = verify_ledger()
            # Use an explicit if/else: Streamlit's magic renderer would
            # display the DeltaGenerator from a ternary expression as source.
            if all(checks.values()):
                st.success(f"All checks passed · {len(blocks)} blocks")
            else:
                st.error("Integrity issue — audit requires attention")
            events = pd.DataFrame([
                {"Block": b["index"], "UTC": b["timestamp_utc"][:16], "Event": b["event_type"]}
                for b in blocks[-5:][::-1]
            ])
            st.dataframe(events, hide_index=True, width="stretch", height=178)
        except (OSError, ValueError, KeyError) as exc:
            st.error(f"Ledger verification error: {exc}")
    st.caption("Research demonstration only. Bid feed, historical simulation, hypothetical fees. No live quotes, real orders or validated future profits.")


def market_page():
    header("Market Analysis", "Historical OHLC and validated technical indicators")
    market_chart(compact=False)
    st.info("The price series uses XOMUSUSD Bid data from the validation simulator. It is not a live market feed.")


def agents_page(decisions):
    header("AI & Autonomous Agents", "Explore historical predictions and agent decisions")
    st.markdown("**Agent architecture:** Forecast → Strategy → Risk → Execution → Portfolio")
    st.caption("Forecast models are ML/DL; strategy, risk, and execution agents follow programmed rules.")
    actions = decisions["strategy_action"].fillna("UNKNOWN").value_counts()
    cols = st.columns(3)
    for c, action in zip(cols, ["BUY", "SELL", "HOLD"]):
        with c:
            st.metric(action + " decisions", int(actions.get(action, 0)))
    event = st.selectbox("Filter decision events", ["ALL"] + sorted(decisions["event"].dropna().unique().tolist()))
    filtered = decisions if event == "ALL" else decisions[decisions["event"].eq(event)]
    display_cols = ["timestamp_ms", "event", "predicted_class", "strategy_action", "risk_status", "execution_status", "system_state"]
    st.dataframe(filtered[display_cols].tail(100).iloc[::-1], hide_index=True, width="stretch")
    st.caption("Only historical recorded decisions are shown. No new orders are submitted.")


def portfolio_page(summary, transactions):
    header("Portfolio & Risk", "Audited simulated portfolio and horizon comparison")
    results = summary["results"]
    cols = st.columns(2)
    for c, label in zip(cols, ["30m", "60m"]):
        with c:
            r = results[label]
            st.subheader(f"Horizon {label}")
            st.metric("Net return", f"{r['net_return_pct']:+.4f}%")
            st.metric("Closed trades", r["completed_trades"])
            st.metric("Max drawdown", f"{r['max_observed_drawdown_pct']:.4f}%")
    st.subheader("Historical portfolio capital after closed trades (60m)")
    capital_chart(transactions, height=350)
    st.subheader("Simulated transaction log")
    st.dataframe(transactions.tail(80).iloc[::-1], hide_index=True, width="stretch")
    st.warning("Validation-only exploratory backtesting; the common 132-signal cohort was selected using future candle availability.")


def twin_page(summary):
    header("Digital Investment Twin", "Read-only digital simulation of the decision pipeline")
    st.markdown("### Model → Strategy → Risk → Execution → Ledger")
    st.info("This page currently visualizes saved historical decisions. Interactive What-If scenarios will be developed in a later stage.")
    r = summary["results"]["60m"]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Completed simulated trades", r["completed_trades"])
    with c2:
        st.metric("Simulated final equity", f"${r['final_equity_usd']:,.2f}")
    with c3:
        st.metric("Kill Switch", r["final_system_state"])


def blockchain_page():
    header("Blockchain Explorer", "Inspect local SHA-256 linked historical decision records")
    try:
        blocks, audit, checks = verify_ledger()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"Cannot load the blockchain ledger: {exc}")
        st.stop()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total blocks", len(blocks))
    with c2:
        st.metric("Decision blocks", len(blocks) - 1)
    with c3:
        st.metric("Verification", "PASS" if all(checks.values()) else "FAIL")
    if all(checks.values()):
        st.success("Block chain, final block hash and all source file hashes match the saved audit.")
    else:
        st.error("Integrity mismatch detected. Review the check results below.")
    st.dataframe(pd.DataFrame([{"check": name, "passed": ok} for name, ok in checks.items()]), hide_index=True, width="stretch")
    records = []
    for block in blocks:
        payload = block.get("payload", {})
        records.append({
            "Index": int(block["index"]), "Timestamp UTC": block["timestamp_utc"],
            "Event": block["event_type"],
            "Action": payload.get("strategy", {}).get("action", "—"),
            "Hash": block["hash"][:16] + "…",
        })
    data = pd.DataFrame(records)
    event = st.selectbox("Event type", ["ALL"] + sorted(data["Event"].unique().tolist()))
    filtered = data if event == "ALL" else data[data["Event"] == event]
    st.dataframe(filtered.iloc[::-1], hide_index=True, width="stretch", height=270)
    selected = st.selectbox("Inspect block", filtered["Index"].tolist()[::-1])
    st.json(blocks[int(selected)])
    st.caption("Local reconstructed audit chain; not distributed consensus or an immutable public blockchain.")


required = [SUMMARY_PATH, DECISIONS_PATH, TRANSACTIONS_PATH, LEDGER_PATH, AUDIT_PATH]
missing = [str(p) for p in required if not p.is_file()]
if missing:
    st.error("Project artifacts missing:")
    st.code("\n".join(missing))
    st.stop()

with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-x">XOM</span>.ai</div><div class="brand-small">AURUM · INVESTMENT TWIN</div>', unsafe_allow_html=True)
    st.divider()
    page = st.radio("NAVIGATION", [
        "🏠 Dashboard", "📈 Market Analysis", "🧠 AI & Autonomous Agents",
        "🏭 Digital Twin", "💼 Portfolio & Risk", "🔗 Blockchain Explorer",
    ])
    st.divider()
    st.caption("DATA: XOMUSUSD Bid 5m")
    st.caption("MODE: HISTORICAL · VALIDATION")
    st.caption("NO LIVE TRADES / NO REAL MONEY")

summary, decisions, transactions = load_reports()
if page == "🏠 Dashboard":
    dashboard(summary, decisions, transactions)
elif page == "📈 Market Analysis":
    market_page()
elif page == "🧠 AI & Autonomous Agents":
    agents_page(decisions)
elif page == "🏭 Digital Twin":
    twin_page(summary)
elif page == "💼 Portfolio & Risk":
    portfolio_page(summary, transactions)
else:
    blockchain_page()
