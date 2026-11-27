"""AURUM-AI | Premium NYSE V3 Validation Dashboard.

Installation: replace only frontend/pages/NYSE_V3_Backtesting.py with this file.
Run from project root: python -m streamlit run frontend/app.py

Read-only academic simulation. No live quotes, no Test data, no orders.
Uses the existing frontend.services.model_comparison.load_model_comparison().
"""
from __future__ import annotations

import base64
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# -----------------------------------------------------------------------------
# Paths and configuration (existing AURUM-AI structure)
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.services.model_comparison import load_model_comparison  # noqa: E402

DATA_DIR = (
    ROOT / "artifacts" / "backtesting" / "nyse_v3"
    / "lstm60m_validation_dashboard"
)
BANNER_PATH = ROOT / "frontend" / "assets" / "refinery_banner.jpg"  # optional

st.set_page_config(
    page_title="AURUM-AI | NYSE V3 Investment Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "blue": "#25B8FF",
    "cyan": "#24E3F0",
    "green": "#22E6A1",
    "red": "#FF6079",
    "amber": "#FFBF49",
    "muted": "#95AEC9",
    "text": "#F0F7FF",
    "grid": "#16344C",
}


# -----------------------------------------------------------------------------
# Data: every displayed financial figure comes from Validation artifacts.
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_files(summary_mtime: int, equity_mtime: int, trades_mtime: int):
    """mtime arguments invalidate the cache after regenerating CSV/JSON."""
    with (DATA_DIR / "summary.json").open(encoding="utf-8") as fh:
        summary = json.load(fh)
    equity = pd.read_csv(DATA_DIR / "equity_curve.csv")
    trades = pd.read_csv(DATA_DIR / "trades.csv")
    return summary, equity, trades


def get_data():
    files = [DATA_DIR / "summary.json", DATA_DIR / "equity_curve.csv", DATA_DIR / "trades.csv"]
    if not all(path.is_file() for path in files):
        missing = [str(path) for path in files if not path.is_file()]
        raise FileNotFoundError("Missing Validation artifacts: " + ", ".join(missing))

    summary, equity, trades = load_files(*(path.stat().st_mtime_ns for path in files))
    if summary.get("split") != "validation":
        raise ValueError("Dashboard refuses to load a non-Validation backtest")
    if not {
        "initial_capital_usd", "final_capital_usd", "net_pnl_usd", "return_pct",
        "maximum_drawdown_pct", "trades_closed", "signals", "win_rate_pct",
        "risk_halted", "cost_bps_per_side",
    }.issubset(summary):
        raise ValueError("Missing fields in summary.json")
    required_trades = {
        "side", "entry_timestamp_ms", "exit_timestamp_ms", "entry_price",
        "exit_price", "quantity", "net_pnl_usd", "exit_reason",
    }
    if not required_trades.issubset(trades):
        raise ValueError("Missing fields in trades.csv")
    if not {"timestamp_ms", "equity_usd"}.issubset(equity):
        raise ValueError("Missing fields in equity_curve.csv")
    if len(trades) != int(summary["trades_closed"]):
        raise ValueError("Transaction count inconsistent with summary.json")
    if "market_bars" in summary and len(equity) != int(summary["market_bars"]):
        raise ValueError("Equity bar count inconsistent with summary.json")
    if not equity["timestamp_ms"].is_monotonic_increasing or not equity["timestamp_ms"].is_unique:
        raise ValueError("Invalid equity chronology")
    if not np.isfinite(equity["equity_usd"].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite equity value")
    if not np.isfinite(trades["net_pnl_usd"].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite P&L")
    if abs(float(trades["net_pnl_usd"].sum()) - float(summary["net_pnl_usd"])) > 1e-5:
        raise ValueError("Transaction P&L does not reconcile with summary")
    if abs(float(equity["equity_usd"].iloc[-1]) - float(summary["final_capital_usd"])) > 1e-5:
        raise ValueError("Final equity does not match summary")

    equity = equity.copy()
    equity["datetime_utc"] = pd.to_datetime(equity["timestamp_ms"], unit="ms", utc=True)
    trades = trades.copy().reset_index(drop=True)
    trades["Trade #"] = np.arange(1, len(trades) + 1)
    return summary, equity, trades


try:
    summary, equity, trades = get_data()
    comparison = load_model_comparison()
    if len(comparison) != 12 or len(comparison[["Model", "Horizon (min)"]].drop_duplicates()) != 12:
        raise ValueError("Expected 12 unique model/horizon Validation reports")
except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
    st.error(f"Cannot load AURUM-AI Validation data: {exc}")
    st.stop()

initial = float(summary["initial_capital_usd"])
final = float(summary["final_capital_usd"])
net_pnl = float(summary["net_pnl_usd"])
equity_values = equity["equity_usd"].to_numpy(dtype=float)
peaks = np.maximum.accumulate(np.r_[initial, equity_values])[1:]
drawdown_pct = 100 * (equity_values / peaks - 1)


# -----------------------------------------------------------------------------
# Premium visual system; scoped to the page in which this file is executed.
# -----------------------------------------------------------------------------
st.markdown("""
<style>
:root {color-scheme: dark;}
.stApp {background: #06101d; color: #e8f5ff;}
[data-testid="stAppViewContainer"] {background: radial-gradient(ellipse at 65% 0%, #092842 0%, #06101d 34%, #06101d 100%);}
[data-testid="stHeader"] {background: rgba(6, 16, 29, .83);}
[data-testid="stSidebar"] {background: linear-gradient(180deg,#06182d,#071324 65%,#06101d);border-right:1px solid #19456a;}

[data-testid="stSidebar"] {

.block-container {
    padding-left: 1.25rem !important;
    padding-right: 1.25rem !important;
}

    width: 248px !important;
    min-width: 248px !important;
    max-width: 248px !important;
}

/* AURUM-AI — Full-width Premium Dashboard */
[data-testid="stMainBlockContainer"] {
    max-width: 100% !important;
    width: 100% !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    padding-top: 1rem !important;
}


[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {color:#b1cbe8;}
.block-container {max-width: 1850px; padding-top: 1rem; padding-bottom: 2.3rem;}
#MainMenu, footer {visibility:hidden;}
h1,h2,h3 {color: #eff9ff !important;letter-spacing:-.025em;}
h1 {font-size:1.85rem !important;} h2 {font-size:1.32rem !important;} h3 {font-size:1.06rem !important;}
p {color:#afc5dc;}
[data-testid="stMetric"] {
  background: linear-gradient(125deg,#102b45,#0a1f34 80%);
  border:1px solid #175075;border-radius:12px;padding:15px 14px;
  box-shadow: inset 0 1px 0 rgba(87,218,255,.08),0 8px 22px rgba(0,0,0,.12);
}
[data-testid="stMetricLabel"] p {color:#aac9df !important;font-size:.81rem!important;}
[data-testid="stMetricValue"] {color:#f4fbff!important;font-weight:750;font-size:1.43rem!important;}
[data-testid="stMetricDelta"] {font-size:.76rem;}
[data-testid="stPlotlyChart"] {background:#071b2d;border:1px solid #15476b;border-radius:12px;padding:4px;}
[data-testid="stDataFrame"] {border:1px solid #174466;border-radius:10px;}
[data-testid="stAlert"] {border-radius:10px;}
button[kind="secondary"],button[kind="primary"] {border-radius:10px !important;}
.stButton button {border:1px solid #1b6495;background:#092c4b;color:#e9f8ff;}
[data-baseweb="select"] > div {background:#092238!important;border-color:#205578!important;}
.stTabs [data-baseweb="tab-list"] {gap:.35rem;border-bottom:1px solid #1e4665;}
.stTabs [data-baseweb="tab"] {background:#092339;border-radius:9px 9px 0 0;padding:12px 18px;color:#a8c6e2;}
.stTabs [aria-selected="true"] {background:#10426c!important;color:#fff!important;}
.aurum-hero {
  position:relative;overflow:hidden;min-height:164px;padding:25px 29px 22px;
  border-radius:16px;border:1px solid #286c94;margin-bottom:14px;
  background: linear-gradient(100deg,rgba(6,19,34,.97) 0%,rgba(6,25,44,.86) 49%,rgba(5,30,55,.49) 100%),
              radial-gradient(circle at 80% 70%,#075a86,#09223e 35%,#091528);
  box-shadow:0 8px 34px rgba(0,0,0,.25),inset 0 1px 0 rgba(255,255,255,.07);
}
.aurum-hero:after {content:"";pointer-events:none;position:absolute;right:-100px;top:-45px;
 width:430px;height:240px;background:repeating-linear-gradient(90deg,transparent 0 28px,#1ebaff22 29px 30px);
 transform:skewX(-23deg);opacity:.65;}
.aurum-eyebrow {color:#55d5ff;font-size:11px;font-weight:800;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;}
.aurum-title {font-size:clamp(24px,3vw,40px);line-height:1.1;font-weight:900;color:#fff;letter-spacing:-1.4px;}
.aurum-sub {font-size:14px;color:#b1d4ec;margin:12px 0 12px;}
.aurum-chip {display:inline-block;padding:5px 9px;margin:0 6px 2px 0;border:1px solid #27628a;
 border-radius:6px;font-size:10px;color:#8ee1ff;background:#082945;}
.aurum-status {margin:6px 0 15px;padding:10px 15px;border:1px solid #936f24;border-radius:9px;
 background:linear-gradient(90deg,#292612,#191d19);color:#ffd87c;font-size:12px;font-weight:650;}
.aurum-section {color:#effaff;margin:5px 0 12px;font-size:16px;font-weight:790;letter-spacing:.2px;}
.aurum-note {padding:9px 13px;border:1px solid #215172;border-radius:9px;background:#0c2a43;
 color:#9fc5df;font-size:12px;margin:9px 0 14px;}
.aurum-brand {font-size:27px;font-weight:900;letter-spacing:-1px;color:#f5fbff;line-height:1.1;}
.aurum-brand span {color:#49c9ff;}
.aurum-brand-sub {font-size:10px;color:#8db7d8;letter-spacing:2px;margin-top:4px;}
.aurum-separator {height:1px;background:#173d5b;margin:18px 0;}
.aurum-foot {font-size:11px;color:#7c9fbb;padding:20px 0 0;border-top:1px solid #23435d;}

/* Premium Financial KPI Cards */

[data-testid="stMetric"] {
    position: relative;
    overflow: hidden;
    background: linear-gradient(145deg, #102e4b, #071b30);
    border: 1px solid rgba(37, 184, 255, 0.45);
    border-radius: 14px;
    padding: 18px 16px;
    min-height: 105px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

[data-testid="stMetric"]::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, #00E5FF, #25B8FF, transparent);
}

[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 28px rgba(0, 185, 255, 0.18);
}

</style>
""", unsafe_allow_html=True)


# User may optionally supply a clean refinery photograph as a local file.
# The layout still works without a photograph.
hero_style = ""
if BANNER_PATH.is_file():
    try:
        encoded = base64.b64encode(BANNER_PATH.read_bytes()).decode("ascii")
        hero_style = (
            ' style="background-image:linear-gradient(100deg,rgba(6,19,34,.98),'
            'rgba(5,22,42,.78) 48%,rgba(3,23,40,.12)),'
            f'url(data:image/jpeg;base64,{encoded});background-size:cover;background-position:center;"'
        )
    except OSError:
        pass

with st.sidebar:
    st.markdown('<div class="aurum-brand">◈ AURUM<span>-AI</span></div>'
                '<div class="aurum-brand-sub">QUANTITATIVE INTELLIGENCE</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="aurum-separator"></div>', unsafe_allow_html=True)
    st.caption("NYSE V3 · XOM")
    st.markdown("**🟢 HISTORICAL VALIDATION**")
    st.caption("LSTM 60m · 12 model experiments")
    st.markdown('<div class="aurum-separator"></div>', unsafe_allow_html=True)
    
    st.caption("NAVIGATION")

    navigation_items = [
        ("📊 Cockpit", "🏠 Cockpit"),
        ("📈 Overview", "📈 Overview"),
        ("📋 Transactions", "📋 Transactions"),
        ("🛡️ Risk Management", "🛡️ Risk Management"),
        ("🧠 Model Comparison", "🧠 Model Comparison"),
    ]

    for index, (button_label, tab_label) in enumerate(navigation_items):
        if st.button(
            button_label,
            key=f"aurum_nav_{index}",
            width="stretch",
        ):
            st.session_state["aurum_active_tab"] = tab_label
            st.rerun()
st.markdown('<div class="aurum-separator"></div>', unsafe_allow_html=True)
st.info("Research simulation only...")

st.markdown(
    f'<div class="aurum-hero"{hero_style}>'
    '<div class="aurum-eyebrow">✦ Quantitative Investment Research · NYSE V3</div>'
    '<div class="aurum-title">AURUM-AI | Investment Intelligence</div>'
    '<div class="aurum-sub">XOM · LSTM 60m · Multi-Agent Trading System · Historical Validation</div>'
    '<span class="aurum-chip">XOM / NYSE</span>'
    '<span class="aurum-chip">3 Deep Learning Models</span>'
    '<span class="aurum-chip">12 Experiments</span>'
    '<span class="aurum-chip">Read-only Simulation</span>'
    '</div>', unsafe_allow_html=True,
)
st.markdown('<div class="aurum-status">⚠ VALIDATION ONLY — Historical Bid-open prices, hypothetical costs. No live market data, real orders or investment advice.</div>',
            unsafe_allow_html=True)



def show_kpis():
    return_pct = float(summary["return_pct"])
    drawdown_pct = float(summary["maximum_drawdown_pct"])
    win_rate = float(summary["win_rate_pct"])

    # Dynamic financial colors
    def performance_color(value):
        if value > 0:
            return "#20E6A4"   # Green: positive
        elif value < 0:
            return "#FF6277"   # Red: negative
        return "#B8D4E8"       # Neutral

    cards = [
        (
            "💰 Final Capital",
            f"${final:,.2f}",
            "#38BDF8",
        ),
        (
            "📈 Net P&L",
            f"${net_pnl:+,.2f}",
            performance_color(net_pnl),
        ),
        (
            "↗ Total Return",
            f"{return_pct:+.4f}%",
            performance_color(return_pct),
        ),
        (
            "🛡 Max Drawdown",
            f"{drawdown_pct:.4f}%",
            "#FBBF24",
        ),
        (
            "🎯 Signals / Trades",
            f"{int(summary['signals'])} / {int(summary['trades_closed'])}",
            "#38BDF8",
        ),
        (
            "✓ Win Rate",
            f"{win_rate:.1f}%",
            "#B8D4E8",
        ),
    ]

    st.markdown(
        """
        <style>
        .aurum-smart-kpi {
            background: linear-gradient(
                145deg, #102e4b, #071b30
            );
            border: 1px solid #175075;
            border-top: 3px solid var(--kpi-color);
            border-radius: 14px;
            padding: 17px 12px;
            min-height: 112px;
            transition: transform .2s ease,
                        box-shadow .2s ease;
        }

        .aurum-smart-kpi:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 26px
                        rgba(0,185,255,.15);
        }

        .aurum-smart-kpi-label {
            color: #AAC9DF;
            font-size: .78rem;
            margin-bottom: 9px;
        }

        .aurum-smart-kpi-value {
            color: var(--kpi-color);
            font-size: clamp(1rem, 1.4vw, 1.5rem);
            font-weight: 800;
            white-space: nowrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(6, gap="small")

    for col, (label, value, color) in zip(cols, cards):
        with col:
            st.markdown(
                f'<div class="aurum-smart-kpi" '
                f'style="--kpi-color:{color};">'
                f'<div class="aurum-smart-kpi-label">'
                f'{label}</div>'
                f'<div class="aurum-smart-kpi-value">'
                f'{value}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )



def style_fig(fig: go.Figure, height: int = 350, **kwargs):
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#b8d4e8", size=11),
        margin=dict(l=17, r=15, t=24, b=24),
        hovermode="x unified", showlegend=False,
        xaxis=dict(gridcolor=COLORS["grid"], zeroline=False),
        yaxis=dict(gridcolor=COLORS["grid"], zeroline=False),
        **kwargs,
    )
    return fig


def equity_chart(height=340):
    f = go.Figure()
    f.add_trace(go.Scatter(
        x=equity["datetime_utc"], y=equity["equity_usd"], name="Simulated Equity",
        mode="lines", line=dict(color=COLORS["blue"], width=2),
        hovertemplate="%{x|%Y-%m-%d %H:%M} UTC<br>Equity: $%{y:,.4f}<extra></extra>",
    ))
    f.add_hline(y=initial, line_dash="dash", line_color=COLORS["amber"],
                annotation_text="Initial $10,000", annotation_position="bottom left")
    return style_fig(f, height, xaxis_title="Date (UTC)", yaxis_title="Equity (USD)")


def drawdown_chart(height=305):
    f = go.Figure(go.Scatter(
        x=equity["datetime_utc"], y=drawdown_pct,
        mode="lines", fill="tozeroy", line=dict(color=COLORS["red"], width=1.8),
        fillcolor="rgba(255,96,121,.36)", name="Historical drawdown",
        hovertemplate="%{x|%Y-%m-%d %H:%M} UTC<br>Drawdown: %{y:.5f}%<extra></extra>",
    ))
    return style_fig(f, height, xaxis_title="Date (UTC)", yaxis_title="Drawdown (%)")


def pnl_chart(data: pd.DataFrame, height=325):
    f = go.Figure()
    f.add_trace(go.Bar(
        x=[f"Trade {n}" for n in data["Trade #"]],
        y=data["net_pnl_usd"],
        marker_color=[COLORS["green"] if v >= 0 else COLORS["red"] for v in data["net_pnl_usd"]],
        text=[f"${v:+.4f}" for v in data["net_pnl_usd"]], textposition="outside",
        hovertemplate="%{x}: $%{y:.4f}<extra></extra>",
    ))
    f.add_hline(y=0, line_color="#8fb0c6", line_width=1)
    return style_fig(f, height, xaxis_title="Original trade number", yaxis_title="Net P&L (USD)")


def present_trades(data: pd.DataFrame):
    frame = data.copy()
    frame["Entry (UTC)"] = pd.to_datetime(frame["entry_timestamp_ms"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M")
    frame["Exit (UTC)"] = pd.to_datetime(frame["exit_timestamp_ms"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M")
    frame = frame.rename(columns={
        "side": "Direction", "entry_price": "Entry Price",
        "exit_price": "Exit Price", "quantity": "Quantity",
        "net_pnl_usd": "Net P&L (USD)", "exit_reason": "Exit Reason",
    })
    return frame[[
        "Trade #", "Direction", "Entry (UTC)", "Exit (UTC)",
        "Entry Price", "Exit Price", "Quantity", "Net P&L (USD)", "Exit Reason",
    ]]


TRADE_COLUMNS = {
    "Entry Price": st.column_config.NumberColumn(format="%.3f"),
    "Exit Price": st.column_config.NumberColumn(format="%.3f"),
    "Quantity": st.column_config.NumberColumn(format="%.4f"),
    "Net P&L (USD)": st.column_config.NumberColumn(format="$%.4f"),
}


def winners_table():
    rows = []
    for horizon, g in comparison.groupby("Horizon (min)", sort=True):
        a = g.loc[g["MAE"].idxmin()]
        b = g.loc[g["RMSE"].idxmin()]
        rows.append({
            "Horizon (min)": int(horizon), "Best MAE Model": a["Model"],
            "Best MAE": float(a["MAE"]), "Best RMSE Model": b["Model"],
            "Best RMSE": float(b["RMSE"]),
        })
    return pd.DataFrame(rows)


def heatmap_chart(height=315):
    data = comparison.pivot(index="Model", columns="Horizon (min)", values="MAE Improvement (%)")
    data = data.loc[["LSTM", "GRU", "CNN1D"], [5, 15, 30, 60]]
    z = data.to_numpy(dtype=float)
    lim = max(abs(float(z.min())), abs(float(z.max())), 1e-10)
    f = go.Figure(go.Heatmap(
        z=z, x=["5 min", "15 min", "30 min", "60 min"], y=data.index.tolist(),
        zmin=-lim, zmax=lim, zmid=0, colorscale="RdYlGn",
        text=[[f"{v:+.4f}%" for v in row] for row in z],
        texttemplate="%{text}",
        colorbar=dict(title="MAE gain (%)", len=.85),
        hovertemplate="%{y} / %{x}<br>MAE gain: %{z:.5f}%<extra></extra>",
    ))
    return style_fig(f, height, xaxis_title="Prediction horizon", yaxis_title="Model")


# -----------------------------------------------------------------------------
# Main: compact executive cockpit + the four existing detailed experiences.
# -----------------------------------------------------------------------------
tab_cockpit, tab_overview, tab_trades, tab_risk, tab_models = st.tabs(
    [
        "🏠 Cockpit",
        "📈 Overview",
        "📋 Transactions",
        "🛡️ Risk Management",
        "🧠 Model Comparison",
    ],
    key="aurum_active_tab",
    on_change="rerun",
)

with tab_cockpit:
    show_kpis()
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.18, 1.0], gap="small")
    with left:
        st.markdown('<div class="aurum-section">📈 Portfolio Equity Curve</div>', unsafe_allow_html=True)
        st.plotly_chart(equity_chart(345), width="stretch", key="cockpit_equity")
    with right:
        st.markdown('<div class="aurum-section">📋 Executed Transactions</div>', unsafe_allow_html=True)
        st.dataframe(present_trades(trades)[[
            "Trade #", "Direction", "Entry (UTC)", "Exit (UTC)", "Net P&L (USD)"
        ]], width="stretch", hide_index=True, height=344,
            column_config={"Net P&L (USD)": TRADE_COLUMNS["Net P&L (USD)"]})
    left, right = st.columns([1.0, 1.0], gap="small")
    with left:
        st.markdown('<div class="aurum-section">💹 Net P&L by Transaction</div>', unsafe_allow_html=True)
        st.plotly_chart(pnl_chart(trades, 300), width="stretch", key="cockpit_pnl")
    with right:
        st.markdown('<div class="aurum-section">🛡️ Risk · Historical Drawdown</div>', unsafe_allow_html=True)
        st.plotly_chart(drawdown_chart(300), width="stretch", key="cockpit_drawdown")
    left, right = st.columns([1.0, 1.05], gap="small")
    with left:
        st.markdown('<div class="aurum-section">🧠 Best Models by Horizon</div>', unsafe_allow_html=True)
        st.dataframe(winners_table(), width="stretch", hide_index=True,
                     column_config={"Best MAE": st.column_config.NumberColumn(format="%.8f"),
                                    "Best RMSE": st.column_config.NumberColumn(format="%.8f")})
        st.caption("Winners selected on Validation only; not independent Test results.")
    with right:
        st.markdown('<div class="aurum-section">🔬 MAE Improvement Heatmap</div>', unsafe_allow_html=True)
        st.plotly_chart(heatmap_chart(280), width="stretch", key="cockpit_heatmap")
    st.markdown('<div class="aurum-note">The Cockpit is an overview of historical simulation. Model forecast error and simulated trading P&L are different metrics. Five trades cannot establish real-world profitability.</div>', unsafe_allow_html=True)

with tab_overview:
    st.subheader("Portfolio Performance · Historical Simulation")
    show_kpis()
    st.markdown("### Portfolio Equity Curve")
    st.plotly_chart(equity_chart(475), width="stretch", key="overview_equity")
    st.caption(f"Initial capital ${initial:,.2f}. {len(equity):,} historical validation bars; Bid-open simulation with hypothetical costs.")
    st.markdown("### Historical Drawdown")
    st.plotly_chart(drawdown_chart(320), width="stretch", key="overview_drawdown")

with tab_trades:
    st.subheader("Trading Activity · Validation")
    a, b, c = st.columns(3)
    a.metric("Trading Signals", int(summary["signals"]))
    b.metric("Closed Trades", int(summary["trades_closed"]))
    c.metric("Win Rate (5 trades)", f"{float(summary['win_rate_pct']):.1f}%")
    st.markdown("### Executed Trades")
    direction = st.selectbox("Filter by Trade Direction", ["ALL", "LONG", "SHORT"], key="aurum_trade_filter")
    filtered = trades if direction == "ALL" else trades.loc[trades["side"] == direction]
    a, b = st.columns(2)
    a.metric("Displayed Trades", f"{len(filtered)} / {len(trades)}")
    b.metric("Filtered Net P&L", f"${float(filtered['net_pnl_usd'].sum()):+,.4f}")
    display = present_trades(filtered)
    st.dataframe(display, width="stretch", hide_index=True, column_config=TRADE_COLUMNS)
    st.download_button(
        "📥 Download Transactions CSV", data=display.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
        file_name=f"aurum_ai_trades_{direction.lower()}_validation.csv", mime="text/csv",
        key="aurum_download_trades",
    )
    st.caption("Downloaded CSV uses semicolon delimiters and comma decimals (French Excel format). Simulated transactions only.")
    st.markdown("### Net P&L by Transaction")
    st.plotly_chart(pnl_chart(filtered, 410), width="stretch", key="trades_pnl")
    st.caption("Trade numbers retain their original identifiers after applying a direction filter.")

with tab_risk:
    st.subheader("Risk Management · Simulated Portfolio")
    a, b, c = st.columns(3)
    a.metric("Maximum Drawdown", f"{float(summary['maximum_drawdown_pct']):.6f}%")
    b.metric("Backtest Cost", f"{float(summary['cost_bps_per_side']):g} bps / side")
    c.metric("Kill Switch Triggered", "YES" if summary["risk_halted"] else "NO")
    st.markdown('<div class="aurum-note">Actual spreads, slippage, borrow fees, financing, market impact and live order execution are not fully modeled.</div>', unsafe_allow_html=True)

    st.markdown("### Transaction Cost Sensitivity")
    selected_cost = st.slider(
        "Hypothetical Cost (bps per side)", min_value=0, max_value=40,
        value=int(summary["cost_bps_per_side"]), step=5, key="aurum_cost_slider",
    )
    original_cost = float(summary["cost_bps_per_side"])
    total_notional = float((trades["quantity"] * (trades["entry_price"] + trades["exit_price"])).sum())
    gross_pnl = net_pnl + original_cost * total_notional / 10000
    adjusted_pnl = gross_pnl - selected_cost * total_notional / 10000
    break_even = gross_pnl * 10000 / total_notional if total_notional > 0 else math.nan
    a, b = st.columns(2)
    a.metric("Adjusted Net P&L", f"${adjusted_pnl:+,.4f}")
    b.metric("Break-Even Cost (per side)", f"{break_even:.2f} bps" if math.isfinite(break_even) else "N/A")

    cost_levels = list(range(0, 41, 5))
    costs_pnl = [gross_pnl - cost * total_notional / 10000 for cost in cost_levels]
    cost_fig = go.Figure()
    cost_fig.add_trace(go.Scatter(
        x=cost_levels, y=costs_pnl, mode="lines+markers", name="Hypothetical net P&L",
        line=dict(color=COLORS["blue"], width=3), marker=dict(size=8),
    ))
    cost_fig.add_hline(y=0, line_color=COLORS["red"], line_dash="dash", annotation_text="Zero P&L")
    if math.isfinite(break_even):
        cost_fig.add_vline(x=break_even, line_color=COLORS["amber"], line_dash="dot",
                           annotation_text="Break-even")
    cost_fig.add_trace(go.Scatter(
        x=[selected_cost], y=[adjusted_pnl], mode="markers", name="Selected scenario",
        marker=dict(color=COLORS["green"] if adjusted_pnl >= 0 else COLORS["red"],
                    size=15, symbol="diamond"),
    ))
    cost_fig = style_fig(cost_fig, 410, xaxis_title="Cost (bps per side)", yaxis_title="Net P&L (USD)")
    cost_fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(cost_fig, width="stretch", key="risk_cost_chart")
    st.caption("This varies costs for the same 5 executed trades and quantities; it is not a new backtest. Historical drawdown below uses the original 10 bps scenario.")
    st.markdown("### Historical Drawdown")
    st.plotly_chart(drawdown_chart(365), width="stretch", key="risk_drawdown")

with tab_models:
    st.subheader("Deep Learning Model Comparison · Validation")
    st.caption("3 models × 4 forecast horizons = 12 experiments. All reported model selection uses Validation; Test remains unused.")
    a, b, c = st.columns(3)
    a.metric("Total Experiments", len(comparison))
    b.metric("Models Tested", comparison["Model"].nunique())
    c.metric("Prediction Horizons", comparison["Horizon (min)"].nunique())

    st.markdown("### Validation Performance")
    st.dataframe(comparison, width="stretch", hide_index=True, column_config={
        "MAE": st.column_config.NumberColumn(format="%.8f"),
        "RMSE": st.column_config.NumberColumn(format="%.8f"),
        "Zero Return MAE": st.column_config.NumberColumn(format="%.8f"),
        "Zero Return RMSE": st.column_config.NumberColumn(format="%.8f"),
        "MAE Improvement (%)": st.column_config.NumberColumn(format="%.4f %%"),
    })

    st.markdown("### Model Performance by Horizon")
    horizon = st.selectbox("Select Prediction Horizon", [5, 15, 30, 60],
                           format_func=lambda h: f"{h} minutes", key="aurum_models_horizon")
    selected = comparison.loc[comparison["Horizon (min)"] == horizon].sort_values("Model")
    gain_fig = go.Figure(go.Bar(
        x=selected["Model"], y=selected["MAE Improvement (%)"],
        marker_color=[COLORS["green"] if v > 0 else COLORS["red"] for v in selected["MAE Improvement (%)"]],
        text=[f"{v:+.4f}%" for v in selected["MAE Improvement (%)"]], textposition="outside",
    ))
    gain_fig.add_hline(y=0, line_color="#a2b9c9", line_dash="dash")
    st.plotly_chart(style_fig(gain_fig, 355, yaxis_title="MAE gain vs Zero Return (%)"),
                    width="stretch", key="models_gain_chart")

    st.markdown("### Prediction Error Comparison")
    metric = st.radio("Select Evaluation Metric", ["MAE", "RMSE"], horizontal=True,
                      key="aurum_error_metric")
    baseline = float(selected[f"Zero Return {metric}"].iloc[0])
    best = selected.loc[selected[metric].idxmin()]
    improvement = (baseline - float(best[metric])) / baseline * 100 if baseline > 0 else math.nan
    a, b = st.columns(2)
    a.metric(f"Lowest {metric} — Validation", str(best["Model"]))
    b.metric(f"Best {metric} Value", f"{float(best[metric]):.8f}",
             delta=f"{improvement:+.4f}% vs Zero Return" if math.isfinite(improvement) else None,
             delta_color="normal")
    error_fig = go.Figure(go.Bar(
        x=selected["Model"], y=selected[metric], marker_color=COLORS["blue"],
        text=[f"{v:.8f}" for v in selected[metric]], textposition="outside",
    ))
    error_fig.add_hline(y=baseline, line_color=COLORS["amber"], line_dash="dash",
                        annotation_text="Zero Return Baseline")
    error_fig = style_fig(error_fig, 390, yaxis_title=metric)
    error_fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(error_fig, width="stretch", key="models_error_chart")
    st.caption("Lower MAE/RMSE is better. The y-axis begins at zero: tiny differences may be hard to see, so consult the precise table values.")

    st.markdown("### Best Models by Horizon")
    st.dataframe(winners_table(), width="stretch", hide_index=True, column_config={
        "Best MAE": st.column_config.NumberColumn(format="%.8f"),
        "Best RMSE": st.column_config.NumberColumn(format="%.8f"),
    })
    st.caption("Winners are selected on the same Validation set shown above, not independently tested.")
    st.markdown("### MAE Improvement Heatmap")
    st.plotly_chart(heatmap_chart(375), width="stretch", key="models_heatmap")
    st.caption("Green means lower MAE than Zero Return; red means higher MAE. Color magnitude is relative and improvements remain small.")
    st.download_button(
        "📥 Download Model Comparison CSV",
        data=comparison.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
        file_name="aurum_ai_model_comparison_validation.csv", mime="text/csv",
        key="aurum_download_models",
    )
    st.info("Validation-selected models: small forecast-error improvements are not statistically established as trading gains. No Test dataset was used in this view.")

st.markdown(
    '<div class="aurum-foot">◈ AURUM-AI · NYSE V3 · Academic investment simulation · '
    'Validation only · Bid-open execution approximation · No real trading · '
    'Past simulated results do not predict future profitability.</div>',
    unsafe_allow_html=True,
)
