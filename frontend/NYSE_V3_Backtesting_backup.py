from frontend.services.model_comparison import load_model_comparison
from pathlib import Path
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# AURUM-AI — NYSE V3 BACKTESTING DASHBOARD
#
# Model: LSTM
# Horizon: 60 minutes
# Asset: XOM
# Dataset: Validation only
#
# Academic financial simulation — no real trading
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT
    / "artifacts"
    / "backtesting"
    / "nyse_v3"
    / "lstm60m_validation_dashboard"
)


# ============================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AURUM-AI | NYSE V3",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# AURUM-AI — Professional KPI Cards Styling
# ============================================================

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: #102338;
        border: 1px solid #214463;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 8px;
    }

    div[data-testid="stMetricLabel"] {
        color: #9DB2C9;
        font-weight: 500;
    }

    div[data-testid="stMetricValue"] {
        color: #F2F6FC;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# 3. DATA LOADING
# ============================================================

@st.cache_data
def load_data():

    # Load performance metrics
    with open(
        DATA_DIR / "summary.json",
        encoding="utf-8",
    ) as file:
        summary = json.load(file)

    # Load portfolio equity curve
    equity = pd.read_csv(
        DATA_DIR / "equity_curve.csv"
    )

    # Load simulated transactions
    trades = pd.read_csv(
        DATA_DIR / "trades.csv"
    )

    # Convert equity timestamps to datetime
    equity["datetime_utc"] = pd.to_datetime(
        equity["datetime_utc"],
        utc=True,
    )

    return summary, equity, trades


try:
    summary, equity, trades = load_data()

except (OSError, ValueError, KeyError) as error:
    st.error(
        f"Dashboard data loading error: {error}"
    )
    st.stop()


# ============================================================
# 4. HEADER
# ============================================================

st.title(
    "AURUM-AI | NYSE V3 Backtesting"
)

st.caption(
    "XOM • LSTM 60m • Multi-Agent Trading System"
)

st.warning(
    "VALIDATION ONLY — Academic simulation. "
    "Historical Bid prices, hypothetical trading costs, "
    "and no real trading."
)


# ============================================================
# 5. DASHBOARD NAVIGATION
# ============================================================


overview_tab, trades_tab, risk_tab, model_tab = st.tabs(
    [
        "📊 Overview",
        "📋 Transactions",
        "🛡️ Risk",
        "🧠 Model Comparison",
    ]
)



# ============================================================
# 6. OVERVIEW TAB
# ============================================================

with overview_tab:

    st.subheader(
        "Portfolio Performance"
    )

    # ---------------- KPI METRICS ----------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Final Capital",
            f"${summary['final_capital_usd']:,.2f}",
        )

    with col2:
        st.metric(
            "Net P&L",
            f"${summary['net_pnl_usd']:+,.2f}",
        )

    with col3:
        st.metric(
            "Return",
            f"{summary['return_pct']:+.4f}%",
        )

    with col4:
        st.metric(
            "Max Drawdown",
            f"{summary['maximum_drawdown_pct']:.4f}%",
        )

    st.divider()

    # ---------------- EQUITY CURVE ----------------

    st.subheader(
        "Portfolio Equity Curve"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=equity["datetime_utc"],
            y=equity["equity_usd"],
            mode="lines",
            name="Portfolio Equity",
            line=dict(
                color="#38BDF8",
                width=2,
            ),
        )
    )

    # Initial capital reference line
    fig.add_hline(
        y=float(summary["initial_capital_usd"]),
        line_dash="dash",
        line_color="#FBBF24",
        line_width=2,
        annotation_text="Initial Capital ($10,000)",
        annotation_position="top left",
    )


    fig.update_layout(
        template="plotly_dark",
        height=440,
        xaxis_title="Date (UTC)",
        yaxis_title="Equity (USD)",
        hovermode="x unified",
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10,
        ),
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )


# ============================================================
# 7. TRANSACTIONS TAB
# ============================================================

with trades_tab:

    st.subheader(
        "Trading Activity"
    )

    # ---------------- TRADING METRICS ----------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Trading Signals",
            summary["signals"],
        )

    with col2:
        st.metric(
            "Closed Trades",
            summary["trades_closed"],
        )

    with col3:
        st.metric(
            "Win Rate",
            f"{summary['win_rate_pct']:.1f}%",
        )

    st.divider()

    # ---------------- TRADES TABLE ----------------

    st.subheader(
        "Executed Trades"
    )

    # Create a display copy.
    # Never modify the original transaction data.
    display_trades = trades.copy()

    # Interactive filter for trade direction
    direction_filter = st.selectbox(
        "Filter by Trade Direction",
        options=["ALL", "LONG", "SHORT"],
        index=0,
    )

    if direction_filter != "ALL":
        display_trades = display_trades[
            display_trades["side"] == direction_filter
        ].copy()
    
    # Dynamic metrics for the selected trade direction
    filtered_count = len(display_trades)
    filtered_pnl = display_trades["net_pnl_usd"].sum()

    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric(
            "Displayed Trades",
            f"{filtered_count} / {len(trades)}",
        )

    with metric2:
        st.metric(
            "Filtered Net P&L",
            f"${filtered_pnl:+,.4f}",
        )



    # Convert Unix milliseconds to readable UTC dates.
    display_trades["Entry (UTC)"] = pd.to_datetime(
        display_trades["entry_timestamp_ms"],
        unit="ms",
        utc=True,
    ).dt.strftime("%Y-%m-%d %H:%M")

    display_trades["Exit (UTC)"] = pd.to_datetime(
        display_trades["exit_timestamp_ms"],
        unit="ms",
        utc=True,
    ).dt.strftime("%Y-%m-%d %H:%M")

    # Rename columns for the interface.
    display_trades = display_trades.rename(
        columns={
            "side": "Direction",
            "entry_price": "Entry Price",
            "exit_price": "Exit Price",
            "quantity": "Quantity",
            "net_pnl_usd": "Net P&L (USD)",
            "exit_reason": "Exit Reason",
        }
    )

    # Show only useful columns in a clear order.
    display_trades = display_trades[
        [
            "Direction",
            "Entry (UTC)",
            "Exit (UTC)",
            "Entry Price",
            "Exit Price",
            "Quantity",
            "Net P&L (USD)",
            "Exit Reason",
        ]
    ]

    # Display the formatted transaction table.
    st.dataframe(
        display_trades,
        width="stretch",
        hide_index=True,
        column_config={
            "Entry Price": st.column_config.NumberColumn(
                format="%.3f"
            ),
            "Exit Price": st.column_config.NumberColumn(
                format="%.3f"
            ),
            "Quantity": st.column_config.NumberColumn(
                format="%.4f"
            ),
            "Net P&L (USD)": st.column_config.NumberColumn(
                format="$%.4f"
            ),
        },
    )

    # Download currently filtered transactions 

    csv_data = display_trades.to_csv(
        index=False,
        sep=";",
        decimal=",",
    ).encode("utf-8-sig")
    st.download_button(
        label="📥 Download Transactions CSV",
        data=csv_data,
        file_name=f"aurum_ai_trades_{direction_filter.lower()}_validation.csv",
        mime="text/csv",
    )


    st.caption(
        "All transactions are hypothetical "
        "and were generated using Validation data."
    )

    # ========================================================
    # P&L PER TRANSACTION — INTERACTIVE CHART
    # ========================================================

    st.divider()
    st.subheader("Net P&L by Transaction")

    chart_data = display_trades.copy()

    chart_data["Trade"] = [
        f"Trade {i + 1}"
        for i in chart_data.index
    ]


    pnl_values = chart_data["Net P&L (USD)"]

    colors = [
        "#22C55E" if pnl >= 0 else "#EF4444"
        for pnl in pnl_values
    ]

    pnl_fig = go.Figure()

    pnl_fig.add_trace(
        go.Bar(
            x=chart_data["Trade"],
            y=pnl_values,
            marker_color=colors,
            text=[f"${value:+.4f}" for value in pnl_values],
            textposition="outside",
            hovertemplate=(
                "%{x}<extra></extra>"
                "<br>Net P&L: $%{y:.4f}"
            ),
        )
    )

    pnl_fig.add_hline(
        y=0,
        line_color="#94A3B8",
        line_width=1,
    )

    pnl_fig.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="Transaction",
        yaxis_title="Net P&L (USD)",
        showlegend=False,
        margin=dict(l=20, r=20, t=35, b=20),
    )

    st.plotly_chart(
        pnl_fig,
        width="stretch",
    )


# ============================================================
# 8. RISK MANAGEMENT TAB
# ============================================================

with risk_tab:

    st.subheader(
        "Risk Overview"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Maximum Drawdown",
            f"{summary['maximum_drawdown_pct']:.6f}%",
        )

    with col2:
        st.metric(
            "Transaction Cost",
            f"{summary['cost_bps_per_side']:.0f} bps / side",
        )

    with col3:
        st.metric(
            "Kill Switch Triggered",
            (
                "YES"
                if summary["risk_halted"]
                else "NO"
            ),
        )

    st.divider()

    st.info(
        "Trading results use simulated execution "
        "at historical Bid-open prices. "
        "Actual spreads, short-borrow costs, "
        "slippage and real market liquidity "
        "are not fully modeled."
    )

    # Interactive transaction cost sensitivity
    st.divider()
    st.subheader("Transaction Cost Sensitivity")

    selected_cost = st.slider(
        "Hypothetical Cost (bps per side)",
        min_value=0,
        max_value=40,
        value=10,
        step=5,
    )

    original_cost = float(
        summary["cost_bps_per_side"]
    )

    total_notional = (
        trades["quantity"]
        * (trades["entry_price"] + trades["exit_price"])
    ).sum()

    adjusted_pnl = (
        trades["net_pnl_usd"].sum()
        + (original_cost - selected_cost)
        * total_notional / 10000
    )

    st.metric(
        "Adjusted Net P&L",
        f"${adjusted_pnl:+,.4f}",
    )
    # Break-even transaction cost for the same trades
    gross_pnl = (
        trades["net_pnl_usd"].sum()
        + original_cost * total_notional / 10000
    )

    break_even_bps = (
        gross_pnl * 10000 / total_notional
        if total_notional > 0
        else 0.0
    )

    # ========================================================
    # TRANSACTION COST SENSITIVITY CHART
    # ========================================================

    st.subheader("Net P&L vs Transaction Costs")

    cost_levels = list(range(0, 41, 5))

    pnl_levels = [
        gross_pnl - cost * total_notional / 10000
        for cost in cost_levels
    ]

    cost_fig = go.Figure()

    # Net P&L sensitivity curve
    cost_fig.add_trace(
        go.Scatter(
            x=cost_levels,
            y=pnl_levels,
            mode="lines+markers",
            name="Adjusted Net P&L",
            line=dict(color="#38BDF8", width=3),
            marker=dict(size=8),
        )
    )

    # Zero-profit reference
    cost_fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text="Zero P&L",
    )

    # Break-even cost reference
    cost_fig.add_vline(
        x=break_even_bps,
        line_dash="dot",
        line_color="#FBBF24",
        annotation_text="Break-Even",
    )

    # Currently selected slider scenario
    cost_fig.add_trace(
        go.Scatter(
            x=[selected_cost],
            y=[adjusted_pnl],
            mode="markers",
            name="Selected Scenario",
            marker=dict(
                color="#22C55E",
                size=14,
                symbol="diamond",
            ),
        )
    )

    cost_fig.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="Cost (bps per side)",
        yaxis_title="Net P&L (USD)",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=35, b=20),
    )

    st.plotly_chart(
        cost_fig,
        width="stretch",
    )



    st.caption(
        "Sensitivity analysis using the same 5 historical "
        "transactions and quantities. This is not a new "
        "backtest and does not model actual market spreads."
    )


    # Historical portfolio drawdown
    st.subheader("Historical Drawdown")

    values = equity["equity_usd"].astype(float).reset_index(
        drop=True
    )

    initial = float(summary["initial_capital_usd"])

    peaks = pd.concat(
        [pd.Series([initial]), values],
        ignore_index=True,
    ).cummax().iloc[1:].reset_index(drop=True)

    drawdown_pct = (values / peaks - 1.0) * 100

    drawdown_fig = go.Figure()

    drawdown_fig.add_trace(
        go.Scatter(
            x=equity["datetime_utc"],
            y=drawdown_pct,
            mode="lines",
            fill="tozeroy",
            name="Drawdown",
            line=dict(color="#FB7185", width=2),
        )
    )

    drawdown_fig.update_layout(
        template="plotly_dark",
        height=380,
        xaxis_title="Date (UTC)",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
    )

    st.plotly_chart(
        drawdown_fig,
        width="stretch",
    )

# ============================================================
# MODEL COMPARISON — VALIDATION
# ============================================================

with model_tab:

    st.subheader("Deep Learning Model Comparison")

    st.caption(
        "Validation results for LSTM, GRU and CNN1D "
        "across 5, 15, 30 and 60-minute horizons."
    )

    comparison = load_model_comparison()

    st.metric(
        "Total Experiments",
        len(comparison),
    )

    st.subheader("Validation Performance")

    st.dataframe(
        comparison,
        width="stretch",
        hide_index=True,
        column_config={
            "MAE": st.column_config.NumberColumn(
                format="%.8f"
            ),
            "RMSE": st.column_config.NumberColumn(
                format="%.8f"
            ),
            "Zero Return MAE": st.column_config.NumberColumn(
                format="%.8f"
            ),
            "Zero Return RMSE": st.column_config.NumberColumn(
                format="%.8f"
            ),
            "MAE Improvement (%)": st.column_config.NumberColumn(
                format="%.4f %%"
            ),
        },
    )

    # ========================================================
    # INTERACTIVE MODEL COMPARISON
    # ========================================================

    st.divider()
    st.subheader("Model Performance by Horizon")

    selected_horizon = st.selectbox(
        "Select Prediction Horizon",
        options=[5, 15, 30, 60],
        format_func=lambda x: f"{x} minutes",
    )

    filtered_models = comparison[
        comparison["Horizon (min)"] == selected_horizon
    ].copy()

    filtered_models = filtered_models.sort_values("Model")

    colors = [
        "#22C55E" if value > 0 else "#EF4444"
        for value in filtered_models["MAE Improvement (%)"]
    ]

    model_fig = go.Figure()

    model_fig.add_trace(
        go.Bar(
            x=filtered_models["Model"],
            y=filtered_models["MAE Improvement (%)"],
            marker_color=colors,
            text=[
                f"{v:+.4f}%"
                for v in filtered_models["MAE Improvement (%)"]
            ],
            textposition="outside",
            name="MAE Improvement",
        )
    )

    model_fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#94A3B8",
    )

    model_fig.update_layout(
        template="plotly_dark",
        height=430,
        xaxis_title="Deep Learning Model",
        yaxis_title="MAE Improvement vs Zero Return (%)",
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
    )

    st.plotly_chart(
        model_fig,
        width="stretch",
    )

    # ========================================================
    # MAE / RMSE COMPARISON
    # ========================================================

    st.subheader("Prediction Error Comparison")

    selected_metric = st.radio(
        "Select Evaluation Metric",
        options=["MAE", "RMSE"],
        horizontal=True,
    )

    baseline_column = f"Zero Return {selected_metric}"

    baseline_value = float(
        filtered_models[baseline_column].iloc[0]
    )

    # Best model for the selected horizon and metric
    best_row = filtered_models.loc[
        filtered_models[selected_metric].idxmin()
    ]

    best_model = best_row["Model"]
    best_error = float(best_row[selected_metric])

    improvement_pct = (
        (baseline_value - best_error)
        / baseline_value
        * 100
    )

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            f"Lowest {selected_metric} — Validation",
            best_model,
        )

    with c2:
        st.metric(
            f"Best {selected_metric} Value",
            f"{best_error:.8f}",
            delta=f"{improvement_pct:+.4f}% vs Zero Return",
        )


    error_fig = go.Figure()

    error_fig.add_trace(
        go.Bar(
            x=filtered_models["Model"],
            y=filtered_models[selected_metric],
            marker_color="#38BDF8",
            text=[
                f"{value:.8f}"
                for value in filtered_models[selected_metric]
            ],
            textposition="outside",
            name=selected_metric,
        )
    )

    error_fig.add_hline(
        y=baseline_value,
        line_dash="dash",
        line_color="#FBBF24",
        annotation_text="Zero Return Baseline",
    )

    error_fig.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="Deep Learning Model",
        yaxis_title=selected_metric,
        yaxis=dict(rangemode="tozero"),
        margin=dict(l=20, r=20, t=45, b=20),
    )

    st.plotly_chart(
        error_fig,
        width="stretch",
    )




    st.caption(
        "Lower prediction error is better. "
        "The chart starts at zero, so small differences "
        "may be difficult to see; use the numerical values "
        "for precise comparisons."
    )


    st.caption(
        "Green: lower MAE than Zero Return. "
        "Red: higher MAE than Zero Return. "
        "Differences are small and evaluated on Validation."
    )
    
    # ========================================================
    # BEST MODELS BY PREDICTION HORIZON
    # ========================================================

    st.divider()
    st.subheader("Best Models by Horizon")

    winners = []

    for horizon, group in comparison.groupby(
        "Horizon (min)", sort=True
    ):
        best_mae = group.loc[group["MAE"].idxmin()]
        best_rmse = group.loc[group["RMSE"].idxmin()]

        winners.append({
            "Horizon (min)": int(horizon),
            "Best MAE Model": best_mae["Model"],
            "Best MAE": float(best_mae["MAE"]),
            "Best RMSE Model": best_rmse["Model"],
            "Best RMSE": float(best_rmse["RMSE"]),
        })

    winners_df = pd.DataFrame(winners)

    st.dataframe(
        winners_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Best MAE": st.column_config.NumberColumn(
                format="%.8f"
            ),
            "Best RMSE": st.column_config.NumberColumn(
                format="%.8f"
            ),
        },
    )

    st.caption(
        "Winners are selected separately for each horizon "
        "on Validation. This is not an independent Test "
        "evaluation or proof of trading profitability."
    )

    # ========================================================
    # MAE IMPROVEMENT HEATMAP — VALIDATION
    # ========================================================

    st.divider()
    st.subheader("MAE Improvement Heatmap")

    heatmap_data = comparison.pivot(
        index="Model",
        columns="Horizon (min)",
        values="MAE Improvement (%)",
    )

    heatmap_data = heatmap_data.loc[
        ["LSTM", "GRU", "CNN1D"],
        [5, 15, 30, 60],
    ]

    z_values = heatmap_data.to_numpy()

    max_abs = max(abs(z_values.min()), abs(z_values.max()))

    heatmap_fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=["5 min", "15 min", "30 min", "60 min"],
            y=heatmap_data.index.tolist(),
            colorscale="RdYlGn",
            zmin=-max_abs,
            zmax=max_abs,
            zmid=0,
            text=[
                [f"{v:+.4f}%" for v in row]
                for row in z_values
            ],
            texttemplate="%{text}",
            colorbar=dict(title="MAE gain (%)"),
        )
    )

    heatmap_fig.update_layout(
        template="plotly_dark",
        height=350,
        xaxis_title="Prediction Horizon",
        yaxis_title="Deep Learning Model",
        margin=dict(l=20, r=20, t=20, b=20),
    )

    st.plotly_chart(
        heatmap_fig,
        width="stretch",
    )

    st.caption(
        "Green means lower MAE than Zero Return; "
        "red means higher MAE. Improvements are small, "
        "selected on Validation, and are not proof "
        "of statistical significance or profitability."
    )

    # Export all 12 model comparison results
    model_csv = comparison.to_csv(
        index=False,
        sep=";",
        decimal=",",
    ).encode("utf-8-sig")

    st.download_button(
        label="📥 Download Model Comparison CSV",
        data=model_csv,
        file_name="aurum_ai_model_comparison_validation.csv",
        mime="text/csv",
        key="download_model_comparison",
    )



    st.info(
        "Lower MAE and RMSE indicate smaller prediction "
        "errors on Validation. These results were selected "
        "using Validation and do not establish statistical "
        "significance or trading profitability. "
        "The Test dataset remains unused."
    )




# ============================================================
# 9. FOOTER
# ============================================================

st.divider()

st.caption(
    "AURUM-AI | NYSE V3 | Validation dataset | "
    "Historical simulated performance is not "
    "evidence of future profitability."
)
