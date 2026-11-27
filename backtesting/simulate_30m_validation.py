
from pathlib import Path
import json

import numpy as np
import pandas as pd


# =====================================================
# AURUM-AI — VALIDATION TRADING SIMULATION
# Long-only | One position at a time
# CNN1D vs Logistic Regression
# No real money, no test dataset
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT / "reports"
    / "backtest_30m_validation_inputs.parquet"
)

OUTPUT_DIR = ROOT / "reports" / "backtesting"

SUMMARY_PATH = (
    OUTPUT_DIR / "simulation_30m_validation_summary.json"
)

INITIAL_CAPITAL = 10000.0
POSITION_FRACTION = 0.05

# Hypothetical all-in trading friction per side.
# These are NOT measured Dukascopy spreads.
COST_BPS_PER_SIDE = 10.0
COST_RATE = COST_BPS_PER_SIDE / 10000

MAX_DAILY_LOSS = 0.02
MAX_DRAWDOWN = 0.10

BAR_MS = 5 * 60 * 1000
HORIZON_MS = 30 * 60 * 1000

UP_CLASS = 2


def load_inputs():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    df = pd.read_parquet(INPUT_PATH)

    required = [
        "signal_timestamp_ms",
        "entry_timestamp_ms",
        "exit_timestamp_ms",
        "session_date",
        "segment_id",
        "entry_open_bid",
        "exit_close_bid",
        "cnn_class",
        "logistic_class",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df.empty:
        raise ValueError("Backtest inputs are empty.")

    if df[required].isna().any().any():
        raise ValueError("Missing values in inputs.")

    df = df.sort_values(
        "signal_timestamp_ms"
    ).reset_index(drop=True)

    if df["signal_timestamp_ms"].duplicated().any():
        raise ValueError("Duplicate signal timestamps.")

    if not (
        df["entry_timestamp_ms"]
        - df["signal_timestamp_ms"]
        == BAR_MS
    ).all():
        raise ValueError("Invalid entry alignment.")

    if not (
        df["exit_timestamp_ms"]
        - df["signal_timestamp_ms"]
        == HORIZON_MS
    ).all():
        raise ValueError("Invalid exit alignment.")

    if (
        (df["entry_open_bid"] <= 0).any()
        or (df["exit_close_bid"] <= 0).any()
    ):
        raise ValueError("Invalid market prices.")

    for col in ("cnn_class", "logistic_class"):
        if not df[col].isin([0, 1, 2]).all():
            raise ValueError(f"Invalid class: {col}")

    return df


def run_simulation(df, model_name, class_column):

    capital = INITIAL_CAPITAL
    peak_capital = INITIAL_CAPITAL

    current_day = None
    day_start_capital = INITIAL_CAPITAL

    # New signal must arrive after the exit candle
    # has fully completed.
    next_available_signal = -1

    trades = []

    skipped_busy = 0
    skipped_non_up = 0
    rejected_daily_loss = 0
    rejected_drawdown = 0

    max_realized_drawdown = 0.0

    for row in df.itertuples(index=False):

        signal_ts = int(row.signal_timestamp_ms)
        entry_ts = int(row.entry_timestamp_ms)
        exit_ts = int(row.exit_timestamp_ms)

        session_date = str(row.session_date)

        predicted_class = int(
            getattr(row, class_column)
        )

        if current_day != session_date:
            current_day = session_date
            day_start_capital = capital

        # A signal generated while a previous position
        # is still active cannot open another trade.
        if signal_ts < next_available_signal:
            skipped_busy += 1
            continue

        # Initial long-only strategy:
        # UP -> candidate BUY
        # DOWN / NEUTRAL -> stay in cash
        if predicted_class != UP_CLASS:
            skipped_non_up += 1
            continue

        daily_return = (
            capital / day_start_capital - 1.0
        )

        drawdown_before = (
            1.0 - capital / peak_capital
        )

        # Basic risk controls.
        if daily_return <= -MAX_DAILY_LOSS:
            rejected_daily_loss += 1
            continue

        if drawdown_before >= MAX_DRAWDOWN:
            rejected_drawdown += 1
            continue

        entry_bid = float(row.entry_open_bid)
        exit_bid = float(row.exit_close_bid)

        # Allocate at most 5% of current equity.
        allocation = capital * POSITION_FRACTION

        # Fractional shares are assumed.
        # The entry-side friction is included
        # in the allocated cash.
        quantity = (
            allocation
            / (entry_bid * (1.0 + COST_RATE))
        )

        gross_exit_value = quantity * exit_bid

        # Exit-side hypothetical friction.
        proceeds = (
            gross_exit_value * (1.0 - COST_RATE)
        )

        pnl = proceeds - allocation

        capital_before = capital
        capital += pnl

        peak_capital = max(
            peak_capital, capital
        )

        realized_drawdown = (
            1.0 - capital / peak_capital
        )

        max_realized_drawdown = max(
            max_realized_drawdown,
            realized_drawdown,
        )

        # Conservative time gating:
        # Wait until the exit candle completes
        # before accepting a new signal.
        next_available_signal = (
            exit_ts + BAR_MS
        )

        gross_return_pct = (
            exit_bid / entry_bid - 1.0
        ) * 100.0

        net_trade_return_pct = (
            pnl / allocation
        ) * 100.0

        trades.append({
            "model": model_name,
            "session_date": session_date,
            "signal_timestamp_ms": signal_ts,
            "entry_timestamp_ms": entry_ts,
            "exit_timestamp_ms": exit_ts,
            "action": "BUY_THEN_FIXED_EXIT",
            "quantity": quantity,
            "entry_bid_reference": entry_bid,
            "exit_bid_reference": exit_bid,
            "allocated_capital": allocation,
            "gross_exit_value": gross_exit_value,
            "net_exit_proceeds": proceeds,
            "gross_return_pct": gross_return_pct,
            "net_trade_return_pct": net_trade_return_pct,
            "pnl_usd": pnl,
            "capital_before": capital_before,
            "capital_after": capital,
            "realized_drawdown_pct":
                realized_drawdown * 100.0,
        })

    trade_df = pd.DataFrame(trades)

    if len(trade_df) > 0:

        wins = int(
            (trade_df["pnl_usd"] > 0).sum()
        )

        losses = int(
            (trade_df["pnl_usd"] < 0).sum()
        )

        win_rate = (
            wins / len(trade_df) * 100.0
        )

        average_trade_return = float(
            trade_df["net_trade_return_pct"].mean()
        )

        total_allocations = float(
            trade_df["allocated_capital"].sum()
        )

        total_pnl = float(
            trade_df["pnl_usd"].sum()
        )

    else:

        wins = 0
        losses = 0
        win_rate = 0.0
        average_trade_return = 0.0
        total_allocations = 0.0
        total_pnl = 0.0

    net_return_pct = (
        capital / INITIAL_CAPITAL - 1.0
    ) * 100.0

    # Check the cash accounting.
    if not np.isclose(
        INITIAL_CAPITAL + total_pnl,
        capital,
        atol=1e-7,
        rtol=0,
    ):
        raise ValueError("Portfolio accounting mismatch.")

    summary = {
        "model": model_name,
        "strategy": "Long-only UP, fixed 30m exit",
        "initial_capital_usd": INITIAL_CAPITAL,
        "final_capital_usd": float(capital),
        "net_profit_usd": float(
            capital - INITIAL_CAPITAL
        ),
        "net_return_pct": float(net_return_pct),
        "trades": len(trade_df),
        "winning_trades": wins,
        "losing_trades": losses,
        "win_rate_pct": float(win_rate),
        "average_trade_return_pct":
            average_trade_return,
        "total_allocated_usd": total_allocations,
        "max_realized_drawdown_pct": float(
            max_realized_drawdown * 100.0
        ),
        "skipped_busy": skipped_busy,
        "skipped_non_up": skipped_non_up,
        "risk_rejected_daily_loss":
            rejected_daily_loss,
        "risk_rejected_drawdown":
            rejected_drawdown,
        "position_fraction": POSITION_FRACTION,
        "cost_bps_per_side": COST_BPS_PER_SIDE,
    }

    return trade_df, summary


def main():

    print(
        "\n=== AURUM-AI | TRADING SIMULATION ==="
    )

    output_files = [
        SUMMARY_PATH,
        OUTPUT_DIR / "cnn1d_trades_validation.csv",
        OUTPUT_DIR / "logistic_trades_validation.csv",
    ]

    for path in output_files:
        if path.exists():
            raise FileExistsError(
                f"Existing artifact preserved: {path}"
            )

    df = load_inputs()

    print("\n=== INPUT VALIDATION ===")
    print(f"Signals available: {len(df)}")
    print(
        f"Trading days: "
        f"{df['session_date'].nunique()}"
    )

    print(f"Initial capital: {INITIAL_CAPITAL:.2f} USD")
    print(
        f"Position size: "
        f"{POSITION_FRACTION * 100:.1f}%"
    )
    print(
        f"Assumed cost: "
        f"{COST_BPS_PER_SIDE:.1f} bps per side"
    )

    OUTPUT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    results = {}

    models = [
        (
            "CNN1D",
            "cnn_class",
            OUTPUT_DIR / "cnn1d_trades_validation.csv",
        ),
        (
            "Logistic Regression",
            "logistic_class",
            OUTPUT_DIR / "logistic_trades_validation.csv",
        ),
    ]

    for model_name, column, output_path in models:

        trades, summary = run_simulation(
            df,
            model_name,
            column,
        )

        trades.to_csv(
            output_path,
            index=False,
        )

        key = (
            "cnn1d"
            if model_name == "CNN1D"
            else "logistic_regression"
        )

        results[key] = summary

        print(f"\n=== {model_name.upper()} ===")

        print(f"Executed trades: {summary['trades']}")

        print(
            f"Final capital: "
            f"{summary['final_capital_usd']:.2f} USD"
        )

        print(
            f"Net profit: "
            f"{summary['net_profit_usd']:+.2f} USD"
        )

        print(
            f"Net return: "
            f"{summary['net_return_pct']:+.4f}%"
        )

        print(
            f"Win rate: "
            f"{summary['win_rate_pct']:.2f}%"
        )

        print(
            f"Max realized drawdown: "
            f"{summary['max_realized_drawdown_pct']:.4f}%"
        )

        print(
            f"Risk rejected (daily loss): "
            f"{summary['risk_rejected_daily_loss']}"
        )

        print(
            f"Risk rejected (drawdown): "
            f"{summary['risk_rejected_drawdown']}"
        )

    report = {
        "project": "AURUM-AI",
        "experiment": "30m validation trading simulation",
        "type": "historical virtual simulation",
        "results": results,
        "price_source": "Dukascopy XOMUSUSD Bid OHLC",
        "entry": "Next 5m candle Bid open reference",
        "exit": "30m target candle Bid close reference",
        "capital_type": "simulated",
        "real_money": False,
        "test_dataset_used": False,
        "genuine_walk_forward": False,
        "existing_agents_integrated": False,
        "limitations": [
            "Models selected using validation data.",
            "Bid-only data; real Ask/spread unavailable.",
            "Cost is a hypothetical proxy.",
            "Idealized next-open and future-close fills.",
            "No intrabar stops or intratrade risk monitoring.",
            "Max drawdown measured on realized equity only.",
            "Fractional shares assumed.",
            "No short selling.",
            "No dividends or financing modeled.",
            "No benchmark or Sharpe ratio computed yet.",
            "Risk checks are local; existing RiskAgent not yet integrated.",
        ],
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(report, file, indent=2)

    print("\n=== FILES SAVED ===")
    for path in output_files:
        print(path)

    print("\nNo real trades executed.")
    print("Test dataset was NOT used.")
    print("This is NOT a walk-forward test.")
    print("\nTRADING SIMULATION COMPLETED")


if __name__ == "__main__":
    main()
