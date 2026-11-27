
from pathlib import Path
import json

import numpy as np
import pandas as pd

from backtesting.simulate_multi_agent_30m import load_inputs
from backtesting.simulate_multi_agent_30m_v2 import simulate_model


# ============================================================
# AURUM-AI — B.17.6
# CONTROLLED PORTFOLIO BACKTEST: 30M VS 60M
#
# Dataset: XOM only
# Split: VALIDATION only
# Strategy: CNN1D UP + Logistic UP + Technical Trend UP
#
# Same BUY signals, risk rules, costs and capital.
# Only the scheduled exit horizon changes.
#
# No model retraining.
# No real trades.
# No test data.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

BAR_MS = 5 * 60 * 1000

FEATURES_PATH = (
    ROOT / "data" / "03_features" / "xom_features_v2.parquet"
)

OUTPUT_DIR = (
    ROOT / "reports" / "backtesting"
    / "combined_holding_30m_60m"
)

SUMMARY_PATH = OUTPUT_DIR / "comparison_summary.json"


def prepare_common_signals(signals, prices):

    print("\n=== PREPARING COMMON BUY SIGNALS ===")

    features = pd.read_parquet(
        FEATURES_PATH,
        columns=[
            "timestamp",
            "session_date",
            "segment_id",
            "bar_in_segment",
            "close",
            "sma_5",
            "ema_10",
            "volatility_10",
        ],
    ).set_index("timestamp", verify_integrity=True)

    if signals["signal_timestamp_ms"].duplicated().any():
        raise RuntimeError("Duplicate prediction timestamps.")

    aligned = features.reindex(
        signals["signal_timestamp_ms"].to_numpy()
    )

    if aligned["segment_id"].isna().any():
        raise RuntimeError("Missing signal features.")

    same_segment = (
        aligned["segment_id"].to_numpy()
        == signals["segment_id"].to_numpy()
    )

    trend_up = (
        aligned["close"].gt(aligned["sma_5"])
        & aligned["sma_5"].gt(aligned["ema_10"])
        & aligned["volatility_10"].notna()
        & aligned["bar_in_segment"].ge(10)
    ).to_numpy()

    combined_up = (
        signals["cnn_class"].eq(2).to_numpy()
        & signals["logistic_class"].eq(2).to_numpy()
        & trend_up
        & same_segment
    )

    selected = signals.loc[combined_up].copy()

    print(f"Original combined signals: {len(selected)}")

    if len(selected) != 157:
        raise RuntimeError(
            "Combined signal count differs from B.17.5."
        )

    timestamps = selected[
        "signal_timestamp_ms"
    ].to_numpy(dtype=np.int64)

    entry_timestamps = selected[
        "entry_execution_timestamp_ms"
    ].to_numpy(dtype=np.int64)

    original_exit_timestamps = selected[
        "scheduled_exit_timestamp_ms"
    ].to_numpy(dtype=np.int64)

    if not np.array_equal(
        entry_timestamps,
        timestamps + 2 * BAR_MS,
    ):
        raise RuntimeError(
            "Unexpected delayed-entry convention."
        )

    if not np.array_equal(
        original_exit_timestamps,
        timestamps + 7 * BAR_MS,
    ):
        raise RuntimeError(
            "Unexpected 30m scheduled-exit convention."
        )

    # Price data are already scoped to validation
    # by the existing load_inputs() function.
    market = prices.set_index(
        "timestamp",
        verify_integrity=True,
    )

    base = features.reindex(timestamps)

    segment_ids = selected["segment_id"].to_numpy()

    session_dates = (
        selected["session_date"]
        .astype(str)
        .to_numpy()
    )

    base_bar_positions = (
        base["bar_in_segment"].to_numpy(dtype=float)
    )

    valid = (
        np.isfinite(base_bar_positions)
        & (
            base["segment_id"].to_numpy()
            == segment_ids
        )
    )

    # A comparable signal must have all required
    # candles in its original uninterrupted segment.
    #
    # Signal t
    # Entry  t + 10 minutes
    # 30m exit t + 35 minutes
    # 60m exit t + 65 minutes

    for offset in (0, 2, 7, 13):

        required_ts = timestamps + offset * BAR_MS

        feature_rows = features.reindex(required_ts)

        market_rows = market.reindex(required_ts)

        valid &= (
            feature_rows["segment_id"].to_numpy()
            == segment_ids
        )

        valid &= (
            feature_rows["bar_in_segment"]
            .to_numpy(dtype=float)
            == base_bar_positions + offset
        )

        valid &= (
            feature_rows["session_date"]
            .astype(str)
            .to_numpy()
            == session_dates
        )

        valid &= (
            market_rows["session_date"]
            .astype(str)
            .to_numpy()
            == session_dates
        )

        market_open = market_rows[
            "open"
        ].to_numpy(dtype=float)

        valid &= (
            np.isfinite(market_open)
            & (market_open > 0)
        )

    selected = selected.loc[valid].copy()

    print(f"Common eligible signals: {len(selected)}")
    print(f"Excluded signals: {int((~valid).sum())}")

    if len(selected) != 132:
        raise RuntimeError(
            "Common signal count differs from B.17.5. "
            "Check validation price coverage."
        )

    selected["combined_class"] = 2

    selected = selected.sort_values(
        "entry_execution_timestamp_ms"
    ).reset_index(drop=True)

    return selected


def run_horizon(signals, prices, horizon_minutes):

    candidates = signals.copy()

    if horizon_minutes == 30:
        exit_offset_bars = 7
    elif horizon_minutes == 60:
        exit_offset_bars = 13
    else:
        raise ValueError("Unsupported horizon.")

    candidates["scheduled_exit_timestamp_ms"] = (
        candidates["signal_timestamp_ms"].astype(np.int64)
        + exit_offset_bars * BAR_MS
    )

    # Combined class is always UP for selected events.
    # No model-driven early DOWN exit.
    return simulate_model(
        model_name=f"Consensus + Trend {horizon_minutes}m",
        class_column="combined_class",
        candidates=candidates,
        prices=prices,
    )


def main():

    print(
        "\n=== AURUM-AI | B.17.6 PORTFOLIO HOLDING AUDIT ==="
    )

    targets = {
        "summary": SUMMARY_PATH,
        "30m_decisions": OUTPUT_DIR / "30m_decisions.csv",
        "30m_transactions": OUTPUT_DIR / "30m_transactions.csv",
        "60m_decisions": OUTPUT_DIR / "60m_decisions.csv",
        "60m_transactions": OUTPUT_DIR / "60m_transactions.csv",
    }

    # Never replace an earlier result.
    for path in targets.values():
        if path.exists():
            raise FileExistsError(
                f"Existing experiment preserved: {path}"
            )

    signals, prices = load_inputs()

    common_signals = prepare_common_signals(
        signals,
        prices,
    )

    results = {}
    frames = {}

    for horizon in (30, 60):

        print("\n" + "=" * 56)
        print(f"PORTFOLIO HORIZON: {horizon} MIN")
        print("=" * 56)

        decisions, transactions, summary = run_horizon(
            common_signals,
            prices,
            horizon,
        )

        if summary["early_down_exits"] != 0:
            raise RuntimeError(
                "Unexpected early DOWN exit."
            )

        if (
            summary["scheduled_exits"]
            != summary["completed_trades"]
        ):
            raise RuntimeError(
                "Scheduled exit accounting mismatch."
            )

        key = f"{horizon}m"

        results[key] = summary
        frames[f"{key}_decisions"] = decisions
        frames[f"{key}_transactions"] = transactions

    print("\n=== B.17.6 FAIR PORTFOLIO COMPARISON ===")

    print(
        f"{'Indicator':<26}"
        f"{'30m':>15}"
        f"{'60m':>15}"
    )

    print("-" * 56)

    metrics = [
        ("completed_trades", "Completed trades"),
        ("win_rate_pct", "Win rate %"),
        ("net_return_pct", "Net return %"),
        ("max_observed_drawdown_pct", "Max drawdown %"),
        ("total_fees_usd", "Trading fees USD"),
        ("final_equity_usd", "Final capital USD"),
    ]

    for metric, label in metrics:

        value_30 = results["30m"][metric]
        value_60 = results["60m"][metric]

        print(
            f"{label:<26}"
            f"{value_30:>15.4f}"
            f"{value_60:>15.4f}"
        )

    improvement = (
        results["60m"]["net_return_pct"]
        - results["30m"]["net_return_pct"]
    )

    print("\n=== RETURN DIFFERENCE ===")

    print(
        "60m minus 30m (percentage points): "
        f"{improvement:+.4f}"
    )

    print("\n=== SYSTEM STATES ===")

    for horizon in ("30m", "60m"):
        print(
            f"{horizon}: "
            f"{results[horizon]['final_system_state']}"
        )

    report = {
        "project": "AURUM-AI",
        "stage": "B.17.6",
        "dataset": "XOM",
        "evaluation_split": "validation",
        "strategy": (
            "CNN1D UP and Logistic UP "
            "and Close > SMA5 > EMA10"
        ),
        "common_signals": int(len(common_signals)),
        "entry_delay_minutes": 10,
        "exit_offsets_from_signal_minutes": {
            "30m": 35,
            "60m": 65,
        },
        "actual_holding_minutes": {
            "30m": 25,
            "60m": 55,
        },
        "cost_bps_per_side": 10.0,
        "execution_engine": "Historical Multi-Agent V2",
        "risk_engine": "Risk Agent + Kill Switch",
        "results": results,
        "return_difference_60m_minus_30m_pp": float(
            improvement
        ),
        "previous_experiments_preserved": True,
        "test_dataset_used": False,
        "real_trading": False,
        "genuine_walk_forward": False,
        "limitations": [
            "Exploratory validation experiment.",
            "30m classification models reused for 60m exits.",
            "Common cohort selected using historical future-bar availability.",
            "Cohort eligibility is not an ex-ante tradable rule.",
            "Model and strategies previously explored on validation.",
            "Not independent out-of-sample evidence.",
            "Trading fees are hypothetical.",
            "Bid-only historical data.",
            "Idealized delayed candle-open execution.",
            "No intrabar execution or slippage modeling.",
        ],
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for key, frame in frames.items():
        frame.to_csv(
            targets[key],
            index=False,
        )

    SUMMARY_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== RESULTS SAVED ===")
    print(SUMMARY_PATH)

    print("\nOriginal XOM dataset preserved.")
    print("Existing models and reports preserved.")
    print("Test dataset NOT used.")
    print("No real orders executed.")

    print(
        "\nB.17.6 PORTFOLIO HOLDING AUDIT COMPLETED"
    )


if __name__ == "__main__":
    main()
