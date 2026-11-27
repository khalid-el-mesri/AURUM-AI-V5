
from pathlib import Path
import json

import numpy as np
import pandas as pd

from backtesting.simulate_multi_agent_30m import load_inputs
from backtesting.simulate_multi_agent_30m_v2 import simulate_model


# ============================================================
# AURUM-AI — B.17.2
# TREND-FOLLOWING VS MULTI-MODEL CONSENSUS
#
# Dataset: XOM only
# Evaluation: VALIDATION only
# Execution: Historical Multi-Agent V2
#
# All strategies:
#   - Long-only
#   - Fixed 30m horizon
#   - Same simulated entry/exit prices
#   - Same hypothetical costs (10 bps/side)
#   - Same Risk Agent and persistent Kill Switch
#
# No real trading. No test evaluation.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = (
    ROOT / "data" / "03_features"
    / "xom_features_v2.parquet"
)

OUTPUT_DIR = (
    ROOT / "reports" / "backtesting"
    / "trend_consensus_30m"
)

SUMMARY_PATH = OUTPUT_DIR / "summary.json"

REFERENCE_PATH = (
    ROOT / "reports" / "backtesting"
    / "fixed_exit_30m_comparison"
    / "comparison_summary.json"
)


def load_signal_features(signals):

    print("\n=== FEATURE ALIGNMENT AUDIT ===")

    features = pd.read_parquet(
        FEATURES_PATH,
        columns=[
            "timestamp",
            "segment_id",
            "bar_in_segment",
            "close",
            "sma_5",
            "ema_10",
            "volatility_10",
        ],
    )

    if features["timestamp"].duplicated().any():
        raise ValueError("Duplicate feature timestamps.")

    # Features must correspond to the same candle
    # that generated each historical prediction.
    aligned = features.set_index(
        "timestamp",
        verify_integrity=True,
    ).reindex(
        signals["signal_timestamp_ms"].to_numpy()
    )

    if aligned["segment_id"].isna().any():
        raise RuntimeError(
            "Missing feature rows for validation signals."
        )

    if not np.array_equal(
        aligned["segment_id"].to_numpy(),
        signals["segment_id"].to_numpy(),
    ):
        raise RuntimeError(
            "Segment mismatch between features and signals."
        )

    # Require mature indicators; do not replace
    # rolling warm-up NaNs with zero.
    usable = (
        aligned["close"].notna()
        & aligned["sma_5"].notna()
        & aligned["ema_10"].notna()
        & aligned["volatility_10"].notna()
        & aligned["bar_in_segment"].ge(10)
    ).to_numpy()

    close = aligned["close"].to_numpy(dtype=float)
    sma = aligned["sma_5"].to_numpy(dtype=float)
    ema = aligned["ema_10"].to_numpy(dtype=float)

    trend_up = (
        usable
        & np.isfinite(close)
        & np.isfinite(sma)
        & np.isfinite(ema)
        & (close > sma)
        & (sma > ema)
    )

    print(f"Validation signals: {len(signals)}")
    print(f"Usable feature rows: {int(usable.sum())}")
    print(f"Excluded feature rows: {int((~usable).sum())}")
    print(f"Technical UP signals: {int(trend_up.sum())}")

    print(
        "Feature timestamp: signal candle start; "
        "its close is known before delayed entry."
    )

    return trend_up


def main():

    print(
        "\n=== AURUM-AI | B.17.2 TREND COMPARISON ==="
    )

    signals, prices = load_inputs()
    signals = signals.copy()

    trend_up = load_signal_features(signals)

    cnn_up = signals["cnn_class"].eq(2).to_numpy()
    logistic_up = signals["logistic_class"].eq(2).to_numpy()

    consensus_up = cnn_up & logistic_up
    combined_up = consensus_up & trend_up

    # Adapter conventions:
    # UP=2 -> potential BUY
    # NEUTRAL=1 -> HOLD
    #
    # No DOWN class -> no model-triggered
    # early liquidation; fixed exits only.

    signals["trend_class"] = np.where(
        trend_up, 2, 1
    ).astype(np.int64)

    signals["consensus_class"] = np.where(
        consensus_up, 2, 1
    ).astype(np.int64)

    signals["combined_class"] = np.where(
        combined_up, 2, 1
    ).astype(np.int64)

    strategies = [
        ("Technical Trend", "trend_class", "trend"),
        (
            "CNN1D + Logistic",
            "consensus_class",
            "consensus",
        ),
        (
            "Consensus + Trend",
            "combined_class",
            "combined",
        ),
    ]

    # Never overwrite existing experiment results.
    outputs = {"summary": SUMMARY_PATH}

    for _, _, key in strategies:
        outputs[f"{key}_decisions"] = (
            OUTPUT_DIR / f"{key}_decisions.csv"
        )
        outputs[f"{key}_transactions"] = (
            OUTPUT_DIR / f"{key}_transactions.csv"
        )

    for path in outputs.values():
        if path.exists():
            raise FileExistsError(
                f"Existing result preserved: {path}"
            )

    results = {}
    frames = {}

    for name, class_column, key in strategies:

        potential_buys = int(
            signals[class_column].eq(2).sum()
        )

        print("\n" + "=" * 55)
        print(f"STRATEGY: {name}")
        print(f"Potential BUY signals: {potential_buys}")

        decisions, transactions, summary = simulate_model(
            model_name=name,
            class_column=class_column,
            candidates=signals,
            prices=prices,
        )

        if summary["early_down_exits"] != 0:
            raise RuntimeError("Unexpected early exit.")

        if (
            summary["scheduled_exits"]
            != summary["completed_trades"]
        ):
            raise RuntimeError(
                "Trade / scheduled-exit mismatch."
            )

        summary["potential_buy_signals"] = potential_buys

        results[key] = summary
        frames[f"{key}_decisions"] = decisions
        frames[f"{key}_transactions"] = transactions

    # Verify that the consensus reference reproduces
    # the existing controlled B.16.3 experiment.
    if REFERENCE_PATH.exists():

        reference = json.loads(
            REFERENCE_PATH.read_text(encoding="utf-8")
        )

        expected = reference["results"]["consensus"]

        for metric in (
            "completed_trades",
            "net_return_pct",
            "max_observed_drawdown_pct",
        ):
            if not np.isclose(
                results["consensus"][metric],
                expected[metric],
                rtol=0,
                atol=1e-7,
            ):
                raise RuntimeError(
                    f"Consensus reference mismatch: {metric}"
                )

        print("\nConsensus reference reproduction: PASSED")

    print("\n=== B.17.2 FAIR PORTFOLIO COMPARISON ===")

    print(
        f"{'Strategy':<19}"
        f"{'Trades':>8}"
        f"{'Return %':>13}"
        f"{'Drawdown %':>14}"
    )

    print("-" * 54)

    for key, label in (
        ("trend", "Trend"),
        ("consensus", "Consensus"),
        ("combined", "Consensus + Trend"),
    ):

        result = results[key]

        print(
            f"{label:<19}"
            f"{result['completed_trades']:>8}"
            f"{result['net_return_pct']:>+13.4f}"
            f"{result['max_observed_drawdown_pct']:>14.4f}"
        )

    print("\n=== COMBINED VS CONSENSUS ===")

    difference = (
        results["combined"]["net_return_pct"]
        - results["consensus"]["net_return_pct"]
    )

    print(
        "Return difference (percentage points): "
        f"{difference:+.4f}"
    )

    report = {
        "project": "AURUM-AI",
        "stage": "B.17.2",
        "dataset": "XOM",
        "evaluation_split": "validation",
        "valid_execution_signals": int(len(signals)),
        "technical_rule": "Close > SMA5 > EMA10",
        "consensus_rule": "CNN1D UP and Logistic UP",
        "combined_rule": "Consensus UP and Technical Trend UP",
        "indicator_warmup_required": True,
        "forecast_horizon_minutes": 30,
        "cost_bps_per_side": 10.0,
        "execution_engine": "Historical Multi-Agent V2",
        "results": results,
        "combined_minus_consensus_pp": float(difference),
        "real_trading": False,
        "test_dataset_used": False,
        "genuine_walk_forward": False,
        "limitations": [
            "Exploratory validation comparison.",
            "Strategies were examined after prior validation results.",
            "Not an independent profitability demonstration.",
            "The technical trend rule uses signal-close information.",
            "Execution is delayed until a later bar open.",
            "Fixed exits; no early DOWN liquidation.",
            "Hypothetical 10 bps transaction costs per side.",
            "Bid-only data; real Ask/spread unavailable.",
            "No intrabar execution or stop-loss.",
            "Risk controls cannot guarantee a fixed loss cap.",
        ],
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for key, frame in frames.items():
        frame.to_csv(outputs[key], index=False)

    SUMMARY_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== SAVED RESULTS ===")
    print(SUMMARY_PATH)

    print("\nOriginal XOM data preserved.")
    print("Previous reports preserved.")
    print("No real orders executed.")
    print("Test dataset NOT used.")

    print("\nB.17.2 TREND COMPARISON COMPLETED")


if __name__ == "__main__":
    main()
