
from pathlib import Path
import json

import numpy as np

from backtesting.simulate_multi_agent_30m import load_inputs
from backtesting.simulate_multi_agent_30m_v2 import simulate_model


# =====================================================
# AURUM-AI — B.16.2
# MULTI-MODEL CONSENSUS BACKTEST
#
# CNN1D + Logistic Regression
# Dataset: XOM (unchanged)
# Split: VALIDATION ONLY
#
# BOTH UP -> BUY
# Otherwise -> HOLD
#
# Scheduled position exit after the 30m horizon.
# Risk Agent + persistent Kill Switch provided by V2.
#
# No real trading.
# No independent test evaluation.
# =====================================================


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "backtesting"

DECISIONS_PATH = (
    OUTPUT_DIR / "consensus_30m_validation_decisions.csv"
)

TRANSACTIONS_PATH = (
    OUTPUT_DIR / "consensus_30m_validation_transactions.csv"
)

REPORT_PATH = (
    OUTPUT_DIR / "consensus_30m_validation_summary.json"
)

V2_REPORT_PATH = (
    OUTPUT_DIR / "multi_agent_30m_validation_v2_summary.json"
)


def main():

    print("\n=== AURUM-AI | MULTI-MODEL CONSENSUS ===")

    # Preserve previous experiments.
    for path in (
        DECISIONS_PATH,
        TRANSACTIONS_PATH,
        REPORT_PATH,
    ):
        if path.exists():
            raise FileExistsError(
                f"Existing experiment preserved: {path}"
            )

    # Load the same validated historical windows.
    signals, prices = load_inputs()

    signals = signals.copy()

    # =================================================
    # 1. CONSENSUS STRATEGY
    # =================================================

    both_up = (
        signals["cnn_class"].eq(2)
        & signals["logistic_class"].eq(2)
    )

    # ClassificationStrategyAdapter:
    # UP = 2
    # NEUTRAL = 1
    #
    # No DOWN-generated early exit:
    # open positions use the fixed scheduled exit.

    signals["consensus_class"] = np.where(
        both_up,
        2,
        1,
    ).astype(np.int64)

    consensus_count = int(both_up.sum())

    print("\n=== CONSENSUS SIGNALS ===")
    print(f"Total valid signals: {len(signals)}")
    print(f"Both models predict UP: {consensus_count}")
    print(
        "Other signals (HOLD): "
        f"{len(signals) - consensus_count}"
    )

    # =================================================
    # 2. SEQUENTIAL PORTFOLIO SIMULATION
    # =================================================

    decisions, transactions, summary = simulate_model(
        model_name="CNN1D + Logistic Consensus",
        class_column="consensus_class",
        candidates=signals,
        prices=prices,
    )

    # =================================================
    # 3. COMPARE WITH PREVIOUS V2 EXPERIMENTS
    # =================================================

    if not V2_REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Missing V2 comparison report: {V2_REPORT_PATH}"
        )

    previous_report = json.loads(
        V2_REPORT_PATH.read_text(encoding="utf-8")
    )

    cnn = previous_report["results"]["cnn"]
    logistic = previous_report["results"]["logistic"]

    print("\n=== PORTFOLIO COMPARISON ===")

    comparison = {
        "CNN1D V2": cnn,
        "Logistic V2": logistic,
        "Consensus V2": summary,
    }

    for name, result in comparison.items():

        print(f"\n{name}")
        print(
            "Net return: "
            f"{result['net_return_pct']:+.4f}%"
        )
        print(
            "Completed trades: "
            f"{result['completed_trades']}"
        )
        print(
            "Max drawdown: "
            f"{result['max_observed_drawdown_pct']:.4f}%"
        )
        print(
            "Final capital: "
            f"{result['final_equity_usd']:.2f} USD"
        )

    # =================================================
    # 4. REPORT
    # =================================================

    report = {
        "project": "AURUM-AI",
        "stage": "B.16.2",
        "dataset": "XOM",
        "split": "validation",
        "strategy": "CNN1D and Logistic both predict UP",
        "forecast_horizon_minutes": 30,
        "total_valid_signals": int(len(signals)),
        "consensus_up_signals": consensus_count,
        "execution_method": "Historical Multi-Agent V2",
        "consensus_result": summary,
        "reference_results": {
            "cnn1d_v2": cnn,
            "logistic_v2": logistic,
        },
        "test_dataset_used": False,
        "real_orders_executed": False,
        "genuine_walk_forward": False,
        "limitations": [
            "Exploratory validation experiment.",
            "Consensus rule examined after validation results.",
            "Models previously selected on validation.",
            "No independent out-of-sample confirmation.",
            "Bid-only historical price data.",
            "Idealized candle-open fills.",
            "Hypothetical transaction costs.",
            "No intrabar stop-loss execution.",
            "One long position at a time.",
            "No early model-triggered SELL.",
            "Kill Switch prevents new BUY orders.",
        ],
    }

    # =================================================
    # 5. SAVE NEW ARTIFACTS
    # =================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    decisions.to_csv(
        DECISIONS_PATH,
        index=False,
    )

    transactions.to_csv(
        TRANSACTIONS_PATH,
        index=False,
    )

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== SAVED FILES ===")
    print(DECISIONS_PATH)
    print(TRANSACTIONS_PATH)
    print(REPORT_PATH)

    print("\nOriginal XOM dataset preserved.")
    print("Previous experiments preserved.")
    print("Test dataset NOT used.")
    print("No real trading.")

    print(
        "\nB.16.2 CONSENSUS BACKTEST COMPLETED"
    )


if __name__ == "__main__":
    main()

