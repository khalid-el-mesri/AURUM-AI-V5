
from pathlib import Path
import json

import numpy as np

from backtesting.simulate_multi_agent_30m import load_inputs
from backtesting.simulate_multi_agent_30m_v2 import simulate_model


# ============================================================
# AURUM-AI — B.16.3
# CONTROLLED COMPARISON OF THREE STRATEGIES
#
# Same dataset: XOM
# Same split: Validation
# Same horizon: 30 minutes
# Same entry and exit execution rules
# Same simulated trading costs and risk limits
#
# Difference: BUY selection rule only
#
# No early DOWN exits in any strategy.
# No real trading.
# No test dataset.
# ============================================================


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    ROOT
    / "reports"
    / "backtesting"
    / "fixed_exit_30m_comparison"
)

REPORT_PATH = OUTPUT_DIR / "comparison_summary.json"


def main():

    print(
        "\n=== AURUM-AI | B.16.3 CONTROLLED COMPARISON ==="
    )

    # --------------------------------------------------------
    # 1. Load identical validation market data.
    # --------------------------------------------------------

    signals, prices = load_inputs()

    signals = signals.copy()

    cnn_up = signals["cnn_class"].eq(2)
    logistic_up = signals["logistic_class"].eq(2)

    # The classification adapter interprets:
    # 2 = UP -> potentially BUY
    # 1 = NEUTRAL -> HOLD
    #
    # Class 0 is intentionally not used here.
    # Therefore no model-driven early SELL occurs.

    signals["cnn_fixed_class"] = np.where(
        cnn_up, 2, 1
    )

    signals["logistic_fixed_class"] = np.where(
        logistic_up, 2, 1
    )

    signals["consensus_fixed_class"] = np.where(
        cnn_up & logistic_up, 2, 1
    )

    strategies = [
        ("CNN1D", "cnn_fixed_class", "cnn"),
        (
            "Logistic Regression",
            "logistic_fixed_class",
            "logistic",
        ),
        (
            "CNN1D + Logistic Consensus",
            "consensus_fixed_class",
            "consensus",
        ),
    ]

    # --------------------------------------------------------
    # 2. Define separate output artifacts.
    # --------------------------------------------------------

    files = {"summary": REPORT_PATH}

    for _, _, key in strategies:

        files[f"{key}_decisions"] = (
            OUTPUT_DIR / f"{key}_decisions.csv"
        )

        files[f"{key}_transactions"] = (
            OUTPUT_DIR / f"{key}_transactions.csv"
        )

    # Preserve any previous results.

    for path in files.values():

        if path.exists():
            raise FileExistsError(
                f"Existing file preserved: {path}"
            )

    # --------------------------------------------------------
    # 3. Run identical execution rules.
    # --------------------------------------------------------

    results = {}
    outputs = {}

    for name, class_column, key in strategies:

        print("\n" + "=" * 54)
        print(f"STRATEGY: {name}")
        print("=" * 54)

        selected_signals = int(
            signals[class_column].eq(2).sum()
        )

        print(f"Potential BUY signals: {selected_signals}")

        decisions, transactions, summary = simulate_model(
            model_name=name,
            class_column=class_column,
            candidates=signals,
            prices=prices,
        )

        # All three strategies must use fixed exits.
        if summary["early_down_exits"] != 0:
            raise RuntimeError(
                f"Unexpected early exit for {name}."
            )

        if summary["scheduled_exits"] != (
            summary["completed_trades"]
        ):
            raise RuntimeError(
                f"Scheduled-exit mismatch for {name}."
            )

        summary["potential_buy_signals"] = (
            selected_signals
        )

        results[key] = summary

        outputs[f"{key}_decisions"] = decisions
        outputs[f"{key}_transactions"] = transactions

    # --------------------------------------------------------
    # 4. Compare results.
    # --------------------------------------------------------

    print("\n=== FAIR PORTFOLIO COMPARISON ===")

    print(
        f"{'Strategy':<15}"
        f"{'Trades':>9}"
        f"{'Return %':>13}"
        f"{'Drawdown %':>14}"
    )

    print("-" * 54)

    for key, name in [
        ("cnn", "CNN1D"),
        ("logistic", "Logistic"),
        ("consensus", "Consensus"),
    ]:

        r = results[key]

        print(
            f"{name:<15}"
            f"{r['completed_trades']:>9}"
            f"{r['net_return_pct']:>+13.4f}"
            f"{r['max_observed_drawdown_pct']:>14.4f}"
        )

    consensus_return = results[
        "consensus"
    ]["net_return_pct"]

    cnn_return = results[
        "cnn"
    ]["net_return_pct"]

    logistic_return = results[
        "logistic"
    ]["net_return_pct"]

    print("\n=== CONSENSUS DIFFERENCES ===")

    print(
        "Consensus minus CNN1D (percentage points): "
        f"{consensus_return - cnn_return:+.4f}"
    )

    print(
        "Consensus minus Logistic (percentage points): "
        f"{consensus_return - logistic_return:+.4f}"
    )

    print("\nNo-trade cash baseline return: 0.0000%")
    print("Cash baseline excludes interest and inflation.")

    # --------------------------------------------------------
    # 5. Save the experiment.
    # --------------------------------------------------------

    report = {
        "project": "AURUM-AI",
        "stage": "B.16.3",
        "dataset": "XOM",
        "evaluation_split": "validation",
        "horizon_minutes": 30,
        "valid_signals": int(len(signals)),
        "comparison_type": "Controlled fixed-exit comparison",
        "shared_rules": {
            "entry": "Signal + 10 minutes, Bid open",
            "exit": "Signal + 35 minutes, Bid open",
            "early_down_exit": False,
            "allocation_fraction": 0.05,
            "cost_bps_per_side": 10.0,
            "risk_management": "RiskAgent + persistent Kill Switch",
            "long_only": True,
            "max_concurrent_positions": 1,
        },
        "results": results,
        "no_trade_cash_baseline_return_pct": 0.0,
        "real_trading": False,
        "test_dataset_used": False,
        "genuine_walk_forward": False,
        "limitations": [
            "Models were previously selected on validation.",
            "Consensus rule was explored on validation.",
            "Not independent out-of-sample evidence.",
            "Trading costs are hypothetical.",
            "Market data contain Bid OHLC only.",
            "No real Bid/Ask spread observations.",
            "Candle-open fills are idealized.",
            "No intrabar risk execution.",
            "Fixed exits do not guarantee profits.",
        ],
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for key, frame in outputs.items():

        frame.to_csv(
            files[key],
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

    print("\n=== EXPERIMENT SAVED ===")
    print(REPORT_PATH)

    print("\nOriginal XOM dataset preserved.")
    print("Previous V1 and V2 reports preserved.")
    print("No test dataset used.")
    print("No real trading.")

    print(
        "\nB.16.3 CONTROLLED COMPARISON COMPLETED"
    )


if __name__ == "__main__":
    main()
