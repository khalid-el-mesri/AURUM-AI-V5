
import numpy as np
import pandas as pd

from backtesting.simulate_multi_agent_30m import load_inputs


# ======================================================
# AURUM-AI — B.15.2
# FINANCIAL QUALITY OF UP PREDICTIONS
#
# XOM validation only.
# No new model training.
# No test dataset.
# No real trading.
# ======================================================

COST_BPS_PER_SIDE = 10.0


def analyze_model(
    signals,
    market_prices,
    model_name,
    prediction_column,
    score_column,
):

    print(f"\n=== {model_name} — UP SCORE AUDIT ===")

    # Keep only model predictions classified as UP.
    data = signals.loc[
        signals[prediction_column] == 2
    ].copy()

    if data.empty:
        raise ValueError("No UP predictions found.")

    # Historical Bid opening prices.
    opens = market_prices.set_index("timestamp")["open"]

    data["entry_price"] = (
        data["entry_execution_timestamp_ms"].map(opens)
    )

    data["exit_price"] = (
        data["scheduled_exit_timestamp_ms"].map(opens)
    )

    if data[["entry_price", "exit_price"]].isna().any().any():
        raise ValueError("Missing execution prices.")

    if (
        (data["entry_price"] <= 0).any()
        or (data["exit_price"] <= 0).any()
    ):
        raise ValueError("Invalid execution prices.")

    # Hypothetical BUY at entry open,
    # SELL at the planned exit open.
    # Fees on both sides, not observed spread.
    cost = COST_BPS_PER_SIDE / 10000.0

    price_ratio = (
        data["exit_price"] / data["entry_price"]
    )

    data["gross_return_pct"] = (
        price_ratio - 1.0
    ) * 100.0

    data["net_return_pct"] = (
        price_ratio * (1.0 - cost) / (1.0 + cost)
        - 1.0
    ) * 100.0

    # Divide UP signals into five score groups.
    # Q1: lowest UP scores.
    # Q5: highest UP scores.
    data["score_quintile"] = (
        pd.qcut(
            data[score_column],
            q=5,
            labels=False,
            duplicates="drop",
        ) + 1
    )

    results = data.groupby("score_quintile").agg(
        signals=(score_column, "size"),
        score_min=(score_column, "min"),
        score_max=(score_column, "max"),
        avg_gross_pct=("gross_return_pct", "mean"),
        avg_net_pct=("net_return_pct", "mean"),
        profitable_pct=(
            "net_return_pct",
            lambda values: 100.0 * (values > 0).mean(),
        ),
    )

    print(f"UP predictions analyzed: {len(data)}")
    print(f"Cost per side: {COST_BPS_PER_SIDE} bps")

    print("\n=== SCORE QUINTILES ===")
    print(results.round(4).to_string())

    print("\n=== ALL UP SIGNALS ===")
    print(
        "Average gross return: "
        f"{data['gross_return_pct'].mean():+.4f}%"
    )
    print(
        "Average net return: "
        f"{data['net_return_pct'].mean():+.4f}%"
    )
    print(
        "Profitable opportunities after costs: "
        f"{100 * (data['net_return_pct'] > 0).mean():.2f}%"
    )

    print("\nIMPORTANT:")
    print("Signals are evaluated independently.")
    print("Overlapping positions are not simulated.")
    print("These figures are NOT portfolio returns.")


def main():

    print(
        "\n=== AURUM-AI | B.15.2 UP SCORE QUALITY ==="
    )

    signals, prices = load_inputs()

    models = [
        (
            "CNN1D",
            "cnn_class",
            "cnn_score_up",
        ),
        (
            "Logistic Regression",
            "logistic_class",
            "logistic_score_up",
        ),
    ]

    for model_name, prediction_col, score_col in models:

        analyze_model(
            signals=signals,
            market_prices=prices,
            model_name=model_name,
            prediction_column=prediction_col,
            score_column=score_col,
        )

    print("\nXOM original dataset preserved.")
    print("No files modified.")
    print("No test dataset used.")
    print("No real trading.")

    print("\nB.15.2 SCORE QUALITY AUDIT COMPLETED")


if __name__ == "__main__":
    main()
