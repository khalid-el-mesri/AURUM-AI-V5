
from pathlib import Path

import numpy as np
import pandas as pd


# AURUM-AI — COST SENSITIVITY ANALYSIS
# Existing validation trades only
# No new trades, no training, no test data

ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "CNN1D": ROOT / "reports/backtesting/cnn1d_trades_validation.csv",
    "Logistic Regression": ROOT / "reports/backtesting/logistic_trades_validation.csv",
}

COST_SCENARIOS_BPS = [0, 1, 2, 5, 10]


def analyze_model(name, path):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = [
        "entry_bid_reference",
        "exit_bid_reference",
        "net_trade_return_pct",
    ]

    if df.empty or df[required].isna().any().any():
        raise ValueError(f"Invalid trade data: {name}")

    entry = df["entry_bid_reference"].to_numpy()
    exit_price = df["exit_bid_reference"].to_numpy()

    if np.any(entry <= 0) or np.any(exit_price <= 0):
        raise ValueError("Prices must be positive.")

    gross_ratio = exit_price / entry

    print(f"\n=== {name.upper()} ===")
    print(f"Existing trades: {len(df)}")

    gross_return = (gross_ratio - 1) * 100

    print(
        f"Average gross return: "
        f"{gross_return.mean():+.5f}%"
    )

    print("\nCost sensitivity (per side):")

    for bps in COST_SCENARIOS_BPS:
        cost = bps / 10000.0

        net_ratio = (
            gross_ratio * (1 - cost) / (1 + cost)
        )

        net_returns = (net_ratio - 1) * 100

        mean_return = float(net_returns.mean())
        win_rate = float(np.mean(net_returns > 0) * 100)

        print(
            f"{bps:>2} bps | "
            f"Avg net trade: {mean_return:+.5f}% | "
            f"Win rate: {win_rate:.2f}%"
        )

        if bps == 10:
            recorded = df["net_trade_return_pct"].mean()

            if not np.isclose(
                mean_return,
                recorded,
                rtol=0,
                atol=1e-7,
            ):
                raise ValueError(
                    "Cost calculation mismatch."
                )

    mean_ratio = float(gross_ratio.mean())

    if mean_ratio > 1:
        break_even_bps = (
            (mean_ratio - 1)
            / (mean_ratio + 1)
            * 10000
        )

        print(
            f"\nApprox break-even cost per side: "
            f"{break_even_bps:.4f} bps"
        )
    else:
        print(
            "\nBreak-even: not achievable with "
            "nonnegative costs on these fixed trades."
        )


def main():
    print("=== AURUM-AI | COST SENSITIVITY ===")

    for name, path in FILES.items():
        analyze_model(name, path)

    print("\n=== IMPORTANT ===")
    print("Same executed trades in all cost scenarios.")
    print("Risk decisions were NOT recalculated.")
    print("These are per-trade metrics, not portfolio returns.")
    print("No real trades. No test dataset used.")
    print("\nCOST SENSITIVITY COMPLETED")


if __name__ == "__main__":
    main()
