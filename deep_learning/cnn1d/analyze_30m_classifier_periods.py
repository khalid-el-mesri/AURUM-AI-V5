
from pathlib import Path
import csv
import json
from datetime import date

import numpy as np


# ====================================================
# AURUM-AI — CLASSIFIER TEMPORAL ROBUSTNESS
# Four chronological validation periods
# No retraining, no test data
# ====================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT / "reports"
    / "cnn1d_30m_classifier_daily_validation.csv"
)

OUTPUT_PATH = (
    ROOT / "reports"
    / "cnn1d_30m_temporal_robustness_validation.json"
)

N_PERIODS = 4


def main():

    print("\n=== AURUM-AI | TEMPORAL ROBUSTNESS ===")

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Existing report preserved: {OUTPUT_PATH}"
        )

    with open(INPUT_PATH, "r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if len(rows) < N_PERIODS:
        raise ValueError("Insufficient validation days.")

    rows.sort(key=lambda row: row["date"])

    dates = [
        date.fromisoformat(row["date"])
        for row in rows
    ]

    if len(set(dates)) != len(dates):
        raise ValueError("Duplicate dates detected.")

    results = []

    groups = np.array_split(
        np.arange(len(rows)), N_PERIODS
    )

    for number, indices in enumerate(groups, start=1):

        period = [rows[int(i)] for i in indices]

        windows = sum(
            int(row["windows"])
            for row in period
        )

        cnn_correct = sum(
            int(row["cnn_correct"])
            for row in period
        )

        baseline_correct = sum(
            int(row["baseline_correct"])
            for row in period
        )

        if (
            windows <= 0
            or cnn_correct > windows
            or baseline_correct > windows
            or cnn_correct < 0
            or baseline_correct < 0
        ):
            raise ValueError("Invalid daily counts.")

        cnn_accuracy = 100 * cnn_correct / windows

        baseline_accuracy = (
            100 * baseline_correct / windows
        )

        difference = (
            cnn_accuracy - baseline_accuracy
        )

        better_days = sum(
            float(row["difference_pp"]) > 0
            for row in period
        )

        worse_days = sum(
            float(row["difference_pp"]) < 0
            for row in period
        )

        equal_days = (
            len(period) - better_days - worse_days
        )

        result = {
            "period": number,
            "first_date": period[0]["date"],
            "last_date": period[-1]["date"],
            "trading_days": len(period),
            "windows": windows,
            "cnn_accuracy_pct": cnn_accuracy,
            "baseline_accuracy_pct": baseline_accuracy,
            "difference_pp": difference,
            "cnn_better_days": better_days,
            "baseline_better_days": worse_days,
            "equal_days": equal_days,
        }

        results.append(result)

        print(f"\n=== PERIOD {number} ===")
        print(
            f"Dates: {result['first_date']} "
            f"to {result['last_date']}"
        )
        print(f"Trading days: {len(period)}")
        print(f"Windows: {windows}")

        print(
            f"CNN1D Accuracy: "
            f"{cnn_accuracy:.4f}%"
        )

        print(
            f"Baseline Accuracy: "
            f"{baseline_accuracy:.4f}%"
        )

        print(
            f"Improvement: {difference:+.4f} pp"
        )

        print(
            f"Better / Worse / Equal days: "
            f"{better_days} / {worse_days} / {equal_days}"
        )

    positive_periods = sum(
        result["difference_pp"] > 0
        for result in results
    )

    print("\n=== TEMPORAL SUMMARY ===")

    print(
        f"Positive periods: "
        f"{positive_periods}/{N_PERIODS}"
    )

    total_windows = sum(
        row["windows"] for row in results
    )

    overall_difference = sum(
        row["difference_pp"] * row["windows"]
        for row in results
    ) / total_windows

    print(
        f"Overall improvement: "
        f"{overall_difference:+.4f} pp"
    )

    report = {
        "project": "AURUM-AI",
        "model": "CNN1D Classifier 30m",
        "baseline": "Train-majority DOWN",
        "evaluation_split": "validation",
        "analysis": "Chronological four-period comparison",
        "positive_periods": positive_periods,
        "total_periods": N_PERIODS,
        "overall_improvement_pp": overall_difference,
        "periods": results,
        "test_dataset_used": False,
        "limitations": [
            "Model and horizon selected on validation.",
            "Overlapping forecasts create dependence.",
            "Not an independent out-of-sample test.",
        ],
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    with open(
        OUTPUT_PATH, "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    print(f"\nReport saved: {OUTPUT_PATH}")
    print("Test dataset was NOT used.")
    print("\nTEMPORAL ROBUSTNESS COMPLETED")


if __name__ == "__main__":
    main()
