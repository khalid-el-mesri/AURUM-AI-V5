
from pathlib import Path
from datetime import date
import csv
import json
import math

import numpy as np


# ====================================================
# AURUM-AI — CNN1D 30M CLASSIFIER BLOCK BOOTSTRAP
# Exploratory validation analysis only
# No training, no test data, no trades
# ====================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT / "reports"
    / "cnn1d_30m_classifier_daily_validation.csv"
)

ORIGINAL_REPORT = (
    ROOT / "reports"
    / "cnn1d_30m_classification_validation.json"
)

OUTPUT_PATH = (
    ROOT / "reports"
    / "cnn1d_30m_classifier_bootstrap_validation.json"
)

BLOCK_SIZE = 10
N_BOOTSTRAP = 3000
SEED = 42
CONFIDENCE_LEVEL = 95


def load_daily_data():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    with open(INPUT_PATH, "r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("Daily report is empty.")

    rows.sort(key=lambda row: row["date"])

    dates = [
        date.fromisoformat(row["date"])
        for row in rows
    ]

    if len(set(dates)) != len(dates):
        raise ValueError("Duplicate trading dates.")

    windows = np.array(
        [int(row["windows"]) for row in rows],
        dtype=np.int64,
    )

    cnn_correct = np.array(
        [int(row["cnn_correct"]) for row in rows],
        dtype=np.int64,
    )

    baseline_correct = np.array(
        [int(row["baseline_correct"]) for row in rows],
        dtype=np.int64,
    )

    if np.any(windows <= 0):
        raise ValueError("Invalid window counts.")

    if np.any(cnn_correct < 0) or np.any(
        cnn_correct > windows
    ):
        raise ValueError("Invalid CNN1D correct counts.")

    if np.any(baseline_correct < 0) or np.any(
        baseline_correct > windows
    ):
        raise ValueError("Invalid baseline correct counts.")

    return dates, windows, cnn_correct, baseline_correct


def calculate_difference(
    indices,
    windows,
    cnn_correct,
    baseline_correct,
):
    total = int(np.sum(windows[indices]))

    cnn_accuracy = (
        np.sum(cnn_correct[indices]) / total
    )

    baseline_accuracy = (
        np.sum(baseline_correct[indices]) / total
    )

    # Positive value means CNN1D performs better.
    return float(
        100 * (cnn_accuracy - baseline_accuracy)
    )


def main():

    print(
        "\n=== AURUM-AI | CLASSIFIER BLOCK BOOTSTRAP ==="
    )

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Existing report preserved: {OUTPUT_PATH}"
        )

    dates, windows, cnn, baseline = load_daily_data()

    n_days = len(dates)

    if n_days < BLOCK_SIZE:
        raise ValueError("Not enough trading days.")

    all_indices = np.arange(n_days)

    observed_difference = calculate_difference(
        all_indices,
        windows,
        cnn,
        baseline,
    )

    print(f"\nTrading days: {n_days}")
    print(f"Validation windows: {int(windows.sum())}")
    print(f"Block size: {BLOCK_SIZE} trading days")
    print(f"Bootstrap repetitions: {N_BOOTSTRAP}")

    print(
        f"\nObserved accuracy difference: "
        f"{observed_difference:+.4f} percentage points"
    )

    # Verify consistency with the original classifier report.
    with open(
        ORIGINAL_REPORT, "r", encoding="utf-8"
    ) as file:
        original = json.load(file)

    recorded_difference = 100 * (
        original["classifier_metrics"]["accuracy"]
        - original["baseline_metrics"]["accuracy"]
    )

    if not math.isclose(
        observed_difference,
        recorded_difference,
        abs_tol=1e-8,
        rel_tol=0,
    ):
        raise ValueError(
            "Daily accuracy does not match original report."
        )

    rng = np.random.default_rng(SEED)

    n_blocks = math.ceil(n_days / BLOCK_SIZE)

    bootstrap_differences = np.empty(
        N_BOOTSTRAP,
        dtype=np.float64,
    )

    print("\nRunning moving block bootstrap...")

    for iteration in range(N_BOOTSTRAP):

        starts = rng.integers(
            0,
            n_days - BLOCK_SIZE + 1,
            size=n_blocks,
        )

        indices = (
            starts[:, None]
            + np.arange(BLOCK_SIZE)[None, :]
        ).reshape(-1)[:n_days]

        bootstrap_differences[iteration] = (
            calculate_difference(
                indices,
                windows,
                cnn,
                baseline,
            )
        )

    alpha = (1 - CONFIDENCE_LEVEL / 100) / 2

    lower, upper = np.quantile(
        bootstrap_differences,
        [alpha, 1 - alpha],
    )

    includes_zero = bool(
        lower <= 0 <= upper
    )

    positive_percentage = float(
        np.mean(bootstrap_differences > 0) * 100
    )

    print("\n=== BLOCK BOOTSTRAP RESULTS ===")

    print(
        f"Observed improvement: "
        f"{observed_difference:+.4f} pp"
    )

    print(
        f"95% CI: [{lower:+.4f}, {upper:+.4f}] pp"
    )

    print(
        f"Confidence interval includes zero: "
        f"{includes_zero}"
    )

    print(
        f"Bootstrap samples with positive improvement: "
        f"{positive_percentage:.2f}%"
    )

    if includes_zero:
        conclusion = (
            "No clear positive improvement in "
            "this exploratory bootstrap."
        )
    elif lower > 0:
        conclusion = (
            "Positive improvement in the exploratory "
            "bootstrap; independent testing still needed."
        )
    else:
        conclusion = (
            "Negative improvement in the exploratory "
            "bootstrap."
        )

    print(f"\nConclusion: {conclusion}")

    report = {
        "project": "AURUM-AI",
        "model": "CNN1D 30m Classifier",
        "baseline": "Train-majority DOWN",
        "evaluation_split": "validation",
        "method": "Moving Block Bootstrap",
        "metric": "accuracy_difference_pp",
        "trading_days": n_days,
        "total_windows": int(windows.sum()),
        "block_size": BLOCK_SIZE,
        "bootstrap_repetitions": N_BOOTSTRAP,
        "seed": SEED,
        "confidence_level": CONFIDENCE_LEVEL,
        "observed_improvement_pp": observed_difference,
        "confidence_interval_pp": [
            float(lower),
            float(upper),
        ],
        "ci_includes_zero": includes_zero,
        "bootstrap_positive_percentage":
            positive_percentage,
        "conclusion": conclusion,
        "limitations": [
            "Model selected using the same validation period.",
            "Overlapping 30-minute forecasts are dependent.",
            "Block length is an exploratory choice.",
            "Interval is not a formal post-selection test.",
            "No trading profitability has been measured.",
        ],
        "test_dataset_used": False,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH, "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    print(f"\nReport saved: {OUTPUT_PATH}")
    print("Test dataset was NOT used.")
    print("\nCLASSIFIER BOOTSTRAP COMPLETED")


if __name__ == "__main__":
    main()
    