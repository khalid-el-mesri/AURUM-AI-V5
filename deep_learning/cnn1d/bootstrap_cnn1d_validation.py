
from pathlib import Path
import csv
import json

import numpy as np


# ====================================================
# AURUM-AI — CNN1D MOVING BLOCK BOOTSTRAP
# Validation only — Test data is NOT used
# ====================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "reports" / "cnn1d_daily_validation.csv"
)

OUTPUT_FILE = (
    ROOT / "reports"
    / "cnn1d_block_bootstrap_validation.json"
)

SEED = 42
N_BOOTSTRAP = 3000
BLOCK_SIZE = 10
CONFIDENCE_LEVEL = 95


def load_daily_results():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    with open(
        INPUT_FILE, "r", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("Daily validation report is empty.")

    rows.sort(key=lambda row: row["date"])

    dates = [row["date"] for row in rows]

    if len(set(dates)) != len(dates):
        raise ValueError("Duplicate dates detected.")

    windows = np.array(
        [int(row["windows"]) for row in rows],
        dtype=np.int64,
    )

    cnn_mae = np.array(
        [float(row["cnn_mae"]) for row in rows],
        dtype=np.float64,
    )

    zero_mae = np.array(
        [float(row["zero_mae"]) for row in rows],
        dtype=np.float64,
    )

    if not np.all(windows > 0):
        raise ValueError("Invalid window counts.")

    if not np.all(np.isfinite(cnn_mae)):
        raise ValueError("Invalid CNN1D MAE.")

    if not np.all(np.isfinite(zero_mae)):
        raise ValueError("Invalid Zero Return MAE.")

    if np.any(cnn_mae < 0) or np.any(zero_mae < 0):
        raise ValueError("Negative MAE detected.")

    return dates, windows, cnn_mae, zero_mae


def calculate_improvement(indices, windows, cnn, zero):
    # Each daily MAE is weighted by the number of windows.
    weights = windows[indices]

    cnn_errors = np.sum(
        weights * cnn[indices]
    )

    zero_errors = np.sum(
        weights * zero[indices]
    )

    total_windows = np.sum(weights)

    cnn_weighted_mae = cnn_errors / total_windows
    zero_weighted_mae = zero_errors / total_windows

    # Positive difference means CNN1D is better.
    absolute_improvement = (
        zero_weighted_mae - cnn_weighted_mae
    )

    if zero_weighted_mae <= 0:
        raise ValueError("Zero Return MAE must be positive.")

    percentage_improvement = (
        absolute_improvement
        / zero_weighted_mae
        * 100
    )

    return absolute_improvement, percentage_improvement


def sample_block_indices(rng, n_days, block_size):
    indices = []

    while len(indices) < n_days:
        start = int(
            rng.integers(0, n_days - block_size + 1)
        )

        indices.extend(
            range(start, start + block_size)
        )

    return np.array(
        indices[:n_days],
        dtype=np.int64,
    )


def main():
    print("\n=== AURUM-AI | BLOCK BOOTSTRAP ===")

    dates, windows, cnn, zero = load_daily_results()

    n_days = len(dates)

    if n_days < BLOCK_SIZE:
        raise ValueError(
            "Insufficient trading days for bootstrap."
        )

    print(f"\nTrading days: {n_days}")
    print(f"Total windows: {int(windows.sum())}")
    print(f"Bootstrap repetitions: {N_BOOTSTRAP}")
    print(f"Block size: {BLOCK_SIZE} trading days")
    print(f"Confidence level: {CONFIDENCE_LEVEL}%")

    all_indices = np.arange(n_days)

    observed_absolute, observed_pct = (
        calculate_improvement(
            all_indices, windows, cnn, zero
        )
    )

    rng = np.random.default_rng(SEED)

    bootstrap_absolute = []
    bootstrap_percentage = []

    print("\nRunning bootstrap...")

    for _ in range(N_BOOTSTRAP):
        sample_indices = sample_block_indices(
            rng, n_days, BLOCK_SIZE
        )

        absolute, percentage = calculate_improvement(
            sample_indices,
            windows,
            cnn,
            zero,
        )

        bootstrap_absolute.append(absolute)
        bootstrap_percentage.append(percentage)

    bootstrap_absolute = np.array(
        bootstrap_absolute
    )

    bootstrap_percentage = np.array(
        bootstrap_percentage
    )

    alpha = (100 - CONFIDENCE_LEVEL) / 200

    absolute_ci = np.quantile(
        bootstrap_absolute,
        [alpha, 1 - alpha],
    )

    percentage_ci = np.quantile(
        bootstrap_percentage,
        [alpha, 1 - alpha],
    )

    includes_zero = bool(
        percentage_ci[0] <= 0 <= percentage_ci[1]
    )

    print("\n=== OBSERVED PERFORMANCE ===")

    print(
        f"Absolute MAE improvement: "
        f"{observed_absolute:.12f}"
    )

    print(
        f"Relative MAE improvement: "
        f"{observed_pct:.6f}%"
    )

    print("\n=== BLOCK BOOTSTRAP RESULTS ===")

    print(
        f"95% CI absolute MAE: "
        f"[{absolute_ci[0]:.12f}, "
        f"{absolute_ci[1]:.12f}]"
    )

    print(
        f"95% CI relative improvement: "
        f"[{percentage_ci[0]:.6f}%, "
        f"{percentage_ci[1]:.6f}%]"
    )

    print(
        f"Confidence interval includes zero: "
        f"{includes_zero}"
    )

    if includes_zero:
        print(
            "\nConclusion: No clear improvement "
            "over Zero Return from this diagnostic."
        )
    elif percentage_ci[0] > 0:
        print(
            "\nConclusion: Positive improvement "
            "in this exploratory bootstrap."
        )
    else:
        print(
            "\nConclusion: CNN1D underperforms "
            "in this exploratory bootstrap."
        )

    report = {
        "project": "AURUM-AI",
        "model": "CNN1D",
        "baseline": "Zero Return",
        "evaluation_split": "validation",
        "method": "Moving Block Bootstrap",
        "block_size": BLOCK_SIZE,
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "confidence_level": CONFIDENCE_LEVEL,
        "trading_days": n_days,
        "total_windows": int(windows.sum()),
        "observed_absolute_improvement": float(
            observed_absolute
        ),
        "observed_improvement_pct": float(
            observed_pct
        ),
        "absolute_improvement_ci": (
            absolute_ci.tolist()
        ),
        "percentage_improvement_ci": (
            percentage_ci.tolist()
        ),
        "ci_includes_zero": includes_zero,
        "test_dataset_used": False,
        "limitations": [
            "The model was already selected using validation.",
            "Bootstrap results are exploratory.",
            "Temporal market regimes may change.",
            "No trading profitability is evaluated.",
        ],
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    with open(
        OUTPUT_FILE, "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    print(f"\nReport saved: {OUTPUT_FILE}")
    print("Test dataset was NOT used.")
    print("\nBLOCK BOOTSTRAP COMPLETED")


if __name__ == "__main__":
    main()
