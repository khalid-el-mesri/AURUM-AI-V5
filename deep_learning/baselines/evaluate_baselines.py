
from pathlib import Path
import json

import numpy as np


# ==================================================
# AURUM-AI — BASELINE MODEL EVALUATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "05_model_data"

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "model_baselines_validation.json"
)


def evaluate(y_true, y_pred):
    """Calculate prediction errors."""

    mae = np.mean(np.abs(y_true - y_pred))

    rmse = np.sqrt(
        np.mean((y_true - y_pred) ** 2)
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
    }


def load_data(name):
    path = DATA_DIR / f"{name}_normalized.npz"

    with np.load(path, allow_pickle=False) as file:
        return {
            key: file[key]
            for key in file.files
        }


def main():

    print("\n=== AURUM-AI | BASELINE EVALUATION ===")

    # 1. Load Train and Validation only
    train = load_data("train")
    validation = load_data("validation")

    y_train = train["y"].astype(np.float64)
    y_val = validation["y"].astype(np.float64)

    X_val = validation["X"].astype(np.float64)

    # 2. Load normalization parameters
    stats_path = DATA_DIR / "normalization_stats.json"

    with open(stats_path, "r", encoding="utf-8") as file:
        stats = json.load(file)

    feature_names = stats["feature_names"]

    if validation["feature_names"].tolist() != feature_names:
        raise ValueError("Feature order mismatch.")

    # Find log-return feature
    log_return_index = feature_names.index(
        "log_return_1"
    )

    # 3. Baseline A: Always predict zero return
    zero_predictions = np.zeros_like(y_val)

    # 4. Baseline B: Mean return from TRAIN only
    train_mean = y_train.mean()

    mean_predictions = np.full_like(
        y_val,
        train_mean,
    )

    # 5. Baseline C: Repeat the last observed return
    # Undo normalization using TRAIN parameters

    last_normalized_return = X_val[
        :, -1, log_return_index
    ]

    last_return_predictions = (
        last_normalized_return
        * stats["std"][log_return_index]
        + stats["mean"][log_return_index]
    )

    # 6. Evaluate all baselines
    models = {
        "Zero Return": zero_predictions,
        "Historical Mean": mean_predictions,
        "Last Return": last_return_predictions,
    }

    results = {}

    print(f"\nTraining samples: {len(y_train):,}")
    print(f"Validation samples: {len(y_val):,}")

    print("\nBASELINE RESULTS — VALIDATION")
    print("-" * 48)

    for name, predictions in models.items():

        if not np.isfinite(predictions).all():
            raise ValueError(
                f"Invalid predictions: {name}"
            )

        metrics = evaluate(y_val, predictions)

        results[name] = metrics

        print(f"\n{name}")
        print(f"MAE:  {metrics['MAE']:.8f}")
        print(f"RMSE: {metrics['RMSE']:.8f}")

    # 7. Identify best validation MAE
    best_model = min(
        results,
        key=lambda name: results[name]["MAE"],
    )

    print("\nBEST BASELINE BY VALIDATION MAE")
    print(best_model)

    # 8. Save reproducible results
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "evaluation_split": "validation",
        "target": "next_5min_log_return",
        "train_samples": len(y_train),
        "validation_samples": len(y_val),
        "metrics": results,
        "best_baseline_by_mae": best_model,
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=2)

    print(f"\nReport saved: {REPORT_PATH}")
    print("Test dataset was not used.")
    print("\nBASELINE EVALUATION COMPLETED")


if __name__ == "__main__":
    main()
