
from pathlib import Path
import csv
import json
import os

import numpy as np
import pandas as pd
import torch

from train_cnn1d_30m_classifier import CNN1DClassifier


# =====================================================
# AURUM-AI — CNN1D 30M DAILY CLASSIFICATION ANALYSIS
# Validation only — No retraining, no test usage
# =====================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "05_model_data"

TRAIN_PATH = DATA_DIR / "train_30m_classification.npz"
VAL_PATH = DATA_DIR / "validation_30m_classification.npz"

MODEL_PATH = (
    ROOT / "artifacts" / "models"
    / "cnn1d_30m_classifier_best.pt"
)

ORIGINAL_REPORT = (
    ROOT / "reports"
    / "cnn1d_30m_classification_validation.json"
)

CSV_PATH = (
    ROOT / "reports"
    / "cnn1d_30m_classifier_daily_validation.csv"
)

JSON_PATH = (
    ROOT / "reports"
    / "cnn1d_30m_classifier_daily_validation.json"
)

CLASS_NAMES = ["DOWN", "NEUTRAL", "UP"]


def main():

    print("\n=== AURUM-AI | DAILY CLASSIFIER ANALYSIS ===")

    torch.set_num_threads(
        min(4, os.cpu_count() or 1)
    )

    # Preserve previous results
    if CSV_PATH.exists() or JSON_PATH.exists():
        raise FileExistsError(
            "Daily analysis already exists. "
            "Existing results were preserved."
        )

    # Determine majority class using TRAIN only
    with np.load(TRAIN_PATH, allow_pickle=False) as data:
        y_train = data["y_class"].astype(np.int64)

    train_counts = np.bincount(
        y_train, minlength=3
    )

    majority_class = int(np.argmax(train_counts))

    print(
        f"Train majority baseline: "
        f"{CLASS_NAMES[majority_class]}"
    )

    # Read VALIDATION only
    with np.load(VAL_PATH, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y_class"].astype(np.int64)
        end_ts = data["end_timestamp_ms"].astype(np.int64)
        features = data["feature_names"].tolist()
        threshold = float(data["threshold_log_return"])

    if not (len(X) == len(y) == len(end_ts)):
        raise ValueError("Dataset length mismatch.")

    if not np.isfinite(X).all():
        raise ValueError("Invalid input features.")

    if not np.isin(y, [0, 1, 2]).all():
        raise ValueError("Invalid validation labels.")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["feature_names"] != features:
        raise ValueError("Feature order mismatch.")

    if checkpoint["window_size"] != X.shape[1]:
        raise ValueError("Window size mismatch.")

    if checkpoint["input_size"] != X.shape[2]:
        raise ValueError("Input size mismatch.")

    if not np.isclose(
        checkpoint["neutral_threshold"],
        threshold,
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("Threshold mismatch.")

    # Restore the trained model
    model = CNN1DClassifier(
        input_size=X.shape[2]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    predictions = []

    with torch.no_grad():
        for start in range(0, len(X), 512):
            xb = torch.from_numpy(
                X[start:start + 512]
            )

            logits = model(xb)

            labels = torch.argmax(
                logits, dim=1
            )

            predictions.append(
                labels.cpu().numpy()
            )

    cnn_pred = np.concatenate(predictions)

    baseline_pred = np.full_like(
        y, majority_class
    )

    cnn_accuracy = float(
        np.mean(cnn_pred == y)
    )

    baseline_accuracy = float(
        np.mean(baseline_pred == y)
    )

    # Confirm consistency with the original report
    with open(
        ORIGINAL_REPORT, "r", encoding="utf-8"
    ) as file:
        original = json.load(file)

    recorded_accuracy = original[
        "classifier_metrics"
    ]["accuracy"]

    if not np.isclose(
        cnn_accuracy,
        recorded_accuracy,
        atol=1e-8,
        rtol=0,
    ):
        raise ValueError(
            "Predictions do not match original report."
        )

    # Convert timestamps to New York trading dates
    dates = np.asarray(
        pd.to_datetime(
            end_ts, unit="ms", utc=True
        ).tz_convert(
            "America/New_York"
        ).strftime("%Y-%m-%d")
    )

    unique_dates = np.unique(dates)

    daily_results = []

    for date in unique_dates:

        mask = dates == date
        actual = y[mask]

        cnn_correct = int(
            np.sum(cnn_pred[mask] == actual)
        )

        baseline_correct = int(
            np.sum(baseline_pred[mask] == actual)
        )

        n = int(mask.sum())

        cnn_acc = cnn_correct / n
        baseline_acc = baseline_correct / n

        difference_pp = (
            cnn_acc - baseline_acc
        ) * 100

        daily_results.append({
            "date": str(date),
            "windows": n,
            "cnn_correct": cnn_correct,
            "baseline_correct": baseline_correct,
            "cnn_accuracy": cnn_acc,
            "baseline_accuracy": baseline_acc,
            "difference_pp": difference_pp,
        })

    better_days = sum(
        row["difference_pp"] > 0
        for row in daily_results
    )

    worse_days = sum(
        row["difference_pp"] < 0
        for row in daily_results
    )

    equal_days = (
        len(daily_results)
        - better_days
        - worse_days
    )

    print("\n=== OVERALL VALIDATION ===")

    print(f"Windows: {len(y)}")
    print(f"CNN1D accuracy: {cnn_accuracy * 100:.4f}%")
    print(
        f"Baseline accuracy: "
        f"{baseline_accuracy * 100:.4f}%"
    )

    print(
        f"Accuracy improvement: "
        f"{(cnn_accuracy - baseline_accuracy) * 100:+.4f} pp"
    )

    print("\n=== DAILY STABILITY ===")

    print(f"Trading days: {len(daily_results)}")
    print(f"Days CNN1D better: {better_days}")
    print(f"Days baseline better: {worse_days}")
    print(f"Equal days: {equal_days}")

    CSV_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    with open(
        CSV_PATH, "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(daily_results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(daily_results)

    report = {
        "project": "AURUM-AI",
        "model": "CNN1D 30m Classifier",
        "baseline": CLASS_NAMES[majority_class],
        "evaluation_split": "validation",
        "total_windows": int(len(y)),
        "trading_days": len(daily_results),
        "cnn_accuracy": cnn_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "accuracy_difference_pp": float(
            (cnn_accuracy - baseline_accuracy) * 100
        ),
        "cnn_better_days": better_days,
        "baseline_better_days": worse_days,
        "equal_days": equal_days,
        "test_dataset_used": False,
        "note": (
            "Exploratory validation analysis. "
            "Not an independent out-of-sample test. "
            "Overlapping forecast horizons may "
            "create temporal dependence."
        ),
    }

    with open(
        JSON_PATH, "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    print("\n=== FILES SAVED ===")
    print(CSV_PATH)
    print(JSON_PATH)

    print("\nTest dataset was NOT used.")
    print("DAILY CLASSIFICATION ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()
