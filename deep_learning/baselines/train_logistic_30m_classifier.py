
from pathlib import Path
import json

import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)


# AURUM-AI — Logistic Regression 30m Classification
# Same inputs as CNN1D; Train + Validation only

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "05_model_data"

CNN_REPORT = (
    ROOT / "reports"
    / "cnn1d_30m_classification_validation.json"
)

MODEL_PATH = (
    ROOT / "artifacts" / "models"
    / "logistic_30m_classifier.joblib"
)

REPORT_PATH = (
    ROOT / "reports"
    / "logistic_30m_classification_validation.json"
)

CLASS_NAMES = ["DOWN", "NEUTRAL", "UP"]


def load_data(split):
    path = DATA_DIR / f"{split}_30m_classification.npz"

    with np.load(path, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y_class"].astype(np.int64)
        features = data["feature_names"].tolist()
        end = data["end_timestamp_ms"].astype(np.int64)
        target = data["target_timestamp_ms"].astype(np.int64)
        threshold = float(data["threshold_log_return"])

    if X.ndim != 3 or X.shape[1:] != (20, 6):
        raise ValueError(f"Invalid X shape: {split}")

    if not (len(X) == len(y) == len(end) == len(target)):
        raise ValueError(f"Length mismatch: {split}")

    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite features: {split}")

    if not np.isin(y, [0, 1, 2]).all():
        raise ValueError(f"Invalid labels: {split}")

    if not np.all(target - end == 30 * 60_000):
        raise ValueError(f"Invalid horizon: {split}")

    if len(np.unique(end)) != len(end):
        raise ValueError(f"Duplicate timestamps: {split}")

    return X, y, features, end, target, threshold


def evaluate(actual, predicted):
    return {
        "accuracy": float(
            accuracy_score(actual, predicted)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(actual, predicted)
        ),
        "macro_f1": float(
            f1_score(
                actual,
                predicted,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
    }


def print_metrics(name, metrics):
    print(f"\n{name}")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(
        f"Balanced Accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )
    print(f"Macro-F1:          {metrics['macro_f1']:.4f}")


def main():
    print("\n=== AURUM-AI | LOGISTIC REGRESSION 30M ===")

    for path in [MODEL_PATH, REPORT_PATH]:
        if path.exists():
            raise FileExistsError(
                f"Existing artifact preserved: {path}"
            )

    (
        X_train, y_train, train_features,
        train_end, train_target, train_threshold
    ) = load_data("train")

    (
        X_val, y_val, val_features,
        val_end, val_target, val_threshold
    ) = load_data("validation")

    if train_features != val_features:
        raise ValueError("Feature order mismatch")

    if not np.isclose(
        train_threshold, val_threshold,
        rtol=0, atol=1e-12,
    ):
        raise ValueError("Threshold mismatch")

    if train_target.max() >= val_end.min():
        raise ValueError("Chronological separation failed")

    print(f"\nTrain: {X_train.shape}")
    print(f"Validation: {X_val.shape}")

    # Flatten 20 x 6 into 120 input features.
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)

    # Fixed hyperparameters — no grid search.
    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=700,
    )

    print("\nTraining Logistic Regression...")
    model.fit(X_train_flat, y_train)

    logistic_pred = model.predict(X_val_flat)

    logistic_metrics = evaluate(y_val, logistic_pred)

    # Majority class selected from TRAIN only.
    majority_class = int(
        np.argmax(np.bincount(y_train, minlength=3))
    )

    baseline_pred = np.full_like(
        y_val, majority_class
    )

    baseline_metrics = evaluate(y_val, baseline_pred)

    # Read the existing CNN1D validation results.
    with open(CNN_REPORT, "r", encoding="utf-8") as file:
        cnn_report = json.load(file)

    if (
        cnn_report["validation_samples"] != len(y_val)
        or cnn_report["train_samples"] != len(y_train)
    ):
        raise ValueError("CNN1D sample count mismatch")

    if cnn_report["baseline_class"] != CLASS_NAMES[majority_class]:
        raise ValueError("CNN1D baseline class mismatch")

    cnn_metrics = cnn_report["classifier_metrics"]

    for key in ("accuracy", "balanced_accuracy", "macro_f1"):
        if not np.isclose(
            cnn_report["baseline_metrics"][key],
            baseline_metrics[key],
            rtol=0,
            atol=1e-8,
        ):
            raise ValueError(
                f"CNN1D baseline mismatch: {key}"
            )

    print("\n=== FINAL VALIDATION COMPARISON ===")

    print_metrics("Majority Baseline", baseline_metrics)
    print_metrics("Logistic Regression", logistic_metrics)
    print_metrics("CNN1D Classifier", cnn_metrics)

    comparison = {}

    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        comparison[metric] = float(
            cnn_metrics[metric] - logistic_metrics[metric]
        )

    print("\n=== CNN1D VS LOGISTIC REGRESSION ===")
    print(
        "Accuracy difference: "
        f"{comparison['accuracy'] * 100:+.4f} pp"
    )
    print(
        "Balanced Accuracy difference: "
        f"{comparison['balanced_accuracy'] * 100:+.4f} pp"
    )
    print(
        "Macro-F1 difference: "
        f"{comparison['macro_f1']:+.4f}"
    )

    print(
        "\nLogistic Regression confusion matrix "
        "(DOWN, NEUTRAL, UP):"
    )

    cm = confusion_matrix(
        y_val, logistic_pred, labels=[0, 1, 2]
    )
    print(cm)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "feature_names": train_features,
            "window_size": 20,
            "horizon_minutes": 30,
            "neutral_threshold": train_threshold,
            "classes": CLASS_NAMES,
        },
        MODEL_PATH,
    )

    report = {
        "project": "AURUM-AI",
        "task": "30m classification",
        "evaluation_split": "validation",
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "models": {
            "majority": baseline_metrics,
            "logistic_regression": logistic_metrics,
            "cnn1d": cnn_metrics,
        },
        "cnn_minus_logistic": comparison,
        "logistic_confusion_matrix": cm.tolist(),
        "logistic_hyperparameters": {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 700,
        },
        "test_dataset_used": False,
        "limitations": [
            "CNN1D was selected using validation.",
            "Comparisons are exploratory.",
            "Overlapping 30m targets are dependent.",
        ],
    }

    with open(
        REPORT_PATH, "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Report saved: {REPORT_PATH}")
    print("Test dataset was NOT used.")
    print("\nLOGISTIC COMPARISON COMPLETED")


if __name__ == "__main__":
    main()
