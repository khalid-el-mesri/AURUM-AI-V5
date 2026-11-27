
from pathlib import Path
import json
import math

import joblib
import numpy as np
import pandas as pd
import torch

from train_cnn1d_30m_classifier import CNN1DClassifier


# AURUM-AI — Paired Block Bootstrap
# CNN1D vs Logistic Regression
# Validation only

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "data/05_model_data/"
    "validation_30m_classification.npz"
)

CNN_PATH = (
    ROOT / "artifacts/models/"
    "cnn1d_30m_classifier_best.pt"
)

LOGISTIC_PATH = (
    ROOT / "artifacts/models/"
    "logistic_30m_classifier.joblib"
)

CNN_REPORT = (
    ROOT / "reports/"
    "cnn1d_30m_classification_validation.json"
)

LOGISTIC_REPORT = (
    ROOT / "reports/"
    "logistic_30m_classification_validation.json"
)

OUTPUT_PATH = (
    ROOT / "reports/"
    "cnn1d_vs_logistic_paired_bootstrap.json"
)

BLOCK_DAYS = 10
REPETITIONS = 3000
SEED = 42


def confusion(actual, predicted):
    return np.bincount(
        3 * actual + predicted,
        minlength=9,
    ).reshape(3, 3)


def metrics(matrix):
    diagonal = np.diag(matrix)
    actual_counts = matrix.sum(axis=1)
    predicted_counts = matrix.sum(axis=0)

    accuracy = diagonal.sum() / matrix.sum()

    recall = np.divide(
        diagonal,
        actual_counts,
        out=np.zeros(3, dtype=float),
        where=actual_counts > 0,
    )

    f1 = np.divide(
        2 * diagonal,
        actual_counts + predicted_counts,
        out=np.zeros(3, dtype=float),
        where=(actual_counts + predicted_counts) > 0,
    )

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }


def main():
    print("\n=== AURUM-AI | PAIRED BOOTSTRAP ===")

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Existing report preserved: {OUTPUT_PATH}"
        )

    torch.set_num_threads(4)

    with np.load(DATA_PATH, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y_class"].astype(np.int64)
        timestamps = data[
            "end_timestamp_ms"
        ].astype(np.int64)
        feature_names = data["feature_names"].tolist()
        threshold = float(data["threshold_log_return"])

    if X.ndim != 3 or X.shape[1:] != (20, 6):
        raise ValueError("Invalid window dimensions.")

    if not (len(X) == len(y) == len(timestamps)):
        raise ValueError("Dataset length mismatch.")

    if not np.isfinite(X).all():
        raise ValueError("Invalid input values.")

    if not np.isin(y, [0, 1, 2]).all():
        raise ValueError("Unexpected class labels.")

    if len(np.unique(timestamps)) != len(timestamps):
        raise ValueError("Duplicate timestamps.")

    # Load CNN1D checkpoint
    checkpoint = torch.load(
        CNN_PATH,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["feature_names"] != feature_names:
        raise ValueError("CNN1D feature mismatch.")

    if checkpoint["input_size"] != 6:
        raise ValueError("CNN1D input size mismatch.")

    if not np.isclose(
        checkpoint["neutral_threshold"],
        threshold,
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("CNN1D threshold mismatch.")

    cnn = CNN1DClassifier(input_size=6)
    cnn.load_state_dict(checkpoint["model_state_dict"])
    cnn.eval()

    cnn_predictions = []

    with torch.no_grad():
        for start in range(0, len(X), 512):
            batch = torch.from_numpy(
                X[start:start + 512]
            )
            predictions = cnn(batch).argmax(dim=1)
            cnn_predictions.append(
                predictions.cpu().numpy()
            )

    cnn_predictions = np.concatenate(cnn_predictions)

    # Load locally created, trusted Logistic model
    logistic_artifact = joblib.load(LOGISTIC_PATH)

    if logistic_artifact["feature_names"] != feature_names:
        raise ValueError("Logistic feature mismatch.")

    if not np.isclose(
        logistic_artifact["neutral_threshold"],
        threshold,
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("Logistic threshold mismatch.")

    logistic = logistic_artifact["model"]

    logistic_predictions = logistic.predict(
        X.reshape(len(X), -1)
    ).astype(np.int64)

    if not np.isin(
        cnn_predictions, [0, 1, 2]
    ).all():
        raise ValueError("Invalid CNN predictions.")

    if not np.isin(
        logistic_predictions, [0, 1, 2]
    ).all():
        raise ValueError("Invalid Logistic predictions.")

    # Verify predictions against saved validation reports
    with open(CNN_REPORT, encoding="utf-8") as file:
        cnn_saved = json.load(file)

    with open(LOGISTIC_REPORT, encoding="utf-8") as file:
        logistic_saved = json.load(file)

    cnn_scores = metrics(confusion(y, cnn_predictions))
    logistic_scores = metrics(
        confusion(y, logistic_predictions)
    )

    expected_cnn = cnn_saved["classifier_metrics"]
    expected_logistic = logistic_saved[
        "models"
    ]["logistic_regression"]

    for key in ("accuracy", "balanced_accuracy", "macro_f1"):
        if not np.isclose(
            cnn_scores[key], expected_cnn[key], atol=1e-8
        ):
            raise ValueError(f"CNN report mismatch: {key}")

        if not np.isclose(
            logistic_scores[key],
            expected_logistic[key],
            atol=1e-8,
        ):
            raise ValueError(
                f"Logistic report mismatch: {key}"
            )

    # Group paired predictions by NYSE calendar day
    dates = np.asarray(
        pd.to_datetime(
            timestamps, unit="ms", utc=True
        ).tz_convert(
            "America/New_York"
        ).strftime("%Y-%m-%d")
    )

    unique_days = np.unique(dates)

    cnn_daily = []
    logistic_daily = []

    for day in unique_days:
        mask = dates == day
        cnn_daily.append(
            confusion(y[mask], cnn_predictions[mask])
        )
        logistic_daily.append(
            confusion(y[mask], logistic_predictions[mask])
        )

    cnn_daily = np.stack(cnn_daily)
    logistic_daily = np.stack(logistic_daily)

    n_days = len(unique_days)

    if n_days < BLOCK_DAYS:
        raise ValueError("Insufficient trading days.")

    observed_accuracy_pp = 100 * (
        cnn_scores["accuracy"]
        - logistic_scores["accuracy"]
    )

    observed_macro_f1 = (
        cnn_scores["macro_f1"]
        - logistic_scores["macro_f1"]
    )

    rng = np.random.default_rng(SEED)
    n_blocks = math.ceil(n_days / BLOCK_DAYS)

    accuracy_differences = np.empty(REPETITIONS)
    f1_differences = np.empty(REPETITIONS)

    for i in range(REPETITIONS):
        starts = rng.integers(
            0,
            n_days - BLOCK_DAYS + 1,
            size=n_blocks,
        )

        sample = (
            starts[:, None]
            + np.arange(BLOCK_DAYS)[None, :]
        ).reshape(-1)[:n_days]

        cnn_matrix = cnn_daily[sample].sum(axis=0)
        logistic_matrix = (
            logistic_daily[sample].sum(axis=0)
        )

        a = metrics(cnn_matrix)
        b = metrics(logistic_matrix)

        accuracy_differences[i] = 100 * (
            a["accuracy"] - b["accuracy"]
        )

        f1_differences[i] = (
            a["macro_f1"] - b["macro_f1"]
        )

    accuracy_ci = np.percentile(
        accuracy_differences, [2.5, 97.5]
    )

    f1_ci = np.percentile(
        f1_differences, [2.5, 97.5]
    )

    print("\n=== MODEL COMPARISON ===")
    print(f"Trading days: {n_days}")
    print(f"Validation windows: {len(y)}")
    print(f"CNN1D Accuracy: {cnn_scores['accuracy']:.4f}")
    print(
        f"Logistic Accuracy: "
        f"{logistic_scores['accuracy']:.4f}"
    )

    print("\n=== PAIRED BOOTSTRAP RESULTS ===")
    print(
        f"Accuracy difference: "
        f"{observed_accuracy_pp:+.4f} pp"
    )
    print(
        f"95% Accuracy CI: "
        f"[{accuracy_ci[0]:+.4f}, "
        f"{accuracy_ci[1]:+.4f}] pp"
    )

    print(
        f"Macro-F1 difference: "
        f"{observed_macro_f1:+.4f}"
    )
    print(
        f"95% Macro-F1 CI: "
        f"[{f1_ci[0]:+.4f}, "
        f"{f1_ci[1]:+.4f}]"
    )

    print(
        "Accuracy CI includes zero:",
        bool(accuracy_ci[0] <= 0 <= accuracy_ci[1]),
    )

    report = {
        "project": "AURUM-AI",
        "comparison": "CNN1D vs Logistic Regression",
        "evaluation": "validation",
        "trading_days": int(n_days),
        "windows": int(len(y)),
        "block_days": BLOCK_DAYS,
        "repetitions": REPETITIONS,
        "cnn1d_metrics": cnn_scores,
        "logistic_metrics": logistic_scores,
        "accuracy_difference_pp":
            float(observed_accuracy_pp),
        "accuracy_ci_95_pp": accuracy_ci.tolist(),
        "macro_f1_difference":
            float(observed_macro_f1),
        "macro_f1_ci_95": f1_ci.tolist(),
        "test_dataset_used": False,
        "limitations": [
            "Exploratory validation comparison.",
            "Models were selected using validation.",
            "Overlapping targets create dependence.",
            "Interval does not correct for model selection.",
            "No financial profitability measured.",
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
    print("\nPAIRED BOOTSTRAP COMPLETED")


if __name__ == "__main__":
    main()
