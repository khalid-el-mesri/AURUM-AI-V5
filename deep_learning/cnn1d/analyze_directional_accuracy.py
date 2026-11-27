
from pathlib import Path
import json
import os

import numpy as np
import torch

from train_cnn1d import CNN1DForecaster


# AURUM-AI — DIRECTIONAL ACCURACY DIAGNOSTIC
# Validation only. No trading or retraining.

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "data" / "05_model_data"
    / "validation_normalized.npz"
)

MODEL_PATH = (
    ROOT / "artifacts" / "models" / "cnn1d_best.pt"
)

REPORT_PATH = (
    ROOT / "reports" / "cnn1d_directional_validation.json"
)


def main():
    print("\n=== AURUM-AI | DIRECTIONAL ANALYSIS ===")

    torch.set_num_threads(min(4, os.cpu_count() or 1))

    with np.load(DATA_PATH, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float64)
        feature_names = data["feature_names"].tolist()

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite validation data.")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["feature_names"] != feature_names:
        raise ValueError("Feature names do not match.")

    if checkpoint["window_size"] != X.shape[1]:
        raise ValueError("Window size does not match.")

    if checkpoint["input_size"] != X.shape[2]:
        raise ValueError("Input size does not match.")

    model = CNN1DForecaster(
        input_size=X.shape[2]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    scale = float(checkpoint["target_scale"])

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid target scale.")

    predictions = []

    with torch.no_grad():
        for start in range(0, len(X), 512):
            batch = torch.from_numpy(
                X[start:start + 512]
            )

            output = model(batch).cpu().numpy()

            predictions.append(
                output.astype(np.float64) / scale
            )

    y_pred = np.concatenate(predictions)

    if not np.isfinite(y_pred).all():
        raise ValueError("Invalid predictions.")

    # Exclude zero actual returns:
    # they have no upward or downward direction.
    mask = y != 0

    actual_sign = np.sign(y[mask])
    predicted_sign = np.sign(y_pred[mask])

    correct = predicted_sign == actual_sign

    directional_accuracy = float(
        np.mean(correct) * 100
    )

    positive_count = int(np.sum(actual_sign > 0))
    negative_count = int(np.sum(actual_sign < 0))

    # Majority-direction baseline on the same examples.
    majority_accuracy = (
        max(positive_count, negative_count)
        / len(actual_sign)
        * 100
    )

    if np.std(y_pred) > 0 and np.std(y) > 0:
        correlation = float(
            np.corrcoef(y, y_pred)[0, 1]
        )
    else:
        correlation = None

    # Diagnostic on the 10% largest predicted magnitudes.
    cutoff = np.percentile(np.abs(y_pred), 90)

    strong_mask = mask & (np.abs(y_pred) >= cutoff)

    strong_count = int(np.sum(strong_mask))

    strong_accuracy = (
        float(
            np.mean(
                np.sign(y_pred[strong_mask])
                == np.sign(y[strong_mask])
            ) * 100
        )
        if strong_count > 0
        else None
    )

    print(f"\nTotal validation windows: {len(y)}")
    print(f"Nonzero actual returns: {int(mask.sum())}")
    print(f"Excluded zero returns: {int((~mask).sum())}")

    print("\n=== DIRECTIONAL RESULTS ===")

    print(
        f"CNN1D directional accuracy: "
        f"{directional_accuracy:.4f}%"
    )

    print(
        f"Majority-direction baseline: "
        f"{majority_accuracy:.4f}%"
    )

    print(
        f"Difference: "
        f"{directional_accuracy - majority_accuracy:+.4f} pp"
    )

    print(
        f"Pearson correlation: {correlation}"
    )

    print("\n=== TOP 10% PREDICTION MAGNITUDES ===")

    print(f"Minimum magnitude: {cutoff:.10f}")
    print(f"Nonzero-return samples: {strong_count}")

    if strong_accuracy is not None:
        print(
            f"Directional accuracy: "
            f"{strong_accuracy:.4f}%"
        )

    report = {
        "project": "AURUM-AI",
        "model": "CNN1D",
        "split": "validation",
        "total_windows": int(len(y)),
        "nonzero_return_windows": int(mask.sum()),
        "directional_accuracy_pct": directional_accuracy,
        "majority_baseline_accuracy_pct": majority_accuracy,
        "pearson_correlation": correlation,
        "top_10pct_prediction_magnitude_accuracy_pct":
            strong_accuracy,
        "top_10pct_nonzero_samples": strong_count,
        "test_dataset_used": False,
        "trades_executed": False,
    }

    REPORT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"\nReport saved: {REPORT_PATH}")
    print("Test dataset was NOT used.")
    print("\nDIRECTIONAL ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()
