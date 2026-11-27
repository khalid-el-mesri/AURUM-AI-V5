
from pathlib import Path
import json
import os

import numpy as np
import torch

from train_cnn1d import CNN1DForecaster


# AURUM-AI — CNN1D SIGNAL DISTRIBUTION
# Validation only. No trades executed.

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "data" / "05_model_data"
    / "validation_normalized.npz"
)

MODEL_PATH = (
    ROOT / "artifacts" / "models" / "cnn1d_best.pt"
)

REPORT_PATH = (
    ROOT / "reports"
    / "cnn1d_signal_distribution_validation.json"
)

# Exploratory thresholds, not measured trading costs
THRESHOLDS = [0.0001, 0.0003, 0.0005]


def main():
    print("\n=== AURUM-AI | CNN1D SIGNAL ANALYSIS ===")

    torch.set_num_threads(min(4, os.cpu_count() or 1))

    with np.load(DATA_PATH, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        feature_names = data["feature_names"].tolist()

    if not np.isfinite(X).all():
        raise ValueError("Invalid validation features.")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["feature_names"] != feature_names:
        raise ValueError("Feature mismatch.")

    if checkpoint["window_size"] != X.shape[1]:
        raise ValueError("Window size mismatch.")

    model = CNN1DForecaster(
        input_size=X.shape[2]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    scale = float(checkpoint["target_scale"])

    if scale <= 0:
        raise ValueError("Invalid target scale.")

    predictions = []

    with torch.no_grad():
        for start in range(0, len(X), 512):
            batch = torch.from_numpy(
                X[start:start + 512]
            )

            output = model(batch).cpu().numpy()
            predictions.append(output / scale)

    predicted_returns = np.concatenate(predictions)

    if not np.isfinite(predicted_returns).all():
        raise ValueError("Invalid model predictions.")

    print(f"\nValidation windows: {len(predicted_returns)}")
    print(f"Mean prediction: {predicted_returns.mean():.10f}")
    print(f"Min prediction:  {predicted_returns.min():.10f}")
    print(f"Max prediction:  {predicted_returns.max():.10f}")

    percentiles = np.percentile(
        np.abs(predicted_returns),
        [50, 90, 95, 99],
    )

    print("\nABSOLUTE PREDICTION PERCENTILES")

    for percentile, value in zip(
        [50, 90, 95, 99], percentiles
    ):
        print(
            f"P{percentile}: {value:.10f} "
            f"({value * 100:.6f}%)"
        )

    results = []

    print("\n=== HYPOTHETICAL SIGNAL COUNTS ===")

    for threshold in THRESHOLDS:
        buy = int(
            np.sum(predicted_returns > threshold)
        )
        sell = int(
            np.sum(predicted_returns < -threshold)
        )
        hold = len(predicted_returns) - buy - sell

        total = len(predicted_returns)

        print(f"\nThreshold: {threshold * 100:.4f}%")
        print(f"BUY:  {buy} ({100 * buy / total:.2f}%)")
        print(f"SELL: {sell} ({100 * sell / total:.2f}%)")
        print(f"HOLD: {hold} ({100 * hold / total:.2f}%)")

        results.append({
            "threshold_log_return": threshold,
            "BUY": buy,
            "SELL": sell,
            "HOLD": hold,
        })

    report = {
        "project": "AURUM-AI",
        "model": "CNN1D",
        "evaluation_split": "validation",
        "total_windows": len(predicted_returns),
        "prediction_percentiles": {
            str(p): float(v)
            for p, v in zip([50, 90, 95, 99], percentiles)
        },
        "threshold_analysis": results,
        "trades_executed": False,
        "costs_estimated": False,
        "test_dataset_used": False,
    }

    REPORT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    with open(
        REPORT_PATH, "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    print(f"\nReport saved: {REPORT_PATH}")
    print("Test dataset was NOT used.")
    print("\nSIGNAL ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()
