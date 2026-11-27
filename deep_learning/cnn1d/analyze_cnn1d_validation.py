
from pathlib import Path
import csv
import json
import os

import numpy as np
import torch

from train_cnn1d import CNN1DForecaster


# AURUM-AI — DAILY CNN1D VALIDATION ANALYSIS

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "data" / "05_model_data"
    / "validation_normalized.npz"
)

MODEL_PATH = (
    ROOT / "artifacts" / "models" / "cnn1d_best.pt"
)

CSV_PATH = ROOT / "reports" / "cnn1d_daily_validation.csv"

JSON_PATH = (
    ROOT / "reports" / "cnn1d_daily_validation.json"
)

BATCH_SIZE = 512


def calculate_metrics(actual, predicted):
    error = actual - predicted

    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
    }


def main():

    print("\n=== AURUM-AI | DAILY VALIDATION ANALYSIS ===")

    torch.set_num_threads(min(4, os.cpu_count() or 1))

    # Only the VALIDATION dataset is loaded.
    with np.load(DATA_PATH, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float64)

        features = data["feature_names"].tolist()

        end_ts = data["end_timestamp_ms"]
        target_ts = data["target_timestamp_ms"]

        current_close = data["current_close"]
        next_close = data["next_close"]

    if not np.isfinite(X).all():
        raise ValueError("Invalid features")

    if not np.isfinite(y).all():
        raise ValueError("Invalid target values")

    if not np.all(target_ts - end_ts == 300000):
        raise ValueError(
            "Some targets are not 5 minutes ahead."
        )

    if not np.all(
        np.isfinite(current_close)
        & np.isfinite(next_close)
        & (current_close > 0)
        & (next_close > 0)
    ):
        raise ValueError("Invalid close prices")

    expected_returns = np.log(
        next_close / current_close
    )

    if not np.allclose(
        y, expected_returns, atol=2e-7, rtol=1e-4
    ):
        raise ValueError(
            "Target values do not match close prices."
        )

    print(f"\nValidation windows: {len(y)}")
    print(f"Window shape: {X.shape[1:]}")
    print("Target alignment: verified")

    # Load the saved CNN1D model.
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["feature_names"] != features:
        raise ValueError("Feature order mismatch")

    if checkpoint["window_size"] != X.shape[1]:
        raise ValueError("Window size mismatch")

    model = CNN1DForecaster(
        input_size=X.shape[2]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    target_scale = checkpoint["target_scale"]

    # Predict on validation, without training.
    predictions = []

    with torch.no_grad():
        for start in range(0, len(X), BATCH_SIZE):

            batch = torch.from_numpy(
                X[start:start + BATCH_SIZE]
            )

            output = model(batch)

            predictions.append(
                output.numpy().astype(np.float64)
                / target_scale
            )

    cnn_predictions = np.concatenate(predictions)

    zero_predictions = np.zeros_like(y)

    cnn_metrics = calculate_metrics(
        y, cnn_predictions
    )

    zero_metrics = calculate_metrics(
        y, zero_predictions
    )

    print("\n=== OVERALL VALIDATION ===")

    print(f"Zero Return MAE: {zero_metrics['MAE']:.8f}")
    print(f"CNN1D MAE:       {cnn_metrics['MAE']:.8f}")

    print(f"Zero Return RMSE: {zero_metrics['RMSE']:.8f}")
    print(f"CNN1D RMSE:       {cnn_metrics['RMSE']:.8f}")

    # Convert UTC timestamps into calendar dates.
    # NY regular sessions fall on the same UTC date.
    dates = target_ts.astype(
        "datetime64[ms]"
    ).astype("datetime64[D]")

    unique_dates = np.unique(dates)

    daily_results = []

    for date in unique_dates:

        mask = dates == date

        actual_day = y[mask]
        cnn_day = cnn_predictions[mask]

        cnn_day_metrics = calculate_metrics(
            actual_day, cnn_day
        )

        zero_day_metrics = calculate_metrics(
            actual_day,
            np.zeros_like(actual_day)
        )

        cnn_mae = cnn_day_metrics["MAE"]
        zero_mae = zero_day_metrics["MAE"]

        daily_results.append({
            "date": str(date),
            "windows": int(mask.sum()),
            "cnn_mae": cnn_mae,
            "zero_mae": zero_mae,
            "mae_difference": cnn_mae - zero_mae,
            "cnn_rmse": cnn_day_metrics["RMSE"],
            "zero_rmse": zero_day_metrics["RMSE"],
        })

    better_days = sum(
        row["cnn_mae"] < row["zero_mae"]
        for row in daily_results
    )

    worse_days = sum(
        row["cnn_mae"] > row["zero_mae"]
        for row in daily_results
    )

    equal_days = len(daily_results) - better_days - worse_days

    print("\n=== DAILY STABILITY ===")

    print(f"Trading days analyzed: {len(daily_results)}")
    print(f"Days CNN1D better: {better_days}")
    print(f"Days Zero Return better: {worse_days}")
    print(f"Equal days: {equal_days}")

    if zero_metrics["MAE"] > 0:
        improvement = (
            (zero_metrics["MAE"] - cnn_metrics["MAE"])
            / zero_metrics["MAE"]
        ) * 100
    else:
        improvement = None

    if improvement is not None:
        print(f"Overall MAE improvement: {improvement:.6f}%")

    # Save the daily table.
    CSV_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    with open(
        CSV_PATH, "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(daily_results[0].keys())
        )

        writer.writeheader()
        writer.writerows(daily_results)

    report = {
        "model": "CNN1D",
        "baseline": "Zero Return",
        "evaluation_split": "validation",
        "total_windows": len(y),
        "trading_days": len(daily_results),
        "cnn_better_days": better_days,
        "zero_better_days": worse_days,
        "equal_days": equal_days,
        "cnn_metrics": cnn_metrics,
        "zero_metrics": zero_metrics,
        "overall_mae_improvement_pct": improvement,
        "note": (
            "Exploratory validation diagnostic, "
            "not an independent significance test."
        ),
        "test_dataset_used": False,
    }

    with open(
        JSON_PATH, "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    print("\n=== FILES SAVED ===")
    print(CSV_PATH)
    print(JSON_PATH)

    print("\nTest dataset was NOT used.")
    print("DAILY VALIDATION ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()
