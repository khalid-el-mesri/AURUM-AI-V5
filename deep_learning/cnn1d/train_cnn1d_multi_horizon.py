
from pathlib import Path
import argparse
import json
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_cnn1d import CNN1DForecaster


# =====================================================
# AURUM-AI — CNN1D MULTI-HORIZON TRAINING
# Supported horizons: 30m, 60m
# Train + Validation only
# =====================================================

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "05_model_data"

BATCH_SIZE = 256
MAX_EPOCHS = 10
PATIENCE = 3
LEARNING_RATE = 0.001
TARGET_SCALE = 1000.0
SEED = 42


def load_data(split, horizon):
    path = DATA_DIR / f"{split}_{horizon}m_normalized.npz"

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32)
        features = data["feature_names"].tolist()
        end_ts = data["end_timestamp_ms"].astype(np.int64)
        target_ts = data["target_timestamp_ms"].astype(np.int64)

    if X.ndim != 3 or X.shape[1:] != (20, 6):
        raise ValueError("Invalid input window dimensions.")

    if len(X) != len(y) or len(y) != len(end_ts):
        raise ValueError("Dataset length mismatch.")

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite values detected.")

    if not np.all(target_ts - end_ts == horizon * 60_000):
        raise ValueError("Target timestamps are not aligned.")

    if len(np.unique(end_ts)) != len(end_ts):
        raise ValueError("Duplicate window timestamps.")

    return X, y, features, end_ts


def calculate_metrics(actual, predicted):
    error = actual - predicted

    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
    }


def predict(model, X):
    model.eval()
    outputs = []

    with torch.no_grad():
        for start in range(0, len(X), 512):
            batch = torch.from_numpy(X[start:start + 512])
            result = model(batch).cpu().numpy()
            outputs.append(result / TARGET_SCALE)

    return np.concatenate(outputs).astype(np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--horizon",
        type=int,
        choices=[30, 60],
        required=True,
    )

    args = parser.parse_args()
    horizon = args.horizon

    model_path = (
        ROOT / "artifacts" / "models"
        / f"cnn1d_{horizon}m_best.pt"
    )

    report_path = (
        ROOT / "reports"
        / f"cnn1d_{horizon}m_validation.json"
    )

    print(
        f"\n=== AURUM-AI | CNN1D {horizon}M TRAINING ==="
    )

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    # Never overwrite an existing experiment.
    if model_path.exists() or report_path.exists():
        raise FileExistsError(
            "An experiment already exists for this horizon. "
            "Existing artifacts were preserved."
        )

    X_train, y_train, train_features, train_ts = load_data(
        "train", horizon
    )

    X_val, y_val, val_features, val_ts = load_data(
        "validation", horizon
    )

    if train_features != val_features:
        raise ValueError("Feature order mismatch.")

    if X_train.shape[1:] != X_val.shape[1:]:
        raise ValueError("Window shape mismatch.")

    if train_ts.max() >= val_ts.min():
        raise ValueError("Chronological split violation.")

    print("\nDevice: CPU")
    print(f"Train: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Forecast horizon: {horizon} minutes")

    dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train * TARGET_SCALE),
    )

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    model = CNN1DForecaster(input_size=X_train.shape[2])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001,
    )

    criterion = nn.SmoothL1Loss()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    best_mae = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    print("\nTraining CNN1D...")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()

            output = model(xb)
            loss = criterion(output, yb)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()
            total_loss += loss.item() * len(xb)

        train_loss = total_loss / len(X_train)

        val_predictions = predict(model, X_val)

        scores = calculate_metrics(
            y_val.astype(np.float64),
            val_predictions,
        )

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MAE: {scores['MAE']:.8f} | "
            f"Val RMSE: {scores['RMSE']:.8f}",
            flush=True,
        )

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "validation_mae": scores["MAE"],
            "validation_rmse": scores["RMSE"],
        })

        if scores["MAE"] < best_mae:
            best_mae = scores["MAE"]
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "input_size": int(X_train.shape[2]),
                "window_size": int(X_train.shape[1]),
                "feature_names": train_features,
                "target_scale": TARGET_SCALE,
                "forecast_horizon_minutes": horizon,
            }, model_path)

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("\nEarly stopping activated.")
            break

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    actual = y_val.astype(np.float64)
    final_predictions = predict(model, X_val)

    cnn_scores = calculate_metrics(actual, final_predictions)
    zero_scores = calculate_metrics(
        actual, np.zeros_like(actual)
    )

    improvement = (
        (zero_scores["MAE"] - cnn_scores["MAE"])
        / zero_scores["MAE"]
        * 100
    )

    print(
        f"\n=== FINAL VALIDATION RESULTS — {horizon}M ==="
    )

    print("\nZero Return")
    print(f"MAE:  {zero_scores['MAE']:.8f}")
    print(f"RMSE: {zero_scores['RMSE']:.8f}")

    print("\nCNN1D")
    print(f"MAE:  {cnn_scores['MAE']:.8f}")
    print(f"RMSE: {cnn_scores['RMSE']:.8f}")

    print(f"\nMAE improvement: {improvement:+.6f}%")
    print(f"Best epoch: {best_epoch}")

    report = {
        "project": "AURUM-AI",
        "model": "CNN1D",
        "forecast_horizon_minutes": horizon,
        "evaluation_split": "validation",
        "target": f"next_{horizon}min_log_return",
        "train_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "best_epoch": best_epoch,
        "cnn1d_validation": cnn_scores,
        "zero_return_validation": zero_scores,
        "mae_improvement_pct": float(improvement),
        "training_history": history,
        "test_dataset_used": False,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nModel saved: {model_path}")
    print(f"Report saved: {report_path}")
    print("Test dataset was NOT used.")
    print("\nMULTI-HORIZON TRAINING COMPLETED")


if __name__ == "__main__":
    main()
