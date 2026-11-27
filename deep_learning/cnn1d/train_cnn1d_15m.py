
from pathlib import Path
import json
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_cnn1d import CNN1DForecaster


# ========================================================
# AURUM-AI — CNN1D 15-MINUTE FORECASTING
# New model, same architecture as original CNN1D
# TRAIN + VALIDATION ONLY
# ========================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "05_model_data"

TRAIN_PATH = DATA_DIR / "train_15m_normalized.npz"
VAL_PATH = DATA_DIR / "validation_15m_normalized.npz"

MODEL_PATH = (
    ROOT / "artifacts" / "models" / "cnn1d_15m_best.pt"
)

REPORT_PATH = (
    ROOT / "reports" / "cnn1d_15m_validation.json"
)

BATCH_SIZE = 256
MAX_EPOCHS = 10
PATIENCE = 3
LEARNING_RATE = 0.001
TARGET_SCALE = 1000.0
SEED = 42


def load_dataset(path):

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:

        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32)

        feature_names = data["feature_names"].tolist()

        end_ts = data[
            "end_timestamp_ms"
        ].astype(np.int64)

        target_ts = data[
            "target_timestamp_ms"
        ].astype(np.int64)

    if X.ndim != 3 or X.shape[1:] != (20, 6):
        raise ValueError("Expected windows of shape (N,20,6)")

    if len(X) != len(y):
        raise ValueError("X/y length mismatch")

    if not np.isfinite(X).all():
        raise ValueError("Invalid input features")

    if not np.isfinite(y).all():
        raise ValueError("Invalid prediction targets")

    # 15 minutes = 900,000 milliseconds
    if not np.all(target_ts - end_ts == 900000):
        raise ValueError("Invalid 15-minute target alignment")

    if len(np.unique(end_ts)) != len(end_ts):
        raise ValueError("Duplicate window timestamps")

    return X, y, feature_names, end_ts


def calculate_metrics(y_true, y_pred):

    error = y_true - y_pred

    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
    }


def predict(model, X):

    model.eval()
    outputs = []

    with torch.no_grad():

        for start in range(0, len(X), 512):

            batch = torch.from_numpy(
                X[start:start + 512]
            )

            predicted_scaled = model(batch)

            predicted_returns = (
                predicted_scaled.cpu().numpy()
                / TARGET_SCALE
            )

            outputs.append(predicted_returns)

    return np.concatenate(outputs).astype(np.float64)


def main():

    print("\n=== AURUM-AI | CNN1D 15M TRAINING ===")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    torch.set_num_threads(
        min(4, os.cpu_count() or 1)
    )

    # Preserve previous experiment artifacts
    for output in (MODEL_PATH, REPORT_PATH):
        if output.exists():
            raise FileExistsError(
                f"Existing artifact will not be overwritten: {output}"
            )

    X_train, y_train, train_features, train_ts = (
        load_dataset(TRAIN_PATH)
    )

    X_val, y_val, val_features, val_ts = (
        load_dataset(VAL_PATH)
    )

    if train_features != val_features:
        raise ValueError("Feature order mismatch")

    if train_ts.max() >= val_ts.min():
        raise ValueError("Chronological split violation")

    print("\nDevice: CPU")
    print(f"Train: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print("Forecast horizon: 15 minutes")

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train * TARGET_SCALE),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    # Fresh model, no pretrained weights
    model = CNN1DForecaster(
        input_size=X_train.shape[2]
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001,
    )

    criterion = nn.SmoothL1Loss()

    MODEL_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    REPORT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    best_mae = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = []

    print("\nTraining CNN1D for 15-minute returns...")

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()

        total_loss = 0.0

        for xb, yb in train_loader:

            optimizer.zero_grad()

            predictions = model(xb)

            loss = criterion(predictions, yb)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_loss += loss.item() * len(xb)

        train_loss = total_loss / len(X_train)

        val_predictions = predict(
            model, X_val
        )

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
                "forecast_horizon_minutes": 15,
            }, MODEL_PATH)

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("\nEarly stopping activated.")
            break

    # Restore best checkpoint
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    final_predictions = predict(
        model, X_val
    )

    actual = y_val.astype(np.float64)

    cnn_metrics = calculate_metrics(
        actual,
        final_predictions,
    )

    zero_metrics = calculate_metrics(
        actual,
        np.zeros_like(actual),
    )

    improvement = (
        (zero_metrics["MAE"] - cnn_metrics["MAE"])
        / zero_metrics["MAE"]
        * 100
    )

    print("\n=== FINAL VALIDATION RESULTS — 15M ===")

    print("\nZero Return 15m")
    print(f"MAE:  {zero_metrics['MAE']:.8f}")
    print(f"RMSE: {zero_metrics['RMSE']:.8f}")

    print("\nCNN1D 15m")
    print(f"MAE:  {cnn_metrics['MAE']:.8f}")
    print(f"RMSE: {cnn_metrics['RMSE']:.8f}")

    print(f"\nMAE improvement: {improvement:+.6f}%")
    print(f"Best epoch: {best_epoch}")

    report = {
        "project": "AURUM-AI",
        "model": "CNN1D",
        "forecast_horizon_minutes": 15,
        "evaluation_split": "validation",
        "target": "next_15min_log_return",
        "training_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "window_size": 20,
        "feature_names": train_features,
        "best_epoch": best_epoch,
        "validation_metrics": cnn_metrics,
        "zero_return_validation": zero_metrics,
        "mae_improvement_pct": float(improvement),
        "training_history": history,
        "test_dataset_used": False,
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(report, file, indent=2)

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Report saved: {REPORT_PATH}")

    print("\nOriginal 5m model was NOT modified.")
    print("Test dataset was NOT used.")
    print("\nCNN1D 15M TRAINING COMPLETED")


if __name__ == "__main__":
    main()
