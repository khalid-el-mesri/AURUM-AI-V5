
from pathlib import Path
import json
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# AURUM-AI — CNN1D MODEL

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "05_model_data"

MODEL_PATH = ROOT / "artifacts" / "models" / "cnn1d_best.pt"
REPORT_PATH = ROOT / "reports" / "cnn1d_validation.json"

BATCH_SIZE = 256
MAX_EPOCHS = 10
PATIENCE = 3
TARGET_SCALE = 1000.0
SEED = 42


def load_data(name):
    path = DATA_DIR / f"{name}_normalized.npz"

    with np.load(path, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32)
        features = data["feature_names"].tolist()

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"Invalid data in {name}")

    return X, y, features


class CNN1DForecaster(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # (batch, time, features) -> (batch, features, time)
        x = x.transpose(1, 2)
        return self.network(x).squeeze(-1)


def metrics(y_true, y_pred):
    error = y_true - y_pred

    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
    }


def predict(model, loader):
    model.eval()
    results = []

    with torch.no_grad():
        for (xb,) in loader:
            predictions = model(xb) / TARGET_SCALE
            results.append(predictions.numpy())

    return np.concatenate(results)


def main():
    print("\n=== AURUM-AI | CNN1D TRAINING ===")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    X_train, y_train, features_train = load_data("train")
    X_val, y_val, features_val = load_data("validation")

    if features_train != features_val:
        raise ValueError("Feature order mismatch")

    if X_train.shape[1:] != X_val.shape[1:]:
        raise ValueError("Window shape mismatch")

    print("\nDevice: CPU")
    print(f"Train: {X_train.shape}")
    print(f"Validation: {X_val.shape}")

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train * TARGET_SCALE),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = CNN1DForecaster(input_size=X_train.shape[2])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001,
    )

    criterion = nn.SmoothL1Loss()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

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

        val_predictions = predict(model, val_loader)
        scores = metrics(
            y_val.astype(np.float64),
            val_predictions.astype(np.float64),
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
                "input_size": X_train.shape[2],
                "window_size": X_train.shape[1],
                "target_scale": TARGET_SCALE,
                "feature_names": features_train,
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

    model.load_state_dict(checkpoint["model_state_dict"])

    final_predictions = predict(model, val_loader)

    cnn_metrics = metrics(y_val, final_predictions)
    zero_metrics = metrics(y_val, np.zeros_like(y_val))

    print("\n=== FINAL VALIDATION RESULTS ===")

    print("\nZero Return")
    print(f"MAE:  {zero_metrics['MAE']:.8f}")
    print(f"RMSE: {zero_metrics['RMSE']:.8f}")

    print("\nCNN1D")
    print(f"MAE:  {cnn_metrics['MAE']:.8f}")
    print(f"RMSE: {cnn_metrics['RMSE']:.8f}")

    print(f"\nBest epoch: {best_epoch}")

    if cnn_metrics["MAE"] < zero_metrics["MAE"]:
        print("CNN1D beats Zero Return on validation MAE.")
    else:
        print("Zero Return remains better on validation MAE.")

    report = {
        "model": "CNN1D",
        "evaluation_split": "validation",
        "target": "next_5min_log_return",
        "best_epoch": best_epoch,
        "validation_metrics": cnn_metrics,
        "zero_return_validation": zero_metrics,
        "training_history": history,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Report saved: {REPORT_PATH}")
    print("Test dataset was NOT used.")
    print("\nCNN1D TRAINING COMPLETED")


if __name__ == "__main__":
    main()
