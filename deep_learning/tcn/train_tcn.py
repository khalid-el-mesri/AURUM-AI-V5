
from pathlib import Path
import json
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# AURUM-AI — TCN TRAINING ON CPU

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "05_model_data"

MODEL_PATH = ROOT / "artifacts" / "models" / "tcn_best.pt"
REPORT_PATH = ROOT / "reports" / "tcn_validation.json"

BATCH_SIZE = 256
MAX_EPOCHS = 10
PATIENCE = 3
TARGET_SCALE = 1000.0
SEED = 42
CHANNELS = 16


def load_data(name):
    path = DATA_DIR / f"{name}_normalized.npz"

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32)
        feature_names = data["feature_names"].tolist()

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"Invalid data: {name}")

    return X, y, feature_names


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()

        self.padding = 2 * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            dilation=dilation,
            padding=self.padding,
        )

    def forward(self, x):
        x = self.conv(x)

        # Remove right padding to preserve causality
        return x[:, :, :-self.padding]


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()

        self.network = nn.Sequential(
            CausalConv1d(in_channels, out_channels, dilation),
            nn.ReLU(),
            CausalConv1d(out_channels, out_channels, dilation),
            nn.ReLU(),
        )

        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        return torch.relu(self.network(x) + self.skip(x))


class TCNForecaster(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            TCNBlock(input_size, CHANNELS, dilation=1),
            TCNBlock(CHANNELS, CHANNELS, dilation=2),
            TCNBlock(CHANNELS, CHANNELS, dilation=4),
        )

        self.output = nn.Linear(CHANNELS, 1)

    def forward(self, x):
        # (batch, time, features) -> (batch, features, time)
        x = x.transpose(1, 2)

        x = self.network(x)

        # Predict from the final time step
        x = x[:, :, -1]

        return self.output(x).squeeze(-1)


def calculate_metrics(y_true, y_pred):
    error = y_true - y_pred

    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
    }


def predict(model, loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for (xb,) in loader:
            output = model(xb) / TARGET_SCALE
            predictions.append(output.cpu().numpy())

    return np.concatenate(predictions)


def main():
    print("\n=== AURUM-AI | TCN TRAINING ===")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    X_train, y_train, train_features = load_data("train")
    X_val, y_val, val_features = load_data("validation")

    if train_features != val_features:
        raise ValueError("Feature mismatch.")

    if X_train.shape[1:] != X_val.shape[1:]:
        raise ValueError("Window dimension mismatch.")

    print("\nDevice: CPU")
    print(f"Train: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"TCN channels: {CHANNELS}")

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train * TARGET_SCALE),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = TCNForecaster(X_train.shape[2])

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

    print("\nTraining TCN...")

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
            model, validation_loader
        )

        scores = calculate_metrics(
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
                "input_size": int(X_train.shape[2]),
                "channels": CHANNELS,
                "window_size": int(X_train.shape[1]),
                "target_scale": TARGET_SCALE,
                "feature_names": train_features,
            }, MODEL_PATH)

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("\nEarly stopping activated.")
            break

    # Restore the best model
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    final_predictions = predict(model, validation_loader)

    tcn_metrics = calculate_metrics(y_val, final_predictions)

    zero_metrics = calculate_metrics(
        y_val, np.zeros_like(y_val)
    )

    print("\n=== FINAL VALIDATION RESULTS ===")

    print("\nZero Return")
    print(f"MAE:  {zero_metrics['MAE']:.8f}")
    print(f"RMSE: {zero_metrics['RMSE']:.8f}")

    print("\nTCN")
    print(f"MAE:  {tcn_metrics['MAE']:.8f}")
    print(f"RMSE: {tcn_metrics['RMSE']:.8f}")

    print(f"\nBest epoch: {best_epoch}")

    if tcn_metrics["MAE"] < zero_metrics["MAE"]:
        print("TCN beats Zero Return on validation MAE.")
    else:
        print("Zero Return remains better on validation MAE.")

    report = {
        "model": "TCN",
        "evaluation_split": "validation",
        "device": "cpu",
        "target": "next_5min_log_return",
        "window_size": int(X_train.shape[1]),
        "feature_names": train_features,
        "channels": CHANNELS,
        "best_epoch": best_epoch,
        "validation_metrics": tcn_metrics,
        "zero_return_validation": zero_metrics,
        "training_history": history,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Report saved: {REPORT_PATH}")
    print("Test dataset was NOT used.")
    print("\nTCN TRAINING COMPLETED")


if __name__ == "__main__":
    main()
