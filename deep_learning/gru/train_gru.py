
from pathlib import Path
import json
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# =====================================================
# AURUM-AI — GRU TRAINING ON CPU
# =====================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "05_model_data"
MODEL_PATH = ROOT / "artifacts" / "models" / "gru_best.pt"
REPORT_PATH = ROOT / "reports" / "gru_validation.json"

HIDDEN_SIZE = 32
BATCH_SIZE = 256
MAX_EPOCHS = 10
PATIENCE = 3
LEARNING_RATE = 0.001
TARGET_SCALE = 1000.0
SEED = 42


def load_data(name):
    path = DATA_DIR / f"{name}_normalized.npz"

    with np.load(path, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32)
        features = data["feature_names"].tolist()

    if not np.isfinite(X).all():
        raise ValueError(f"Invalid features in {name}")

    if not np.isfinite(y).all():
        raise ValueError(f"Invalid targets in {name}")

    return X, y, features


class GRUForecaster(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
        )

        self.output = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        sequence_output, _ = self.gru(x)
        last_step = sequence_output[:, -1, :]
        return self.output(last_step).squeeze(-1)


def evaluate(y_true, y_pred):
    return {
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
        "RMSE": float(
            np.sqrt(np.mean((y_true - y_pred) ** 2))
        ),
    }


def predict(model, loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for X_batch, _ in loader:
            outputs = model(X_batch)
            predictions.append(
                (outputs / TARGET_SCALE).numpy()
            )

    return np.concatenate(predictions)


def main():
    print("\n=== AURUM-AI | GRU TRAINING ===")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    X_train, y_train, train_features = load_data("train")
    X_val, y_val, val_features = load_data("validation")

    if train_features != val_features:
        raise ValueError("Feature order mismatch.")

    if X_train.shape[1:] != X_val.shape[1:]:
        raise ValueError("Window dimensions mismatch.")

    print("\nDevice: CPU")
    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train * TARGET_SCALE),
    )

    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val * TARGET_SCALE),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = GRUForecaster(X_train.shape[2])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001,
    )

    criterion = nn.SmoothL1Loss()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_mae = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    print("\nTraining GRU...")

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()
            total_loss += loss.item() * len(X_batch)

        train_loss = total_loss / len(train_dataset)

        val_predictions = predict(model, val_loader)
        scores = evaluate(y_val, val_predictions)

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
                "hidden_size": HIDDEN_SIZE,
                "window_size": int(X_train.shape[1]),
                "target_scale": TARGET_SCALE,
                "feature_names": train_features,
            }, MODEL_PATH)

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("\nEarly stopping activated.")
            break

    # Reload the best checkpoint
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    final_predictions = predict(model, val_loader)

    gru_metrics = evaluate(y_val, final_predictions)
    zero_metrics = evaluate(
        y_val, np.zeros_like(y_val)
    )

    print("\n=== FINAL VALIDATION RESULTS ===")

    print("\nZero Return")
    print(f"MAE:  {zero_metrics['MAE']:.8f}")
    print(f"RMSE: {zero_metrics['RMSE']:.8f}")

    print("\nGRU")
    print(f"MAE:  {gru_metrics['MAE']:.8f}")
    print(f"RMSE: {gru_metrics['RMSE']:.8f}")

    print(f"\nBest epoch: {best_epoch}")

    if gru_metrics["MAE"] < zero_metrics["MAE"]:
        print("GRU beats Zero Return on validation MAE.")
    else:
        print("Zero Return remains better on validation MAE.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "model": "GRU",
        "evaluation_split": "validation",
        "target": "next_5min_log_return",
        "device": "cpu",
        "hidden_size": HIDDEN_SIZE,
        "window_size": int(X_train.shape[1]),
        "feature_names": train_features,
        "target_scale": TARGET_SCALE,
        "best_epoch": best_epoch,
        "validation_metrics": gru_metrics,
        "zero_return_validation": zero_metrics,
        "training_history": history,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Report saved: {REPORT_PATH}")
    print("Test dataset was NOT used.")
    print("\nGRU TRAINING COMPLETED")


if __name__ == "__main__":
    main()
