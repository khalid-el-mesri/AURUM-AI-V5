
from pathlib import Path
import json
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# AURUM-AI — LSTM TRAINING ON CPU
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "05_model_data"

MODEL_PATH = (
    PROJECT_ROOT / "artifacts" / "models" / "lstm_best.pt"
)

REPORT_PATH = (
    PROJECT_ROOT / "reports" / "lstm_validation.json"
)

# Light architecture for CPU training
HIDDEN_SIZE = 32
BATCH_SIZE = 256
MAX_EPOCHS = 10
PATIENCE = 3
LEARNING_RATE = 0.001

# Scale targets during training for numerical stability
# Predictions are converted back before evaluation
TARGET_SCALE = 1000.0

SEED = 42


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Avoid excessive CPU thread usage
    torch.set_num_threads(
        min(4, os.cpu_count() or 1)
    )


def load_data(name):
    path = DATA_DIR / f"{name}_normalized.npz"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    with np.load(path, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32)
        features = data["feature_names"].tolist()

    if not np.isfinite(X).all():
        raise ValueError(f"Invalid X in {name}")

    if not np.isfinite(y).all():
        raise ValueError(f"Invalid y in {name}")

    return X, y, features


class LSTMForecaster(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
        )

        self.output = nn.Linear(
            HIDDEN_SIZE,
            1,
        )

    def forward(self, x):
        sequence_output, _ = self.lstm(x)

        last_step = sequence_output[:, -1, :]

        prediction = self.output(last_step)

        return prediction.squeeze(-1)


def calculate_metrics(y_true, y_pred):
    errors = y_true - y_pred

    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(
            np.sqrt(np.mean(errors ** 2))
        ),
    }


def predict(model, loader):
    model.eval()

    predictions = []

    with torch.no_grad():
        for X_batch, _ in loader:
            output = model(X_batch)

            # Convert back to original return units
            output = output / TARGET_SCALE

            predictions.append(
                output.cpu().numpy()
            )

    return np.concatenate(predictions)


def main():

    print("\n=== AURUM-AI | LSTM TRAINING ===")

    set_seed()

    device = torch.device("cpu")

    print(f"\nDevice: {device}")
    print(f"PyTorch: {torch.__version__}")

    # 1. Load TRAIN and VALIDATION only
    X_train, y_train, features_train = load_data(
        "train"
    )

    X_val, y_val, features_val = load_data(
        "validation"
    )

    if features_train != features_val:
        raise ValueError("Feature order mismatch.")

    if X_train.shape[1:] != X_val.shape[1:]:
        raise ValueError("Window dimensions mismatch.")

    print(f"\nTrain shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")

    # 2. Convert arrays into PyTorch tensors
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train * TARGET_SCALE),
    )

    validation_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val * TARGET_SCALE),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # 3. Initialize LSTM
    model = LSTMForecaster(
        input_size=X_train.shape[2]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001,
    )

    criterion = nn.SmoothL1Loss()

    print("\nMODEL CONFIGURATION")
    print(f"Hidden size: {HIDDEN_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max epochs: {MAX_EPOCHS}")
    print(f"Early stopping patience: {PATIENCE}")
    print(f"Target training scale: {TARGET_SCALE}")

    # 4. Train with early stopping
    best_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    MODEL_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    history = []

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:

            optimizer.zero_grad()

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch,
            )

            loss.backward()

            # Limit exploding gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_loss += (
                loss.item() * len(X_batch)
            )

        train_loss = total_loss / len(train_dataset)

        # Evaluate on validation
        val_predictions = predict(
            model,
            validation_loader,
        )

        metrics = calculate_metrics(
            y_val.astype(np.float64),
            val_predictions.astype(np.float64),
        )

        val_mae = metrics["MAE"]

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MAE: {val_mae:.8f} | "
            f"Val RMSE: {metrics['RMSE']:.8f}",
            flush=True,
        )

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "validation_mae": val_mae,
            "validation_rmse": metrics["RMSE"],
        })

        # Save best validation checkpoint
        if val_mae < best_mae:

            best_mae = val_mae
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": X_train.shape[2],
                    "hidden_size": HIDDEN_SIZE,
                    "window_size": X_train.shape[1],
                    "target_scale": TARGET_SCALE,
                    "feature_names": features_train,
                },
                MODEL_PATH,
            )

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping activated.")
            break

    # 5. Reload best model
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    best_predictions = predict(
        model,
        validation_loader,
    )

    lstm_metrics = calculate_metrics(
        y_val.astype(np.float64),
        best_predictions.astype(np.float64),
    )

    zero_metrics = calculate_metrics(
        y_val.astype(np.float64),
        np.zeros_like(y_val, dtype=np.float64),
    )

    # 6. Final comparison
    print("\n=== FINAL VALIDATION RESULTS ===")

    print("\nZero Return")
    print(f"MAE:  {zero_metrics['MAE']:.8f}")
    print(f"RMSE: {zero_metrics['RMSE']:.8f}")

    print("\nLSTM")
    print(f"MAE:  {lstm_metrics['MAE']:.8f}")
    print(f"RMSE: {lstm_metrics['RMSE']:.8f}")

    print(f"\nBest epoch: {best_epoch}")

    if lstm_metrics["MAE"] < zero_metrics["MAE"]:
        print("\nLSTM beats Zero Return on validation MAE.")
    else:
        print("\nZero Return remains better on validation MAE.")

    # 7. Save evaluation report
    REPORT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    report = {
        "model": "LSTM",
        "device": "cpu",
        "evaluation_split": "validation",
        "target": "next_5min_log_return",
        "window_size": int(X_train.shape[1]),
        "feature_names": features_train,
        "hidden_size": HIDDEN_SIZE,
        "target_scale": TARGET_SCALE,
        "best_epoch": best_epoch,
        "validation_metrics": lstm_metrics,
        "zero_return_validation": zero_metrics,
        "training_history": history,
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=2)

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Report saved: {REPORT_PATH}")

    print("\nTest dataset was NOT used.")
    print("LSTM TRAINING COMPLETED")


if __name__ == "__main__":
    main()
