
from pathlib import Path
import json
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# AURUM-AI — LIGHTWEIGHT TRANSFORMER

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "05_model_data"

MODEL_PATH = ROOT / "artifacts" / "models" / "transformer_best.pt"
REPORT_PATH = ROOT / "reports" / "transformer_validation.json"

D_MODEL = 24
N_HEADS = 4
N_LAYERS = 1
BATCH_SIZE = 256
MAX_EPOCHS = 10
PATIENCE = 3
TARGET_SCALE = 1000.0
SEED = 42


def load_data(split):
    path = DATA_DIR / f"{split}_normalized.npz"

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32)
        features = data["feature_names"].tolist()

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"Invalid data in {split}")

    return X, y, features


class TransformerForecaster(nn.Module):
    def __init__(self, input_size, window_size):
        super().__init__()

        self.window_size = window_size

        self.input_projection = nn.Linear(
            input_size, D_MODEL
        )

        self.position_embedding = nn.Embedding(
            window_size, D_MODEL
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=48,
            dropout=0.1,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=N_LAYERS,
        )

        self.output = nn.Linear(D_MODEL, 1)

    def forward(self, x):
        batch_size, window_size, _ = x.shape

        if window_size != self.window_size:
            raise ValueError("Unexpected window size")

        positions = torch.arange(
            window_size, device=x.device
        )

        x = (
            self.input_projection(x)
            + self.position_embedding(positions)
        )

        # True means attention is blocked.
        # Tokens cannot attend to later time steps.
        causal_mask = torch.triu(
            torch.ones(
                window_size,
                window_size,
                dtype=torch.bool,
                device=x.device,
            ),
            diagonal=1,
        )

        x = self.encoder(x, mask=causal_mask)

        return self.output(x[:, -1, :]).squeeze(-1)


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
    print("\n=== AURUM-AI | TRANSFORMER TRAINING ===")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    torch.set_num_threads(
        min(4, os.cpu_count() or 1)
    )

    X_train, y_train, train_features = load_data("train")
    X_val, y_val, val_features = load_data("validation")

    if train_features != val_features:
        raise ValueError("Feature order mismatch")

    if X_train.shape[1:] != X_val.shape[1:]:
        raise ValueError("Window dimensions mismatch")

    print("\nDevice: CPU")
    print(f"Train: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"D_MODEL: {D_MODEL}")
    print(f"Attention heads: {N_HEADS}")

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

    model = TransformerForecaster(
        input_size=X_train.shape[2],
        window_size=X_train.shape[1],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0005,
        weight_decay=0.0001,
    )

    criterion = nn.SmoothL1Loss()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_mae = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    print("\nTraining Transformer...")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()

            output = model(xb)
            loss = criterion(output, yb)

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
                "window_size": int(X_train.shape[1]),
                "d_model": D_MODEL,
                "n_heads": N_HEADS,
                "n_layers": N_LAYERS,
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

    final_predictions = predict(
        model, validation_loader
    )

    transformer_metrics = calculate_metrics(
        y_val.astype(np.float64),
        final_predictions.astype(np.float64),
    )

    zero_metrics = calculate_metrics(
        y_val.astype(np.float64),
        np.zeros_like(y_val, dtype=np.float64),
    )

    print("\n=== FINAL VALIDATION RESULTS ===")

    print("\nZero Return")
    print(f"MAE:  {zero_metrics['MAE']:.8f}")
    print(f"RMSE: {zero_metrics['RMSE']:.8f}")

    print("\nTransformer")
    print(f"MAE:  {transformer_metrics['MAE']:.8f}")
    print(f"RMSE: {transformer_metrics['RMSE']:.8f}")

    print(f"\nBest epoch: {best_epoch}")

    if transformer_metrics["MAE"] < zero_metrics["MAE"]:
        print("Transformer beats Zero Return on validation MAE.")
    else:
        print("Zero Return remains better on validation MAE.")

    report = {
        "model": "Transformer",
        "evaluation_split": "validation",
        "target": "next_5min_log_return",
        "device": "cpu",
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "window_size": int(X_train.shape[1]),
        "feature_names": train_features,
        "target_scale": TARGET_SCALE,
        "best_epoch": best_epoch,
        "validation_metrics": transformer_metrics,
        "zero_return_validation": zero_metrics,
        "training_history": history,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Report saved: {REPORT_PATH}")
    print("Test dataset was NOT used.")
    print("\nTRANSFORMER TRAINING COMPLETED")


if __name__ == "__main__":
    main()
