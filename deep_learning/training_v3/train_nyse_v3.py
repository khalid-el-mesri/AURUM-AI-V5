"""AURUM-AI E.6: isolated, validation-only NYSE V3 trainer.

Use:
    python -m deep_learning.training_v3.train_nyse_v3 --model lstm --horizon 5 --dry-run
    python -m deep_learning.training_v3.train_nyse_v3 --model lstm --horizon 5

Never loads held-out test. Never writes to original model/report paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "05_model_data" / "nyse_v3"
EXPERIMENTS_DIR = PROJECT_ROOT / "artifacts" / "experiments" / "nyse_v3"
EXPECTED_FEATURES = (
    "return_1", "range_pct", "body_pct", "log_return_1", "volatility_10", "sma_5_distance"
)
WINDOW_SIZE = 20
BAR_MS = 300_000
TARGET_SCALE = 1000.0
HIDDEN_SIZE = 32
BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
MAX_EPOCHS = 10
PATIENCE = 3
SEED = 42


class LSTMForecaster(nn.Module):
    """Original LSTM architecture: one layer, 32 hidden units, last step."""

    def __init__(self, input_size: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=HIDDEN_SIZE,
            num_layers=1, batch_first=True,
        )
        self.output = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_output, _ = self.lstm(x)
        return self.output(sequence_output[:, -1, :]).squeeze(-1)


class GRUForecaster(nn.Module):
    """Original GRU architecture: one layer, 32 hidden units, last step."""

    def __init__(self, input_size: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=HIDDEN_SIZE,
            num_layers=1, batch_first=True,
        )
        self.output = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_output, _ = self.gru(x)
        return self.output(sequence_output[:, -1, :]).squeeze(-1)


class CNN1DForecaster(nn.Module):
    """Original CNN1D architecture: 2 conv 3/pad 1, ReLU, average pool, linear."""

    def __init__(self, input_size: int):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.transpose(1, 2)).squeeze(-1)


def make_model(name: str) -> nn.Module:
    models = {"lstm": LSTMForecaster, "gru": GRUForecaster, "cnn1d": CNN1DForecaster}
    if name not in models:
        raise ValueError(f"Unsupported model: {name}")
    return models[name](len(EXPECTED_FEATURES))


def filename_for(split: str, horizon: int) -> str:
    if split not in {"train", "validation"} or horizon not in {5, 15, 30, 60}:
        raise ValueError("Unsupported split/horizon")
    suffix = "" if horizon == 5 else f"_{horizon}m"
    return f"{split}{suffix}_normalized.npz"


def load_split(path: Path, horizon: int) -> dict:
    """Load and independently verify data contracts; no test-file access."""
    if not path.is_file():
        raise FileNotFoundError(f"Expected NYSE V3 NPZ: {path}")
    required = {
        "X", "y", "feature_names", "end_timestamp_ms", "target_timestamp_ms",
        "signal_available_at_ms", "target_available_at_ms", "current_close", "next_close",
    }
    with np.load(path, allow_pickle=False) as src:
        missing = required - set(src.files)
        if missing:
            raise ValueError(f"{path.name}: missing keys {sorted(missing)}")
        d = {name: src[name] for name in required}

    n = len(d["y"])
    X = d["X"]
    if n == 0 or X.shape != (n, WINDOW_SIZE, len(EXPECTED_FEATURES)):
        raise ValueError(f"{path.name}: bad X shape {X.shape}")
    if tuple(d["feature_names"].tolist()) != EXPECTED_FEATURES:
        raise ValueError(f"{path.name}: feature order mismatch")
    if not np.isfinite(X).all() or not np.isfinite(d["y"]).all():
        raise ValueError(f"{path.name}: nonfinite X/y")
    if not np.issubdtype(X.dtype, np.floating) or not np.issubdtype(d["y"].dtype, np.floating):
        raise ValueError(f"{path.name}: nonfloating X/y")
    for key in ("end_timestamp_ms", "target_timestamp_ms", "signal_available_at_ms", "target_available_at_ms", "current_close", "next_close"):
        if d[key].shape != (n,):
            raise ValueError(f"{path.name}: bad metadata shape for {key}")

    end = d["end_timestamp_ms"]
    target = d["target_timestamp_ms"]
    signal_at = d["signal_available_at_ms"]
    target_at = d["target_available_at_ms"]
    if np.any(np.diff(end) <= 0):
        raise ValueError(f"{path.name}: timestamps not strictly chronological")
    if not np.all(target - end == horizon * 60_000):
        raise ValueError(f"{path.name}: wrong target horizon")
    if not np.all(signal_at == end + BAR_MS) or not np.all(target_at == target + BAR_MS):
        raise ValueError(f"{path.name}: candle availability timestamp mismatch")
    if np.any(signal_at >= target_at):
        raise ValueError(f"{path.name}: target known before prediction")
    current, future = d["current_close"], d["next_close"]
    if not np.isfinite(current).all() or not np.isfinite(future).all() or np.any(current <= 0) or np.any(future <= 0):
        raise ValueError(f"{path.name}: invalid prices")
    expected_y = np.log(future / current)
    if float(np.max(np.abs(d["y"].astype(np.float64) - expected_y))) >= 1e-7:
        raise ValueError(f"{path.name}: y does not equal future log return")
    return d


def load_pair(data_dir: Path, horizon: int) -> tuple[dict, dict]:
    scaler_path = data_dir / "normalization_stats.json"
    if not scaler_path.is_file():
        raise FileNotFoundError(f"Missing V3 scaler manifest: {scaler_path}")
    stats = json.loads(scaler_path.read_text(encoding="utf-8"))
    if tuple(stats.get("feature_names", ())) != EXPECTED_FEATURES or stats.get("window_size") != WINDOW_SIZE:
        raise ValueError("V3 normalization stats do not match expected architecture")
    if stats.get("fitted_on") != "train 5m windows only":
        raise ValueError("Scaler is not declared Train-only")
    for key in ("mean", "std"):
        arr = np.array(stats.get(key, []), dtype=np.float64)
        if arr.shape != (len(EXPECTED_FEATURES),) or not np.isfinite(arr).all():
            raise ValueError(f"Invalid scaler {key}")
        if key == "std" and np.any(arr <= 0):
            raise ValueError("Invalid scaler std")

    train = load_split(data_dir / filename_for("train", horizon), horizon)
    val = load_split(data_dir / filename_for("validation", horizon), horizon)
    if int(train["target_available_at_ms"].max()) >= int(val["signal_available_at_ms"].min()):
        raise ValueError("Train labels overlap with Validation signal availability")
    return train, val


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return {"MAE": float(np.abs(err).mean()), "RMSE": float(np.sqrt(np.square(err).mean()))}


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    result = []
    with torch.no_grad():
        for first in range(0, len(X), 512):
            xb = torch.from_numpy(X[first:first + 512].astype(np.float32, copy=False))
            result.append((model(xb).cpu().numpy() / TARGET_SCALE).astype(np.float64))
    return np.concatenate(result)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for part in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def train_experiment(*, model_name: str, horizon: int, data_dir: Path, experiments_dir: Path,
                     epochs: int = MAX_EPOCHS, patience: int = PATIENCE, seed: int = SEED,
                     run_id: str | None = None, dry_run: bool = False) -> dict:
    if epochs < 1 or patience < 1:
        raise ValueError("epochs and patience must be >= 1")
    if model_name not in {"lstm", "gru", "cnn1d"} or horizon not in {5, 15, 30, 60}:
        raise ValueError("Unsupported model/horizon")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    run_id = run_id or f"{model_name}_{horizon}m_seed{seed}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        raise ValueError("Invalid run ID: use letters, digits, '_' or '-' only")
    run_dir = experiments_dir / run_id
    if run_dir.exists():
        raise FileExistsError(f"Experiment already exists (preserved): {run_dir}")

    train, val = load_pair(data_dir, horizon)
    X_train = train["X"].astype(np.float32, copy=False)
    y_train = train["y"].astype(np.float32, copy=False)
    X_val = val["X"].astype(np.float32, copy=False)
    y_val = val["y"].astype(np.float64)
    zero_scores = metrics(y_val, np.zeros_like(y_val))
    info = {
        "model": model_name.upper(), "horizon_minutes": horizon, "run_id": run_id,
        "train_samples": len(X_train), "validation_samples": len(X_val),
        "window_shape": list(X_train.shape[1:]),
        "features": list(EXPECTED_FEATURES),
        "zero_return_validation": zero_scores,
        "test_dataset_used": False,
    }
    print("=== AURUM-AI E.6 | NYSE V3 TRAINING ===", flush=True)
    print(f"Mode: {'DRY RUN / READ ONLY' if dry_run else 'TRAIN / NEW ARTIFACTS ONLY'}", flush=True)
    print(f"Model: {model_name.upper()}  horizon: {horizon}m  seed: {seed}", flush=True)
    print(f"Train: {X_train.shape}  Validation: {X_val.shape}", flush=True)
    print(f"Zero Return Validation MAE: {zero_scores['MAE']:.8f}", flush=True)
    print("Held-out Test not read. Existing V2 files remain untouched.", flush=True)
    if dry_run:
        print("No training, directories or artifacts created.", flush=True)
        return info

    set_seed(seed)
    model = make_model(model_name)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train * TARGET_SCALE)),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.SmoothL1Loss()
    best_mae = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(xb)
        val_pred = predict(model, X_val)
        scores = metrics(y_val, val_pred)
        train_loss = loss_sum / len(X_train)
        history.append({"epoch": epoch, "train_loss_scaled": train_loss,
                        "validation_MAE": scores["MAE"], "validation_RMSE": scores["RMSE"]})
        print(f"Epoch {epoch:02d}/{epochs} | scaled Train Loss {train_loss:.6f} | "
              f"Val MAE {scores['MAE']:.8f} | Val RMSE {scores['RMSE']:.8f}", flush=True)
        if scores["MAE"] < best_mae:
            best_mae = scores["MAE"]
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            print("Early stopping activated.", flush=True)
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    predictions = predict(model, X_val)
    final_scores = metrics(y_val, predictions)
    info.update({
        "status": "COMPLETED", "best_epoch": best_epoch,
        "validation_metrics_selected_on_validation": final_scores,
        "selection_metric": "minimum validation MAE across epochs",
        "evaluation_note": "Validation selected epoch; not an unbiased final Test estimate.",
        "mae_improvement_vs_zero_pct": (
            float(100 * (zero_scores["MAE"] - final_scores["MAE"]) / zero_scores["MAE"])
            if zero_scores["MAE"] else None
        ),
        "training_history": history,
        "hyperparameters": {
            "batch_size": BATCH_SIZE, "max_epochs": epochs, "patience": patience,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "target_scale": TARGET_SCALE, "hidden_or_conv_channels": HIDDEN_SIZE,
            "seed": seed, "gradient_clip_norm": 1.0, "loss": "SmoothL1Loss",
        },
        "dataset": {
            "name": "NYSE V3", "normalization": "Train 5m z-score, reused across horizons",
            "train_npz_sha256": hash_file(data_dir / filename_for("train", horizon)),
            "validation_npz_sha256": hash_file(data_dir / filename_for("validation", horizon)),
            "normalization_stats_sha256": hash_file(data_dir / "normalization_stats.json"),
        },
        "framework": {"torch": torch.__version__, "numpy": np.__version__, "device": "cpu"},
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    # Create only a new run directory, and refuse to overwrite any existing one.
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = {
        "model_state_dict": best_state, "model_name": model_name,
        "input_size": len(EXPECTED_FEATURES), "hidden_size": HIDDEN_SIZE,
        "window_size": WINDOW_SIZE, "target_scale": TARGET_SCALE,
        "feature_names": list(EXPECTED_FEATURES), "forecast_horizon_minutes": horizon,
        "seed": seed, "best_epoch": best_epoch, "data_version": "nyse_v3",
    }
    torch.save(checkpoint, run_dir / "best.pt")
    np.savez_compressed(
        run_dir / "validation_predictions.npz",
        y_true=y_val.astype(np.float32), y_pred=predictions.astype(np.float32),
        end_timestamp_ms=val["end_timestamp_ms"], target_timestamp_ms=val["target_timestamp_ms"],
        signal_available_at_ms=val["signal_available_at_ms"],
        target_available_at_ms=val["target_available_at_ms"],
        current_close=val["current_close"], next_close=val["next_close"],
    )
    (run_dir / "report.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Best epoch: {best_epoch} | MAE: {final_scores['MAE']:.8f}")
    print(f"Created (new run only): {run_dir}")
    print("Test dataset was NOT used. No profit claim.")
    return info


def main():
    parser = argparse.ArgumentParser(description="NYSE V3 isolated CPU training; no Test evaluation")
    parser.add_argument("--model", choices=("lstm", "gru", "cnn1d"), required=True)
    parser.add_argument("--horizon", type=int, choices=(5, 15, 30, 60), required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs only, no training or output")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--run-id", default=None, help="Unique output folder name for a repeat experiment")
    args = parser.parse_args()
    train_experiment(model_name=args.model, horizon=args.horizon, data_dir=DATA_DIR,
                     experiments_dir=EXPERIMENTS_DIR, epochs=args.epochs, patience=args.patience,
                     seed=args.seed, run_id=args.run_id, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
