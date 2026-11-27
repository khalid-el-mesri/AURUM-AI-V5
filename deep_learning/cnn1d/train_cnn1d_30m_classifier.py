
from pathlib import Path
import json
import os
import random

import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ==================================================
# AURUM-AI — CNN1D CLASSIFICATION 30 MINUTES
# Classes: DOWN / NEUTRAL / UP
# Training and validation ONLY
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "05_model_data"

TRAIN_PATH = (
    DATA_DIR / "train_30m_classification.npz"
)

VAL_PATH = (
    DATA_DIR / "validation_30m_classification.npz"
)

MODEL_PATH = (
    ROOT / "artifacts" / "models"
    / "cnn1d_30m_classifier_best.pt"
)

REPORT_PATH = (
    ROOT / "reports"
    / "cnn1d_30m_classification_validation.json"
)

CLASS_NAMES = ["DOWN", "NEUTRAL", "UP"]

BATCH_SIZE = 256
MAX_EPOCHS = 12
PATIENCE = 4
LEARNING_RATE = 0.001
SEED = 42


def load_data(path):

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:

        X = data["X"].astype(np.float32)

        y = data["y_class"].astype(np.int64)

        features = data["feature_names"].tolist()

        end_ts = data[
            "end_timestamp_ms"
        ].astype(np.int64)

        target_ts = data[
            "target_timestamp_ms"
        ].astype(np.int64)

        threshold = float(
            data["threshold_log_return"]
        )

    if X.ndim != 3 or X.shape[1:] != (20, 6):
        raise ValueError("Expected shape (N,20,6).")

    if not (
        len(X) == len(y) == len(end_ts) == len(target_ts)
    ):
        raise ValueError("Dataset length mismatch.")

    if not np.isfinite(X).all():
        raise ValueError("Invalid feature values.")

    if not np.isin(y, [0, 1, 2]).all():
        raise ValueError("Invalid class labels.")

    if not np.all(target_ts - end_ts == 30 * 60_000):
        raise ValueError("Invalid horizon alignment.")

    if len(np.unique(end_ts)) != len(end_ts):
        raise ValueError("Duplicate timestamps.")

    return X, y, features, end_ts, target_ts, threshold


class CNN1DClassifier(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(
                input_size, 32,
                kernel_size=3, padding=1
            ),
            nn.ReLU(),

            nn.Conv1d(
                32, 32,
                kernel_size=3, padding=1
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),

            nn.Linear(32, 3),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.network(x)


def predict(model, X):

    model.eval()

    predictions = []

    with torch.no_grad():

        for start in range(0, len(X), 512):

            batch = torch.from_numpy(
                X[start:start + 512]
            )

            logits = model(batch)

            predicted_classes = torch.argmax(
                logits, dim=1
            )

            predictions.append(
                predicted_classes.cpu().numpy()
            )

    return np.concatenate(predictions)


def evaluate(y_true, y_pred):

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
    }


def main():

    print(
        "\n=== AURUM-AI | CNN1D 30M CLASSIFIER ==="
    )

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    torch.set_num_threads(
        min(4, os.cpu_count() or 1)
    )

    # Do not overwrite completed experiments.
    for path in (MODEL_PATH, REPORT_PATH):
        if path.exists():
            raise FileExistsError(
                f"Existing artifact preserved: {path}"
            )

    (
        X_train, y_train, train_features,
        train_end, train_target, train_threshold
    ) = load_data(TRAIN_PATH)

    (
        X_val, y_val, val_features,
        val_end, val_target, val_threshold
    ) = load_data(VAL_PATH)

    if train_features != val_features:
        raise ValueError("Feature order mismatch.")

    if not np.isclose(
        train_threshold,
        val_threshold,
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("Classification threshold mismatch.")

    if train_target.max() >= val_end.min():
        raise ValueError("Chronological split violation.")

    print("\nDEVICE: CPU")
    print(f"Train: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Neutral threshold: {train_threshold:.10f}")

    print("\n=== CLASS DISTRIBUTION ===")

    train_counts = np.bincount(
        y_train, minlength=3
    )

    val_counts = np.bincount(
        y_val, minlength=3
    )

    for index, name in enumerate(CLASS_NAMES):

        print(
            f"{name}: "
            f"Train={train_counts[index]} "
            f"Validation={val_counts[index]}"
        )

    # Majority baseline chosen from TRAIN only.
    majority_class = int(
        np.argmax(train_counts)
    )

    baseline_predictions = np.full_like(
        y_val, majority_class
    )

    baseline_metrics = evaluate(
        y_val, baseline_predictions
    )

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    model = CNN1DClassifier(
        input_size=X_train.shape[2]
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001,
    )

    criterion = nn.CrossEntropyLoss()

    MODEL_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    REPORT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    best_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    history = []

    print("\nTraining classifier...")

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:

            optimizer.zero_grad()

            logits = model(xb)

            loss = criterion(logits, yb)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_loss += (
                loss.item() * len(xb)
            )

        train_loss = (
            total_loss / len(X_train)
        )

        val_predictions = predict(
            model, X_val
        )

        metrics = evaluate(
            y_val, val_predictions
        )

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Loss: {train_loss:.6f} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Balanced Acc: "
            f"{metrics['balanced_accuracy']:.4f} | "
            f"Macro-F1: {metrics['macro_f1']:.4f}",
            flush=True,
        )

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            **metrics,
        })

        # Select checkpoint by Validation Macro-F1.
        if metrics["macro_f1"] > best_f1 + 1e-8:

            best_f1 = metrics["macro_f1"]
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "input_size": int(X_train.shape[2]),
                "window_size": int(X_train.shape[1]),
                "feature_names": train_features,
                "class_names": CLASS_NAMES,
                "neutral_threshold": train_threshold,
                "horizon_minutes": 30,
            }, MODEL_PATH)

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("\nEarly stopping activated.")
            break

    # Reload best validation checkpoint.
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

    final_metrics = evaluate(
        y_val, final_predictions
    )

    confusion = confusion_matrix(
        y_val,
        final_predictions,
        labels=[0, 1, 2],
    )

    class_report = classification_report(
        y_val,
        final_predictions,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )

    print(
        "\n=== FINAL VALIDATION RESULTS — CLASSIFICATION ==="
    )

    print("\nMajority Baseline")
    print(f"Class: {CLASS_NAMES[majority_class]}")
    print(
        f"Accuracy: "
        f"{baseline_metrics['accuracy']:.4f}"
    )
    print(
        f"Balanced Accuracy: "
        f"{baseline_metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"Macro-F1: "
        f"{baseline_metrics['macro_f1']:.4f}"
    )

    print("\nCNN1D Classifier")
    print(
        f"Accuracy: "
        f"{final_metrics['accuracy']:.4f}"
    )
    print(
        f"Balanced Accuracy: "
        f"{final_metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"Macro-F1: "
        f"{final_metrics['macro_f1']:.4f}"
    )

    print(f"\nBest epoch: {best_epoch}")

    print(
        "\nCONFUSION MATRIX "
        "(rows=actual, columns=predicted)"
    )

    print(confusion)

    report = {
        "project": "AURUM-AI",
        "model": "CNN1D Classifier",
        "task": "UP/DOWN/NEUTRAL",
        "horizon_minutes": 30,
        "evaluation_split": "validation",
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "class_names": CLASS_NAMES,
        "neutral_threshold": train_threshold,
        "best_epoch": best_epoch,
        "selection_metric": "macro_f1",
        "baseline_class": CLASS_NAMES[majority_class],
        "baseline_metrics": baseline_metrics,
        "classifier_metrics": final_metrics,
        "confusion_matrix": confusion.tolist(),
        "classification_report": class_report,
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
    print("Test dataset was NOT used.")
    print("\nCNN1D CLASSIFICATION COMPLETED")


if __name__ == "__main__":
    main()
