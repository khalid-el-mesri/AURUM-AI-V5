
from pathlib import Path
import json

import numpy as np
from xgboost import XGBRegressor


# ============================================================
# AURUM-AI — XGBOOST TRAINING & VALIDATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "05_model_data"

REPORT_PATH = (
    PROJECT_ROOT / "reports" / "xgboost_validation.json"
)

MODEL_PATH = (
    PROJECT_ROOT / "artifacts" / "models"
    / "xgboost_model.json"
)


def load_data(name):
    path = DATA_DIR / f"{name}_normalized.npz"

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    with np.load(path, allow_pickle=False) as data:
        return {
            key: data[key]
            for key in data.files
        }


def calculate_metrics(y_true, y_pred):
    errors = y_true - y_pred

    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
    }


def main():

    print("\n=== AURUM-AI | XGBOOST MODEL ===")

    # 1. Load training and validation data ONLY
    train = load_data("train")
    validation = load_data("validation")

    X_train = train["X"]
    y_train = train["y"].astype(np.float64)

    X_val = validation["X"]
    y_val = validation["y"].astype(np.float64)

    if train["feature_names"].tolist() != (
        validation["feature_names"].tolist()
    ):
        raise ValueError("Feature names mismatch.")

    if X_train.ndim != 3 or X_val.ndim != 3:
        raise ValueError("Expected 3D windows.")

    if X_train.shape[1:] != X_val.shape[1:]:
        raise ValueError("Window dimensions mismatch.")

    # 2. Flatten 3D windows into 2D for XGBoost
    X_train_flat = X_train.reshape(
        X_train.shape[0], -1
    )

    X_val_flat = X_val.reshape(
        X_val.shape[0], -1
    )

    for array in (
        X_train_flat, X_val_flat, y_train, y_val
    ):
        if not np.isfinite(array).all():
            raise ValueError("NaN or infinity detected.")

    print(f"\nTraining samples: {len(y_train):,}")
    print(f"Validation samples: {len(y_val):,}")
    print(f"Features per sample: {X_train_flat.shape[1]}")

    # 3. Configure XGBoost
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=10.0,
        min_child_weight=10,
        tree_method="hist",
        eval_metric="rmse",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=4,
    )

    # 4. Train the model
    print("\nTraining XGBoost...")

    model.fit(
        X_train_flat,
        y_train,
        eval_set=[(X_val_flat, y_val)],
        verbose=False,
    )

    # 5. Predictions
    train_predictions = model.predict(X_train_flat)
    val_predictions = model.predict(X_val_flat)

    train_metrics = calculate_metrics(
        y_train, train_predictions
    )

    xgboost_metrics = calculate_metrics(
        y_val, val_predictions
    )

    # Zero Return baseline for fair comparison
    baseline_metrics = calculate_metrics(
        y_val, np.zeros_like(y_val)
    )

    # 6. Display results
    print("\n=== VALIDATION RESULTS ===")

    print("\nZero Return")
    print(f"MAE:  {baseline_metrics['MAE']:.8f}")
    print(f"RMSE: {baseline_metrics['RMSE']:.8f}")

    print("\nXGBoost")
    print(f"MAE:  {xgboost_metrics['MAE']:.8f}")
    print(f"RMSE: {xgboost_metrics['RMSE']:.8f}")

    print("\nXGBoost TRAIN metrics")
    print(f"MAE:  {train_metrics['MAE']:.8f}")
    print(f"RMSE: {train_metrics['RMSE']:.8f}")

    print(f"\nBest iteration: {model.best_iteration}")

    if xgboost_metrics["MAE"] < baseline_metrics["MAE"]:
        print("\nXGBoost beats Zero Return on validation MAE.")
    else:
        print("\nZero Return remains better on validation MAE.")

    # 7. Save model and validation report
    MODEL_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    REPORT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    model.save_model(str(MODEL_PATH))

    report = {
        "model": "XGBoost",
        "evaluation_split": "validation",
        "target": "next_5min_log_return",
        "window_size": int(X_train.shape[1]),
        "features_per_sample": int(X_train_flat.shape[1]),
        "feature_names": train["feature_names"].tolist(),
        "best_iteration": int(model.best_iteration),
        "train_metrics": train_metrics,
        "validation_metrics": xgboost_metrics,
        "zero_return_validation": baseline_metrics,
        "model_path": str(MODEL_PATH),
    }

    with open(
        REPORT_PATH, "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Report saved: {REPORT_PATH}")

    print("\nTest dataset was NOT used.")
    print("XGBOOST EVALUATION COMPLETED")


if __name__ == "__main__":
    main()
