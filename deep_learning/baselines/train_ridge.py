
from pathlib import Path
import json

import numpy as np
from sklearn.linear_model import Ridge


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "05_model_data"
REPORT_PATH = PROJECT_ROOT / "reports" / "ridge_validation.json"


def load_data(name):
    path = DATA_DIR / f"{name}_normalized.npz"

    with np.load(path, allow_pickle=False) as data:
        return data["X"], data["y"]


def metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
    }


def main():
    print("\n=== AURUM-AI | RIDGE REGRESSION ===")

    # 1. Load Train and Validation only
    X_train, y_train = load_data("train")
    X_val, y_val = load_data("validation")

    print(f"\nTrain: {X_train.shape}")
    print(f"Validation: {X_val.shape}")

    # 2. Convert 3D windows into 2D tabular inputs
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)

    print(f"\nFeatures per sample: {X_train_flat.shape[1]}")

    # 3. Train Ridge model
    model = Ridge(alpha=100.0)

    model.fit(X_train_flat, y_train)

    # 4. Predict on Validation
    predictions = model.predict(X_val_flat)

    # 5. Compare with Zero Return baseline
    ridge_results = metrics(y_val, predictions)
    zero_results = metrics(y_val, np.zeros_like(y_val))

    print("\nVALIDATION RESULTS")

    print("\nZero Return")
    print(f"MAE:  {zero_results['MAE']:.8f}")
    print(f"RMSE: {zero_results['RMSE']:.8f}")

    print("\nRidge Regression")
    print(f"MAE:  {ridge_results['MAE']:.8f}")
    print(f"RMSE: {ridge_results['RMSE']:.8f}")

    if ridge_results["MAE"] < zero_results["MAE"]:
        print("\nRidge beats Zero Return on validation MAE.")
    else:
        print("\nZero Return remains better on validation MAE.")

    # 6. Save report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "model": "Ridge Regression",
        "alpha": 100.0,
        "evaluation_split": "validation",
        "target": "next_5min_log_return",
        "features_per_sample": int(X_train_flat.shape[1]),
        "ridge": ridge_results,
        "zero_return": zero_results,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"\nReport saved: {REPORT_PATH}")
    print("Test dataset was not used.")
    print("\nRIDGE EVALUATION COMPLETED")


if __name__ == "__main__":
    main()
