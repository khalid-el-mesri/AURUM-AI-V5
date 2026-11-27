
from pathlib import Path
import json

import numpy as np


# ============================================================
# AURUM-AI — TRAIN-ONLY NORMALIZATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "04_windows"
OUTPUT_DIR = PROJECT_ROOT / "data" / "05_model_data"

SPLIT_NAMES = ["train", "validation", "test"]


def load_split(name):
    path = INPUT_DIR / f"{name}_windows.npz"

    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")

    with np.load(path, allow_pickle=False) as file:
        return {key: file[key] for key in file.files}


def normalize_windows():
    print("\n=== AURUM-AI | TRAIN-ONLY NORMALIZATION ===")

    # 1. Load training data
    train = load_split("train")

    X_train = train["X"]

    if X_train.ndim != 3:
        raise ValueError("Expected 3D training windows.")

    if not np.isfinite(X_train).all():
        raise ValueError("Training features contain NaN or infinity.")

    feature_names = train["feature_names"].tolist()

    # 2. Fit scaler on training inputs ONLY
    train_values = X_train.astype(np.float64)

    means = train_values.mean(axis=(0, 1))
    stds = train_values.std(axis=(0, 1))

    # Prevent division by zero
    safe_stds = np.where(stds < 1e-12, 1.0, stds)

    print("\nSCALER PARAMETERS (TRAIN ONLY)")

    for name, mean, std in zip(feature_names, means, stds):
        print(f"{name:20s} mean={mean:.8f} std={std:.8f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Save reproducible scaler parameters
    scaler_info = {
        "method": "z_score",
        "fitted_on": "train_only",
        "feature_names": feature_names,
        "window_size": int(X_train.shape[1]),
        "mean": means.tolist(),
        "std": safe_stds.tolist(),
        "original_std": stds.tolist(),
    }

    stats_path = OUTPUT_DIR / "normalization_stats.json"

    with open(stats_path, "w", encoding="utf-8") as file:
        json.dump(scaler_info, file, indent=2)

    # 4. Transform each split using the SAME parameters
    for name in SPLIT_NAMES:
        data = load_split(name)

        X = data["X"]
        y = data["y"]

        if X.ndim != 3:
            raise ValueError(f"Invalid shape for {name}")

        if X.shape[1:] != X_train.shape[1:]:
            raise ValueError(f"Feature shape mismatch: {name}")

        if data["feature_names"].tolist() != feature_names:
            raise ValueError(f"Feature order mismatch: {name}")

        if not np.isfinite(X).all():
            raise ValueError(f"Invalid inputs: {name}")

        if not np.isfinite(y).all():
            raise ValueError(f"Invalid targets: {name}")

        # Normalize input features only
        X_scaled = (
            (X.astype(np.float64) - means) / safe_stds
        ).astype(np.float32)

        if not np.isfinite(X_scaled).all():
            raise ValueError(f"Normalization failed: {name}")

        # Preserve target, timestamps, and prices
        output_data = dict(data)
        output_data["X"] = X_scaled

        output_path = (
            OUTPUT_DIR / f"{name}_normalized.npz"
        )

        np.savez_compressed(output_path, **output_data)

        print(f"\n{name.upper()}")
        print(f"Input shape: {X_scaled.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Saved: {output_path.name}")

        if name == "train":
            print("\nNORMALIZED TRAIN FEATURE MEANS")
            print(
                np.round(
                    X_scaled.mean(axis=(0, 1), dtype=np.float64),
                    6,
                )
            )

            print("\nNORMALIZED TRAIN FEATURE STDS")
            print(
                np.round(
                    X_scaled.std(axis=(0, 1), dtype=np.float64),
                    6,
                )
            )

    print("\nNORMALIZATION COMPLETED")
    print("Scaler fitted on TRAIN only.")
    print("Targets and price metadata unchanged.")
    print("Original datasets unchanged.")


if __name__ == "__main__":
    normalize_windows()
