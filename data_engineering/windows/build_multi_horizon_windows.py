
from pathlib import Path
import json

import numpy as np
import pandas as pd


# =====================================================
# AURUM-AI — MULTI-HORIZON MODEL WINDOWS
# Reuse existing normalized 20x6 windows
# Train + Validation ONLY
# =====================================================

ROOT = Path(__file__).resolve().parents[2]

TARGETS_PATH = (
    ROOT / "data" / "04_windows"
    / "multi_horizon_targets_train_validation.parquet"
)

MODEL_DIR = ROOT / "data" / "05_model_data"

REPORT_PATH = (
    ROOT / "reports"
    / "multi_horizon_windows_audit.json"
)

HORIZONS = (15, 30, 60)
SPLITS = ("train", "validation")


def main():

    print("\n=== AURUM-AI | MULTI-HORIZON WINDOWS ===")

    required_columns = [
        "timestamp",
        "split",
        "close",
        "target_log_return_15m",
        "target_log_return_30m",
        "target_log_return_60m",
    ]

    targets = pd.read_parquet(
        TARGETS_PATH,
        columns=required_columns,
    )

    if targets["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps.")

    targets["timestamp"] = (
        targets["timestamp"].astype(np.int64)
    )

    if not targets["split"].isin(SPLITS).all():
        raise ValueError("Unexpected split.")

    # Timestamp is the alignment key.
    targets = targets.set_index(
        "timestamp",
        verify_integrity=True,
    )

    audit = {
        "project": "AURUM-AI",
        "method": "Reuse normalized 5-minute input windows",
        "splits": {},
        "test_dataset_used": False,
    }

    for split in SPLITS:

        input_path = (
            MODEL_DIR / f"{split}_normalized.npz"
        )

        with np.load(
            input_path, allow_pickle=False
        ) as data:

            X = data["X"].astype(np.float32)

            end_ts = data[
                "end_timestamp_ms"
            ].astype(np.int64)

            current_close = data[
                "current_close"
            ].astype(np.float64)

            feature_names = data[
                "feature_names"
            ].copy()

        if len(X) != len(end_ts):
            raise ValueError("Window count mismatch.")

        if len(np.unique(end_ts)) != len(end_ts):
            raise ValueError("Duplicate window timestamps.")

        if not np.isfinite(X).all():
            raise ValueError("Invalid input features.")

        # Match existing input windows to new targets.
        positions = targets.index.get_indexer(end_ts)

        if np.any(positions < 0):
            raise ValueError(
                f"Missing target timestamps in {split}."
            )

        aligned = targets.iloc[positions]

        if not (aligned["split"] == split).all():
            raise ValueError(
                f"Split mismatch in {split}."
            )

        if not np.allclose(
            aligned["close"].to_numpy(
                dtype=np.float64
            ),
            current_close,
            rtol=1e-6,
            atol=1e-5,
        ):
            raise ValueError(
                f"Close-price alignment mismatch: {split}"
            )

        audit["splits"][split] = {}

        print(f"\n=== {split.upper()} ===")
        print(f"Original 5m windows: {len(X)}")

        for minutes in HORIZONS:

            column = (
                f"target_log_return_{minutes}m"
            )

            target_values = aligned[
                column
            ].to_numpy(dtype=np.float64)

            # Only windows having a valid future target.
            valid = np.isfinite(target_values)

            if not valid.any():
                raise ValueError(
                    f"No valid {minutes}m windows."
                )

            X_new = X[valid]

            y_new = target_values[
                valid
            ].astype(np.float32)

            end_new = end_ts[valid]

            target_ts_new = (
                end_new + minutes * 60_000
            )

            close_new = current_close[valid]

            if X_new.shape[1:] != (20, 6):
                raise ValueError(
                    "Unexpected feature window shape."
                )

            output_path = (
                MODEL_DIR
                / f"{split}_{minutes}m_normalized.npz"
            )

            # Never overwrite existing 5m datasets.
            if output_path.exists():
                raise FileExistsError(
                    f"Output already exists: {output_path}"
                )

            np.savez_compressed(
                output_path,
                X=X_new,
                y=y_new,
                feature_names=feature_names,
                end_timestamp_ms=end_new,
                target_timestamp_ms=target_ts_new,
                current_close=close_new,
            )

            audit["splits"][split][
                str(minutes)
            ] = {
                "total_original_windows": int(len(X)),
                "valid_windows": int(len(y_new)),
                "excluded_windows": int(
                    len(X) - len(y_new)
                ),
                "X_shape": list(X_new.shape),
                "target": (
                    f"next_{minutes}min_log_return"
                ),
                "file": str(output_path),
            }

            print(
                f"{minutes}m | "
                f"X: {X_new.shape} | "
                f"y: {y_new.shape} | "
                f"Excluded: {len(X) - len(y_new)}"
            )

    REPORT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    with open(
        REPORT_PATH, "w", encoding="utf-8"
    ) as file:
        json.dump(audit, file, indent=2)

    print("\n=== FILES SAVED ===")
    print(f"Audit: {REPORT_PATH}")

    print("\nOriginal 5m files were NOT modified.")
    print("Test dataset was NOT used.")
    print("\nMULTI-HORIZON WINDOWS COMPLETED")


if __name__ == "__main__":
    main()
