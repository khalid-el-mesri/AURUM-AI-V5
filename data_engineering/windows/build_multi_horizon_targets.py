
from pathlib import Path
import json

import numpy as np
import pandas as pd


# =====================================================
# AURUM-AI — MULTI-HORIZON TARGET GENERATION
# Train + Validation only
# No test evaluation
# =====================================================

ROOT = Path(__file__).resolve().parents[2]

SOURCE_PATH = (
    ROOT / "data" / "03_features"
    / "xom_features_v2.parquet"
)

MODEL_DATA_DIR = ROOT / "data" / "05_model_data"

OUTPUT_PATH = (
    ROOT / "data" / "04_windows"
    / "multi_horizon_targets_train_validation.parquet"
)

REPORT_PATH = (
    ROOT / "reports"
    / "multi_horizon_targets_audit.json"
)

FEATURES = [
    "return_1",
    "range_pct",
    "body_pct",
    "log_return_1",
    "volatility_10",
    "sma_5_distance",
]

HORIZONS = {
    15: 3,
    30: 6,
    60: 12,
}

BAR_DURATION_MS = 300000


def get_split_days(name):
    path = MODEL_DATA_DIR / f"{name}_normalized.npz"

    with np.load(path, allow_pickle=False) as data:
        timestamps = data["end_timestamp_ms"]

    dates = (
        pd.to_datetime(timestamps, unit="ms", utc=True)
        .tz_convert("America/New_York")
        .strftime("%Y-%m-%d")
    )

    return set(dates)


def main():
    print("\n=== AURUM-AI | MULTI-HORIZON TARGETS ===")

    train_days = get_split_days("train")
    validation_days = get_split_days("validation")

    if train_days & validation_days:
        raise ValueError(
            "Train and validation dates overlap."
        )

    if max(train_days) >= min(validation_days):
        raise ValueError(
            "Invalid chronological split."
        )

    print("\nEXISTING CHRONOLOGICAL SPLIT")
    print(f"Train dates: {len(train_days)}")
    print(f"Validation dates: {len(validation_days)}")

    columns = [
        "timestamp",
        "session_date",
        "segment_id",
        "bar_in_segment",
        "close",
        *FEATURES,
    ]

    df = pd.read_parquet(
        SOURCE_PATH,
        columns=columns,
    )

    df["session_date"] = (
        df["session_date"].astype(str)
    )

    # Keep only known Train and Validation days.
    # The Test period is excluded before computing targets.
    allowed_days = train_days | validation_days

    df = df[
        df["session_date"].isin(allowed_days)
    ].copy()

    df["split"] = np.where(
        df["session_date"].isin(train_days),
        "train",
        "validation",
    )

    df = df.sort_values(
        ["segment_id", "timestamp"]
    ).reset_index(drop=True)

    if df["timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate timestamps detected."
        )

    # Verify that each segment belongs to one day.
    days_per_segment = (
        df.groupby("segment_id")["session_date"]
        .nunique()
    )

    if (days_per_segment != 1).any():
        raise ValueError(
            "A segment contains multiple sessions."
        )

    print(f"\nSelected rows: {len(df)}")
    print("Only Train and Validation dates selected.")

    groups = df.groupby(
        ["split", "segment_id"],
        sort=False,
    )

    audit = {
        "project": "AURUM-AI",
        "source": str(SOURCE_PATH),
        "splits": ["train", "validation"],
        "train_dates": len(train_days),
        "validation_dates": len(validation_days),
        "horizons": {},
        "test_dataset_used_for_model_selection": False,
    }

    for minutes, steps in HORIZONS.items():

        future_close = groups["close"].shift(-steps)
        future_timestamp = groups["timestamp"].shift(
            -steps
        )

        # Reject any target crossing a gap
        # or the end of the trading session.
        valid = (
            (future_timestamp - df["timestamp"])
            == steps * BAR_DURATION_MS
        )

        valid = valid & (
            (df["close"] > 0)
            & (future_close > 0)
        )

        column = f"target_log_return_{minutes}m"

        df[column] = np.nan

        df.loc[valid, column] = np.log(
            future_close.loc[valid]
            / df.loc[valid, "close"]
        )

        valid_count = int(valid.sum())
        excluded_count = int((~valid).sum())

        audit["horizons"][str(minutes)] = {
            "future_bars": steps,
            "valid_targets": valid_count,
            "excluded_rows": excluded_count,
            "train_valid_targets": int(
                (
                    valid
                    & (df["split"] == "train")
                ).sum()
            ),
            "validation_valid_targets": int(
                (
                    valid
                    & (df["split"] == "validation")
                ).sum()
            ),
        }

        print(f"\nHORIZON {minutes} MINUTES")
        print(f"Future bars: {steps}")
        print(f"Valid targets: {valid_count}")
        print(f"Excluded rows: {excluded_count}")

    # Check that every generated target is finite.
    for minutes in HORIZONS:
        column = f"target_log_return_{minutes}m"

        values = df[column].dropna().to_numpy()

        if not np.isfinite(values).all():
            raise ValueError(
                f"Invalid targets for {minutes} minutes."
            )

    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    REPORT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    # Preserve the original feature dataset.
    df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(audit, file, indent=2)

    print("\n=== FILES SAVED ===")
    print(f"Targets: {OUTPUT_PATH}")
    print(f"Audit: {REPORT_PATH}")

    print("\nOriginal dataset was NOT modified.")
    print("Test period was NOT included.")
    print("\nMULTI-HORIZON TARGETS COMPLETED")


if __name__ == "__main__":
    main()
