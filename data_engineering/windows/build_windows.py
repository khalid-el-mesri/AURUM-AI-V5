
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# AURUM-AI — WINDOWING & CHRONOLOGICAL SPLIT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "03_features"
    / "xom_features_v2.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "04_windows"
)

# 20 candles of 5 minutes = 100 minutes of history
WINDOW_SIZE = 20

# Relative features used by the models
FEATURE_COLUMNS = [
    "return_1",
    "range_pct",
    "body_pct",
    "log_return_1",
    "volatility_10",
    "sma_5_distance",
]


def build_windows():

    print("\n=== AURUM-AI | WINDOWING & SPLIT ===")

    # --------------------------------------------------------
    # 1. Load feature dataset
    # --------------------------------------------------------

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_PATH}"
        )

    df = pd.read_parquet(INPUT_PATH)

    print(f"\nInput observations: {len(df):,}")

    # --------------------------------------------------------
    # 2. Validate dataset
    # --------------------------------------------------------

    required_columns = FEATURE_COLUMNS + [
        "segment_id",
        "session_date",
        "timestamp",
        "close",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps are not sorted.")

    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps detected.")

    # --------------------------------------------------------
    # 3. Chronological Train / Validation / Test split
    # --------------------------------------------------------

    # Split by full trading sessions, never by random rows
    dates = sorted(df["session_date"].unique())

    n_dates = len(dates)

    train_end = int(n_dates * 0.70)
    validation_end = int(n_dates * 0.85)

    train_dates = set(dates[:train_end])

    validation_dates = set(
        dates[train_end:validation_end]
    )

    test_dates = set(
        dates[validation_end:]
    )

    if not train_dates or not validation_dates or not test_dates:
        raise ValueError(
            "Not enough trading sessions for splitting."
        )

    splits = {
        "train": {
            "dates": train_dates,
            "X": [],
            "y": [],
            "end_ts": [],
            "target_ts": [],
            "current_close": [],
            "next_close": [],
        },
        "validation": {
            "dates": validation_dates,
            "X": [],
            "y": [],
            "end_ts": [],
            "target_ts": [],
            "current_close": [],
            "next_close": [],
        },
        "test": {
            "dates": test_dates,
            "X": [],
            "y": [],
            "end_ts": [],
            "target_ts": [],
            "current_close": [],
            "next_close": [],
        },
    }

    # --------------------------------------------------------
    # 4. Build windows inside continuous time segments
    # --------------------------------------------------------

    skipped_windows = 0

    for segment_id, group in df.groupby(
        "segment_id",
        sort=False
    ):

        # A segment must belong to one trading session
        if group["session_date"].nunique() != 1:
            raise ValueError(
                f"Segment {segment_id} spans multiple sessions."
            )

        session_date = group["session_date"].iloc[0]

        # Determine which chronological split it belongs to
        if session_date in train_dates:
            split_name = "train"

        elif session_date in validation_dates:
            split_name = "validation"

        elif session_date in test_dates:
            split_name = "test"

        else:
            raise ValueError(
                f"Unknown trading session: {session_date}"
            )

        bucket = splits[split_name]

        features = group[FEATURE_COLUMNS].to_numpy(
            dtype=np.float64
        )

        closes = group["close"].to_numpy(
            dtype=np.float64
        )

        timestamps = group["timestamp"].to_numpy(
            dtype=np.int64
        )

        # Validate prices
        if (
            not np.isfinite(closes).all()
            or (closes <= 0).any()
        ):
            raise ValueError(
                f"Invalid close prices in segment {segment_id}"
            )

        # Each consecutive candle must be 5 minutes apart
        if not np.all(
            np.diff(timestamps) == 300_000
        ):
            raise ValueError(
                f"Non-continuous segment: {segment_id}"
            )

        # ----------------------------------------------------
        # 5. Create sliding windows
        # ----------------------------------------------------

        # Input: 20 candles ending at time t
        # Target: logarithmic return from t to t+1

        for t in range(
            WINDOW_SIZE - 1,
            len(group) - 1
        ):

            window = features[
                t - WINDOW_SIZE + 1:t + 1
            ]

            # Skip windows containing NaN or infinity
            if not np.isfinite(window).all():
                skipped_windows += 1
                continue

            # The target is the next 5-minute log return
            target = np.log(
                closes[t + 1] / closes[t]
            )

            if not np.isfinite(target):
                skipped_windows += 1
                continue

            # Store input features and prediction target
            bucket["X"].append(
                window.astype(np.float32)
            )

            bucket["y"].append(target)

            # Keep timestamps for future backtesting
            bucket["end_ts"].append(
                timestamps[t]
            )

            bucket["target_ts"].append(
                timestamps[t + 1]
            )

            # Keep prices for virtual portfolio simulation
            bucket["current_close"].append(
                closes[t]
            )

            bucket["next_close"].append(
                closes[t + 1]
            )

    # --------------------------------------------------------
    # 6. Save datasets
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nWINDOW CONFIGURATION")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Skipped windows: {skipped_windows:,}")

    for name, bucket in splits.items():

        if not bucket["X"]:
            raise ValueError(
                f"No valid windows for {name}."
            )

        X = np.stack(bucket["X"])

        y = np.asarray(
            bucket["y"],
            dtype=np.float32
        )

        end_timestamps = np.asarray(
            bucket["end_ts"],
            dtype=np.int64
        )

        target_timestamps = np.asarray(
            bucket["target_ts"],
            dtype=np.int64
        )

        # Final safety checks
        if X.shape[1:] != (
            WINDOW_SIZE,
            len(FEATURE_COLUMNS)
        ):
            raise ValueError(
                f"Unexpected input shape: {X.shape}"
            )

        if not (
            len(X)
            == len(y)
            == len(end_timestamps)
            == len(target_timestamps)
        ):
            raise ValueError(
                f"Inconsistent arrays for {name}"
            )

        if not np.all(
            target_timestamps - end_timestamps == 300_000
        ):
            raise ValueError(
                f"Invalid prediction horizon in {name}"
            )

        if not np.isfinite(X).all():
            raise ValueError(
                f"Non-finite features in {name}"
            )

        if not np.isfinite(y).all():
            raise ValueError(
                f"Non-finite targets in {name}"
            )

        output_path = (
            OUTPUT_DIR / f"{name}_windows.npz"
        )

        np.savez_compressed(
            output_path,
            X=X,
            y=y,
            feature_names=np.array(FEATURE_COLUMNS),
            end_timestamp_ms=end_timestamps,
            target_timestamp_ms=target_timestamps,
            current_close=np.asarray(
                bucket["current_close"]
            ),
            next_close=np.asarray(
                bucket["next_close"]
            ),
        )

        print(f"\n{name.upper()}")
        print(
            f"Trading dates: {len(bucket['dates']):,}"
        )
        print(
            f"Date range: {min(bucket['dates'])}"
            f" -> {max(bucket['dates'])}"
        )
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"Saved: {output_path.name}")

    print("\nWINDOW GENERATION COMPLETED")
    print("Original datasets unchanged.")
    print("Normalization has NOT been applied yet.")


if __name__ == "__main__":
    build_windows()
