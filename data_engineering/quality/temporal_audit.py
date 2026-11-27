
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "01_raw"
    / "xomususd-m5-bid-2017-01-26-2026-10-01.csv"
)


def audit_timestamps():
    # Read only the timestamp column
    df = pd.read_csv(DATASET_PATH, usecols=["timestamp"])

    timestamps = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
        errors="coerce",
    )

    print("\n=== AURUM-AI | TEMPORAL AUDIT ===")

    print(f"\nTotal observations: {len(df):,}")
    print(f"Invalid timestamps: {timestamps.isna().sum():,}")

    print(f"\nFirst timestamp: {timestamps.iloc[0]}")
    print(f"Last timestamp:  {timestamps.iloc[-1]}")

    print(f"\nMinimum date: {timestamps.min()}")
    print(f"Maximum date: {timestamps.max()}")

    print("\nCHRONOLOGICAL ORDER")
    print(f"Monotonically increasing: {timestamps.is_monotonic_increasing}")

    print("\nDUPLICATE TIMESTAMPS")
    print(f"Duplicate timestamps: {timestamps.duplicated().sum():,}")

    # Analyze consecutive intervals in original row order
    deltas = timestamps.diff()
    expected_interval = pd.Timedelta(minutes=5)

    print("\nTIME INTERVALS")
    print("Most common intervals:")
    print(deltas.value_counts().head(10))

    print(f"\nExactly 5 minutes: {(deltas == expected_interval).sum():,}")
    print(f"Intervals > 5 minutes: {(deltas > expected_interval).sum():,}")
    print(f"Intervals < 5 minutes: {(deltas < expected_interval).sum():,}")

    print("\nTEMPORAL GAPS")
    gaps = pd.DataFrame(
        {
            "previous_timestamp": timestamps.shift(1),
            "current_timestamp": timestamps,
            "gap": deltas,
        }
    )

    gaps = gaps[gaps["gap"] > expected_interval]

    print(f"Number of gaps > 5 minutes: {len(gaps):,}")

    if not gaps.empty:
        print("\nLargest 10 gaps:")
        print(gaps.nlargest(10, "gap").to_string(index=False))


if __name__ == "__main__":
    audit_timestamps()
