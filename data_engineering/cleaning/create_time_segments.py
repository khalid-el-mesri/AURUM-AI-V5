
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "02_clean"
    / "xom_regular_session.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "02_clean"
    / "xom_segmented.parquet"
)


def create_time_segments():
    # 1. Load filtered data
    df = pd.read_csv(INPUT_PATH)

    # 2. Convert timestamps to UTC
    df["datetime_utc"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
    )

    # 3. Verify chronological order
    if not df["datetime_utc"].is_monotonic_increasing:
        raise ValueError("Timestamps are not sorted.")

    if df["datetime_utc"].duplicated().any():
        raise ValueError("Duplicate timestamps detected.")

    # 4. Identify local New York trading dates
    ny_time = df["datetime_utc"].dt.tz_convert(
        "America/New_York"
    )

    df["session_date"] = ny_time.dt.strftime("%Y-%m-%d")

    # 5. Detect changes of day
    new_day = df["session_date"].ne(
        df["session_date"].shift()
    )

    # 6. Detect interruptions between observations
    time_difference = df["datetime_utc"].diff()

    expected_interval = pd.Timedelta(minutes=5)

    time_break = time_difference.ne(expected_interval)

    # 7. Create a new segment at each day or interruption
    new_segment = new_day | time_break

    df["segment_id"] = new_segment.cumsum()

    # Position of each candle inside its segment
    df["bar_in_segment"] = df.groupby(
        "segment_id"
    ).cumcount()

    # 8. Verify continuity within each segment
    within_segment_diff = df.groupby(
        "segment_id"
    )["datetime_utc"].diff()

    valid_intervals = (
        within_segment_diff.dropna() == expected_interval
    ).all()

    if not valid_intervals:
        raise ValueError("A segment contains an invalid time interval.")

    # 9. Analyze the generated segments
    segments = df.groupby("segment_id").agg(
        session_date=("session_date", "first"),
        first_timestamp=("datetime_utc", "first"),
        last_timestamp=("datetime_utc", "last"),
        candles=("timestamp", "size"),
    )

    intraday_breaks = ((~new_day) & time_break).sum()

    print("\n=== AURUM-AI | TIME SEGMENTATION ===")

    print(f"\nInput observations: {len(df):,}")
    print(f"Trading dates present: {df['session_date'].nunique():,}")
    print(f"Total segments: {len(segments):,}")
    print(f"Intraday interruptions: {intraday_breaks:,}")

    print("\nSEGMENT LENGTH DISTRIBUTION")
    print(
        segments["candles"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nSEGMENT SIZE STATISTICS")
    print(segments["candles"].describe().to_string())

    print("\nFIRST 5 SEGMENTS")
    print(segments.head().to_string())

    # 10. Save a new derived dataset
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(OUTPUT_PATH, index=False)

    print("\nVALIDATION")
    print("All segments contain consecutive 5-minute observations.")
    print(f"Output: {OUTPUT_PATH}")
    print("Original CSV files unchanged.")


if __name__ == "__main__":
    create_time_segments()
