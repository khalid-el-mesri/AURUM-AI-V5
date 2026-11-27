
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = (
    PROJECT_ROOT
    / "data"
    / "01_raw"
    / "xomususd-m5-bid-2017-01-26-2026-10-01.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "02_clean"
    / "xom_regular_session.csv"
)


def filter_regular_session():
    # Read original data without modifying RAW
    df = pd.read_csv(RAW_PATH)

    # Convert Unix milliseconds to UTC
    utc_time = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
    )

    # Convert to New York timezone, including DST
    ny_time = utc_time.dt.tz_convert("America/New_York")

    # Minutes since midnight in New York
    minutes = ny_time.dt.hour * 60 + ny_time.dt.minute

    # Monday to Friday and 09:30 <= time < 16:00
    regular_session = (
        (ny_time.dt.dayofweek < 5)
        & (minutes >= 9 * 60 + 30)
        & (minutes < 16 * 60)
    )

    # Preserve original OHLC and timestamp columns
    clean_df = df.loc[regular_session].copy()

    print("\n=== AURUM-AI | REGULAR SESSION FILTER ===")
    print(f"RAW observations: {len(df):,}")
    print(f"Selected observations: {len(clean_df):,}")
    print(f"Excluded observations: {len(df) - len(clean_df):,}")

    # Create output directory if necessary
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save a separate working dataset
    clean_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved to: {OUTPUT_PATH}")
    print("RAW dataset unchanged.")


if __name__ == "__main__":
    filter_regular_session()
