
from pathlib import Path

import pandas as pd


# 1. Locate the filtered dataset
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "02_clean"
    / "xom_regular_session.csv"
)

EXPECTED_CANDLES = 78


def audit_daily_coverage():
    # 2. Load the filtered dataset
    df = pd.read_csv(DATASET_PATH)

    # 3. Convert Unix timestamps to New York time
    utc_time = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
    )

    ny_time = utc_time.dt.tz_convert("America/New_York")

    # 4. Extract local trading dates and times
    df["date_ny"] = ny_time.dt.date
    df["time_ny"] = ny_time.dt.strftime("%H:%M")

    # 5. Count candles for each available trading date
    daily = df.groupby("date_ny").agg(
        candles=("timestamp", "size"),
        first_candle=("time_ny", "min"),
        last_candle=("time_ny", "max"),
    )

    # 6. Identify complete and partial days
    complete_days = daily[daily["candles"] == EXPECTED_CANDLES]
    incomplete_days = daily[daily["candles"] < EXPECTED_CANDLES]
    extra_days = daily[daily["candles"] > EXPECTED_CANDLES]

    # 7. Print general statistics
    print("\n=== AURUM-AI | DAILY COVERAGE AUDIT ===")

    print(f"\nTotal observations: {len(df):,}")
    print(f"Total dates present: {len(daily):,}")

    print("\nSESSION COMPLETENESS")
    print(f"Days with 78 candles: {len(complete_days):,}")
    print(f"Days with fewer than 78: {len(incomplete_days):,}")
    print(f"Days with more than 78: {len(extra_days):,}")

    # 8. Show distribution of candles per day
    print("\nCANDLE COUNT DISTRIBUTION")
    print(
        daily["candles"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # 9. Show ALL incomplete days
    print("\nALL INCOMPLETE DAYS")

    if incomplete_days.empty:
        print("No incomplete days detected.")
    else:
        print(incomplete_days.to_string())

    # 10. Calculate yearly coverage statistics
    daily["year"] = pd.to_datetime(daily.index).year

    yearly = daily.groupby("year")["candles"].agg(
        days="count",
        total_candles="sum",
        min_candles="min",
        max_candles="max",
    )

    print("\nCOVERAGE BY YEAR")
    print(yearly.to_string())

    # 11. Final message
    print("\nAUDIT COMPLETED")
    print("Dataset was read only. No data modified.")


if __name__ == "__main__":
    audit_daily_coverage()
