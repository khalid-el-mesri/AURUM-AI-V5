
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "01_raw"
    / "xomususd-m5-bid-2017-01-26-2026-10-01.csv"
)


def audit_market_hours():
    df = pd.read_csv(DATASET_PATH)

    # Convert Unix milliseconds into UTC timestamps
    utc_time = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
    )

    # Convert UTC to New York local time, including DST
    ny_time = utc_time.dt.tz_convert("America/New_York")

    df["ny_time"] = ny_time
    df["year"] = ny_time.dt.year

    # Regular-session clock window:
    # 09:30 inclusive, 16:00 exclusive.
    # This is a time-of-day check, not a holiday calendar.
    minutes = ny_time.dt.hour * 60 + ny_time.dt.minute

    weekday = ny_time.dt.dayofweek < 5

    df["regular_hours"] = (
        weekday
        & (minutes >= 9 * 60 + 30)
        & (minutes < 16 * 60)
    )

    df["flat"] = (
        (df["open"] == df["high"])
        & (df["high"] == df["low"])
        & (df["low"] == df["close"])
    )

    print("\n=== AURUM-AI | MARKET HOURS AUDIT ===")

    print(f"\nTotal observations: {len(df):,}")
    print(f"Inside regular hours: {df['regular_hours'].sum():,}")
    print(f"Outside regular hours: {(~df['regular_hours']).sum():,}")

    summary = df.groupby("regular_hours")["flat"].agg(
        total="count",
        flat="sum",
        flat_percentage="mean",
    )

    summary["flat_percentage"] *= 100

    print("\nFLAT CANDLES BY SESSION CATEGORY")
    print(summary.round(2).to_string())

    print("\nREGULAR-HOURS OBSERVATIONS BY YEAR")
    print(
        df.groupby("year")["regular_hours"]
        .sum()
        .to_string()
    )


if __name__ == "__main__":
    audit_market_hours()
