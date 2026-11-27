
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "01_raw"
    / "xomususd-m5-bid-2017-01-26-2026-10-01.csv"
)


def analyze_flat_candles():
    df = pd.read_csv(DATASET_PATH)

    df["datetime"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
    )

    df["year"] = df["datetime"].dt.year
    df["hour_utc"] = df["datetime"].dt.hour

    df["is_flat"] = (
        (df["open"] == df["high"])
        & (df["high"] == df["low"])
        & (df["low"] == df["close"])
    )

    print("\n=== AURUM-AI | FLAT CANDLES ANALYSIS ===")

    print("\nFLAT CANDLES BY YEAR")
    yearly = df.groupby("year")["is_flat"].agg(
        total="count",
        flat="sum",
        flat_percentage="mean",
    )

    yearly["flat_percentage"] *= 100
    print(yearly.round(2).to_string())

    print("\nFLAT CANDLES BY UTC HOUR")
    hourly = df.groupby("hour_utc")["is_flat"].agg(
        total="count",
        flat="sum",
        flat_percentage="mean",
    )

    hourly["flat_percentage"] *= 100
    print(hourly.round(2).to_string())


if __name__ == "__main__":
    analyze_flat_candles()
