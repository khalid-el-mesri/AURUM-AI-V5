
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "01_raw"
    / "xomususd-m5-bid-2017-01-26-2026-10-01.csv"
)


def analyze_sessions():
    df = pd.read_csv(DATASET_PATH)

    df["datetime"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
    )

    df["weekday"] = df["datetime"].dt.day_name()
    df["hour_utc"] = df["datetime"].dt.hour

    df["is_active"] = ~(
        (df["open"] == df["high"])
        & (df["high"] == df["low"])
        & (df["low"] == df["close"])
    )

    print("\n=== AURUM-AI | SESSION ANALYSIS ===")

    print("\nOBSERVATIONS BY WEEKDAY")
    print(
        df.groupby("weekday")
        .agg(
            total=("is_active", "size"),
            active=("is_active", "sum"),
        )
        .to_string()
    )

    print("\nACTIVE CANDLES BY UTC HOUR")
    print(
        df.groupby("hour_utc")
        .agg(
            total=("is_active", "size"),
            active=("is_active", "sum"),
        )
        .to_string()
    )

    print("\nFIRST AND LAST ACTIVE HOUR BY YEAR")
    active = df[df["is_active"]].copy()

    active["year"] = active["datetime"].dt.year

    print(
        active.groupby("year")["hour_utc"]
        .agg(["min", "max"])
        .to_string()
    )


if __name__ == "__main__":
    analyze_sessions()
