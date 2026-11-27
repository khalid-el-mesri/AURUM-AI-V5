
from pathlib import Path

import numpy as np
import pandas as pd


# 1. Locate the original dataset
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "01_raw"
    / "xomususd-m5-bid-2017-01-26-2026-10-01.csv"
)


def audit_ohlc():
    # 2. Read the CSV without modifying it
    df = pd.read_csv(DATASET_PATH)

    print("\n=== AURUM-AI | OHLC DATA QUALITY AUDIT ===")
    print(f"\nTotal observations: {len(df):,}")

    # 3. Verify the expected columns
    required_columns = ["timestamp", "open", "high", "low", "close"]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    # 4. Convert prices to numeric values for validation
    price_columns = ["open", "high", "low", "close"]
    prices = df[price_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    # 5. Check missing and invalid numeric values
    missing_prices = prices.isna().any(axis=1)
    non_finite_prices = ~np.isfinite(prices.to_numpy()).all(axis=1)
    invalid_numeric = missing_prices | non_finite_prices

    # 6. Check prices greater than zero
    non_positive_prices = (
        (prices <= 0).any(axis=1)
    ).fillna(False)

    # 7. Check OHLC consistency
    invalid_high = (
        (prices["high"] < prices["open"])
        | (prices["high"] < prices["close"])
        | (prices["high"] < prices["low"])
    )

    invalid_low = (
        (prices["low"] > prices["open"])
        | (prices["low"] > prices["close"])
        | (prices["low"] > prices["high"])
    )

    # 8. Identify flat candles
    flat_candles = (
        (prices["open"] == prices["high"])
        & (prices["high"] == prices["low"])
        & (prices["low"] == prices["close"])
    )

    # 9. Identify duplicate complete rows
    duplicate_rows = df.duplicated()

    # 10. Overall invalid rows
    invalid_rows = (
        invalid_numeric
        | non_positive_prices
        | invalid_high
        | invalid_low
    )

    print("\n--- PRICE VALIDATION ---")
    print(f"Missing prices (rows): {missing_prices.sum():,}")
    print(f"Non-finite prices (rows): {non_finite_prices.sum():,}")
    print(f"Non-positive prices (rows): {non_positive_prices.sum():,}")

    print("\n--- OHLC CONSISTENCY ---")
    print(f"Invalid HIGH (rows): {invalid_high.sum():,}")
    print(f"Invalid LOW (rows): {invalid_low.sum():,}")
    print(f"Total invalid rows: {invalid_rows.sum():,}")

    print("\n--- ADDITIONAL CHECKS ---")
    print(f"Flat candles: {flat_candles.sum():,}")
    print(f"Duplicate complete rows: {duplicate_rows.sum():,}")

    print("\n--- PRICE RANGES ---")
    print(prices.agg(["min", "max"]).to_string())

    print("\n--- AUDIT RESULT ---")

    if invalid_rows.sum() == 0:
        print("PASS: No invalid OHLC price rows detected.")
    else:
        print("WARNING: Invalid OHLC rows detected.")
        print("\nFirst 10 invalid rows:")
        print(df.loc[invalid_rows].head(10).to_string(index=False))


if __name__ == "__main__":
    audit_ohlc()
