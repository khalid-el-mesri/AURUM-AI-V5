
from pathlib import Path

import pandas as pd


# 1. Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "02_clean"
    / "xom_segmented.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "03_features"
    / "xom_features.parquet"
)


def build_features():
    # 2. Load segmented data
    df = pd.read_parquet(INPUT_PATH)

    print("\n=== AURUM-AI | FEATURE ENGINEERING ===")

    print(f"\nInput observations: {len(df):,}")

    # 3. Verify input data
    required = [
        "timestamp", "open", "high", "low",
        "close", "segment_id"
    ]

    if not all(column in df.columns for column in required):
        raise ValueError("Required columns are missing.")

    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Non-positive prices detected.")

    # 4. Group data by continuous segment
    grouped = df.groupby("segment_id", sort=False)

    # Feature 1: Return from previous candle
    df["return_1"] = grouped["close"].pct_change(
        fill_method=None
    )

    # Feature 2: Candle price range
    df["range_pct"] = (
        (df["high"] - df["low"]) / df["open"]
    )

    # Feature 3: Candle body
    df["body_pct"] = (
        (df["close"] - df["open"]) / df["open"]
    )

    # Feature 4: Simple Moving Average (5 candles)
    df["sma_5"] = grouped["close"].transform(
        lambda prices: prices.rolling(
            window=5,
            min_periods=5,
        ).mean()
    )

    # 5. Display feature statistics
    feature_columns = [
        "return_1",
        "range_pct",
        "body_pct",
        "sma_5",
    ]

    print("\nNEW FEATURES")
    print(feature_columns)

    print("\nFIRST 10 ROWS")
    print(
        df[
            ["segment_id", "close"] + feature_columns
        ].head(10).to_string(index=False)
    )

    print("\nMISSING VALUES IN FEATURES")
    print(df[feature_columns].isna().sum().to_string())

    print("\nFEATURE STATISTICS")
    print(
        df[feature_columns].describe().to_string()
    )

    # 6. Ensure no original observation was removed
    original_count = len(pd.read_parquet(INPUT_PATH))

    if len(df) != original_count:
        raise ValueError("Observation count changed.")

    # 7. Save new feature dataset
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(OUTPUT_PATH, index=False)

    print("\nVALIDATION")
    print(f"Output observations: {len(df):,}")
    print(f"Unique segments: {df['segment_id'].nunique():,}")
    print(f"Saved to: {OUTPUT_PATH}")
    print("Original datasets unchanged.")


if __name__ == "__main__":
    build_features()
