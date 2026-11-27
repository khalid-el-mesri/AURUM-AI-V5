
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "03_features"
    / "xom_features.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "03_features"
    / "xom_features_v2.parquet"
)


def build_advanced_features():
    # 1. Load existing features
    df = pd.read_parquet(INPUT_PATH)

    print("\n=== AURUM-AI | ADVANCED FEATURES ===")
    print(f"\nInput observations: {len(df):,}")

    # 2. Validate input
    required = [
        "segment_id",
        "close",
        "sma_5",
    ]

    if not all(column in df.columns for column in required):
        raise ValueError("Required columns are missing.")

    if (df["close"] <= 0).any():
        raise ValueError("Close prices must be positive.")

    # 3. Group by continuous trading segments
    grouped = df.groupby("segment_id", sort=False)

    # Feature 1: Logarithmic return
    df["log_return_1"] = grouped["close"].transform(
        lambda prices: np.log(prices).diff()
    )

    # Feature 2: Rolling volatility over 10 returns
    df["volatility_10"] = (
        df.groupby("segment_id")["log_return_1"]
        .transform(
            lambda returns: returns.rolling(
                window=10,
                min_periods=10,
            ).std(ddof=0)
        )
    )

    # Feature 3: Relative distance from SMA
    df["sma_5_distance"] = (
        df["close"] / df["sma_5"] - 1
    )

    # Feature 4: Exponential moving average
    df["ema_10"] = grouped["close"].transform(
        lambda prices: prices.ewm(
            span=10,
            adjust=False,
        ).mean()
    )

    # 4. Validate the new features
    new_features = [
        "log_return_1",
        "volatility_10",
        "sma_5_distance",
        "ema_10",
    ]

    for column in new_features:
        values = df[column].dropna().to_numpy()

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite values detected in {column}"
            )

    print("\nNEW FEATURES")
    print(new_features)

    print("\nMISSING VALUES")
    print(df[new_features].isna().sum().to_string())

    print("\nFIRST 12 OBSERVATIONS")
    print(
        df[
            ["segment_id", "close"] + new_features
        ].head(12).to_string(index=False)
    )

    print("\nFEATURE STATISTICS")
    print(df[new_features].describe().to_string())

    # 5. Preserve all observations
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(OUTPUT_PATH, index=False)

    print("\nVALIDATION")
    print(f"Output observations: {len(df):,}")
    print(f"Segments: {df['segment_id'].nunique():,}")
    print(f"Output: {OUTPUT_PATH}")
    print("Original datasets unchanged.")


if __name__ == "__main__":
    build_advanced_features()
