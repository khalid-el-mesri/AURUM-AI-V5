
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# AURUM-AI — NYSE V3 MARKET LOADER
#
# Dataset: XOM historical 5-minute Bid OHLC
# Split: VALIDATION ONLY
#
# Goals:
# - Load complete NYSE Validation sessions
# - Preserve chronological order
# - Validate historical market prices
# - Never modify the source dataset
# - Never load the held-out Test split
#
# Academic simulation only — no real trading
# ============================================================


ROOT = Path(__file__).resolve().parents[2]

VALIDATION_PATH = (
    ROOT
    / "data"
    / "05_model_data"
    / "nyse_v3"
    / "validation_60m_normalized.npz"
)

PRICES_PATH = (
    ROOT
    / "data"
    / "02_clean"
    / "xom_segmented_nyse_v3.parquet"
)

DAY_MS = 86_400_000


def load_validation_market():
    """
    Load complete historical Bid candles for Validation sessions.

    The first Validation day starts at 00:00 UTC for loading,
    so its early market candles are not accidentally excluded.

    This function is read-only.
    """

    # --------------------------------------------------------
    # STEP 1 — Load Validation timestamps
    # --------------------------------------------------------

    with np.load(
        VALIDATION_PATH,
        allow_pickle=False,
    ) as data:

        starts = data["end_timestamp_ms"]
        ends = data["target_available_at_ms"]

    if len(starts) == 0 or len(ends) == 0:
        raise ValueError("Empty Validation dataset")

    if len(starts) != len(ends):
        raise ValueError("Inconsistent Validation timestamps")

    # --------------------------------------------------------
    # STEP 2 — Determine historical period
    # --------------------------------------------------------

    # FIX: Begin at midnight UTC of the first Validation day.
    # This includes all 78 candles of the first NYSE session.

    first_ms = int(
        starts.min() // DAY_MS * DAY_MS
    )

    last_ms = int(ends.max())

    # Dates derived exclusively from Validation windows.
    validation_dates = set(
        pd.to_datetime(
            starts,
            unit="ms",
            utc=True,
        ).strftime("%Y-%m-%d")
    )

    # --------------------------------------------------------
    # STEP 3 — Read historical NYSE V3 Bid prices
    # --------------------------------------------------------

    prices = pd.read_parquet(
        PRICES_PATH,
        columns=[
            "timestamp",
            "open",
            "session_date",
            "segment_id",
            "bar_in_segment",
        ],
        filters=[
            ("timestamp", ">=", first_ms),
            ("timestamp", "<=", last_ms),
        ],
    )

    # Keep only Validation sessions.
    prices = prices[
        prices["session_date"]
        .astype(str)
        .isin(validation_dates)
    ].copy()

    # --------------------------------------------------------
    # STEP 4 — Chronological ordering
    # --------------------------------------------------------

    prices = (
        prices
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if prices.empty:
        raise ValueError(
            "No Validation market prices found"
        )

    # --------------------------------------------------------
    # STEP 5 — Data integrity checks
    # --------------------------------------------------------

    if prices["timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate price timestamps"
        )

    if not prices["timestamp"].is_monotonic_increasing:
        raise ValueError(
            "Market prices are not chronological"
        )

    if prices.isna().any().any():
        raise ValueError(
            "Missing market data"
        )

    if not np.isfinite(
        prices["open"].to_numpy()
    ).all():
        raise ValueError(
            "Non-finite market prices"
        )

    if (prices["open"] <= 0).any():
        raise ValueError(
            "Invalid Bid open prices"
        )

    # --------------------------------------------------------
    # STEP 6 — Return validated market candles
    # --------------------------------------------------------

    return prices
