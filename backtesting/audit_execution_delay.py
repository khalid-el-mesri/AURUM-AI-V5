
from pathlib import Path

import numpy as np
import pandas as pd


# AURUM-AI — EXECUTION DELAY AUDIT
# Validation only, no new trades
# Compare next-bar open with one extra bar delay

ROOT = Path(__file__).resolve().parents[1]

PRICES_PATH = (
    ROOT / "data/02_clean/xom_segmented.parquet"
)

TRADES_DIR = ROOT / "reports/backtesting"

MODELS = {
    "CNN1D": "cnn1d_trades_validation.csv",
    "Logistic": "logistic_trades_validation.csv",
}

BAR_MS = 300_000
COST_BPS = 10
COST = COST_BPS / 10000


def evaluate_trades(name, trades, prices):

    signal_ts = trades[
        "signal_timestamp_ms"
    ].to_numpy(dtype=np.int64)

    original_entry_ts = trades[
        "entry_timestamp_ms"
    ].to_numpy(dtype=np.int64)

    exit_ts = trades[
        "exit_timestamp_ms"
    ].to_numpy(dtype=np.int64)

    delayed_entry_ts = signal_ts + 2 * BAR_MS

    if not np.array_equal(
        original_entry_ts, signal_ts + BAR_MS
    ):
        raise ValueError("Unexpected original entry timing.")

    signal = prices.reindex(signal_ts)
    original_entry = prices.reindex(original_entry_ts)
    delayed_entry = prices.reindex(delayed_entry_ts)
    exit_rows = prices.reindex(exit_ts)

    for label, frame in [
        ("Signal", signal),
        ("Original entry", original_entry),
        ("Delayed entry", delayed_entry),
        ("Exit", exit_rows),
    ]:
        if frame.isna().any().any():
            raise ValueError(f"Missing candle: {label}")

    signal_segment = signal["segment_id"].to_numpy()

    for frame in (original_entry, delayed_entry, exit_rows):
        if not np.array_equal(
            signal_segment,
            frame["segment_id"].to_numpy()
        ):
            raise ValueError("Cross-session execution detected.")

    if not np.all(
        delayed_entry["bar_in_segment"].to_numpy()
        - signal["bar_in_segment"].to_numpy() == 2
    ):
        raise ValueError("Delayed candle alignment failed.")

    original_price = original_entry[
        "open"
    ].to_numpy(dtype=np.float64)

    delayed_price = delayed_entry[
        "open"
    ].to_numpy(dtype=np.float64)

    exit_price = exit_rows[
        "close"
    ].to_numpy(dtype=np.float64)

    if not np.allclose(
        original_price,
        trades["entry_bid_reference"].to_numpy(),
        rtol=0,
        atol=1e-6,
    ):
        raise ValueError("Original entry prices mismatch.")

    if not np.allclose(
        exit_price,
        trades["exit_bid_reference"].to_numpy(),
        rtol=0,
        atol=1e-6,
    ):
        raise ValueError("Exit prices mismatch.")

    if np.any(
        (original_price <= 0)
        | (delayed_price <= 0)
        | (exit_price <= 0)
    ):
        raise ValueError("Non-positive prices.")

    def metrics(entry_price):
        gross = (
            exit_price / entry_price - 1
        ) * 100

        net = (
            (exit_price / entry_price)
            * (1 - COST) / (1 + COST) - 1
        ) * 100

        return {
            "gross": float(gross.mean()),
            "net": float(net.mean()),
            "win_rate": float((net > 0).mean() * 100),
        }

    original = metrics(original_price)
    delayed = metrics(delayed_price)

    recorded_net = float(
        trades["net_trade_return_pct"].mean()
    )

    if not np.isclose(
        original["net"], recorded_net,
        rtol=0,
        atol=1e-6,
    ):
        raise ValueError("Original cost accounting mismatch.")

    print(f"\n=== {name.upper()} ===")
    print(f"Existing trades: {len(trades)}")

    print("\nOriginal entry: next candle open")
    print(f"Average gross: {original['gross']:+.5f}%")
    print(f"Average net: {original['net']:+.5f}%")
    print(f"Net win rate: {original['win_rate']:.2f}%")

    print("\nDelayed entry: one extra 5m candle")
    print(f"Average gross: {delayed['gross']:+.5f}%")
    print(f"Average net: {delayed['net']:+.5f}%")
    print(f"Net win rate: {delayed['win_rate']:.2f}%")

    difference = delayed["net"] - original["net"]

    print(
        f"\nDelay impact on average net trade: "
        f"{difference:+.5f} percentage points"
    )


def main():

    print("\n=== AURUM-AI | EXECUTION DELAY AUDIT ===")

    trades_by_model = {}

    for name, filename in MODELS.items():
        path = TRADES_DIR / filename
        if not path.exists():
            raise FileNotFoundError(path)

        trades_by_model[name] = pd.read_csv(path)

    all_trades = pd.concat(
        trades_by_model.values(),
        ignore_index=True,
    )

    # Read only the validation trade time interval.
    first_ts = int(
        all_trades["signal_timestamp_ms"].min()
    )
    last_ts = int(
        all_trades["exit_timestamp_ms"].max()
    )

    prices = pd.read_parquet(
        PRICES_PATH,
        columns=[
            "timestamp", "open", "close",
            "segment_id", "bar_in_segment",
        ],
        filters=[
            ("timestamp", ">=", first_ts),
            ("timestamp", "<=", last_ts),
        ],
    )

    if prices["timestamp"].duplicated().any():
        raise ValueError("Duplicate price timestamps.")

    prices = prices.set_index(
        "timestamp",
        verify_integrity=True,
    )

    for name, trades in trades_by_model.items():
        evaluate_trades(name, trades, prices)

    print("\n=== LIMITATIONS ===")
    print("Same previously selected trades.")
    print("No new risk decisions or portfolio simulation.")
    print("10 bps per side remains hypothetical.")
    print("No real trades, no test data used.")
    print("\nEXECUTION DELAY AUDIT COMPLETED")


if __name__ == "__main__":
    main()
