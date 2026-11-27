
import numpy as np
import pandas as pd

from backtesting.simulate_multi_agent_30m import load_inputs


# =====================================================
# AURUM-AI — B.15.5
# FAIR HORIZON COMPARISON
#
# Same XOM validation signals for:
# 15m / 30m / 60m
#
# No trading, no retraining, no test evaluation.
# =====================================================

BAR_MS = 300_000
COST_BPS_PER_SIDE = 10.0

HORIZONS = {
    15: 4,
    30: 7,
    60: 13,
}


def main():

    print("\n=== AURUM-AI | FAIR HORIZON COMPARISON ===")

    signals, _ = load_inputs()

    timestamps = signals[
        "signal_timestamp_ms"
    ].to_numpy(dtype=np.int64)

    segments = signals[
        "segment_id"
    ].to_numpy()

    validation_days = set(
        signals["session_date"].astype(str)
    )

    market = pd.read_parquet(
        "data/02_clean/xom_segmented.parquet",
        columns=[
            "timestamp",
            "open",
            "segment_id",
            "bar_in_segment",
            "session_date",
        ],
        filters=[
            ("timestamp", ">=", int(timestamps.min())),
            (
                "timestamp",
                "<=",
                int(timestamps.max() + 13 * BAR_MS),
            ),
        ],
    )

    market = market[
        market["session_date"].astype(str).isin(
            validation_days
        )
    ]

    market = market.set_index(
        "timestamp",
        verify_integrity=True,
    )

    signal_rows = market.reindex(timestamps)

    entry_rows = market.reindex(
        timestamps + 2 * BAR_MS
    )

    signal_bars = signal_rows[
        "bar_in_segment"
    ].to_numpy()

    entry_prices = entry_rows["open"].to_numpy()

    valid = (
        (signal_rows["segment_id"].to_numpy() == segments)
        & (entry_rows["segment_id"].to_numpy() == segments)
        & (
            entry_rows["bar_in_segment"].to_numpy()
            == signal_bars + 2
        )
        & np.isfinite(entry_prices)
        & (entry_prices > 0)
    )

    exits = {}

    for horizon, offset in HORIZONS.items():

        exit_rows = market.reindex(
            timestamps + offset * BAR_MS
        )

        exit_prices = exit_rows["open"].to_numpy()

        exits[horizon] = exit_prices

        valid &= (
            exit_rows["segment_id"].to_numpy()
            == segments
        )

        valid &= (
            exit_rows["bar_in_segment"].to_numpy()
            == signal_bars + offset
        )

        valid &= np.isfinite(exit_prices)

        valid &= exit_prices > 0

    print(f"\nOriginal candidate signals: {len(signals)}")
    print(f"Common valid signals: {int(valid.sum())}")
    print(f"Excluded from comparison: {int((~valid).sum())}")

    if not valid.any():
        raise RuntimeError("No common execution windows.")

    cost = COST_BPS_PER_SIDE / 10000.0

    print("\n=== SAME SIGNALS — ALL HORIZONS ===")

    for horizon in HORIZONS:

        ratio = (
            exits[horizon][valid]
            / entry_prices[valid]
        )

        gross_pct = (ratio - 1.0) * 100.0

        net_pct = (
            ratio * (1.0 - cost) / (1.0 + cost)
            - 1.0
        ) * 100.0

        print(f"\n--- HORIZON {horizon} MIN ---")

        print(f"Signals: {int(valid.sum())}")

        print(
            "Mean absolute move: "
            f"{np.mean(np.abs(gross_pct)):.4f}%"
        )

        print(
            "Mean gross LONG return: "
            f"{np.mean(gross_pct):+.4f}%"
        )

        print(
            "Mean net LONG return: "
            f"{np.mean(net_pct):+.4f}%"
        )

        print(
            "Profitable LONG windows: "
            f"{100 * np.mean(net_pct > 0):.2f}%"
        )

        print(
            "Absolute moves > 0.20%: "
            f"{100 * np.mean(np.abs(gross_pct) > 0.20):.2f}%"
        )

    print("\nIMPORTANT:")
    print("Identical signals compared across horizons.")
    print("These are market opportunities, not model trades.")
    print("Overlapping windows are possible.")
    print("Fees are hypothetical.")
    print("No real transactions.")
    print("No test dataset evaluated.")

    print("\nB.15.5 FAIR HORIZON COMPARISON COMPLETED")


if __name__ == "__main__":
    main()
