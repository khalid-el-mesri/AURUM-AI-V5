
import numpy as np
import pandas as pd

from backtesting.simulate_multi_agent_30m import load_inputs


# ======================================================
# AURUM-AI — B.15.4
# MARKET OPPORTUNITY AUDIT
#
# Original XOM dataset
# Validation only
# Horizons: 15, 30, 60 minutes
# No trades or model retraining
# ======================================================

BAR_MS = 300_000
COST_BPS = 10.0

HORIZONS = [15, 30, 60]


def main():

    print(
        "\n=== AURUM-AI | HORIZON OPPORTUNITY AUDIT ==="
    )

    signals, _ = load_inputs()

    timestamps = signals[
        "signal_timestamp_ms"
    ].to_numpy(dtype=np.int64)

    segments = signals[
        "segment_id"
    ].to_numpy()

    source = (
        "data/02_clean/xom_segmented.parquet"
    )

    # Read prices only around validation signals.
    market = pd.read_parquet(
        source,
        columns=[
            "timestamp",
            "open",
            "segment_id",
            "bar_in_segment",
        ],
        filters=[
            (
                "timestamp",
                ">=",
                int(timestamps.min()),
            ),
            (
                "timestamp",
                "<=",
                int(timestamps.max() + 13 * BAR_MS),
            ),
        ],
    )

    market = market.set_index(
        "timestamp",
        verify_integrity=True,
    )

    signal_rows = market.reindex(timestamps)

    entry_rows = market.reindex(
        timestamps + 2 * BAR_MS
    )

    entry_prices = entry_rows[
        "open"
    ].to_numpy()

    cost = COST_BPS / 10000.0

    common = (
        signal_rows["segment_id"].to_numpy()
        == segments
    )

    common &= (
        entry_rows["segment_id"].to_numpy()
        == segments
    )

    common &= (
        entry_rows["bar_in_segment"].to_numpy()
        == signal_rows["bar_in_segment"].to_numpy() + 2
    )

    common &= np.isfinite(entry_prices)

    print("\nHorizon | Valid | Mean abs move |")
    print("         Long profitable | Abs move > 0.20%")
    print("-" * 65)

    for horizon in HORIZONS:

        # Signal bar starts at t.
        # Its close is known at t + 5 minutes.
        # Delayed entry: t + 10 minutes.
        # Exit: open after the target candle closes.
        exit_offset_bars = horizon // 5 + 1

        exit_rows = market.reindex(
            timestamps + exit_offset_bars * BAR_MS
        )

        exit_prices = exit_rows[
            "open"
        ].to_numpy()

        valid = common.copy()

        valid &= (
            exit_rows["segment_id"].to_numpy()
            == segments
        )

        valid &= (
            exit_rows["bar_in_segment"].to_numpy()
            == (
                signal_rows["bar_in_segment"].to_numpy()
                + exit_offset_bars
            )
        )

        valid &= np.isfinite(exit_prices)

        if not valid.any():
            print(f"{horizon}m: No valid windows")
            continue

        ratio = (
            exit_prices[valid]
            / entry_prices[valid]
        )

        gross_pct = (ratio - 1.0) * 100.0

        net_long_pct = (
            ratio * (1.0 - cost) / (1.0 + cost)
            - 1.0
        ) * 100.0

        mean_absolute = np.mean(
            np.abs(gross_pct)
        )

        long_profitable = (
            100.0 * np.mean(net_long_pct > 0)
        )

        large_movements = (
            100.0 * np.mean(
                np.abs(gross_pct) > 0.20
            )
        )

        print(
            f"{horizon:>3}m    | "
            f"{valid.sum():>5} | "
            f"{mean_absolute:>12.4f}% | "
            f"{long_profitable:>12.2f}% | "
            f"{large_movements:>12.2f}%"
        )

    print("\nIMPORTANT:")
    print("Historical market movements only.")
    print("No strategy or forecast performance measured.")
    print("Signals may overlap.")
    print("Costs are hypothetical.")
    print("No real trading.")
    print("No test labels or test predictions used.")

    print(
        "\nB.15.4 HORIZON OPPORTUNITY AUDIT COMPLETED"
    )


if __name__ == "__main__":
    main()
