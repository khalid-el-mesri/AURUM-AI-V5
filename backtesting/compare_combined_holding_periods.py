
import numpy as np
import pandas as pd

from backtesting.simulate_multi_agent_30m import load_inputs


# =====================================================
# AURUM-AI — B.17.5
# SAME SIGNALS, DIFFERENT HOLDING HORIZONS
#
# Dataset: XOM
# Split: Validation only
# BUY: CNN1D UP + Logistic UP + Technical Trend UP
#
# Compare 30m and 60m scheduled exits.
# No portfolio simulation.
# =====================================================

BAR_MS = 300_000
COST_BPS = 10.0


def main():

    print(
        "\n=== AURUM-AI | B.17.5 HOLDING PERIOD AUDIT ==="
    )

    signals, _ = load_inputs()

    features = pd.read_parquet(
        "data/03_features/xom_features_v2.parquet",
        columns=[
            "timestamp",
            "segment_id",
            "close",
            "sma_5",
            "ema_10",
            "volatility_10",
            "bar_in_segment",
        ],
    ).set_index("timestamp", verify_integrity=True)

    aligned = features.reindex(
        signals["signal_timestamp_ms"].to_numpy()
    )

    same_segment = (
        aligned["segment_id"].to_numpy()
        == signals["segment_id"].to_numpy()
    )

    trend_up = (
        aligned["close"].gt(aligned["sma_5"])
        & aligned["sma_5"].gt(aligned["ema_10"])
        & aligned["volatility_10"].notna()
        & same_segment
    ).to_numpy()

    consensus_up = (
        signals["cnn_class"].eq(2).to_numpy()
        & signals["logistic_class"].eq(2).to_numpy()
    )

    combined = signals.loc[
        consensus_up & trend_up
    ].copy()

    print(f"Combined BUY signals: {len(combined)}")

    if combined.empty:
        raise RuntimeError("No combined signals.")

    timestamps = combined[
        "signal_timestamp_ms"
    ].to_numpy(dtype=np.int64)

    segment_ids = combined[
        "segment_id"
    ].to_numpy()

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

    validation_days = set(
        signals["session_date"].astype(str)
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

    signal = market.reindex(timestamps)

    entry = market.reindex(
        timestamps + 2 * BAR_MS
    )

    exit_30 = market.reindex(
        timestamps + 7 * BAR_MS
    )

    exit_60 = market.reindex(
        timestamps + 13 * BAR_MS
    )

    signal_bar = signal[
        "bar_in_segment"
    ].to_numpy()

    valid = (
        signal["segment_id"].to_numpy()
        == segment_ids
    )

    valid &= np.isfinite(signal_bar)

    for row, offset in (
        (entry, 2),
        (exit_30, 7),
        (exit_60, 13),
    ):

        valid &= (
            row["segment_id"].to_numpy()
            == segment_ids
        )

        valid &= (
            row["bar_in_segment"].to_numpy()
            == signal_bar + offset
        )

        valid &= (
            row["session_date"].astype(str).to_numpy()
            == combined["session_date"].astype(str).to_numpy()
        )

        prices = row["open"].to_numpy(dtype=float)

        valid &= (
            np.isfinite(prices)
            & (prices > 0)
        )

    print(f"Comparable signals: {int(valid.sum())}")
    print(
        "Excluded (no common 60m window): "
        f"{int((~valid).sum())}"
    )

    if not valid.any():
        raise RuntimeError("No comparable signals.")

    entry_price = entry["open"].to_numpy()[valid]

    cost = COST_BPS / 10000.0

    print("\n=== SAME SIGNALS — 30M VS 60M ===")

    for horizon, exit_rows in (
        (30, exit_30),
        (60, exit_60),
    ):

        exit_price = exit_rows[
            "open"
        ].to_numpy()[valid]

        ratio = exit_price / entry_price

        gross_pct = (
            ratio - 1.0
        ) * 100.0

        net_pct = (
            ratio * (1.0 - cost) / (1.0 + cost)
            - 1.0
        ) * 100.0

        print(f"\n--- {horizon} MIN ---")

        print(
            "Average gross return: "
            f"{np.mean(gross_pct):+.4f}%"
        )

        print(
            "Average net return: "
            f"{np.mean(net_pct):+.4f}%"
        )

        print(
            "Profitable opportunities: "
            f"{100 * np.mean(net_pct > 0):.2f}%"
        )

        print(
            "Average absolute movement: "
            f"{np.mean(np.abs(gross_pct)):.4f}%"
        )

    print("\nValidation only.")
    print("Same entry signals for both horizons.")
    print("No overlapping portfolio positions simulated.")
    print("30m classifier signals reused for 60m exits.")
    print("Trading costs are hypothetical.")
    print("No real trading.")
    print("No test data used.")

    print(
        "\nB.17.5 HOLDING PERIOD AUDIT COMPLETED"
    )


if __name__ == "__main__":
    main()
