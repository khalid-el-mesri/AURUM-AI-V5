
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from deep_learning.cnn1d.train_cnn1d import (
    CNN1DForecaster,
)


# ==========================================================
# AURUM-AI — B.15.7
# CNN1D 60M INVESTMENT SIGNAL ANALYSIS
#
# Dataset: XOM only
# Split: VALIDATION only
# Model: Pretrained CNN1D 60m
#
# Historical simulation of individual signals.
# NOT a portfolio backtest.
# NOT genuine walk-forward.
# NO real orders.
# ==========================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "artifacts" / "models"
    / "cnn1d_60m_best.pt"
)

VALIDATION_PATH = (
    ROOT / "data" / "05_model_data"
    / "validation_60m_normalized.npz"
)

MARKET_PATH = (
    ROOT / "data" / "02_clean"
    / "xom_segmented.parquet"
)

REPORT_PATH = (
    ROOT / "reports"
    / "cnn1d_60m_validation.json"
)

OUTPUT_DIR = ROOT / "reports" / "backtesting"

SIGNALS_OUTPUT = (
    OUTPUT_DIR
    / "cnn1d_60m_validation_signal_audit.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "cnn1d_60m_validation_investment_audit.json"
)

BAR_MS = 300_000
COST_BPS_PER_SIDE = 10.0

# Fixed exploratory thresholds expressed as log returns.
# 0.002 corresponds approximately to +0.20%.
THRESHOLDS = [0.0, 0.001, 0.002, 0.003]


def load_validation():

    print("\n=== LOADING 60M VALIDATION ===")

    with np.load(
        VALIDATION_PATH,
        allow_pickle=False,
    ) as data:

        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float64)

        features = [
            str(item)
            for item in data["feature_names"]
        ]

        signal_ts = data[
            "end_timestamp_ms"
        ].astype(np.int64)

        target_ts = data[
            "target_timestamp_ms"
        ].astype(np.int64)

        current_close = data[
            "current_close"
        ].astype(np.float64)

    n = len(X)

    if X.ndim != 3 or X.shape[1:] != (20, 6):
        raise ValueError("Unexpected input dimensions.")

    if not all(
        len(array) == n
        for array in (
            y, signal_ts, target_ts, current_close
        )
    ):
        raise ValueError("Inconsistent validation arrays.")

    if not np.isfinite(X).all():
        raise ValueError("Non-finite model features.")

    if not np.isfinite(y).all():
        raise ValueError("Non-finite validation targets.")

    if not np.isfinite(current_close).all():
        raise ValueError("Invalid reference closing prices.")

    if len(np.unique(signal_ts)) != n:
        raise ValueError("Duplicate signal timestamps.")

    if not np.all(np.diff(signal_ts) > 0):
        raise ValueError("Signals are not chronological.")

    if not np.all(
        target_ts - signal_ts == 12 * BAR_MS
    ):
        raise ValueError(
            "Unexpected 60-minute target alignment."
        )

    print(f"Validation samples: {n}")
    print(f"Input shape: {X.shape}")
    print(f"Features: {features}")

    return (
        X,
        y,
        features,
        signal_ts,
        target_ts,
        current_close,
    )


def generate_predictions(X, features):

    print("\n=== LOADING CNN1D CHECKPOINT ===")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    if int(
        checkpoint["forecast_horizon_minutes"]
    ) != 60:
        raise ValueError("Wrong checkpoint horizon.")

    if int(checkpoint["input_size"]) != X.shape[2]:
        raise ValueError("Checkpoint feature count mismatch.")

    if int(checkpoint["window_size"]) != X.shape[1]:
        raise ValueError("Checkpoint window mismatch.")

    checkpoint_features = [
        str(item)
        for item in checkpoint["feature_names"]
    ]

    if checkpoint_features != features:
        raise ValueError("Feature order mismatch.")

    scale = float(checkpoint["target_scale"])

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid target scale.")

    model = CNN1DForecaster(
        input_size=int(checkpoint["input_size"])
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    predictions = []

    with torch.inference_mode():

        for start in range(0, len(X), 512):

            batch = torch.from_numpy(
                X[start:start + 512]
            )

            outputs = model(batch)

            predictions.append(
                outputs.cpu().numpy().reshape(-1)
                / scale
            )

    predicted = np.concatenate(
        predictions
    ).astype(np.float64)

    if not np.isfinite(predicted).all():
        raise ValueError("Non-finite predictions.")

    print("Checkpoint loaded successfully.")
    print(f"Target scale: {scale}")
    print(f"Predictions generated: {len(predicted)}")

    return predicted


def verify_model_metrics(y, predicted):

    print("\n=== MODEL REPRODUCIBILITY CHECK ===")

    mae = float(np.mean(np.abs(y - predicted)))

    rmse = float(
        np.sqrt(np.mean((y - predicted) ** 2))
    )

    report = json.loads(
        REPORT_PATH.read_text(encoding="utf-8")
    )

    expected = report["cnn1d_validation"]

    if not np.isclose(
        mae,
        float(expected["MAE"]),
        rtol=1e-4,
        atol=1e-8,
    ):
        raise RuntimeError(
            "MAE does not match saved validation report."
        )

    if not np.isclose(
        rmse,
        float(expected["RMSE"]),
        rtol=1e-4,
        atol=1e-8,
    ):
        raise RuntimeError(
            "RMSE does not match saved validation report."
        )

    baseline_mae = float(
        np.mean(np.abs(y))
    )

    print(f"CNN1D MAE: {mae:.8f}")
    print(f"CNN1D RMSE: {rmse:.8f}")
    print(f"Zero Return MAE: {baseline_mae:.8f}")
    print("Saved report reproduction: PASSED")

    return mae, rmse


def prepare_execution_windows(
    signal_ts,
    target_ts,
    current_close,
    predicted,
):

    print("\n=== HISTORICAL EXECUTION ALIGNMENT ===")

    market = pd.read_parquet(
        MARKET_PATH,
        columns=[
            "timestamp",
            "open",
            "close",
            "segment_id",
            "bar_in_segment",
            "session_date",
        ],
        filters=[
            (
                "timestamp",
                ">=",
                int(signal_ts.min()),
            ),
            (
                "timestamp",
                "<=",
                int(signal_ts.max() + 13 * BAR_MS),
            ),
        ],
    )

    if market["timestamp"].duplicated().any():
        raise ValueError("Duplicate market timestamps.")

    market = market.set_index(
        "timestamp",
        verify_integrity=True,
    )

    signal_rows = market.reindex(signal_ts)

    # The signal is based on the close of the
    # candle starting at signal_ts.
    #
    # Delayed entry:
    # signal_ts + 10 minutes, next Bid open.
    #
    # Exit:
    # target candle close is known at
    # target_ts + 5 minutes.
    #
    # Therefore the hypothetical exit uses
    # the Bid open at target_ts + 5 minutes.

    entry_ts = signal_ts + 2 * BAR_MS
    exit_ts = target_ts + BAR_MS

    entry_rows = market.reindex(entry_ts)
    exit_rows = market.reindex(exit_ts)

    signal_segment = signal_rows[
        "segment_id"
    ].to_numpy()

    signal_bar = signal_rows[
        "bar_in_segment"
    ].to_numpy()

    entry_price = entry_rows[
        "open"
    ].to_numpy(dtype=float)

    exit_price = exit_rows[
        "open"
    ].to_numpy(dtype=float)

    valid = (
        np.isfinite(signal_bar)
        & np.isfinite(entry_price)
        & np.isfinite(exit_price)
        & (entry_price > 0)
        & (exit_price > 0)
    )

    valid &= (
        entry_rows["segment_id"].to_numpy()
        == signal_segment
    )

    valid &= (
        exit_rows["segment_id"].to_numpy()
        == signal_segment
    )

    valid &= (
        entry_rows["bar_in_segment"].to_numpy()
        == signal_bar + 2
    )

    valid &= (
        exit_rows["bar_in_segment"].to_numpy()
        == signal_bar + 13
    )

    valid &= (
        entry_rows["session_date"].astype(str).to_numpy()
        == signal_rows["session_date"].astype(str).to_numpy()
    )

    valid &= (
        exit_rows["session_date"].astype(str).to_numpy()
        == signal_rows["session_date"].astype(str).to_numpy()
    )

    observed_close = signal_rows[
        "close"
    ].to_numpy(dtype=float)

    valid &= np.isclose(
        observed_close,
        current_close,
        rtol=1e-6,
        atol=1e-6,
    )

    print(f"Original signals: {len(signal_ts)}")
    print(f"Valid execution windows: {int(valid.sum())}")
    print(f"Excluded windows: {int((~valid).sum())}")

    if not valid.any():
        raise RuntimeError("No valid execution windows.")

    # Ex-post returns: used ONLY for evaluation.
    # Not available to the strategy at signal time.
    cost = COST_BPS_PER_SIDE / 10000.0

    ratio = (
        exit_price[valid] / entry_price[valid]
    )

    gross_pct = (ratio - 1.0) * 100.0

    net_pct = (
        ratio * (1.0 - cost) / (1.0 + cost)
        - 1.0
    ) * 100.0

    frame = pd.DataFrame({
        "signal_timestamp_ms": signal_ts[valid],
        "target_timestamp_ms": target_ts[valid],
        "entry_timestamp_ms": entry_ts[valid],
        "exit_timestamp_ms": exit_ts[valid],
        "predicted_log_return_60m": predicted[valid],
        "entry_bid_open": entry_price[valid],
        "exit_bid_open": exit_price[valid],
        "gross_long_return_pct": gross_pct,
        "net_long_return_pct": net_pct,
    })

    return frame, int((~valid).sum())


def evaluate_thresholds(frame):

    print("\n=== INVESTMENT SIGNAL ANALYSIS ===")

    print(
        "\nThreshold | Signals | Gross mean | "
        "Net mean | Profitable"
    )

    print("-" * 68)

    results = []

    for threshold in THRESHOLDS:

        # Only predictions are used to select
        # potential BUY signals.
        selected = frame[
            frame["predicted_log_return_60m"]
            > threshold
        ]

        count = len(selected)

        if count == 0:

            print(
                f"{threshold:.4f}   | "
                "0 signals | No candidates"
            )

            results.append({
                "threshold_log_return": threshold,
                "signals": 0,
                "average_gross_pct": None,
                "average_net_pct": None,
                "profitable_pct": None,
            })

            continue

        gross = float(
            selected["gross_long_return_pct"].mean()
        )

        net = float(
            selected["net_long_return_pct"].mean()
        )

        profitable = float(
            100.0
            * (
                selected["net_long_return_pct"] > 0
            ).mean()
        )

        print(
            f"{threshold:>9.4f} | "
            f"{count:>7} | "
            f"{gross:>+10.4f}% | "
            f"{net:>+8.4f}% | "
            f"{profitable:>8.2f}%"
        )

        results.append({
            "threshold_log_return": threshold,
            "signals": count,
            "average_gross_pct": gross,
            "average_net_pct": net,
            "profitable_pct": profitable,
        })

    return results


def main():

    print(
        "\n=== AURUM-AI | CNN1D 60M INVESTMENT AUDIT ==="
    )

    # Preserve existing experiments.
    for path in (SIGNALS_OUTPUT, SUMMARY_OUTPUT):
        if path.exists():
            raise FileExistsError(
                f"Existing artifact preserved: {path}"
            )

    (
        X,
        y,
        features,
        signal_ts,
        target_ts,
        current_close,
    ) = load_validation()

    predicted = generate_predictions(
        X,
        features,
    )

    mae, rmse = verify_model_metrics(
        y,
        predicted,
    )

    frame, excluded = prepare_execution_windows(
        signal_ts,
        target_ts,
        current_close,
        predicted,
    )

    print("\n=== PREDICTION DISTRIBUTION ===")

    percentiles = np.quantile(
        predicted,
        [0, 0.10, 0.50, 0.90, 0.99, 1],
    )

    for name, value in zip(
        ["min", "p10", "median", "p90", "p99", "max"],
        percentiles,
    ):
        print(f"{name}: {value:+.7f}")

    results = evaluate_thresholds(frame)

    report = {
        "project": "AURUM-AI",
        "stage": "B.15.7",
        "dataset": "XOM",
        "model": "CNN1D",
        "horizon_minutes": 60,
        "split": "validation",
        "validation_samples": int(len(X)),
        "valid_execution_windows": int(len(frame)),
        "excluded_execution_windows": excluded,
        "validation_mae": mae,
        "validation_rmse": rmse,
        "cost_bps_per_side": COST_BPS_PER_SIDE,
        "threshold_results": results,
        "real_trading": False,
        "test_dataset_used": False,
        "limitations": [
            "Exploratory validation analysis.",
            "Overlapping investment signals.",
            "Not a sequential portfolio backtest.",
            "No position sizing or portfolio risk rules.",
            "Returns depend on idealized Bid open fills.",
            "Transaction costs are hypothetical.",
            "Predicted signal-to-target log return is not "
            "identical to delayed-entry realized return.",
            "Threshold results are not independent "
            "out-of-sample evidence.",
        ],
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        SIGNALS_OUTPUT,
        index=False,
    )

    SUMMARY_OUTPUT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== OUTPUT FILES ===")
    print(SIGNALS_OUTPUT)
    print(SUMMARY_OUTPUT)

    print("\nOriginal XOM dataset preserved.")
    print("Existing models preserved.")
    print("No test dataset used.")
    print("No real trading.")

    print(
        "\nB.15.7 CNN1D 60M INVESTMENT AUDIT COMPLETED"
    )


if __name__ == "__main__":
    main()
