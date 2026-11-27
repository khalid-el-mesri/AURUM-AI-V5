
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch

from train_cnn1d_30m_classifier import CNN1DClassifier


# ==================================================
# AURUM-AI — BACKTEST INPUT PREPARATION
# CNN1D vs Logistic Regression
# Validation only — no trades executed
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "data" / "05_model_data"
    / "validation_30m_classification.npz"
)

PRICES_PATH = (
    ROOT / "data" / "02_clean"
    / "xom_segmented.parquet"
)

CNN_PATH = (
    ROOT / "artifacts" / "models"
    / "cnn1d_30m_classifier_best.pt"
)

LOGISTIC_PATH = (
    ROOT / "artifacts" / "models"
    / "logistic_30m_classifier.joblib"
)

OUTPUT_PATH = (
    ROOT / "reports"
    / "backtest_30m_validation_inputs.parquet"
)

AUDIT_PATH = (
    ROOT / "reports"
    / "backtest_30m_validation_inputs_audit.json"
)

BAR_MS = 5 * 60 * 1000
HORIZON_MS = 30 * 60 * 1000
CLASS_NAMES = ["DOWN", "NEUTRAL", "UP"]


def check_output_files():
    for path in (OUTPUT_PATH, AUDIT_PATH):
        if path.exists():
            raise FileExistsError(
                f"Existing file preserved: {path}"
            )


def load_validation():
    with np.load(DATA_PATH, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y_class"].astype(np.int64)
        returns = data["y_log_return"].astype(np.float64)
        end = data["end_timestamp_ms"].astype(np.int64)
        target = data["target_timestamp_ms"].astype(np.int64)
        features = data["feature_names"].tolist()
        threshold = float(data["threshold_log_return"])

    if X.ndim != 3 or X.shape[1:] != (20, 6):
        raise ValueError("Invalid window shape.")

    if not (
        len(X) == len(y) == len(returns)
        == len(end) == len(target)
    ):
        raise ValueError("Validation length mismatch.")

    if not np.isfinite(X).all():
        raise ValueError("Invalid feature values.")

    if not np.isfinite(returns).all():
        raise ValueError("Invalid target returns.")

    if not np.isin(y, [0, 1, 2]).all():
        raise ValueError("Invalid class labels.")

    if not np.all(target - end == HORIZON_MS):
        raise ValueError("Incorrect forecast horizon.")

    if len(np.unique(end)) != len(end):
        raise ValueError("Duplicate signal timestamps.")

    return X, y, returns, end, target, features, threshold


def load_prices(end, target):
    # Read only the date interval needed for validation.
    prices = pd.read_parquet(
        PRICES_PATH,
        columns=[
            "timestamp",
            "open",
            "close",
            "segment_id",
            "bar_in_segment",
            "session_date",
        ],
        filters=[
            ("timestamp", ">=", int(end.min())),
            ("timestamp", "<=", int(target.max())),
        ],
    )

    if prices.empty:
        raise ValueError("No matching price candles.")

    if prices["timestamp"].duplicated().any():
        raise ValueError("Duplicate price timestamps.")

    return prices.set_index(
        "timestamp",
        verify_integrity=True,
    )


def get_aligned_prices(prices, end, target, returns):
    entry_ts = end + BAR_MS

    signal = prices.reindex(end)
    entry = prices.reindex(entry_ts)
    exit_rows = prices.reindex(target)

    for name, frame in [
        ("signal", signal),
        ("entry", entry),
        ("exit", exit_rows),
    ]:
        if frame[
            ["open", "close", "segment_id", "bar_in_segment"]
        ].isna().any().any():
            raise ValueError(
                f"Missing or invalid {name} candles."
            )

    signal_segment = signal["segment_id"].to_numpy()
    entry_segment = entry["segment_id"].to_numpy()
    exit_segment = exit_rows["segment_id"].to_numpy()

    if not (
        np.array_equal(signal_segment, entry_segment)
        and np.array_equal(signal_segment, exit_segment)
    ):
        raise ValueError("A trade would cross a segment.")

    signal_bar = signal["bar_in_segment"].to_numpy()
    entry_bar = entry["bar_in_segment"].to_numpy()
    exit_bar = exit_rows["bar_in_segment"].to_numpy()

    if not np.all(entry_bar - signal_bar == 1):
        raise ValueError("Entry bar is not consecutive.")

    if not np.all(exit_bar - signal_bar == 6):
        raise ValueError("Exit bar alignment failed.")

    signal_close = signal["close"].to_numpy(
        dtype=np.float64
    )

    entry_open = entry["open"].to_numpy(
        dtype=np.float64
    )

    exit_close = exit_rows["close"].to_numpy(
        dtype=np.float64
    )

    if (
        np.any(signal_close <= 0)
        or np.any(entry_open <= 0)
        or np.any(exit_close <= 0)
    ):
        raise ValueError("Non-positive market prices.")

    # The recorded 30m label must match the real
    # future close in our price dataset.
    calculated_return = np.log(
        exit_close / signal_close
    )

    if not np.allclose(
        calculated_return,
        returns,
        rtol=0,
        atol=2e-6,
    ):
        raise ValueError(
            "Future prices do not match stored targets."
        )

    return (
        signal,
        entry,
        exit_rows,
        entry_ts,
        signal_close,
        entry_open,
        exit_close,
    )


def cnn_predict(X, features, threshold):
    checkpoint = torch.load(
        CNN_PATH,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["feature_names"] != features:
        raise ValueError("CNN feature mismatch.")

    if checkpoint["horizon_minutes"] != 30:
        raise ValueError("Incorrect CNN horizon.")

    if not np.isclose(
        checkpoint["neutral_threshold"],
        threshold,
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("CNN threshold mismatch.")

    model = CNN1DClassifier(input_size=6)
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    probabilities = []

    with torch.no_grad():
        for start in range(0, len(X), 512):
            batch = torch.from_numpy(
                X[start:start + 512]
            )

            logits = model(batch)

            probs = torch.softmax(
                logits, dim=1
            ).cpu().numpy()

            probabilities.append(probs)

    probs = np.concatenate(probabilities)

    return probs.argmax(axis=1), probs


def logistic_predict(X, features, threshold):
    # Load only our own locally trained artifact.
    artifact = joblib.load(LOGISTIC_PATH)

    if artifact["feature_names"] != features:
        raise ValueError("Logistic feature mismatch.")

    if not np.isclose(
        artifact["neutral_threshold"],
        threshold,
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("Logistic threshold mismatch.")

    if artifact["horizon_minutes"] != 30:
        raise ValueError("Incorrect Logistic horizon.")

    model = artifact["model"]

    if not np.array_equal(model.classes_, [0, 1, 2]):
        raise ValueError("Unexpected class order.")

    flat = X.reshape(len(X), -1)
    probs = model.predict_proba(flat)

    return probs.argmax(axis=1), probs


def main():
    print("\n=== AURUM-AI | BACKTEST PREPARATION ===")

    check_output_files()
    torch.set_num_threads(4)

    (
        X, y, returns, end, target,
        features, threshold
    ) = load_validation()

    prices = load_prices(end, target)

    (
        signal, entry, exit_rows,
        entry_ts, signal_close,
        entry_open, exit_close
    ) = get_aligned_prices(
        prices, end, target, returns
    )

    cnn_labels, cnn_probs = cnn_predict(
        X, features, threshold
    )

    logistic_labels, logistic_probs = logistic_predict(
        X, features, threshold
    )

    # Cross-check model performance against saved reports.
    cnn_accuracy = float(np.mean(cnn_labels == y))
    logistic_accuracy = float(
        np.mean(logistic_labels == y)
    )

    with open(
        ROOT / "reports"
        / "cnn1d_30m_classification_validation.json",
        encoding="utf-8",
    ) as file:
        cnn_report = json.load(file)

    with open(
        ROOT / "reports"
        / "logistic_30m_classification_validation.json",
        encoding="utf-8",
    ) as file:
        logistic_report = json.load(file)

    if not np.isclose(
        cnn_accuracy,
        cnn_report["classifier_metrics"]["accuracy"],
        atol=1e-8,
        rtol=0,
    ):
        raise ValueError("CNN accuracy mismatch.")

    if not np.isclose(
        logistic_accuracy,
        logistic_report["models"][
            "logistic_regression"
        ]["accuracy"],
        atol=1e-8,
        rtol=0,
    ):
        raise ValueError("Logistic accuracy mismatch.")

    signal_dates = signal[
        "session_date"
    ].astype(str).to_numpy()

    entry_dates = entry[
        "session_date"
    ].astype(str).to_numpy()

    exit_dates = exit_rows[
        "session_date"
    ].astype(str).to_numpy()

    if not (
        np.array_equal(signal_dates, entry_dates)
        and np.array_equal(signal_dates, exit_dates)
    ):
        raise ValueError("Session date mismatch.")

    # Only model outputs available at signal time
    # are used as decision inputs.
    result = pd.DataFrame({
        "signal_timestamp_ms": end,
        "entry_timestamp_ms": entry_ts,
        "exit_timestamp_ms": target,
        "session_date": signal_dates,
        "segment_id": signal[
            "segment_id"
        ].to_numpy(),
        "signal_close_bid": signal_close,
        "entry_open_bid": entry_open,
        "exit_close_bid": exit_close,
        "cnn_class": cnn_labels.astype(np.int64),
        "logistic_class": logistic_labels.astype(np.int64),
        "cnn_score_down": cnn_probs[:, 0],
        "cnn_score_neutral": cnn_probs[:, 1],
        "cnn_score_up": cnn_probs[:, 2],
        "logistic_score_down": logistic_probs[:, 0],
        "logistic_score_neutral": logistic_probs[:, 1],
        "logistic_score_up": logistic_probs[:, 2],
    })

    result = result.sort_values(
        "signal_timestamp_ms"
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    report = {
        "project": "AURUM-AI",
        "stage": "Backtest input preparation",
        "evaluation_split": "validation",
        "windows": int(len(result)),
        "trading_days": int(
            result["session_date"].nunique()
        ),
        "cnn_validation_accuracy": cnn_accuracy,
        "logistic_validation_accuracy": logistic_accuracy,
        "price_source": "Dukascopy XOMUSUSD Bid OHLC",
        "entry_rule": "Next 5m candle Bid open",
        "reference_exit_rule": "30m target candle Bid close",
        "execution_status": "NOT EXECUTED",
        "scores_calibrated": False,
        "real_spread_available": False,
        "test_dataset_used": False,
        "limitations": [
            "Validation used during model selection.",
            "Bid-only quotes, no observed Ask or spread.",
            "Next-open fill assumes ideal execution timing.",
            "Scores are not calibrated probabilities.",
            "No actual trade has been simulated yet.",
            "Not yet a genuine walk-forward evaluation.",
        ],
    }

    with open(
        AUDIT_PATH, "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    print("\n=== VALIDATION ALIGNMENT ===")
    print(f"Windows: {len(result)}")
    print(f"Trading days: {report['trading_days']}")
    print("Entry: next 5m candle open")
    print("Reference exit: 30m target candle close")
    print("Same segment: VERIFIED")
    print("Future target consistency: VERIFIED")

    print("\n=== FORECAST MODELS ===")
    print(f"CNN1D Accuracy: {cnn_accuracy:.4f}")
    print(f"Logistic Accuracy: {logistic_accuracy:.4f}")

    for name, labels in [
        ("CNN1D", cnn_labels),
        ("Logistic", logistic_labels),
    ]:
        counts = np.bincount(labels, minlength=3)
        print(
            f"{name} predictions: "
            f"DOWN={counts[0]}, "
            f"NEUTRAL={counts[1]}, "
            f"UP={counts[2]}"
        )

    print("\n=== FILES SAVED ===")
    print(OUTPUT_PATH)
    print(AUDIT_PATH)

    print("\nNo trades executed.")
    print("Test dataset was NOT used.")
    print("\nBACKTEST INPUT PREPARATION COMPLETED")


if __name__ == "__main__":
    main()
