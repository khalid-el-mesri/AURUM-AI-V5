
from pathlib import Path

import numpy as np

from backtesting.nyse_v3.strategy import StrategyAgentV3


ROOT = Path(__file__).resolve().parents[2]

VALIDATION_PATH = (
    ROOT
    / "artifacts"
    / "experiments"
    / "nyse_v3"
    / "lstm_60m_seed42"
    / "validation_predictions.npz"
)

BAR_MS = 300_000


def load_validation_signals(threshold=0.002):
    """Load validation-only predictions as historical signal events."""

    with np.load(VALIDATION_PATH, allow_pickle=False) as data:
        predictions = data["y_pred"].astype(float)
        signal_times = data["signal_available_at_ms"].astype(np.int64)
        target_times = data["target_available_at_ms"].astype(np.int64)

    if not (
        len(predictions)
        == len(signal_times)
        == len(target_times)
    ):
        raise ValueError("Inconsistent prediction lengths")

    if not np.isfinite(predictions).all():
        raise ValueError("Invalid predictions")

    if np.any(np.diff(signal_times) <= 0):
        raise ValueError("Signal times must be strictly increasing")

    if np.any(signal_times + BAR_MS >= target_times):
        raise ValueError("Invalid entry/exit chronology")

    strategy = StrategyAgentV3(threshold=threshold)
    events = []

    for prediction, signal_time, target_time in zip(
        predictions, signal_times, target_times
    ):
        action = strategy.decide(float(prediction))

        if action == "NO_TRADE":
            continue

        events.append({
            "side": action,
            "predicted_return": float(prediction),
            "signal_available_at_ms": int(signal_time),
            "entry_timestamp_ms": int(signal_time + BAR_MS),
            "exit_timestamp_ms": int(target_time),
        })

    return events
