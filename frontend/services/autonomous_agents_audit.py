"""Read-only historical Autonomous Agents data for AURUM-AI V5.

No order execution; never modifies files or uses test predictions.
The regression StrategyAgent and the classification ExecutionAgent are different
pipelines, and must not be connected without a validated adapter.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd


EXPERIMENT_RE = re.compile(r"^(lstm|gru|cnn1d)_(5|15|30|60)m_seed\d+$")
REQUIRED = ("y_pred", "y_true", "end_timestamp_ms", "target_timestamp_ms",
            "signal_available_at_ms", "target_available_at_ms", "current_close", "next_close")


def available_experiments(root: Path) -> dict[tuple[str, int], Path]:
    root = Path(root)
    result = {}
    if not root.is_dir():
        return result
    for path in sorted(root.glob("*/validation_predictions.npz")):
        match = EXPERIMENT_RE.fullmatch(path.parent.name)
        if match:
            model, horizon = match.group(1).upper(), int(match.group(2))
            result[(model, horizon)] = path
    return result


def load_validation_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        missing = set(REQUIRED).difference(data.files)
        if missing:
            raise ValueError(f"Missing fields in validation NPZ: {sorted(missing)}")
        arrays = {key: np.asarray(data[key]) for key in REQUIRED}
    n = len(arrays["y_pred"])
    if n == 0:
        raise ValueError("Empty validation predictions")
    if any(x.ndim != 1 or len(x) != n for x in arrays.values()):
        raise ValueError("Inconsistent array shapes in validation predictions")
    if any(not np.issubdtype(arrays[k].dtype, np.number) for k in REQUIRED):
        raise ValueError("Non-numeric prediction data")
    if any(not np.isfinite(arrays[k]).all() for k in REQUIRED):
        raise ValueError("Non-finite values in predictions")
    if (arrays["current_close"] <= 0).any() or (arrays["next_close"] <= 0).any():
        raise ValueError("Prices must be strictly positive")
    return arrays


def timing_audit(arrays: dict[str, np.ndarray], horizon: int) -> dict[str, int]:
    end = arrays["end_timestamp_ms"]
    target = arrays["target_timestamp_ms"]
    signal = arrays["signal_available_at_ms"]
    available = arrays["target_available_at_ms"]
    expected = horizon * 60_000
    return {
        "observations": len(end),
        "non_monotonic_signal_times": int(np.sum(np.diff(signal) < 0)),
        "duplicate_signal_times": int(len(signal) - len(np.unique(signal))),
        "target_at_or_before_signal": int(np.sum(available <= signal)),
        "signal_before_feature_end": int(np.sum(signal < end)),
        "target_before_target_bar": int(np.sum(available < target)),
        "wrong_target_horizon": int(np.sum(target - end != expected)),
    }


def signal_decisions(arrays: dict[str, np.ndarray], threshold: float) -> pd.DataFrame:
    """Apply the *existing* regression StrategyAgent only; no virtual trades."""
    from agents.strategy_agent import StrategyAgent

    agent = StrategyAgent(threshold=float(threshold))
    preds = arrays["y_pred"]
    decisions = [agent.decide(float(value)) for value in preds]
    df = pd.DataFrame({
        "signal_available_utc": pd.to_datetime(arrays["signal_available_at_ms"], unit="ms", utc=True),
        "target_available_utc": pd.to_datetime(arrays["target_available_at_ms"], unit="ms", utc=True),
        "predicted_log_return": preds.astype(float),
        "actual_log_return": arrays["y_true"].astype(float),
        "action_proposed": [d.action for d in decisions],
        "strategy_reason": [d.reason for d in decisions],
        "current_close_usd": arrays["current_close"].astype(float),
        "actual_target_close_usd": arrays["next_close"].astype(float),
        "predicted_target_close_usd": (arrays["current_close"] * np.exp(preds)).astype(float),
    })
    return df


def signal_summary(frame: pd.DataFrame) -> dict:
    counts = Counter(frame["action_proposed"].tolist())
    n = len(frame)
    err = frame["predicted_log_return"] - frame["actual_log_return"]
    baseline = frame["actual_log_return"]
    return {
        "observations": n,
        "BUY": counts["BUY"], "SELL": counts["SELL"], "HOLD": counts["HOLD"],
        "mae_log_return": float(np.abs(err).mean()),
        "zero_return_mae": float(np.abs(baseline).mean()),
        "direction_accuracy_percent": float((np.sign(frame["predicted_log_return"]) == np.sign(frame["actual_log_return"])).mean() * 100),
    }


def price_target_consistency(arrays: dict[str, np.ndarray], atol: float = 1e-6) -> int:
    implied = np.log(arrays["next_close"] / arrays["current_close"])
    return int((np.abs(implied - arrays["y_true"]) > atol).sum())


def _get(obj: object, *keys, default=None):
    cur = obj
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key, default)
    return cur


def parse_historical_ledger(raw: object) -> pd.DataFrame:
    """Extract readable decisions; DOES NOT verify block hashes or chain validity."""
    if isinstance(raw, list):
        blocks = raw
    elif isinstance(raw, dict):
        blocks = next((raw[k] for k in ("chain", "blocks", "entries", "ledger") if isinstance(raw.get(k), list)), None)
        if blocks is None:
            raise ValueError("JSON ledger has no recognized list of blocks")
    else:
        raise ValueError("Unexpected ledger format")
    records = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        payload = block.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}
        records.append({
            "index": block.get("index"),
            "timestamp_utc": block.get("timestamp_utc"),
            "event_type": block.get("event_type", block.get("type", "")),
            "strategy_action": _get(payload, "strategy", "action", default=""),
            "risk_status": _get(payload, "risk", "status", default=""),
            "execution_status": _get(payload, "execution", "status", default=""),
            "forecast_class": _get(payload, "forecast", "predicted_class", default=""),
            "block_hash": block.get("hash", ""),
        })
    return pd.DataFrame.from_records(records, columns=["index", "timestamp_utc", "event_type", "strategy_action", "risk_status", "execution_status", "forecast_class", "block_hash"])


def read_historical_ledger(path: Path) -> pd.DataFrame:
    with Path(path).open("r", encoding="utf-8") as f:
        return parse_historical_ledger(json.load(f))
