"""AURUM-AI E.5: READ-ONLY scientific pipeline audit.

Run from project root:
    python scripts/audit_e5_temporal_integrity.py

Does not modify source data, reports, model artifacts, or test split.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BAR_MS = 300_000
FEATURES = (
    "return_1", "range_pct", "body_pct", "log_return_1",
    "volatility_10", "sma_5_distance",
)


def require(condition: bool, msg: str) -> None:
    if not bool(condition):
        raise ValueError(msg)


def check_candle_table(df: pd.DataFrame) -> dict:
    required = {"timestamp", "session_date", "segment_id", "open", "high", "low", "close", *FEATURES}
    require(required.issubset(df.columns), f"Missing feature data columns: {sorted(required - set(df.columns))}")
    require(len(df) > 0, "Empty feature dataset")
    ts = df["timestamp"].to_numpy(dtype=np.int64)
    require(bool(np.all(np.diff(ts) > 0)), "Timestamps must be strictly increasing")
    groups = df.groupby("segment_id", sort=False)
    require(bool((groups["session_date"].nunique() == 1).all()), "Segment crosses NY session dates")
    diffs = groups["timestamp"].diff().dropna().to_numpy(dtype=np.float64)
    require(bool(np.all(diffs == BAR_MS)), "Non-contiguous intrasegment candles")
    utc = pd.to_datetime(ts, unit="ms", utc=True)
    ny_dates = utc.tz_convert("America/New_York").strftime("%Y-%m-%d")
    require(bool(np.array_equal(ny_dates, df["session_date"].astype(str).to_numpy())),
            "Recorded session_date differs from NY date")
    ohlc = df[["open", "high", "low", "close"]].to_numpy(dtype=np.float64)
    require(bool(np.isfinite(ohlc).all() and np.all(ohlc > 0)), "Invalid OHLC")
    require(bool(np.all(ohlc[:, 1] >= np.maximum(ohlc[:, 0], ohlc[:, 3])) and
                 np.all(ohlc[:, 2] <= np.minimum(ohlc[:, 0], ohlc[:, 3])) and
                 np.all(ohlc[:, 1] >= ohlc[:, 2])), "Inconsistent OHLC extrema")
    return {"bars": len(df), "segments": int(df["segment_id"].nunique()),
            "session_dates": int(df["session_date"].nunique())}


def calendar_violations(df: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Find candle *opening timestamps* outside the actual NYSE sessions.

    schedule must have UTC market_open/market_close and a date index.
    A candle starting exactly at market_close is inadmissible.
    """
    dates = df["session_date"].astype(str)
    calendar = schedule.copy()
    calendar.index = pd.Index(pd.to_datetime(calendar.index).strftime("%Y-%m-%d"))
    starts = pd.to_datetime(dates.map(calendar["market_open"]), utc=True)
    stops = pd.to_datetime(dates.map(calendar["market_close"]), utc=True)
    candle_utc = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    bad = starts.isna().to_numpy() | stops.isna().to_numpy() | (candle_utc < starts).to_numpy() | (candle_utc >= stops).to_numpy()
    # A 5m candle must be completed by the scheduled market close.
    bad |= (candle_utc + pd.Timedelta(minutes=5) > stops).to_numpy()
    return df.loc[bad, ["timestamp", "session_date", "segment_id"]].copy()


def check_features_on_prefixes(df: pd.DataFrame, samples: int = 12) -> int:
    """Recalculate selected feature vectors from historical prefixes only."""
    eligible = df.groupby("segment_id", sort=False).filter(lambda g: len(g) >= 31)
    require(not eligible.empty, "No long enough segments for causal feature checks")
    # Pick reproducible sample rows, avoiding the warm-up period of each segment.
    eligible = eligible[eligible.groupby("segment_id", sort=False).cumcount() >= 30]
    positions = np.unique(np.linspace(0, len(eligible) - 1, min(samples, len(eligible)), dtype=int))
    # Use the ORIGINAL complete segment and cut strictly at the sampled row.
    by_segment = {sid: g for sid, g in df.groupby("segment_id", sort=False)}
    passed = 0
    for position in positions:
        row = eligible.iloc[int(position)]
        group = by_segment[row["segment_id"]]
        offset = int(np.flatnonzero(group.index.to_numpy() == row.name)[0])
        prefix = group.iloc[:offset + 1]
        close = prefix["close"].to_numpy(dtype=np.float64)
        o, hi, lo, last = (float(row[c]) for c in ("open", "high", "low", "close"))
        values = {
            "return_1": last / close[-2] - 1,
            "range_pct": (hi - lo) / o,
            "body_pct": (last - o) / o,
            "log_return_1": np.log(last) - np.log(close[-2]),
            "volatility_10": float(np.diff(np.log(close[-11:])).std(ddof=0)),
            "sma_5_distance": last / close[-5:].mean() - 1,
        }
        for name, expected in values.items():
            actual = float(row[name])
            require(np.isfinite(actual) and np.isclose(actual, expected, rtol=2e-5, atol=1e-8),
                    f"Non-causal or inconsistent feature {name} at timestamp {int(row['timestamp'])}: "
                    f"actual={actual} expected={expected}")
        passed += 1
    return passed


def check_npz_alignment(path: Path, df: pd.DataFrame, scaler: dict) -> dict:
    """Read TRAIN or VALIDATION only, never TEST. No fitting or saving."""
    require(path.is_file(), f"Missing model dataset: {path}")
    with np.load(path, allow_pickle=False) as z:
        require({"X", "y", "end_timestamp_ms", "target_timestamp_ms", "current_close", "feature_names"}.issubset(z.files),
                f"Missing NPZ keys in {path.name}")
        x = z["X"]
        y = z["y"]
        end = z["end_timestamp_ms"].astype(np.int64)
        target = z["target_timestamp_ms"].astype(np.int64)
        current_close = z["current_close"].astype(np.float64)
        names = [str(f) for f in z["feature_names"].tolist()]
    n = len(end)
    require(x.ndim == 3 and x.shape[1:] == (20, 6), f"Unexpected X shape: {path.name}: {x.shape}")
    require(n > 0 and len(x) == len(y) == len(target) == len(current_close), f"NPZ dimensions mismatch: {path.name}")
    require(bool(np.all(np.diff(end) > 0)), f"Unsorted or duplicate end timestamps: {path.name}")
    require(bool(np.all(target - end == BAR_MS)), f"Target gap must be 5min: {path.name}")
    require(bool(np.isfinite(x).all() and np.isfinite(y).all()), f"Non-finite NPZ values: {path.name}")
    require(names == list(FEATURES) == list(scaler["feature_names"]), f"Feature names/order mismatch: {path.name}")
    source = df.set_index("timestamp", verify_integrity=True)
    # End timestamp is the OPEN of the last 5-minute candle.
    idx = source.index.get_indexer(end)
    require(bool((idx >= 0).all()), f"Unknown end timestamp in {path.name}")
    source_last = source.iloc[idx]
    require(bool(np.allclose(source_last["close"].to_numpy(dtype=float), current_close, rtol=1e-6, atol=1e-5)),
            f"Current-close mismatch: {path.name}")
    # Compare normalized last-candle features with original features for a deterministic sample.
    indices = np.unique(np.linspace(0, n-1, min(n, 256), dtype=int))
    means = np.asarray(scaler["mean"], dtype=float)
    stds = np.asarray(scaler["std"], dtype=float)
    require(means.shape == stds.shape == (6,) and bool(np.all(stds > 0)), "Invalid scaler metadata")
    expected_last = (source_last.iloc[indices][list(FEATURES)].to_numpy(dtype=float) - means) / stds
    require(bool(np.allclose(x[indices, -1, :], expected_last, atol=2e-4, rtol=2e-4)),
            f"Normalized final input candle not aligned to feature dataset: {path.name}")
    return {"windows": n, "first_end": int(end[0]), "last_end": int(end[-1]),
            "last_target": int(target[-1]), "first_available_at": int(end[0] + BAR_MS)}



def check_horizon_npz(path: Path, df: pd.DataFrame, horizon_minutes: int) -> dict:
    """Reconcile TRAIN/VALIDATION horizon labels with exact future closes.

    These future closes are used exclusively to VERIFY targets, never to
    construct forecast features or decide which trading signals to execute.
    """
    require(horizon_minutes in (15, 30, 60), "Unsupported horizon")
    require(path.exists(), f"Missing horizon NPZ: {path}")
    with np.load(path, allow_pickle=False) as z:
        require({"X", "y", "end_timestamp_ms", "target_timestamp_ms", "current_close"}.issubset(z.files),
                f"Missing NPZ keys in {path.name}")
        x_shape = z["X"].shape
        end = z["end_timestamp_ms"].astype(np.int64)
        target = z["target_timestamp_ms"].astype(np.int64)
        y = z["y"].astype(np.float64)
        closes = z["current_close"].astype(np.float64)
    n = len(end)
    require(n and x_shape == (n, 20, 6) and len(y) == len(target) == len(closes),
            f"Invalid horizon array dimensions: {path.name}")
    require(bool(np.all(np.diff(end) > 0)), f"Non-chronological horizon windows: {path.name}")
    require(bool(np.all(target - end == horizon_minutes * 60_000)),
            f"Incorrect {horizon_minutes}m offset: {path.name}")
    require(bool(np.isfinite(y).all() and np.isfinite(closes).all() and (closes > 0).all()),
            f"Invalid target/close: {path.name}")
    source = df.set_index("timestamp", verify_integrity=True)
    at_start = source.index.get_indexer(end)
    at_target = source.index.get_indexer(target)
    require(bool(np.all(at_start >= 0) and np.all(at_target >= 0)),
            f"Missing historical price observation for {path.name}")
    start_rows = source.iloc[at_start]
    target_rows = source.iloc[at_target]
    require(bool(np.all(start_rows["segment_id"].to_numpy() == target_rows["segment_id"].to_numpy())),
            f"{horizon_minutes}m target crosses data segment: {path.name}")
    actual_start = start_rows["close"].to_numpy(dtype=np.float64)
    actual_end = target_rows["close"].to_numpy(dtype=np.float64)
    require(bool(np.allclose(closes, actual_start, atol=1e-5, rtol=1e-6)),
            f"Current close does not match price source: {path.name}")
    expected = np.log(actual_end / actual_start)
    require(bool(np.allclose(y, expected, atol=2e-7, rtol=2e-5)),
            f"Target log returns do not match horizon prices: {path.name}")
    return {"windows": n, "first_end": int(end[0]), "last_target": int(target[-1])}


def main() -> int:
    print("=== AURUM-AI E.5 | READ-ONLY TEMPORAL INTEGRITY AUDIT ===")
    print("Scope: features_v2 + TRAIN/VALIDATION 5m NPZ; TEST NOT OPENED")
    features_path = ROOT / "data/03_features/xom_features_v2.parquet"
    stats_path = ROOT / "data/05_model_data/normalization_stats.json"
    try:
        require(features_path.exists(), f"Missing features: {features_path}")
        require(stats_path.exists(), f"Missing scaler: {stats_path}")
        df = pd.read_parquet(features_path)
        counts = check_candle_table(df)
        print(f"PASS: OHLC, chronology, continuous segments: {counts}")
        n = check_features_on_prefixes(df)
        print(f"PASS: {n} historical-prefix feature checks")
        try:
            import pandas_market_calendars as mcal
        except ImportError:
            print("REVIEW: pandas_market_calendars missing; NYSE early-close audit not run")
        else:
            first = str(df["session_date"].iloc[0])
            last = str(df["session_date"].iloc[-1])
            sched = mcal.get_calendar("NYSE").schedule(start_date=first, end_date=last)
            issues = calendar_violations(df, sched)
            if len(issues):
                print(f"REVIEW: {len(issues)} feature bars outside NYSE official schedule")
                print(issues.head(8).to_string(index=False))
            else:
                print("PASS: all existing 5m bars within NYSE official schedule")
        scaler = json.loads(stats_path.read_text(encoding="utf-8"))
        require(scaler.get("fitted_on") == "train_only", "Scaler not declared train-only")
        result = {}
        for name in ("train", "validation"):
            path = ROOT / "data/05_model_data" / f"{name}_normalized.npz"
            result[name] = check_npz_alignment(path, df, scaler)
            print(f"PASS: {name.upper()} 5m window metadata and last-candle alignment: {result[name]['windows']} windows")
        require(result["train"]["last_target"] < result["validation"]["first_end"],
                "TRAIN target timestamp overlaps VALIDATION input timestamps")
        print("PASS: chronology / target-time separation between TRAIN and VALIDATION")
        for minutes in (15, 30, 60):
            for split in ("train", "validation"):
                p = ROOT / "data/05_model_data" / f"{split}_{minutes}m_normalized.npz"
                if p.exists():
                    horizon_result = check_horizon_npz(p, df, minutes)
                    print(f"PASS: {split} {minutes}m label-price alignment: {horizon_result['windows']} windows")
                else:
                    print(f"REVIEW: missing {split} {minutes}m NPZ; this horizon was not verified")
        print("REVIEW (DESIGN): Candle timestamp = OPEN time; last OHLC feature becomes available at OPEN+5min.")
        print("REVIEW (DESIGN): Verify inference signal timestamps and entry times before claiming no leakage.")
        print("NOTE: Test split was NOT opened; no dataset or report was overwritten.")
        print("E.5 AUDIT COMPLETED (findings may require review).")
        return 0
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        print("No dataset or report was overwritten. Test split not opened.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
