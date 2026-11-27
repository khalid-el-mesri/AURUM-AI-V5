"""Synthetic tests only; no real XOM files required."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_FILE = Path(__file__).resolve().parents[1] / "scripts/audit_e5_temporal_integrity.py"
# When copied to the real AURUM-AI/tests folder, the script is in ../scripts.
if not MODULE_FILE.exists():
    MODULE_FILE = Path(__file__).resolve().parents[1] / "scripts/audit_e5_temporal_integrity.py"
spec = importlib.util.spec_from_file_location("audit_e5_temporal_integrity", MODULE_FILE)
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


def sample_df(n=40):
    ts = pd.date_range("2024-07-02 13:30", periods=n, freq="5min", tz="UTC")
    close = np.arange(n, dtype=float) + 100
    price_open = close - 0.2
    df = pd.DataFrame({
        "timestamp": ts.astype("int64") // 1_000_000,
        "session_date": ["2024-07-02"] * n,
        "segment_id": [1] * n,
        "open": price_open, "high": close + 0.1, "low": price_open - 0.1, "close": close,
    })
    g = df["close"]
    df["return_1"] = g.pct_change(fill_method=None)
    df["range_pct"] = (df.high - df.low) / df.open
    df["body_pct"] = (df.close - df.open) / df.open
    df["log_return_1"] = np.log(g).diff()
    df["volatility_10"] = df.log_return_1.rolling(10, min_periods=10).std(ddof=0)
    df["sma_5_distance"] = g / g.rolling(5, min_periods=5).mean() - 1
    return df


class TemporalAuditTests(unittest.TestCase):
    def test_sorted_segments_pass(self):
        self.assertEqual(audit.check_candle_table(sample_df())["bars"], 40)

    def test_duplicate_timestamp_rejected(self):
        df = sample_df()
        df.loc[1, "timestamp"] = df.loc[0, "timestamp"]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            audit.check_candle_table(df)

    def test_gap_within_segment_rejected(self):
        df = sample_df()
        df.loc[15:, "timestamp"] += 300000
        with self.assertRaisesRegex(ValueError, "Non-contiguous"):
            audit.check_candle_table(df)

    def test_invalid_ohlc_rejected(self):
        df = sample_df()
        df.loc[6, "high"] = df.loc[6, "low"] - 1
        with self.assertRaisesRegex(ValueError, "Inconsistent OHLC"):
            audit.check_candle_table(df)

    def test_correct_causal_features_pass(self):
        self.assertEqual(audit.check_features_on_prefixes(sample_df(), samples=5), 5)

    def test_changed_feature_rejected(self):
        df = sample_df()
        df.loc[30:, "return_1"] += .2
        with self.assertRaisesRegex(ValueError, "Non-causal or inconsistent feature"):
            audit.check_features_on_prefixes(df, samples=5)

    def test_calendar_early_close_rejected(self):
        df = sample_df(n=40).copy()
        df["session_date"] = "2024-07-03"
        # 13:00 New York = 17:00 UTC during summer, early close on Jul 3 2024.
        sched = pd.DataFrame({
            "market_open": [pd.Timestamp("2024-07-03 13:30", tz="UTC")],
            "market_close": [pd.Timestamp("2024-07-03 17:00", tz="UTC")],
        }, index=pd.DatetimeIndex(["2024-07-03"]))
        # Test one 17:00 UTC opening => should be rejected.
        df.loc[0, "timestamp"] = pd.Timestamp("2024-07-03 17:00", tz="UTC").value // 1_000_000
        issues = audit.calendar_violations(df, sched)
        self.assertTrue((issues["timestamp"] == df.loc[0, "timestamp"]).any())

    def test_missing_calendar_date_rejected(self):
        df = sample_df(n=2)
        sched = pd.DataFrame(columns=["market_open", "market_close"], index=pd.DatetimeIndex([]))
        self.assertEqual(len(audit.calendar_violations(df, sched)), 2)

    def test_horizon_target_math_pass(self):
        df = sample_df(n=40)
        end = df.timestamp.to_numpy(dtype=np.int64)[15:18]
        target = end + 15 * 60_000
        starts = df.close.to_numpy(dtype=np.float64)[15:18]
        futures = df.close.to_numpy(dtype=np.float64)[18:21]
        with tempfile.TemporaryDirectory() as directory:
            f = Path(directory) / "train_15m_normalized.npz"
            np.savez_compressed(f, X=np.zeros((3,20,6), dtype=np.float32),
                                y=np.log(futures/starts), end_timestamp_ms=end,
                                target_timestamp_ms=target, current_close=starts)
            self.assertEqual(audit.check_horizon_npz(f, df, 15)["windows"], 3)

    def test_horizon_target_wrong_future_rejected(self):
        df = sample_df(n=40)
        end = df.timestamp.to_numpy(dtype=np.int64)[15:18]
        target = end + 15 * 60_000
        starts = df.close.to_numpy(dtype=np.float64)[15:18]
        with tempfile.TemporaryDirectory() as directory:
            f = Path(directory) / "train_15m_normalized.npz"
            np.savez_compressed(f, X=np.zeros((3,20,6), dtype=np.float32),
                                y=np.zeros(3), end_timestamp_ms=end,
                                target_timestamp_ms=target, current_close=starts)
            with self.assertRaisesRegex(ValueError, "Target log returns"):
                audit.check_horizon_npz(f, df, 15)

    def test_require_helper(self):
        audit.require(True, "ok")
        with self.assertRaisesRegex(ValueError, "nope"):
            audit.require(False, "nope")


if __name__ == "__main__":
    unittest.main()
