"""Ex-ante cohort and regression controls for AURUM-AI 30m/60m V5.

Synthetic tests. Full historical integration requires the user's validation XOM data.
"""
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from backtesting.compare_portfolio_holding_30m_60m_v5 import (
    BAR_MS, EXIT_OFFSETS, prepare_ex_ante_common_signals,
    run_horizon, verify_result, extend_validation_prices,
)


def _input():
    # Five signals during the July 2 normal NYSE session.
    dates = ["2024-07-02 09:45", "2024-07-02 09:50", "2024-07-02 14:45",
             "2024-07-02 14:50", "2024-07-02 15:00"]
    ts = (pd.to_datetime(dates).tz_localize("America/New_York")
          .tz_convert("UTC").astype("int64") // 1_000_000).to_numpy()
    sig = pd.DataFrame({
        "signal_timestamp_ms": ts,
        "entry_execution_timestamp_ms": ts + 2 * BAR_MS,
        "session_date": ["2024-07-02"] * 5,
        "segment_id": [101] * 5,
        "signal_bar_in_segment": [15, 16, 75, 76, 78],
        "cnn_class": [2, 2, 2, 2, 2],
        "logistic_class": [2, 0, 2, 2, 2],
    })
    features = pd.DataFrame({
        "timestamp": ts,
        "session_date": ["2024-07-02"] * 5,
        "segment_id": [101] * 5,
        "bar_in_segment": [15, 16, 75, 76, 78],
        "close": [103., 104., 99., 106., 107.],
        "sma_5": [102.] * 5,
        "ema_10": [101.] * 5,
        "volatility_10": [0.01] * 5,
    })
    return sig, features


def _calendar_only(frame, *, exit_offset_bars):
    assert exit_offset_bars == 13, "60m calendar cutoff must determine common cohort"
    close = pd.Timestamp("2024-07-02 16:00", tz="America/New_York").tz_convert("UTC")
    close_ms = int(close.timestamp() * 1000)
    return frame["signal_timestamp_ms"] + exit_offset_bars * BAR_MS < close_ms


class TestExAnteSelection(unittest.TestCase):
    def test_selects_consensus_trend_with_nyse_cutoff(self):
        signals, features = _input()
        cohort, info = prepare_ex_ante_common_signals(signals, features, _calendar_only)
        self.assertEqual(len(cohort), 2)
        self.assertEqual(info["consensus_and_trend_signals"], 3)
        self.assertEqual(info["excluded_by_60m_nyse_cutoff"], 1)
        self.assertEqual(info["common_ex_ante_eligible_signals"], 2)
        self.assertEqual(cohort["combined_class"].tolist(), [2, 2])
        self.assertEqual(cohort["signal_timestamp_ms"].tolist(),
                         signals["signal_timestamp_ms"].iloc[[0, 3]].tolist())

    def test_selection_does_not_need_any_future_execution_candles(self):
        # No future bar or future OHLC is supplied whatsoever.
        signals, features = _input()
        selected, _ = prepare_ex_ante_common_signals(signals, features, _calendar_only)
        self.assertEqual(len(selected), 2)

    def test_input_frames_are_not_mutated(self):
        signals, features = _input()
        s = signals.copy(deep=True)
        f = features.copy(deep=True)
        prepare_ex_ante_common_signals(signals, features, _calendar_only)
        pd.testing.assert_frame_equal(signals, s)
        pd.testing.assert_frame_equal(features, f)

    def test_segment_context_mismatch_fails_closed(self):
        signals, features = _input()
        features.loc[2, "segment_id"] = 999
        with self.assertRaisesRegex(ValueError, "conflict"):
            prepare_ex_ante_common_signals(signals, features, _calendar_only)

    def test_duplicate_signal_timestamp_rejected(self):
        signals, features = _input()
        signals.loc[1, "signal_timestamp_ms"] = signals.loc[0, "signal_timestamp_ms"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            prepare_ex_ante_common_signals(signals, features, _calendar_only)

    def test_invalid_entry_delay_rejected(self):
        signals, features = _input()
        signals.loc[0, "entry_execution_timestamp_ms"] += BAR_MS
        with self.assertRaisesRegex(ValueError, "entry schedule"):
            prepare_ex_ante_common_signals(signals, features, _calendar_only)

    def test_calendar_mask_length_rejected(self):
        signals, features = _input()
        with self.assertRaisesRegex(ValueError, "calendar mask"):
            prepare_ex_ante_common_signals(signals, features,
                                           lambda f, **kwargs: np.array([True]))

    def test_empty_calendar_cohort_rejected(self):
        signals, features = _input()
        with self.assertRaisesRegex(ValueError, "No 60m"):
            prepare_ex_ante_common_signals(signals, features,
                                           lambda f, **kwargs: np.zeros(len(f), dtype=bool))


class TestHorizonRunner(unittest.TestCase):
    def test_identical_cohort_and_correct_exit_offsets(self):
        signals, features = _input()
        cohort, _ = prepare_ex_ante_common_signals(signals, features, _calendar_only)
        original = cohort.copy(deep=True)
        prices = pd.DataFrame({"timestamp": [1], "open": [100.]})
        observed = {}

        def spy(**kwargs):
            h = 60 if "60m" in kwargs["model_name"] else 30
            observed[h] = kwargs["candidates"].copy()
            self.assertEqual(kwargs["class_column"], "combined_class")
            self.assertIs(kwargs["prices"], prices)
            return (pd.DataFrame(), pd.DataFrame(), {})

        run_horizon(cohort, prices, 30, spy)
        run_horizon(cohort, prices, 60, spy)
        self.assertEqual(observed[30]["signal_timestamp_ms"].tolist(),
                         observed[60]["signal_timestamp_ms"].tolist())
        for h, offset in EXIT_OFFSETS.items():
            self.assertTrue(np.array_equal(observed[h]["scheduled_exit_timestamp_ms"],
                observed[h]["signal_timestamp_ms"] + offset * BAR_MS))
        pd.testing.assert_frame_equal(cohort, original)

    def test_unsupported_horizon_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported horizon"):
            run_horizon(pd.DataFrame(), pd.DataFrame(), 45, Mock())


class TestAccounting(unittest.TestCase):
    def _valid(self):
        t = pd.DataFrame([
            {"timestamp_ms": 100, "action": "BUY", "price": 100., "quantity": 1.,
             "fee_usd": 0.10, "cash_after_usd": 9899.90, "holdings_after": 1.},
            {"timestamp_ms": 200, "action": "SELL", "price": 101., "quantity": 1.,
             "fee_usd": 0.101, "cash_after_usd": 10000.799, "holdings_after": 0.},
        ])
        summary = {"completed_trades": 1, "total_fees_usd": .201,
                   "final_equity_usd": 10000.799, "early_down_exits": 0,
                   "scheduled_exits": 1, "kill_switch_emergency_exits": 0,
                   "halt_timestamp_ms": None}
        return t, summary

    def test_valid_accounting(self):
        trans, summary = self._valid()
        self.assertTrue(verify_result(pd.DataFrame(), trans, summary))

    def test_fee_discrepancy_rejected(self):
        trans, summary = self._valid()
        summary["total_fees_usd"] = 99
        with self.assertRaisesRegex(RuntimeError, "fees"):
            verify_result(pd.DataFrame(), trans, summary)

    def test_halt_blocks_new_buy(self):
        trans, summary = self._valid()
        summary["halt_timestamp_ms"] = 100
        with self.assertRaisesRegex(RuntimeError, "BUY after"):
            verify_result(pd.DataFrame(), trans, summary)

    def test_exit_category_mismatch_rejected(self):
        trans, summary = self._valid()
        summary["scheduled_exits"] = 0
        with self.assertRaisesRegex(RuntimeError, "exit-category"):
            verify_result(pd.DataFrame(), trans, summary)


class TestHistoricalPriceReplay(unittest.TestCase):
    def _fixture(self):
        base = int(pd.Timestamp("2024-07-02T13:45:00Z").timestamp() * 1000)
        signals = pd.DataFrame({
            "signal_timestamp_ms": [base], "session_date": ["2024-07-02"],
        })
        rows = []
        for i in range(14):
            rows.append({"timestamp": base + i * BAR_MS, "open": 100.0 + i / 10,
                         "session_date": "2024-07-02", "segment_id": 1,
                         "bar_in_segment": i + 3})
        full = pd.DataFrame(rows)
        outside = full.iloc[:1].copy()
        outside["session_date"] = "2024-07-03"
        outside["timestamp"] = base + 3 * BAR_MS
        source = pd.concat([full, outside], ignore_index=True)
        reference = full.iloc[:8].copy()
        return signals, reference, source

    def test_extended_replay_includes_exit_without_using_it_for_selection(self):
        signals, reference, source = self._fixture()
        with patch("pandas.read_parquet", return_value=source):
            observed = extend_validation_prices(signals, reference, Path("unused.parquet"))
        self.assertEqual(len(observed), 14)
        self.assertEqual(observed["session_date"].unique().tolist(), ["2024-07-02"])
        self.assertEqual(int(observed.iloc[-1]["timestamp"]),
                         int(signals.iloc[0]["signal_timestamp_ms"]) + 13 * BAR_MS)

    def test_extended_replay_rejects_changed_price(self):
        signals, reference, source = self._fixture()
        source.loc[0, "open"] += 1.0
        with patch("pandas.read_parquet", return_value=source):
            with self.assertRaisesRegex(ValueError, "disagree"):
                extend_validation_prices(signals, reference, Path("unused.parquet"))


if __name__ == "__main__":
    unittest.main()
