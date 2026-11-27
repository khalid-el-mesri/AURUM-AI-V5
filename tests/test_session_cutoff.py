"""Focused tests for calendar-only session cutoff, plus legacy comparison."""

import unittest

import pandas as pd

from backtesting.session_cutoff import session_cutoff_mask


def one_signal(day: str, new_york_time: str) -> pd.DataFrame:
    dt = pd.Timestamp(f"{day} {new_york_time}", tz="America/New_York")
    return pd.DataFrame(
        {"signal_timestamp_ms": [dt.value // 1_000_000], "session_date": [day]},
        index=["sample"],
    )


class TestNYSESessionCutoff(unittest.TestCase):
    def test_normal_day_last_admissible_30m_signal(self):
        self.assertTrue(bool(session_cutoff_mask(one_signal("2024-07-02", "15:20")).iloc[0]))

    def test_normal_day_first_inadmissible_30m_signal(self):
        self.assertFalse(bool(session_cutoff_mask(one_signal("2024-07-02", "15:25")).iloc[0]))

    def test_early_close_last_admissible_30m_signal(self):
        self.assertTrue(bool(session_cutoff_mask(one_signal("2024-07-03", "12:20")).iloc[0]))

    def test_early_close_first_inadmissible_30m_signal(self):
        self.assertFalse(bool(session_cutoff_mask(one_signal("2024-07-03", "12:25")).iloc[0]))

    def test_normal_day_60m_exit(self):
        self.assertTrue(bool(session_cutoff_mask(one_signal("2024-07-02", "14:50"), exit_offset_bars=13).iloc[0]))
        self.assertFalse(bool(session_cutoff_mask(one_signal("2024-07-02", "14:55"), exit_offset_bars=13).iloc[0]))

    def test_early_close_60m_exit(self):
        self.assertTrue(bool(session_cutoff_mask(one_signal("2024-07-03", "11:50"), exit_offset_bars=13).iloc[0]))
        self.assertFalse(bool(session_cutoff_mask(one_signal("2024-07-03", "11:55"), exit_offset_bars=13).iloc[0]))

    def test_signal_before_open_is_rejected(self):
        self.assertFalse(bool(session_cutoff_mask(one_signal("2024-07-02", "09:20")).iloc[0]))

    def test_holiday_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "No NYSE trading session"):
            session_cutoff_mask(one_signal("2024-07-04", "10:00"))

    def test_input_is_not_mutated_and_index_preserved(self):
        signals = one_signal("2024-07-02", "15:20")
        before = signals.copy(deep=True)
        result = session_cutoff_mask(signals)
        pd.testing.assert_frame_equal(signals, before)
        self.assertTrue(result.index.equals(signals.index))
        self.assertEqual(result.dtype, bool)

    def test_invalid_offset_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            session_cutoff_mask(one_signal("2024-07-02", "15:20"), exit_offset_bars=0)

    def test_missing_columns_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing required"):
            session_cutoff_mask(pd.DataFrame({"session_date": ["2024-07-02"]}))

    def test_current_validation_matches_existing_filter(self):
        """Read-only regression: compare to the legacy 30m candidate set."""
        from backtesting.simulate_multi_agent_30m import INPUT_PATH, load_inputs

        legacy_candidates, _ = load_inputs()
        all_signals = pd.read_parquet(INPUT_PATH)
        eligible = session_cutoff_mask(all_signals, exit_offset_bars=7)
        legacy = all_signals["signal_timestamp_ms"].isin(
            legacy_candidates["signal_timestamp_ms"]
        )
        self.assertEqual(int(eligible.sum()), int(legacy.sum()))
        self.assertEqual(int((eligible != legacy).sum()), 0)


if __name__ == "__main__":
    unittest.main()
