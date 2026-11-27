"""Synthetic, source-independent segmentation unit tests."""
import unittest

import numpy as np
import pandas as pd

from data_engineering.cleaning.create_time_segments_nyse_v3 import make_segments


def market_calendar(dates):
    days = [pd.Timestamp(d) for d in dates]
    opens = [pd.Timestamp(f'{d.date()} 09:30', tz='America/New_York').tz_convert('UTC') for d in days]
    closes = [pd.Timestamp(f'{d.date()} {"13:00" if str(d.date()) in ("2019-07-03", "2019-11-29") else "16:00"}', tz='America/New_York').tz_convert('UTC') for d in days]
    return pd.DataFrame({'market_open': opens, 'market_close': closes}, index=days)


def frame(local_times):
    return pd.DataFrame({
        'timestamp': pd.DatetimeIndex([pd.Timestamp(t, tz='America/New_York') for t in local_times]).tz_convert('UTC').asi8 // 1_000_000,
        'open': [100.0] * len(local_times), 'high': [101.0] * len(local_times),
        'low': [99.0] * len(local_times), 'close': [100.5] * len(local_times),
    })


class SegmentationV3Tests(unittest.TestCase):
    def test_continuous_candles_one_segment(self):
        d = frame(['2024-07-02 09:30', '2024-07-02 09:35', '2024-07-02 09:40'])
        out, audit = make_segments(d, market_calendar(['2024-07-02']))
        self.assertEqual(out.segment_id.tolist(), [1, 1, 1])
        self.assertEqual(out.bar_in_segment.tolist(), [0, 1, 2])
        self.assertEqual(audit['segments'], 1)

    def test_gap_creates_new_segment(self):
        out, audit = make_segments(frame(['2024-07-02 09:30', '2024-07-02 09:40']), market_calendar(['2024-07-02']))
        self.assertEqual(out.segment_id.tolist(), [1, 2])
        self.assertEqual(audit['intraday_interruptions'], 1)

    def test_next_session_creates_new_segment(self):
        out, audit = make_segments(frame(['2024-07-02 15:55', '2024-07-03 09:30']), market_calendar(['2024-07-02', '2024-07-03']))
        self.assertEqual(out.segment_id.tolist(), [1, 2])
        self.assertEqual(audit['intraday_interruptions'], 0)

    def test_july_early_close_last_bar_allowed(self):
        out, _ = make_segments(frame(['2019-07-03 12:55']), market_calendar(['2019-07-03']))
        self.assertEqual(len(out), 1)

    def test_july_early_close_bar_rejected(self):
        with self.assertRaisesRegex(ValueError, 'NYSE-inadmissible'):
            make_segments(frame(['2019-07-03 13:00']), market_calendar(['2019-07-03']))

    def test_black_friday_early_close_bar_rejected(self):
        with self.assertRaisesRegex(ValueError, 'NYSE-inadmissible'):
            make_segments(frame(['2019-11-29 13:00']), market_calendar(['2019-11-29']))

    def test_price_integrity_rejected(self):
        d = frame(['2024-07-02 09:30']); d.loc[0, 'high'] = 50.0
        with self.assertRaisesRegex(ValueError, 'OHLC high'):
            make_segments(d, market_calendar(['2024-07-02']))

    def test_duplicate_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            make_segments(frame(['2024-07-02 09:30', '2024-07-02 09:30']), market_calendar(['2024-07-02']))

    def test_unsorted_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unsorted'):
            make_segments(frame(['2024-07-02 09:35', '2024-07-02 09:30']), market_calendar(['2024-07-02']))

    def test_holiday_rejected(self):
        with self.assertRaisesRegex(ValueError, 'NYSE-inadmissible'):
            make_segments(frame(['2024-07-04 09:30']), market_calendar(['2024-07-02']))

    def test_source_immutable(self):
        d = frame(['2024-07-02 09:30']); original = d.copy(deep=True)
        make_segments(d, market_calendar(['2024-07-02']))
        pd.testing.assert_frame_equal(d, original)

    def test_missing_columns(self):
        with self.assertRaisesRegex(ValueError, 'Missing required'):
            make_segments(pd.DataFrame({'timestamp': [1]}), market_calendar(['2024-07-02']))

    def test_extra_raw_column_preserved(self):
        d = frame(['2024-07-02 09:30']); d['extra'] = ['keep']
        out, _ = make_segments(d, market_calendar(['2024-07-02']))
        self.assertEqual(out['extra'].tolist(), ['keep'])

    def test_nonfinite_prices_rejected(self):
        d = frame(['2024-07-02 09:30']); d.loc[0, 'close'] = np.nan
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            make_segments(d, market_calendar(['2024-07-02']))


if __name__ == '__main__':
    unittest.main()
