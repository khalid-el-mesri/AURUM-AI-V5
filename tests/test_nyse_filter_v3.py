import unittest

import pandas as pd

from data_engineering.cleaning.filter_regular_session_nyse_v3 import apply_nyse_filter


def bars(local_times):
    ts = pd.DatetimeIndex(local_times, tz='America/New_York')
    return pd.DataFrame({'timestamp': (ts.tz_convert('UTC').asi8 // 10**6).astype('int64'), 'close': 100.0})


class NYSEFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sessions = [
            ('2019-07-01', '09:30', '16:00'),
            ('2019-07-03', '09:30', '13:00'),
            ('2019-11-04', '09:30', '16:00'),
            ('2019-11-29', '09:30', '13:00'),
        ]
        def ny_to_utc(day, time):
            return pd.Timestamp(f'{day} {time}', tz='America/New_York').tz_convert('UTC')
        cls.cal = pd.DataFrame({
            'market_open': [ny_to_utc(d, o) for d, o, c in sessions],
            'market_close': [ny_to_utc(d, c) for d, o, c in sessions],
        }, index=pd.DatetimeIndex([d for d, _, _ in sessions]))

    def test_early_close_july(self):
        df = bars(['2019-07-03 12:55', '2019-07-03 13:00', '2019-07-03 14:40'])
        kept, excluded = apply_nyse_filter(df, self.cal)
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(excluded), 2)

    def test_early_close_black_friday(self):
        df = bars(['2019-11-29 12:55', '2019-11-29 13:00'])
        kept, excluded = apply_nyse_filter(df, self.cal)
        self.assertEqual((len(kept), len(excluded)), (1, 1))

    def test_normal_close(self):
        df = bars(['2019-07-01 15:55', '2019-07-01 16:00'])
        kept, excluded = apply_nyse_filter(df, self.cal)
        self.assertEqual((len(kept), len(excluded)), (1, 0))

    def test_holiday(self):
        df = bars(['2019-07-04 10:00'])
        kept, excluded = apply_nyse_filter(df, self.cal)
        self.assertEqual((len(kept), len(excluded)), (0, 1))

    def test_weekend(self):
        df = bars(['2019-07-06 10:00'])
        kept, excluded = apply_nyse_filter(df, self.cal)
        self.assertEqual((len(kept), len(excluded)), (0, 0))

    def test_dst(self):
        df = bars(['2019-11-04 09:30', '2019-11-04 15:55'])
        kept, excluded = apply_nyse_filter(df, self.cal)
        self.assertEqual((len(kept), len(excluded)), (2, 0))

    def test_source_unchanged(self):
        df = bars(['2019-07-03 12:55', '2019-07-03 13:00'])
        before = df.copy(deep=True)
        apply_nyse_filter(df, self.cal)
        pd.testing.assert_frame_equal(df, before)

    def test_duplicate_rejected(self):
        df = bars(['2019-07-03 12:55', '2019-07-03 12:55'])
        with self.assertRaises(ValueError):
            apply_nyse_filter(df, self.cal)

    def test_unsorted_rejected(self):
        df = bars(['2019-07-03 13:00', '2019-07-03 12:55'])
        with self.assertRaises(ValueError):
            apply_nyse_filter(df, self.cal)

    def test_missing_timestamp(self):
        with self.assertRaises(ValueError):
            apply_nyse_filter(pd.DataFrame({'close': [1]}), self.cal)

    def test_misaligned_candle(self):
        df = bars(['2019-07-01 09:31'])
        kept, excluded = apply_nyse_filter(df, self.cal)
        self.assertEqual((len(kept), len(excluded)), (0, 1))


if __name__ == '__main__':
    unittest.main()
