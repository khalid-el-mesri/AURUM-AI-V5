"""Synthetic, no-user-data tests for standalone V3 features."""
import unittest

import numpy as np
import pandas as pd

from data_engineering.features.build_features_nyse_v3 import build_feature_frame, summarize, BAR_MS, FEATURE_COLUMNS


def sample(n=15, two_sessions=False):
    base = pd.Timestamp('2024-07-02 09:30', tz='America/New_York').tz_convert('UTC').value // 1_000_000
    t = [int(base + BAR_MS * i) for i in range(n)]
    group = [1] * n
    pos = list(range(n))
    day = ['2024-07-02'] * n
    if two_sessions:
        next_base = pd.Timestamp('2024-07-03 09:30', tz='America/New_York').tz_convert('UTC').value // 1_000_000
        t.extend([int(next_base + BAR_MS * i) for i in range(n)])
        group.extend([2] * n)
        pos.extend(range(n))
        day.extend(['2024-07-03'] * n)
    p = np.linspace(100.0, 103.0, len(t))
    return pd.DataFrame({
        'timestamp': np.array(t, dtype='int64'), 'open': p,
        'high': p + 0.3, 'low': p - 0.3, 'close': p + 0.1,
        'session_date': day, 'segment_id': group, 'bar_in_segment': pos,
    })


class FeaturesV3Tests(unittest.TestCase):
    def test_pure_no_change_to_source(self):
        inp = sample()
        saved = inp.copy(deep=True)
        build_feature_frame(inp)
        pd.testing.assert_frame_equal(inp, saved)

    def test_preserves_input_columns_and_row_count(self):
        inp = sample()
        out = build_feature_frame(inp)
        self.assertEqual(len(out), len(inp))
        pd.testing.assert_frame_equal(out[inp.columns], inp)

    def test_feature_names_match_legacy_v2(self):
        out = build_feature_frame(sample())
        self.assertTrue(all(x in out.columns for x in FEATURE_COLUMNS))

    def test_return_and_log_return_math(self):
        out = build_feature_frame(sample())
        self.assertTrue(np.isnan(out['return_1'].iloc[0]))
        self.assertAlmostEqual(out['return_1'].iloc[1], out['close'].iloc[1] / out['close'].iloc[0] - 1)
        self.assertAlmostEqual(out['log_return_1'].iloc[1], np.log(out['close'].iloc[1] / out['close'].iloc[0]))

    def test_sma_and_ema(self):
        out = build_feature_frame(sample())
        self.assertTrue(out['sma_5'].iloc[:4].isna().all())
        self.assertAlmostEqual(out['sma_5'].iloc[4], out['close'].iloc[:5].mean())
        self.assertAlmostEqual(out['ema_10'].iloc[0], out['close'].iloc[0])

    def test_volatility_warmup(self):
        out = build_feature_frame(sample())
        self.assertTrue(out['volatility_10'].iloc[:10].isna().all())
        self.assertAlmostEqual(out['volatility_10'].iloc[10], np.std(out['log_return_1'].iloc[1:11], ddof=0))

    def test_segment_resets(self):
        out = build_feature_frame(sample(two_sessions=True))
        self.assertTrue(np.isnan(out['return_1'].iloc[15]))
        self.assertTrue(np.isnan(out['sma_5'].iloc[15]))
        self.assertAlmostEqual(out['ema_10'].iloc[15], out['close'].iloc[15])

    def test_prefix_invariance_future_candles_not_used(self):
        inp = sample(n=25)
        whole = build_feature_frame(inp)
        prefix = build_feature_frame(inp.iloc[:16].copy())
        pd.testing.assert_frame_equal(whole[FEATURE_COLUMNS].iloc[:16], prefix[FEATURE_COLUMNS])

    def test_feature_availability_is_at_candle_end(self):
        out = build_feature_frame(sample())
        self.assertTrue(np.array_equal(out.feature_available_at_ms, out.timestamp + BAR_MS))

    def test_duplicates_rejected(self):
        x = sample()
        x.loc[5, 'timestamp'] = x.loc[4, 'timestamp']
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            build_feature_frame(x)

    def test_gap_inside_segment_rejected(self):
        x = sample()
        x.loc[5:, 'timestamp'] += BAR_MS
        with self.assertRaisesRegex(ValueError, 'Nonconsecutive'):
            build_feature_frame(x)

    def test_out_of_order_rejected(self):
        x = sample().iloc[::-1].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            build_feature_frame(x)

    def test_ohlc_violation_rejected(self):
        x = sample()
        x.loc[3, 'high'] = x.loc[3, 'low'] - 1
        with self.assertRaisesRegex(ValueError, 'OHLC high'):
            build_feature_frame(x)

    def test_invalid_bar_position_rejected(self):
        x = sample()
        x.loc[4, 'bar_in_segment'] = 50
        with self.assertRaisesRegex(ValueError, 'bar_in_segment'):
            build_feature_frame(x)

    def test_nan_prices_rejected(self):
        x = sample()
        x.loc[4, 'close'] = np.nan
        with self.assertRaisesRegex(ValueError, 'Invalid OHLC'):
            build_feature_frame(x)

    def test_summary_does_not_report_model_training(self):
        x = sample()
        s = summarize(x, build_feature_frame(x))
        self.assertFalse(s['model_retrained'])
        self.assertFalse(s['test_npz_opened'])
        self.assertEqual(s['output_rows'], len(x))


if __name__ == '__main__':
    unittest.main()
