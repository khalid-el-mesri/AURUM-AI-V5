"""Synthetic unit tests for the versioned NYSE V3 Train+Validation windows."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from data_engineering.windows import build_windows_nyse_v3 as mod


def sample_frame(sessions=8, bars=55):
    records = []
    dates = pd.bdate_range('2024-06-24', periods=sessions)
    for k, date in enumerate(dates):
        base = pd.Timestamp(f'{date.date()} 09:30', tz='America/New_York')
        for i in range(bars):
            open_time = base + pd.Timedelta(minutes=5 * i)
            ms = open_time.tz_convert('UTC').value // 1_000_000
            price = 100 + k * 2 + 0.1 * i
            records.append({
                'timestamp': int(ms), 'session_date': str(date.date()),
                'segment_id': k + 1, 'bar_in_segment': i, 'close': price,
                'feature_available_at_ms': int(ms) + mod.BAR_MS,
                'return_1': price / (price - .1) - 1 if i else np.nan,
                'range_pct': .002 + k * .0001, 'body_pct': .001 + i * .00001,
                'log_return_1': np.log(price / (price - .1)) if i else np.nan,
                'volatility_10': .002 if i >= 10 else np.nan,
                'sma_5_distance': .003 if i >= 4 else np.nan,
            })
    return pd.DataFrame.from_records(records)


class TestNYSEWindowsV3(unittest.TestCase):
    def setUp(self):
        self.df = sample_frame()
        self.splits = mod.chronological_splits(self.df)

    def test_valid_dataset(self):
        mod.validate_features(self.df)

    def test_split_chronologically(self):
        self.assertEqual(tuple(map(len, self.splits.values())), (5, 1, 2))
        self.assertLess(self.splits['train'][-1], self.splits['validation'][0])
        self.assertLess(self.splits['validation'][-1], self.splits['reserved_test_dates'][0])

    def test_5m_window_shape_and_counts(self):
        d = mod.make_windows(self.df, self.splits['train'], 5)
        self.assertEqual(d['X'].shape, (5 * (55 - 30), 20, 6))
        self.assertEqual(d['y'].dtype, np.float32)
        self.assertEqual(list(d['feature_names']), list(mod.FEATURES))

    def test_all_horizons_and_correct_target_timestamp(self):
        for horizon in mod.HORIZONS:
            d = mod.make_windows(self.df, self.splits['validation'], horizon)
            expected_rows = 55 - 29 - (horizon // 5)
            self.assertEqual(len(d['y']), expected_rows)
            self.assertTrue(np.all(d['target_timestamp_ms'] - d['end_timestamp_ms'] == horizon * 60000))
            self.assertTrue(np.allclose(d['y'], np.log(d['next_close'] / d['current_close']), atol=1e-8))

    def test_signal_and_label_availability(self):
        d = mod.make_windows(self.df, self.splits['train'], 60)
        self.assertTrue(np.array_equal(d['signal_available_at_ms'], d['end_timestamp_ms'] + mod.BAR_MS))
        self.assertTrue(np.array_equal(d['target_available_at_ms'], d['target_timestamp_ms'] + mod.BAR_MS))
        self.assertTrue(np.all(d['target_available_at_ms'] > d['signal_available_at_ms']))

    def test_no_cross_session_window(self):
        d = mod.make_windows(self.df, self.splits['train'], 60)
        f = pd.to_datetime(d['end_timestamp_ms'], unit='ms', utc=True).tz_convert('America/New_York')
        t = pd.to_datetime(d['target_timestamp_ms'], unit='ms', utc=True).tz_convert('America/New_York')
        self.assertTrue(np.array_equal(f.date, t.date))

    def test_future_price_does_not_enter_features(self):
        a = mod.make_windows(self.df, self.splits['train'], 5)
        later = self.df.copy()
        first_session = later['segment_id'].eq(1)
        first_target_idx = later.index[first_session][30]
        later.loc[first_target_idx, 'close'] += 4
        b = mod.make_windows(later, self.splits['train'], 5)
        self.assertTrue(np.array_equal(a['X'][0], b['X'][0]))
        self.assertNotEqual(float(a['y'][0]), float(b['y'][0]))

    def test_nan_warmup_skips_affected_windows(self):
        a = mod.make_windows(self.df, self.splits['train'], 5)
        changed = self.df.copy()
        changed.loc[10, 'range_pct'] = np.nan
        b = mod.make_windows(changed, self.splits['train'], 5)
        self.assertEqual(len(a['y']) - len(b['y']), 1)

    def test_no_date_is_included_from_reserved_test_split(self):
        a = mod.make_windows(self.df, self.splits['train'], 5)
        b = mod.make_windows(self.df, self.splits['validation'], 5)
        test_first = pd.Timestamp(self.splits['reserved_test_dates'][0] + ' 00:00', tz='America/New_York').tz_convert('UTC').value // 1_000_000
        self.assertTrue(np.all(a['target_timestamp_ms'] < test_first))
        self.assertTrue(np.all(b['target_timestamp_ms'] < test_first))

    def test_duplicate_timestamp_rejected(self):
        bad = self.df.copy()
        bad.loc[1, 'timestamp'] = bad.loc[0, 'timestamp']
        with self.assertRaisesRegex(ValueError, 'Unsorted or duplicate'):
            mod.validate_features(bad)

    def test_gap_within_segment_rejected(self):
        bad = self.df.copy()
        bad.loc[5, 'timestamp'] += mod.BAR_MS
        with self.assertRaisesRegex(ValueError, 'Unsorted or duplicate|Gap within'):
            mod.validate_features(bad)

    def test_feature_availability_rejected(self):
        bad = self.df.copy()
        bad.loc[0, 'feature_available_at_ms'] = bad.loc[0, 'timestamp']
        with self.assertRaisesRegex(ValueError, 'available'):
            mod.validate_features(bad)

    def test_wrong_bar_positions_rejected(self):
        bad = self.df.copy()
        bad.loc[2, 'bar_in_segment'] = 200
        with self.assertRaisesRegex(ValueError, 'bar_in_segment'):
            mod.validate_features(bad)

    def test_wrong_feature_order_rejected(self):
        train = mod.make_windows(self.df, self.splits['train'], 5)
        mean, safe, _ = mod.fit_training_scaler(train)
        train['feature_names'] = train['feature_names'][::-1]
        with self.assertRaisesRegex(ValueError, 'order'):
            mod.apply_scaler(train, mean, safe)

    def test_train_only_scaler_means(self):
        train = mod.make_windows(self.df, self.splits['train'], 5)
        val = mod.make_windows(self.df, self.splits['validation'], 5)
        mean, safe, _ = mod.fit_training_scaler(train)
        self.assertTrue(np.allclose(mean, train['X'].astype('float64').mean(axis=(0, 1))))
        scaled = mod.apply_scaler(train, mean, safe)
        self.assertTrue(np.allclose(scaled['X'].mean(axis=(0, 1)), 0, atol=2e-4))
        self.assertGreater(np.abs(mod.apply_scaler(val, mean, safe)['X'].mean()), 0.01)

    def test_scaling_does_not_change_original_target_or_metadata(self):
        train = mod.make_windows(self.df, self.splits['train'], 5)
        snapshot = {k: v.copy() for k, v in train.items()}
        mean, safe, _ = mod.fit_training_scaler(train)
        out = mod.apply_scaler(train, mean, safe)
        for k in snapshot:
            self.assertTrue(np.array_equal(train[k], snapshot[k]))
            if k != 'X':
                self.assertTrue(np.array_equal(out[k], snapshot[k]))

    def test_versioned_manifest_no_test_outputs(self):
        filenames = [str(p) for p in mod.output_manifest()]
        self.assertEqual(len(filenames), 18)
        self.assertTrue(all('nyse_v3' in p or 'windows_nyse_v3_audit' in p for p in filenames))
        self.assertTrue(all('test_' not in Path(p).name for p in filenames))
        self.assertTrue(all('test_normalized' not in p for p in filenames))

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with patch.object(mod, 'RAW_DIR', root / 'raw'), patch.object(mod, 'MODEL_DIR', root / 'model'), patch.object(mod, 'REPORT', root / 'audit.json'):
                mod.check_no_overwrite()
                conflict = mod.output_file('normalized', 'train', 5)
                conflict.parent.mkdir(parents=True)
                conflict.write_text('existing')
                with self.assertRaises(FileExistsError):
                    mod.check_no_overwrite()
                self.assertEqual(conflict.read_text(), 'existing')

    def test_npz_saved_exclusively_and_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            d = mod.make_windows(self.df, self.splits['train'], 5)
            path = Path(td) / 'dataset.npz'
            mod.save_npz(path, d)
            with np.load(path, allow_pickle=False) as r:
                self.assertEqual(r['X'].shape, d['X'].shape)
                self.assertTrue(np.array_equal(r['target_available_at_ms'], d['target_available_at_ms']))
            with self.assertRaises(FileExistsError):
                mod.save_npz(path, d)


if __name__ == '__main__':
    unittest.main()
