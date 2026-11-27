"""AURUM-AI E.5 — calendar-corrected 5/15/30/60m Train/Validation windows.

DRY RUN by default. --write saves *only* new artifacts under nyse_v3 folders.
The held-out Test NPZ files are neither opened nor created. No training occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data' / '03_features' / 'xom_features_nyse_v3.parquet'
RAW_DIR = ROOT / 'data' / '04_windows' / 'nyse_v3'
MODEL_DIR = ROOT / 'data' / '05_model_data' / 'nyse_v3'
REPORT = ROOT / 'reports' / 'data_quality' / 'windows_nyse_v3_audit.json'
FEATURES = (
    'return_1', 'range_pct', 'body_pct',
    'log_return_1', 'volatility_10', 'sma_5_distance',
)
HORIZONS = (5, 15, 30, 60)
BAR_MS = 300_000
WINDOW = 20
SPLITS = ('train', 'validation')


def validate_features(df: pd.DataFrame) -> None:
    required = {
        'timestamp', 'segment_id', 'bar_in_segment', 'session_date',
        'close', 'feature_available_at_ms', *FEATURES,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing V3 feature columns: {sorted(missing)}')
    if df.empty:
        raise ValueError('Empty feature dataset')
    timestamps = df['timestamp'].to_numpy()
    if not np.issubdtype(timestamps.dtype, np.integer):
        raise ValueError('Timestamps must be integer milliseconds')
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError('Unsorted or duplicate timestamps')
    if np.any(timestamps % BAR_MS != 0):
        raise ValueError('Candle timestamps are off the 5-minute grid')
    available = df['feature_available_at_ms'].to_numpy()
    if not np.array_equal(available, timestamps + BAR_MS):
        raise ValueError('OHLC features are not available at candle close')
    closes = df['close'].to_numpy(dtype=np.float64)
    if not np.isfinite(closes).all() or np.any(closes <= 0):
        raise ValueError('Invalid close prices')
    ids = df['segment_id'].to_numpy()
    changes = np.r_[True, ids[1:] != ids[:-1]]
    if pd.Index(ids[changes]).has_duplicates:
        raise ValueError('Segment ID reused nonconsecutively')
    if not np.all(changes | (np.diff(timestamps, prepend=timestamps[0]) == BAR_MS)):
        raise ValueError('Gap within a segment')
    positions = df.groupby('segment_id', sort=False).cumcount().to_numpy()
    if not np.array_equal(positions, df['bar_in_segment'].to_numpy()):
        raise ValueError('Invalid bar_in_segment')
    if df.groupby('segment_id', sort=False)['session_date'].nunique().gt(1).any():
        raise ValueError('Segment crosses session dates')
    ny_dates = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(
        'America/New_York'
    ).dt.strftime('%Y-%m-%d').to_numpy()
    if not np.array_equal(ny_dates, df['session_date'].astype(str).to_numpy()):
        raise ValueError('Wrong New York session dates')
    features = df[list(FEATURES)].to_numpy(dtype=np.float64)
    if np.isinf(features).any():
        raise ValueError('Infinite feature values')


def chronological_splits(df: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    dates = sorted(df['session_date'].astype(str).unique())
    n = len(dates)
    a, b = int(n * 0.70), int(n * 0.85)
    if not (0 < a < b < n):
        raise ValueError('Not enough sessions for chronological Train / Validation / Test')
    return {
        'train': tuple(dates[:a]),
        'validation': tuple(dates[a:b]),
        # Only a DATE BOUNDARY. No test windows/targets are generated or read.
        'reserved_test_dates': tuple(dates[b:]),
    }


def make_windows(df: pd.DataFrame, selected_dates: tuple[str, ...], horizon: int) -> dict[str, np.ndarray]:
    if horizon not in HORIZONS:
        raise ValueError('Unsupported forecast horizon')
    selected = set(selected_dates)
    if not selected:
        raise ValueError('Empty session selection')
    steps = horizon // 5
    keys = ('X', 'y', 'end_timestamp_ms', 'target_timestamp_ms',
            'current_close', 'next_close', 'signal_available_at_ms',
            'target_available_at_ms')
    acc: dict[str, list] = {key: [] for key in keys}
    for _, group in df.groupby('segment_id', sort=False):
        if str(group['session_date'].iloc[0]) not in selected:
            continue
        if len(group) < WINDOW + steps:
            continue
        feats = group[list(FEATURES)].to_numpy(dtype=np.float64)
        closes = group['close'].to_numpy(dtype=np.float64)
        timestamps = group['timestamp'].to_numpy(dtype=np.int64)
        for t in range(WINDOW - 1, len(group) - steps):
            window = feats[t - WINDOW + 1:t + 1]
            if not np.isfinite(window).all():
                continue
            target = t + steps
            if timestamps[target] - timestamps[t] != horizon * 60_000:
                raise ValueError('Target crosses time discontinuity')
            value = np.log(closes[target] / closes[t])
            if not np.isfinite(value):
                raise ValueError('Nonfinite future return')
            acc['X'].append(window.astype(np.float32))
            acc['y'].append(value)
            acc['end_timestamp_ms'].append(timestamps[t])
            acc['target_timestamp_ms'].append(timestamps[target])
            acc['current_close'].append(closes[t])
            acc['next_close'].append(closes[target])
            acc['signal_available_at_ms'].append(timestamps[t] + BAR_MS)
            acc['target_available_at_ms'].append(timestamps[target] + BAR_MS)
    if not acc['X']:
        raise ValueError(f'No windows for horizon={horizon} and selected sessions')
    out = {
        'X': np.stack(acc['X']),
        'y': np.asarray(acc['y'], dtype=np.float32),
        'feature_names': np.asarray(FEATURES),
    }
    for key in ('end_timestamp_ms', 'target_timestamp_ms',
                'signal_available_at_ms', 'target_available_at_ms'):
        out[key] = np.asarray(acc[key], dtype=np.int64)
    for key in ('current_close', 'next_close'):
        out[key] = np.asarray(acc[key], dtype=np.float64)
    if out['X'].shape[1:] != (WINDOW, len(FEATURES)):
        raise AssertionError('Unexpected window dimensions')
    if np.any(out['signal_available_at_ms'] != out['end_timestamp_ms'] + BAR_MS):
        raise AssertionError('Signal becomes available too early')
    if np.any(out['target_available_at_ms'] != out['target_timestamp_ms'] + BAR_MS):
        raise AssertionError('Target becomes available too early')
    if np.any(out['target_timestamp_ms'] - out['end_timestamp_ms'] != horizon * 60_000):
        raise AssertionError('Wrong horizon')
    if not np.isfinite(out['X']).all() or not np.isfinite(out['y']).all():
        raise AssertionError('Nonfinite output')
    return out


def fit_training_scaler(train_5m: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = train_5m['X'].astype(np.float64)
    if x.ndim != 3 or x.shape[1:] != (WINDOW, len(FEATURES)):
        raise ValueError('Invalid train window shape')
    if not np.isfinite(x).all():
        raise ValueError('Invalid training features')
    means = x.mean(axis=(0, 1))
    std = x.std(axis=(0, 1), ddof=0)
    safe = np.where(std < 1e-12, 1., std)
    return means, safe, std


def apply_scaler(data: dict[str, np.ndarray], means: np.ndarray, safe_stds: np.ndarray) -> dict[str, np.ndarray]:
    if not np.array_equal(data['feature_names'], np.asarray(FEATURES)):
        raise ValueError('Feature order mismatch')
    copied = dict(data)
    copied['X'] = ((data['X'].astype(np.float64) - means) / safe_stds).astype(np.float32)
    if not np.isfinite(copied['X']).all():
        raise ValueError('Invalid scaled values')
    return copied


def output_file(kind: str, split: str, horizon: int) -> Path:
    if kind not in ('raw', 'normalized') or split not in SPLITS or horizon not in HORIZONS:
        raise ValueError('Unexpected output key')
    filename = f'{split}_windows.npz' if horizon == 5 else f'{split}_{horizon}m_windows.npz'
    if kind == 'normalized':
        filename = filename.replace('_windows.npz', '_normalized.npz')
    return (RAW_DIR if kind == 'raw' else MODEL_DIR) / filename


def output_manifest() -> list[Path]:
    return [output_file(kind, split, horizon) for kind in ('raw', 'normalized')
            for split in SPLITS for horizon in HORIZONS] + [MODEL_DIR / 'normalization_stats.json', REPORT]


def check_no_overwrite() -> None:
    existing = [str(path) for path in output_manifest() if path.exists()]
    if existing:
        raise FileExistsError('Existing V3 output(s) preserved: ' + ', '.join(existing))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_npz(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        np.savez_compressed(handle, **data)


def main() -> None:
    parser = argparse.ArgumentParser(description='Read-only by default: build 5/15/30/60m calendar-aware V3 Train+Validation datasets')
    parser.add_argument('--write', action='store_true', help='Create new V3 files only (never overwrite)')
    args = parser.parse_args()
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    if args.write:
        check_no_overwrite()
    frame = pd.read_parquet(SOURCE)
    validate_features(frame)
    splits = chronological_splits(frame)
    if len(splits['train']) != 1450 or len(splits['validation']) != 311:
        raise ValueError('Unexpected session split boundaries compared with V2; stop for review')
    audit = {
        'dataset': 'AURUM-AI XOM V3 NYSE',
        'source_path': str(SOURCE),
        'source_sha256': sha256_file(SOURCE),
        'feature_names': list(FEATURES),
        'window_size_bars': WINDOW,
        'feature_availability': 'candle open + 5 minutes',
        'target_definition': 'ln(close_at_target_bar / close_at_last_input_bar)',
        'split_policy': 'chronological by observed NYSE session; 70% Train, 15% Validation; held-out remainder reserved, never windowed',
        'splits': {split: {'observed_sessions': len(splits[split]),
                           'first': splits[split][0], 'last': splits[split][-1]}
                   for split in SPLITS},
        'reserved_test_session_count': len(splits['reserved_test_dates']),
        'test_npz_opened': False, 'test_windows_built': False,
        'horizons': {}, 'scaler': {}, 'mode': 'WRITE NEW VERSION' if args.write else 'DRY RUN',
    }
    print('=== AURUM-AI E.5 | NYSE V3 MULTI-HORIZON WINDOWS ===')
    print('Mode:', audit['mode'])
    print('Input rows:', len(frame))
    print('Train sessions:', len(splits['train']), splits['train'][0], '->', splits['train'][-1])
    print('Validation sessions:', len(splits['validation']), splits['validation'][0], '->', splits['validation'][-1])
    print('Reserved held-out sessions (NO windows):', len(splits['reserved_test_dates']))

    # Fit on the 5m TRAIN windows, preserving the historical V2 scaling convention.
    train_5m = make_windows(frame, splits['train'], 5)
    means, std_safe, std_original = fit_training_scaler(train_5m)
    audit['scaler'] = {
        'method': 'z-score', 'fitted_on': 'train 5m windows only',
        'mean': means.tolist(), 'std': std_safe.tolist(),
        'original_std': std_original.tolist(),
    }
    for horizon in HORIZONS:
        audit['horizons'][str(horizon)] = {}
        for split in SPLITS:
            data = train_5m if (horizon == 5 and split == 'train') else make_windows(
                frame, splits[split], horizon)
            scaled = apply_scaler(data, means, std_safe)
            audit['horizons'][str(horizon)][split] = {
                'windows': int(len(data['y'])), 'shape': list(data['X'].shape),
                'min_signal_timestamp_ms': int(data['end_timestamp_ms'].min()),
                'max_signal_timestamp_ms': int(data['end_timestamp_ms'].max()),
                'normalized_output': str(output_file('normalized', split, horizon)),
            }
            print(f'{horizon:2d}m {split:10s}: windows={len(data["y"]):6d}, X={data["X"].shape}')
            if args.write:
                save_npz(output_file('raw', split, horizon), data)
                save_npz(output_file('normalized', split, horizon), scaled)
            del data, scaled
    del train_5m
    print('Scaler fitted ONLY on Train 5m; reused without fitting on Validation or other horizons.')
    print('No Test NPZ read or written. No model training or backtesting executed.')
    if not args.write:
        print('No output written. Review results before --write.')
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with (MODEL_DIR / 'normalization_stats.json').open('x', encoding='utf-8') as handle:
        json.dump(audit['scaler'] | {'feature_names': list(FEATURES), 'window_size': WINDOW,
                                    'source': 'train 5m V3, overlapping windows'}, handle, indent=2)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open('x', encoding='utf-8') as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    print('Created V3 Train and Validation raw and normalized NPZ in:', MODEL_DIR)
    print('Created audit:', REPORT)
    print('All V2 files, RAW data, old models and reports left untouched.')


if __name__ == '__main__':
    main()
