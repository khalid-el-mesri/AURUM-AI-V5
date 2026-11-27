"""AURUM-AI E.5: causal V3 features from calendar-corrected 5-minute XOM bars.

Default is DRY RUN. --write exclusively creates new, versioned output and report.
Matches V2 formulas and names; no model/scaler/window rebuilding here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data' / '02_clean' / 'xom_segmented_nyse_v3.parquet'
DEST = ROOT / 'data' / '03_features' / 'xom_features_nyse_v3.parquet'
AUDIT = ROOT / 'reports' / 'data_quality' / 'features_nyse_v3_audit.json'
BAR_MS = 300_000

FEATURE_COLUMNS = [
    'return_1', 'range_pct', 'body_pct', 'sma_5',
    'log_return_1', 'volatility_10', 'sma_5_distance', 'ema_10',
]
MODEL_FEATURE_COLUMNS = [
    'return_1', 'range_pct', 'body_pct',
    'log_return_1', 'volatility_10', 'sma_5_distance',
]
REQUIRED = [
    'timestamp', 'open', 'high', 'low', 'close', 'session_date',
    'segment_id', 'bar_in_segment',
]


def _validate_input(frame: pd.DataFrame) -> None:
    missing = set(REQUIRED).difference(frame.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if frame.empty:
        raise ValueError('Empty input data')
    if any(name in frame.columns for name in FEATURE_COLUMNS + ['feature_available_at_ms']):
        raise ValueError('Input already contains output features/availability metadata')
    if frame['timestamp'].isna().any() or not pd.api.types.is_integer_dtype(frame['timestamp']):
        raise ValueError('Timestamps must be integer milliseconds')
    ts = frame['timestamp']
    if not ts.is_monotonic_increasing or ts.duplicated().any():
        raise ValueError('Timestamps must be strictly increasing')
    if (ts % BAR_MS != 0).any():
        raise ValueError('Timestamps not on 5-minute grid')
    prices = frame[['open', 'high', 'low', 'close']]
    a = prices.to_numpy(dtype=np.float64)
    if not np.isfinite(a).all() or (a <= 0).any():
        raise ValueError('Invalid OHLC prices')
    if (frame['high'] < frame[['open', 'low', 'close']].max(axis=1)).any():
        raise ValueError('OHLC high violates price bounds')
    if (frame['low'] > frame[['open', 'high', 'close']].min(axis=1)).any():
        raise ValueError('OHLC low violates price bounds')
    if frame['segment_id'].isna().any() or frame['session_date'].isna().any():
        raise ValueError('Missing segment or session metadata')
    if frame['bar_in_segment'].isna().any():
        raise ValueError('Missing bar positions')
    seg = frame['segment_id']
    segment_change = seg.ne(seg.shift())
    # An old segment cannot resume after a different segment.
    if seg[segment_change].duplicated().any():
        raise ValueError('Segment IDs are not contiguous blocks')
    diffs = frame.groupby('segment_id', sort=False)['timestamp'].diff().dropna()
    if not diffs.eq(BAR_MS).all():
        raise ValueError('Nonconsecutive 5-minute candles inside a segment')
    if frame.groupby('segment_id', sort=False)['session_date'].nunique().gt(1).any():
        raise ValueError('A segment crosses trading sessions')
    positions = frame.groupby('segment_id', sort=False).cumcount().to_numpy()
    if not np.array_equal(frame['bar_in_segment'].to_numpy(), positions):
        raise ValueError('Incorrect bar_in_segment values')
    local_date = pd.to_datetime(ts, unit='ms', utc=True).dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d')
    if not np.array_equal(local_date.to_numpy(), frame['session_date'].astype(str).to_numpy()):
        raise ValueError('session_date differs from candle local New York date')


def build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Pure, segment-local, backward-looking features identical to legacy V2 formulas."""
    _validate_input(frame)
    df = frame.copy(deep=True)
    group = df.groupby('segment_id', sort=False)
    df['return_1'] = group['close'].pct_change(fill_method=None)
    df['range_pct'] = (df['high'] - df['low']) / df['open']
    df['body_pct'] = (df['close'] - df['open']) / df['open']
    df['sma_5'] = group['close'].transform(lambda s: s.rolling(window=5, min_periods=5).mean())
    df['log_return_1'] = group['close'].transform(lambda s: np.log(s).diff())
    df['volatility_10'] = df.groupby('segment_id', sort=False)['log_return_1'].transform(
        lambda s: s.rolling(window=10, min_periods=10).std(ddof=0)
    )
    df['sma_5_distance'] = df['close'] / df['sma_5'] - 1
    df['ema_10'] = group['close'].transform(lambda s: s.ewm(span=10, adjust=False).mean())

    # Feature at t uses the full OHLC candle [t, t+5m); it is NOT known at candle open.
    df['feature_available_at_ms'] = df['timestamp'].astype('int64') + BAR_MS
    if len(df) != len(frame):
        raise AssertionError('Unexpected row count change')
    if not np.isfinite(df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)[~np.isnan(df[FEATURE_COLUMNS].to_numpy(dtype=np.float64))]).all():
        raise ValueError('Nonfinite features')
    return df


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def summarize(frame: pd.DataFrame, featured: pd.DataFrame) -> dict:
    return {
        'input_rows': int(len(frame)),
        'output_rows': int(len(featured)),
        'segments': int(featured['segment_id'].nunique()),
        'session_dates': int(featured['session_date'].nunique()),
        'first_timestamp_ms': int(featured['timestamp'].iloc[0]),
        'last_timestamp_ms': int(featured['timestamp'].iloc[-1]),
        'feature_columns': FEATURE_COLUMNS,
        'model_feature_columns': MODEL_FEATURE_COLUMNS,
        'missing_by_feature': {c: int(featured[c].isna().sum()) for c in FEATURE_COLUMNS},
        'timestamp_semantics': 'candle OPEN in UTC milliseconds',
        'feature_available_at_ms_rule': 'timestamp + 300000 (candle END)',
        'causal_rule': 'all rolling/EMA/return calculations use current and preceding bars within each segment only',
        'scaler_retrained': False,
        'model_retrained': False,
        'windows_rebuilt': False,
        'test_npz_opened': False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Build separate V3 NYSE-aware historical features')
    parser.add_argument('--write', action='store_true', help='Create versioned Parquet and audit (no overwrite)')
    args = parser.parse_args()
    if not SOURCE.is_file():
        raise FileNotFoundError(f'Corrected segmented input missing: {SOURCE}')
    if args.write and (DEST.exists() or AUDIT.exists()):
        raise FileExistsError(f'Versioned output exists already. Parquet={DEST.exists()}, audit={AUDIT.exists()}')

    frame = pd.read_parquet(SOURCE)
    featured = build_feature_frame(frame)
    stats = summarize(frame, featured)
    print('=== AURUM-AI E.5 | FEATURE ENGINEERING NYSE V3 ===')
    print('Mode:', 'WRITE NEW VERSION' if args.write else 'DRY RUN / READ ONLY')
    print('Source:', SOURCE)
    for k in ('input_rows', 'output_rows', 'segments', 'session_dates'):
        print(f'{k}: {stats[k]}')
    print('Feature columns:', ', '.join(FEATURE_COLUMNS))
    print('Missing by feature:', stats['missing_by_feature'])
    print('Candle timestamps are OPEN; feature availability = OPEN+5min.')
    print('Existing v2 datasets, models, backtests untouched.')
    if not args.write:
        print('No outputs written. Review before --write.')
        return

    DEST.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    stats['source_path'] = str(SOURCE)
    stats['source_sha256'] = _sha256(SOURCE)
    stats['output_path'] = str(DEST)
    with DEST.open('xb') as handle:
        featured.to_parquet(handle, index=False)
    stats['output_sha256'] = _sha256(DEST)
    with AUDIT.open('x', encoding='utf-8') as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)
    print('Created:', DEST)
    print('Created:', AUDIT)
    print('No normalization, windows, training or test evaluation performed.')


if __name__ == '__main__':
    main()
