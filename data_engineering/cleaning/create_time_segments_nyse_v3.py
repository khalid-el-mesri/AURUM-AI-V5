"""AURUM-AI E.5: segment NYSE-filtered 5-minute XOM candles.

Dry-run is the default. Explicit --write creates new versioned outputs, refusing
replacements. No mutation to RAW, prior Parquet files, models or reports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data' / '02_clean' / 'xom_regular_session_nyse_v3.csv'
DEST = ROOT / 'data' / '02_clean' / 'xom_segmented_nyse_v3.parquet'
AUDIT = ROOT / 'reports' / 'data_quality' / 'segmentation_nyse_v3_audit.json'
BAR_MS = 300_000
REQUIRED_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close']


def make_segments(frame: pd.DataFrame, schedule: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Pure transformation: validate and segment a previously NYSE-filtered CSV.

    All original rows and columns retained, with new UTC/session/segment metadata.
    A gap or New York session-date boundary starts a new segment.
    """
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if frame.empty:
        raise ValueError('Empty source dataset')
    out = frame.copy()
    ts = out['timestamp']
    if ts.isna().any() or not pd.api.types.is_integer_dtype(ts.dtype):
        raise ValueError('timestamp must contain integer Unix milliseconds')
    if not ts.is_monotonic_increasing or ts.duplicated().any():
        raise ValueError('Unsorted or duplicate timestamps')
    if (ts % BAR_MS != 0).any():
        raise ValueError('Timestamps must align to five-minute UTC grid')
    prices = out[['open', 'high', 'low', 'close']]
    if prices.isna().any().any() or not np.isfinite(prices.to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite OHLC detected')
    if (prices <= 0).any().any():
        raise ValueError('Nonpositive OHLC detected')
    if (out['high'] < out[['open', 'close', 'low']].max(axis=1)).any():
        raise ValueError('OHLC high constraint violation')
    if (out['low'] > out[['open', 'close', 'high']].min(axis=1)).any():
        raise ValueError('OHLC low constraint violation')

    if schedule.empty or not {'market_open', 'market_close'}.issubset(schedule.columns):
        raise ValueError('Missing official NYSE calendar data')
    dates = pd.to_datetime(ts, unit='ms', utc=True)
    local = dates.dt.tz_convert('America/New_York')
    session = local.dt.strftime('%Y-%m-%d')
    cal = schedule[['market_open', 'market_close']].copy()
    cal['session_date'] = pd.to_datetime(cal.index).strftime('%Y-%m-%d')
    if cal['session_date'].duplicated().any():
        raise ValueError('Duplicate NYSE calendar dates')
    mapped = cal.set_index('session_date')
    opens = pd.to_datetime(session.map(mapped['market_open']), utc=True)
    closes = pd.to_datetime(session.map(mapped['market_close']), utc=True)
    aligned = ((ts - (opens.astype('int64') // 10**6)) % BAR_MS == 0)
    inside = (opens.notna() & closes.notna() & (dates >= opens)
              & ((dates + pd.Timedelta(minutes=5)) <= closes) & aligned)
    if not inside.all():
        first = int(np.flatnonzero(~inside.to_numpy())[0])
        raise ValueError(f'Source contains NYSE-inadmissible bar at row {first}: timestamp={int(ts.iloc[first])}')

    out['datetime_utc'] = dates
    out['session_date'] = session
    start_new = session.ne(session.shift()) | ts.diff().ne(BAR_MS)
    out['segment_id'] = start_new.cumsum().astype('int64')
    out['bar_in_segment'] = out.groupby('segment_id', sort=False).cumcount().astype('int64')
    diffs = out.groupby('segment_id', sort=False)['timestamp'].diff().dropna()
    if not diffs.eq(BAR_MS).all():
        raise ValueError('Internal segment continuity violation')
    summary = {
        'input_rows': int(len(out)),
        'segmented_rows': int(len(out)),
        'session_dates': int(out['session_date'].nunique()),
        'segments': int(out['segment_id'].nunique()),
        'intraday_interruptions': int((~session.ne(session.shift()) & ts.diff().ne(BAR_MS)).sum()),
        'first_timestamp_ms': int(ts.iloc[0]),
        'last_timestamp_ms': int(ts.iloc[-1]),
        'five_minute_bar_ms': BAR_MS,
        'calendar_rule': 'whole [open,open+5min) candle within official NYSE session',
        'timestamp_semantics': 'timestamp = candle OPEN in UTC milliseconds; OHLC available after OPEN+5min',
    }
    return out, summary


def main() -> None:
    parser = argparse.ArgumentParser(description='AURUM-AI NYSE V3 5-minute segmentation; dry-run by default')
    parser.add_argument('--write', action='store_true', help='create versioned Parquet + report, refusing overwrite')
    args = parser.parse_args()
    if not SOURCE.exists():
        raise FileNotFoundError(f'Expected NYSE V3 CSV missing: {SOURCE}')
    if args.write and (DEST.exists() or AUDIT.exists()):
        raise FileExistsError(f'Versioned output already exists. DEST={DEST.exists()}, AUDIT={AUDIT.exists()}')
    import pandas_market_calendars as mcal
    source = pd.read_csv(SOURCE)
    times = pd.to_datetime(source['timestamp'], unit='ms', utc=True).dt.tz_convert('America/New_York')
    schedule = mcal.get_calendar('NYSE').schedule(start_date=times.dt.date.iloc[0], end_date=times.dt.date.iloc[-1])
    segmented, counts = make_segments(source, schedule)
    print('=== AURUM-AI E.5 | NYSE V3 TIME SEGMENTATION ===')
    print('Mode:', 'WRITE NEW VERSION' if args.write else 'DRY RUN / READ ONLY')
    for key, val in counts.items():
        print(f'{key}: {val}')
    print('Old CSVs, Parquets, models and reports are not modified.')
    if not args.write:
        print('No outputs written. Review results before using --write.')
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids accidental replacement of earlier work.
    with DEST.open('xb') as handle:
        segmented.to_parquet(handle, index=False)
    report = {'source_csv': str(SOURCE), 'output_parquet': str(DEST), **counts,
              'features_regenerated': False, 'model_training_performed': False,
              'test_split_npz_opened': False, 'source_csv_modified': False}
    with AUDIT.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print('Created:', DEST)
    print('Created:', AUDIT)
    print('No features, windows or models regenerated.')


if __name__ == '__main__':
    main()
