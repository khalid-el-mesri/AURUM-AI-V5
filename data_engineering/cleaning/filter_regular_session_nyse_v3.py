"""AURUM-AI E.5: NYSE calendar-aware RTH filter, dry-run by default.

Never modifies RAW or previous derived datasets. --write creates a new,
versioned CSV only when it does not already exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/01_raw/xomususd-m5-bid-2017-01-26-2026-10-01.csv'
DEST = ROOT / 'data/02_clean/xom_regular_session_nyse_v3.csv'
AUDIT = ROOT / 'reports/data_quality/nyse_filter_v3_audit.json'
BAR_MS = 300_000


def apply_nyse_filter(frame: pd.DataFrame, calendar: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (valid bar rows, excluded old-RTH candidate rows), without mutating input.

    This validation applies NYSE open/close to the full 5-minute OHLC bar.
    A bar beginning exactly at session close is excluded.
    """
    if 'timestamp' not in frame.columns:
        raise ValueError('Missing timestamp column')
    if frame.empty:
        raise ValueError('Empty market dataset')
    ts = frame['timestamp']
    if ts.isna().any() or not pd.api.types.is_integer_dtype(ts.dtype):
        raise ValueError('timestamp must be integer Unix milliseconds')
    if not ts.is_monotonic_increasing or ts.duplicated().any():
        raise ValueError('Nonchronological or duplicate timestamps')
    if calendar.empty or not {'market_open', 'market_close'}.issubset(calendar.columns):
        raise ValueError('Invalid NYSE schedule')

    times = pd.to_datetime(ts, unit='ms', utc=True)
    local = times.dt.tz_convert('America/New_York')
    day = local.dt.strftime('%Y-%m-%d')
    old_rth = ((local.dt.dayofweek < 5)
               & (local.dt.hour * 60 + local.dt.minute >= 570)
               & (local.dt.hour * 60 + local.dt.minute < 960))

    sessions = calendar[['market_open', 'market_close']].copy()
    sessions['session_date'] = pd.to_datetime(sessions.index).strftime('%Y-%m-%d')
    if sessions.session_date.duplicated().any():
        raise ValueError('Duplicate calendar sessions')
    indexed = sessions.set_index('session_date')
    opens = pd.to_datetime(day.map(indexed.market_open), utc=True)
    closes = pd.to_datetime(day.map(indexed.market_close), utc=True)
    has_session = opens.notna() & closes.notna()
    valid = (has_session & (times >= opens)
             & ((times + pd.Timedelta(milliseconds=BAR_MS)) <= closes))
    # Five-minute start alignment relative to official market open.
    aligned = ((ts - (opens.astype('int64', copy=False) // 10**6)) % BAR_MS == 0)
    valid = valid & aligned
    if (valid & ~old_rth).any():
        raise ValueError('Unexpected NYSE schedule outside previous regular-session definition')
    selected = frame.loc[valid].copy()
    rejected = frame.loc[old_rth & ~valid].copy()
    rejected.insert(0, 'session_date_ny', day.loc[old_rth & ~valid].to_numpy())
    return selected, rejected


def main() -> None:
    parser = argparse.ArgumentParser(description='NYSE RTH audit, dry-run unless --write')
    parser.add_argument('--write', action='store_true', help='create new v3 CSV and audit; never overwrite')
    args = parser.parse_args()
    if not RAW.exists():
        raise FileNotFoundError(RAW)
    if args.write and (DEST.exists() or AUDIT.exists()):
        raise FileExistsError('Versioned destination already exists; refusing to overwrite')

    import pandas_market_calendars as mcal  # required for real historical schedule
    source = pd.read_csv(RAW)
    dates = pd.to_datetime(source.timestamp, unit='ms', utc=True).dt.tz_convert('America/New_York')
    cal = mcal.get_calendar('NYSE').schedule(
        start_date=dates.dt.date.iloc[0], end_date=dates.dt.date.iloc[-1]
    )
    selected, rejected = apply_nyse_filter(source, cal)
    print('=== AURUM-AI E.5 | NYSE CALENDAR V3 ===')
    print('Mode:', 'WRITE NEW VERSION' if args.write else 'DRY RUN / READ ONLY')
    print('RAW rows:', len(source))
    print('Calendar-admissible rows:', len(selected))
    print('Old regular-session candidates excluded by NYSE calendar:', len(rejected))
    print('Removed by NY session date:')
    print(rejected.session_date_ny.value_counts().sort_index().to_string() if len(rejected) else '(none)')
    print('Original RAW / V2 datasets / models / reports unchanged')
    if not args.write:
        print('No output files written. Check results before --write.')
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive file creation; no silent overwrite.
    with DEST.open('x', encoding='utf-8', newline='') as handle:
        selected.to_csv(handle, index=False)
    report = {
        'source': str(RAW), 'output': str(DEST), 'policy': 'NYSE official schedule; entire bar within session',
        'raw_rows': len(source), 'selected_rows': len(selected),
        'excluded_old_rth_rows': len(rejected),
        'excluded_by_date': {str(k): int(v) for k, v in rejected.session_date_ny.value_counts().sort_index().items()},
        'test_split_npz_opened': False, 'raw_includes_all_historical_periods': True, 'raw_modified': False,
        'downstream_pipeline_regenerated': False,
    }
    with AUDIT.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print('Created:', DEST)
    print('Created:', AUDIT)
    print('No features, windows, scaler or models regenerated yet.')


if __name__ == '__main__':
    main()
