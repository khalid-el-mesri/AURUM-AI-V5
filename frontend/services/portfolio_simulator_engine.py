"""AURUM-AI V5 / Portfolio Simulator — read-only historical evidence.

This module never invokes Strategy, Risk, Execution, or Portfolio agents.
Artifacts from 30m and 60m runs are read separately; no mixing of runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path('reports/backtesting/combined_holding_30m_60m')
RUNS = {
    30: ('30m_decisions.csv', '30m_transactions.csv'),
    60: ('60m_decisions.csv', '60m_transactions.csv'),
}
DECISION_COLUMNS = {'timestamp_ms', 'strategy_action', 'risk_status', 'execution_status',
                    'cash_after_usd', 'equity_after_usd', 'event', 'system_state'}
TRADE_COLUMNS = {'timestamp_ms', 'action', 'price', 'quantity', 'fee_usd',
                 'cash_change_usd', 'realized_pnl_usd', 'cash_after_usd', 'holdings_after'}


def _read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'Fichier historique absent : {path}')
    df = pd.read_csv(path, low_memory=False)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f'{path.name} : colonnes absentes : {missing}')
    if df.empty:
        raise ValueError(f'{path.name} : fichier vide')
    ts = pd.to_numeric(df['timestamp_ms'], errors='coerce')
    if ts.isna().any() or not np.isfinite(ts.to_numpy(dtype=float)).all() or (ts % 1 != 0).any():
        raise ValueError(f'{path.name} : horodatage non entier / absent')
    df['timestamp_ms'] = ts.astype('int64')
    return df


def load_run(root: Path, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if horizon not in RUNS:
        raise ValueError('Horizon : seulement 30 ou 60 minutes')
    decision_file, trade_file = RUNS[horizon]
    base = Path(root) / BASE
    d = _read_csv(base / decision_file, DECISION_COLUMNS)
    t = _read_csv(base / trade_file, TRADE_COLUMNS)
    for field in ['price', 'quantity', 'fee_usd', 'cash_change_usd',
                  'realized_pnl_usd', 'cash_after_usd', 'holdings_after']:
        nums = pd.to_numeric(t[field], errors='coerce')
        if nums.isna().any() or not np.isfinite(nums.to_numpy(dtype=float)).all():
            raise ValueError(f'{trade_file} : données numériques invalides dans {field}')
        t[field] = nums.astype('float64')
    if (t['price'] <= 0).any() or (t['quantity'] <= 0).any() or (t['fee_usd'] < 0).any():
        raise ValueError(f'{trade_file} : prix, quantité ou frais invalides')
    if (~t['action'].astype(str).str.upper().isin(['BUY', 'SELL'])).any():
        raise ValueError(f'{trade_file} : action inconnue')
    return d, t


def join_events(decisions: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Exact (timestamp, action, occurrence) matching for EXECUTED rows only."""
    d = decisions.copy().reset_index(drop=True)
    d['_source_row'] = np.arange(len(d))
    d['_action'] = d['strategy_action'].astype('string').fillna('').str.upper().str.strip()
    is_executed = d['execution_status'].astype('string').fillna('').str.upper().str.strip().eq('EXECUTED')
    executed = d.loc[is_executed, ['_source_row', 'timestamp_ms', '_action']].copy()
    executed['_occurrence'] = executed.groupby(['timestamp_ms', '_action']).cumcount()

    t = trades.copy().reset_index(drop=True)
    t['_trade_row'] = np.arange(len(t))
    t['_action'] = t['action'].astype('string').fillna('').str.upper().str.strip()
    t['_occurrence'] = t.groupby(['timestamp_ms', '_action']).cumcount()
    keys = ['timestamp_ms', '_action', '_occurrence']
    matched = executed.merge(t, on=keys, how='left', validate='one_to_one', suffixes=('', '_trade'))
    rename = {'price': 'trade_price', 'quantity': 'trade_quantity',
              'fee_usd': 'trade_fee_usd', 'cash_change_usd': 'trade_cash_change_usd',
              'cash_after_usd': 'trade_cash_after_usd', 'holdings_after': 'trade_holdings_after',
              'realized_pnl_usd': 'trade_realized_pnl_usd', 'action': 'trade_action'}
    matched = matched.rename(columns=rename)
    keep = ['_source_row', '_trade_row'] + list(rename.values())
    joined = d.merge(matched[keep], on='_source_row', how='left', validate='one_to_one')
    joined['matched_trade'] = joined['_trade_row'].notna()
    joined['datetime_utc'] = pd.to_datetime(joined['timestamp_ms'], unit='ms', utc=True)
    return joined.drop(columns=['_source_row', '_action'])


def initial_cash_reference(trades: pd.DataFrame) -> float | None:
    """Infer a *reference* initial capital only when first trade opens from flat."""
    if trades.empty:
        return None
    t = trades.iloc[0]
    if str(t['action']).strip().upper() != 'BUY':
        return None
    qty, holdings = float(t['quantity']), float(t['holdings_after'])
    if not np.isclose(qty, holdings, rtol=0, atol=1e-6):
        return None
    cash_before = float(t['cash_after_usd']) - float(t['cash_change_usd'])
    return cash_before if np.isfinite(cash_before) and cash_before >= 0 else None


def recorded_equity(decisions: pd.DataFrame, gap_days: int = 7) -> pd.DataFrame:
    df = decisions[['timestamp_ms', 'equity_after_usd', 'cash_after_usd']].copy()
    for col in ['equity_after_usd', 'cash_after_usd']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['equity_after_usd']).copy()
    df['datetime_utc'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)
    df['segment'] = (df['timestamp_ms'].diff().fillna(0) > gap_days * 86400000).cumsum()
    return df


def trade_tape(trades: pd.DataFrame) -> pd.DataFrame:
    t = trades.copy()
    t['datetime_utc'] = pd.to_datetime(t['timestamp_ms'], unit='ms', utc=True)
    # This is a position valuation at the recorded transaction price, not a market curve.
    t['trade_time_position_value_usd'] = t['holdings_after'] * t['price']
    t['trade_time_portfolio_value_usd'] = t['cash_after_usd'] + t['trade_time_position_value_usd']
    return t


def audit_run(decisions: pd.DataFrame, trades: pd.DataFrame) -> dict:
    j = join_events(decisions, trades)
    matched = j[j['matched_trade']]
    trade_rows_matched = set(matched['_trade_row'].astype(int).tolist())
    is_executed = decisions['execution_status'].astype('string').fillna('').str.upper().str.strip().eq('EXECUTED')
    difference = (pd.to_numeric(matched['cash_after_usd'], errors='coerce')
                  - pd.to_numeric(matched['trade_cash_after_usd'], errors='coerce')).abs()
    cash_bad = difference.isna() | (difference > 0.011)
    if 'price' in matched:
        dp = pd.to_numeric(matched['price'], errors='coerce')
        tp = pd.to_numeric(matched['trade_price'], errors='coerce')
        price_bad = (dp.notna() & tp.notna() & (dp - tp).abs().gt(0.011))
    else:
        price_bad = pd.Series(False, index=matched.index)
    t = trades.reset_index(drop=True)
    previous_cash = float(t.iloc[0]['cash_after_usd'] - t.iloc[0]['cash_change_usd'])
    expected_cash = previous_cash + t['cash_change_usd'].cumsum()
    flow_error = (expected_cash - t['cash_after_usd']).abs() > 0.011
    action = t['action'].astype(str).str.upper().str.strip()
    theoretical_flow = np.where(action.eq('BUY'), -(t['price']*t['quantity'] + t['fee_usd']),
                                t['price']*t['quantity'] - t['fee_usd'])
    fee_flow_error = (t['cash_change_usd'] - theoretical_flow).abs() > 0.02
    return {
        'decisions': int(len(decisions)),
        'transactions': int(len(t)),
        'executed_decisions': int(is_executed.sum()),
        'matched_transactions': int(len(trade_rows_matched)),
        'unmatched_executed_decisions': int(is_executed.sum() - len(matched)),
        'unmatched_transactions': int(len(t) - len(trade_rows_matched)),
        'cash_mismatch_gt_1_cent': int(cash_bad.sum()),
        'price_mismatch_gt_1_cent': int(price_bad.sum()),
        'cash_flow_mismatch_gt_1_cent': int(flow_error.sum()),
        'trade_cash_flow_mismatch_gt_2_cents': int(fee_flow_error.sum()),
        'negative_holdings': int((t['holdings_after'] < -1e-6).sum()),
        'decision_reverse_time': int((decisions['timestamp_ms'].diff() < 0).sum()),
        'trade_reverse_time': int((t['timestamp_ms'].diff() < 0).sum()),
        'decision_duplicates_timestamps': int(decisions['timestamp_ms'].duplicated().sum()),
    }


def run_metrics(decisions: pd.DataFrame, trades: pd.DataFrame) -> dict:
    curve = recorded_equity(decisions)
    t = trades.reset_index(drop=True)
    initial = initial_cash_reference(t)
    last_equity = float(curve.iloc[-1]['equity_after_usd']) if not curve.empty else None
    return_pct = (100 * (last_equity / initial - 1)) if (last_equity is not None and initial is not None and initial > 0) else None
    # The curve samples only recorded decisions; this is NOT full intraday drawdown.
    drawdown = 0.0
    if not curve.empty:
        peak = curve['equity_after_usd'].cummax()
        valid = peak.gt(0)
        dd = (curve.loc[valid, 'equity_after_usd'] / peak.loc[valid] - 1) * 100
        drawdown = float(-dd.min()) if len(dd) else 0.0
    return {
        'initial_cash_reference_usd': initial,
        'first_recorded_equity_usd': float(curve.iloc[0]['equity_after_usd']) if not curve.empty else None,
        'last_recorded_equity_usd': last_equity,
        'return_vs_reference_pct': return_pct,
        'max_observed_decision_drawdown_pct': drawdown,
        'fees_recorded_usd': float(t['fee_usd'].sum()),
        'realized_pnl_recorded_usd': float(t['realized_pnl_usd'].sum()),
        'last_cash_usd': float(t.iloc[-1]['cash_after_usd']),
        'last_holdings': float(t.iloc[-1]['holdings_after']),
        'buy_count': int(t['action'].astype(str).str.upper().eq('BUY').sum()),
        'sell_count': int(t['action'].astype(str).str.upper().eq('SELL').sum()),
        'market_start_utc': pd.to_datetime(int(decisions.iloc[0]['timestamp_ms']), unit='ms', utc=True).isoformat(),
        'market_end_utc': pd.to_datetime(int(decisions.iloc[-1]['timestamp_ms']), unit='ms', utc=True).isoformat(),
        'transactions_count': len(t),
        'decisions_count': len(decisions),
    }


def report_json(horizon: int, metrics: dict, audit: dict) -> bytes:
    return json.dumps({'simulation_only': True,
                       'source': str(BASE / RUNS[horizon][0]),
                       'transaction_source': str(BASE / RUNS[horizon][1]),
                       'horizon_minutes': horizon,
                       'metrics': metrics, 'audit': audit}, indent=2,
                      ensure_ascii=False, default=str).encode('utf-8')
