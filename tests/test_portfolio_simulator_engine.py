"""Synthetic fixtures only: no real AURUM market data shipped in this ZIP."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frontend.services.portfolio_simulator_engine import (
    BASE, RUNS, _read_csv, audit_run, initial_cash_reference, join_events,
    load_run, recorded_equity, report_json, run_metrics, trade_tape,
)


def synthetic():
    d = pd.DataFrame([
        {'timestamp_ms': 1000000000000, 'strategy_action': 'BUY', 'risk_status': 'APPROVED',
         'execution_status': 'EXECUTED', 'cash_after_usd': 949.9, 'equity_after_usd': 999.9,
         'event': 'FORECAST_DECISION', 'system_state': 'ACTIVE', 'price': 10.0},
        {'timestamp_ms': 1000000300000, 'strategy_action': 'HOLD', 'risk_status': 'NO_ACTION',
         'execution_status': 'NO_ACTION', 'cash_after_usd': 949.9, 'equity_after_usd': 1004.9,
         'event': 'FORECAST_DECISION', 'system_state': 'ACTIVE', 'price': None},
        {'timestamp_ms': 1000000600000, 'strategy_action': 'SELL', 'risk_status': 'APPROVED',
         'execution_status': 'EXECUTED', 'cash_after_usd': 1004.8, 'equity_after_usd': 1004.8,
         'event': 'SCHEDULED_EXIT', 'system_state': 'ACTIVE', 'price': 11.0},
    ])
    t = pd.DataFrame([
        {'timestamp_ms': 1000000000000, 'action': 'BUY', 'price': 10.0,
         'quantity': 5., 'fee_usd': .1, 'cash_change_usd': -50.1, 'realized_pnl_usd': 0.,
         'cash_after_usd': 949.9, 'holdings_after': 5.},
        {'timestamp_ms': 1000000600000, 'action': 'SELL', 'price': 11.0,
         'quantity': 5., 'fee_usd': .1, 'cash_change_usd': 54.9, 'realized_pnl_usd': 4.8,
         'cash_after_usd': 1004.8, 'holdings_after': 0.},
    ])
    return d, t


def test_reconciliation():
    d, t = synthetic()
    j = join_events(d, t)
    assert j['matched_trade'].tolist() == [True, False, True]
    assert j.iloc[2]['trade_fee_usd'] == .1


def test_non_executed_never_matched():
    d, t = synthetic()
    d.loc[0, 'execution_status'] = 'REJECTED'
    assert join_events(d, t)['matched_trade'].tolist() == [False, False, True]


def test_audit_clean():
    d, t = synthetic()
    a = audit_run(d, t)
    assert a['transactions'] == 2
    assert a['matched_transactions'] == 2
    for field in ('unmatched_executed_decisions', 'unmatched_transactions',
                  'cash_mismatch_gt_1_cent', 'price_mismatch_gt_1_cent',
                  'cash_flow_mismatch_gt_1_cent', 'trade_cash_flow_mismatch_gt_2_cents'):
        assert a[field] == 0, field


def test_audit_cash_mismatch():
    d, t = synthetic()
    d.loc[2, 'cash_after_usd'] += 1.0
    assert audit_run(d, t)['cash_mismatch_gt_1_cent'] == 1


def test_audit_flow_mismatch():
    d, t = synthetic()
    t.loc[1, 'cash_after_usd'] += 1.0
    assert audit_run(d, t)['cash_flow_mismatch_gt_1_cent'] == 1


def test_audit_fees():
    d, t = synthetic()
    t.loc[1, 'fee_usd'] += 1.0
    assert audit_run(d, t)['trade_cash_flow_mismatch_gt_2_cents'] == 1


def test_reverse_time():
    d, t = synthetic()
    t = t.iloc[::-1].reset_index(drop=True)
    assert audit_run(d, t)['trade_reverse_time'] == 1


def test_unmatched_transactions():
    d, t = synthetic()
    t.loc[1, 'timestamp_ms'] += 300000
    a = audit_run(d, t)
    assert a['unmatched_transactions'] == 1
    assert a['unmatched_executed_decisions'] == 1


def test_initial_cash_inference():
    d, t = synthetic()
    assert initial_cash_reference(t) == pytest.approx(1000.0)


def test_initial_not_assumed_if_first_sell():
    d, t = synthetic()
    t.loc[0, 'action'] = 'SELL'
    assert initial_cash_reference(t) is None


def test_initial_not_assumed_if_prior_holdings():
    d, t = synthetic()
    t.loc[0, 'holdings_after'] = 7
    assert initial_cash_reference(t) is None


def test_metrics_from_recorded_evidence():
    d, t = synthetic()
    m = run_metrics(d, t)
    assert m['return_vs_reference_pct'] == pytest.approx(.48)
    assert m['fees_recorded_usd'] == pytest.approx(.2)
    assert m['last_holdings'] == 0
    assert m['buy_count'] == 1
    assert m['sell_count'] == 1


def test_drawdown_decisions_only():
    d, t = synthetic()
    d.loc[1, 'equity_after_usd'] = 900.0
    m = run_metrics(d, t)
    assert m['max_observed_decision_drawdown_pct'] == pytest.approx(
        100 * (1 - 900 / 999.9))


def test_gap_segmentation():
    d, _ = synthetic()
    d.loc[2, 'timestamp_ms'] += 8 * 86400000
    assert recorded_equity(d)['segment'].tolist() == [0, 0, 1]


def test_trade_time_valuation_is_not_continuous():
    _, t = synthetic()
    tape = trade_tape(t)
    assert tape.iloc[0]['trade_time_portfolio_value_usd'] == pytest.approx(999.9)


def test_duplicate_keys_stable():
    d, t = synthetic()
    first_d = d.iloc[[0]].copy()
    first_t = t.iloc[[0]].copy()
    d = pd.concat([first_d, first_d, d.iloc[1:]], ignore_index=True)
    t = pd.concat([first_t, first_t, t.iloc[1:]], ignore_index=True)
    j = join_events(d, t)
    assert j['matched_trade'].tolist() == [True, True, False, True]
    assert j.loc[0, '_trade_row'] == 0
    assert j.loc[1, '_trade_row'] == 1


def test_report_is_readonly_and_individual():
    d, t = synthetic()
    obj = json.loads(report_json(60, run_metrics(d, t), audit_run(d, t)))
    assert obj['simulation_only'] is True
    assert '60m_decisions' in obj['source']
    assert obj['horizon_minutes'] == 60


def test_run_selection_rejects_unknown(tmp_path):
    with pytest.raises(ValueError):
        load_run(tmp_path, 5)


def test_missing_data_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path, 60)


def test_both_run_files_separate(tmp_path):
    base = tmp_path / BASE
    base.mkdir(parents=True)
    d, t = synthetic()
    d30, t30 = d.copy(), t.copy()
    d30['strategy_action'] = ['HOLD', 'HOLD', 'SELL']
    for h, data in [(30, (d30, t30)), (60, (d, t))]:
        dn, tn = RUNS[h]
        data[0].to_csv(base/dn, index=False)
        data[1].to_csv(base/tn, index=False)
    l30, _ = load_run(tmp_path, 30)
    l60, _ = load_run(tmp_path, 60)
    assert l30.iloc[0]['strategy_action'] == 'HOLD'
    assert l60.iloc[0]['strategy_action'] == 'BUY'


def test_file_bad_numeric_rejected(tmp_path):
    base = tmp_path/BASE
    base.mkdir(parents=True)
    d, t = synthetic()
    t['price'] = t['price'].astype(object)
    t.loc[0, 'price'] = 'invalid'
    dn, tn = RUNS[60]
    d.to_csv(base/dn, index=False)
    t.to_csv(base/tn, index=False)
    with pytest.raises(ValueError, match='price'):
        load_run(tmp_path, 60)


def test_invalid_timestamp_rejected(tmp_path):
    p = tmp_path/'f.csv'
    p.write_text('timestamp_ms,action\nabc,BUY\n', encoding='utf-8')
    with pytest.raises(ValueError, match='horodatage'):
        _read_csv(p, {'timestamp_ms','action'})
