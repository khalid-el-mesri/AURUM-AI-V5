"""Unit tests: only synthetic rows and ephemeral risk snapshots, no real orders."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frontend.services.risk_management_audit import (
    DECISIONS_FILE, TRADES_FILE, SUMMARY_FILE, RUN_DIR,
    D_REQUIRED, T_REQUIRED, _int_timestamps, load_sources, exact_matches,
    limit_values, audit_history, risk_timeline, report_json,
)
from risk.risk_agent import RiskAgent, RiskSnapshot
from risk.kill_switch import RiskKillSwitch


def fixture_data():
    d = pd.DataFrame([
        dict(timestamp_ms=1000000, strategy_action='BUY', risk_status='APPROVED',
             risk_reason='BUY allowed', risk_timestamp_utc='2026-01-01T00:00:00Z',
             execution_status='EXECUTED', execution_reason='virtual', system_state='ACTIVE',
             cash_after_usd=9499.5, equity_after_usd=9999.5, event='FORECAST_DECISION'),
        dict(timestamp_ms=1300000, strategy_action='HOLD', risk_status='NO_ACTION',
             risk_reason='HOLD', risk_timestamp_utc='2026-01-01T00:00:01Z',
             execution_status='NO_ACTION', execution_reason='No order', system_state='ACTIVE',
             cash_after_usd=9499.5, equity_after_usd=10000.0, event='FORECAST_DECISION'),
        dict(timestamp_ms=1600000, strategy_action='SELL', risk_status='APPROVED',
             risk_reason='SELL allowed', risk_timestamp_utc='2026-01-01T00:00:02Z',
             execution_status='EXECUTED', execution_reason='virtual', system_state='ACTIVE',
             cash_after_usd=10000.0, equity_after_usd=10000.0, event='SCHEDULED_EXIT'),
    ])
    t = pd.DataFrame([
        dict(timestamp_ms=1000000, action='BUY', price=100., quantity=5., fee_usd=.5,
             cash_change_usd=-500.5, realized_pnl_usd=0., cash_after_usd=9499.5,
             holdings_after=5.),
        dict(timestamp_ms=1600000, action='SELL', price=100.2, quantity=5., fee_usd=.5,
             cash_change_usd=500.5, realized_pnl_usd=0., cash_after_usd=10000.,
             holdings_after=0.),
    ])
    s = {
        'evaluation_split':'validation', 'real_trading':False,
        'results': {'60m': {
            'model':'Synthetic 60m', 'initial_capital_usd':10000.,
            'final_equity_usd':10000., 'net_return_pct':0.,
            'completed_trades':1,'total_fees_usd':1.,'blocked_orders':0,
            'max_observed_drawdown_pct':0.5,'final_system_state':'ACTIVE',
            'halt_timestamp_ms':None,'kill_switch_transitions':[]
        }}, 'limitations':['Synthetic only.']
    }
    return d,t,s


def write_files(tmp_path, d=None, t=None, s=None):
    original = fixture_data()
    d = original[0] if d is None else d
    t = original[1] if t is None else t
    s = original[2] if s is None else s
    p = tmp_path / RUN_DIR
    p.mkdir(parents=True, exist_ok=True)
    d.to_csv(p / DECISIONS_FILE, index=False)
    t.to_csv(p / TRADES_FILE, index=False)
    (p / SUMMARY_FILE).write_text(json.dumps(s), encoding='utf8')
    return p


def test_expected_columns_are_explicit():
    d,t,_ = fixture_data()
    assert D_REQUIRED <= set(d) and T_REQUIRED <= set(t)


def test_load_sources_from_synthetic_files(tmp_path):
    write_files(tmp_path)
    d,t,s=load_sources(tmp_path)
    assert len(d)==3 and len(t)==2 and s['evaluation_split']=='validation'


def test_missing_csv_fails_closed(tmp_path):
    write_files(tmp_path)
    (tmp_path/RUN_DIR/DECISIONS_FILE).unlink()
    with pytest.raises(FileNotFoundError): load_sources(tmp_path)


def test_missing_summary_fails_closed(tmp_path):
    write_files(tmp_path)
    (tmp_path/RUN_DIR/SUMMARY_FILE).unlink()
    with pytest.raises(FileNotFoundError): load_sources(tmp_path)


def test_missing_mandatory_column(tmp_path):
    d,t,s=fixture_data();write_files(tmp_path,d.drop(columns=['risk_status']),t,s)
    with pytest.raises(ValueError, match='colonnes absentes'): load_sources(tmp_path)


def test_rejects_test_split(tmp_path):
    d,t,s=fixture_data();s['evaluation_split']='test';write_files(tmp_path,d,t,s)
    with pytest.raises(ValueError, match='Validation'): load_sources(tmp_path)


def test_rejects_real_trading(tmp_path):
    d,t,s=fixture_data();s['real_trading']=True;write_files(tmp_path,d,t,s)
    with pytest.raises(ValueError, match='Validation'): load_sources(tmp_path)


def test_rejects_unknown_trade_action(tmp_path):
    d,t,s=fixture_data();t.loc[0,'action']='SHORT';write_files(tmp_path,d,t,s)
    with pytest.raises(ValueError, match='actions inconnues'):load_sources(tmp_path)


def test_rejects_negative_fees(tmp_path):
    d,t,s=fixture_data();t.loc[0,'fee_usd']=-.1;write_files(tmp_path,d,t,s)
    with pytest.raises(ValueError, match='frais négatifs'):load_sources(tmp_path)


def test_rejects_non_finite_price(tmp_path):
    d,t,s=fixture_data();t.loc[0,'price']=np.inf;write_files(tmp_path,d,t,s)
    with pytest.raises(ValueError):load_sources(tmp_path)


def test_rejects_reverse_time():
    d,_,_=fixture_data()
    with pytest.raises(ValueError, match='ordre chronologique'):
        _int_timestamps(d.iloc[::-1], 'x')


def test_strict_matching_and_not_hold():
    d,t,_=fixture_data()
    matched, orphan=exact_matches(d,t)
    assert matched.matched_trade.tolist()==[True,False,True]
    assert orphan==0


def test_orphan_trade_detected():
    d,t,_=fixture_data(); t.loc[1,'timestamp_ms']=1900000
    matched, orphan=exact_matches(d,t)
    assert orphan==1 and matched.matched_trade.sum()==1


def test_wrong_action_is_not_matched():
    d,t,_=fixture_data();t.loc[0,'action']='SELL'
    matched,orphan=exact_matches(d,t)
    assert orphan==1 and matched.matched_trade.sum()==1


def test_occurrence_matching_duplicate_event():
    d,t,_=fixture_data()
    # Add an independent identical-timestamp occurrence on each side.
    extra_d=d.iloc[[0]].copy();extra_t=t.iloc[[0]].copy()
    d=pd.concat([d.iloc[:1],extra_d,d.iloc[1:]],ignore_index=True)
    t=pd.concat([t.iloc[:1],extra_t,t.iloc[1:]],ignore_index=True)
    result, orphan=exact_matches(d,t)
    assert result.matched_trade.sum()==3 and orphan==0


def test_clean_report():
    d,t,s=fixture_data();r=audit_history(d,t,s)
    assert r['matched']==2 and all(v==0 for v in r['checks'].values())
    assert r['status_counts']=={'APPROVED':2,'NO_ACTION':1}


def test_cash_discrepancy_found():
    d,t,s=fixture_data();d.loc[0,'cash_after_usd']-=2
    assert audit_history(d,t,s)['checks']['Écarts de cash supérieurs à 0,01 USD']==1


def test_fee_discrepancy_found():
    d,t,s=fixture_data();s['results']['60m']['total_fees_usd']=10
    assert audit_history(d,t,s)['checks']['Écart frais / summary > 0,01 USD']==1


def test_final_equity_discrepancy_found():
    d,t,s=fixture_data();d.loc[2,'equity_after_usd']=9995
    assert audit_history(d,t,s)['checks']['Écart equity finale / summary > 0,01 USD']==1


def test_limits_exactly_match_code():
    assert limit_values()=={'max_order_fraction':.05,'max_exposure_fraction':.25,
                            'max_daily_loss_fraction':.02,'max_drawdown_fraction':.10}


@pytest.mark.parametrize('snapshot,expected',[
    (RiskSnapshot(0,.05,0,0),'APPROVED'),
    (RiskSnapshot(0,.0501,0,0),'REJECTED'),
    (RiskSnapshot(.24,.05,0,0),'REJECTED'),
    (RiskSnapshot(0,.05,.02,0),'REJECTED'),
    (RiskSnapshot(0,.05,0,.10),'REJECTED'),
])
def test_actual_risk_agent_buy_boundaries(snapshot,expected):
    assert RiskAgent().decide('BUY',snapshot).status==expected


def test_sell_while_daily_loss_high():
    s=RiskSnapshot(.05,.05,.5,.15,held_fraction=.05)
    assert RiskAgent().decide('SELL',s).status=='APPROVED'


def test_sell_cannot_exceed_holdings():
    s=RiskSnapshot(.05,.06,.5,.15,held_fraction=.05)
    assert RiskAgent().decide('SELL',s).status=='REJECTED'


def test_kill_switch_stays_halted_after_recovery():
    gate=RiskKillSwitch()
    x=gate.evaluate('HOLD', RiskSnapshot(0,0,0,.11),1000000)
    assert x.system_state=='HALTED'
    y=gate.evaluate('BUY', RiskSnapshot(0,.05,0,0),1300000)
    assert y.status=='HALTED' and y.system_state=='HALTED'
    z=gate.evaluate('HOLD', RiskSnapshot(0,0,0,0),1600000)
    assert z.system_state=='HALTED' and len(gate.transitions)==1


def test_risk_timeline_gap_creates_segments():
    d,t,s=fixture_data()
    d.loc[2,'timestamp_ms']+=9*86400000
    r=risk_timeline(d)
    assert r.segment.tolist()==[0,0,1]


def test_report_serializes_valid_json():
    d,t,s=fixture_data();r=audit_history(d,t,s)
    assert json.loads(report_json(r))['recorded_system_final']=='ACTIVE'


def test_no_history_input_reclassified_as_proof():
    d,t,s=fixture_data();r=audit_history(d,t,s)
    assert r['historical_daily_loss_known'] is False
    assert r['historical_order_snapshot_known'] is False


def test_executed_without_risk_approval_is_flagged():
    d,t,s=fixture_data();d.loc[0,'risk_status']='REJECTED'
    r=audit_history(d,t,s)
    assert r['checks']['EXECUTED sans approbation de risque']==1


def test_halted_to_active_without_reset_is_flagged():
    d,t,s=fixture_data()
    d.loc[1,'system_state']='HALTED'
    r=audit_history(d,t,s)
    assert r['checks']['Retour ACTIVE après HALTED (sans reset documenté)']==1
