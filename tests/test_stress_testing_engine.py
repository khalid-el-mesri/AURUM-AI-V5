"""Phase 11: deterministic, offline; no trading or mutation."""
from pathlib import Path
import copy
import math

import numpy as np
import pandas as pd
import pytest

from frontend.services.stress_testing_engine import (
    PRESETS, paired_trades, trade_streak_report, additional_cost_stress,
    simulate_shock_path, json_report, historical_inputs,
)


def transactions():
    return pd.DataFrame([
        dict(timestamp_ms=1000,action="BUY", price=100.,quantity=1.,fee_usd=.1,
             cash_change_usd=-100.1,realized_pnl_usd=0.,holdings_after=1.),
        dict(timestamp_ms=2000,action="SELL",price=105.,quantity=1.,fee_usd=.105,
             cash_change_usd=104.895,realized_pnl_usd=4.795,holdings_after=0.),
        dict(timestamp_ms=3000,action="BUY",price=100.,quantity=1.,fee_usd=.1,
             cash_change_usd=-100.1,realized_pnl_usd=0.,holdings_after=1.),
        dict(timestamp_ms=4000,action="SELL",price=95.,quantity=1.,fee_usd=.095,
             cash_change_usd=94.905,realized_pnl_usd=-5.195,holdings_after=0.),
    ])


def test_pair_count(): assert len(paired_trades(transactions()))==2

def test_pair_pnl():
    p=paired_trades(transactions())
    assert p.realized_net_pnl_usd.tolist()==pytest.approx([4.795,-5.195])


def test_pair_fees(): assert paired_trades(transactions()).recorded_fees_usd.sum()==pytest.approx(.4)

def test_pair_notional(): assert paired_trades(transactions()).notional_buy_usd.sum()==pytest.approx(200)

def test_pair_timestamps(): assert paired_trades(transactions()).entry_time_utc.iloc[0].tzinfo is not None

def test_pair_rejects_odd():
    with pytest.raises(ValueError): paired_trades(transactions().iloc[:-1])


def test_pair_rejects_sell_first():
    t=transactions();t.loc[0,'action']='SELL'
    with pytest.raises(ValueError): paired_trades(t)


def test_pair_rejects_nonchronological():
    t=transactions(); t.loc[1,'timestamp_ms']=999
    with pytest.raises(ValueError): paired_trades(t)


def test_pair_rejects_quantity_mismatch():
    t=transactions();t.loc[1,'quantity']=2
    with pytest.raises(ValueError): paired_trades(t)


def test_pair_rejects_negative_fee():
    t=transactions();t.loc[0,'fee_usd']=-1
    with pytest.raises(ValueError): paired_trades(t)


def test_pair_rejects_open_holdings():
    t=transactions();t.loc[3,'holdings_after']=.2
    with pytest.raises(ValueError): paired_trades(t)


def test_pair_rejects_incorrect_pnl():
    t=transactions();t.loc[3,'realized_pnl_usd']=88
    with pytest.raises(ValueError): paired_trades(t)


def test_pair_rejects_nan():
    t=transactions();t.loc[2,'price']=float('nan')
    with pytest.raises(ValueError): paired_trades(t)


def test_streak_outcome():
    p=paired_trades(transactions()); s=trade_streak_report(p)
    assert (s['completed_trades'],s['losing_trades'],s['profitable_trades'])==(2,1,1)
    assert s['max_consecutive_realized_losses']==1


def test_cost_zero_preserves():
    p=paired_trades(transactions()); df,r=additional_cost_stress(p,1000,0)
    assert r['total_additional_cost_usd']==0
    assert r['total_stressed_pnl_usd']==pytest.approx(-.4)
    assert df.stressed_net_pnl_usd.tolist()==pytest.approx(df.realized_net_pnl_usd.tolist())


def test_cost_100bps_symmetric():
    p=paired_trades(transactions());df,r=additional_cost_stress(p,1000,100)
    expected=(p.notional_buy_usd+p.notional_sell_usd).sum()*.01
    assert r['total_additional_cost_usd']==pytest.approx(expected)


def test_cost_increase_decreases_equity():
    p=paired_trades(transactions());_,a=additional_cost_stress(p,1000,0)
    _,b=additional_cost_stress(p,1000,50)
    assert b['final_trade_close_equity_usd'] < a['final_trade_close_equity_usd']


@pytest.mark.parametrize('bps',[-.1,1000.1,float('nan'),float('inf')])
def test_invalid_bps(bps):
    with pytest.raises(ValueError): additional_cost_stress(paired_trades(transactions()),1000,bps)


def test_cost_does_not_mutate():
    p=paired_trades(transactions()); old=p.copy(deep=True)
    additional_cost_stress(p,1000,25)
    pd.testing.assert_frame_equal(p,old)


def test_shock_zero_step_baseline():
    d,r=simulate_shock_path([0.],.05,10000,100)
    assert d.hypothetical_equity_usd.iloc[-1]==pytest.approx(10000)
    assert r['isolated_final_gate_state']=='ACTIVE'


def test_shock_20pct_5pct_exposure():
    d,r=simulate_shock_path([-20.],.05,10000,100)
    assert d.hypothetical_equity_usd.iloc[-1]==pytest.approx(9900)
    assert r['hypothetical_max_drawdown_pct']==pytest.approx(1.)


def test_bear_at_25pct_triggers_halt():
    d,r=simulate_shock_path(PRESETS['Bear market : 5 baisses de −12 %'],.25,10000,100)
    assert r['isolated_final_gate_state']=='HALTED'
    assert len(r['hypothetical_gate_transitions'])==1
    assert r['hypothetical_new_buy_status']=='HALTED'
    assert r['hypothetical_reducing_sell_status']=='APPROVED'


def test_recovery_stays_halted():
    df,r=simulate_shock_path(PRESETS['Chute suivie de reprise'],.25,10000,100)
    assert r['isolated_final_gate_state']=='HALTED'
    assert df.iloc[-1].hypothetical_drawdown_pct < df.hypothetical_drawdown_pct.max()
    assert df.iloc[-1].isolated_gate_state=='HALTED'


def test_buy_rejected_for_full_exposure_without_halt():
    df,r=simulate_shock_path([-15.],.25,10000,100)
    assert r['isolated_final_gate_state']=='ACTIVE'
    assert r['hypothetical_new_buy_status']=='REJECTED'


def test_sell_honors_existing_holdings():
    _,r=simulate_shock_path([-15.],.05,10000,100)
    assert r['hypothetical_reducing_sell_status']=='APPROVED'


@pytest.mark.parametrize('expo',[-.1,0.,.26,float('nan')])
def test_bad_exposure(expo):
    with pytest.raises(ValueError): simulate_shock_path([-10.],expo,10000,100)


@pytest.mark.parametrize('steps',[[float('nan')],[-100.],[201.],[],[float('inf')]])
def test_bad_shock_steps(steps):
    with pytest.raises(ValueError): simulate_shock_path(steps,.05,10000,100)


def test_invalid_initial_capital():
    with pytest.raises(ValueError): simulate_shock_path([-10.],.05,0.,100)


def test_invalid_reference_price():
    with pytest.raises(ValueError): simulate_shock_path([-10.],.05,10000,0.)


def test_json_export_native():
    result=json_report({'v':np.float64(2.4),'n':np.int64(4),'t':pd.Timestamp('2022-01-01',tz='UTC')})
    assert '2022-01-01' in result and '2.4' in result


def test_historical_files_are_immutable(tmp_path):
    # Inspection only: this fixture intentionally does not create any artifacts
    # or attempt to write to the source dataset.
    t=transactions(); before=t.to_csv(index=False)
    paired_trades(t)
    assert t.to_csv(index=False)==before


def test_existing_validation_reconciliation_when_files_exist():
    root=Path(__file__).resolve().parents[1]
    f=root/'reports/backtesting/combined_holding_30m_60m/60m_transactions.csv'
    if not f.is_file(): pytest.skip('Dataset historique non présent dans le paquet de test isolé')
    d,t,s,a=historical_inputs(root)
    p=paired_trades(t)
    assert len(p)==int(s['results']['60m']['completed_trades'])
    assert len(d)==222 and len(t)==180
    assert sum(a['checks'].values())==0
    _,rep=additional_cost_stress(p,10000,0.)
    assert rep['final_trade_close_equity_usd']==pytest.approx(s['results']['60m']['final_equity_usd'],abs=.01)
