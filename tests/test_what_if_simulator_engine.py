"""Isolated unit tests + optional end-to-end consistency with research artifacts."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frontend.services.what_if_simulator_engine import (
    WhatIfInputs, validate_inputs, validate_pairs, simulate_fixed_trades,
    compare_scenarios, single_shock_mark_to_market, export_report, load_reference,
)


@pytest.fixture
def pairs():
    return pd.DataFrame({
        'trade_id':[1,2,3],
        'entry_time_utc':pd.to_datetime(['2024-01-01T14:00Z','2024-01-02T14:00Z','2024-01-03T14:00Z']),
        'exit_time_utc':pd.to_datetime(['2024-01-01T15:00Z','2024-01-02T15:00Z','2024-01-03T15:00Z']),
        'realized_net_pnl_usd':[5.,-10.,2.],
        'notional_buy_usd':[500.,500.,500.],
        'notional_sell_usd':[506.,491.,503.],
        'recorded_fees_usd':[1.,1.,1.],
    })


def test_default_reference_equity(pairs):
    frame,r=simulate_fixed_trades(pairs,10000,WhatIfInputs())
    assert r['ending_capital_usd']==pytest.approx(9997)
    assert list(frame['equity_after_trade_usd'])==pytest.approx([10005,9995,9997])
    assert r['what_if_return_pct']==pytest.approx(-.03)
    assert not r['actual_risk_agent_executed']


def test_size_double(pairs):
    _,r=simulate_fixed_trades(pairs,10000,WhatIfInputs(size_multiplier=2))
    assert r['what_if_net_pnl_usd']==pytest.approx(-6)
    assert r['max_proxy_buy_fraction_pct']>5


def test_capital_double_preserves_return(pairs):
    _,a=simulate_fixed_trades(pairs,10000,WhatIfInputs())
    _,b=simulate_fixed_trades(pairs,10000,WhatIfInputs(starting_capital_usd=20000))
    assert b['what_if_net_pnl_usd']==pytest.approx(2*a['what_if_net_pnl_usd'])
    assert b['what_if_return_pct']==pytest.approx(a['what_if_return_pct'])


def test_lower_size_dampens_losses(pairs):
    _,r=simulate_fixed_trades(pairs,10000,WhatIfInputs(size_multiplier=.5))
    assert r['what_if_net_pnl_usd']==pytest.approx(-1.5)


def test_cost_bps_applied_both_sides(pairs):
    d,r=simulate_fixed_trades(pairs,10000,WhatIfInputs(extra_cost_bps_per_side=10))
    total_notional=(pairs.notional_buy_usd+pairs.notional_sell_usd).sum()
    assert r['total_additional_cost_usd']==pytest.approx(total_notional*.001)
    assert r['ending_capital_usd']==pytest.approx(9997-total_notional*.001)
    assert d['extra_variable_cost_usd'].iloc[0]==pytest.approx(1.006)


def test_fixed_cost_per_closed_trade(pairs):
    d,r=simulate_fixed_trades(pairs,10000,WhatIfInputs(fixed_extra_cost_usd_per_trade=2))
    assert r['total_additional_cost_usd']==pytest.approx(6)
    assert r['ending_capital_usd']==pytest.approx(9991)
    assert (d.extra_fixed_cost_usd==2).all()


def test_costs_stack(pairs):
    _,r=simulate_fixed_trades(pairs,10000,WhatIfInputs(extra_cost_bps_per_side=10,fixed_extra_cost_usd_per_trade=2))
    total=(pairs.notional_buy_usd+pairs.notional_sell_usd).sum()*.001+6
    assert r['total_additional_cost_usd']==pytest.approx(total)


def test_extra_cost_cannot_increase_capital(pairs):
    _,base=simulate_fixed_trades(pairs,10000,WhatIfInputs())
    _,cost=simulate_fixed_trades(pairs,10000,WhatIfInputs(extra_cost_bps_per_side=10))
    assert cost['ending_capital_usd']<base['ending_capital_usd']


def test_original_not_mutated(pairs):
    original=pairs.copy(deep=True)
    simulate_fixed_trades(pairs,10000,WhatIfInputs())
    pd.testing.assert_frame_equal(pairs,original)


def test_deterministic(pairs):
    a,r1=simulate_fixed_trades(pairs,10000,WhatIfInputs())
    b,r2=simulate_fixed_trades(pairs,10000,WhatIfInputs())
    pd.testing.assert_frame_equal(a,b)
    assert r1==r2


def test_drawdown_nonnegative(pairs):
    f,r=simulate_fixed_trades(pairs,10000,WhatIfInputs())
    assert r['max_trade_close_drawdown_pct']>=0
    assert (f.trade_close_drawdown_pct>=0).all()


def test_proxy_limit_on_double_size(pairs):
    _,r=simulate_fixed_trades(pairs,10000,WhatIfInputs(size_multiplier=2))
    assert r['proxy_trades_above_5pct']>=1


def test_no_daily_loss_assertion(pairs):
    _,r=simulate_fixed_trades(pairs,10000,WhatIfInputs())
    assert r['daily_loss_limit_verified'] is False
    assert r['orders_reexecuted'] is False


@pytest.mark.parametrize('changes',[
    {'starting_capital_usd':0},
    {'starting_capital_usd':float('nan')},
    {'starting_capital_usd':float('inf')},
    {'starting_capital_usd':1000001},
    {'size_multiplier':0},
    {'size_multiplier':2.1},
    {'extra_cost_bps_per_side':-1},
    {'extra_cost_bps_per_side':501},
    {'fixed_extra_cost_usd_per_trade':-1},
    {'fixed_extra_cost_usd_per_trade':1001},
])
def test_reject_bad_inputs(changes):
    with pytest.raises(ValueError):validate_inputs(replace(WhatIfInputs(),**changes))


def test_reject_bool_inputs():
    with pytest.raises(ValueError):validate_inputs(WhatIfInputs(starting_capital_usd=True))


def test_reject_not_parameters():
    with pytest.raises(ValueError):validate_inputs({'starting_capital_usd':10000})


def test_reject_empty_pairs():
    with pytest.raises(ValueError):validate_pairs(pd.DataFrame())


def test_reject_missing_pair_col(pairs):
    with pytest.raises(ValueError):validate_pairs(pairs.drop(columns=['notional_buy_usd']))


def test_reject_nan_pair_col(pairs):
    pairs.loc[1,'notional_buy_usd']=float('nan')
    with pytest.raises(ValueError):validate_pairs(pairs)


def test_reject_unordered_ids(pairs):
    pairs.loc[1,'trade_id']=7
    with pytest.raises(ValueError):validate_pairs(pairs)


def test_reject_negative_notional(pairs):
    pairs.loc[1,'notional_sell_usd']=-7
    with pytest.raises(ValueError):validate_pairs(pairs)


def test_reject_negative_fee(pairs):
    pairs.loc[1,'recorded_fees_usd']=-1
    with pytest.raises(ValueError):validate_pairs(pairs)


def test_reject_inverted_times(pairs):
    pairs.loc[0,'exit_time_utc']=pd.Timestamp('2023-01-01T15:00Z')
    with pytest.raises(ValueError):validate_pairs(pairs)


def test_reject_empty_scenarios(pairs):
    with pytest.raises(ValueError):compare_scenarios(pairs,10000,{})


def test_compare_has_four_rows(pairs):
    out,curves=compare_scenarios(pairs,10000,{'Original':WhatIfInputs(),
                                              'Frais':WhatIfInputs(extra_cost_bps_per_side=5),
                                              'Taille':WhatIfInputs(size_multiplier=.5),
                                              'Capital':WhatIfInputs(starting_capital_usd=20000)})
    assert len(out)==4
    assert set(curves)==set(out.scenario)


def test_reject_bad_scenario_name(pairs):
    with pytest.raises(ValueError):compare_scenarios(pairs,10000,{'':WhatIfInputs()})


def test_reject_too_many_scenarios(pairs):
    with pytest.raises(ValueError):compare_scenarios(pairs,10000,{str(i):WhatIfInputs() for i in range(9)})


def test_shock_zero():
    r=single_shock_mark_to_market(10000,25,0)
    assert r['hypothetical_equity_usd']==pytest.approx(10000)
    assert r['hypothetical_drawdown_pct']==0


def test_shock_minus_20_with_25pct_exposure():
    r=single_shock_mark_to_market(10000,25,-20)
    assert r['hypothetical_equity_usd']==pytest.approx(9500)
    assert r['hypothetical_drawdown_pct']==pytest.approx(5)
    assert not r['reaches_10pct_drawdown_threshold']


def test_shock_minus_40_triggers_numeric_threshold():
    r=single_shock_mark_to_market(10000,25,-40)
    assert r['reaches_10pct_drawdown_threshold']
    assert r['hypothetical_equity_usd']==pytest.approx(9000)


def test_shock_positive():
    r=single_shock_mark_to_market(10000,25,20)
    assert r['hypothetical_equity_usd']==pytest.approx(10500)
    assert r['hypothetical_drawdown_pct']==0


def test_shock_zero_exposure():
    r=single_shock_mark_to_market(10000,0,-90)
    assert r['hypothetical_equity_usd']==10000


def test_shock_invalid_exposure():
    with pytest.raises(ValueError):single_shock_mark_to_market(10000,26,-20)


def test_shock_invalid_move():
    with pytest.raises(ValueError):single_shock_mark_to_market(10000,20,-96)


def test_json_export_roundtrip(pairs):
    params=WhatIfInputs()
    _,r=simulate_fixed_trades(pairs,10000,params)
    t,_=compare_scenarios(pairs,10000,{'Original':params})
    out=export_report(params,r,t,{'initial_capital_usd':10000,'final_equity_usd':9997,
                             'max_observed_drawdown_pct':2,'total_fees_usd':3},
                              single_shock_mark_to_market(10000,25,-15))
    obj=json.loads(out)
    assert obj['phase']=='12'
    assert obj['source_is_independent_test'] is False
    assert obj['comparison'][0]['scenario']=='Original'
    assert obj['what_if']['daily_loss_limit_verified'] is False


def test_real_reference_if_present():
    root=Path(__file__).resolve().parents[1]
    path=root/'reports/backtesting/combined_holding_30m_60m/comparison_summary.json'
    if not path.exists():pytest.skip('Data fixtures not installed')
    trades,source,audit=load_reference(root)
    assert len(trades)==90
    assert audit['matched']==180
    output,result=simulate_fixed_trades(trades,source['initial_capital_usd'],WhatIfInputs())
    assert result['ending_capital_usd']==pytest.approx(source['final_equity_usd'],abs=.02)
    assert result['total_scaled_recorded_fees_usd']==pytest.approx(source['total_fees_usd'],abs=.02)
    assert result['what_if_return_pct']==pytest.approx(source['net_return_pct'],abs=.002)
    # This is trade-close DD, cannot be compared to 5-minute historical DD.
    assert len(output)==90


def test_real_reference_scaled_cost_if_present():
    root=Path(__file__).resolve().parents[1]
    path=root/'reports/backtesting/combined_holding_30m_60m/comparison_summary.json'
    if not path.exists():pytest.skip('Data fixtures not installed')
    pairs,ref,_=load_reference(root)
    _,plain=simulate_fixed_trades(pairs,10000,WhatIfInputs())
    _,custom=simulate_fixed_trades(pairs,10000,WhatIfInputs(extra_cost_bps_per_side=10))
    assert custom['ending_capital_usd']<plain['ending_capital_usd']
    assert custom['total_additional_cost_usd']>0
