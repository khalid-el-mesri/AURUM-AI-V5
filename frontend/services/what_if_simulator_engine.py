"""AURUM-AI V5 — Phase 12, research-only fixed-trade what-if analysis.

Uses the actual, historical Consensus + Trend 60m / Validation CLOSED trade tape.
No strategy, model, RiskAgent, ExecutionAgent, or ledger is run or modified.

Changing capital or sizing is a *linear sensitivity*, NOT executable portfolio
reconstruction. It holds the historical trades/directions/entry/exit times fixed.
Drawdowns below are sampled at trade close, not backtest 5-minute drawdowns.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from frontend.services.stress_testing_engine import historical_inputs, paired_trades


REFERENCE_NAME = 'Consensus + Trend 60m'
RESEARCH_SPLIT = 'validation'


@dataclass(frozen=True)
class WhatIfInputs:
    starting_capital_usd: float = 10000.0
    size_multiplier: float = 1.0
    extra_cost_bps_per_side: float = 0.0
    fixed_extra_cost_usd_per_trade: float = 0.0


def _checked_float(name: str, value: float, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name} doit être un nombre')
    v = float(value)
    if not math.isfinite(v) or not lower <= v <= upper:
        raise ValueError(f'{name} doit être compris entre {lower:g} et {upper:g}')
    return v


def validate_inputs(p: WhatIfInputs) -> WhatIfInputs:
    if not isinstance(p, WhatIfInputs):
        raise ValueError('Paramètres What-If invalides')
    _checked_float('Capital initial', p.starting_capital_usd, 1000, 1_000_000)
    _checked_float('Coefficient de taille', p.size_multiplier, 0.1, 2.0)
    _checked_float('Coût additionnel bps/côté', p.extra_cost_bps_per_side, 0, 500)
    _checked_float('Coût fixe additionnel/trade', p.fixed_extra_cost_usd_per_trade, 0, 1000)
    return p


def validate_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(pairs, pd.DataFrame) or pairs.empty:
        raise ValueError('Aucune transaction clôturée disponible')
    needed = ['trade_id', 'realized_net_pnl_usd', 'notional_buy_usd', 'notional_sell_usd',
              'recorded_fees_usd']
    missing = [col for col in needed if col not in pairs.columns]
    if missing:
        raise ValueError('Colonnes de trades absentes : ' + ', '.join(missing))
    d = pairs.copy().reset_index(drop=True)
    for col in needed:
        d[col] = pd.to_numeric(d[col], errors='coerce')
        if not np.isfinite(d[col].to_numpy(dtype=float)).all():
            raise ValueError('Colonne non finie : ' + col)
    if not np.array_equal(d['trade_id'].to_numpy(), np.arange(1, len(d) + 1)):
        raise ValueError('Identifiants de trades non consécutifs ou désordonnés')
    if (d[['notional_buy_usd', 'notional_sell_usd']] <= 0).any().any():
        raise ValueError('Notionnels de transaction invalides')
    if (d['recorded_fees_usd'] < 0).any():
        raise ValueError('Frais historiques négatifs')
    for col in ('entry_time_utc', 'exit_time_utc'):
        if col in d.columns:
            d[col] = pd.to_datetime(d[col], utc=True, errors='coerce')
            if d[col].isna().any():
                raise ValueError('Horodatages invalides : ' + col)
    if 'entry_time_utc' in d.columns and 'exit_time_utc' in d.columns:
        if (d['entry_time_utc'] >= d['exit_time_utc']).any():
            raise ValueError('Un trade se termine avant son entrée')
        if not d['entry_time_utc'].is_monotonic_increasing:
            raise ValueError('Trades non chronologiques')
    return d


def load_reference(root: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Read existing files and block analysis when Phase 10 audit detects an issue."""
    _, transactions, summary, audit = historical_inputs(Path(root))
    result = summary['results']['60m']
    if result['model'] != REFERENCE_NAME or summary['evaluation_split'] != RESEARCH_SPLIT:
        raise ValueError('La référence attendue est Consensus + Trend 60m / Validation')
    pairs = validate_pairs(paired_trades(transactions))
    if int(result['completed_trades']) != len(pairs):
        raise ValueError('Le nombre de trades ne correspond pas au résumé historique')
    original = float(result['initial_capital_usd']) + float(pairs['realized_net_pnl_usd'].sum())
    if abs(original - float(result['final_equity_usd'])) > 0.02:
        raise ValueError('P&L des trades incohérent avec le capital du résumé')
    if abs(float(pairs['recorded_fees_usd'].sum()) - float(result['total_fees_usd'])) > 0.02:
        raise ValueError('Frais historiques incohérents')
    return pairs, result, audit


def simulate_fixed_trades(pairs: pd.DataFrame, reference_initial_usd: float,
                          inputs: WhatIfInputs) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Scale existing realized trade P&L and notionals; add new hypothetical costs.

    This never creates/replaces trades, changes market prices, or reruns agents.
    Equity is only evaluated after a trade closes; intratrade exposure is unknown.
    """
    validate_inputs(inputs)
    d = validate_pairs(pairs)
    reference_initial_usd = _checked_float('Capital initial de référence', reference_initial_usd, 1, 1_000_000_000)
    base_scale = float(inputs.starting_capital_usd) / reference_initial_usd
    trade_scale = base_scale * float(inputs.size_multiplier)
    d['reference_trade_pnl_usd'] = d['realized_net_pnl_usd']
    d['scaled_buy_notional_usd'] = d['notional_buy_usd'] * trade_scale
    d['scaled_sell_notional_usd'] = d['notional_sell_usd'] * trade_scale
    d['scaled_recorded_fees_usd'] = d['recorded_fees_usd'] * trade_scale
    d['scaled_reference_pnl_usd'] = d['reference_trade_pnl_usd'] * trade_scale
    d['extra_variable_cost_usd'] = (d['scaled_buy_notional_usd'] + d['scaled_sell_notional_usd']) * float(inputs.extra_cost_bps_per_side) / 10_000
    d['extra_fixed_cost_usd'] = float(inputs.fixed_extra_cost_usd_per_trade)
    d['total_extra_cost_usd'] = d['extra_variable_cost_usd'] + d['extra_fixed_cost_usd']
    d['what_if_trade_pnl_usd'] = d['scaled_reference_pnl_usd'] - d['total_extra_cost_usd']
    d['equity_before_trade_usd'] = float(inputs.starting_capital_usd) + d['what_if_trade_pnl_usd'].cumsum().shift(fill_value=0)
    d['equity_after_trade_usd'] = float(inputs.starting_capital_usd) + d['what_if_trade_pnl_usd'].cumsum()
    equity = d['equity_after_trade_usd'].to_numpy(dtype=float)
    peaks = np.maximum.accumulate(np.r_[float(inputs.starting_capital_usd), equity])
    dd = np.maximum(0., (peaks - np.r_[float(inputs.starting_capital_usd), equity]) / peaks * 100)
    d['trade_close_drawdown_pct'] = dd[1:]
    # Proxy ONLY: buy notional / previous *trade-close* capital. Real in-trade
    # order-size check requires contemporaneous RiskSnapshot, unavailable here.
    d['proxy_buy_fraction_pct'] = np.where(d['equity_before_trade_usd'] > 0,
           d['scaled_buy_notional_usd'] / d['equity_before_trade_usd'] * 100, np.nan)
    if (equity <= 0).any() or (d['equity_before_trade_usd'] <= 0).any():
        raise ValueError('Capital hypothétique épuisé : les hypothèses dépassent la portée du modèle linéaire')
    if not np.isfinite(d[['what_if_trade_pnl_usd', 'trade_close_drawdown_pct', 'proxy_buy_fraction_pct']].to_numpy()).all():
        raise ValueError('Résultat non fini')
    ending = float(equity[-1])
    extra = float(d['total_extra_cost_usd'].sum())
    results: dict[str, Any] = {
        'approach': 'fixed_trades_linear_sensitivity',
        'evaluation_split': RESEARCH_SPLIT,
        'reference_experiment': REFERENCE_NAME,
        'starting_capital_usd': float(inputs.starting_capital_usd),
        'reference_initial_capital_usd': reference_initial_usd,
        'size_multiplier': float(inputs.size_multiplier),
        'notional_scale_factor': trade_scale,
        'extra_cost_bps_per_side': float(inputs.extra_cost_bps_per_side),
        'fixed_extra_cost_usd_per_trade': float(inputs.fixed_extra_cost_usd_per_trade),
        'closed_trades': int(len(d)),
        'total_scaled_recorded_fees_usd': float(d['scaled_recorded_fees_usd'].sum()),
        'total_additional_cost_usd': extra,
        'base_scaled_net_pnl_usd': float(d['scaled_reference_pnl_usd'].sum()),
        'what_if_net_pnl_usd': ending - float(inputs.starting_capital_usd),
        'ending_capital_usd': ending,
        'what_if_return_pct': (ending / float(inputs.starting_capital_usd) - 1) * 100,
        'max_trade_close_drawdown_pct': float(dd.max()),
        'max_proxy_buy_fraction_pct': float(d['proxy_buy_fraction_pct'].max()),
        'proxy_trades_above_5pct': int((d['proxy_buy_fraction_pct'] > 5.0 + 1e-9).sum()),
        'trade_close_dd_above_10pct': bool(dd.max() >= 10.0),
        'min_close_equity_usd': float(equity.min()),
        'winning_trades': int((d['what_if_trade_pnl_usd'] > 0).sum()),
        'losing_trades': int((d['what_if_trade_pnl_usd'] < 0).sum()),
        'daily_loss_limit_verified': False,
        'actual_risk_agent_executed': False,
        'orders_reexecuted': False,
    }
    return d, results


def compare_scenarios(pairs: pd.DataFrame, reference_initial_usd: float,
                      scenarios: dict[str, WhatIfInputs]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if not scenarios or len(scenarios) > 8:
        raise ValueError('Entre 1 et 8 scénarios requis')
    rows = []
    curves: dict[str, pd.DataFrame] = {}
    for name, inputs in scenarios.items():
        if not isinstance(name, str) or not name.strip() or len(name) > 80:
            raise ValueError('Nom de scénario invalide')
        curve, result = simulate_fixed_trades(pairs, reference_initial_usd, inputs)
        rows.append({'scenario': name, 'capital_initial_usd': result['starting_capital_usd'],
                     'capital_final_usd': result['ending_capital_usd'],
                     'net_pnl_usd': result['what_if_net_pnl_usd'],
                     'return_pct': result['what_if_return_pct'],
                     'extra_cost_total_usd': result['total_additional_cost_usd'],
                     'drawdown_clotures_pct': result['max_trade_close_drawdown_pct'],
                     'proxy_ordres_sup_5pct': result['proxy_trades_above_5pct']})
        curves[name] = curve
    return pd.DataFrame(rows), curves


def single_shock_mark_to_market(starting_capital_usd: float, exposure_pct: float,
                                shock_pct: float) -> dict[str, float | bool | str]:
    """One-time **synthetic** mark-to-market of a fixed long position, no orders.

    Do NOT identify this as a realized loss, full replay, or RiskAgent output.
    """
    capital = _checked_float('Capital initial', starting_capital_usd, 1000, 1_000_000)
    exposure = _checked_float('Exposition', exposure_pct, 0, 25)
    shock = _checked_float('Choc de prix', shock_pct, -95, 200)
    equity = capital * (1 + exposure / 100 * shock / 100)
    dd = max(0., (capital-equity)/capital*100)
    effective_exposure = (capital * exposure/100 * (1+shock/100)) / equity * 100
    return {'starting_capital_usd':capital,'hypothetical_exposure_pct':exposure,
            'hypothetical_shock_pct':shock,'hypothetical_equity_usd':equity,
            'hypothetical_pnl_usd':equity-capital, 'hypothetical_drawdown_pct':dd,
            'exposure_after_shock_pct': effective_exposure,
            'reaches_10pct_drawdown_threshold': bool(dd >= 10-1e-10),
            'note':'Prix synthétique à un pas, sans exécution et sans replay des agents'}


def _json_native(value: Any) -> Any:
    if isinstance(value, dict): return {str(k):_json_native(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [_json_native(x) for x in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp,)): return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def export_report(inputs: WhatIfInputs, metrics: dict[str, Any],
                  comparison: pd.DataFrame, reference: dict[str, Any],
                  shock: dict[str, Any]) -> str:
    report = {'phase': '12', 'version': 'whatif_read_only_v1',
       'source': 'Consensus + Trend 60m / Validation',
       'source_is_independent_test': False,
       'assumptions': asdict(inputs), 'reference': {
          'initial_capital_usd': reference['initial_capital_usd'],
          'final_equity_usd': reference['final_equity_usd'],
          'max_observed_drawdown_pct': reference['max_observed_drawdown_pct'],
          'total_fees_usd': reference['total_fees_usd']},
       'what_if': metrics, 'comparison':comparison.to_dict(orient='records'),
       'synthetic_shock':shock,
       'limitations':['Historical fixed entry/exit decisions; no order repricing or RiskAgent rerun.',
                      'Extra costs and scaled size are arithmetic sensitivities, not a new backtest.',
                      'Drawdown on trade closes only, not intraday/5-minute historical drawdown.',
                      'Daily loss limits cannot be verified with these summarized inputs.',
                      'Exploratory validation experiment, not independent test evidence.']}
    return json.dumps(_json_native(report), indent=2, ensure_ascii=False, allow_nan=False)
