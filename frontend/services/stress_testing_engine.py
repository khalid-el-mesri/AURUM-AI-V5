"""AURUM-AI V5 Phase 11 — isolated, hypothetical stress diagnostics.

Read-only source: Consensus + Trend 60m / validation from Phase 10.
Historical realized trade tape is never altered. Shock paths are SYNTHETIC,
not observed XOM returns, historical replays, or an investable execution system.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from frontend.services.risk_management_audit import (load_sources, audit_history)
from risk.risk_agent import RiskAgent, RiskLimits, RiskSnapshot
from risk.kill_switch import RiskKillSwitch

EXPERIMENT = "Consensus + Trend 60m"
PRESETS: dict[str, tuple[float, ...]] = {
    "Choc unique −15 %": (-15.0,),
    "Bear market : 5 baisses de −12 %": (-12., -12., -12., -12., -12.),
    "Volatilité : mouvements alternés": (12., -18., 15., -22., 12., -20.),
    "Série adverse : 7 baisses de −8 %": (-8., -8., -8., -8., -8., -8., -8.),
    "Chute suivie de reprise": (-35., -20., 60., 20.),
}


def historical_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    decisions, trades, summary = load_sources(Path(root))
    audit = audit_history(decisions, trades, summary)
    if any(audit["checks"].values()):
        raise ValueError(f"L'audit de référence signale des écarts : {audit['checks']}")
    if summary["results"]["60m"].get("model") != EXPERIMENT:
        raise ValueError("Mauvaise expérience : seul Consensus + Trend 60m est accepté")
    return decisions, trades, summary, audit


def paired_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Produce *closed* trade outcomes from two sequential BUY/SELL transactions.

    Rejects overlapping/open histories rather than guessing matches.
    """
    need = {"timestamp_ms", "action", "price", "quantity", "fee_usd", "cash_change_usd", "realized_pnl_usd", "holdings_after"}
    if need - set(trades):
        raise ValueError("Colonnes absentes pour la construction des trades clôturés")
    if len(trades) == 0 or len(trades) % 2:
        raise ValueError("Il faut un nombre pair de transactions BUY/SELL")
    t = trades.reset_index(drop=True).copy()
    for c in ("price", "quantity", "fee_usd", "cash_change_usd", "realized_pnl_usd", "holdings_after"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
        if not np.isfinite(t[c].to_numpy(dtype=float)).all():
            raise ValueError("Valeur numérique non finie : " + c)
    if not t["timestamp_ms"].is_monotonic_increasing:
        raise ValueError("Transactions non chronologiques")
    rows = []
    for i in range(0, len(t), 2):
        buy, sell = t.iloc[i], t.iloc[i + 1]
        if str(buy.action).upper() != "BUY" or str(sell.action).upper() != "SELL":
            raise ValueError(f"Séquence BUY/SELL non conforme, paire {i//2 + 1}")
        if int(buy.timestamp_ms) >= int(sell.timestamp_ms):
            raise ValueError("SELL doit suivre BUY")
        if min(buy.price, sell.price, buy.quantity, sell.quantity) <= 0:
            raise ValueError("Prix et quantités doivent être positifs")
        if min(buy.fee_usd, sell.fee_usd) < 0:
            raise ValueError("Frais négatifs")
        if not math.isclose(float(buy.quantity), float(sell.quantity), rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError("Quantité clôturée incohérente")
        if abs(float(buy.holdings_after) - float(buy.quantity)) > 1e-6 or abs(float(sell.holdings_after)) > 1e-6:
            raise ValueError("Positions des paires incohérentes")
        net = float(buy.cash_change_usd + sell.cash_change_usd)
        if abs(net - float(sell.realized_pnl_usd)) > 0.02:
            raise ValueError("Le P&L réalisé ne correspond pas aux variations de cash")
        rows.append({
            "trade_id": i // 2 + 1,
            "entry_time_utc": pd.to_datetime(int(buy.timestamp_ms), unit="ms", utc=True),
            "exit_time_utc": pd.to_datetime(int(sell.timestamp_ms), unit="ms", utc=True),
            "entry_price_usd": float(buy.price), "exit_price_usd": float(sell.price),
            "quantity": float(buy.quantity),
            "notional_buy_usd": float(buy.quantity * buy.price),
            "notional_sell_usd": float(sell.quantity * sell.price),
            "recorded_fees_usd": float(buy.fee_usd + sell.fee_usd),
            "realized_net_pnl_usd": net,
        })
    return pd.DataFrame(rows)


def trade_streak_report(pairs: pd.DataFrame) -> dict[str, Any]:
    if pairs.empty: raise ValueError("Aucun trade clôturé")
    pnl = pairs["realized_net_pnl_usd"].to_numpy(dtype=float)
    streak = 0
    best = 0
    end = None
    for index, value in enumerate(pnl):
        if value < 0:
            streak += 1
            if streak > best:
                best = streak
                end = index
        else:
            streak = 0
    return {"completed_trades": len(pnl), "profitable_trades": int((pnl > 0).sum()),
            "losing_trades": int((pnl < 0).sum()), "flat_trades": int((pnl == 0).sum()),
            "max_consecutive_realized_losses": best,
            "worst_trade_pnl_usd": float(pnl.min()),
            "loss_streak_start_trade": None if end is None else int(end + 2 - best),
            "loss_streak_end_trade": None if end is None else int(end + 1)}


def additional_cost_stress(pairs: pd.DataFrame, initial_capital: float, additional_bps_per_side: float) -> tuple[pd.DataFrame, dict[str, float]]:
    """Counterfactual costs ONLY; execution actions/price paths remain fixed.

    pnl already includes recorded 10bps/side hypothetical fees. Additional bps
    are incremental and do not overwrite source fees or run agents.
    """
    bps = float(additional_bps_per_side)
    if not math.isfinite(bps) or not 0 <= bps <= 1000:
        raise ValueError("Glissement supplémentaire en bps hors [0,1000]")
    if not math.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("Capital initial invalide")
    if pairs.empty: raise ValueError("Aucun trade")
    result = pairs.copy()
    result["extra_slippage_usd"] = (result["notional_buy_usd"] + result["notional_sell_usd"]) * bps / 10000
    result["stressed_net_pnl_usd"] = result["realized_net_pnl_usd"] - result["extra_slippage_usd"]
    result["equity_after_trade_usd"] = initial_capital + result["stressed_net_pnl_usd"].cumsum()
    peak = np.maximum.accumulate(np.r_[initial_capital, result["equity_after_trade_usd"].to_numpy()])
    equity = np.r_[initial_capital, result["equity_after_trade_usd"].to_numpy()]
    drawdown = np.maximum(0, (peak - equity) / peak * 100)
    report = {
        "extra_bps_per_side": bps,
        "total_additional_cost_usd": float(result["extra_slippage_usd"].sum()),
        "total_stressed_pnl_usd": float(result["stressed_net_pnl_usd"].sum()),
        "final_trade_close_equity_usd": float(equity[-1]),
        "trade_close_return_pct": float((equity[-1] / initial_capital - 1) * 100),
        "max_trade_close_drawdown_pct": float(drawdown.max()),
        "stressed_losing_trades": int((result["stressed_net_pnl_usd"] < 0).sum()),
    }
    return result, report


def simulate_shock_path(steps_pct: tuple[float, ...] | list[float], exposure_fraction: float,
                        initial_capital_usd: float, initial_price_usd: float,
                        order_fraction: float = 0.05) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Isolated synthetic one-session MTM of a fixed long XOM position.

    No execution, no transaction fees, no funding, no model forecasts. The
    existing RiskKillSwitch observes each stress point via HOLD; final BUY/SELL
    are separate hypothetical decisions, not actual simulated fills.
    """
    if not 0 < exposure_fraction <= 0.25 or not math.isfinite(exposure_fraction):
        raise ValueError("Exposition hypothétique limitée à (0,25]")
    if not 0 < order_fraction <= 0.05 or not math.isfinite(order_fraction):
        raise ValueError("Fraction de nouvel ordre limitée à (0,0.05]")
    if not math.isfinite(initial_capital_usd) or initial_capital_usd <= 0:
        raise ValueError("Capital initial invalide")
    if not math.isfinite(initial_price_usd) or initial_price_usd <= 0:
        raise ValueError("Prix de référence invalide")
    if not steps_pct or len(steps_pct) > 100:
        raise ValueError("Scénario vide ou trop long")
    for s in steps_pct:
        if not math.isfinite(float(s)) or not -99 < float(s) <= 200:
            raise ValueError("Variations de prix hors bornes")
    cash = initial_capital_usd * (1 - exposure_fraction)
    qty = initial_capital_usd * exposure_fraction / initial_price_usd
    price = initial_price_usd
    peak = initial_capital_usd
    switch = RiskKillSwitch(risk_agent=RiskAgent())
    rows = []
    for n, movement in enumerate([0.0] + list(steps_pct)):
        if n:
            price *= (1 + float(movement) / 100)
        equity = cash + qty * price
        peak = max(peak, equity)
        dd = max(0., (peak - equity) / peak)
        daily_loss = min(1., max(0., (initial_capital_usd - equity) / initial_capital_usd))
        exposure = min(1., max(0., qty * price / equity))
        snapshot = RiskSnapshot(current_exposure_fraction=exposure,
            proposed_order_fraction=0., daily_loss_fraction=daily_loss,
            drawdown_fraction=min(dd, 1.0), held_fraction=exposure)
        gate = switch.evaluate("HOLD", snapshot, n * 300000)
        rows.append({"step": n, "hypothetical_return_step_pct": float(movement),
             "hypothetical_price_usd": float(price), "hypothetical_equity_usd": float(equity),
             "hypothetical_drawdown_pct": dd * 100, "hypothetical_daily_loss_pct": daily_loss * 100,
             "hypothetical_exposure_pct": exposure * 100,
             "isolated_gate_state": gate.system_state})
    frame = pd.DataFrame(rows)
    last = frame.iloc[-1]
    last_snap = RiskSnapshot(current_exposure_fraction=float(last.hypothetical_exposure_pct) / 100,
             proposed_order_fraction=order_fraction,
             daily_loss_fraction=float(last.hypothetical_daily_loss_pct) / 100,
             drawdown_fraction=float(last.hypothetical_drawdown_pct) / 100,
             held_fraction=float(last.hypothetical_exposure_pct) / 100)
    buy = switch.evaluate("BUY", last_snap, (len(frame)+1)*300000)
    sell_snap = RiskSnapshot(current_exposure_fraction=last_snap.current_exposure_fraction,
             proposed_order_fraction=last_snap.held_fraction,
             daily_loss_fraction=last_snap.daily_loss_fraction,
             drawdown_fraction=last_snap.drawdown_fraction,
             held_fraction=last_snap.held_fraction)
    sell = switch.evaluate("SELL", sell_snap, (len(frame)+2)*300000)
    report = {"initial_capital_usd": float(initial_capital_usd),
              "initial_price_usd_from_historical_buy": float(initial_price_usd),
              "hypothetical_start_exposure_pct": float(exposure_fraction * 100),
              "hypothetical_end_equity_usd": float(last.hypothetical_equity_usd),
              "hypothetical_max_drawdown_pct": float(frame.hypothetical_drawdown_pct.max()),
              "isolated_final_gate_state": str(switch.state),
              "hypothetical_gate_transitions": list(switch.transitions),
              "hypothetical_new_buy_status": str(buy.status),
              "hypothetical_reducing_sell_status": str(sell.status),
              "hypothetical_new_buy_reason": str(buy.reason),
              "hypothetical_reducing_sell_reason": str(sell.reason),
              "assumptions": ["Price movements explicitly hypothetical, not observed XOM history",
                 "Single synthetic intraday session; daily loss uses scenario start equity as base",
                 "Fixed initial shares, no rebalancing, no execution, fees or slippage in shock path",
                 "BUY and SELL at final checkpoint are isolated rule checks, never orders",
                 "A HALTED state never automatically resets"]}
    return frame, report


def json_report(data: Any) -> str:
    def native(obj: Any) -> Any:
        if isinstance(obj, dict): return {str(k): native(v) for k,v in obj.items()}
        if isinstance(obj, (tuple, list)): return [native(v) for v in obj]
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, pd.Timestamp): return obj.isoformat()
        if isinstance(obj, float) and not math.isfinite(obj): return None
        return obj
    return json.dumps(native(data), ensure_ascii=False, allow_nan=False, indent=2)
