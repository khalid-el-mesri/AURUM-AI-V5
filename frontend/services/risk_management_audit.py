"""AURUM-AI V5 Phase 10 — read-only historical risk analysis.

Strictly reads the *Consensus + Trend 60m* validation run. Never calls execution
engines, replays the Kill Switch on historical rows, or invents unavailable
historical RiskSnapshot inputs. Risk limits are imported from the real V2 agent.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from risk.risk_agent import RiskLimits

RUN_DIR = Path("reports/backtesting/combined_holding_30m_60m")
DECISIONS_FILE = "60m_decisions.csv"
TRADES_FILE = "60m_transactions.csv"
SUMMARY_FILE = "comparison_summary.json"

D_REQUIRED = {
    "timestamp_ms", "strategy_action", "risk_status", "risk_reason",
    "risk_timestamp_utc", "execution_status", "execution_reason",
    "system_state", "cash_after_usd", "equity_after_usd", "event",
}
T_REQUIRED = {
    "timestamp_ms", "action", "price", "quantity", "fee_usd",
    "cash_change_usd", "realized_pnl_usd", "cash_after_usd", "holdings_after",
}


def _int_timestamps(df: pd.DataFrame, name: str) -> pd.DataFrame:
    frame = df.copy()
    raw = pd.to_numeric(frame["timestamp_ms"], errors="coerce")
    if raw.isna().any() or not np.isfinite(raw.to_numpy(dtype=float)).all():
        raise ValueError(f"{name}: horodatages invalides")
    if ((raw % 1) != 0).any() or (raw < 0).any():
        raise ValueError(f"{name}: horodatages non entiers ou négatifs")
    frame["timestamp_ms"] = raw.astype("int64")
    if not frame["timestamp_ms"].is_monotonic_increasing:
        raise ValueError(f"{name}: ordre chronologique non conforme")
    return frame


def _csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Source manquante : {path}")
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise ValueError(f"Fichier vide : {path.name}")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} : colonnes absentes : {missing}")
    return _int_timestamps(frame, path.name)


def _finite_values(frame: pd.DataFrame, cols: list[str], file_name: str,
                   optional: bool = False) -> pd.DataFrame:
    for col in cols:
        if col not in frame.columns:
            continue
        val = pd.to_numeric(frame[col], errors="coerce")
        good = np.isfinite(val.to_numpy(dtype=float)) | (val.isna().to_numpy() if optional else False)
        if not good.all():
            raise ValueError(f"{file_name}: numérique invalide dans {col}")
        if not optional and val.isna().any():
            raise ValueError(f"{file_name}: numérique absent dans {col}")
        frame[col] = val
    return frame


def load_sources(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Read immutable local artifacts; no backtesting is triggered."""
    base = Path(root) / RUN_DIR
    decisions = _csv(base / DECISIONS_FILE, D_REQUIRED)
    trades = _csv(base / TRADES_FILE, T_REQUIRED)
    decisions = _finite_values(decisions, ["cash_after_usd", "equity_after_usd"], DECISIONS_FILE)
    trades = _finite_values(trades, ["price", "quantity", "fee_usd", "cash_change_usd",
                                      "realized_pnl_usd", "cash_after_usd", "holdings_after"], TRADES_FILE)
    if (trades["price"] <= 0).any() or (trades["quantity"] <= 0).any():
        raise ValueError("Transactions : prix ou quantités non positifs")
    if (trades["fee_usd"] < 0).any() or (trades["holdings_after"] < -1e-6).any():
        raise ValueError("Transactions : frais négatifs ou positions impossibles")
    if not trades["action"].astype(str).str.upper().isin(["BUY", "SELL"]).all():
        raise ValueError("Transactions : actions inconnues")
    if not set(decisions["execution_status"].dropna().astype(str)).issubset(
        {"EXECUTED", "BLOCKED", "NO_ACTION", "REJECTED", "SKIPPED", "ERROR"}
    ):
        raise ValueError("Décisions : statut d'exécution inconnu")
    path = base / SUMMARY_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Source manquante : {path}")
    with path.open(encoding="utf-8") as fh:
        summary = json.load(fh)
    if not isinstance(summary, dict) or not isinstance(summary.get("results"), dict):
        raise ValueError("Résumé de simulation invalide")
    if summary.get("evaluation_split") != "validation" or summary.get("real_trading") is not False:
        raise ValueError("La page accepte uniquement une simulation historique Validation")
    if "60m" not in summary["results"]:
        raise ValueError("Résumé 60m absent")
    return decisions, trades, summary


def exact_matches(decisions: pd.DataFrame, trades: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Match only EXECUTED rows to virtual transactions via timestamp/action/occurrence.

    Return decisions enriched with matching trade fields and number of *orphan* trades.
    No nearest-time guesses and no re-execution.
    """
    d = decisions.copy().reset_index(drop=True)
    d["_row"] = np.arange(len(d))
    d["_action"] = d["strategy_action"].fillna("").astype(str).str.strip().str.upper()
    is_exec = d["execution_status"].fillna("").astype(str).str.upper().eq("EXECUTED")
    e = d.loc[is_exec, ["_row", "timestamp_ms", "_action"]].copy()
    e["_occ"] = e.groupby(["timestamp_ms", "_action"]).cumcount()
    t = trades.copy().reset_index(drop=True)
    t["_trade_index"] = np.arange(len(t))
    t["_action"] = t["action"].fillna("").astype(str).str.strip().str.upper()
    t["_occ"] = t.groupby(["timestamp_ms", "_action"]).cumcount()
    cols = ["timestamp_ms", "_action", "_occ"]
    matched = e.merge(t, how="left", on=cols, validate="one_to_one", suffixes=("", "_trade"))
    orphans = len(t) - int(matched["_trade_index"].notna().sum())
    sub = matched[["_row", "_trade_index", "fee_usd", "price", "quantity", "holdings_after", "cash_after_usd"]].copy()
    sub = sub.rename(columns={"fee_usd": "matched_fee_usd", "price": "matched_price",
                              "quantity": "matched_quantity", "holdings_after": "matched_holdings_after",
                              "cash_after_usd": "matched_cash_after_usd"})
    enriched = d.merge(sub, how="left", on="_row", validate="one_to_one")
    enriched["matched_trade"] = enriched["_trade_index"].notna()
    enriched["market_time_utc"] = pd.to_datetime(enriched["timestamp_ms"], unit="ms", utc=True)
    return enriched.drop(columns=["_row", "_action"]), orphans


def limit_values() -> dict[str, float]:
    return asdict(RiskLimits())


def _count(series: pd.Series, value: str) -> int:
    return int(series.fillna("").astype(str).str.strip().str.upper().eq(value).sum())


def audit_history(decisions: pd.DataFrame, trades: pd.DataFrame,
                  summary: dict[str, Any]) -> dict[str, Any]:
    """Audit observable properties without asserting absence of leakage or ex-ante validity."""
    enriched, orphans = exact_matches(decisions, trades)
    is_executed = enriched["execution_status"].astype(str).str.upper().eq("EXECUTED")
    unmatched = int((is_executed & ~enriched["matched_trade"]).sum())
    matched = enriched.loc[enriched["matched_trade"]]
    cash_errors = int((matched["cash_after_usd"] - matched["matched_cash_after_usd"]).abs().gt(0.01).sum())
    results = summary["results"]["60m"]
    fees = float(trades["fee_usd"].sum())
    executed = int(is_executed.sum())
    risk_statuses = decisions["risk_status"].fillna("NON RENSEIGNÉ").astype(str).value_counts().to_dict()
    states = decisions["system_state"].fillna("NON RENSEIGNÉ").astype(str).value_counts().to_dict()
    transitions = results.get("kill_switch_transitions", [])
    if not isinstance(transitions, list):
        raise ValueError("Résumé : transitions Kill Switch invalides")
    execution_status = decisions["execution_status"].fillna("").astype(str).str.upper()
    risk_status = decisions["risk_status"].fillna("").astype(str).str.upper()
    incompatible_exec = int(((execution_status == "EXECUTED") & (risk_status != "APPROVED")).sum())
    # Persistent halt may only remain halted within this historical run.
    state_series = decisions["system_state"].fillna("").astype(str).str.upper()
    active_after_halt = int(((state_series == "ACTIVE") & state_series.eq("HALTED").cummax()).sum())
    checks = {
        "EXECUTED sans approbation de risque": incompatible_exec,
        "Retour ACTIVE après HALTED (sans reset documenté)": active_after_halt,
        "Transactions sans décision EXECUTED rapprochée": orphans,
        "Décisions EXECUTED sans transaction rapprochée": unmatched,
        "Écarts de cash supérieurs à 0,01 USD": cash_errors,
        "Écart nombre exécutées / summary": int(executed != 2 * int(results["completed_trades"])),
        "Écart frais / summary > 0,01 USD": int(abs(fees - float(results["total_fees_usd"])) > .01),
        "Écart equity finale / summary > 0,01 USD": int(abs(float(decisions.iloc[-1]["equity_after_usd"]) - float(results["final_equity_usd"])) > .01),
        "Écart états finaux / summary": int(str(decisions.iloc[-1]["system_state"]) != str(results["final_system_state"])),
        "Écart BLOCKED / summary": int(_count(decisions["execution_status"], "BLOCKED") != int(results.get("blocked_orders", 0))),
    }
    return {
        "evaluation_split": summary["evaluation_split"],
        "experiment": results["model"],
        "decisions": len(decisions), "transactions": len(trades),
        "executed": executed, "matched": int(enriched["matched_trade"].sum()),
        "status_counts": risk_statuses, "state_counts": states,
        "blocked_recorded": _count(decisions["execution_status"], "BLOCKED"),
        "no_action_recorded": _count(decisions["risk_status"], "NO_ACTION"),
        "fee_total_usd": fees,
        "initial_capital_usd": float(results["initial_capital_usd"]),
        "final_equity_usd": float(results["final_equity_usd"]),
        "net_return_pct": float(results["net_return_pct"]),
        "max_observed_drawdown_pct": float(results["max_observed_drawdown_pct"]),
        "recorded_system_final": str(results["final_system_state"]),
        "halt_timestamp_ms": results.get("halt_timestamp_ms"),
        "kill_switch_transitions": transitions,
        "limits": limit_values(), "checks": checks,
        "validation_limitations": summary.get("limitations", []),
        "historical_daily_loss_known": False,
        "historical_order_snapshot_known": False,
    }


def risk_timeline(decisions: pd.DataFrame, gap_days: int = 7) -> pd.DataFrame:
    """Sparse *recorded after-decision equity*, not a continuous risk curve."""
    if not isinstance(gap_days, int) or gap_days < 1:
        raise ValueError("gap_days must be a positive integer")
    frame = decisions[["timestamp_ms", "equity_after_usd", "cash_after_usd", "risk_status", "system_state"]].copy()
    frame["market_time_utc"] = pd.to_datetime(frame["timestamp_ms"], unit="ms", utc=True)
    frame["segment"] = frame["timestamp_ms"].diff().gt(gap_days * 86_400_000).fillna(False).cumsum()
    return frame


def _native(obj: Any) -> Any:
    if isinstance(obj, dict): return {str(k): _native(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_native(v) for v in obj]
    if isinstance(obj, tuple): return [_native(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (pd.Timestamp,)): return obj.isoformat()
    if obj is pd.NA: return None
    if isinstance(obj, float) and not math.isfinite(obj): return None
    return obj


def report_json(report: dict[str, Any]) -> str:
    return json.dumps(_native(report), ensure_ascii=False, indent=2, allow_nan=False)
