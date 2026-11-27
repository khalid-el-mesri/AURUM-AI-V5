"""AURUM-AI V5 — historical digital twin data engine (read only).

Two independent experiments. The engine NEVER joins their artifacts together.
No strategy, execution, risk, portfolio or blockchain agents are invoked.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCENARIO = "Consensus + Trend 60m — historical replay"
BASE = Path("reports/backtesting/combined_holding_30m_60m")
DECISIONS_REL = BASE / "60m_decisions.csv"
TRANSACTIONS_REL = BASE / "60m_transactions.csv"
ALT_BASE = Path("artifacts/backtesting/nyse_v3/lstm60m_validation_dashboard")
ALT_EQUITY_REL = ALT_BASE / "equity_curve.csv"
ALT_TRADES_REL = ALT_BASE / "trades.csv"

DECISION_REQUIRED = {
    "timestamp_ms", "strategy_action", "risk_status", "execution_status",
    "cash_after_usd", "equity_after_usd", "event", "system_state",
}
TRADE_REQUIRED = {
    "timestamp_ms", "action", "price", "quantity", "fee_usd",
    "cash_after_usd", "holdings_after",
}
ALT_EQUITY_REQUIRED = {"timestamp_ms", "equity_usd"}
ALT_TRADES_REQUIRED = {
    "side", "entry_timestamp_ms", "exit_timestamp_ms", "entry_price",
    "exit_price", "quantity", "net_pnl_usd", "exit_reason",
}


def read_csv_validated(path: Path, required: set[str], sort_keys: tuple[str, ...]) -> pd.DataFrame:
    """Read a known local CSV. Sorting is deliberately NOT performed."""
    if not path.is_file():
        raise FileNotFoundError(f"Fichier source absent : {path}")
    frame = pd.read_csv(path, low_memory=False)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name}: colonnes requises absentes : {sorted(missing)}")
    for name in sort_keys:
        numeric = pd.to_numeric(frame[name], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"{path.name}: horodatages absents ou non valides : {name}")
        if ((numeric % 1) != 0).any():
            raise ValueError(f"{path.name}: horodatages non entiers : {name}")
        frame[name] = numeric.astype("int64")
    return frame


def load_consensus(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions = read_csv_validated(root / DECISIONS_REL, DECISION_REQUIRED, ("timestamp_ms",))
    trades = read_csv_validated(root / TRANSACTIONS_REL, TRADE_REQUIRED, ("timestamp_ms",))
    return decisions, trades


def load_lstm_alternate(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    equity = read_csv_validated(root / ALT_EQUITY_REL, ALT_EQUITY_REQUIRED, ("timestamp_ms",))
    trades = read_csv_validated(
        root / ALT_TRADES_REL, ALT_TRADES_REQUIRED,
        ("entry_timestamp_ms", "exit_timestamp_ms"),
    )
    return equity, trades


def _action(frame: pd.DataFrame, col: str) -> pd.Series:
    return frame[col].astype("string").fillna("").str.upper().str.strip()


def attach_recorded_trades(decisions: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Stable exact matching by event market timestamp and action.

    Occurrence index guards against duplicate (timestamp, action) combinations.
    Only explicit EXECUTED decisions are eligible for matching.
    Unmatched data remain unmatched rather than silently inferred.
    """
    events = decisions.copy().reset_index(drop=True)
    events["_event_row"] = np.arange(len(events))
    events["_action"] = _action(events, "strategy_action")
    events["_executed"] = _action(events, "execution_status") == "EXECUTED"

    selected = events.loc[events["_executed"]].copy()
    selected["_occ"] = selected.groupby(["timestamp_ms", "_action"], dropna=False).cumcount()
    tr = trades.copy().reset_index(drop=True)
    tr["_trade_row"] = np.arange(len(tr))
    tr["_action"] = _action(tr, "action")
    tr["_occ"] = tr.groupby(["timestamp_ms", "_action"], dropna=False).cumcount()

    detail_cols = ["_trade_row", "timestamp_ms", "_action", "_occ", "action", "price", "quantity", "fee_usd", "cash_after_usd", "holdings_after"]
    for optional in ("cash_change_usd", "realized_pnl_usd"):
        if optional in tr:
            detail_cols.append(optional)
    matched = selected[["_event_row", "timestamp_ms", "_action", "_occ"]].merge(
        tr[detail_cols], on=["timestamp_ms", "_action", "_occ"], how="left", validate="one_to_one",
        suffixes=("", "_trade"),
    )
    rename = {
        "action": "trade_action", "price": "trade_price", "quantity": "trade_quantity",
        "fee_usd": "trade_fee_usd", "cash_after_usd": "trade_cash_after_usd",
        "holdings_after": "trade_holdings_after",
        "cash_change_usd": "trade_cash_change_usd", "realized_pnl_usd": "trade_realized_pnl_usd",
    }
    matched = matched.rename(columns={k: v for k, v in rename.items() if k in matched})
    matched = matched.drop(columns=["timestamp_ms", "_action", "_occ"])
    events = events.merge(matched, on="_event_row", how="left", validate="one_to_one")
    events["matched_trade"] = events["_trade_row"].notna()
    events["datetime_market_utc"] = pd.to_datetime(events["timestamp_ms"], unit="ms", utc=True)
    events = events.drop(columns=["_action", "_executed", "_event_row"])
    return events


def matched_trade_rows(events: pd.DataFrame) -> set[int]:
    if "_trade_row" not in events:
        return set()
    return set(events["_trade_row"].dropna().astype(int))


def audit_consensus(decisions: pd.DataFrame, trades: pd.DataFrame) -> dict:
    events = attach_recorded_trades(decisions, trades)
    dt = pd.to_numeric(decisions["timestamp_ms"], errors="coerce")
    tt = pd.to_numeric(trades["timestamp_ms"], errors="coerce")
    matched = events.loc[events["matched_trade"]]
    matched_cash = pd.to_numeric(matched["cash_after_usd"], errors="coerce")
    trade_cash = pd.to_numeric(matched["trade_cash_after_usd"], errors="coerce")
    cash_bad = ((matched_cash - trade_cash).abs() > 0.011) | matched_cash.isna() | trade_cash.isna()
    mismatch_price = 0
    if "price" in matched:
        p = pd.to_numeric(matched["price"], errors="coerce")
        tp = pd.to_numeric(matched["trade_price"], errors="coerce")
        mismatch_price = int(((p - tp).abs() > 0.011).fillna(False).sum())
    recorded_exec = int((_action(decisions, "execution_status") == "EXECUTED").sum())
    matched_count = len(matched)
    return {
        "scenario": SCENARIO,
        "decisions": len(decisions),
        "transactions": len(trades),
        "recorded_executed": recorded_exec,
        "matched_transactions": matched_count,
        "unmatched_executed_decisions": recorded_exec - matched_count,
        "unmatched_trade_rows": len(trades) - len(matched_trade_rows(events)),
        "cash_mismatches_gt_1_cent": int(cash_bad.sum()),
        "price_mismatches_gt_1_cent_when_present": mismatch_price,
        "decision_reverse_timestamps": int((dt.diff() < 0).sum()),
        "trade_reverse_timestamps": int((tt.diff() < 0).sum()),
        "decision_duplicate_timestamps": int(dt.duplicated().sum()),
        "long_gaps_gt_7_days": int((dt.diff() > 7 * 24 * 3600 * 1000).sum()),
        "execution_status_is_not_external_proof": True,
    }


def event_replay(decisions: pd.DataFrame, trades: pd.DataFrame, index: int) -> dict:
    events = attach_recorded_trades(decisions, trades)
    if index < 0 or index >= len(events):
        raise IndexError("Index replay hors des bornes")
    row = events.iloc[index]
    fields = [
        "timestamp_ms", "datetime_market_utc", "session_date", "model", "event", "predicted_class",
        "signal_timestamp_ms", "strategy_action", "strategy_reason", "risk_status", "risk_reason",
        "risk_timestamp_utc", "execution_status", "execution_reason", "system_state",
        "price", "quantity", "cash_after_usd", "equity_after_usd", "realized_pnl_usd",
        "matched_trade", "trade_action", "trade_price", "trade_quantity", "trade_fee_usd",
        "trade_cash_after_usd", "trade_holdings_after", "trade_cash_change_usd", "trade_realized_pnl_usd",
    ]
    out = {}
    for field in fields:
        if field not in row.index:
            out[field] = None
            continue
        v = row[field]
        if isinstance(v, (pd.Timestamp,)):
            out[field] = v.isoformat()
        elif pd.isna(v):
            out[field] = None
        elif isinstance(v, (np.integer, np.floating, np.bool_)):
            out[field] = v.item()
        else:
            out[field] = v
    out["source"] = str(DECISIONS_REL)
    out["transaction_source"] = str(TRANSACTIONS_REL)
    out["replay_sequence_position"] = index + 1
    out["mode"] = "READ_ONLY_HISTORICAL"
    return out


def segmented_equity(decisions: pd.DataFrame, gap_days: float = 7.0) -> pd.DataFrame:
    """Only recorded consensus-run portfolio equity, gap-separated in plots."""
    result = decisions[["timestamp_ms", "equity_after_usd"]].copy()
    result["equity_after_usd"] = pd.to_numeric(result["equity_after_usd"], errors="coerce")
    result = result.dropna(subset=["equity_after_usd"]).copy()
    result["datetime_utc"] = pd.to_datetime(result["timestamp_ms"], unit="ms", utc=True)
    result["segment"] = (result["timestamp_ms"].diff().fillna(0) > gap_days * 86400000).cumsum()
    return result


def alternate_audit(equity: pd.DataFrame, trades: pd.DataFrame) -> dict:
    return {
        "scenario": "NYSE V3 LSTM 60m — separate validation experiment",
        "equity_observations": len(equity),
        "trade_rows": len(trades),
        "equity_reverse_timestamps": int((equity["timestamp_ms"].diff() < 0).sum()),
        "equity_duplicate_timestamps": int(equity["timestamp_ms"].duplicated().sum()),
        "invalid_trade_intervals": int((trades["exit_timestamp_ms"] < trades["entry_timestamp_ms"]).sum()),
        "mixed_with_consensus": False,
    }


def summary_json(report: dict) -> bytes:
    return json.dumps({"mode": "READ_ONLY_HISTORICAL", **report}, indent=2,
                      ensure_ascii=False, default=str).encode("utf-8")
