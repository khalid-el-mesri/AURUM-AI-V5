"""AURUM-AI V5 Decision Room: read-only interpretation of a verified local ledger.

IMPORTANT: events in this ledger belong to a historical simulation, not to the
currently selected LSTM/GRU/CNN1D validation prediction file. No execution,
portfolio mutation, model loading, or new decision generation happens here.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


LEDGER_RELATIVE_PATH = Path("reports/blockchain/consensus_trend_60m_validation/ledger.json")


def read_verified_ledger(path: str | Path) -> list[dict[str, Any]]:
    """Require the exact project BlockchainLedger integrity checks before reading."""
    from blockchain.ledger import BlockchainLedger

    ledger = BlockchainLedger.load_json(Path(path))
    # load_json rejects malformed metadata and invalid hashes/links/genesis.
    return ledger.blocks


def _obj(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _txt(value: Any) -> str:
    if value is None or value == "":
        return "Non renseigné"
    return str(value)


def extract_stage_data(block: dict[str, Any]) -> dict[str, dict]:
    """Take only recorded fields; never synthesize an agent decision."""
    payload = _obj(block.get("payload"))
    return {
        "forecast": _obj(payload.get("forecast")),
        "strategy": _obj(payload.get("strategy")),
        "risk": _obj(payload.get("risk")),
        "execution": _obj(payload.get("execution")),
        "portfolio": _obj(payload.get("portfolio")),
    }


def pick_recorded(block: dict, stage: str, *fields: str) -> str:
    """Extract only a nested stage attribute that is explicitly recorded."""
    src = extract_stage_data(block).get(stage, {})
    for name in fields:
        if name in src and src[name] is not None and src[name] != "":
            return _txt(src[name])
    return "Non renseigné"


def _iso_or_none(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return None
        return parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None


def block_table(blocks: list[dict]) -> pd.DataFrame:
    """A compact, faithful ledger index for Streamlit widgets and CSV exports."""
    rows = []
    for block in blocks:
        payload = _obj(block.get("payload"))
        risk = _obj(payload.get("risk"))
        strategy = _obj(payload.get("strategy"))
        execution = _obj(payload.get("execution"))
        forecast = _obj(payload.get("forecast"))
        rows.append({
            "index": block.get("index"),
            "timestamp_utc": block.get("timestamp_utc"),
            "event_type": block.get("event_type"),
            "strategy_action": strategy.get("action"),
            "strategy_reason": strategy.get("reason"),
            "risk_status": risk.get("status"),
            "risk_reason": risk.get("reason"),
            "kill_switch_state_recorded": risk.get("system_state"),
            "execution_status": execution.get("status"),
            "forecast_class": forecast.get("predicted_class"),
            "hash": block.get("hash"),
        })
    return pd.DataFrame(rows, columns=[
        "index", "timestamp_utc", "event_type", "strategy_action", "strategy_reason",
        "risk_status", "risk_reason", "kill_switch_state_recorded", "execution_status",
        "forecast_class", "hash",
    ])


def ledger_summary(blocks: list[dict]) -> dict[str, Any]:
    """Count only events with explicit recorded statuses; GENESIS is not a decision."""
    records = block_table(blocks)
    valid = records[records["event_type"] != "GENESIS"]
    return {
        "block_count": len(blocks),
        "historical_events": len(valid),
        "strategy_actions_recorded": int(valid["strategy_action"].notna().sum()),
        "risk_approvals_recorded": int(valid["risk_status"].eq("APPROVED").sum()),
        "executed_status_recorded": int(valid["execution_status"].eq("EXECUTED").sum()),
        "event_types": dict(Counter(valid["event_type"].fillna("Non renseigné"))),
    }


def chronology_audit(blocks: list[dict]) -> dict[str, int]:
    """Timing quality *in addition* to hash-chain integrity; don't relabel it SHA verification."""
    timestamps = [_iso_or_none(block.get("timestamp_utc")) for block in blocks]
    bad = sum(t is None for t in timestamps)
    reversals = 0
    duplicates = 0
    preceding = None
    known = set()
    for ts in timestamps:
        if ts is None:
            continue
        if preceding is not None and ts < preceding:
            reversals += 1
        if ts in known:
            duplicates += 1
        known.add(ts)
        preceding = ts
    return {"invalid_utc_timestamp": bad, "reverse_event_chronology": reversals,
            "repeated_event_timestamps": duplicates}


def payload_evidence(block: dict) -> list[dict[str, str]]:
    """Stages and their recorded status. Missing stage != stage not executed."""
    stages = extract_stage_data(block)
    mapping = [
        ("Forecast", "forecast", ("predicted_class", "predicted_log_return", "prediction")),
        ("Strategy", "strategy", ("action", "reason")),
        ("Risk / Kill Switch", "risk", ("status", "reason", "system_state")),
        ("Virtual Execution", "execution", ("status", "reason")),
        ("Portfolio", "portfolio", ("equity_usd", "capital_after_usd", "quantity")),
    ]
    return [
        {
            "Stage": title,
            "Présence dans le bloc": "Enregistré" if stages[key] else "Non renseigné",
            "Détails enregistrés": " ; ".join(f"{k}: {_txt(stages[key][k])}" for k in attrs
                                            if k in stages[key]) or "Non renseigné",
        }
        for title, key, attrs in mapping
    ]


def risk_limits_from_code() -> dict[str, float]:
    """Display actual configured RiskLimits defaults; do NOT replay historical risk."""
    from risk.risk_agent import RiskLimits
    limits = RiskLimits()
    return {
        "Fraction maximale d'ordre BUY": limits.max_order_fraction,
        "Exposition maximale": limits.max_exposure_fraction,
        "Perte journalière maximale": limits.max_daily_loss_fraction,
        "Drawdown maximal": limits.max_drawdown_fraction,
    }


def export_selected_event(block: dict[str, Any]) -> str:
    return json.dumps({"mode": "READ_ONLY_HISTORICAL", "data_source": "verified_local_ledger",
                       "note": "Ce bloc n'est pas une nouvelle prédiction ou transaction.",
                       "block": block}, ensure_ascii=False, indent=2, allow_nan=False)
