"""Unit tests for the read-only Decision Room; synthetic events only."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from blockchain.ledger import BlockchainLedger
from frontend.services.decision_room_audit import (
    block_table, chronology_audit, export_selected_event, extract_stage_data,
    ledger_summary, payload_evidence, pick_recorded, read_verified_ledger,
    risk_limits_from_code,
)


def ledger():
    l = BlockchainLedger()
    l.append_event("FORECAST_DECISION", {
        "forecast": {"predicted_class": 2},
        "strategy": {"action": "BUY", "reason": "Model direction"},
        "risk": {"status": "APPROVED", "reason": "Within limits", "system_state": "ACTIVE"},
        "execution": {"status": "EXECUTED"},
        "portfolio": {"quantity": 1.0},
    }, timestamp_utc="2024-11-08T18:30:00+00:00")
    l.append_event("SCHEDULED_EXIT", {
        "strategy": {"action": "SELL", "reason": "Scheduled exit"},
        "risk": {"status": "APPROVED"},
    }, timestamp_utc="2024-11-08T18:40:00+00:00")
    return l


def test_load_valid(tmp_path):
    p = tmp_path / "ledger.json"
    ledger().save_json(p)
    blocks = read_verified_ledger(p)
    assert len(blocks) == 3
    assert blocks[0]["event_type"] == "GENESIS"


def test_reject_tampered_hash(tmp_path):
    p = tmp_path / "ledger.json"
    ledger().save_json(p)
    content = json.loads(p.read_text(encoding="utf-8"))
    content["blocks"][1]["payload"]["strategy"]["action"] = "SELL"
    p.write_text(json.dumps(content), encoding="utf-8")
    import pytest
    with pytest.raises(ValueError):
        read_verified_ledger(p)


def test_summary_counts_only_recorded_execution():
    summary = ledger_summary(ledger().blocks)
    assert summary["block_count"] == 3
    assert summary["historical_events"] == 2
    assert summary["risk_approvals_recorded"] == 2
    assert summary["executed_status_recorded"] == 1


def test_missing_execution_is_not_executed():
    df = block_table(ledger().blocks)
    assert df.loc[2, "execution_status"] is None


def test_exact_original_reason():
    b = ledger().blocks[1]
    assert pick_recorded(b, "strategy", "reason") == "Model direction"


def test_missing_stage_is_marked_not_recorded():
    b = ledger().blocks[2]
    stage = payload_evidence(b)
    exec_row = next(x for x in stage if x["Stage"] == "Virtual Execution")
    assert exec_row["Présence dans le bloc"] == "Non renseigné"


def test_stage_does_not_infer_missing_forecast():
    assert extract_stage_data(ledger().blocks[2])["forecast"] == {}


def test_chronology_standard():
    audit = chronology_audit(ledger().blocks)
    assert audit["invalid_utc_timestamp"] == 0
    assert audit["reverse_event_chronology"] == 0


def test_chronology_reversal_report():
    blocks = copy.deepcopy(ledger().blocks)
    blocks[2]["timestamp_utc"] = "2024-11-07T18:40:00+00:00"
    assert chronology_audit(blocks)["reverse_event_chronology"] == 1


def test_chronology_bad_timestamp_report():
    blocks = copy.deepcopy(ledger().blocks)
    blocks[2]["timestamp_utc"] = "not a date"
    assert chronology_audit(blocks)["invalid_utc_timestamp"] == 1


def test_risk_limit_source_matches_agent():
    values = risk_limits_from_code()
    assert values["Drawdown maximal"] == 0.10
    assert values["Fraction maximale d'ordre BUY"] == 0.05


def test_json_export_keeps_original_payload():
    block = ledger().blocks[1]
    exported = json.loads(export_selected_event(block))
    assert exported["block"] == block
    assert exported["mode"] == "READ_ONLY_HISTORICAL"


def test_no_data_mutation_by_audits():
    before = ledger().blocks
    after = copy.deepcopy(before)
    ledger_summary(before)
    block_table(before)
    payload_evidence(before[1])
    assert before == after
