"""Phase 13 integration tests: use authentic project's blockchain.ledger API.

All test ledgers live in pytest tmp_path, never in the real reports folder.
"""
import json
from hashlib import sha256
from pathlib import Path

import pytest

from blockchain.ledger import BlockchainLedger, calculate_hash
from frontend.services.blockchain_ledger_audit import (
    LEDGER_RELATIVE_PATH, check_one_block, chronology_report, event_action,
    event_rows, export_block_json, export_summary_json, filter_rows,
    read_verified_ledger, source_metadata, summarize,
)


@pytest.fixture
def ledger_file(tmp_path):
    ledger = BlockchainLedger()
    ledger.append_event("FORECAST_DECISION", {"strategy": {"action": "BUY", "reason": "Synthetic"},
                                            "risk": {"status": "APPROVED", "system_state": "ACTIVE"},
                                            "execution": {"status": "EXECUTED"}},
                        "2024-01-02T14:30:00+00:00")
    ledger.append_event("FORECAST_DECISION", {"strategy": {"action": "HOLD"},
                                            "risk": {"status": "NO_ACTION"}},
                        "2024-01-02T14:35:00+00:00")
    ledger.append_event("SCHEDULED_EXIT", {"strategy": {"action": "SELL"},
                                          "risk": {"status": "APPROVED"},
                                          "execution": {"status": "EXECUTED"}},
                        "2024-01-02T14:40:00+00:00")
    p = tmp_path / "ledger.json"
    ledger.save_json(p)
    return p


def _mutate(path, fn):
    d = json.loads(path.read_text(encoding="utf-8"))
    fn(d)
    path.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")


def test_path_is_historical():
    assert LEDGER_RELATIVE_PATH.parts[-2:] == ("consensus_trend_60m_validation", "ledger.json")


def test_ledger_read(ledger_file):
    s = read_verified_ledger(ledger_file)
    assert len(s.blocks) == 4
    assert s.blocks[0]["event_type"] == "GENESIS"


def test_file_checksum_matches_exact_bytes(ledger_file):
    s = read_verified_ledger(ledger_file)
    assert s.file_sha256 == sha256(ledger_file.read_bytes()).hexdigest()
    assert s.size_bytes == len(ledger_file.read_bytes())


def test_read_does_not_modify_file(ledger_file):
    before = ledger_file.read_bytes()
    read_verified_ledger(ledger_file)
    assert before == ledger_file.read_bytes()


def test_summary(ledger_file):
    s = summarize(read_verified_ledger(ledger_file).blocks)
    assert (s["total_blocks"], s["events"]) == (4, 3)
    assert s["risk_approvals"] == 2
    assert s["recorded_executions"] == 2
    assert s["event_types"] == {"FORECAST_DECISION": 2, "SCHEDULED_EXIT": 1}
    assert s["actions"] == {"BUY": 1, "HOLD": 1, "SELL": 1}


def test_summary_empty_rejected():
    with pytest.raises(ValueError):
        summarize([])


def test_genesis_excluded_by_default(ledger_file):
    blocks = read_verified_ledger(ledger_file).blocks
    assert len(event_rows(blocks)) == 3
    assert len(event_rows(blocks, include_genesis=True)) == 4


def test_real_event_fields_are_recorded(ledger_file):
    row = event_rows(read_verified_ledger(ledger_file).blocks)[0]
    assert row["Action"] == "BUY"
    assert row["Risk status"] == "APPROVED"
    assert row["Execution status"] == "EXECUTED"


def test_synthetic_missing_fields_not_invented():
    b = {"payload": {"strategy": {}}, "index": 1, "event_type": "NOTE",
         "timestamp_utc": "2024-01-01T00:00:00+00:00", "hash": "abc"}
    row = event_rows([b])[0]
    assert row["Risk status"] is None
    assert row["Action"] is None
    assert row["Execution status"] is None


@pytest.mark.parametrize("idx", [0, 1, 2, 3])
def test_per_block_local_proofs(ledger_file, idx):
    blocks = read_verified_ledger(ledger_file).blocks
    proof = check_one_block(blocks, idx)
    assert proof["ok"] and all(proof["checks"].values())


@pytest.mark.parametrize("idx", [-1, 4, True, "1", 1.0])
def test_bad_block_index_rejected(ledger_file, idx):
    blocks = read_verified_ledger(ledger_file).blocks
    with pytest.raises(IndexError):
        check_one_block(blocks, idx)


def test_selected_block_export(ledger_file):
    s = read_verified_ledger(ledger_file)
    exported = json.loads(export_block_json(s, 1))
    assert exported["block"] == s.blocks[1]
    assert exported["source_file_sha256"] == s.file_sha256


def test_report_export(ledger_file):
    s = read_verified_ledger(ledger_file)
    report = json.loads(export_summary_json(s))
    assert report["summary"]["total_blocks"] == 4
    assert report["source"]["no_external_anchor_checked"] is True


def test_source_metadata_is_clear(ledger_file):
    d = source_metadata(read_verified_ledger(ledger_file))
    assert d["chain_verification"] == "PASSED_USING_PROJECT_BLOCKCHAIN_LEDGER_LOAD_JSON"
    assert d["source_file_sha256_at_read"]
    assert d["no_external_anchor_checked"]


def test_event_filter(ledger_file):
    data = event_rows(read_verified_ledger(ledger_file).blocks)
    assert len(filter_rows(data, event_type="FORECAST_DECISION")) == 2
    assert len(filter_rows(data, action="SELL")) == 1
    assert len(filter_rows(data, risk_status="APPROVED")) == 2
    assert len(filter_rows(data, search="scheduled_exit")) == 1
    assert len(filter_rows(data, search="not-a-match")) == 0


def test_filter_does_not_change_input(ledger_file):
    data = event_rows(read_verified_ledger(ledger_file).blocks)
    before = json.dumps(data)
    filter_rows(data, search="buy")
    assert json.dumps(data) == before


def test_chronology_normal(ledger_file):
    c = chronology_report(read_verified_ledger(ledger_file).blocks)
    assert c == {"timestamps_invalides": 0, "timestamps_répétés": 0, "inversions_chronologiques": 0}


def test_chronology_detects_repeated_and_backward():
    blocks = [
        {"timestamp_utc": "2024-01-02T10:00:00+00:00"},
        {"timestamp_utc": "2024-01-02T09:00:00+00:00"},
        {"timestamp_utc": "2024-01-02T09:00:00Z"},
        {"timestamp_utc": "bad"},
    ]
    c = chronology_report(blocks)
    assert c == {"timestamps_invalides": 1, "timestamps_répétés": 1, "inversions_chronologiques": 1}


@pytest.mark.parametrize("malformed", [b"", b"not json", b"[]", b'{"schema_version":2,"blocks":[]}'])
def test_invalid_json_rejected(tmp_path, malformed):
    p = tmp_path / "ledger.json"
    p.write_bytes(malformed)
    with pytest.raises((ValueError, TypeError, json.JSONDecodeError)):
        read_verified_ledger(p)


@pytest.mark.parametrize("field,value", [
    ("previous_hash", "0"*64),
    ("hash", "bad-hash"),
    ("index", 9),
    ("event_type", "ALTERED"),
])
def test_integrity_tampering_detected(ledger_file, field, value):
    _mutate(ledger_file, lambda d: d["blocks"][2].__setitem__(field, value))
    with pytest.raises(ValueError):
        read_verified_ledger(ledger_file)


def test_payload_tampering_detected(ledger_file):
    _mutate(ledger_file, lambda d: d["blocks"][1]["payload"]["strategy"].__setitem__("action", "SELL"))
    with pytest.raises(ValueError):
        read_verified_ledger(ledger_file)


def test_missing_genesis_detected(ledger_file):
    _mutate(ledger_file, lambda d: d["blocks"].pop(0))
    with pytest.raises(ValueError):
        read_verified_ledger(ledger_file)


def test_schema_replacement_detected(ledger_file):
    _mutate(ledger_file, lambda d: d.__setitem__("schema_version", 2))
    with pytest.raises(ValueError):
        read_verified_ledger(ledger_file)


def test_local_proof_detects_invalid_neighbor(ledger_file):
    blocks = list(read_verified_ledger(ledger_file).blocks)
    blocks[2] = dict(blocks[2], previous_hash="X" * 64)
    assert not check_one_block(blocks, 1)["ok"]


def test_local_proof_detects_invalid_payload(ledger_file):
    blocks = list(read_verified_ledger(ledger_file).blocks)
    blocks[1]["payload"]["strategy"]["action"] = "FRAUD"
    assert not check_one_block(blocks, 1)["ok"]


def test_export_refuses_altered_block(ledger_file):
    s = read_verified_ledger(ledger_file)
    s.blocks[1]["payload"]["strategy"]["action"] = "CHANGED"
    with pytest.raises(ValueError):
        export_block_json(s, 1)


def test_all_blocks_rehashed_can_pass_without_external_anchor(ledger_file):
    """Document limitation: internal hash chain alone is NOT source authenticity."""
    def rewrite(d):
        blocks = d["blocks"]
        blocks[1]["payload"]["strategy"]["action"] = "FORGED"
        for idx in range(1, len(blocks)):
            blocks[idx]["previous_hash"] = blocks[idx-1]["hash"]
            blocks[idx]["hash"] = calculate_hash(blocks[idx])
    _mutate(ledger_file, rewrite)
    # Chain passes: no independent reference hash was supplied. UI explains this.
    assert read_verified_ledger(ledger_file).blocks[1]["payload"]["strategy"]["action"] == "FORGED"
