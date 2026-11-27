"""Pure tests for the V5 Explorer; never mutate the project's historical ledger."""

import hashlib
import json

import pytest

from frontend.services.blockchain_explorer_audit import (
    SOURCE_REL, block_rows, check_source_hashes, event_counts,
    export_rows_json, file_sha256, filter_block_rows, format_utc,
    previous_link_status,
)


@pytest.fixture
def example_blocks():
    return [
        {"index": 0, "event": "GENESIS", "timestamp": "2024-01-01T00:00:00Z", "hash": "root"},
        {"index": 1, "event_type": "FORECAST_DECISION", "timestamp": 1704067500000,
         "data": {"action": "BUY", "risk_status": "APPROVED"},
         "execution_status": "EXECUTED", "previous_hash": "root", "hash": "hash-1"},
        {"index": 2, "event_type": "SCHEDULED_EXIT", "timestamp": "2024-01-01T00:10:00Z",
         "payload": {"action": "SELL", "risk_status": "APPROVED"},
         "previous_hash": "hash-1", "hash": "hash-2"},
    ]


def test_rows_count(example_blocks):
    assert len(block_rows(example_blocks)) == 3


def test_genesis(example_blocks):
    assert block_rows(example_blocks)[0]["event"] == "GENESIS"


def test_nested_action(example_blocks):
    assert block_rows(example_blocks)[1]["action"] == "BUY"


def test_nested_payload(example_blocks):
    assert block_rows(example_blocks)[2]["action"] == "SELL"


def test_filter_event(example_blocks):
    rows = block_rows(example_blocks)
    assert len(filter_block_rows(rows, event="SCHEDULED_EXIT")) == 1


def test_filter_multiple_constraints(example_blocks):
    rows = block_rows(example_blocks)
    assert [r["index"] for r in filter_block_rows(rows, action="BUY", risk="APPROVED")] == [1]


def test_filter_hash_case_insensitive(example_blocks):
    rows = block_rows(example_blocks)
    assert [r["index"] for r in filter_block_rows(rows, query="HASH-2")] == [2]


def test_filter_multiple_words(example_blocks):
    rows = block_rows(example_blocks)
    assert [r["index"] for r in filter_block_rows(rows, query="FORECAST BUY")] == [1]


def test_filter_without_genesis(example_blocks):
    assert len(filter_block_rows(block_rows(example_blocks), include_genesis=False)) == 2


def test_count_without_genesis(example_blocks):
    assert event_counts(block_rows(example_blocks)) == {"FORECAST_DECISION": 1, "SCHEDULED_EXIT": 1}


def test_previous_hash_match(example_blocks):
    assert previous_link_status(example_blocks, 2) is True


def test_previous_hash_mismatch(example_blocks):
    example_blocks[2]["previous_hash"] = "altered"
    assert previous_link_status(example_blocks, 2) is False


def test_no_previous_link_on_genesis(example_blocks):
    assert previous_link_status(example_blocks, 0) is None


def test_previous_link_missing_field(example_blocks):
    del example_blocks[1]["previous_hash"]
    assert previous_link_status(example_blocks, 1) is None


def test_export_original_selected(example_blocks):
    selected = filter_block_rows(block_rows(example_blocks), action="BUY")
    assert json.loads(export_rows_json(selected, example_blocks)) == [example_blocks[1]]


def test_export_does_not_mutate(example_blocks):
    copy = json.dumps(example_blocks, sort_keys=True)
    export_rows_json(block_rows(example_blocks), example_blocks)
    assert json.dumps(example_blocks, sort_keys=True) == copy


def test_utc_epoch_milliseconds():
    assert format_utc(1704067200000) == "2024-01-01 00:00:00 UTC"


def test_utc_timezone_conversion():
    assert format_utc("2024-01-01T01:00:00+01:00") == "2024-01-01 00:00:00 UTC"


def test_no_assumption_for_naive_datetime():
    assert format_utc("2024-01-01T12:00:00") == "2024-01-01T12:00:00"


def test_reject_non_mapping_block():
    with pytest.raises(ValueError):
        block_rows([123])


def test_file_sha256(tmp_path):
    path = tmp_path / "example"
    path.write_bytes(b"audit")
    assert file_sha256(path) == hashlib.sha256(b"audit").hexdigest()


def test_source_checks_matching_hashes(tmp_path):
    for rel in SOURCE_REL.values():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unchanged")
    expected = hashlib.sha256(b"unchanged").hexdigest()
    checks = check_source_hashes(tmp_path, {name: expected for name in SOURCE_REL})
    assert all(row["status"] == "CONFORME" for row in checks)


def test_source_checks_missing(tmp_path):
    checks = check_source_hashes(tmp_path, {})
    assert all(row["status"] == "FICHIER ABSENT" for row in checks)


def test_source_checks_mismatch(tmp_path):
    for rel in SOURCE_REL.values():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unexpected")
    checks = check_source_hashes(tmp_path, {name: "0" * 64 for name in SOURCE_REL})
    assert all(row["status"] == "DIFFÉRENT" for row in checks)
