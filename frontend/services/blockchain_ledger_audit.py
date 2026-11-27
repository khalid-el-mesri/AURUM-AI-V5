"""AURUM-AI V5 Phase 13 — deterministic, read-only ledger audit.

Source of truth: project's existing blockchain.ledger.BlockchainLedger.
Never append, persist or repair records. A SHA-256 linked file is tamper-evident,
not externally notarized or decentralized; history may be rewritten and rehashed.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

LEDGER_RELATIVE_PATH = Path("reports/blockchain/consensus_trend_60m_validation/ledger.json")


@dataclass(frozen=True)
class VerifiedLedger:
    blocks: tuple[dict, ...]
    file_sha256: str
    size_bytes: int
    source_path: str


def _obj(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _stage(block: dict, name: str) -> dict:
    return _obj(_obj(block.get("payload")).get(name))


def read_verified_ledger(path: str | Path) -> VerifiedLedger:
    """Require original loader and identical bytes across loading; fail closed."""
    from blockchain.ledger import BlockchainLedger

    source = Path(path)
    before = source.read_bytes()
    if not before:
        raise ValueError("Le fichier ledger est vide.")
    source_hash = sha256(before).hexdigest()
    # Explicit UTF-8 and schema check; project loader does full chain verification.
    raw = json.loads(before.decode("utf-8"))
    ledger = BlockchainLedger.load_json(source)
    after = source.read_bytes()
    if before != after:
        raise ValueError("Le ledger a été modifié pendant la lecture.")
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("Version de schéma du ledger non reconnue.")
    blocks = ledger.blocks
    if blocks != raw.get("blocks"):
        raise ValueError("Incohérence entre le contenu lu et le contenu vérifié.")
    if not blocks:
        raise ValueError("La chaîne vérifiée est vide.")
    return VerifiedLedger(tuple(blocks), source_hash, len(before), str(source))


def event_action(block: dict) -> str | None:
    val = _stage(block, "strategy").get("action")
    return str(val) if val is not None and str(val).strip() else None


def event_rows(blocks: tuple[dict, ...] | list[dict], include_genesis: bool = False) -> list[dict]:
    result = []
    for block in blocks:
        if block.get("event_type") == "GENESIS" and not include_genesis:
            continue
        result.append({
            "Bloc": block["index"],
            "Horodatage UTC": block["timestamp_utc"],
            "Événement": block["event_type"],
            "Action": event_action(block),
            "Risk status": _stage(block, "risk").get("status"),
            "Execution status": _stage(block, "execution").get("status"),
            "Kill Switch": _stage(block, "risk").get("system_state"),
            "Hash": block["hash"],
        })
    return result


def summarize(blocks: tuple[dict, ...] | list[dict]) -> dict:
    if not blocks:
        raise ValueError("Impossible de résumer une chaîne vide.")
    events = [b for b in blocks if b["event_type"] != "GENESIS"]
    event_types = Counter(str(b["event_type"]) for b in events)
    actions = Counter(event_action(b) or "Non renseigné" for b in events)
    return {
        "total_blocks": len(blocks),
        "events": len(events),
        "first_hash": blocks[0]["hash"],
        "head_hash": blocks[-1]["hash"],
        "first_event_utc": events[0]["timestamp_utc"] if events else None,
        "last_event_utc": events[-1]["timestamp_utc"] if events else None,
        "event_types": dict(sorted(event_types.items())),
        "actions": dict(sorted(actions.items())),
        "risk_approvals": sum(_stage(b, "risk").get("status") == "APPROVED" for b in events),
        "recorded_executions": sum(_stage(b, "execution").get("status") == "EXECUTED" for b in events),
    }


def check_one_block(blocks: tuple[dict, ...] | list[dict], index: int) -> dict:
    """Verification for display is supplementary to *full chain* load verification."""
    from blockchain.ledger import calculate_hash, GENESIS_PREVIOUS_HASH

    if type(index) is not int or not (0 <= index < len(blocks)):
        raise IndexError("Indice de bloc inexistant.")
    block = blocks[index]
    actual = calculate_hash(block)
    previous = GENESIS_PREVIOUS_HASH if index == 0 else blocks[index - 1]["hash"]
    next_hash_link = None if index + 1 == len(blocks) else blocks[index + 1]["previous_hash"]
    checks = {
        "index_conforme": type(block["index"]) is int and block["index"] == index,
        "hash_bloc_conforme": block["hash"] == actual,
        "previous_hash_conforme": block["previous_hash"] == previous,
        "lien_bloc_suivant_conforme": next_hash_link is None or next_hash_link == block["hash"],
    }
    return {
        "index": index,
        "checks": checks,
        "ok": all(checks.values()),
        "hash_stocke": block["hash"],
        "hash_recalcule": actual,
        "previous_hash": block["previous_hash"],
        "expected_previous_hash": previous,
        "following_index": index + 1 if next_hash_link is not None else None,
    }


def chronology_report(blocks: tuple[dict, ...] | list[dict]) -> dict:
    """Quality advisory, not part of cryptographic integrity."""
    previous = None
    seen = set()
    invalid = repeats = inversions = 0
    for block in blocks:
        raw = block.get("timestamp_utc")
        try:
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                raise ValueError("naive")
            dt = dt.astimezone(timezone.utc)
        except (ValueError, TypeError, OverflowError):
            invalid += 1
            continue
        if dt in seen:
            repeats += 1
        seen.add(dt)
        if previous is not None and dt < previous:
            inversions += 1
        previous = dt
    return {
        "timestamps_invalides": invalid,
        "timestamps_répétés": repeats,
        "inversions_chronologiques": inversions,
    }


def source_metadata(snapshot: VerifiedLedger) -> dict:
    summary = summarize(snapshot.blocks)
    return {
        "project": "AURUM-AI V5",
        "audit_mode": "READ_ONLY",
        "source": snapshot.source_path,
        "source_size_bytes": snapshot.size_bytes,
        "source_file_sha256_at_read": snapshot.file_sha256,
        "schema_version": 1,
        "block_count_verified": summary["total_blocks"],
        "historical_events": summary["events"],
        "genesis_hash": summary["first_hash"],
        "last_block_hash": summary["head_hash"],
        "chain_verification": "PASSED_USING_PROJECT_BLOCKCHAIN_LEDGER_LOAD_JSON",
        "no_external_anchor_checked": True,
        "limitations": [
            "Empreintes et chaînage détectent les altérations incohérentes, pas une réécriture intégralement rehashée.",
            "L'horodatage et la provenance de marché ne sont pas certifiés par une autorité externe.",
            "Pas de blockchain publique, minage, transactions réelles ou preuve de rendement futur.",
        ],
    }


def export_summary_json(snapshot: VerifiedLedger) -> bytes:
    return json.dumps({"source": source_metadata(snapshot), "summary": summarize(snapshot.blocks),
                       "chronology": chronology_report(snapshot.blocks)}, indent=2,
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def export_block_json(snapshot: VerifiedLedger, index: int) -> bytes:
    checked = check_one_block(snapshot.blocks, index)
    if not checked["ok"]:
        raise ValueError("Bloc invalide, export refusé.")
    return json.dumps({"source_file_sha256": snapshot.file_sha256,
                       "validation": "LOCAL_CHAIN_VALIDATED", "block": snapshot.blocks[index]},
                      ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def filter_rows(rows: list[dict], event_type: str | None = None,
                action: str | None = None, risk_status: str | None = None,
                search: str = "") -> list[dict]:
    """Pure filter; no data invention or mutating the ledger."""
    search = search.strip().casefold()
    output = []
    for row in rows:
        if event_type and row["Événement"] != event_type:
            continue
        if action and row["Action"] != action:
            continue
        if risk_status and row["Risk status"] != risk_status:
            continue
        if search and not any(search in str(value).casefold() for value in row.values() if value is not None):
            continue
        output.append(row)
    return output
