"""AURUM-AI V5 Blockchain Explorer — read-only view models and integrity audit.

Only reads the existing ledger and source artifacts. Does not append, repair or
rewrite any blockchain record. Uses the project's BlockchainLedger verifier.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEDGER_REL = Path("reports/blockchain/consensus_trend_60m_validation/ledger.json")
AUDIT_REL = Path("reports/blockchain/consensus_trend_60m_validation/audit_summary.json")
SOURCE_REL = {
    "decisions": Path("reports/backtesting/combined_holding_30m_60m/60m_decisions.csv"),
    "transactions": Path("reports/backtesting/combined_holding_30m_60m/60m_transactions.csv"),
    "summary": Path("reports/backtesting/combined_holding_30m_60m/comparison_summary.json"),
}
UNKNOWN = "—"


@dataclass(frozen=True)
class ExplorerSnapshot:
    blocks: tuple[dict[str, Any], ...]
    rows: tuple[dict[str, Any], ...]
    chain_valid: bool
    audit: dict[str, Any]
    audit_present: bool
    final_hash_matches_audit: bool | None
    source_checks: tuple[dict[str, Any], ...]
    ledger_sha256: str
    audit_sha256: str | None


def file_sha256(path: Path) -> str:
    """SHA-256 of original bytes, in bounded-memory chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_text(value: Any) -> str:
    if value is None or isinstance(value, (dict, list, tuple)):
        return UNKNOWN
    result = str(value).strip()
    return result if result else UNKNOWN


def _field(block: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    """Look for a scalar on the block, then in common event payloads."""
    for key in keys:
        if key in block:
            value = _as_text(block[key])
            if value != UNKNOWN:
                return value
    for container in ("payload", "data", "event_data", "metadata", "decision", "details"):
        nested = block.get(container)
        if isinstance(nested, dict):
            for key in keys:
                if key in nested:
                    value = _as_text(nested[key])
                    if value != UNKNOWN:
                        return value
    return UNKNOWN


def format_utc(value: Any) -> str:
    """Format epoch seconds/ms or an ISO timestamp; leave unknown formats intact."""
    if value is None or str(value).strip() == "":
        return UNKNOWN
    try:
        if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
            number = float(value)
            if abs(number) >= 1e11:  # milliseconds, not seconds
                number /= 1000
            return datetime.fromtimestamp(number, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        if isinstance(value, str):
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                # A timezone-less string cannot be safely converted to UTC.
                return value
            return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except (ValueError, OverflowError, OSError, TypeError):
        pass
    return str(value)


def block_rows(blocks: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, block in enumerate(blocks):
        if not isinstance(block, Mapping):
            raise ValueError(f"Bloc {position} invalide : objet JSON attendu")
        index = block.get("index", position)
        event = _field(block, ("event_type", "event", "type", "kind"))
        if position == 0 and event == UNKNOWN:
            event = "GENESIS"
        timestamp = _field(block, ("timestamp", "timestamp_utc", "created_at", "event_timestamp"))
        row = {
            "position": position,
            "index": index,
            "time_utc": format_utc(timestamp if timestamp != UNKNOWN else None),
            "event": event,
            "action": _field(block, ("action", "strategy_action", "decision_action")),
            "risk": _field(block, ("risk_status", "risk_decision", "risk_state")),
            "execution": _field(block, ("execution_status", "execution_state")),
            "kill_switch": _field(block, ("kill_switch", "kill_switch_state", "kill_switch_status")),
            "hash": _field(block, ("hash",)),
            "previous_hash": _field(block, ("previous_hash", "prev_hash")),
        }
        # Search within both metadata and original payload without discarding fields.
        row["search_text"] = json.dumps(block, ensure_ascii=False, sort_keys=True, default=str).casefold()
        rows.append(row)
    return rows


def filter_block_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    query: str = "",
    event: str | None = None,
    action: str | None = None,
    risk: str | None = None,
    execution: str | None = None,
    include_genesis: bool = True,
) -> list[dict[str, Any]]:
    terms = query.casefold().strip().split()
    result: list[dict[str, Any]] = []
    for row in rows:
        if not include_genesis and row["position"] == 0:
            continue
        if event and row["event"] != event:
            continue
        if action and row["action"] != action:
            continue
        if risk and row["risk"] != risk:
            continue
        if execution and row["execution"] != execution:
            continue
        haystack = row.get("search_text", "").casefold() + " " + " ".join(
            str(row.get(k, "")) for k in ("position", "index", "time_utc")
        ).casefold()
        if not all(term in haystack for term in terms):
            continue
        result.append(dict(row))
    return result


def previous_link_status(blocks: tuple[dict[str, Any], ...] | list[dict[str, Any]], position: int) -> bool | None:
    """Checks only a block's previous-hash pointer; NOT its own SHA-256."""
    if position < 0 or position >= len(blocks):
        raise IndexError(position)
    if position == 0:
        return None
    current = blocks[position]
    previous = blocks[position - 1]
    pointer = current.get("previous_hash", current.get("prev_hash"))
    previous_hash = previous.get("hash")
    if not isinstance(pointer, str) or not isinstance(previous_hash, str):
        return None
    return pointer == previous_hash


def check_source_hashes(root: Path, manifest: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Inspect source files against declared hashes, without modifying them."""
    hashes = manifest if isinstance(manifest, Mapping) else {}
    results: list[dict[str, Any]] = []
    for name, relative in SOURCE_REL.items():
        path = root / relative
        expected = hashes.get(name)
        exists = path.is_file()
        observed = file_sha256(path) if exists else None
        if not exists:
            status = "FICHIER ABSENT"
        elif not isinstance(expected, str) or not expected.strip():
            status = "EMPREINTE NON FOURNIE"
        elif observed == expected.lower():
            status = "CONFORME"
        else:
            status = "DIFFÉRENT"
        results.append({
            "source": name,
            "path": str(relative),
            "status": status,
            "expected_sha256": expected,
            "observed_sha256": observed,
        })
    return results


def load_snapshot(root: Path = PROJECT_ROOT) -> ExplorerSnapshot:
    """Load the existing verified engine; never create or save a block."""
    from blockchain.ledger import BlockchainLedger

    root = Path(root)
    ledger_path = root / LEDGER_REL
    audit_path = root / AUDIT_REL
    if not ledger_path.is_file():
        raise FileNotFoundError(f"Ledger introuvable : {ledger_path}")
    ledger = BlockchainLedger.load_json(ledger_path)
    blocks = tuple(dict(block) for block in ledger.blocks)
    if not blocks:
        raise ValueError("Ledger vide : aucun bloc GENESIS disponible")
    valid = bool(ledger.verify_integrity())
    audit_present = audit_path.is_file()
    if audit_present:
        with audit_path.open("r", encoding="utf-8") as source:
            audit = json.load(source)
        if not isinstance(audit, dict):
            raise ValueError("audit_summary.json doit contenir un objet JSON")
    else:
        audit = {}
    expected_final_hash = audit.get("final_block_hash")
    final_matches = (
        blocks[-1].get("hash") == expected_final_hash
        if isinstance(expected_final_hash, str) else None
    )
    checks = check_source_hashes(root, audit.get("source_file_sha256"))
    return ExplorerSnapshot(
        blocks=blocks,
        rows=tuple(block_rows(blocks)),
        chain_valid=valid,
        audit=audit,
        audit_present=audit_present,
        final_hash_matches_audit=final_matches,
        source_checks=tuple(checks),
        ledger_sha256=file_sha256(ledger_path),
        audit_sha256=file_sha256(audit_path) if audit_present else None,
    )


def event_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    return dict(Counter(row["event"] for row in rows if row["position"] != 0))


def export_rows_json(rows: Iterable[Mapping[str, Any]], blocks: Iterable[dict[str, Any]]) -> str:
    """Export selected original blocks, not a reconstructed/modified ledger."""
    original = list(blocks)
    selected = [original[row["position"]] for row in rows]
    return json.dumps(selected, ensure_ascii=False, indent=2, default=str)
