
import copy
import hashlib
import json
import tempfile

from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# AURUM-AI — C.2
# LOCAL BLOCKCHAIN LEDGER
#
# SHA-256 hash chaining
# JSON persistence
# Integrity verification
#
# This is a simulated local audit ledger.
# No cryptocurrency, no mining, no real trading.
# ============================================================


GENESIS_PREVIOUS_HASH = "0" * 64
GENESIS_TIMESTAMP = "1970-01-01T00:00:00+00:00"

BLOCK_FIELDS = (
    "index",
    "timestamp_utc",
    "event_type",
    "payload",
    "previous_hash",
)


def canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def calculate_hash(block):
    body = {
        field: block[field]
        for field in BLOCK_FIELDS
    }

    encoded = canonical_json(body).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def normalize_timestamp(timestamp):

    if timestamp is None:
        return datetime.now(timezone.utc).isoformat()

    if not isinstance(timestamp, str):
        raise TypeError("Timestamp must be a string.")

    parsed = datetime.fromisoformat(
        timestamp.replace("Z", "+00:00")
    )

    if parsed.tzinfo is None:
        raise ValueError(
            "Timestamp must contain timezone information."
        )

    return parsed.astimezone(timezone.utc).isoformat()


class BlockchainLedger:

    def __init__(self):

        genesis = self._create_block(
            index=0,
            timestamp_utc=GENESIS_TIMESTAMP,
            event_type="GENESIS",
            payload={
                "project": "AURUM-AI",
                "ledger_version": 1,
            },
            previous_hash=GENESIS_PREVIOUS_HASH,
        )

        self._blocks = [genesis]

    @staticmethod
    def _create_block(
        index,
        timestamp_utc,
        event_type,
        payload,
        previous_hash,
    ):

        body = {
            "index": index,
            "timestamp_utc": timestamp_utc,
            "event_type": event_type,
            "payload": copy.deepcopy(payload),
            "previous_hash": previous_hash,
        }

        body["hash"] = calculate_hash(body)

        return body

    @property
    def blocks(self):

        # Return copies to prevent accidental modification.
        return copy.deepcopy(self._blocks)

    def append_event(
        self,
        event_type,
        payload,
        timestamp_utc=None,
    ):

        if not self.verify_integrity():
            raise RuntimeError(
                "Cannot append to an invalid blockchain."
            )

        if (
            not isinstance(event_type, str)
            or not event_type.strip()
            or event_type == "GENESIS"
        ):
            raise ValueError("Invalid event type.")

        if not isinstance(payload, dict):
            raise TypeError("Payload must be a dictionary.")

        # Verify that the payload can be serialized
        # deterministically without NaN or Infinity.
        canonical_json(payload)

        normalized_time = normalize_timestamp(
            timestamp_utc
        )

        previous_block = self._blocks[-1]

        block = self._create_block(
            index=len(self._blocks),
            timestamp_utc=normalized_time,
            event_type=event_type,
            payload=payload,
            previous_hash=previous_block["hash"],
        )

        self._blocks.append(block)

        return copy.deepcopy(block)

    def verify_integrity(self):

        if not self._blocks:
            return False

        expected_genesis = BlockchainLedger()._blocks[0] \
            if len(self._blocks) > 1 else None

        if expected_genesis is None:
            expected_genesis = self._create_block(
                index=0,
                timestamp_utc=GENESIS_TIMESTAMP,
                event_type="GENESIS",
                payload={
                    "project": "AURUM-AI",
                    "ledger_version": 1,
                },
                previous_hash=GENESIS_PREVIOUS_HASH,
            )

        if self._blocks[0] != expected_genesis:
            return False

        previous_hash = GENESIS_PREVIOUS_HASH

        for index, block in enumerate(self._blocks):

            try:
                if set(block) != set(BLOCK_FIELDS) | {"hash"}:
                    return False

                if type(block["index"]) is not int:
                    return False

                if block["index"] != index:
                    return False

                if block["previous_hash"] != previous_hash:
                    return False

                if index > 0 and block["event_type"] == "GENESIS":
                    return False

                if block["hash"] != calculate_hash(block):
                    return False

                previous_hash = block["hash"]

            except (KeyError, TypeError, ValueError):
                return False

        return True

    def save_json(self, path):

        if not self.verify_integrity():
            raise RuntimeError(
                "Cannot save an invalid blockchain."
            )

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # "x" prevents overwriting an existing ledger.
        with path.open("x", encoding="utf-8") as file:
            json.dump(
                {
                    "schema_version": 1,
                    "blocks": self._blocks,
                },
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

    @classmethod
    def load_json(cls, path):

        with Path(path).open(
            "r", encoding="utf-8"
        ) as file:
            data = json.load(file)

        if (
            not isinstance(data, dict)
            or data.get("schema_version") != 1
            or not isinstance(data.get("blocks"), list)
        ):
            raise ValueError("Invalid ledger file format.")

        ledger = cls()
        ledger._blocks = copy.deepcopy(data["blocks"])

        if not ledger.verify_integrity():
            raise ValueError(
                "Blockchain integrity verification failed."
            )

        return ledger


def main():

    print("\n=== AURUM-AI | BLOCKCHAIN LEDGER TEST ===")

    ledger = BlockchainLedger()

    print("Genesis created: PASSED")

    # Synthetic example only.
    ledger.append_event(
        event_type="INVESTMENT_DECISION",
        timestamp_utc="2024-06-03T14:00:00+00:00",
        payload={
            "asset": "XOMUSD",
            "prediction": "UP",
            "strategy_signal": "BUY",
            "risk_status": "APPROVED",
            "quantity": 2.0,
            "capital_before_usd": 10000.0,
            "capital_after_usd": 9900.0,
            "agent": "ExecutionAgent",
            "execution": "SIMULATED",
        },
    )

    assert len(ledger.blocks) == 2
    assert ledger.verify_integrity()

    print("Investment block appended: PASSED")
    print("SHA-256 chain verification: PASSED")

    # Test JSON persistence in a temporary directory.
    with tempfile.TemporaryDirectory() as tmp:

        path = Path(tmp) / "test_ledger.json"

        ledger.save_json(path)

        restored = BlockchainLedger.load_json(path)

        assert restored.blocks == ledger.blocks
        assert restored.verify_integrity()

        print("JSON save and reload: PASSED")

    # External copies must not alter internal blocks.
    copied_blocks = ledger.blocks

    copied_blocks[1]["payload"]["quantity"] = 999.0

    assert ledger.verify_integrity()

    print("External copy protection: PASSED")

    # Deliberately corrupt an internal block to test
    # detection. This affects the test instance only.
    ledger._blocks[1]["payload"]["quantity"] = 999.0

    assert not ledger.verify_integrity()

    print("Tampering detection: PASSED")

    print("\nAll blockchain tests PASSED.")
    print("Synthetic data only.")
    print("No project dataset modified.")
    print("No real trading.")

    print("\nC.2 BLOCKCHAIN LEDGER COMPLETED")


if __name__ == "__main__":
    main()
