
import copy
import json
import tempfile
import unittest

from pathlib import Path

from blockchain.ledger import (
    BlockchainLedger,
    calculate_hash,
)


# ============================================================
# AURUM-AI — C.3
# BLOCKCHAIN LEDGER UNIT TESTS
#
# Synthetic data only.
# No real trading.
# No modification of existing project artifacts.
# ============================================================


class TestBlockchainLedger(unittest.TestCase):

    def setUp(self):
        self.ledger = BlockchainLedger()

    def add_sample_event(self):
        return self.ledger.append_event(
            event_type="INVESTMENT_DECISION",
            timestamp_utc="2024-06-03T14:00:00+00:00",
            payload={
                "asset": "XOMUSD",
                "prediction": "UP",
                "strategy_signal": "BUY",
                "risk_status": "APPROVED",
                "quantity": 2.0,
                "execution": "SIMULATED",
            },
        )

    # TEST 1
    def test_genesis_block(self):
        blocks = self.ledger.blocks

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["index"], 0)
        self.assertEqual(
            blocks[0]["event_type"], "GENESIS"
        )
        self.assertTrue(
            self.ledger.verify_integrity()
        )

    # TEST 2
    def test_append_event(self):
        block = self.add_sample_event()

        self.assertEqual(len(self.ledger.blocks), 2)
        self.assertEqual(block["index"], 1)
        self.assertEqual(
            block["event_type"],
            "INVESTMENT_DECISION",
        )

    # TEST 3
    def test_hash_chain_link(self):
        self.add_sample_event()

        blocks = self.ledger.blocks

        self.assertEqual(
            blocks[1]["previous_hash"],
            blocks[0]["hash"],
        )

        self.assertTrue(
            self.ledger.verify_integrity()
        )

    # TEST 4
    def test_hash_recalculation(self):
        block = self.add_sample_event()

        self.assertEqual(
            block["hash"],
            calculate_hash(block),
        )

    # TEST 5
    def test_multiple_events(self):
        self.add_sample_event()

        self.ledger.append_event(
            event_type="RISK_DECISION",
            payload={
                "status": "APPROVED",
                "simulation": True,
            },
        )

        self.assertEqual(
            len(self.ledger.blocks), 3
        )

        self.assertTrue(
            self.ledger.verify_integrity()
        )

    # TEST 6
    def test_external_copy_protection(self):
        self.add_sample_event()

        blocks = self.ledger.blocks

        blocks[1]["payload"]["quantity"] = 999

        self.assertEqual(
            self.ledger.blocks[1]["payload"]["quantity"],
            2.0,
        )

        self.assertTrue(
            self.ledger.verify_integrity()
        )

    # TEST 7
    def test_tampering_detection(self):
        self.add_sample_event()

        self.ledger._blocks[1]["payload"][
            "quantity"
        ] = 999

        self.assertFalse(
            self.ledger.verify_integrity()
        )

    # TEST 8
    def test_broken_previous_hash(self):
        self.add_sample_event()

        self.ledger._blocks[1][
            "previous_hash"
        ] = "invalid_hash"

        self.assertFalse(
            self.ledger.verify_integrity()
        )

    # TEST 9
    def test_invalid_event_type(self):
        with self.assertRaises(ValueError):
            self.ledger.append_event(
                event_type="",
                payload={"test": True},
            )

        with self.assertRaises(ValueError):
            self.ledger.append_event(
                event_type="GENESIS",
                payload={"test": True},
            )

    # TEST 10
    def test_invalid_payload(self):
        with self.assertRaises(TypeError):
            self.ledger.append_event(
                event_type="TEST",
                payload="invalid",
            )

        with self.assertRaises(ValueError):
            self.ledger.append_event(
                event_type="TEST",
                payload={"value": float("nan")},
            )

    # TEST 11
    def test_json_save_and_reload(self):
        self.add_sample_event()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.json"

            self.ledger.save_json(path)

            restored = BlockchainLedger.load_json(
                path
            )

            self.assertEqual(
                restored.blocks,
                self.ledger.blocks,
            )

            self.assertTrue(
                restored.verify_integrity()
            )

    # TEST 12
    def test_existing_file_not_overwritten(self):
        self.add_sample_event()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.json"

            self.ledger.save_json(path)

            with self.assertRaises(FileExistsError):
                self.ledger.save_json(path)

    # TEST 13
    def test_corrupted_file_rejected(self):
        self.add_sample_event()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corrupt.json"

            corrupted = {
                "schema_version": 1,
                "blocks": copy.deepcopy(
                    self.ledger.blocks
                ),
            }

            corrupted["blocks"][1]["payload"][
                "quantity"
            ] = 999

            path.write_text(
                json.dumps(corrupted),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                BlockchainLedger.load_json(path)

    # TEST 14
    def test_append_rejected_on_corrupt_chain(self):
        self.add_sample_event()

        self.ledger._blocks[1]["payload"][
            "quantity"
        ] = 999

        with self.assertRaises(RuntimeError):
            self.ledger.append_event(
                event_type="EXECUTION",
                payload={"status": "SIMULATED"},
            )


if __name__ == "__main__":
    unittest.main()
