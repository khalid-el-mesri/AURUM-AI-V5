
import hashlib
import json
import unittest

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from blockchain.ledger import BlockchainLedger


# ============================================================
# AURUM-AI — C.4.3
# BLOCKCHAIN INTEGRATION TESTS
#
# Checks saved blockchain against historical
# Consensus + Trend 60m validation backtest.
#
# Read-only tests.
# No simulation rerun.
# No real trading.
# ============================================================


ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT / "reports" / "backtesting"
    / "combined_holding_30m_60m"
)

LEDGER_DIR = (
    ROOT / "reports" / "blockchain"
    / "consensus_trend_60m_validation"
)


class TestBlockchainIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.source_paths = {
            "decisions": SOURCE / "60m_decisions.csv",
            "transactions": SOURCE / "60m_transactions.csv",
            "summary": SOURCE / "comparison_summary.json",
        }

        cls.decisions = pd.read_csv(
            cls.source_paths["decisions"]
        )

        cls.transactions = pd.read_csv(
            cls.source_paths["transactions"]
        )

        cls.backtest = json.loads(
            cls.source_paths["summary"].read_text(
                encoding="utf-8"
            )
        )

        cls.audit = json.loads(
            (LEDGER_DIR / "audit_summary.json").read_text(
                encoding="utf-8"
            )
        )

        cls.ledger = BlockchainLedger.load_json(
            LEDGER_DIR / "ledger.json"
        )

        cls.blocks = cls.ledger.blocks

        cls.reference = cls.backtest["results"]["60m"]

    # TEST 1
    def test_genesis_and_block_count(self):

        self.assertTrue(self.ledger.verify_integrity())

        self.assertEqual(
            self.blocks[0]["event_type"],
            "GENESIS",
        )

        self.assertEqual(len(self.blocks), 223)

        self.assertEqual(
            len(self.blocks),
            len(self.decisions) + 1,
        )

        self.assertEqual(
            len(self.blocks),
            self.audit["block_count_including_genesis"],
        )

    # TEST 2
    def test_source_file_hashes(self):

        for name, path in self.source_paths.items():

            actual_hash = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()

            expected_hash = self.audit[
                "source_file_sha256"
            ][name]

            self.assertEqual(
                actual_hash,
                expected_hash,
            )

    # TEST 3
    def test_event_provenance(self):

        rows = self.decisions.to_dict("records")

        for number, (row, block) in enumerate(
            zip(rows, self.blocks[1:]),
            start=1,
        ):

            payload = block["payload"]

            self.assertEqual(
                block["index"],
                number,
            )

            self.assertEqual(
                block["event_type"],
                row["event"],
            )

            self.assertEqual(
                payload["historical_timestamp_ms"],
                int(row["timestamp_ms"]),
            )

            self.assertEqual(
                payload["strategy"]["action"],
                row["strategy_action"],
            )

            self.assertEqual(
                payload["execution"]["status"],
                row["execution_status"],
            )

            self.assertEqual(
                payload["provenance"]["source_row_number"],
                number,
            )

    # TEST 4
    def test_historical_timestamps(self):

        for block in self.blocks[1:]:

            historical_ms = block["payload"][
                "historical_timestamp_ms"
            ]

            expected = datetime.fromtimestamp(
                historical_ms / 1000,
                tz=timezone.utc,
            )

            actual = datetime.fromisoformat(
                block["timestamp_utc"]
            )

            self.assertEqual(actual, expected)

            # Processing timestamps must not be
            # confused with historical trading times.
            self.assertNotIn(
                "risk_timestamp_utc",
                block["payload"],
            )

    # TEST 5
    def test_transaction_reconciliation(self):

        executed = [
            block
            for block in self.blocks[1:]
            if block["payload"]["execution"][
                "transaction"
            ] is not None
        ]

        self.assertEqual(
            len(executed),
            len(self.transactions),
        )

        for block, row in zip(
            executed,
            self.transactions.to_dict("records"),
        ):

            transaction = block["payload"][
                "execution"
            ]["transaction"]

            self.assertEqual(
                block["payload"]["historical_timestamp_ms"],
                int(row["timestamp_ms"]),
            )

            self.assertEqual(
                transaction["action"],
                row["action"],
            )

            for field in (
                "price",
                "quantity",
                "fee_usd",
                "cash_after_usd",
                "realized_pnl_usd",
            ):
                self.assertAlmostEqual(
                    transaction[field],
                    float(row[field]),
                    delta=1e-6,
                )

    # TEST 6
    def test_buy_sell_counts(self):

        counts = self.transactions[
            "action"
        ].value_counts()

        self.assertEqual(int(counts["BUY"]), 90)
        self.assertEqual(int(counts["SELL"]), 90)

        self.assertEqual(
            self.reference["completed_trades"],
            90,
        )

        self.assertEqual(
            self.audit["executed_buy_count"],
            90,
        )

        self.assertEqual(
            self.audit["executed_sell_count"],
            90,
        )

    # TEST 7
    def test_portfolio_accounting(self):

        fees = float(
            self.transactions["fee_usd"].sum()
        )

        pnl = float(
            self.transactions["realized_pnl_usd"].sum()
        )

        final_cash = float(
            self.transactions.iloc[-1]["cash_after_usd"]
        )

        self.assertAlmostEqual(
            fees,
            self.reference["total_fees_usd"],
            delta=1e-4,
        )

        self.assertAlmostEqual(
            pnl,
            self.reference["net_profit_usd"],
            delta=1e-4,
        )

        self.assertAlmostEqual(
            final_cash,
            self.reference["final_equity_usd"],
            delta=1e-4,
        )

    # TEST 8
    def test_final_hash_matches_audit(self):

        self.assertEqual(
            self.blocks[-1]["hash"],
            self.audit["final_block_hash"],
        )

        self.assertTrue(
            self.audit["blockchain_integrity_verified"]
        )

    # TEST 9
    def test_external_copy_protection(self):

        copied = self.ledger.blocks

        copied[1]["payload"]["asset"] = "MODIFIED"

        self.assertTrue(
            self.ledger.verify_integrity()
        )

        self.assertEqual(
            self.ledger.blocks[1]["payload"]["asset"],
            "XOMUSD",
        )

    # TEST 10
    def test_tampering_detection(self):

        # Load another instance so the shared
        # test ledger remains unchanged.
        damaged = BlockchainLedger.load_json(
            LEDGER_DIR / "ledger.json"
        )

        damaged._blocks[1]["payload"][
            "asset"
        ] = "MODIFIED"

        self.assertFalse(
            damaged.verify_integrity()
        )

    # TEST 11
    def test_validation_only(self):

        self.assertEqual(
            self.audit["source_split"],
            "validation",
        )

        self.assertFalse(
            self.audit["test_dataset_used"]
        )

        self.assertFalse(
            self.audit["real_trading"]
        )


if __name__ == "__main__":
    unittest.main()
