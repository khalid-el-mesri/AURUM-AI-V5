
from collections import defaultdict, deque
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd

from blockchain.ledger import BlockchainLedger


# ============================================================
# AURUM-AI — C.4.2
# HISTORICAL AGENT DECISIONS -> BLOCKCHAIN LEDGER
#
# Source: Consensus + Trend 60m
# Dataset: XOM
# Evaluation split: VALIDATION
#
# Read-only access to existing backtest artifacts.
# No simulation rerun.
# No real trading.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = (
    ROOT / "reports" / "backtesting"
    / "combined_holding_30m_60m"
)

DECISIONS_PATH = SOURCE_DIR / "60m_decisions.csv"
TRANSACTIONS_PATH = SOURCE_DIR / "60m_transactions.csv"
BACKTEST_PATH = SOURCE_DIR / "comparison_summary.json"

OUTPUT_DIR = (
    ROOT / "reports" / "blockchain"
    / "consensus_trend_60m_validation"
)

LEDGER_PATH = OUTPUT_DIR / "ledger.json"
AUDIT_PATH = OUTPUT_DIR / "audit_summary.json"


def file_sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)

    return digest.hexdigest()


def clean_value(value):
    """Convert pandas and NumPy scalars into JSON-safe values."""

    if value is None:
        return None

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return None
        return float(value)

    if pd.isna(value):
        return None

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    return str(value)


def optional_timestamp(value):
    if pd.isna(value):
        return None
    return int(value)


def assert_close(actual, expected, label, atol=1e-4):
    if not np.isclose(
        float(actual),
        float(expected),
        rtol=0,
        atol=atol,
    ):
        raise RuntimeError(
            f"{label} mismatch: {actual} != {expected}"
        )


def load_historical_artifacts():

    for path in (
        DECISIONS_PATH,
        TRANSACTIONS_PATH,
        BACKTEST_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    decisions = pd.read_csv(DECISIONS_PATH)
    transactions = pd.read_csv(TRANSACTIONS_PATH)

    report = json.loads(
        BACKTEST_PATH.read_text(encoding="utf-8")
    )

    required_decision_columns = {
        "timestamp_ms",
        "session_date",
        "predicted_class",
        "strategy_action",
        "strategy_reason",
        "risk_status",
        "risk_reason",
        "execution_status",
        "execution_reason",
        "quantity",
        "cash_after_usd",
        "equity_after_usd",
        "realized_pnl_usd",
        "model",
        "event",
        "signal_timestamp_ms",
        "system_state",
    }

    required_transaction_columns = {
        "timestamp_ms",
        "action",
        "price",
        "quantity",
        "fee_usd",
        "cash_change_usd",
        "realized_pnl_usd",
        "cash_after_usd",
        "holdings_after",
    }

    if not required_decision_columns.issubset(decisions.columns):
        missing = required_decision_columns - set(decisions.columns)
        raise RuntimeError(f"Missing decision columns: {missing}")

    if not required_transaction_columns.issubset(
        transactions.columns
    ):
        missing = (
            required_transaction_columns
            - set(transactions.columns)
        )
        raise RuntimeError(
            f"Missing transaction columns: {missing}"
        )

    if decisions.empty or transactions.empty:
        raise RuntimeError("Empty historical artifacts.")

    # Validate chronology before importing anything.
    if not decisions["timestamp_ms"].is_monotonic_increasing:
        raise RuntimeError("Decisions are not chronological.")

    if not transactions["timestamp_ms"].is_monotonic_increasing:
        raise RuntimeError("Transactions are not chronological.")

    if report.get("evaluation_split") != "validation":
        raise RuntimeError("Unexpected evaluation split.")

    reference = report["results"]["60m"]

    return decisions, transactions, reference


def build_transaction_index(transactions):

    index = defaultdict(deque)

    for row in transactions.to_dict("records"):

        action = str(row["action"])
        timestamp_ms = int(row["timestamp_ms"])

        if action not in ("BUY", "SELL"):
            raise RuntimeError(
                f"Unexpected transaction action: {action}"
            )

        key = (timestamp_ms, action)
        index[key].append(row)

    return index


def create_payload(row, transaction, source_row_number):

    timestamp_ms = int(row["timestamp_ms"])

    execution_details = {
        "status": clean_value(row["execution_status"]),
        "reason": clean_value(row["execution_reason"]),
        "transaction": None,
    }

    if transaction is not None:

        execution_details["transaction"] = {
            "action": str(transaction["action"]),
            "price": float(transaction["price"]),
            "quantity": float(transaction["quantity"]),
            "fee_usd": float(transaction["fee_usd"]),
            "cash_change_usd": float(
                transaction["cash_change_usd"]
            ),
            "realized_pnl_usd": float(
                transaction["realized_pnl_usd"]
            ),
            "cash_after_usd": float(
                transaction["cash_after_usd"]
            ),
            "holdings_after": float(
                transaction["holdings_after"]
            ),
        }

    return {
        "asset": "XOMUSD",
        "source": "AURUM-AI historical backtest",
        "simulation": True,
        "evaluation_split": "validation",
        "historical_timestamp_ms": timestamp_ms,
        "session_date": str(row["session_date"]),
        "model": clean_value(row["model"]),
        "forecast": {
            "signal_timestamp_ms": optional_timestamp(
                row["signal_timestamp_ms"]
            ),
            "predicted_class": clean_value(
                row["predicted_class"]
            ),
        },
        "strategy": {
            "action": clean_value(row["strategy_action"]),
            "reason": clean_value(row["strategy_reason"]),
        },
        "risk": {
            "status": clean_value(row["risk_status"]),
            "reason": clean_value(row["risk_reason"]),
            "system_state": clean_value(row["system_state"]),
        },
        "execution": execution_details,
        "portfolio": {
            "cash_after_usd": clean_value(
                row["cash_after_usd"]
            ),
            "equity_after_usd": clean_value(
                row["equity_after_usd"]
            ),
            "realized_pnl_usd": clean_value(
                row["realized_pnl_usd"]
            ),
        },
        "provenance": {
            "source": "60m_decisions.csv",
            "source_row_number": source_row_number,
            "event": clean_value(row["event"]),
            "risk_timestamp_note": (
                "Processing timestamp intentionally excluded; "
                "block uses historical market-event timestamp."
            ),
        },
    }


def main():

    print("\n=== AURUM-AI | C.4.2 BLOCKCHAIN INTEGRATION ===")

    # Never overwrite earlier artifacts.
    for path in (LEDGER_PATH, AUDIT_PATH):
        if path.exists():
            raise FileExistsError(
                f"Existing artifact preserved: {path}"
            )

    decisions, transactions, reference = (
        load_historical_artifacts()
    )

    print("\n=== SOURCE AUDIT ===")
    print("Decisions:", len(decisions))
    print("Transactions:", len(transactions))

    transaction_index = build_transaction_index(
        transactions
    )

    ledger = BlockchainLedger()

    executed_decisions = 0
    hold_or_other_decisions = 0
    buy_count = 0
    sell_count = 0

    print("\n=== BUILDING BLOCKCHAIN ===")

    for source_row_number, row in enumerate(
        decisions.to_dict("records"),
        start=1,
    ):

        timestamp_ms = int(row["timestamp_ms"])

        action = str(row["strategy_action"])
        status = str(row["execution_status"])

        transaction = None

        if status == "EXECUTED":

            if action not in ("BUY", "SELL"):
                raise RuntimeError(
                    f"Unexpected executed action: {action}"
                )

            key = (timestamp_ms, action)

            if not transaction_index[key]:
                raise RuntimeError(
                    f"Missing transaction for {key}"
                )

            transaction = transaction_index[key].popleft()

            assert_close(
                row["quantity"],
                transaction["quantity"],
                "Executed quantity",
                atol=1e-6,
            )

            assert_close(
                row["cash_after_usd"],
                transaction["cash_after_usd"],
                "Cash after execution",
            )

            executed_decisions += 1

            if action == "BUY":
                buy_count += 1
            else:
                sell_count += 1

        else:
            hold_or_other_decisions += 1

        payload = create_payload(
            row=row,
            transaction=transaction,
            source_row_number=source_row_number,
        )

        historical_time = pd.to_datetime(
            timestamp_ms,
            unit="ms",
            utc=True,
        ).isoformat()

        ledger.append_event(
            event_type=str(row["event"]),
            payload=payload,
            timestamp_utc=historical_time,
        )

    remaining = sum(
        len(queue)
        for queue in transaction_index.values()
    )

    if remaining != 0:
        raise RuntimeError(
            f"Unmatched transactions: {remaining}"
        )

    if executed_decisions != len(transactions):
        raise RuntimeError("Executed decision count mismatch.")

    if buy_count != sell_count:
        raise RuntimeError("Unbalanced BUY/SELL transactions.")

    if buy_count != reference["completed_trades"]:
        raise RuntimeError("Backtest trade count mismatch.")

    assert_close(
        transactions["fee_usd"].sum(),
        reference["total_fees_usd"],
        "Total fees",
    )

    assert_close(
        transactions["realized_pnl_usd"].sum(),
        reference["net_profit_usd"],
        "Realized P&L",
    )

    assert_close(
        transactions.iloc[-1]["cash_after_usd"],
        reference["final_equity_usd"],
        "Final equity",
    )

    if not ledger.verify_integrity():
        raise RuntimeError("Blockchain integrity failure.")

    print("\n=== ACCOUNTING VERIFICATION ===")
    print("Executed BUY:", buy_count)
    print("Executed SELL:", sell_count)
    print("Other decisions:", hold_or_other_decisions)
    print("Backtest reconciliation: PASSED")
    print("Blockchain integrity: PASSED")

    # Persist the blockchain in separate new files.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ledger.save_json(LEDGER_PATH)

    restored = BlockchainLedger.load_json(
        LEDGER_PATH
    )

    if restored.blocks != ledger.blocks:
        raise RuntimeError(
            "Saved blockchain differs from original."
        )

    audit = {
        "project": "AURUM-AI",
        "stage": "C.4.2",
        "asset": "XOMUSD",
        "source_split": "validation",
        "strategy": "Consensus + Trend 60m",
        "decision_count": int(len(decisions)),
        "transaction_count": int(len(transactions)),
        "executed_buy_count": buy_count,
        "executed_sell_count": sell_count,
        "other_decision_count": hold_or_other_decisions,
        "block_count_including_genesis": len(
            ledger.blocks
        ),
        "blockchain_integrity_verified": True,
        "json_reload_verified": True,
        "final_block_hash": ledger.blocks[-1]["hash"],
        "backtest_final_equity_usd": float(
            reference["final_equity_usd"]
        ),
        "source_file_sha256": {
            "decisions": file_sha256(DECISIONS_PATH),
            "transactions": file_sha256(
                TRANSACTIONS_PATH
            ),
            "summary": file_sha256(BACKTEST_PATH),
        },
        "historical_timestamp_source": "timestamp_ms",
        "processing_risk_timestamp_excluded": True,
        "test_dataset_used": False,
        "real_trading": False,
        "limitations": [
            "Local SHA-256 linked audit ledger.",
            "Reconstructed from historical CSV files.",
            "Not live event-stream recording.",
            "No distributed consensus mechanism.",
            "No external immutable hash anchor.",
            "A fully rewritten chain cannot be detected "
            "by internal hashing alone.",
        ],
    }

    with AUDIT_PATH.open("x", encoding="utf-8") as file:
        json.dump(
            audit,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    print("\n=== BLOCKCHAIN SUMMARY ===")
    print("Blocks including Genesis:", len(ledger.blocks))
    print("Final hash:", ledger.blocks[-1]["hash"])
    print("JSON reload: PASSED")

    print("\n=== OUTPUT FILES ===")
    print(LEDGER_PATH)
    print(AUDIT_PATH)

    print("\nOriginal XOM dataset preserved.")
    print("Historical simulation files preserved.")
    print("No new investment simulation executed.")
    print("Test dataset NOT used.")
    print("No real trading.")

    print("\nC.4.2 BLOCKCHAIN INTEGRATION COMPLETED")


if __name__ == "__main__":
    main()
