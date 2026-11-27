"""AURUM-AI Digital Twin: unit tests on synthetic histories only."""
from pathlib import Path

import pandas as pd
import pytest

from frontend.services.digital_twin_audit import (
    DECISIONS_REL, TRANSACTIONS_REL, ALT_EQUITY_REL, ALT_TRADES_REL,
    alternate_audit, attach_recorded_trades, audit_consensus, event_replay,
    load_consensus, load_lstm_alternate, read_csv_validated,
    segmented_equity, summary_json,
)

T0 = 1669833600000


def sample():
    decisions = pd.DataFrame([
        {"timestamp_ms": T0, "strategy_action": "BUY", "risk_status": "APPROVED",
         "execution_status": "EXECUTED", "cash_after_usd": 9500.0, "equity_after_usd": 9999.5,
         "price": 111.1, "event": "FORECAST_DECISION", "system_state": "ACTIVE",
         "risk_timestamp_utc": "2026-10-13T00:00:00+00:00", "strategy_reason": "UP",
         "predicted_class": "UP", "model": "Cons.+Trend 60m"},
        {"timestamp_ms": T0+300000, "strategy_action": "HOLD", "risk_status": "NO_ACTION",
         "execution_status": "NO_ACTION", "cash_after_usd": 9500.0, "equity_after_usd": 10003.,
         "price": 111.3, "event": "FORECAST_DECISION", "system_state": "ACTIVE"},
        {"timestamp_ms": T0+600000, "strategy_action": "SELL", "risk_status": "APPROVED",
         "execution_status": "EXECUTED", "cash_after_usd": 10001.0, "equity_after_usd": 10001.0,
         "price": None, "event": "SCHEDULED_EXIT", "system_state": "ACTIVE"},
    ])
    trades = pd.DataFrame([
        {"timestamp_ms": T0, "action": "BUY", "price": 111.1, "quantity": 4.5,
         "fee_usd": .5, "cash_after_usd": 9500., "holdings_after": 4.5},
        {"timestamp_ms": T0+600000, "action": "SELL", "price": 111.4, "quantity": 4.5,
         "fee_usd": .5, "cash_after_usd": 10001., "holdings_after": 0.0},
    ])
    return decisions, trades


def test_join_only_explicit_executions():
    d, t = sample()
    events = attach_recorded_trades(d, t)
    assert events["matched_trade"].tolist() == [True, False, True]
    assert events.iloc[2]["trade_price"] == pytest.approx(111.4)


def test_counts_are_from_source_not_guessed():
    d, t = sample()
    report = audit_consensus(d, t)
    assert (report["decisions"], report["transactions"]) == (3, 2)
    assert report["recorded_executed"] == 2
    assert report["matched_transactions"] == 2


def test_all_cash_balances_match():
    d, t = sample()
    assert audit_consensus(d, t)["cash_mismatches_gt_1_cent"] == 0


def test_cash_mismatch_is_reported():
    d, t = sample()
    t.loc[0, "cash_after_usd"] = 9497.5
    assert audit_consensus(d, t)["cash_mismatches_gt_1_cent"] == 1


def test_action_mismatch_never_fabricates_join():
    d, t = sample()
    t.loc[0, "action"] = "SELL"
    report = audit_consensus(d, t)
    assert report["unmatched_trade_rows"] >= 1
    assert report["unmatched_executed_decisions"] >= 1


def test_hold_not_matched_even_when_trade_timestamp_same():
    d, t = sample()
    t.loc[0, "timestamp_ms"] = T0+300000
    t.loc[0, "action"] = "HOLD"
    e = attach_recorded_trades(d, t)
    assert not e.iloc[1]["matched_trade"]


def test_replay_respects_historical_audit_time():
    d, t = sample()
    item = event_replay(d, t, 0)
    assert item["risk_timestamp_utc"] == "2026-10-13T00:00:00+00:00"
    assert item["timestamp_ms"] == T0
    assert item["mode"] == "READ_ONLY_HISTORICAL"


def test_missing_is_not_inferred():
    d, t = sample()
    item = event_replay(d, t, 1)
    assert item["matched_trade"] is False
    assert item["trade_holdings_after"] is None


def test_replay_out_of_bounds():
    d, t = sample()
    with pytest.raises(IndexError):
        event_replay(d, t, 99)


def test_long_gap_breaks_recorded_curve():
    d, _ = sample()
    d.loc[2, "timestamp_ms"] += 9 * 86400000
    out = segmented_equity(d)
    assert out["segment"].tolist() == [0, 0, 1]


def test_missing_equity_is_not_interpolated():
    d, _ = sample()
    d.loc[1, "equity_after_usd"] = None
    assert len(segmented_equity(d)) == 2


def test_reverse_chronology_is_detected():
    d, t = sample()
    d.loc[2, "timestamp_ms"] = T0-300000
    assert audit_consensus(d, t)["decision_reverse_timestamps"] == 1


def test_read_schema_rejects_missing_columns(tmp_path):
    filename = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp_ms": [T0]}).to_csv(filename, index=False)
    with pytest.raises(ValueError, match="absentes"):
        read_csv_validated(filename, {"timestamp_ms", "equity_usd"}, ("timestamp_ms",))


def test_read_rejects_fractional_timestamps(tmp_path):
    filename = tmp_path / "invalid.csv"
    pd.DataFrame({"timestamp_ms": [T0 + .5], "equity_usd": [100.] }).to_csv(filename, index=False)
    with pytest.raises(ValueError, match="non entiers"):
        read_csv_validated(filename, {"timestamp_ms", "equity_usd"}, ("timestamp_ms",))


def test_source_files_never_mutated(tmp_path):
    d, t = sample()
    dp, tp = tmp_path / DECISIONS_REL, tmp_path / TRANSACTIONS_REL
    dp.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(dp, index=False)
    t.to_csv(tp, index=False)
    before = dp.read_bytes(), tp.read_bytes()
    decisions, trades = load_consensus(tmp_path)
    audit_consensus(decisions, trades)
    event_replay(decisions, trades, 0)
    assert before == (dp.read_bytes(), tp.read_bytes())


def test_lstm_alternate_remains_independent(tmp_path):
    ep, tp = tmp_path / ALT_EQUITY_REL, tmp_path / ALT_TRADES_REL
    ep.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp_ms": [T0, T0+300000], "equity_usd": [10000., 10001.]}).to_csv(ep, index=False)
    pd.DataFrame({"side": ["LONG"], "entry_timestamp_ms": [T0], "exit_timestamp_ms": [T0+300000],
                  "entry_price": [100.], "exit_price": [101.], "quantity": [1.],
                  "net_pnl_usd": [.5], "exit_reason": ["SCHEDULED_EXIT"]}).to_csv(tp, index=False)
    eq, trades = load_lstm_alternate(tmp_path)
    audit = alternate_audit(eq, trades)
    assert audit["equity_observations"] == 2
    assert audit["trade_rows"] == 1
    assert audit["mixed_with_consensus"] is False


def test_json_marked_readonly():
    d, t = sample()
    import json
    data = json.loads(summary_json(audit_consensus(d, t)))
    assert data["mode"] == "READ_ONLY_HISTORICAL"
