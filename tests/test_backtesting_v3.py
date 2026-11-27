"""AURUM-AI V3: isolated event-stream tests with simulated agents.

Run from the project root:
    python -m unittest discover -s tests -p "test_backtesting_v3.py" -v

All market data below are synthetic. Tests do not alter XOM data or run the
historical backtest, and the execution stub is not a substitute for the real
agents' integration tests.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import backtesting.simulate_multi_agent_30m_v3 as v3


BAR = 300_000
DAY = "2024-07-02"
ORIGIN = pd.Timestamp("2024-07-02 09:30:00", tz="America/New_York").value // 1_000_000


def signals():
    return pd.DataFrame([{
        "signal_timestamp_ms": int(ORIGIN),
        "exit_timestamp_ms": int(ORIGIN + 6 * BAR),
        "session_date": DAY,
        "segment_id": 7,
        "cnn_class": 2,
        "logistic_class": 2,
    }])


def candles(missing=()):
    return pd.DataFrame([
        {"timestamp": int(ORIGIN + n * BAR), "open": float(100 + n * 0.1),
         "segment_id": 7, "session_date": DAY, "bar_in_segment": n}
        for n in range(13) if n not in missing
    ])


def candidates():
    s = signals()
    s["entry_execution_timestamp_ms"] = s["signal_timestamp_ms"] + 2 * BAR
    s["scheduled_exit_timestamp_ms"] = s["signal_timestamp_ms"] + 7 * BAR
    s["signal_bar_in_segment"] = 0
    return s


class FakePortfolio:
    def __init__(self, initial_capital):
        self.cash = float(initial_capital)
        self.quantity = 0.0
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        self.max_drawdown = 0.0
        self.transactions = []

    def equity(self, price):
        return self.cash + self.quantity * price

    def mark_to_market(self, price, timestamp_ms):
        equity = self.equity(price)
        return {"equity_usd": equity, "exposure_fraction":
                (self.quantity * price / equity) if equity else 0.0,
                "drawdown_fraction": self.max_drawdown}

    def buy(self, price, timestamp_ms):
        self.cash -= price
        self.quantity += 1
        self.transactions.append({"action": "BUY", "realized_pnl_usd": 0.0})

    def sell(self, price, timestamp_ms):
        assert self.quantity == 1
        self.quantity = 0
        self.cash += price
        self.realized_pnl = self.cash - 10000.0
        self.transactions.append({"action": "SELL", "realized_pnl_usd": self.realized_pnl})


class FakeRiskAdapter:
    def __init__(self, risk_agent):
        self.state = "ACTIVE"
        self.switch = SimpleNamespace(transitions=[], halted_at_ms=None)

    def observe(self, **kwargs):
        return None

    def set_event_timestamp(self, timestamp_ms):
        return None


class FakeExecution:
    def __init__(self, portfolio, risk_agent, allocation_fraction, cost_bps):
        self.portfolio = portfolio
        self.current_day = None
        self.day_start_equity = None
        self.risk_agent = risk_agent

    def execute_forecast(self, predicted_class, price, timestamp_ms, session_date):
        if predicted_class == 2 and not self.portfolio.quantity:
            self.portfolio.buy(price, timestamp_ms)
            action, status = "BUY", "EXECUTED"
        else:
            action, status = "HOLD", "NO_ACTION"
        return SimpleNamespace(strategy_action=action, execution_status=status,
                               to_dict=lambda: {"timestamp_ms": timestamp_ms,
                                                "strategy_action": action,
                                                "execution_status": status,
                                                "risk_status": "APPROVED"})

    def close_scheduled(self, price, timestamp_ms):
        self.portfolio.sell(price, timestamp_ms)
        return {"event": "SCHEDULED_EXIT", "timestamp_ms": timestamp_ms,
                "strategy_action": "SELL", "execution_status": "EXECUTED",
                "risk_status": "APPROVED"}


class TestV3Loader(unittest.TestCase):
    def test_missing_future_exit_does_not_filter_eligible_signal(self):
        raw, prices = signals(), candles(missing=(7,))
        def read_parquet(path, **kwargs):
            return raw.copy() if path == v3.INPUT_PATH else prices.copy()
        with patch.object(v3.pd, "read_parquet", side_effect=read_parquet), \
                patch.object(v3, "session_cutoff_mask", return_value=pd.Series([True])):
            candidates_loaded, p = v3.load_inputs_v3()
        self.assertEqual(len(candidates_loaded), 1)
        self.assertEqual(len(p), 12)
        self.assertEqual(int(candidates_loaded.iloc[0].signal_bar_in_segment), 0)

    def test_real_validation_matches_legacy_candidates_when_available(self):
        """Read-only check against XOM data on the user's workstation."""
        from pathlib import Path
        from backtesting.simulate_multi_agent_30m import INPUT_PATH, PRICES_PATH, load_inputs
        if not Path(INPUT_PATH).exists() or not Path(PRICES_PATH).exists():
            self.skipTest("Real XOM validation inputs not available here")
        previous, _ = load_inputs()
        current, _ = v3.load_inputs_v3()
        self.assertEqual(
            set(previous["signal_timestamp_ms"].astype("int64")),
            set(current["signal_timestamp_ms"].astype("int64")),
            "V3 eligibility diverges from previously diagnosed NYSE calendar baseline",
        )
        self.assertEqual(len(current), 13026)

    def test_nyse_cutoff_exclusion(self):
        raw, prices = signals(), candles()
        def read_parquet(path, **kwargs):
            return raw.copy() if path == v3.INPUT_PATH else prices.copy()
        with patch.object(v3.pd, "read_parquet", side_effect=read_parquet), \
                patch.object(v3, "session_cutoff_mask", return_value=pd.Series([False])):
            with self.assertRaisesRegex(ValueError, "No calendar-eligible"):
                v3.load_inputs_v3()


class TestV3EventStream(unittest.TestCase):
    def simulate(self, missing=(), signal_data=None, price_data=None):
        with patch.object(v3, "KillSwitchExecutionAdapter", FakeRiskAdapter), \
                patch.object(v3, "RiskAgent", lambda: object()), \
                patch.object(v3, "RiskSnapshot", lambda **kwargs: object()), \
                patch.object(v3, "PortfolioAgent", FakePortfolio), \
                patch.object(v3, "HistoricalExecutionAgent", FakeExecution):
            return v3.simulate_model("CNN1D", "cnn_class",
                                     candidates() if signal_data is None else signal_data,
                                     candles(missing) if price_data is None else price_data)

    def test_normal_buy_and_scheduled_exit(self):
        decisions, transactions, summary = self.simulate()
        self.assertEqual(summary["completed_trades"], 1)
        self.assertEqual(summary["missing_entry_bars"], 0)
        self.assertEqual(summary["data_interruption_exits"], 0)
        self.assertEqual(len(transactions), 2)
        self.assertIn("SCHEDULED_EXIT", decisions["event"].tolist())

    def test_second_entry_on_scheduled_exit_bar_is_skipped_and_logged(self):
        two = candidates()
        next_signal = two.iloc[[0]].copy()
        next_signal["signal_timestamp_ms"] = ORIGIN + 5 * BAR
        next_signal["entry_execution_timestamp_ms"] = ORIGIN + 7 * BAR
        next_signal["scheduled_exit_timestamp_ms"] = ORIGIN + 12 * BAR
        next_signal["signal_bar_in_segment"] = 5
        combined = pd.concat([two, next_signal], ignore_index=True)
        decisions, transactions, summary = self.simulate(signal_data=combined)
        self.assertEqual(summary["completed_trades"], 1)
        self.assertEqual(summary["skipped_after_exit"], 1)
        self.assertIn("ENTRY_SKIPPED_SAME_BAR_AS_EXIT", decisions["event"].tolist())

    def test_missing_entry_bar_is_audited_without_buy(self):
        decisions, transactions, summary = self.simulate(missing=(2,))
        self.assertEqual(summary["completed_trades"], 0)
        self.assertEqual(summary["missing_entry_bars"], 1)
        self.assertTrue(transactions.empty)
        self.assertIn("MISSING_ENTRY_BAR", decisions["event"].tolist())

    def test_gap_while_holding_forces_exit_at_next_observed_open(self):
        decisions, transactions, summary = self.simulate(missing=(4,))
        self.assertEqual(summary["completed_trades"], 1)
        self.assertEqual(summary["data_interruption_exits"], 1)
        item = decisions[decisions["event"] == "DATA_INTERRUPTION_EXIT"].iloc[0]
        self.assertEqual(int(item.timestamp_ms), ORIGIN + 5 * BAR)

    def test_missing_scheduled_exit_is_flagged(self):
        decisions, transactions, summary = self.simulate(missing=(7,))
        self.assertEqual(summary["data_interruption_exits"], 1)
        item = decisions[decisions["event"] == "DATA_INTERRUPTION_EXIT"].iloc[0]
        self.assertEqual(int(item.exit_delay_ms), BAR)
        self.assertEqual(summary["completed_trades"], 1)

    def test_duplicate_entry_events_are_rejected(self):
        duplicated = pd.concat([candidates(), candidates()], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate forecast entry"):
            self.simulate(signal_data=duplicated)

    def test_nonconsecutive_entry_is_rejected(self):
        # Observed bar at the exact entry time, but the preceding candle is absent.
        decisions, transactions, summary = self.simulate(missing=(1,))
        self.assertEqual(summary["invalid_entry_contexts"], 1)
        self.assertEqual(summary["completed_trades"], 0)
        self.assertIn("INVALID_ENTRY_CONTEXT", decisions["event"].tolist())

    def test_unclosed_position_is_an_error_not_an_invented_exit(self):
        # No available price after entry: V3 refuses to fabricate a closing trade.
        with self.assertRaisesRegex(RuntimeError, "scheduled position remained open"):
            self.simulate(price_data=candles().iloc[:3].copy())

    def test_unsorted_candles_fail_closed(self):
        p = candles()
        p.iloc[[2, 3]] = p.iloc[[3, 2]].to_numpy()
        with self.assertRaisesRegex(RuntimeError, "not strictly chronological"):
            self.simulate(price_data=p)


if __name__ == "__main__":
    unittest.main()
