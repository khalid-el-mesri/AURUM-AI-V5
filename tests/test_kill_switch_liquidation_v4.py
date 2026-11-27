"""AURUM-AI V4 — isolated emergency-liquidation tests (synthetic bars).

Run from the AURUM-AI project root:
  python -m unittest discover -s tests -p 'test_kill_switch_liquidation_v4.py' -v

No files are modified by the tests, and this does not claim live execution.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import backtesting.simulate_multi_agent_30m_v4 as v4

BAR = 300_000
DAY = "2024-07-02"
ORIGIN = pd.Timestamp("2024-07-02 09:30", tz="America/New_York").value // 1_000_000


def signals():
    return pd.DataFrame([{
        "signal_timestamp_ms": int(ORIGIN),
        "entry_execution_timestamp_ms": int(ORIGIN + 2 * BAR),
        "scheduled_exit_timestamp_ms": int(ORIGIN + 7 * BAR),
        "signal_bar_in_segment": 0,
        "session_date": DAY,
        "segment_id": 7,
        "cnn_class": 2,
    }])


def bars(missing=()):
    return pd.DataFrame([
        {"timestamp": int(ORIGIN + j * BAR), "open": 100.0 + j * .1,
         "segment_id": 7, "session_date": DAY, "bar_in_segment": j}
        for j in range(13) if j not in missing
    ])


class FakePortfolio:
    def __init__(self, initial_capital):
        self.cash = float(initial_capital)
        self.quantity = 0.0
        self.max_drawdown = 0.0
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        self.transactions = []

    def equity(self, price):
        return self.cash + self.quantity * price

    def mark_to_market(self, price, timestamp_ms):
        e = self.equity(price)
        return {"equity_usd": e, "exposure_fraction": self.quantity * price / e,
                "drawdown_fraction": self.max_drawdown}

    def buy(self, price, timestamp_ms):
        assert self.quantity == 0
        self.cash -= float(price)
        self.quantity = 1.0
        self.transactions.append({"action": "BUY", "timestamp_ms": timestamp_ms,
                                  "realized_pnl_usd": 0.0})

    def sell(self, price, timestamp_ms):
        assert self.quantity > 0
        self.cash += float(price) * self.quantity
        self.quantity = 0.0
        self.realized_pnl = self.cash - 10000.0
        self.transactions.append({"action": "SELL", "timestamp_ms": timestamp_ms,
                                  "realized_pnl_usd": self.realized_pnl})


class ScriptedHaltRisk:
    trigger_ms = ORIGIN + 4 * BAR

    def __init__(self, risk_agent):
        self.state = "ACTIVE"
        self.switch = SimpleNamespace(transitions=[], halted_at_ms=None)

    def observe(self, timestamp_ms, snapshot):
        if self.state == "ACTIVE" and timestamp_ms >= self.trigger_ms:
            self.state = "HALTED"
            self.switch.halted_at_ms = int(timestamp_ms)
            self.switch.transitions.append({"event": "HALTED", "timestamp_ms": timestamp_ms})

    def set_event_timestamp(self, timestamp_ms):
        pass


class FakeExecution:
    def __init__(self, portfolio, risk_agent, allocation_fraction, cost_bps):
        self.portfolio = portfolio
        self.risk_agent = risk_agent
        self.day_start_equity = None
        self.current_day = None

    def execute_forecast(self, predicted_class, price, timestamp_ms, session_date):
        if predicted_class == 2 and self.risk_agent.state == "HALTED":
            action, status = "BUY", "BLOCKED"
        elif predicted_class == 2 and not self.portfolio.quantity:
            self.portfolio.buy(price, timestamp_ms)
            action, status = "BUY", "EXECUTED"
        else:
            action, status = "HOLD", "NO_ACTION"
        return SimpleNamespace(strategy_action=action, execution_status=status,
            to_dict=lambda: {"timestamp_ms": timestamp_ms, "strategy_action": action,
                             "execution_status": status, "risk_status": "APPROVED"})

    def close_scheduled(self, price, timestamp_ms):
        if self.portfolio.quantity <= 0:
            raise AssertionError("Duplicate SELL")
        self.portfolio.sell(price, timestamp_ms)
        return {"timestamp_ms": timestamp_ms, "event": "SCHEDULED_EXIT",
                "strategy_action": "SELL", "execution_status": "EXECUTED",
                "risk_status": "APPROVED"}


class TestKillSwitchEmergencyLiquidationV4(unittest.TestCase):
    def run_case(self, trigger_bars=4, signal_data=None, market=None, executor=FakeExecution):
        risk_cls = type("HaltAtCandle", (ScriptedHaltRisk,), {"trigger_ms": ORIGIN + trigger_bars * BAR})
        with patch.object(v4, "KillSwitchExecutionAdapter", risk_cls), \
             patch.object(v4, "RiskAgent", lambda: object()), \
             patch.object(v4, "RiskSnapshot", lambda **kwargs: object()), \
             patch.object(v4, "PortfolioAgent", FakePortfolio), \
             patch.object(v4, "HistoricalExecutionAgent", executor):
            return v4.simulate_model("CNN1D", "cnn_class",
                signals() if signal_data is None else signal_data,
                bars() if market is None else market)

    def test_forced_sell_without_strategy_sell(self):
        decision, transaction, stats = self.run_case()
        self.assertEqual(stats["kill_switch_emergency_exits"], 1)
        self.assertEqual(stats["scheduled_exits"], 0)
        self.assertEqual(stats["early_down_exits"], 0)
        self.assertEqual(stats["completed_trades"], 1)
        self.assertEqual(stats["final_system_state"], "HALTED")
        self.assertEqual(transaction.action.tolist(), ["BUY", "SELL"])
        self.assertEqual(int(transaction.iloc[1].timestamp_ms), ORIGIN + 4 * BAR)
        self.assertEqual(decision.event.tolist().count("KILL_SWITCH_EMERGENCY_EXIT"), 1)
        self.assertEqual(decision.event.tolist().count("KILL_SWITCH_ACTIVATED"), 1)

    def test_no_double_sell_when_scheduled_exit_same_bar(self):
        decision, transaction, stats = self.run_case(trigger_bars=7)
        self.assertEqual(stats["kill_switch_emergency_exits"], 1)
        self.assertEqual(stats["scheduled_exits"], 0)
        self.assertEqual(len(transaction), 2)
        self.assertFalse(decision.event.eq("SCHEDULED_EXIT").any())

    def test_no_double_sell_when_forecast_arrives_on_halt_bar(self):
        sig = signals()
        second = sig.copy()
        second["signal_timestamp_ms"] = ORIGIN + 2 * BAR
        second["entry_execution_timestamp_ms"] = ORIGIN + 4 * BAR
        second["signal_bar_in_segment"] = 2
        second["scheduled_exit_timestamp_ms"] = ORIGIN + 9 * BAR
        second["cnn_class"] = 0
        sig = pd.concat([sig, second], ignore_index=True)
        decision, transaction, stats = self.run_case(signal_data=sig)
        self.assertEqual(len(transaction), 2)
        self.assertEqual(stats["kill_switch_emergency_exits"], 1)
        self.assertIn("ENTRY_SKIPPED_SAME_BAR_AS_EXIT", decision.event.tolist())

    def test_halt_while_flat_does_not_create_fake_sell(self):
        decision, transaction, stats = self.run_case(trigger_bars=1)
        self.assertEqual(stats["kill_switch_emergency_exits"], 0)
        self.assertEqual(stats["completed_trades"], 0)
        self.assertTrue(transaction.empty)
        self.assertTrue(decision.event.eq("KILL_SWITCH_ACTIVATED").any())

    def test_missing_trigger_bar_exits_first_observed_open(self):
        decision, transaction, stats = self.run_case(market=bars(missing=(4,)))
        self.assertEqual(stats["kill_switch_emergency_exits"], 1)
        self.assertEqual(int(transaction.iloc[-1].timestamp_ms), ORIGIN + 5 * BAR)
        self.assertEqual(stats["data_interruption_exits"], 0)  # kill switch takes precedence

    def test_buy_signal_after_halt_is_blocked_and_audited(self):
        first = signals()
        extra = first.copy()
        extra["signal_timestamp_ms"] = ORIGIN + 3 * BAR
        extra["entry_execution_timestamp_ms"] = ORIGIN + 5 * BAR
        extra["scheduled_exit_timestamp_ms"] = ORIGIN + 10 * BAR
        extra["signal_bar_in_segment"] = 3
        combined = pd.concat([first, extra], ignore_index=True)
        decisions, transactions, stats = self.run_case(signal_data=combined)
        self.assertEqual(stats["kill_switch_emergency_exits"], 1)
        self.assertEqual(stats["blocked_orders"], 1)
        self.assertEqual(transactions.action.tolist(), ["BUY", "SELL"])
        self.assertEqual(decisions[(decisions.strategy_action == "BUY") &
                                   (decisions.execution_status == "BLOCKED")].shape[0], 1)

    def test_failed_liquidation_raises_error_instead_of_inventing_flat_portfolio(self):
        class RefusesSell(FakeExecution):
            def close_scheduled(self, price, timestamp_ms):
                raise RuntimeError("SELL execution rejected")
        with self.assertRaisesRegex(RuntimeError, "SELL execution rejected"):
            self.run_case(executor=RefusesSell)


if __name__ == "__main__":
    unittest.main()
