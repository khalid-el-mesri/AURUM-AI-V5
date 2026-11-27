
from datetime import datetime, timezone

from risk.kill_switch import RiskKillSwitch
from risk.risk_agent import RiskAgent, RiskDecision, RiskSnapshot
from agents.execution_agent import ExecutionAgent
from agents.portfolio_agent import PortfolioAgent


# =====================================================
# AURUM-AI — KILL SWITCH / EXECUTION ADAPTER
#
# Integrates persistent system halt protection
# without modifying ExecutionAgent or RiskAgent V2.
#
# Academic simulation only. No real trading.
# =====================================================


class KillSwitchExecutionAdapter:

    def __init__(self, risk_agent=None):

        self.switch = RiskKillSwitch(
            risk_agent=risk_agent or RiskAgent()
        )

        self.current_event_timestamp_ms = None
        self.last_gate_decision = None

        self.limits = self.switch.risk_agent.limits

    @property
    def state(self):
        return self.switch.state

    def set_event_timestamp(self, timestamp_ms):

        if (
            isinstance(timestamp_ms, bool)
            or not isinstance(timestamp_ms, int)
            or timestamp_ms < 0
        ):
            raise ValueError("Invalid event timestamp.")

        previous = self.switch.last_timestamp_ms

        if previous is not None and timestamp_ms < previous:
            raise ValueError("Non-chronological event.")

        self.current_event_timestamp_ms = timestamp_ms

    def observe(self, timestamp_ms, snapshot):
        """
        Check risk at an observed market event,
        even when no new BUY/SELL signal exists.

        The caller must construct the snapshot
        using only information available at that time.
        """

        self.set_event_timestamp(timestamp_ms)

        result = self.switch.evaluate(
            action="HOLD",
            snapshot=snapshot,
            timestamp_ms=timestamp_ms,
        )

        self.last_gate_decision = result

        return result

    def decide(self, action, snapshot):
        """
        Compatibility layer for ExecutionAgent.

        ExecutionAgent understands:
        APPROVED / REJECTED / NO_ACTION.

        An internal HALTED decision is translated
        to REJECTED, with an explicit system reason.
        """

        if self.current_event_timestamp_ms is None:
            raise RuntimeError(
                "Set the event timestamp before risk evaluation."
            )

        result = self.switch.evaluate(
            action=action,
            snapshot=snapshot,
            timestamp_ms=self.current_event_timestamp_ms,
        )

        self.last_gate_decision = result

        if result.status == "HALTED":

            status = "REJECTED"

            reason = (
                "SYSTEM_HALTED: "
                + result.reason
            )

        else:

            status = result.status
            reason = result.reason

        # Runtime audit time; not historical market time.
        return RiskDecision(
            action=action,
            status=status,
            reason=reason,
            timestamp_utc=datetime.now(
                timezone.utc
            ).isoformat(),
        )


# =====================================================
# SYNTHETIC INTEGRATION TESTS
# =====================================================


def main():

    print(
        "\n=== AURUM-AI | KILL SWITCH + EXECUTION ==="
    )

    guarded_risk = KillSwitchExecutionAdapter()

    execution = ExecutionAgent(
        portfolio=PortfolioAgent(
            initial_capital=10000.0
        ),
        risk_agent=guarded_risk,
        allocation_fraction=0.05,
        cost_bps=10.0,
    )

    session = "2026-10-08"

    # ----------------------------------------
    # TEST 1 — Normal BUY
    # ----------------------------------------

    guarded_risk.set_event_timestamp(1000000)

    result = execution.execute_forecast(
        predicted_class=2,
        price=100.0,
        timestamp_ms=1000000,
        session_date=session,
    )

    assert result.execution_status == "EXECUTED"
    assert guarded_risk.state == "ACTIVE"

    print("\n=== TEST 1 — NORMAL BUY ===")
    print(f"System: {guarded_risk.state}")
    print(f"Execution: {result.execution_status}")
    print("Result: PASSED")

    # ----------------------------------------
    # TEST 2 — Simulated risk shock
    # ----------------------------------------

    # Artificial risk snapshot used ONLY to test
    # the halt mechanism; it is not calculated
    # from the portfolio or XOM prices.

    exposure = execution.portfolio.equity(100.0)

    held_fraction = (
        execution.portfolio.quantity * 100.0
        / exposure
    )

    shock = RiskSnapshot(
        current_exposure_fraction=held_fraction,
        proposed_order_fraction=0.0,
        daily_loss_fraction=0.0,
        drawdown_fraction=0.11,
        held_fraction=held_fraction,
    )

    halt_result = guarded_risk.observe(
        timestamp_ms=1300000,
        snapshot=shock,
    )

    assert halt_result.system_state == "HALTED"
    assert len(guarded_risk.switch.transitions) == 1

    print("\n=== TEST 2 — RISK SHOCK ===")
    print(f"System: {guarded_risk.state}")
    print("Drawdown snapshot: 11% (synthetic)")
    print("Result: PASSED")

    # ----------------------------------------
    # TEST 3 — SELL while HALTED
    # ----------------------------------------

    guarded_risk.set_event_timestamp(1600000)

    result = execution.execute_forecast(
        predicted_class=0,
        price=100.0,
        timestamp_ms=1600000,
        session_date=session,
    )

    assert result.strategy_action == "SELL"
    assert result.risk_status == "APPROVED"
    assert result.execution_status == "EXECUTED"
    assert execution.portfolio.quantity == 0.0

    print("\n=== TEST 3 — SELL WHILE HALTED ===")
    print(f"System: {guarded_risk.state}")
    print(f"Risk: {result.risk_status}")
    print(f"Execution: {result.execution_status}")
    print("Result: PASSED")

    # ----------------------------------------
    # TEST 4 — BUY must remain blocked
    # ----------------------------------------

    guarded_risk.set_event_timestamp(1900000)

    result = execution.execute_forecast(
        predicted_class=2,
        price=100.0,
        timestamp_ms=1900000,
        session_date=session,
    )

    assert result.execution_status == "BLOCKED"
    assert result.risk_status == "REJECTED"
    assert "SYSTEM_HALTED" in result.risk_reason

    print("\n=== TEST 4 — BUY BLOCKED ===")
    print(f"System: {guarded_risk.state}")
    print(f"Execution: {result.execution_status}")
    print(f"Reason: {result.risk_reason}")
    print("Result: PASSED")

    # ----------------------------------------
    # TEST 5 — Recovery does NOT reset state
    # ----------------------------------------

    recovery_snapshot = RiskSnapshot(
        current_exposure_fraction=0.0,
        proposed_order_fraction=0.0,
        daily_loss_fraction=0.0,
        drawdown_fraction=0.01,
        held_fraction=0.0,
    )

    guarded_risk.observe(
        timestamp_ms=2200000,
        snapshot=recovery_snapshot,
    )

    assert guarded_risk.state == "HALTED"

    guarded_risk.set_event_timestamp(2500000)

    result = execution.execute_forecast(
        predicted_class=2,
        price=100.0,
        timestamp_ms=2500000,
        session_date=session,
    )

    assert result.execution_status == "BLOCKED"

    print("\n=== TEST 5 — PERSISTENT HALT ===")
    print(f"System: {guarded_risk.state}")
    print(f"Execution: {result.execution_status}")
    print("Automatic reset: DISABLED")
    print("Result: PASSED")

    # ----------------------------------------
    # FINAL CHECKS
    # ----------------------------------------

    assert len(execution.portfolio.transactions) == 2
    assert len(execution.decision_log) == 4
    assert len(guarded_risk.switch.transitions) == 1

    print("\n=== INTEGRATION SUMMARY ===")
    print("Tests passed: 5/5")
    print(f"System state: {guarded_risk.state}")
    print(
        "Executed transactions: "
        f"{len(execution.portfolio.transactions)}"
    )
    print(
        "Decision logs: "
        f"{len(execution.decision_log)}"
    )
    print(
        "Kill Switch transitions: "
        f"{len(guarded_risk.switch.transitions)}"
    )

    print("\nSynthetic risk scenario only.")
    print("No real orders executed.")
    print("No test dataset used.")

    print(
        "\nKILL SWITCH + EXECUTION INTEGRATION COMPLETED"
    )


if __name__ == "__main__":
    main()
