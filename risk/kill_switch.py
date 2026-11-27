
from dataclasses import dataclass, asdict
import math

from risk.risk_agent import RiskAgent, RiskSnapshot


# ======================================================
# AURUM-AI — PERSISTENT RISK KILL SWITCH
#
# State: ACTIVE -> HALTED
# No automatic reset
# Risk-reducing SELL remains possible
# Academic simulation only
# ======================================================


@dataclass
class KillSwitchDecision:
    event_timestamp_ms: int
    action: str
    system_state: str
    status: str
    reason: str
    underlying_risk_status: str

    def to_dict(self):
        return asdict(self)


class RiskKillSwitch:

    def __init__(self, risk_agent=None):
        self.risk_agent = (
            risk_agent
            if risk_agent is not None
            else RiskAgent()
        )

        self.state = "ACTIVE"
        self.halted_at_ms = None
        self.halt_reason = None
        self.last_timestamp_ms = None
        self.transitions = []

    def evaluate(self, action, snapshot, timestamp_ms):

        if (
            isinstance(timestamp_ms, bool)
            or not isinstance(timestamp_ms, int)
            or timestamp_ms < 0
        ):
            raise ValueError("Invalid event timestamp.")

        if (
            self.last_timestamp_ms is not None
            and timestamp_ms < self.last_timestamp_ms
        ):
            raise ValueError(
                "Risk events must be chronological."
            )

        if not isinstance(snapshot, RiskSnapshot):
            raise TypeError("Expected RiskSnapshot.")

        # Delegate normal validation to Risk Agent V2.
        base = self.risk_agent.decide(action, snapshot)

        limit = self.risk_agent.limits.max_drawdown_fraction
        drawdown = snapshot.drawdown_fraction

        if not math.isfinite(drawdown):
            raise ValueError("Invalid drawdown.")

        # Persistent ACTIVE -> HALTED transition.
        if self.state == "ACTIVE" and drawdown >= limit:

            self.state = "HALTED"
            self.halted_at_ms = timestamp_ms

            self.halt_reason = (
                "Maximum drawdown limit reached."
            )

            self.transitions.append({
                "event_timestamp_ms": timestamp_ms,
                "from_state": "ACTIVE",
                "to_state": "HALTED",
                "reason": self.halt_reason,
                "observed_drawdown_fraction": drawdown,
            })

        self.last_timestamp_ms = timestamp_ms

        # The Kill Switch blocks any new BUY while halted.
        if self.state == "HALTED" and action == "BUY":

            return KillSwitchDecision(
                event_timestamp_ms=timestamp_ms,
                action=action,
                system_state=self.state,
                status="HALTED",
                reason=(
                    "New BUY blocked: persistent "
                    "drawdown Kill Switch is active."
                ),
                underlying_risk_status=base.status,
            )

        # SELL still follows Risk Agent V2 checks,
        # including the available holdings constraint.
        return KillSwitchDecision(
            event_timestamp_ms=timestamp_ms,
            action=action,
            system_state=self.state,
            status=base.status,
            reason=base.reason,
            underlying_risk_status=base.status,
        )


def main():

    print("\n=== AURUM-AI | KILL SWITCH TEST ===")

    gate = RiskKillSwitch()

    cases = [
        (
            "Normal BUY",
            "BUY",
            RiskSnapshot(0, 0.05, 0, 0),
            1000000,
            "ACTIVE",
            "APPROVED",
        ),
        (
            "Drawdown reaches 11% while HOLD",
            "HOLD",
            RiskSnapshot(0.05, 0, 0, 0.11, 0.05),
            1300000,
            "HALTED",
            "NO_ACTION",
        ),
        (
            "New BUY while halted",
            "BUY",
            RiskSnapshot(0, 0.05, 0, 0.11),
            1600000,
            "HALTED",
            "HALTED",
        ),
        (
            "Risk-reducing SELL while halted",
            "SELL",
            RiskSnapshot(0.05, 0.05, 0.03, 0.12, 0.05),
            1900000,
            "HALTED",
            "APPROVED",
        ),
        (
            "BUY after apparent recovery",
            "BUY",
            RiskSnapshot(0, 0.05, 0, 0.01),
            2200000,
            "HALTED",
            "HALTED",
        ),
        (
            "SELL without holdings",
            "SELL",
            RiskSnapshot(0, 0.05, 0, 0.01, 0),
            2500000,
            "HALTED",
            "REJECTED",
        ),
    ]

    for number, case in enumerate(cases, start=1):

        (
            name, action, snapshot, timestamp,
            expected_state, expected_status
        ) = case

        result = gate.evaluate(
            action=action,
            snapshot=snapshot,
            timestamp_ms=timestamp,
        )

        assert result.system_state == expected_state, (
            f"State mismatch in test {number}"
        )

        assert result.status == expected_status, (
            f"Status mismatch in test {number}"
        )

        print(f"\n=== TEST {number} ===")
        print(f"Scenario: {name}")
        print(f"System state: {result.system_state}")
        print(f"Action: {result.action}")
        print(f"Status: {result.status}")
        print(f"Reason: {result.reason}")
        print("Result: PASSED")

    assert len(gate.transitions) == 1
    assert gate.halted_at_ms == 1300000

    print("\n=== KILL SWITCH SUMMARY ===")
    print(f"Final state: {gate.state}")
    print(f"State transitions: {len(gate.transitions)}")
    print(f"Halt event timestamp: {gate.halted_at_ms}")
    print("Automatic reset: DISABLED")

    print("\nAll 6 Kill Switch tests passed.")
    print("Synthetic risk scenarios only.")
    print("No real orders executed.")
    print("No test dataset used.")
    print("\nKILL SWITCH TEST COMPLETED")


if __name__ == "__main__":
    main()
