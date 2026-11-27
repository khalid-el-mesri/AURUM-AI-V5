
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import math


# ======================================================
# AURUM-AI — RISK AGENT V2
# Academic simulation only — no real trading
# ======================================================


@dataclass
class RiskLimits:
    max_order_fraction: float = 0.05
    max_exposure_fraction: float = 0.25
    max_daily_loss_fraction: float = 0.02
    max_drawdown_fraction: float = 0.10


@dataclass
class RiskSnapshot:
    current_exposure_fraction: float
    proposed_order_fraction: float
    daily_loss_fraction: float
    drawdown_fraction: float
    held_fraction: float = 0.0


@dataclass
class RiskDecision:
    action: str
    status: str
    reason: str
    timestamp_utc: str

    def to_dict(self):
        return asdict(self)


class RiskAgent:

    def __init__(self, limits=None):
        self.limits = limits or RiskLimits()

    def decide(self, action, snapshot):

        if action not in ("BUY", "SELL", "HOLD"):
            raise ValueError("Invalid strategy action.")

        if not isinstance(snapshot, RiskSnapshot):
            raise TypeError("Expected RiskSnapshot.")

        for name, value in asdict(snapshot).items():

            if not math.isfinite(value):
                raise ValueError(
                    f"Invalid numerical value: {name}"
                )

            if not 0 <= value <= 1:
                raise ValueError(
                    f"{name} must be between 0 and 1."
                )

        timestamp = datetime.now(
            timezone.utc
        ).isoformat()

        def decision(status, reason):
            return RiskDecision(
                action=action,
                status=status,
                reason=reason,
                timestamp_utc=timestamp,
            )

        # =========================================
        # 1. HOLD — No order
        # =========================================

        if action == "HOLD":
            return decision(
                "NO_ACTION",
                "Strategy requested HOLD."
            )

        # =========================================
        # 2. Validate order quantity
        # =========================================

        if snapshot.proposed_order_fraction <= 0:
            return decision(
                "REJECTED",
                "Order size must be positive."
            )

        # =========================================
        # 3. SELL — Risk-reducing exit
        # =========================================

        if action == "SELL":

            if (
                snapshot.proposed_order_fraction
                > snapshot.held_fraction
            ):
                return decision(
                    "REJECTED",
                    "Insufficient holdings for SELL."
                )

            # Exiting existing holdings must not
            # be blocked by daily loss or drawdown.
            return decision(
                "APPROVED",
                "Risk-reducing SELL approved. "
                "Existing holdings can be reduced "
                "even when loss limits are reached."
            )

        # =========================================
        # 4. BUY — Apply risk limits
        # =========================================

        if (
            snapshot.daily_loss_fraction
            >= self.limits.max_daily_loss_fraction
        ):
            return decision(
                "REJECTED",
                "Daily loss limit reached."
            )

        if (
            snapshot.drawdown_fraction
            >= self.limits.max_drawdown_fraction
        ):
            return decision(
                "REJECTED",
                "Maximum drawdown limit reached."
            )

        if (
            snapshot.proposed_order_fraction
            > self.limits.max_order_fraction
        ):
            return decision(
                "REJECTED",
                "Order size exceeds risk limit."
            )

        new_exposure = (
            snapshot.current_exposure_fraction
            + snapshot.proposed_order_fraction
        )

        if new_exposure > self.limits.max_exposure_fraction:
            return decision(
                "REJECTED",
                "Portfolio exposure limit exceeded."
            )

        return decision(
            "APPROVED",
            "All configured BUY risk checks passed."
        )


if __name__ == "__main__":

    agent = RiskAgent()

    cases = [
        (
            "BUY — Normal",
            "BUY",
            RiskSnapshot(0, 0.05, 0, 0),
        ),
        (
            "BUY — Daily loss exceeded",
            "BUY",
            RiskSnapshot(0, 0.05, 0.03, 0),
        ),
        (
            "SELL — Daily loss exceeded",
            "SELL",
            RiskSnapshot(
                0.05, 0.05, 0.03, 0,
                held_fraction=0.05,
            ),
        ),
        (
            "SELL — Drawdown exceeded",
            "SELL",
            RiskSnapshot(
                0.05, 0.05, 0, 0.11,
                held_fraction=0.05,
            ),
        ),
        (
            "SELL — Insufficient holdings",
            "SELL",
            RiskSnapshot(
                0.05, 0.08, 0, 0,
                held_fraction=0.05,
            ),
        ),
        (
            "HOLD",
            "HOLD",
            RiskSnapshot(0, 0, 0, 0),
        ),
    ]

    expected = [
        "APPROVED",
        "REJECTED",
        "APPROVED",
        "APPROVED",
        "REJECTED",
        "NO_ACTION",
    ]

    print("\n=== AURUM-AI | RISK AGENT V2 TEST ===")

    for index, (name, action, snapshot) in enumerate(
        cases, start=1
    ):

        result = agent.decide(action, snapshot)

        print(f"\nCase {index}: {name}")
        print(f"Action: {result.action}")
        print(f"Status: {result.status}")
        print(f"Reason: {result.reason}")

        assert result.status == expected[index - 1], (
            f"Unexpected status in case {index}"
        )

    print("\nAll 6 Risk Agent V2 tests passed.")
    print("No orders executed.")
    print("\nRISK AGENT V2 TEST COMPLETED")
