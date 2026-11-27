
from dataclasses import dataclass, asdict
from datetime import date
import math

from agents.portfolio_agent import PortfolioAgent
from agents.classification_strategy_adapter import (
    ClassificationStrategyAdapter,
)
from risk.risk_agent import RiskAgent, RiskSnapshot


# ======================================================
# AURUM-AI — EXECUTION AGENT V1
#
# Forecast -> Strategy -> Risk -> Execution -> Portfolio
#
# Long-only virtual simulation.
# No live brokerage connection.
# No real orders.
# ======================================================


@dataclass
class ExecutionDecision:
    timestamp_ms: int
    session_date: str
    predicted_class: str
    strategy_action: str
    strategy_reason: str
    risk_status: str
    risk_reason: str
    risk_timestamp_utc: str
    execution_status: str
    execution_reason: str
    price: float
    quantity: float
    cash_after_usd: float
    equity_after_usd: float
    realized_pnl_usd: float

    def to_dict(self):
        return asdict(self)


class ExecutionAgent:

    def __init__(
        self,
        portfolio=None,
        risk_agent=None,
        allocation_fraction=0.05,
        cost_bps=10.0,
    ):

        if (
            not math.isfinite(allocation_fraction)
            or not 0 < allocation_fraction <= 1
        ):
            raise ValueError(
                "Invalid allocation fraction."
            )

        if (
            not math.isfinite(cost_bps)
            or not 0 <= cost_bps < 10000
        ):
            raise ValueError("Invalid transaction cost.")

        self.portfolio = (
            portfolio
            if portfolio is not None
            else PortfolioAgent()
        )

        self.risk_agent = (
            risk_agent
            if risk_agent is not None
            else RiskAgent()
        )

        self.strategy = ClassificationStrategyAdapter()

        self.allocation_fraction = allocation_fraction
        self.cost_bps = cost_bps

        self.current_day = None
        self.day_start_equity = None
        self.last_event_timestamp = None

        self.decision_log = []

    def execute_forecast(
        self,
        predicted_class,
        price,
        timestamp_ms,
        session_date,
    ):

        # ------------------------------------------
        # 1. Validate historical event
        # ------------------------------------------

        if not math.isfinite(price) or price <= 0:
            raise ValueError("Invalid execution price.")

        if (
            isinstance(timestamp_ms, bool)
            or not isinstance(timestamp_ms, int)
            or timestamp_ms < 0
        ):
            raise ValueError("Invalid event timestamp.")

        if (
            self.last_event_timestamp is not None
            and timestamp_ms <= self.last_event_timestamp
        ):
            raise ValueError(
                "Events must be strictly chronological."
            )

        event_day = date.fromisoformat(session_date)

        if (
            self.current_day is not None
            and event_day < self.current_day
        ):
            raise ValueError(
                "Trading dates cannot go backwards."
            )

        # ------------------------------------------
        # 2. Update portfolio market value
        # ------------------------------------------

        state = self.portfolio.mark_to_market(
            price=float(price),
            timestamp_ms=timestamp_ms,
        )

        equity = state["equity_usd"]

        if equity <= 0:
            raise ValueError(
                "Portfolio equity must be positive."
            )

        # Daily reference is equity at the first
        # processed event of the session.
        if self.current_day != event_day:

            self.current_day = event_day
            self.day_start_equity = equity

        daily_loss = max(
            0.0,
            1.0 - equity / self.day_start_equity,
        )

        drawdown = state["drawdown_fraction"]

        exposure = state["exposure_fraction"]

        held_fraction = exposure

        # ------------------------------------------
        # 3. Classification Strategy Agent
        # ------------------------------------------

        strategy_decision = self.strategy.decide(
            predicted_class=predicted_class,
            held_fraction=held_fraction,
        )

        action = strategy_decision.action

        # ------------------------------------------
        # 4. Calculate proposed order
        # ------------------------------------------

        if action == "BUY":

            allocation = min(
                equity * self.allocation_fraction,
                self.portfolio.cash,
            )

            proposed_fraction = allocation / equity

        elif action == "SELL":

            # Full liquidation of existing long
            # holdings when a DOWN signal arrives.
            allocation = 0.0

            proposed_fraction = held_fraction

        else:

            allocation = 0.0
            proposed_fraction = 0.0

        # ------------------------------------------
        # 5. Existing Risk Agent V2
        # ------------------------------------------

        snapshot = RiskSnapshot(
            current_exposure_fraction=exposure,
            proposed_order_fraction=proposed_fraction,
            daily_loss_fraction=daily_loss,
            drawdown_fraction=drawdown,
            held_fraction=held_fraction,
        )

        risk_decision = self.risk_agent.decide(
            action=action,
            snapshot=snapshot,
        )

        # ------------------------------------------
        # 6. Execution decision
        # ------------------------------------------

        status = risk_decision.status

        quantity = 0.0
        realized_pnl = 0.0

        if status == "APPROVED":

            if action == "BUY":

                transaction = self.portfolio.buy(
                    price=float(price),
                    allocation_usd=allocation,
                    timestamp_ms=timestamp_ms,
                    cost_bps=self.cost_bps,
                )

                quantity = transaction.quantity

                execution_status = "EXECUTED"

                execution_reason = (
                    "Risk-approved virtual BUY executed."
                )

            elif action == "SELL":

                transaction = self.portfolio.sell(
                    price=float(price),
                    quantity=self.portfolio.quantity,
                    timestamp_ms=timestamp_ms,
                    cost_bps=self.cost_bps,
                )

                quantity = transaction.quantity

                realized_pnl = (
                    transaction.realized_pnl_usd
                )

                execution_status = "EXECUTED"

                execution_reason = (
                    "Risk-approved virtual SELL executed."
                )

            else:
                raise RuntimeError(
                    "Risk approved an invalid action."
                )

        elif status == "REJECTED":

            execution_status = "BLOCKED"

            execution_reason = (
                "Order blocked by Risk Agent: "
                + risk_decision.reason
            )

        elif status == "NO_ACTION":

            if action != "HOLD":
                raise RuntimeError(
                    "Unexpected NO_ACTION for an order."
                )

            execution_status = "NO_ACTION"

            execution_reason = (
                "No order: Strategy selected HOLD."
            )

        else:
            raise RuntimeError(
                "Unknown Risk Agent status."
            )

        # ------------------------------------------
        # 7. Updated portfolio snapshot
        # ------------------------------------------

        final_state = self.portfolio.mark_to_market(
            price=float(price),
            timestamp_ms=timestamp_ms,
        )

        result = ExecutionDecision(
            timestamp_ms=timestamp_ms,
            session_date=session_date,
            predicted_class=(
                strategy_decision.predicted_class
            ),
            strategy_action=action,
            strategy_reason=(
                strategy_decision.reason
            ),
            risk_status=status,
            risk_reason=risk_decision.reason,
            risk_timestamp_utc=(
                risk_decision.timestamp_utc
            ),
            execution_status=execution_status,
            execution_reason=execution_reason,
            price=float(price),
            quantity=float(quantity),
            cash_after_usd=final_state["cash_usd"],
            equity_after_usd=final_state["equity_usd"],
            realized_pnl_usd=float(realized_pnl),
        )

        self.decision_log.append(result.to_dict())

        self.last_event_timestamp = timestamp_ms

        return result


# ======================================================
# SYNTHETIC INTEGRATION TESTS
# ======================================================


def main():

    print(
        "\n=== AURUM-AI | EXECUTION AGENT V1 ==="
    )

    execution = ExecutionAgent(
        portfolio=PortfolioAgent(
            initial_capital=10000.0
        ),
        risk_agent=RiskAgent(),
        allocation_fraction=0.05,
        cost_bps=10.0,
    )

    # These prices are SYNTHETIC, not XOM data.
    # Class: DOWN=0, NEUTRAL=1, UP=2
    #
    # Same fictitious trading day throughout.

    cases = [
        (
            "UP - open position",
            2, 100.0, 1000000, "EXECUTED"
        ),
        (
            "UP - already holding",
            2, 102.0, 1300000, "NO_ACTION"
        ),
        (
            "NEUTRAL - hold",
            1, 101.0, 1600000, "NO_ACTION"
        ),
        (
            "DOWN - close position",
            0, 101.0, 1900000, "EXECUTED"
        ),
        (
            "DOWN - empty portfolio",
            0, 100.0, 2200000, "NO_ACTION"
        ),
        (
            "UP - open second position",
            2, 100.0, 2500000, "EXECUTED"
        ),
        (
            "DOWN - close at a loss",
            0, 40.0, 2800000, "EXECUTED"
        ),
        (
            "UP - daily loss protection",
            2, 40.0, 3100000, "BLOCKED"
        ),
    ]

    for index, case in enumerate(
        cases, start=1
    ):

        (
            name,
            cls,
            price,
            timestamp,
            expected_status,
        ) = case

        result = execution.execute_forecast(
            predicted_class=cls,
            price=price,
            timestamp_ms=timestamp,
            session_date="2026-10-08",
        )

        print(f"\n=== CASE {index} ===")
        print(f"Scenario: {name}")
        print(f"Forecast: {result.predicted_class}")
        print(
            f"Strategy: {result.strategy_action}"
        )
        print(f"Risk: {result.risk_status}")
        print(
            f"Risk reason: {result.risk_reason}"
        )
        print(
            f"Execution: {result.execution_status}"
        )
        print(
            f"Quantity: {result.quantity:.6f}"
        )
        print(
            f"Cash: {result.cash_after_usd:.2f} USD"
        )
        print(
            f"Equity: {result.equity_after_usd:.2f} USD"
        )
        print(
            f"Realized P&L: "
            f"{result.realized_pnl_usd:+.2f} USD"
        )

        assert (
            result.execution_status
            == expected_status
        ), (
            f"Case {index}: "
            f"Expected {expected_status}, "
            f"got {result.execution_status}"
        )

        print("Test Result: PASSED")

    # ------------------------------------------
    # Final consistency checks
    # ------------------------------------------

    portfolio = execution.portfolio

    assert len(execution.decision_log) == 8

    assert len(portfolio.transactions) == 4

    assert portfolio.quantity == 0.0

    assert execution.decision_log[-1][
        "risk_status"
    ] == "REJECTED"

    assert math.isclose(
        portfolio.cash,
        portfolio.initial_capital
        + portfolio.realized_pnl,
        rel_tol=0,
        abs_tol=1e-6,
    )

    print("\n=== EXECUTION SUMMARY ===")

    print(
        f"Initial capital: "
        f"{portfolio.initial_capital:.2f} USD"
    )

    print(
        f"Final capital: "
        f"{portfolio.cash:.2f} USD"
    )

    print(
        f"Realized P&L: "
        f"{portfolio.realized_pnl:+.2f} USD"
    )

    print(
        f"Transactions: "
        f"{len(portfolio.transactions)}"
    )

    print(
        f"Decision logs: "
        f"{len(execution.decision_log)}"
    )

    print(
        f"Max drawdown: "
        f"{portfolio.max_drawdown * 100:.4f}%"
    )

    print("\nAll 8 Execution Agent tests passed.")
    print("All market prices were synthetic.")
    print("No real orders executed.")
    print("No test dataset used.")

    print(
        "\nEXECUTION AGENT V1 TEST COMPLETED"
    )


if __name__ == "__main__":
    main()
