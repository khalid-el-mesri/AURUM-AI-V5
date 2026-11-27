
import math


class RiskExecutionV3:
    """Connect historical risk monitoring to portfolio liquidation."""

    def __init__(self, portfolio, risk_agent):
        self.portfolio = portfolio
        self.risk = risk_agent
        self.pending_liquidation = False
        self.last_timestamp_ms = None

    def process_bar(self, price, session_date, timestamp_ms):
        if not math.isfinite(price) or price <= 0:
            raise ValueError("Invalid market price")

        if (
            isinstance(timestamp_ms, bool)
            or not isinstance(timestamp_ms, int)
            or timestamp_ms < 0
        ):
            raise ValueError("Invalid bar timestamp")

        if (
            self.last_timestamp_ms is not None
            and timestamp_ms <= self.last_timestamp_ms
        ):
            raise ValueError("Bars must be strictly chronological")

        event = "NO_ACTION"
        realized_pnl = None

        # Liquidate at the NEXT observed bar after a risk halt.
        if self.pending_liquidation:
            if self.portfolio.position is None:
                raise RuntimeError("Pending liquidation without position")

            realized_pnl = self.portfolio.close_position(
                price=price,
                timestamp_ms=timestamp_ms,
            )
            self.pending_liquidation = False
            event = "EMERGENCY_EXIT"

        equity = self.portfolio.equity(price)

        self.risk.observe(
            equity=equity,
            session_date=session_date,
            timestamp_ms=timestamp_ms,
        )

        # Schedule liquidation; do not fill on the trigger bar.
        if self.risk.halted and self.portfolio.position is not None:
            self.pending_liquidation = True
            event = "LIQUIDATION_SCHEDULED"

        self.last_timestamp_ms = timestamp_ms

        return {
            "event": event,
            "timestamp_ms": timestamp_ms,
            "equity_usd": equity,
            "risk_halted": self.risk.halted,
            "pending_liquidation": self.pending_liquidation,
            "realized_pnl_usd": realized_pnl,
        }
