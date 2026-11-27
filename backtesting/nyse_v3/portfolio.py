
import math

from backtesting.nyse_v3.position import Position


class PortfolioV3:
    """Simulated LONG/SHORT portfolio. One position maximum."""

    def __init__(self, initial_capital=10000.0):
        if not math.isfinite(initial_capital) or initial_capital <= 0:
            raise ValueError("Invalid initial capital")

        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.position = None
        self.entry_fee = 0.0

    def open_position(
        self,
        side,
        price,
        timestamp_ms,
        allocation_fraction=0.05,
        cost_bps_per_side=10.0,
    ):
        if self.position is not None:
            raise ValueError("A position is already open")

        if (
            not math.isfinite(allocation_fraction)
            or not 0 < allocation_fraction <= 1
        ):
            raise ValueError("Invalid allocation fraction")

        if not math.isfinite(price) or price <= 0:
            raise ValueError("Invalid entry price")

        if (
            not math.isfinite(cost_bps_per_side)
            or not 0 <= cost_bps_per_side < 10000
        ):
            raise ValueError("Invalid transaction cost")

        notional = self.cash * allocation_fraction
        quantity = notional / price

        position = Position(
            side=side,
            quantity=quantity,
            entry_price=price,
            entry_timestamp_ms=timestamp_ms,
        )

        self.entry_fee = notional * cost_bps_per_side / 10000.0
        self.cash -= self.entry_fee
        self.position = position

        return position

    def equity(self, price):
        if not math.isfinite(price) or price <= 0:
            raise ValueError("Invalid market price")

        if self.position is None:
            return self.cash

        direction = 1 if self.position.side == "LONG" else -1

        unrealized_pnl = (
            direction
            * self.position.quantity
            * (price - self.position.entry_price)
        )

        return self.cash + unrealized_pnl

    def close_position(
        self,
        price,
        timestamp_ms,
        cost_bps_per_side=10.0,
    ):
        if self.position is None:
            raise ValueError("No open position")

        if (
            isinstance(timestamp_ms, bool)
            or not isinstance(timestamp_ms, int)
            or timestamp_ms <= self.position.entry_timestamp_ms
        ):
            raise ValueError("Invalid exit timestamp")

        net_pnl = self.position.pnl_usd(
            exit_price=price,
            cost_bps_per_side=cost_bps_per_side,
        )

        # Entry fee was already deducted when opening.
        self.cash += net_pnl + self.entry_fee

        self.position = None
        self.entry_fee = 0.0

        return net_pnl
