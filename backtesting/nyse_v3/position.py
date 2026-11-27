
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Position:
    side: str
    quantity: float
    entry_price: float
    entry_timestamp_ms: int

    def __post_init__(self):
        if self.side not in ("LONG", "SHORT"):
            raise ValueError("Invalid position side")

        if not math.isfinite(self.quantity) or self.quantity <= 0:
            raise ValueError("Quantity must be positive")

        if not math.isfinite(self.entry_price) or self.entry_price <= 0:
            raise ValueError("Entry price must be positive")

        if (
            isinstance(self.entry_timestamp_ms, bool)
            or not isinstance(self.entry_timestamp_ms, int)
            or self.entry_timestamp_ms < 0
        ):
            raise ValueError("Invalid entry timestamp")

    def pnl_usd(self, exit_price, cost_bps_per_side=10.0):
        if not math.isfinite(exit_price) or exit_price <= 0:
            raise ValueError("Invalid exit price")

        if not math.isfinite(cost_bps_per_side) or cost_bps_per_side < 0:
            raise ValueError("Invalid transaction cost")

        direction = 1 if self.side == "LONG" else -1

        gross = direction * self.quantity * (
            exit_price - self.entry_price
        )

        fees = (
            self.quantity
            * (self.entry_price + exit_price)
            * cost_bps_per_side
            / 10000.0
        )

        return gross - fees
