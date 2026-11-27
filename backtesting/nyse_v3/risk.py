
from dataclasses import dataclass
from datetime import date
import math


@dataclass(frozen=True)
class RiskLimitsV3:
    max_order_fraction: float = 0.05
    max_exposure_fraction: float = 0.25
    max_daily_loss_fraction: float = 0.02
    max_drawdown_fraction: float = 0.10

    def __post_init__(self):
        for value in (
            self.max_order_fraction,
            self.max_exposure_fraction,
            self.max_daily_loss_fraction,
            self.max_drawdown_fraction,
        ):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError("Invalid risk limit")


class RiskAgentV3:
    """Historical risk checks for simulated LONG and SHORT positions."""

    def __init__(self, limits=None):
        self.limits = limits or RiskLimitsV3()

        self.peak_equity = 0.0
        self.day_start_equity = None
        self.current_day = None
        self.last_timestamp_ms = None

        self.daily_loss_fraction = 0.0
        self.drawdown_fraction = 0.0
        self.halted = False

    def observe(self, equity, session_date, timestamp_ms):
        """Update risk using equity observable at this historical event."""

        if not math.isfinite(equity) or equity <= 0:
            raise ValueError("Invalid portfolio equity")

        if (
            isinstance(timestamp_ms, bool)
            or not isinstance(timestamp_ms, int)
            or timestamp_ms < 0
        ):
            raise ValueError("Invalid timestamp")

        if (
            self.last_timestamp_ms is not None
            and timestamp_ms <= self.last_timestamp_ms
        ):
            raise ValueError("Risk events must be chronological")

        day = date.fromisoformat(str(session_date))

        if self.current_day is not None and day < self.current_day:
            raise ValueError("Session dates cannot go backwards")

        if day != self.current_day:
            self.current_day = day
            self.day_start_equity = equity

        self.peak_equity = max(self.peak_equity, equity)

        self.daily_loss_fraction = max(
            0.0, 1.0 - equity / self.day_start_equity
        )

        self.drawdown_fraction = max(
            0.0, 1.0 - equity / self.peak_equity
        )

        if self.drawdown_fraction >= self.limits.max_drawdown_fraction:
            self.halted = True

        self.last_timestamp_ms = timestamp_ms

    def approve_entry(self, side, order_fraction,
                      current_exposure_fraction=0.0):
        """Approve or reject a proposed entry, never an exit."""

        if side not in ("LONG", "SHORT"):
            raise ValueError("Invalid position side")

        for value in (order_fraction, current_exposure_fraction):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Invalid exposure or order fraction")

        if self.last_timestamp_ms is None:
            raise RuntimeError("Observe market equity before trading")

        if self.halted:
            return False, "KILL_SWITCH_HALTED"

        if self.daily_loss_fraction >= self.limits.max_daily_loss_fraction:
            return False, "DAILY_LOSS_LIMIT"

        if order_fraction <= 0:
            return False, "NO_TRADE"

        if order_fraction > self.limits.max_order_fraction:
            return False, "ORDER_SIZE_LIMIT"

        if (current_exposure_fraction + order_fraction
                > self.limits.max_exposure_fraction):
            return False, "EXPOSURE_LIMIT"

        return True, "APPROVED"
