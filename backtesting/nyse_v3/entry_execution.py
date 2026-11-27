
BAR_MS = 300_000


class EntryExecutionV3:
    """Risk-controlled historical LONG/SHORT entry."""

    def __init__(self, portfolio, risk_agent):
        self.portfolio = portfolio
        self.risk = risk_agent

    def execute_entry(
        self,
        side,
        price,
        timestamp_ms,
        signal_available_at_ms,
        allocation_fraction=0.05,
        cost_bps_per_side=10.0,
    ):
        if side == "NO_TRADE":
            return "NO_ACTION", "NO_TRADE"

        if side not in ("LONG", "SHORT"):
            raise ValueError("Invalid trading side")

        for value in (timestamp_ms, signal_available_at_ms):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("Invalid timestamp")

        if self.portfolio.position is not None:
            return "BLOCKED", "POSITION_ALREADY_OPEN"

        # Only the scheduled bar can execute the signal.
        if timestamp_ms != signal_available_at_ms + BAR_MS:
            return "SKIPPED", "INVALID_ENTRY_TIME"

        # Risk must have observed this historical market event.
        if self.risk.last_timestamp_ms != timestamp_ms:
            return "BLOCKED", "RISK_NOT_UPDATED"

        approved, reason = self.risk.approve_entry(
            side=side,
            order_fraction=allocation_fraction,
            current_exposure_fraction=0.0,
        )

        if not approved:
            return "BLOCKED", reason

        self.portfolio.open_position(
            side=side,
            price=price,
            timestamp_ms=timestamp_ms,
            allocation_fraction=allocation_fraction,
            cost_bps_per_side=cost_bps_per_side,
        )

        return "EXECUTED", "RISK_APPROVED"
