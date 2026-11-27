
from backtesting.nyse_v3.market import load_validation_market
from backtesting.nyse_v3.signals import load_validation_signals
from backtesting.nyse_v3.portfolio import PortfolioV3
from backtesting.nyse_v3.risk import RiskAgentV3
from backtesting.nyse_v3.risk_execution import RiskExecutionV3
from backtesting.nyse_v3.entry_execution import EntryExecutionV3


BAR_MS = 300_000


class BacktestEngineV3:
    """
    AURUM-AI: chronological NYSE V3 backtester.

    Validation only.
    Historical Bid-open prices and hypothetical execution costs.
    No real orders.
    """

    def __init__(
        self,
        initial_capital=10000.0,
        signal_threshold=0.002,
    ):
        self.market = load_validation_market()
        self.signals = load_validation_signals(
            threshold=signal_threshold
        )

        self.portfolio = PortfolioV3(
            initial_capital=initial_capital
        )
        self.risk = RiskAgentV3()

        self.risk_execution = RiskExecutionV3(
            portfolio=self.portfolio,
            risk_agent=self.risk,
        )

        self.entry_execution = EntryExecutionV3(
            portfolio=self.portfolio,
            risk_agent=self.risk,
        )

        self.events = []
        self.trades = []
        self.equity_curve = []

        self._active_signal = None
        self._has_run = False

        self.signals_by_entry = {}

        available_timestamps = set(
            self.market["timestamp"].astype("int64")
        )

        for signal in self.signals:
            entry_time = signal["entry_timestamp_ms"]
            exit_time = signal["exit_timestamp_ms"]

            if entry_time not in available_timestamps:
                raise ValueError(
                    f"Missing entry candle: {entry_time}"
                )

            if exit_time not in available_timestamps:
                raise ValueError(
                    f"Missing exit candle: {exit_time}"
                )

            if entry_time in self.signals_by_entry:
                raise ValueError(
                    f"Duplicate entry time: {entry_time}"
                )

            self.signals_by_entry[entry_time] = signal

    def status(self):
        return {
            "market_bars": len(self.market),
            "validation_sessions": (
                self.market["session_date"].nunique()
            ),
            "signals": len(self.signals),
            "initial_capital_usd": (
                self.portfolio.initial_capital
            ),
            "risk_agent_connected": (
                self.entry_execution.risk is self.risk
                and self.risk_execution.risk is self.risk
            ),
            "portfolio_connected": (
                self.entry_execution.portfolio
                is self.portfolio
                and self.risk_execution.portfolio
                is self.portfolio
            ),
        }

    def _record_close(
        self,
        position,
        exit_price,
        exit_timestamp_ms,
        net_pnl,
        reason,
    ):
        self.trades.append({
            "side": position.side,
            "entry_timestamp_ms": position.entry_timestamp_ms,
            "exit_timestamp_ms": exit_timestamp_ms,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "quantity": position.quantity,
            "net_pnl_usd": net_pnl,
            "exit_reason": reason,
        })

        self.events.append({
            "timestamp_ms": exit_timestamp_ms,
            "event": reason,
            "net_pnl_usd": net_pnl,
        })

    def run(self):
        """Process every Validation candle in chronological order."""

        if self._has_run:
            raise RuntimeError(
                "Engine already executed. Create a new instance."
            )

        self._has_run = True
        previous_bar = None

        for bar in self.market.itertuples(index=False):
            timestamp_ms = int(bar.timestamp)
            price = float(bar.open)
            session_date = str(bar.session_date)
            segment_id = bar.segment_id

            # Prevent holding through missing candles,
            # segment gaps or overnight boundaries.
            if (
                self.portfolio.position is not None
                and previous_bar is not None
            ):
                previous_time, previous_day, previous_segment = (
                    previous_bar
                )

                if (
                    timestamp_ms - previous_time != BAR_MS
                    or session_date != previous_day
                    or segment_id != previous_segment
                ):
                    raise RuntimeError(
                        "Open position crossed a market gap"
                    )

            # 1. Normal exit at the scheduled candle.
            # Exit first when no emergency liquidation is pending.
            if (
                self.portfolio.position is not None
                and self._active_signal is not None
                and not self.risk_execution.pending_liquidation
                and timestamp_ms
                == self._active_signal["exit_timestamp_ms"]
            ):
                position = self.portfolio.position

                pnl = self.portfolio.close_position(
                    price=price,
                    timestamp_ms=timestamp_ms,
                )

                self._record_close(
                    position,
                    price,
                    timestamp_ms,
                    pnl,
                    "SCHEDULED_EXIT",
                )

                self._active_signal = None

            # 2. Observe market and execute pending emergency exits.
            position_before_risk = self.portfolio.position

            risk_result = self.risk_execution.process_bar(
                price=price,
                session_date=session_date,
                timestamp_ms=timestamp_ms,
            )

            if risk_result["event"] == "EMERGENCY_EXIT":
                if position_before_risk is None:
                    raise RuntimeError(
                        "Emergency exit without a position"
                    )

                self._record_close(
                    position_before_risk,
                    price,
                    timestamp_ms,
                    risk_result["realized_pnl_usd"],
                    "EMERGENCY_EXIT",
                )

                self._active_signal = None

            elif risk_result["event"] == "LIQUIDATION_SCHEDULED":
                self.events.append({
                    "timestamp_ms": timestamp_ms,
                    "event": "LIQUIDATION_SCHEDULED",
                })

            # 3. Execute a new signal only after risk observation.
            signal = self.signals_by_entry.get(timestamp_ms)

            if signal is not None:
                outcome, reason = (
                    self.entry_execution.execute_entry(
                        side=signal["side"],
                        price=price,
                        timestamp_ms=timestamp_ms,
                        signal_available_at_ms=(
                            signal["signal_available_at_ms"]
                        ),
                    )
                )

                self.events.append({
                    "timestamp_ms": timestamp_ms,
                    "event": "ENTRY_" + outcome,
                    "side": signal["side"],
                    "reason": reason,
                })

                if outcome == "EXECUTED":
                    self._active_signal = signal

            # 4. Check that a planned exit was not missed.
            if (
                self.portfolio.position is not None
                and self._active_signal is not None
                and timestamp_ms
                > self._active_signal["exit_timestamp_ms"]
            ):
                # A pending emergency exit can legitimately occur
                # after the scheduled exit time.
                if not self.risk_execution.pending_liquidation:
                    raise RuntimeError(
                        "Scheduled exit was missed"
                    )

            # 5. Record portfolio equity after all actions.
            self.equity_curve.append({
                "timestamp_ms": timestamp_ms,
                "equity_usd": self.portfolio.equity(price),
            })

            previous_bar = (
                timestamp_ms,
                session_date,
                segment_id,
            )

        # Do not report a completed backtest with an open position.
        if (
            self.portfolio.position is not None
            or self.risk_execution.pending_liquidation
        ):
            raise RuntimeError(
                "Unclosed position at end of Validation"
            )

        return {
            "bars_processed": len(self.equity_curve),
            "signals": len(self.signals),
            "trades_closed": len(self.trades),
            "final_capital_usd": self.portfolio.cash,
            "net_pnl_usd": (
                self.portfolio.cash
                - self.portfolio.initial_capital
            ),
            "risk_halted": self.risk.halted,
        }
