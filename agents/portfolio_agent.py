
from dataclasses import dataclass, asdict
import math


# ======================================================
# AURUM-AI — PORTFOLIO AGENT V1
#
# Long-only virtual portfolio.
# Fractional shares allowed.
# Bid prices are simulation references.
# No real orders or financial transactions.
# ======================================================


@dataclass
class PortfolioTransaction:
    timestamp_ms: int
    action: str
    price: float
    quantity: float
    fee_usd: float
    cash_change_usd: float
    realized_pnl_usd: float
    cash_after_usd: float
    holdings_after: float

    def to_dict(self):
        return asdict(self)


class PortfolioAgent:

    def __init__(self, initial_capital=10000.0):

        self._positive(initial_capital, "Initial capital")

        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.quantity = 0.0

        # Total acquisition cost, including entry fees.
        self.position_cost_basis = 0.0

        self.realized_pnl = 0.0
        self.total_fees = 0.0

        self.peak_equity = float(initial_capital)
        self.max_drawdown = 0.0

        self.last_timestamp_ms = None
        self.last_mark_price = None

        self.transactions = []

    @staticmethod
    def _positive(value, name):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive.")

    @staticmethod
    def _validate_cost(cost_bps):
        if (
            not math.isfinite(cost_bps)
            or cost_bps < 0
            or cost_bps >= 10000
        ):
            raise ValueError("Invalid cost in bps.")

    def _validate_time(self, timestamp_ms):

        if isinstance(timestamp_ms, bool) or not isinstance(
            timestamp_ms, int
        ):
            raise TypeError("Timestamp must be an integer.")

        if timestamp_ms < 0:
            raise ValueError("Timestamp cannot be negative.")

        if (
            self.last_timestamp_ms is not None
            and timestamp_ms < self.last_timestamp_ms
        ):
            raise ValueError(
                "Portfolio events cannot go backwards in time."
            )

    def equity(self, price):

        self._positive(price, "Market price")

        return float(
            self.cash + self.quantity * price
        )

    def mark_to_market(self, price, timestamp_ms):

        self._positive(price, "Market price")
        self._validate_time(timestamp_ms)

        equity = self.equity(price)

        self.peak_equity = max(
            self.peak_equity, equity
        )

        drawdown = (
            (self.peak_equity - equity)
            / self.peak_equity
        )

        self.max_drawdown = max(
            self.max_drawdown, drawdown
        )

        self.last_mark_price = float(price)
        self.last_timestamp_ms = timestamp_ms

        unrealized_pnl = (
            self.quantity * price
            - self.position_cost_basis
        )

        exposure = (
            self.quantity * price / equity
            if equity > 0
            else 0.0
        )

        return {
            "timestamp_ms": timestamp_ms,
            "cash_usd": float(self.cash),
            "quantity": float(self.quantity),
            "market_price": float(price),
            "equity_usd": float(equity),
            "exposure_fraction": float(exposure),
            "realized_pnl_usd": float(self.realized_pnl),
            "unrealized_pnl_usd": float(unrealized_pnl),
            "total_fees_usd": float(self.total_fees),
            "return_pct": float(
                (equity / self.initial_capital - 1) * 100
            ),
            "drawdown_fraction": float(drawdown),
            "max_drawdown_fraction": float(
                self.max_drawdown
            ),
        }

    def buy(
        self,
        price,
        allocation_usd,
        timestamp_ms,
        cost_bps=10.0,
    ):

        self._positive(price, "Entry price")
        self._positive(allocation_usd, "Allocation")
        self._validate_cost(cost_bps)
        self._validate_time(timestamp_ms)

        if allocation_usd > self.cash + 1e-8:
            raise ValueError(
                "Insufficient cash for BUY."
            )

        allocation_usd = min(
            float(allocation_usd),
            self.cash,
        )

        cost = cost_bps / 10000.0

        # Allocation includes the entry-side fee.
        quantity = (
            allocation_usd / (price * (1 + cost))
        )

        gross_value = quantity * price
        fee = allocation_usd - gross_value

        self.cash -= allocation_usd
        self.quantity += quantity

        self.position_cost_basis += allocation_usd
        self.total_fees += fee

        self.last_timestamp_ms = timestamp_ms

        record = PortfolioTransaction(
            timestamp_ms=timestamp_ms,
            action="BUY",
            price=float(price),
            quantity=float(quantity),
            fee_usd=float(fee),
            cash_change_usd=-allocation_usd,
            realized_pnl_usd=0.0,
            cash_after_usd=float(self.cash),
            holdings_after=float(self.quantity),
        )

        self.transactions.append(record.to_dict())

        self.mark_to_market(price, timestamp_ms)

        return record

    def sell(
        self,
        price,
        quantity,
        timestamp_ms,
        cost_bps=10.0,
    ):

        self._positive(price, "Exit price")
        self._positive(quantity, "Sell quantity")
        self._validate_cost(cost_bps)
        self._validate_time(timestamp_ms)

        if quantity > self.quantity + 1e-10:
            raise ValueError(
                "Insufficient holdings for SELL."
            )

        quantity = min(
            float(quantity),
            self.quantity,
        )

        old_quantity = self.quantity

        # Allocate acquisition cost proportionally.
        cost_basis_removed = (
            self.position_cost_basis
            * quantity / old_quantity
        )

        gross_proceeds = quantity * price

        fee = (
            gross_proceeds * cost_bps / 10000.0
        )

        net_proceeds = gross_proceeds - fee

        realized_pnl = (
            net_proceeds - cost_basis_removed
        )

        self.cash += net_proceeds
        self.quantity -= quantity

        self.position_cost_basis -= cost_basis_removed

        self.realized_pnl += realized_pnl
        self.total_fees += fee

        # Avoid residual floating-point dust.
        if self.quantity < 1e-10:
            self.quantity = 0.0
            self.position_cost_basis = 0.0

        self.last_timestamp_ms = timestamp_ms

        record = PortfolioTransaction(
            timestamp_ms=timestamp_ms,
            action="SELL",
            price=float(price),
            quantity=float(quantity),
            fee_usd=float(fee),
            cash_change_usd=float(net_proceeds),
            realized_pnl_usd=float(realized_pnl),
            cash_after_usd=float(self.cash),
            holdings_after=float(self.quantity),
        )

        self.transactions.append(record.to_dict())

        self.mark_to_market(price, timestamp_ms)

        return record


# ======================================================
# TESTS — SYNTHETIC PORTFOLIO ONLY
# ======================================================

def main():

    print("\n=== AURUM-AI | PORTFOLIO AGENT V1 ===")

    portfolio = PortfolioAgent(
        initial_capital=10000.0
    )

    # ------------------------------------
    # TEST 1 — Initial portfolio
    # ------------------------------------

    initial = portfolio.mark_to_market(
        price=100.0,
        timestamp_ms=1000000,
    )

    assert initial["equity_usd"] == 10000.0
    assert initial["quantity"] == 0

    print("\n=== TEST 1: INITIAL PORTFOLIO ===")
    print(f"Cash: {portfolio.cash:.2f} USD")
    print(f"Equity: {initial['equity_usd']:.2f} USD")
    print("Result: PASSED")

    # ------------------------------------
    # TEST 2 — Virtual BUY
    # ------------------------------------

    buy = portfolio.buy(
        price=100.0,
        allocation_usd=500.0,
        timestamp_ms=1000000,
        cost_bps=10.0,
    )

    assert math.isclose(
        portfolio.cash, 9500.0,
        abs_tol=1e-7,
    )

    assert portfolio.quantity > 0
    assert buy.fee_usd > 0

    print("\n=== TEST 2: BUY ===")
    print(f"Quantity: {portfolio.quantity:.6f}")
    print(f"Cash: {portfolio.cash:.2f} USD")
    print(f"Entry fee: {buy.fee_usd:.4f} USD")
    print("Result: PASSED")

    # ------------------------------------
    # TEST 3 — Market price decreases
    # ------------------------------------

    down_state = portfolio.mark_to_market(
        price=95.0,
        timestamp_ms=1300000,
    )

    assert down_state["unrealized_pnl_usd"] < 0
    assert down_state["drawdown_fraction"] > 0

    print("\n=== TEST 3: UNREALIZED LOSS ===")
    print(
        f"Unrealized P&L: "
        f"{down_state['unrealized_pnl_usd']:+.2f} USD"
    )
    print(
        f"Drawdown: "
        f"{down_state['drawdown_fraction'] * 100:.4f}%"
    )
    print("Result: PASSED")

    # ------------------------------------
    # TEST 4 — Virtual SELL / close
    # ------------------------------------

    portfolio.mark_to_market(
        price=102.0,
        timestamp_ms=1600000,
    )

    sell = portfolio.sell(
        price=102.0,
        quantity=portfolio.quantity,
        timestamp_ms=1600000,
        cost_bps=10.0,
    )

    final_state = portfolio.mark_to_market(
        price=102.0,
        timestamp_ms=1600000,
    )

    assert portfolio.quantity == 0.0

    assert math.isclose(
        final_state["equity_usd"],
        final_state["cash_usd"],
        abs_tol=1e-7,
    )

    assert math.isclose(
        portfolio.initial_capital + portfolio.realized_pnl,
        portfolio.cash,
        abs_tol=1e-7,
    )

    print("\n=== TEST 4: SELL ===")
    print(
        f"Realized P&L: "
        f"{sell.realized_pnl_usd:+.2f} USD"
    )
    print(
        f"Final cash: "
        f"{portfolio.cash:.2f} USD"
    )
    print(
        f"Total fees: "
        f"{portfolio.total_fees:.4f} USD"
    )
    print("Result: PASSED")

    # ------------------------------------
    # TEST 5 — Prevent naked SELL
    # ------------------------------------

    rejected = False

    try:
        portfolio.sell(
            price=102.0,
            quantity=1.0,
            timestamp_ms=1600000,
        )
    except ValueError:
        rejected = True

    assert rejected

    print("\n=== TEST 5: NO SHORT SELLING ===")
    print("SELL without holdings: BLOCKED")
    print("Result: PASSED")

    # ------------------------------------
    # FINAL SUMMARY
    # ------------------------------------

    print("\n=== PORTFOLIO SUMMARY ===")

    print(f"Initial capital: 10000.00 USD")
    print(f"Final equity: {final_state['equity_usd']:.2f} USD")
    print(f"Realized P&L: {portfolio.realized_pnl:+.2f} USD")
    print(f"Transactions logged: {len(portfolio.transactions)}")
    print(
        f"Max observed drawdown: "
        f"{portfolio.max_drawdown * 100:.4f}%"
    )

    print("\nAll 5 Portfolio Agent tests passed.")
    print("All market prices above were synthetic.")
    print("No real trading.")
    print("No test dataset used.")
    print("\nPORTFOLIO AGENT V1 TEST COMPLETED")


if __name__ == "__main__":
    main()
