
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting.nyse_v3.engine import BacktestEngineV3


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "backtesting"
    / "nyse_v3"
    / "lstm60m_validation_dashboard"
)


def export_dashboard_data():
    """Export Validation backtesting results for Streamlit."""

    # Execute the baseline backtest, not the stress test.
    engine = BacktestEngineV3()
    result = engine.run()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Transaction history
    trades = pd.DataFrame(engine.trades)
    trades.to_csv(
        OUTPUT_DIR / "trades.csv",
        index=False,
    )

    # 2. Historical portfolio equity
    equity = pd.DataFrame(engine.equity_curve)

    equity["datetime_utc"] = pd.to_datetime(
        equity["timestamp_ms"],
        unit="ms",
        utc=True,
    )

    equity.to_csv(
        OUTPUT_DIR / "equity_curve.csv",
        index=False,
    )

    # 3. Performance metrics
    values = equity["equity_usd"].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Invalid equity values")

    initial = float(engine.portfolio.initial_capital)

    peaks = np.maximum.accumulate(
        np.r_[initial, values]
    )[1:]

    drawdowns = (peaks - values) / peaks

    total_trades = len(trades)

    winning_trades = sum(
        trade["net_pnl_usd"] > 0
        for trade in engine.trades
    )

    summary = {
        "project": "AURUM-AI",
        "dataset": "XOM NYSE V3",
        "split": "validation",
        "model": "LSTM",
        "horizon_minutes": 60,
        "initial_capital_usd": initial,
        "final_capital_usd": float(
            result["final_capital_usd"]
        ),
        "net_pnl_usd": float(
            result["net_pnl_usd"]
        ),
        "return_pct": float(
            100 * result["net_pnl_usd"] / initial
        ),
        "maximum_drawdown_pct": float(
            100 * drawdowns.max()
        ),
        "market_bars": int(result["bars_processed"]),
        "signals": int(result["signals"]),
        "trades_closed": int(total_trades),
        "winning_trades": int(winning_trades),
        "win_rate_pct": float(
            100 * winning_trades / total_trades
            if total_trades else 0
        ),
        "risk_halted": bool(result["risk_halted"]),
        "cost_bps_per_side": 10.0,
        "execution_price": "historical Bid open",
        "simulation_only": True,
    }

    with open(
        OUTPUT_DIR / "summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
           ensure_ascii=False,
            allow_nan=False,
        )

    print("DASHBOARD EXPORT: PASS")
    print("Output:", OUTPUT_DIR)
    print("Trades:", total_trades)
    print("Equity points:", len(equity))
    print("Net PnL:", round(summary["net_pnl_usd"], 4))
    print("Max drawdown:", round(
        summary["maximum_drawdown_pct"], 6
    ), "%")

    return summary


if __name__ == "__main__":
    export_dashboard_data()
