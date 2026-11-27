
from pathlib import Path
import json

import numpy as np
import pandas as pd

from agents.execution_agent import ExecutionAgent
from agents.portfolio_agent import PortfolioAgent
from risk.risk_agent import RiskAgent, RiskSnapshot


# =====================================================
# AURUM-AI — HISTORICAL MULTI-AGENT SIMULATION
#
# Forecast -> Strategy -> Risk -> Execution -> Portfolio
#
# XOM historical Bid OHLC
# Validation only
# Long-only, simulated capital
# NOT a genuine walk-forward backtest
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT / "reports"
    / "backtest_30m_validation_inputs.parquet"
)

PRICES_PATH = (
    ROOT / "data" / "02_clean"
    / "xom_segmented.parquet"
)

OUTPUT_DIR = ROOT / "reports" / "backtesting"

SUMMARY_PATH = (
    OUTPUT_DIR
    / "multi_agent_30m_validation_summary.json"
)

BAR_MS = 300_000

INITIAL_CAPITAL = 10000.0
ALLOCATION_FRACTION = 0.05
COST_BPS = 10.0

CLASS_NAMES = {
    0: "DOWN",
    1: "NEUTRAL",
    2: "UP",
}


class HistoricalExecutionAgent(ExecutionAgent):
    """Adds an explicit scheduled exit to ExecutionAgent."""

    def close_scheduled(self, price, timestamp_ms):

        if self.portfolio.quantity <= 0:
            raise ValueError("No position to close.")

        state = self.portfolio.mark_to_market(
            price=float(price),
            timestamp_ms=int(timestamp_ms),
        )

        exposure = state["exposure_fraction"]
        equity = state["equity_usd"]

        if self.day_start_equity is None:
            raise ValueError("Missing daily equity reference.")

        daily_loss = max(
            0.0,
            1.0 - equity / self.day_start_equity,
        )

        snapshot = RiskSnapshot(
            current_exposure_fraction=exposure,
            proposed_order_fraction=exposure,
            daily_loss_fraction=daily_loss,
            drawdown_fraction=state["drawdown_fraction"],
            held_fraction=exposure,
        )

        risk_decision = self.risk_agent.decide(
            action="SELL",
            snapshot=snapshot,
        )

        if risk_decision.status != "APPROVED":
            raise RuntimeError(
                "Scheduled risk-reducing exit was blocked."
            )

        transaction = self.portfolio.sell(
            price=float(price),
            quantity=self.portfolio.quantity,
            timestamp_ms=int(timestamp_ms),
            cost_bps=self.cost_bps,
        )

        result = {
            "event": "SCHEDULED_EXIT",
            "timestamp_ms": int(timestamp_ms),
            "strategy_action": "SELL",
            "strategy_reason": "Fixed-horizon exit.",
            "risk_status": risk_decision.status,
            "risk_reason": risk_decision.reason,
            "risk_timestamp_utc":
                risk_decision.timestamp_utc,
            "execution_status": "EXECUTED",
            "quantity": transaction.quantity,
            "realized_pnl_usd":
                transaction.realized_pnl_usd,
            "cash_after_usd": self.portfolio.cash,
            "equity_after_usd":
                self.portfolio.equity(price),
        }

        self.decision_log.append(result)

        return result


def load_inputs():

    print("\n=== LOADING VALIDATION DATA ===")

    signals = pd.read_parquet(INPUT_PATH)

    required = [
        "signal_timestamp_ms",
        "exit_timestamp_ms",
        "session_date",
        "segment_id",
        "cnn_class",
        "logistic_class",
    ]

    if signals.empty:
        raise ValueError("Empty signal dataset.")

    if signals[required].isna().any().any():
        raise ValueError("Missing signal data.")

    if signals["signal_timestamp_ms"].duplicated().any():
        raise ValueError("Duplicate signal timestamps.")

    signals = signals.copy()

    signals["session_date"] = (
        signals["session_date"].astype(str)
    )

    for col in ("cnn_class", "logistic_class"):
        if not signals[col].isin([0, 1, 2]).all():
            raise ValueError("Invalid forecast classes.")

    signals["entry_execution_timestamp_ms"] = (
        signals["signal_timestamp_ms"] + 2 * BAR_MS
    )

    signals["scheduled_exit_timestamp_ms"] = (
        signals["signal_timestamp_ms"] + 7 * BAR_MS
    )

    first_ts = int(signals["signal_timestamp_ms"].min())
    last_ts = int(
        signals["scheduled_exit_timestamp_ms"].max()
    )

    prices = pd.read_parquet(
        PRICES_PATH,
        columns=[
            "timestamp",
            "open",
            "segment_id",
            "session_date",
            "bar_in_segment",
        ],
        filters=[
            ("timestamp", ">=", first_ts),
            ("timestamp", "<=", last_ts),
        ],
    )

    # Explicitly keep validation dates only.
    validation_days = set(signals["session_date"])

    prices["session_date"] = (
        prices["session_date"].astype(str)
    )

    prices = prices[
        prices["session_date"].isin(validation_days)
    ].copy()

    if prices.empty:
        raise ValueError("No validation prices found.")

    if prices["timestamp"].duplicated().any():
        raise ValueError("Duplicate market timestamps.")

    prices = prices.sort_values("timestamp")

    if (
        prices["open"].isna().any()
        or (prices["open"] <= 0).any()
    ):
        raise ValueError("Invalid Bid open prices.")

    price_index = prices.set_index(
        "timestamp",
        verify_integrity=True,
    )

    signal_timestamps = (
        signals["signal_timestamp_ms"].to_numpy(
            dtype=np.int64
        )
    )

    signal_rows = price_index.reindex(
        signal_timestamps
    )

    valid = signal_rows["segment_id"].notna().to_numpy()

    expected_segments = signals[
        "segment_id"
    ].to_numpy()

    valid &= (
        signal_rows["segment_id"].to_numpy()
        == expected_segments
    )

    valid &= (
        signal_rows["session_date"].to_numpy()
        == signals["session_date"].to_numpy()
    )

    # Validate every candle from signal through exit.
    # This is retrospective data-quality filtering,
    # not information given to the forecasting model.
    for offset in range(1, 8):

        future = price_index.reindex(
            signal_timestamps + offset * BAR_MS
        )

        valid &= future["segment_id"].notna().to_numpy()

        valid &= (
            future["segment_id"].to_numpy()
            == expected_segments
        )

        valid &= (
            future["bar_in_segment"].to_numpy()
            == signal_rows["bar_in_segment"].to_numpy()
            + offset
        )

        valid &= (
            future["session_date"].to_numpy()
            == signals["session_date"].to_numpy()
        )

        valid &= future["open"].notna().to_numpy()

    candidates = signals.loc[valid].copy()

    candidates = candidates.sort_values(
        "entry_execution_timestamp_ms"
    )

    print(f"Original signals: {len(signals)}")
    print(f"Valid execution signals: {len(candidates)}")
    print(f"Excluded signals: {int((~valid).sum())}")

    if candidates.empty:
        raise ValueError("No valid execution candidates.")

    if candidates[
        "entry_execution_timestamp_ms"
    ].duplicated().any():
        raise ValueError("Duplicate entry event timestamps.")

    return candidates, prices


def run_model(model_name, class_column, candidates, prices):

    print(f"\n=== SIMULATING {model_name.upper()} ===")

    execution = HistoricalExecutionAgent(
        portfolio=PortfolioAgent(INITIAL_CAPITAL),
        risk_agent=RiskAgent(),
        allocation_fraction=ALLOCATION_FRACTION,
        cost_bps=COST_BPS,
    )

    # Each entry event uses a forecast created earlier.
    events = {
        int(row.entry_execution_timestamp_ms): row
        for row in candidates.itertuples(index=False)
    }

    scheduled_exit_ts = None
    records = []

    scheduled_exits = 0
    early_exits = 0
    blocked_orders = 0
    skipped_after_exit = 0

    for bar in prices.itertuples(index=False):

        ts = int(bar.timestamp)
        price = float(bar.open)
        session_date = str(bar.session_date)

        # Update equity on every available 5m open.
        execution.portfolio.mark_to_market(
            price=price,
            timestamp_ms=ts,
        )

        exited_this_bar = False

        # ------------------------------------------
        # 1. Scheduled exit before new decisions
        # ------------------------------------------

        if scheduled_exit_ts is not None:

            if ts > scheduled_exit_ts:
                raise RuntimeError(
                    "Missed a scheduled exit."
                )

            if ts == scheduled_exit_ts:

                result = execution.close_scheduled(
                    price=price,
                    timestamp_ms=ts,
                )

                result["model"] = model_name
                result["session_date"] = session_date

                records.append(result)

                scheduled_exits += 1
                scheduled_exit_ts = None
                exited_this_bar = True

        # ------------------------------------------
        # 2. Process forecast decision at entry time
        # ------------------------------------------

        signal = events.get(ts)

        if signal is None:
            continue

        # Prevent an exit and a new entry at the
        # exact same candle-open price.
        if exited_this_bar:
            skipped_after_exit += 1
            continue

        predicted_class = int(
            getattr(signal, class_column)
        )

        result = execution.execute_forecast(
            predicted_class=predicted_class,
            price=price,
            timestamp_ms=ts,
            session_date=session_date,
        )

        record = result.to_dict()

        record["model"] = model_name
        record["event"] = "FORECAST_DECISION"
        record["signal_timestamp_ms"] = int(
            signal.signal_timestamp_ms
        )

        records.append(record)

        if result.execution_status == "BLOCKED":
            blocked_orders += 1

        if result.execution_status == "EXECUTED":

            if result.strategy_action == "BUY":

                if scheduled_exit_ts is not None:
                    raise RuntimeError(
                        "Overlapping positions detected."
                    )

                scheduled_exit_ts = int(
                    signal.scheduled_exit_timestamp_ms
                )

            elif result.strategy_action == "SELL":

                # A real DOWN prediction closes the position
                # before its originally scheduled exit.
                scheduled_exit_ts = None
                early_exits += 1

    if scheduled_exit_ts is not None:
        raise RuntimeError("Unclosed scheduled position.")

    portfolio = execution.portfolio

    if portfolio.quantity != 0.0:
        raise RuntimeError("Portfolio not flat at the end.")

    trades = pd.DataFrame(portfolio.transactions)
    decisions = pd.DataFrame(records)

    buys = trades[trades["action"] == "BUY"]
    sells = trades[trades["action"] == "SELL"]

    if len(buys) != len(sells):
        raise RuntimeError("BUY/SELL accounting mismatch.")

    if len(sells):
        win_rate = float(
            (sells["realized_pnl_usd"] > 0).mean()
            * 100.0
        )
    else:
        win_rate = 0.0

    final_equity = float(portfolio.cash)

    if not np.isclose(
        final_equity,
        INITIAL_CAPITAL + portfolio.realized_pnl,
        rtol=0,
        atol=1e-6,
    ):
        raise RuntimeError("Portfolio cash reconciliation failed.")

    summary = {
        "model": model_name,
        "initial_capital_usd": INITIAL_CAPITAL,
        "final_equity_usd": final_equity,
        "net_profit_usd": float(portfolio.realized_pnl),
        "net_return_pct": float(
            (final_equity / INITIAL_CAPITAL - 1) * 100
        ),
        "completed_trades": int(len(sells)),
        "win_rate_pct": win_rate,
        "max_observed_drawdown_pct": float(
            portfolio.max_drawdown * 100
        ),
        "total_fees_usd": float(portfolio.total_fees),
        "scheduled_exits": scheduled_exits,
        "early_down_exits": early_exits,
        "risk_blocked_orders": blocked_orders,
        "signals_skipped_after_exit": skipped_after_exit,
        "decisions_logged": int(len(decisions)),
        "executed_transactions": int(len(trades)),
    }

    print(f"Completed trades: {len(sells)}")
    print(f"Scheduled exits: {scheduled_exits}")
    print(f"Early DOWN exits: {early_exits}")
    print(f"Risk-blocked orders: {blocked_orders}")
    print(f"Final equity: {final_equity:.2f} USD")
    print(
        f"Net return: "
        f"{summary['net_return_pct']:+.4f}%"
    )
    print(f"Win rate: {win_rate:.2f}%")
    print(
        f"Max observed drawdown: "
        f"{summary['max_observed_drawdown_pct']:.4f}%"
    )

    return decisions, trades, summary


def main():

    print(
        "\n=== AURUM-AI | HISTORICAL MULTI-AGENT ==="
    )

    outputs = {
        "cnn_decisions":
            OUTPUT_DIR / "multi_agent_cnn1d_decisions.csv",
        "cnn_trades":
            OUTPUT_DIR / "multi_agent_cnn1d_transactions.csv",
        "logistic_decisions":
            OUTPUT_DIR / "multi_agent_logistic_decisions.csv",
        "logistic_trades":
            OUTPUT_DIR / "multi_agent_logistic_transactions.csv",
        "summary": SUMMARY_PATH,
    }

    for path in outputs.values():
        if path.exists():
            raise FileExistsError(
                f"Existing artifact preserved: {path}"
            )

    candidates, prices = load_inputs()

    results = {}

    models = [
        (
            "CNN1D",
            "cnn_class",
            "cnn_decisions",
            "cnn_trades",
            "cnn1d",
        ),
        (
            "Logistic Regression",
            "logistic_class",
            "logistic_decisions",
            "logistic_trades",
            "logistic_regression",
        ),
    ]

    for name, class_column, d_key, t_key, result_key in models:

        decisions, transactions, summary = run_model(
            name,
            class_column,
            candidates,
            prices,
        )

        results[result_key] = summary

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        decisions.to_csv(
            outputs[d_key],
            index=False,
        )

        transactions.to_csv(
            outputs[t_key],
            index=False,
        )

    report = {
        "project": "AURUM-AI",
        "stage": "Historical Multi-Agent Simulation",
        "evaluation_split": "validation",
        "original_signals": 13337,
        "valid_signals": int(len(candidates)),
        "excluded_signals": int(
            13337 - len(candidates)
        ),
        "initial_capital_usd": INITIAL_CAPITAL,
        "cost_bps_per_side": COST_BPS,
        "results": results,
        "execution_assumptions": {
            "entry": "Signal timestamp + 10 minutes, Bid open",
            "exit": "Signal timestamp + 35 minutes, Bid open",
            "early_exit": "DOWN forecast when holdings exist",
            "same_session_required": True,
            "fractional_shares": True,
        },
        "real_trading": False,
        "genuine_walk_forward": False,
        "test_dataset_used": False,
        "limitations": [
            "Models and horizon selected on validation.",
            "Bid-only OHLC; real spread unavailable.",
            "Hypothetical 10 bps per-side costs.",
            "Idealized candle-open fills.",
            "No intrabar stop-loss.",
            "Drawdown sampled at available bar opens.",
            "Validation-only model comparison.",
            "Retrospective filtering of missing candles.",
            "Historical event and runtime audit times differ.",
            "Not an independent out-of-sample result.",
        ],
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=2)

    print("\n=== FILES SAVED ===")
    for path in outputs.values():
        print(path)

    print("\nHistorical decision logs recorded.")
    print("No real orders executed.")
    print("Test dataset was NOT used.")
    print("This is NOT genuine walk-forward testing.")

    print(
        "\nHISTORICAL MULTI-AGENT SIMULATION COMPLETED"
    )


if __name__ == "__main__":
    main()
