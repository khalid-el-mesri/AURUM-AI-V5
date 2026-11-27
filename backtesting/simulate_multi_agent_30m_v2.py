
from pathlib import Path
from datetime import date
import json

import pandas as pd

from backtesting.simulate_multi_agent_30m import (
    load_inputs,
    HistoricalExecutionAgent,
)
from agents.portfolio_agent import PortfolioAgent
from risk.risk_agent import RiskAgent, RiskSnapshot
from risk.kill_switch_execution_adapter import (
    KillSwitchExecutionAdapter,
)


# ============================================================
# AURUM-AI — HISTORICAL MULTI-AGENT SIMULATION V2
#
# Dataset: XOM historical Bid OHLC
# Dataset split: VALIDATION ONLY
# Forecast models: CNN1D and Logistic Regression
#
# Architecture:
# Forecast -> Strategy -> Risk -> Execution -> Portfolio
#
# Additional protection:
# Persistent Kill Switch at 10% drawdown
#
# All transactions are 100% simulated.
# No real trading.
# No test dataset used.
# Not a genuine walk-forward backtest.
# ============================================================


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "reports" / "backtesting"

INITIAL_CAPITAL = 10000.0
ALLOCATION_FRACTION = 0.05
COST_BPS = 10.0


# ============================================================
# 1. HISTORICAL SIMULATION
# ============================================================


def simulate_model(
    model_name,
    class_column,
    candidates,
    prices,
):

    print(f"\n=== {model_name.upper()} V2 ===")

    # --------------------------------------------------------
    # Initialize Risk Agent + Persistent Kill Switch
    # --------------------------------------------------------

    guarded_risk = KillSwitchExecutionAdapter(
        risk_agent=RiskAgent()
    )

    # --------------------------------------------------------
    # Initialize Portfolio and Execution Agent
    # --------------------------------------------------------

    execution = HistoricalExecutionAgent(
        portfolio=PortfolioAgent(
            initial_capital=INITIAL_CAPITAL
        ),
        risk_agent=guarded_risk,
        allocation_fraction=ALLOCATION_FRACTION,
        cost_bps=COST_BPS,
    )

    # --------------------------------------------------------
    # Historical forecast events
    # --------------------------------------------------------

    events = {
        int(row.entry_execution_timestamp_ms): row
        for row in candidates.itertuples(index=False)
    }

    scheduled_exit_ts = None

    records = []

    scheduled_exits = 0
    early_exits = 0
    skipped_after_exit = 0
    observed_bars = 0

    # --------------------------------------------------------
    # Iterate chronologically through historical candles
    # --------------------------------------------------------

    for bar in prices.itertuples(index=False):

        ts = int(bar.timestamp)
        price = float(bar.open)

        session_date = str(bar.session_date)

        # IMPORTANT FIX:
        # ExecutionAgent uses datetime.date,
        # not string, for its internal current_day.
        event_day = date.fromisoformat(session_date)

        observed_bars += 1

        # ====================================================
        # STEP 1 — UPDATE PORTFOLIO
        # ====================================================

        state = execution.portfolio.mark_to_market(
            price=price,
            timestamp_ms=ts,
        )

        equity = float(state["equity_usd"])

        # ====================================================
        # STEP 2 — UPDATE DAILY EQUITY BASELINE
        # ====================================================

        # Keep current_day as datetime.date.
        # Never assign a string to execution.current_day.

        if execution.current_day != event_day:

            execution.current_day = event_day

            execution.day_start_equity = equity

        if (
            execution.day_start_equity is None
            or execution.day_start_equity <= 0
        ):
            raise RuntimeError(
                "Invalid daily equity baseline."
            )

        daily_loss = max(
            0.0,
            1.0
            - equity / execution.day_start_equity,
        )

        exposure = float(
            state["exposure_fraction"]
        )

        drawdown = float(
            state["drawdown_fraction"]
        )

        # ====================================================
        # STEP 3 — CONTINUOUS RISK OBSERVATION
        # ====================================================

        # Observe risk on every available 5-minute
        # bar, even when no forecast is available.

        snapshot = RiskSnapshot(
            current_exposure_fraction=exposure,
            proposed_order_fraction=0.0,
            daily_loss_fraction=daily_loss,
            drawdown_fraction=drawdown,
            held_fraction=exposure,
        )

        previous_system_state = guarded_risk.state

        guarded_risk.observe(
            timestamp_ms=ts,
            snapshot=snapshot,
        )

        # ====================================================
        # STEP 4 — DETECT KILL SWITCH ACTIVATION
        # ====================================================

        if (
            previous_system_state == "ACTIVE"
            and guarded_risk.state == "HALTED"
        ):

            records.append({
                "event": "KILL_SWITCH_ACTIVATED",
                "timestamp_ms": ts,
                "session_date": session_date,
                "model": model_name,
                "strategy_action": "HOLD",
                "risk_status": "HALTED",
                "risk_reason": (
                    "Maximum drawdown limit reached."
                ),
                "execution_status": "NO_ACTION",
                "system_state": "HALTED",
                "price": price,
                "equity_usd": equity,
                "drawdown_fraction": drawdown,
            })

        # ====================================================
        # STEP 5 — SCHEDULED POSITION EXIT
        # ====================================================

        exited_this_bar = False

        if scheduled_exit_ts is not None:

            if ts > scheduled_exit_ts:

                raise RuntimeError(
                    "Scheduled exit timestamp missed."
                )

            if ts == scheduled_exit_ts:

                guarded_risk.set_event_timestamp(ts)

                result = execution.close_scheduled(
                    price=price,
                    timestamp_ms=ts,
                )

                result["model"] = model_name

                result["session_date"] = session_date

                result["system_state"] = (
                    guarded_risk.state
                )

                records.append(result)

                scheduled_exits += 1

                scheduled_exit_ts = None

                exited_this_bar = True

        # ====================================================
        # STEP 6 — CHECK FORECAST AVAILABILITY
        # ====================================================

        signal = events.get(ts)

        if signal is None:
            continue

        # Avoid opening a new position on the same
        # candle where a scheduled exit occurred.

        if exited_this_bar:

            skipped_after_exit += 1

            continue

        predicted_class = int(
            getattr(signal, class_column)
        )

        # ====================================================
        # STEP 7 — EXECUTE MULTI-AGENT DECISION
        # ====================================================

        guarded_risk.set_event_timestamp(ts)

        decision = execution.execute_forecast(
            predicted_class=predicted_class,
            price=price,
            timestamp_ms=ts,
            session_date=session_date,
        )

        record = decision.to_dict()

        record["model"] = model_name

        record["event"] = "FORECAST_DECISION"

        record["signal_timestamp_ms"] = int(
            signal.signal_timestamp_ms
        )

        record["system_state"] = guarded_risk.state

        records.append(record)

        # ====================================================
        # STEP 8 — MANAGE POSITION LIFECYCLE
        # ====================================================

        if decision.execution_status == "EXECUTED":

            if decision.strategy_action == "BUY":

                if scheduled_exit_ts is not None:

                    raise RuntimeError(
                        "Overlapping positions detected."
                    )

                scheduled_exit_ts = int(
                    signal.scheduled_exit_timestamp_ms
                )

            elif decision.strategy_action == "SELL":

                # Actual DOWN forecast closes
                # an existing position early.

                scheduled_exit_ts = None

                early_exits += 1

    # ========================================================
    # STEP 9 — FINAL PORTFOLIO VALIDATION
    # ========================================================

    if scheduled_exit_ts is not None:

        raise RuntimeError(
            "A scheduled position remained open."
        )

    portfolio = execution.portfolio

    if abs(portfolio.quantity) > 1e-10:

        raise RuntimeError(
            "Portfolio is not flat at simulation end."
        )

    decisions = pd.DataFrame(records)

    transactions = pd.DataFrame(
        portfolio.transactions
    )

    # ========================================================
    # STEP 10 — TRANSACTION STATISTICS
    # ========================================================

    if transactions.empty:

        buy_count = 0
        sell_count = 0
        win_rate = 0.0

    else:

        buys = transactions[
            transactions["action"] == "BUY"
        ]

        sells = transactions[
            transactions["action"] == "SELL"
        ]

        buy_count = len(buys)

        sell_count = len(sells)

        win_rate = (
            float(
                (
                    sells["realized_pnl_usd"] > 0
                ).mean() * 100.0
            )
            if sell_count > 0
            else 0.0
        )

    if buy_count != sell_count:

        raise RuntimeError(
            "BUY/SELL counts do not match."
        )

    # ========================================================
    # STEP 11 — ACCOUNTING VERIFICATION
    # ========================================================

    final_equity = float(portfolio.cash)

    expected_equity = (
        INITIAL_CAPITAL
        + float(portfolio.realized_pnl)
    )

    if abs(final_equity - expected_equity) > 1e-5:

        raise RuntimeError(
            "Portfolio accounting mismatch."
        )

    # ========================================================
    # STEP 12 — RISK STATISTICS
    # ========================================================

    blocked_orders = int(
        (
            decisions["execution_status"] == "BLOCKED"
        ).sum()
    )

    transitions = guarded_risk.switch.transitions

    halt_timestamp_ms = (
        guarded_risk.switch.halted_at_ms
    )

    halt_time_utc = (
        pd.to_datetime(
            halt_timestamp_ms,
            unit="ms",
            utc=True,
        ).isoformat()
        if halt_timestamp_ms is not None
        else None
    )

    # No executed BUY should occur after
    # activation of the persistent Kill Switch.

    if halt_timestamp_ms is not None:

        buys_after_halt = decisions[
            (decisions["strategy_action"] == "BUY")
            & (
                decisions["execution_status"]
                == "EXECUTED"
            )
            & (
                decisions["timestamp_ms"]
                >= halt_timestamp_ms
            )
        ]

        if not buys_after_halt.empty:

            raise RuntimeError(
                "BUY executed after Kill Switch activation."
            )

    # ========================================================
    # STEP 13 — FINAL SUMMARY
    # ========================================================

    net_return_pct = (
        final_equity / INITIAL_CAPITAL - 1.0
    ) * 100.0

    max_drawdown_pct = float(
        portfolio.max_drawdown * 100.0
    )

    summary = {
        "model": model_name,
        "initial_capital_usd": INITIAL_CAPITAL,
        "final_equity_usd": final_equity,
        "net_profit_usd": float(
            portfolio.realized_pnl
        ),
        "net_return_pct": float(
            net_return_pct
        ),
        "completed_trades": int(
            sell_count
        ),
        "win_rate_pct": float(
            win_rate
        ),
        "scheduled_exits": int(
            scheduled_exits
        ),
        "early_down_exits": int(
            early_exits
        ),
        "blocked_orders": int(
            blocked_orders
        ),
        "skipped_after_exit": int(
            skipped_after_exit
        ),
        "observed_5min_bars": int(
            observed_bars
        ),
        "max_observed_drawdown_pct": float(
            max_drawdown_pct
        ),
        "total_fees_usd": float(
            portfolio.total_fees
        ),
        "final_system_state": guarded_risk.state,
        "halt_timestamp_ms": halt_timestamp_ms,
        "halt_timestamp_utc": halt_time_utc,
        "kill_switch_transitions": transitions,
    }

    # ========================================================
    # STEP 14 — TERMINAL OUTPUT
    # ========================================================

    print(f"Observed bars: {observed_bars}")

    print(f"Completed trades: {sell_count}")

    print(f"Scheduled exits: {scheduled_exits}")

    print(f"Early DOWN exits: {early_exits}")

    print(f"Risk-blocked orders: {blocked_orders}")

    print(f"System state: {guarded_risk.state}")

    print(f"Halt time UTC: {halt_time_utc}")

    print(
        f"Final equity: {final_equity:.2f} USD"
    )

    print(
        f"Net return: {net_return_pct:+.4f}%"
    )

    print(
        f"Win rate: {win_rate:.2f}%"
    )

    print(
        "Max observed drawdown: "
        f"{max_drawdown_pct:.4f}%"
    )

    return decisions, transactions, summary


# ============================================================
# 2. MAIN PROGRAM
# ============================================================


def main():

    print(
        "\n=== AURUM-AI | MULTI-AGENT V2 + KILL SWITCH ==="
    )

    # --------------------------------------------------------
    # Output files: NEW VERSION
    # --------------------------------------------------------

    output_files = {
        "cnn_decisions":
            OUTPUT_DIR
            / "multi_agent_cnn1d_v2_decisions.csv",

        "cnn_transactions":
            OUTPUT_DIR
            / "multi_agent_cnn1d_v2_transactions.csv",

        "logistic_decisions":
            OUTPUT_DIR
            / "multi_agent_logistic_v2_decisions.csv",

        "logistic_transactions":
            OUTPUT_DIR
            / "multi_agent_logistic_v2_transactions.csv",

        "summary":
            OUTPUT_DIR
            / "multi_agent_30m_validation_v2_summary.json",
    }

    # Never silently overwrite previous experiment files.

    for path in output_files.values():

        if path.exists():

            raise FileExistsError(
                f"Existing V2 file preserved: {path}"
            )

    # --------------------------------------------------------
    # Load historical validation dataset
    # --------------------------------------------------------

    candidates, prices = load_inputs()

    results = {}

    outputs = {}

    models = [
        (
            "CNN1D",
            "cnn_class",
            "cnn",
        ),
        (
            "Logistic Regression",
            "logistic_class",
            "logistic",
        ),
    ]

    # --------------------------------------------------------
    # Run both forecasting models
    # --------------------------------------------------------

    for model_name, class_column, prefix in models:

        decisions, transactions, summary = simulate_model(
            model_name=model_name,
            class_column=class_column,
            candidates=candidates,
            prices=prices,
        )

        outputs[
            f"{prefix}_decisions"
        ] = decisions

        outputs[
            f"{prefix}_transactions"
        ] = transactions

        results[prefix] = summary

    # --------------------------------------------------------
    # Build historical audit report
    # --------------------------------------------------------

    report = {
        "project": "AURUM-AI",
        "simulation": "Multi-Agent V2",
        "dataset": "XOM",
        "evaluation_split": "validation",
        "valid_signals": int(
            len(candidates)
        ),
        "initial_capital_usd": INITIAL_CAPITAL,
        "cost_bps_per_side": COST_BPS,
        "risk_monitoring":
            "Every available 5-minute Bid open",
        "kill_switch":
            "Persistent 10% drawdown halt",
        "results": results,
        "real_trading": False,
        "test_dataset_used": False,
        "genuine_walk_forward": False,
        "limitations": [
            "Models were selected using validation.",
            "The source contains Bid OHLC only.",
            "Historical Ask prices are unavailable.",
            "Transaction costs are hypothetical.",
            "Candle-open executions are idealized.",
            "Risk is monitored at bar opens.",
            "Intrabar price extremes are not monitored.",
            "Kill Switch blocks BUY, not existing SELL.",
            "The 10% threshold is not a guaranteed loss cap.",
            "No independent test-set result is reported.",
        ],
    }

    # --------------------------------------------------------
    # Save outputs only after simulations succeed
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for key, frame in outputs.items():

        frame.to_csv(
            output_files[key],
            index=False,
        )

    with open(
        output_files["summary"],
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    print("\n=== V2 OUTPUT FILES ===")

    for path in output_files.values():

        print(path)

    print("\nOriginal V1 files preserved.")

    print("Original XOM dataset preserved.")

    print("No real orders executed.")

    print("Test dataset was NOT used.")

    print(
        "\nMULTI-AGENT V2 SIMULATION COMPLETED"
    )


if __name__ == "__main__":

    main()
