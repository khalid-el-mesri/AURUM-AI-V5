
from pathlib import Path
from datetime import date
import json

import numpy as np
import pandas as pd

from backtesting.simulate_multi_agent_30m import (
    HistoricalExecutionAgent,
    INPUT_PATH,
    PRICES_PATH,
    BAR_MS,
)
from backtesting.session_cutoff import session_cutoff_mask
from agents.portfolio_agent import PortfolioAgent
from risk.risk_agent import RiskAgent, RiskSnapshot
from risk.kill_switch_execution_adapter import (
    KillSwitchExecutionAdapter,
)


# ============================================================
# AURUM-AI — HISTORICAL MULTI-AGENT SIMULATION V3
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
# Candidate eligibility: NYSE calendar only (no future OHLC inspection)
# Historical gaps: visible operational events, no invented fills
#
# All transactions are 100% simulated.
# No real trading.
# No test dataset used.
# Not a genuine walk-forward backtest.
# ============================================================


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "reports" / "backtesting" / "calendar_cutoff_v3"

INITIAL_CAPITAL = 10000.0
ALLOCATION_FRACTION = 0.05
COST_BPS = 10.0


# ============================================================
# VALIDATION-ONLY INPUTS — POINT-IN-TIME CANDIDATE FILTER
# ============================================================


def load_inputs_v3():
    """Read validation inputs without asking whether future candles exist.

    Eligibility checks use the historical signal candle (already observed)
    and the exchange's published session calendar. Entry/exit data integrity
    is handled *as events unfold* by simulate_model(), never in this loader.
    """
    signals = pd.read_parquet(INPUT_PATH)
    required = {
        "signal_timestamp_ms", "exit_timestamp_ms", "session_date",
        "segment_id", "cnn_class", "logistic_class",
    }
    absent = sorted(required.difference(signals.columns))
    if absent:
        raise ValueError(f"Missing input signal columns: {absent}")
    if signals.empty or signals[list(required)].isna().any().any():
        raise ValueError("Signals are empty or contain missing required values")
    if signals["signal_timestamp_ms"].duplicated().any():
        raise ValueError("Duplicate signal timestamps")

    signals = signals.copy()
    signals["session_date"] = signals["session_date"].astype(str)
    for field in ("cnn_class", "logistic_class"):
        if not signals[field].isin((0, 1, 2)).all():
            raise ValueError(f"Invalid predicted classes: {field}")

    # The model's signal is at the start of its observed 5-minute candle.
    signals["entry_execution_timestamp_ms"] = (
        signals["signal_timestamp_ms"] + 2 * BAR_MS
    )
    signals["scheduled_exit_timestamp_ms"] = (
        signals["signal_timestamp_ms"] + 7 * BAR_MS
    )

    prices = pd.read_parquet(
        PRICES_PATH,
        columns=["timestamp", "open", "segment_id", "session_date", "bar_in_segment"],
        filters=[
            ("timestamp", ">=", int(signals["signal_timestamp_ms"].min())),
            ("timestamp", "<=", int(signals["scheduled_exit_timestamp_ms"].max())),
        ],
    )
    prices = prices.copy()
    prices["session_date"] = prices["session_date"].astype(str)
    prices = prices[prices["session_date"].isin(set(signals["session_date"]))]
    if prices.empty or prices["timestamp"].duplicated().any():
        raise ValueError("Empty or duplicate validation price timestamps")
    if prices[["timestamp", "open", "segment_id", "session_date", "bar_in_segment"]].isna().any().any():
        raise ValueError("Invalid missing fields in validation prices")
    if (prices["open"] <= 0).any():
        raise ValueError("Invalid non-positive Bid open prices")
    prices = prices.sort_values("timestamp").reset_index(drop=True)

    # Only verify that the candle *at the time of the signal* exists and
    # belongs to the recorded segment/session. No offsets into the future.
    observed = prices.set_index("timestamp", verify_integrity=True).reindex(
        signals["signal_timestamp_ms"].to_numpy(dtype=np.int64)
    )
    known_at_signal = (
        observed["open"].notna().to_numpy()
        & (observed["segment_id"].to_numpy() == signals["segment_id"].to_numpy())
        & (observed["session_date"].to_numpy() == signals["session_date"].to_numpy())
        & observed["bar_in_segment"].notna().to_numpy()
    )
    calendar_eligible = session_cutoff_mask(signals, exit_offset_bars=7)
    mask = known_at_signal & calendar_eligible.to_numpy(dtype=bool)
    signals["signal_bar_in_segment"] = observed["bar_in_segment"].to_numpy()
    candidates = signals.loc[mask].copy().sort_values("entry_execution_timestamp_ms")
    if candidates.empty:
        raise ValueError("No calendar-eligible validation signal candidates")
    if candidates["entry_execution_timestamp_ms"].duplicated().any():
        raise ValueError("Duplicate entry event timestamps")

    print("\n=== V3 CALENDAR-ONLY VALIDATION LOADER ===")
    print(f"Original signals: {len(signals)}")
    print(f"Rejected by NYSE calendar: {int((~calendar_eligible).sum())}")
    print(f"Rejected using already-known signal data: {int((~known_at_signal).sum())}")
    print(f"Eligible candidates: {len(candidates)}")
    print("Future execution candles: NOT inspected to select candidates")
    return candidates, prices


def _missed_entry_record(event_time, signal, model_name, observed_time=None):
    """Audit an absent entry candle, without pretending to execute a fill."""
    return {
        "event": "MISSING_ENTRY_BAR",
        "timestamp_ms": int(event_time),
        "observed_after_timestamp_ms": int(observed_time) if observed_time is not None else None,
        "session_date": str(signal.session_date),
        "signal_timestamp_ms": int(signal.signal_timestamp_ms),
        "model": model_name,
        "strategy_action": "HOLD",
        "risk_status": "NOT_EVALUATED",
        "risk_reason": "No candle open observed at the scheduled entry timestamp.",
        "execution_status": "SKIPPED_MISSING_MARKET_DATA",
        "system_state": "NO_TRADE",
    }


# ============================================================
# 1. HISTORICAL SIMULATION
# ============================================================


def simulate_model(
    model_name,
    class_column,
    candidates,
    prices,
):

    print(f"\n=== {model_name.upper()} V3 ===")

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

    if len(events) != len(candidates):
        raise ValueError("Duplicate forecast entry timestamps; no silent overwrites")

    scheduled_exit_ts = None

    # Sorted event list detects absent entry timestamps as time advances.
    sorted_entry_times = sorted(events)
    next_entry_index = 0
    last_bar_context = None

    records = []

    scheduled_exits = 0
    early_exits = 0
    skipped_after_exit = 0
    observed_bars = 0
    missed_entries = 0
    rejected_entry_context = 0
    data_interruption_exits = 0
    detected_market_discontinuities = 0

    # --------------------------------------------------------
    # Iterate chronologically through historical candles
    # --------------------------------------------------------

    for bar in prices.itertuples(index=False):

        ts = int(bar.timestamp)
        price = float(bar.open)

        session_date = str(bar.session_date)
        bar_segment = bar.segment_id
        bar_index = int(bar.bar_in_segment)

        # Verify a strictly chronological execution stream.
        if last_bar_context is not None and ts <= last_bar_context[0]:
            raise RuntimeError("Market price bars are not strictly chronological")

        # Detect forecast entries that were *not observed* when their scheduled
        # time passed. Recording this does not influence prior trade decisions.
        while (
            next_entry_index < len(sorted_entry_times)
            and sorted_entry_times[next_entry_index] < ts
        ):
            entry_ts = sorted_entry_times[next_entry_index]
            records.append(_missed_entry_record(entry_ts, events[entry_ts], model_name, ts))
            missed_entries += 1
            next_entry_index += 1

        consecutive = (
            last_bar_context is None
            or (
                ts - last_bar_context[0] == BAR_MS
                and session_date == last_bar_context[1]
                and bar_segment == last_bar_context[2]
                and bar_index == last_bar_context[3] + 1
            )
        )
        if not consecutive:
            detected_market_discontinuities += 1
        last_bar_context = (ts, session_date, bar_segment, bar_index)

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
        # STEP 5 — EXIT: ON TIME OR AT NEXT AVAILABLE OPEN
        # ====================================================

        exited_this_bar = False

        if scheduled_exit_ts is not None:
            # No previous candle in the current uninterrupted segment means
            # monitoring cannot certify normal bar-by-bar risk behavior.
            interrupted = not consecutive
            exit_due = ts >= scheduled_exit_ts
            if interrupted or exit_due:
                guarded_risk.set_event_timestamp(ts)
                scheduled_was = scheduled_exit_ts
                result = execution.close_scheduled(price=price, timestamp_ms=ts)
                if interrupted:
                    result["event"] = "DATA_INTERRUPTION_EXIT"
                    result["strategy_reason"] = (
                        "Data continuity failed; liquidated at first subsequent "
                        "observed Bid open. Execution is only a simulation."
                    )
                    data_interruption_exits += 1
                elif ts > scheduled_was:
                    result["event"] = "LATE_SCHEDULED_EXIT"
                    result["strategy_reason"] = (
                        "Scheduled price missing; exited at first subsequent observed Bid open."
                    )
                result["scheduled_exit_timestamp_ms"] = int(scheduled_was)
                result["exit_delay_ms"] = int(max(0, ts - scheduled_was))
                result["model"] = model_name
                result["session_date"] = session_date
                result["system_state"] = guarded_risk.state
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
        # This entry candle exists, regardless of whether a trade is taken.
        next_entry_index += 1

        # No trade on a discontinuous or wrong-segment entry candle.
        # The model's original segment and bar index were known at signal time.
        if (
            not consecutive
            or str(signal.session_date) != session_date
            or signal.segment_id != bar_segment
            or int(signal.signal_bar_in_segment) + 2 != bar_index
        ):
            records.append({
                "event": "INVALID_ENTRY_CONTEXT",
                "timestamp_ms": ts,
                "session_date": session_date,
                "signal_timestamp_ms": int(signal.signal_timestamp_ms),
                "model": model_name,
                "strategy_action": "HOLD",
                "risk_status": "NOT_EVALUATED",
                "risk_reason": "Entry time exists but session/segment/bar continuity is invalid.",
                "execution_status": "SKIPPED_DATA_DISCONTINUITY",
                "system_state": guarded_risk.state,
            })
            rejected_entry_context += 1
            continue

        # Avoid opening a new position on the same
        # candle where a scheduled exit occurred.

        if exited_this_bar:
            skipped_after_exit += 1
            records.append({
                "event": "ENTRY_SKIPPED_SAME_BAR_AS_EXIT",
                "timestamp_ms": ts,
                "session_date": session_date,
                "signal_timestamp_ms": int(signal.signal_timestamp_ms),
                "model": model_name,
                "strategy_action": "HOLD",
                "risk_status": "NOT_EVALUATED",
                "risk_reason": "Exit and re-entry at the same candle open are prohibited.",
                "execution_status": "SKIPPED_SAME_BAR_AS_EXIT",
                "system_state": guarded_risk.state,
            })
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

    # Remaining entry timestamps were never observed in the price stream.
    while next_entry_index < len(sorted_entry_times):
        entry_ts = sorted_entry_times[next_entry_index]
        records.append(_missed_entry_record(entry_ts, events[entry_ts], model_name))
        missed_entries += 1
        next_entry_index += 1

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
        (decisions.get("execution_status", pd.Series(dtype=str)) == "BLOCKED").sum()
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

    if halt_timestamp_ms is not None and not decisions.empty:

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
        "observed_5min_bars": int(observed_bars),
        "missing_entry_bars": int(missed_entries),
        "invalid_entry_contexts": int(rejected_entry_context),
        "data_interruption_exits": int(data_interruption_exits),
        "observed_market_discontinuities": int(detected_market_discontinuities),
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
        "\n=== AURUM-AI | MULTI-AGENT V3 + NYSE CUTOFF + KILL SWITCH ==="
    )

    # --------------------------------------------------------
    # Output files: V3 in a new dedicated directory
    # --------------------------------------------------------

    output_files = {
        "cnn_decisions": OUTPUT_DIR / "cnn1d_decisions_v3.csv",
        "cnn_transactions": OUTPUT_DIR / "cnn1d_transactions_v3.csv",
        "logistic_decisions": OUTPUT_DIR / "logistic_decisions_v3.csv",
        "logistic_transactions": OUTPUT_DIR / "logistic_transactions_v3.csv",
        "summary": OUTPUT_DIR / "validation_summary_v3.json",
    }

    # Never silently overwrite previous experiment files.

    for path in output_files.values():

        if path.exists():

            raise FileExistsError(
                f"Existing V3 file preserved: {path}"
            )

    # --------------------------------------------------------
    # Load historical validation dataset
    # --------------------------------------------------------

    candidates, prices = load_inputs_v3()

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
        "simulation": "Multi-Agent V3 Calendar Cutoff",
        "dataset": "XOM",
        "evaluation_split": "validation",
        "valid_signals": int(len(candidates)),
        "candidate_selection": "NYSE exchange calendar plus signal-time evidence only",
        "entry_data_policy": "Skip and audit absent/invalid entry candles at observation time",
        "exit_data_policy": "At first available Bid open after scheduled exit or data interruption; flag risk",
        "historical_result_comparison": "Exploratory validation; not independently validated",
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
            "Unknown trading halts cannot be predicted from exchange calendars.",
            "Price gaps may force exits at later observed opens, with unmeasured execution slippage.",
            "Risk checks only happen when historical bars are observable.",
            "This V3 corrects 30m base loader and runner only; separate 60m strategy is not patched.",
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

    print("\n=== V3 OUTPUT FILES ===")

    for path in output_files.values():

        print(path)

    print("\nOriginal V1/V2 files preserved.")

    print("Original XOM dataset preserved.")

    print("No real orders executed.")

    print("Test dataset was NOT used.")

    print(
        "\nMULTI-AGENT V3 SIMULATION COMPLETED"
    )


if __name__ == "__main__":

    main()
