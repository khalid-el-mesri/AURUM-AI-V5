"""AURUM-AI | V5: ex-ante 30m vs 60m validation comparison.

The SAME pre-trade eligible signals are simulated at both holding horizons.
A published NYSE calendar, not future candle availability, determines cohort
eligibility. Actual missing execution candles are handled during the V4
chronological simulation, not silently removed from the cohort.

Research only. Historical Bid bars; no real orders or untouched test split.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "03_features" / "xom_features_v2.parquet"
OUTPUT_DIR = ROOT / "reports" / "backtesting" / "combined_holding_30m_60m_v5_calendar"
SUMMARY_PATH = OUTPUT_DIR / "comparison_summary_v5.json"
BAR_MS = 300_000
EXIT_OFFSETS = {30: 7, 60: 13}


def prepare_ex_ante_common_signals(
    candidate_signals: pd.DataFrame,
    features: pd.DataFrame,
    calendar_mask_fn=None,
) -> tuple[pd.DataFrame, dict]:
    """Identify common 60m-eligible consensus+trend signals without future bars.

    candidate_signals must come from V4's point-in-time 30m loader and contain
    signal_bar_in_segment, entry_execution_timestamp_ms and relevant predictions.
    features are read only at the signal timestamp. With 5m candles labelled by
    their START time, the signal candle's close is known at t+5m; entry is t+10m.
    """
    if calendar_mask_fn is None:
        from backtesting.session_cutoff import session_cutoff_mask
        calendar_mask_fn = session_cutoff_mask
    if candidate_signals.empty:
        raise ValueError("No validation signals supplied")
    required = {
        "signal_timestamp_ms", "entry_execution_timestamp_ms", "session_date",
        "segment_id", "signal_bar_in_segment", "cnn_class", "logistic_class",
    }
    missing = sorted(required - set(candidate_signals.columns))
    if missing:
        raise ValueError(f"Missing candidate columns: {missing}")
    if candidate_signals["signal_timestamp_ms"].duplicated().any():
        raise ValueError("Duplicate signal timestamps")
    if not np.array_equal(
        candidate_signals["entry_execution_timestamp_ms"].to_numpy(dtype=np.int64),
        candidate_signals["signal_timestamp_ms"].to_numpy(dtype=np.int64) + 2 * BAR_MS,
    ):
        raise ValueError("Unexpected entry schedule")

    required_features = {
        "timestamp", "session_date", "segment_id", "bar_in_segment",
        "close", "sma_5", "ema_10", "volatility_10",
    }
    missing_features = sorted(required_features - set(features.columns))
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    if features["timestamp"].duplicated().any():
        raise ValueError("Duplicate feature timestamps")

    # Positions in both arrays refer to the same signal; NEVER compare
    # Future feature or future execution price rows here.
    original = candidate_signals.reset_index(drop=True).copy()
    observed = features.set_index("timestamp", verify_integrity=True).reindex(
        original["signal_timestamp_ms"].to_numpy(dtype=np.int64)
    )
    if observed["segment_id"].isna().any():
        raise ValueError("Missing observed features at one or more signals")
    consistent = (
        (observed["segment_id"].to_numpy() == original["segment_id"].to_numpy())
        & (observed["session_date"].astype(str).to_numpy()
           == original["session_date"].astype(str).to_numpy())
        & (observed["bar_in_segment"].to_numpy()
           == original["signal_bar_in_segment"].to_numpy())
    )
    if not consistent.all():
        raise ValueError("Signal-time features conflict with recorded context")

    consensus = (
        original["cnn_class"].eq(2).to_numpy()
        & original["logistic_class"].eq(2).to_numpy()
    )
    trend = (
        observed["close"].gt(observed["sma_5"])
        & observed["sma_5"].gt(observed["ema_10"])
        & observed["volatility_10"].notna()
        & observed["bar_in_segment"].ge(10)
    ).to_numpy(dtype=bool)

    chosen = original.loc[consensus & trend].copy()
    if chosen.empty:
        raise ValueError("No consensus + trend signals")

    # 60m's exit (t+65m) must fit an exchange session. Since 30m exits
    # earlier, this produces an identical EX-ANTE-eligible cohort for both.
    calendar_mask = calendar_mask_fn(chosen, exit_offset_bars=EXIT_OFFSETS[60])
    calendar_mask = np.asarray(calendar_mask, dtype=bool)
    if len(calendar_mask) != len(chosen):
        raise ValueError("Incorrect NYSE calendar mask shape")
    eligible = chosen.loc[calendar_mask].copy()
    if eligible.empty:
        raise ValueError("No 60m calendar-eligible consensus + trend signals")

    eligible["combined_class"] = 2
    eligible = eligible.sort_values("entry_execution_timestamp_ms").reset_index(drop=True)
    info = {
        "v4_30m_candidates": len(original),
        "consensus_and_trend_signals": len(chosen),
        "excluded_by_60m_nyse_cutoff": int((~calendar_mask).sum()),
        "common_ex_ante_eligible_signals": len(eligible),
    }
    return eligible, info


def extend_validation_prices(
    candidate_signals: pd.DataFrame,
    reference_prices: pd.DataFrame,
    price_path: Path,
) -> pd.DataFrame:
    """Observe validation candles through t+65m; never pre-screen signal exits.

    Loading data in advance for historical replay does not itself make those
    data available to trading decisions. V4 iterates over candles in time order.
    """
    if candidate_signals.empty or reference_prices.empty:
        raise ValueError("Empty validation source")
    required = {"timestamp", "open", "session_date", "segment_id", "bar_in_segment"}
    if required - set(reference_prices.columns):
        raise ValueError("Reference prices lack expected columns")
    dates = set(candidate_signals["session_date"].astype(str))
    start = int(reference_prices["timestamp"].min())
    end = int(candidate_signals["signal_timestamp_ms"].max()) + EXIT_OFFSETS[60] * BAR_MS
    observed = pd.read_parquet(
        price_path,
        columns=["timestamp", "open", "session_date", "segment_id", "bar_in_segment"],
        filters=[("timestamp", ">=", start), ("timestamp", "<=", end)],
    )
    observed = observed.copy()
    observed["session_date"] = observed["session_date"].astype(str)
    observed = observed.loc[observed["session_date"].isin(dates)].copy()
    observed = observed.sort_values("timestamp").reset_index(drop=True)
    if observed.empty or observed["timestamp"].duplicated().any():
        raise ValueError("Extended price stream empty or duplicated")
    if observed[list(required)].isna().any().any() or (observed["open"] <= 0).any():
        raise ValueError("Extended price stream contains invalid observations")

    # Cross-check the V4 execution price series to detect accidental
    # changes in source or chronology; check only overlapping bars.
    old = reference_prices.copy()
    old["session_date"] = old["session_date"].astype(str)
    old = old.set_index("timestamp", verify_integrity=True)
    new = observed.set_index("timestamp", verify_integrity=True)
    common = old.index.intersection(new.index)
    if len(common) == 0:
        raise ValueError("No overlapping V4 bars")
    if (old.index.difference(new.index).size and
            old.index.max() <= end and
            old.loc[old.index.difference(new.index), "session_date"].isin(dates).any()):
        raise ValueError("V4 validation market bars missing from expanded stream")
    if not np.allclose(
        old.loc[common, "open"].to_numpy(dtype=float),
        new.loc[common, "open"].to_numpy(dtype=float),
        rtol=0, atol=1e-9,
    ):
        raise ValueError("Extended market prices disagree with V4")
    for col in ("segment_id", "bar_in_segment", "session_date"):
        if not (old.loc[common, col].to_numpy() == new.loc[common, col].to_numpy()).all():
            raise ValueError(f"Extended {col} disagrees with V4")
    return observed


def run_horizon(common_signals, prices, horizon_minutes, simulate_fn):
    if horizon_minutes not in EXIT_OFFSETS:
        raise ValueError("Unsupported horizon")
    candidate = common_signals.copy()
    candidate["scheduled_exit_timestamp_ms"] = (
        candidate["signal_timestamp_ms"].astype(np.int64)
        + EXIT_OFFSETS[horizon_minutes] * BAR_MS
    )
    return simulate_fn(
        model_name=f"Consensus + Trend {horizon_minutes}m (V4 risk)",
        class_column="combined_class",
        candidates=candidate,
        prices=prices,
    )


def verify_result(decisions: pd.DataFrame, transactions: pd.DataFrame, summary: dict):
    buys = transactions.loc[transactions["action"].eq("BUY")]
    sells = transactions.loc[transactions["action"].eq("SELL")]
    if len(buys) != len(sells) or len(sells) != summary["completed_trades"]:
        raise RuntimeError("BUY/SELL counts do not match the summary")
    if not np.isclose(transactions["fee_usd"].sum(), summary["total_fees_usd"], atol=1e-6):
        raise RuntimeError("Transaction fees mismatch")
    if len(transactions):
        if not np.isclose(transactions["cash_after_usd"].iloc[-1], summary["final_equity_usd"], atol=1e-6):
            raise RuntimeError("Final cash/equity mismatch")
        if abs(transactions["holdings_after"].iloc[-1]) > 1e-8:
            raise RuntimeError("Unclosed position")
    if summary["early_down_exits"] != 0:
        raise RuntimeError("Unexpected early DOWN exit for UP-only candidate stream")
    if (summary["scheduled_exits"] + summary["kill_switch_emergency_exits"]
            + summary["early_down_exits"] != summary["completed_trades"]):
        raise RuntimeError("Trade exit-category count mismatch")
    if summary["halt_timestamp_ms"] is not None and not transactions.empty:
        late_buys = transactions.loc[
            transactions["action"].eq("BUY")
            & transactions["timestamp_ms"].ge(summary["halt_timestamp_ms"])
        ]
        if len(late_buys):
            raise RuntimeError("BUY after risk HALT")
    return True


def main():
    # Imports delayed to keep the selection routines independently testable.
    from backtesting.simulate_multi_agent_30m import PRICES_PATH
    from backtesting.simulate_multi_agent_30m_v4 import load_inputs_v4, simulate_model

    paths = {
        "30m_decisions": OUTPUT_DIR / "30m_decisions_v5.csv",
        "30m_transactions": OUTPUT_DIR / "30m_transactions_v5.csv",
        "60m_decisions": OUTPUT_DIR / "60m_decisions_v5.csv",
        "60m_transactions": OUTPUT_DIR / "60m_transactions_v5.csv",
        "summary": SUMMARY_PATH,
    }
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"Existing output preserved: {path}")

    print("\n=== AURUM-AI | 30m vs 60m V5 EX-ANTE COMPARISON ===")
    signals30, prices30 = load_inputs_v4()
    features = pd.read_parquet(
        FEATURES_PATH,
        columns=["timestamp", "session_date", "segment_id", "bar_in_segment",
                 "close", "sma_5", "ema_10", "volatility_10"],
    )
    common, cohort = prepare_ex_ante_common_signals(signals30, features)
    print("\n=== COHORT AUDIT ===")
    for key, value in cohort.items():
        print(f"{key}: {value}")
    prices = extend_validation_prices(signals30, prices30, PRICES_PATH)
    print(f"Historical validation bars replayed: {len(prices)}")

    outputs = {}
    results = {}
    for horizon in (30, 60):
        print(f"\n=== HORIZON {horizon}m ===")
        decisions, transactions, summary = run_horizon(common, prices, horizon, simulate_model)
        verify_result(decisions, transactions, summary)
        key = f"{horizon}m"
        results[key] = summary
        outputs[f"{key}_decisions"] = decisions
        outputs[f"{key}_transactions"] = transactions

    difference = results["60m"]["net_return_pct"] - results["30m"]["net_return_pct"]
    report = {
        "project": "AURUM-AI", "experiment": "E.3 V5 30m vs 60m ex-ante validation",
        "dataset": "XOMUSUSD Bid 5min, original dataset unchanged",
        "evaluation_split": "validation", "strategy": "CNN1D UP + Logistic UP + close>SMA5>EMA10",
        "cohort_rule": "Same consensus+trend signals satisfying published NYSE 60m session cutoff; NO future-bar availability prefilter",
        "cohort_counts": cohort,
        "selection_times": "Features of bar labelled t known only after bar closes at t+5m; simulated entry t+10m",
        "exit_offsets_signal_minutes": {"30m": 35, "60m": 65},
        "actual_holding_minutes": {"30m": 25, "60m": 55},
        "execution_engine": "V4 historical multi-agent simulator, Risk + persistent Kill Switch + emergency exit",
        "cost_bps_per_side": 10.0, "initial_capital_usd": 10000.0,
        "results": results,
        "return_difference_60m_minus_30m_pp": difference,
        "cash_baseline_return_pct": 0.0,
        "legacy_results_untouched": True, "real_trading": False,
        "test_dataset_used": False, "genuine_walk_forward": False,
        "limitations": [
            "Exploratory validation result; previous tuning/selection was performed on validation.",
            "30m trained classification signals reused for both 30m and 60m exits.",
            "Same 60m-eligible cohort is fair for exits, but omits 30m-only opportunities.",
            "Cohort is selected by published calendar, not future price availability; missing bars are audited during execution.",
            "Full price series is read for REPLAY only; simulator decisions are chronological.",
            "Signal-bar close features are available at t+5m, before entry t+10m; upstream feature-generation leakage not independently audited here.",
            "NYSE early closes and DST modeled by pandas_market_calendars.",
            "Bid-only candles; no Ask spread, true fill dynamics or intrabar stop-loss.",
            "Hypothetical 10 bps per side, idealized Bid-open fill and bar-open risk monitoring.",
            "Emergency liquidation at first observed bar open is not a guaranteed drawdown cap.",
            "Market gaps may produce delayed/overnight fills; inspect any data-interruption exits.",
            "Cash at 0% is a research baseline; Buy & Hold remains to be evaluated consistently.",
            "Do not interpret as independent out-of-sample performance or profitable trading.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, df in outputs.items():
        df.to_csv(paths[key], index=False)
    paths["summary"].write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== V5 RESULT SUMMARY ===")
    for horizon in ("30m", "60m"):
        s = results[horizon]
        print(f"{horizon}: trades={s['completed_trades']}, fees=${s['total_fees_usd']:.2f}, "
              f"return={s['net_return_pct']:+.4f}%, maxDD={s['max_observed_drawdown_pct']:.4f}%, "
              f"system={s['final_system_state']}")
    print(f"60m - 30m: {difference:+.4f} percentage points")
    print(f"Saved to: {OUTPUT_DIR}")
    print("No test dataset, no real orders, older results unchanged.")


if __name__ == "__main__":
    main()
