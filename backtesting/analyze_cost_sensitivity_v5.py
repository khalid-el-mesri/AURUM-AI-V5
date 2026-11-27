"""AURUM-AI — E.4 V5 fixed-execution cost and slippage sensitivity.

Exploratory VALIDATION sensitivity, using ONLY recorded 30m/60m V5 trades.
This is NOT a full strategy rerun: forecasts, orders, quantities, risk limits
and kill-switch decisions are held fixed at their V5 historical values.

Historical execution reference is XOM 5m Bid-open, not an actual Ask quote.
Adverse hypothetical slippage: BUY price increases, SELL price decreases.
Cost bps is a hypothetical per-side *additional* fee on repriced notionals.
No training, test-split access, real trading, or source file modifications.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "reports" / "backtesting" / "combined_holding_30m_60m_v5_calendar"
OUTPUT_DIR = ROOT / "reports" / "backtesting" / "cost_sensitivity_v5_fixed_trades"
FEE_GRID_BPS = (0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0)
SLIPPAGE_GRID_BPS = (0.0, 2.0, 5.0, 10.0)
BASE_FEE_BPS = 10.0
TOL = 1e-6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_v5_transactions(frame: pd.DataFrame, summary: dict, capital: float) -> int:
    """Fail closed unless source matches V5 accounting and trade sequencing."""
    required = {
        "timestamp_ms", "action", "price", "quantity", "fee_usd",
        "cash_change_usd", "realized_pnl_usd", "cash_after_usd", "holdings_after",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing V5 transaction fields: {missing}")
    if len(frame) == 0 or len(frame) % 2:
        raise ValueError("Expected a nonempty, even number of BUY/SELL events")
    d = frame.reset_index(drop=True)
    numerical = ["timestamp_ms", "price", "quantity", "fee_usd", "cash_change_usd",
                 "realized_pnl_usd", "cash_after_usd", "holdings_after"]
    for col in numerical:
        values = pd.to_numeric(d[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite or invalid {col}")
    if not np.all(np.diff(d["timestamp_ms"].to_numpy(dtype=np.int64)) > 0):
        raise ValueError("Transaction timestamps must increase strictly")
    actions = d.action.to_numpy()
    if not ((actions[::2] == "BUY").all() and (actions[1::2] == "SELL").all()):
        raise ValueError("Expected alternating BUY then SELL (flat-to-flat trades)")
    prices = d.price.to_numpy(dtype=float)
    qty = d.quantity.to_numpy(dtype=float)
    fees = d.fee_usd.to_numpy(dtype=float)
    if (prices <= 0).any() or (qty <= 0).any() or (fees < 0).any():
        raise ValueError("Invalid price, quantity, or fees")
    if not np.allclose(qty[::2], qty[1::2], rtol=0, atol=1e-8):
        raise ValueError("BUY/SELL quantities mismatch")
    signed_qty = np.where(actions == "BUY", qty, -qty)
    if not np.allclose(np.cumsum(signed_qty), d.holdings_after, rtol=0, atol=1e-7):
        raise ValueError("Recorded holdings are inconsistent")
    if not np.allclose(d.holdings_after.to_numpy(dtype=float)[1::2], 0, rtol=0, atol=1e-7):
        raise ValueError("Every trade must end with zero holdings")
    computed_fee = prices * qty * BASE_FEE_BPS / 10000.0
    if not np.allclose(computed_fee, fees, rtol=0, atol=TOL):
        raise ValueError("Recorded V5 fees do not match 10bps per side")
    cash_change = np.where(actions == "BUY", -1.0, 1.0) * prices * qty - fees
    if not np.allclose(cash_change, d.cash_change_usd, rtol=0, atol=TOL):
        raise ValueError("Recorded cash changes inconsistent with prices and fees")
    if not np.allclose(capital + np.cumsum(cash_change), d.cash_after_usd, rtol=0, atol=TOL):
        raise ValueError("Recorded cash balances inconsistent")
    pnl_sell = (prices[1::2] - prices[::2]) * qty[::2] - fees[::2] - fees[1::2]
    if not np.allclose(pnl_sell, d.realized_pnl_usd.to_numpy(dtype=float)[1::2], rtol=0, atol=TOL):
        raise ValueError("Recorded realized P&L inconsistent")
    if not np.allclose(d.realized_pnl_usd.to_numpy(dtype=float)[::2], 0, rtol=0, atol=TOL):
        raise ValueError("BUY realized P&L must be zero")
    trades = len(d) // 2
    if trades != int(summary["completed_trades"]):
        raise ValueError("Trade count differs from V5 summary")
    if not np.isclose(float(fees.sum()), float(summary["total_fees_usd"]), rtol=0, atol=TOL):
        raise ValueError("Fees differ from V5 summary")
    if not np.isclose(float(d.cash_after_usd.iloc[-1]), float(summary["final_equity_usd"]), rtol=0, atol=TOL):
        raise ValueError("Final capital differs from V5 summary")
    if not np.isclose(float(summary["net_return_pct"]),
                      (float(d.cash_after_usd.iloc[-1]) / capital - 1.0) * 100.0,
                      rtol=0, atol=1e-7):
        raise ValueError("Net return differs from V5 summary")
    return trades


def scenario(frame: pd.DataFrame, capital: float, fee_bps: float, slip_bps: float) -> dict:
    """Reprice the SAME quantities/trades with adverse per-side slippage/fees.

    This is a frozen-trade, non-reoptimized stress analysis. It does NOT replay
    risk, modify sizing, or infer achievable returns under changed assumptions.
    """
    if (not np.isfinite(fee_bps) or not np.isfinite(slip_bps)
            or fee_bps < 0 or slip_bps < 0 or capital <= 0):
        raise ValueError("Invalid cost, slippage, or capital")
    d = frame.reset_index(drop=True)
    if len(d) == 0 or len(d) % 2:
        raise ValueError("Invalid transaction sequence")
    actions = d.action.to_numpy()
    prices = d.price.to_numpy(dtype=float)
    qty = d.quantity.to_numpy(dtype=float)
    factor = np.where(actions == "BUY", 1 + slip_bps / 10000, 1 - slip_bps / 10000)
    if np.any(factor <= 0):
        raise ValueError("Slippage makes synthetic execution price non-positive")
    stressed_price = prices * factor
    fee = stressed_price * qty * fee_bps / 10000
    change = np.where(actions == "BUY", -1.0, 1.0) * stressed_price * qty - fee
    cash_path = capital + np.cumsum(change)
    realized = change[::2] + change[1::2]
    final_capital = float(cash_path[-1])
    return {
        "trades": int(len(d) // 2),
        "fees_bps_per_side": float(fee_bps),
        "adverse_slippage_bps_per_side": float(slip_bps),
        "total_fees_usd": float(fee.sum()),
        "realized_pnl_usd": float(realized.sum()),
        "fixed_quantity_final_cash_usd": final_capital,
        "fixed_quantity_return_pct": float(100 * (final_capital / capital - 1)),
        "win_rate_pct": float(100 * (realized > 0).mean()),
        "minimum_synthetic_cash_usd": float(cash_path.min()),
        "sufficient_cash_at_each_recorded_trade": bool((cash_path >= -TOL).all()),
    }


def build_report(source: Path = SOURCE_DIR) -> tuple[pd.DataFrame, dict]:
    summary_path = source / "comparison_summary_v5.json"
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    if report.get("evaluation_split") != "validation" or report.get("test_dataset_used") is not False:
        raise ValueError("Expected validation-only V5 report with unused test split")
    if float(report["cost_bps_per_side"]) != BASE_FEE_BPS:
        raise ValueError("Unexpected reference fee assumptions")
    capital = float(report["initial_capital_usd"])
    if not np.isfinite(capital) or capital <= 0:
        raise ValueError("Invalid initial capital")
    results = []
    hashes = {summary_path.name: sha256(summary_path)}
    validations = {}
    for horizon in ("30m", "60m"):
        path = source / f"{horizon}_transactions_v5.csv"
        hashes[path.name] = sha256(path)
        frame = pd.read_csv(path)
        trades = audit_v5_transactions(frame, report["results"][horizon], capital)
        validations[horizon] = {"trades": trades, "reference_total_fees": report["results"][horizon]["total_fees_usd"],
                                "reference_final_equity": report["results"][horizon]["final_equity_usd"]}
        reference = scenario(frame, capital, BASE_FEE_BPS, 0)
        if not np.isclose(reference["fixed_quantity_final_cash_usd"],
                          report["results"][horizon]["final_equity_usd"], rtol=0, atol=TOL):
            raise RuntimeError("Zero-slippage scenario did not reproduce V5 cash")
        for cost in FEE_GRID_BPS:
            for slip in SLIPPAGE_GRID_BPS:
                results.append({"horizon": horizon, **scenario(frame, capital, cost, slip)})
    meta = {
        "project": "AURUM-AI", "analysis": "E.4 V5 frozen-trade cost and slippage sensitivity",
        "source_split": "validation", "source_report": str(summary_path), "sha256": hashes,
        "source_audit": validations, "initial_capital_usd": capital,
        "historical_price": "XOMUSUSD 5m Bid open", "baseline_fee_bps_per_side": BASE_FEE_BPS,
        "scenarios_fee_bps_per_side": list(FEE_GRID_BPS),
        "scenarios_adverse_slippage_bps_per_side": list(SLIPPAGE_GRID_BPS),
        "recomputed_trades": False, "recomputed_risk": False, "recomputed_positions": False,
        "test_dataset_used": False, "real_orders": False,
        "limitations": [
            "FIXED executed trades and quantities from V5; no re-optimization, position resizing or agent/risk recomputation.",
            "Hypothetical per-side fees and adverse price slippage; original Bid-only data contain no Ask or true spread.",
            "Synthetic capital outcomes are accounting sensitivity, NOT results of a fresh backtest.",
            "Fixed quantities may become infeasible if changed costs exhaust available cash; see scenario flag.",
            "Risk/kill-switch timestamps, entry counts and position sizes can change under a true strategy rerun.",
            "Additional intrabar slippage, market impact, queue priority and live market fills are unknown.",
            "Validation was previously used for strategy selection; not independent out-of-sample evidence.",
        ],
    }
    return pd.DataFrame(results), meta


def main():
    print("\n=== AURUM-AI E.4 | COST + SLIPPAGE SENSITIVITY (V5 FROZEN TRADES) ===")
    # Never overwrite research outputs. Verify the source report before saving anything.
    output_csv = OUTPUT_DIR / "cost_slippage_grid_v5.csv"
    output_json = OUTPUT_DIR / "assumptions_and_source_audit.json"
    for output in (output_csv, output_json):
        if output.exists():
            raise FileExistsError(f"Previous audit preserved: {output}")
    table, audit = build_report()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    for horizon in ("30m", "60m"):
        chosen = table.loc[table.horizon.eq(horizon) & table.adverse_slippage_bps_per_side.eq(0)]
        print(f"\n=== {horizon.upper()}: FEES (0 bps SLIPPAGE, FIXED TRADES) ===")
        print(chosen[["fees_bps_per_side", "trades", "fixed_quantity_return_pct", "total_fees_usd"]].round(4).to_string(index=False))
        chosen = table.loc[table.horizon.eq(horizon) & table.fees_bps_per_side.eq(10)]
        print(f"\n=== {horizon.upper()}: SLIPPAGE (10 bps FEES PER SIDE, FIXED TRADES) ===")
        print(chosen[["adverse_slippage_bps_per_side", "fixed_quantity_return_pct", "minimum_synthetic_cash_usd"]].round(4).to_string(index=False))
    print(f"\nSaved CSV: {output_csv}")
    print(f"Saved source audit: {output_json}")
    print("WARNING: Frozen trades, NOT full portfolio/risk resimulation. Test split untouched.")


if __name__ == "__main__":
    main()
