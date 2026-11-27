"""Tests for validation-only, fixed-trade V5 sensitivity audit."""
import tempfile
import unittest
from pathlib import Path
import json

import numpy as np
import pandas as pd

from backtesting.analyze_cost_sensitivity_v5 import (
    audit_v5_transactions, scenario, build_report, BASE_FEE_BPS,
)


def sample():
    q = 4.0
    entry, exit_ = 100.0, 101.0
    cost = BASE_FEE_BPS / 10000
    buy_fee, sell_fee = entry*q*cost, exit_*q*cost
    cash1 = 10000 - entry*q - buy_fee
    cash2 = cash1 + exit_*q - sell_fee
    trades = pd.DataFrame({
        "timestamp_ms": [1000, 2000],
        "action": ["BUY", "SELL"],
        "price": [entry, exit_], "quantity": [q, q],
        "fee_usd": [buy_fee, sell_fee],
        "cash_change_usd": [-entry*q-buy_fee, exit_*q-sell_fee],
        "realized_pnl_usd": [0, (exit_-entry)*q-buy_fee-sell_fee],
        "cash_after_usd": [cash1, cash2], "holdings_after": [q, 0],
    })
    summary = {"completed_trades": 1, "total_fees_usd": buy_fee+sell_fee,
               "final_equity_usd": cash2, "net_return_pct": (cash2 / 10000 -1)*100}
    return trades, summary


class TestCostSensitivityV5(unittest.TestCase):
    def test_baseline_accounting_exact(self):
        d,s = sample()
        self.assertEqual(audit_v5_transactions(d,s,10000),1)
        r = scenario(d,10000,10,0)
        self.assertAlmostEqual(r["fixed_quantity_final_cash_usd"],s["final_equity_usd"])

    def test_zero_fee_and_slippage(self):
        d,_ = sample()
        r = scenario(d,10000,0,0)
        self.assertAlmostEqual(r["fixed_quantity_final_cash_usd"],10004)
        self.assertAlmostEqual(r["total_fees_usd"],0)

    def test_adverse_slippage_reduces_equity(self):
        d,_ = sample()
        self.assertLess(scenario(d,10000,10,5)["fixed_quantity_final_cash_usd"],
                        scenario(d,10000,10,0)["fixed_quantity_final_cash_usd"])

    def test_higher_fees_reduce_equity(self):
        d,_ = sample()
        self.assertLess(scenario(d,10000,20,0)["fixed_quantity_final_cash_usd"],
                        scenario(d,10000,5,0)["fixed_quantity_final_cash_usd"])

    def test_invalid_costs_rejected(self):
        d,_ = sample()
        for c,slip in [(-1,0),(10,-2),(10,10001),(float("nan"),0)]:
            with self.subTest(c=c,slip=slip),self.assertRaises(ValueError):
                scenario(d,10000,c,slip)

    def test_source_is_unchanged(self):
        d,_ = sample()
        before = d.copy(deep=True)
        scenario(d,10000,10,5)
        pd.testing.assert_frame_equal(d,before)

    def test_sell_without_buy_rejected(self):
        d,s = sample()
        d.loc[0,"action"]="SELL"
        with self.assertRaises(ValueError):
            audit_v5_transactions(d,s,10000)

    def test_fee_tampering_rejected(self):
        d,s = sample()
        d.loc[1,"fee_usd"]+=.1
        with self.assertRaises(ValueError):
            audit_v5_transactions(d,s,10000)

    def test_unbalanced_quantity_rejected(self):
        d,s = sample()
        d.loc[1,"quantity"]+=.01
        with self.assertRaises(ValueError):
            audit_v5_transactions(d,s,10000)

    def test_nonchronological_events_rejected(self):
        d,s = sample()
        d.loc[1,"timestamp_ms"] = 500
        with self.assertRaises(ValueError):
            audit_v5_transactions(d,s,10000)

    def test_validation_report_rejects_test_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/"comparison_summary_v5.json").write_text(json.dumps({"evaluation_split":"test","test_dataset_used":True}))
            with self.assertRaises(ValueError):
                build_report(p)

    def test_missing_fields_rejected(self):
        d,s=sample()
        with self.assertRaises(ValueError):
            audit_v5_transactions(d.drop(columns="fee_usd"),s,10000)


if __name__ == "__main__":
    unittest.main()
