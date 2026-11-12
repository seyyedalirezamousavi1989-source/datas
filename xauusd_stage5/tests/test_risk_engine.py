import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fake_mt5 import FakeMT5, gold_info
from test_broker_context import mk, call
from broker_context.risk_engine import RiskPolicy, AccountState


def sz(t, state=None, policy=None, **kw):
    a = dict(side="long", entry_price=2000.0, stop_price=1995.0, account_state=state or {"balance": 10000, "equity": 10000, "margin_used": 0})
    if policy is not None:
        a["policy"] = policy
    a.update(kw)
    return call(t, "calculate_position_size", **a)


class T(unittest.TestCase):
    def setUp(self):
        self.t, *_ = mk(FakeMT5())

    def test_fixed_fractional_exact(self):
        r = sz(self.t, policy={"risk_pct": 1.0})["data"]
        # risk 100 USD, stop 5.0 -> 500 USD/lot -> 0.2 lots
        self.assertAlmostEqual(r["lots"], 0.20)
        self.assertAlmostEqual(r["risk_money"], 100.0)
        self.assertEqual(r["binding_constraint"], "policy_risk_pct")

    def test_balance_vs_equity_base(self):
        st = {"balance": 10000, "equity": 8000, "margin_used": 0}
        e = sz(self.t, st, {"risk_pct": 1.0, "base": "equity"})["data"]["lots"]
        b = sz(self.t, st, {"risk_pct": 1.0, "base": "balance"})["data"]["lots"]
        self.assertAlmostEqual(e, 0.16)
        self.assertAlmostEqual(b, 0.20)

    def test_rounds_down_to_step_and_min_lot_block(self):
        r = sz(self.t, {"balance": 10000, "equity": 10000, "margin_used": 0}, {"risk_pct": 1.0}, stop_price=1993.0)["data"]  # 100/700 = 0.1428
        self.assertAlmostEqual(r["lots"], 0.14)
        small = sz(self.t, {"balance": 100, "equity": 100, "margin_used": 0}, {"risk_pct": 0.5})
        self.assertEqual(small["status"], "blocked")
        self.assertEqual(small["data"]["lots"], 0.0)

    def test_incomplete_without_account_state(self):
        r = sz(self.t, {"margin_used": 0})
        self.assertEqual(r["status"], "incomplete")
        self.assertEqual(r["data"]["lots"], 0.0)

    def test_daily_loss_limit_reduces_then_blocks(self):
        st = {"balance": 10000, "equity": 10000, "margin_used": 0, "realized_loss_today": 150}
        r = sz(self.t, st, {"risk_pct": 1.0, "max_daily_loss_pct": 2.0})["data"]
        self.assertEqual(r["binding_constraint"], "daily_loss")
        self.assertAlmostEqual(r["risk_money"], 50.0, places=2)
        st["realized_loss_today"] = 200
        self.assertEqual(sz(self.t, st, {"risk_pct": 1.0, "max_daily_loss_pct": 2.0})["status"], "blocked")

    def test_weekly_and_aggregate(self):
        st = {"balance": 10000, "equity": 10000, "margin_used": 0, "realized_loss_week": 390}
        r = sz(self.t, st, {"risk_pct": 1.0, "max_weekly_loss_pct": 4.0})["data"]
        self.assertEqual(r["binding_constraint"], "weekly_loss")
        st = {"balance": 10000, "equity": 10000, "margin_used": 0, "open_positions": [{"side": "long", "lots": 0.3, "entry": 2000, "stop": 1995}]}
        r = sz(self.t, st, {"risk_pct": 1.0, "max_aggregate_open_risk_pct": 2.0, "max_daily_loss_pct": 9, "max_weekly_loss_pct": 9})["data"]
        self.assertEqual(r["binding_constraint"], "aggregate_open_risk")
        self.assertAlmostEqual(r["risk_money"], 50.0, places=2)   # 200 budget - 150 open

    def test_max_positions_and_same_direction(self):
        pos = [{"side": "long", "lots": 0.01, "entry": 2000, "stop": 1995}] * 2
        st = {"balance": 10000, "equity": 10000, "margin_used": 0, "open_positions": pos}
        self.assertEqual(sz(self.t, st, {"max_same_direction_positions": 2})["status"], "blocked")
        self.assertNotEqual(sz(self.t, st, {"max_same_direction_positions": 2}, side="short", entry_price=2000.0, stop_price=2005.0)["status"], "blocked")
        pos3 = pos + [{"side": "short", "lots": 0.01, "entry": 2000, "stop": 2005}]
        self.assertEqual(sz(self.t, dict(st, open_positions=pos3), {"max_positions": 3, "max_same_direction_positions": 5})["status"], "blocked")

    def test_drawdown_block(self):
        st = {"balance": 9000, "equity": 8900, "margin_used": 0, "peak_equity": 10000}
        self.assertEqual(sz(self.t, st, {"max_drawdown_pct": 10.0})["status"], "blocked")
        self.assertNotEqual(sz(self.t, st, {"max_drawdown_pct": 20.0})["status"], "blocked")

    def test_volatility_and_spread_reduction_only_shrinks(self):
        base = sz(self.t, policy={"risk_pct": 1.0})["data"]["lots"]
        v = sz(self.t, policy={"risk_pct": 1.0}, atr_ratio=2.0)["data"]
        self.assertLess(v["lots"], base)
        self.assertTrue(v["reductions"])
        w = sz(self.t, policy={"risk_pct": 1.0}, spread_price=0.8, atr=3.0)["data"]
        self.assertLess(w["lots"], base)
        blocked = sz(self.t, policy={"risk_pct": 1.0}, spread_price=2.0, atr=3.0)
        self.assertEqual(blocked["status"], "blocked")

    def test_margin_safety_reduces_lots(self):
        st = {"balance": 10000, "equity": 10000, "margin_used": 0}
        open_ = sz(self.t, st, {"risk_pct": 1.0, "max_margin_usage_pct": 100, "min_margin_level_pct": 1})["data"]
        # margin = lots*100*2000/500 = lots*400
        self.assertAlmostEqual(open_["margin"]["required"], open_["lots"] * 400)
        tight = sz(self.t, st, {"risk_pct": 1.0, "max_margin_usage_pct": 0.5, "min_margin_level_pct": 1})["data"]
        self.assertLess(tight["lots"], open_["lots"])
        self.assertEqual(tight["binding_constraint"], "margin_safety")
        self.assertLessEqual(tight["margin"]["required"], 50 + 1e-6)

    def test_margin_unverified_when_state_missing(self):
        r = sz(self.t, {"balance": 10000, "equity": 10000})
        self.assertEqual(r["status"], "complete_with_gaps")
        self.assertTrue(any("margin_used" in w for w in r["warnings"]))

    def test_no_martingale_possible(self):
        for k in ("martingale_factor", "loss_recovery", "double_after_loss", "escalate_after_loss"):
            with self.assertRaises(ValueError):
                RiskPolicy.from_dict({k: 2})
        with self.assertRaises(ValueError):
            RiskPolicy.from_dict({"vol_reduce_factor": 1.5})
        r = sz(self.t, policy={"martingale": True})
        self.assertEqual(r["status"], "invalid_arguments")
        # after losing, size never increases
        a = sz(self.t, {"balance": 10000, "equity": 10000, "margin_used": 0})["data"]["lots"]
        b = sz(self.t, {"balance": 9800, "equity": 9800, "margin_used": 0, "realized_loss_today": 200})["data"]
        self.assertLessEqual(b["lots"], a)

    def test_non_usd_account_requires_fx_for_margin(self):
        t, *_ = mk(FakeMT5(currency="EUR", symbols={"XAUUSD": gold_info(trade_tick_value=0.9)}))
        r = sz(t, {"balance": 10000, "equity": 10000, "margin_used": 0})
        self.assertTrue(any("margin" in w for w in r["warnings"]))
        r2 = sz(t, {"balance": 10000, "equity": 10000, "margin_used": 0}, fx_rate_margin_to_account=0.9)
        self.assertIsNotNone(r2["data"]["margin"]["required"])

    def test_wrong_stop_side(self):
        self.assertEqual(sz(self.t, stop_price=2005.0)["status"], "invalid")

    def test_portfolio_validation(self):
        st = {"balance": 10000, "equity": 9000, "peak_equity": 10000, "realized_loss_today": 250,
              "open_positions": [{"side": "long", "lots": 0.5, "entry": 2000, "stop": 1995}]}
        r = call(self.t, "validate_portfolio_risk", account_state=st, policy={"max_daily_loss_pct": 2.0})
        self.assertEqual(r["status"], "violation")
        self.assertIn("daily_loss_pct", r["data"]["violations"])
        self.assertAlmostEqual(r["data"]["exposure"]["net_lots"], 0.5)

    def test_mt5_account_state_read(self):
        m = FakeMT5()
        import types
        base = m.account_info
        m.account_info = lambda: types.SimpleNamespace(**dict(vars(base()), balance=1234.5, equity=1200.0, margin=10.0, margin_free=1190.0, margin_level=12000.0))
        t, *_ = mk(m)
        v = t.provider.fetch_account_state()
        self.assertEqual(v["values"]["balance"], 1234.5)
        self.assertNotIn("login", v["values"])


if __name__ == "__main__":
    unittest.main()
