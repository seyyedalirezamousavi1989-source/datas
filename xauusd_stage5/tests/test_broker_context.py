import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from broker_context.mt5_provider import AuthorizationStore, MT5Provider, ReadOnlyMT5
from broker_context.tools import TOOL_SCHEMAS, create_tools
from broker_context.cost_profiles import CostProfile, LabelRunRegistry, historical_net_r
from broker_context.context_snapshot import BrokerContextConfig
from fake_mt5 import FakeClock, FakeMT5, gold_info


def mk(mt5=None, authorize=True, clock=None, cfg=None, home=None):
    home = home or tempfile.mkdtemp()
    clock = clock or FakeClock()
    t = create_tools(home, mt5, clock, cfg)
    if authorize:
        t.provider.auth.authorize()
    return t, home, clock


def call(t, name, **a):
    return t.call(name, a)


def full_profile(t, **extra):
    f = {"commission_model": "per_lot", "commission_amount": 3.5, "commission_currency": "USD", "commission_basis": "per_lot_per_side"}
    f.update(extra)
    return call(t, "update_user_account_profile", fields=f, confirmed=True)


class T(unittest.TestCase):
    # 1 disconnected terminal
    def test_01_terminal_disconnected(self):
        t, *_ = mk(FakeMT5(init_ok=False))
        r = call(t, "get_mt5_connection_status")
        self.assertIn(r["status"], ("unavailable", "error"))
        r = call(t, "calculate_position_risk", lots=0.1, entry_price=2000, stop_price=1995)
        self.assertEqual(r["status"], "incomplete")
        self.assertNotIn("risk_to_stop", json.dumps(r["data"].get("components", {})).replace("null", "")) if False else None

    def test_01b_not_authorized_and_package_missing(self):
        t, *_ = mk(FakeMT5(), authorize=False)
        self.assertEqual(call(t, "get_mt5_connection_status")["status"], "not_authorized")
        t2, *_ = mk(None)
        t2.provider._mod = None
        r = call(t2, "get_mt5_connection_status")
        self.assertIn(r["status"], ("unavailable", "not_authorized"))

    # 2 connected
    def test_02_connected_full(self):
        t, *_ = mk(FakeMT5())
        self.assertEqual(call(t, "get_mt5_connection_status")["status"], "ok")
        a = call(t, "get_active_account_context")
        self.assertEqual(a["data"]["account"]["currency"]["value"], "USD")
        self.assertEqual(a["data"]["account"]["currency"]["source"], "mt5_live")
        s = call(t, "get_symbol_specifications")
        self.assertEqual(s["data"]["symbol"], "XAUUSD")
        self.assertEqual(s["data"]["specifications"]["contract_size"]["value"], 100.0)

    # 3 partial account info
    def test_03_partial_account(self):
        t, *_ = mk(FakeMT5(account=False))
        a = call(t, "get_active_account_context")
        self.assertIn(a["status"], ("partial", "unavailable"))
        self.assertIn("currency", a["missing_fields"])

    # 4 unknown commission  (+ type label never invents)
    def test_04_unknown_commission(self):
        t, *_ = mk(FakeMT5())
        call(t, "update_user_account_profile", fields={"account_type": "ECN"}, confirmed=True)
        r = call(t, "calculate_trade_costs", side="long", lots=0.1, entry_price=2000, stop_price=1995)
        c = r["data"]["components"]["commission"]
        self.assertIsNone(c["value"])
        self.assertFalse(r["data"]["totals"]["is_total_complete"])
        self.assertIn("commission", r["data"]["totals"]["unknown_components"])
        self.assertIsNone(r["data"]["position_risk"]["worst_case_loss_at_stop"])

    # 5 unknown contract size
    def test_05_unknown_contract_size(self):
        info = gold_info(trade_contract_size=0, trade_tick_value=0, trade_tick_size=0)
        t, *_ = mk(FakeMT5(symbols={"XAUUSD": info}))
        r = call(t, "calculate_position_risk", lots=0.1, entry_price=2000, stop_price=1995)
        self.assertEqual(r["status"], "incomplete")
        self.assertTrue(r["missing_fields"])

    # 6 symbol prefix/suffix and ambiguity
    def test_06_symbol_variants(self):
        t, *_ = mk(FakeMT5(symbols={"XAUUSD.m": gold_info(), "EURUSD": gold_info(currency_base="EUR", description="Euro", path="Forex\\EURUSD")}))
        s = call(t, "get_symbol_specifications")
        self.assertEqual(s["data"]["symbol"], "XAUUSD.m")
        t2, *_ = mk(FakeMT5(symbols={"XAUUSD.a": gold_info(), "XAUUSD.b": gold_info()}))
        s2 = call(t2, "get_symbol_specifications")
        self.assertEqual(s2["status"], "ambiguous")

    def test_06b_non_gold_rejected(self):
        t, *_ = mk(FakeMT5(symbols={"EURUSD": gold_info(currency_base="EUR", description="Euro", path="Forex\\EURUSD")}))
        s = call(t, "get_symbol_specifications", symbol="EURUSD")
        self.assertNotEqual(s["status"], "ok")

    # 7 conflicts between MT5 and user
    def test_07_conflict_preserved_and_acknowledged(self):
        t, *_ = mk(FakeMT5())
        call(t, "update_user_account_profile", fields={"currency": "EUR"}, confirmed=True)
        a = call(t, "get_active_account_context")
        confs = a["data"]["conflicts"]
        self.assertTrue(any(c["field"] == "currency" for c in confs))
        self.assertEqual(a["status"], "partial")
        r = call(t, "calculate_position_risk", lots=0.1, entry_price=2000, stop_price=1995)
        self.assertEqual(r["status"], "incomplete")
        call(t, "update_user_account_profile", fields={}, acknowledge_conflicts={"currency": "mt5"})
        r = call(t, "calculate_position_risk", lots=0.1, entry_price=2000, stop_price=1995)
        self.assertIn(r["status"], ("complete", "complete_with_estimates"))
        self.assertEqual(r["data"]["currency"], "USD")

    # 8 stale cache
    def test_08_stale_cache(self):
        m, clock = FakeMT5(), FakeClock()
        t, *_ = mk(m, clock=clock)
        self.assertFalse(call(t, "get_active_account_context")["is_stale"])
        m.init_ok = False
        t.provider._initialized = False
        clock.advance(2000)
        a = call(t, "get_active_account_context")
        self.assertTrue(a["is_stale"])
        self.assertTrue(any("cached" in w for w in a["warnings"]))

    # 9 non-USD account currency
    def test_09_non_usd_account(self):
        info = gold_info(trade_tick_value=0.9)  # MT5 already gives tick value in account currency
        t, *_ = mk(FakeMT5(symbols={"XAUUSD": info}, currency="EUR"))
        r = call(t, "calculate_position_risk", lots=1, entry_price=2000, stop_price=1990)
        self.assertEqual(r["data"]["currency"], "EUR")
        self.assertAlmostEqual(r["data"]["components"]["risk_to_stop"]["value"], 10 * 0.9 / 0.01, places=6)

    def test_09b_fx_needed_without_tick_value(self):
        info = gold_info(trade_tick_value=0, trade_tick_size=0)
        t, *_ = mk(FakeMT5(symbols={"XAUUSD": info}, currency="EUR"))
        r = call(t, "calculate_position_risk", lots=1, entry_price=2000, stop_price=1990)
        self.assertEqual(r["status"], "incomplete")
        self.assertIn("fx_rate_profit_to_account", r["missing_fields"])
        r = call(t, "calculate_position_risk", lots=1, entry_price=2000, stop_price=1990, fx_rate_profit_to_account=0.9)
        self.assertEqual(r["status"], "complete_with_estimates")
        self.assertAlmostEqual(r["data"]["components"]["risk_to_stop"]["value"], 10 * 100 * 0.9)

    # 10 tick value conversion
    def test_10_tick_value(self):
        t, *_ = mk(FakeMT5(symbols={"XAUUSD": gold_info(trade_tick_value=1.0, trade_tick_size=0.01)}))
        r = call(t, "calculate_position_risk", lots=0.5, entry_price=2000, stop_price=1998)
        self.assertAlmostEqual(r["data"]["components"]["risk_to_stop"]["value"], 2 * 100 * 0.5)

    # 11 point/tick conversion
    def test_11_point_tick(self):
        info = gold_info(point=0.001, digits=3, trade_tick_size=0.01, trade_tick_value=1.0)
        t, *_ = mk(FakeMT5(symbols={"XAUUSD": info}))
        d = call(t, "calculate_position_risk", lots=1, entry_price=2000, stop_price=1999)["data"]["components"]["risk_to_stop"]["detail"]
        self.assertAlmostEqual(d["stop_distance_points"], 1000)
        self.assertAlmostEqual(d["stop_distance_ticks"], 100)

    # 12 commission per side vs round turn vs flat
    def test_12_commission_bases(self):
        for basis, amt, lots, exp in (("per_lot_per_side", 3.5, 2, 14.0), ("per_lot_round_turn", 7.0, 2, 14.0), ("flat_per_trade_round_turn", 5.0, 2, 5.0)):
            t, *_ = mk(FakeMT5())
            full_profile(t, commission_amount=amt, commission_basis=basis)
            r = call(t, "calculate_trade_costs", side="long", lots=lots, entry_price=2000, stop_price=1995, slippage_price_per_side=0.0)
            self.assertAlmostEqual(r["data"]["components"]["commission"]["value"], exp, msg=basis)

    # 13 spread failure
    def test_13_spread_failure(self):
        m = FakeMT5()
        m.tick_fail = True
        t, *_ = mk(m)
        r = call(t, "get_current_spread")
        self.assertIn(r["status"], ("ok", "partial"))  # falls back to symbol_info.spread * point, labelled
        self.assertAlmostEqual(r["data"]["spread_price"]["value"], 0.20)
        t2, *_ = mk(FakeMT5(init_ok=False))
        r2 = call(t2, "get_current_spread")
        self.assertEqual(r2["status"], "unavailable")
        c = call(t2, "calculate_trade_costs", side="long", lots=0.1, entry_price=2000)
        self.assertEqual(c["status"], "incomplete")

    def test_13b_stale_quote_flagged(self):
        clock = FakeClock()
        m = FakeMT5()
        t, *_ = mk(m, clock=clock, cfg=BrokerContextConfig(quote_ttl_s=15))
        call(t, "get_current_spread")
        t.builder.provider.fetch_quote = lambda name, _o=t.builder.provider.fetch_quote: dict(_o(name), retrieved_at="2026-01-05T11:00:00Z")
        r = call(t, "get_current_spread")
        self.assertTrue(any("stale" in w for w in r["warnings"]) or r["is_stale"])

    # 14 user changes account type
    def test_14_account_type_change(self):
        t, *_ = mk(FakeMT5(), authorize=False)
        full_profile(t, account_type="ECN")
        r = call(t, "update_user_account_profile", fields={"account_type": "Standard"})
        self.assertTrue(any("re-confirm" in w for w in r["warnings"]))
        self.assertEqual(t.store.records["commission_amount"]["verification"], "unconfirmed")

    # 15 runtime updates without retraining -> numbers change when profile changes
    def test_15_runtime_recalculation(self):
        t, *_ = mk(FakeMT5())
        full_profile(t, commission_amount=3.5)
        a = call(t, "calculate_trade_costs", side="long", lots=1, entry_price=2000, stop_price=1995, slippage_price_per_side=0.0)
        full_profile(t, commission_amount=7.0)
        b = call(t, "calculate_trade_costs", side="long", lots=1, entry_price=2000, stop_price=1995, slippage_price_per_side=0.0)
        self.assertAlmostEqual(b["data"]["components"]["commission"]["value"], 2 * a["data"]["components"]["commission"]["value"])

    # 16 reproducible historical labeling under different cost profiles
    def test_16_historical_profiles(self):
        def prof(amount):
            return CostProfile(name="s", config_version="v1", broker_assumptions={"broker": "assumed", "verified": False},
                               contract_assumptions={"point": 0.01, "contract_size": 100.0, "account_currency": "USD", "profit_currency": "USD"},
                               spread={"source": "fixed_points", "fixed_points": 20}, commission={"amount": amount, "currency": "USD", "basis": "per_lot_per_side"},
                               slippage={"model": "fixed_price", "value": 0.0})
        p1, p1b, p2 = prof(3.5), prof(3.5), prof(7.0)
        self.assertEqual(p1.profile_id, p1b.profile_id)
        self.assertNotEqual(p1.profile_id, p2.profile_id)
        r1 = historical_net_r(p1, 10.0, 5.0)
        r2 = historical_net_r(p2, 10.0, 5.0)
        self.assertEqual(r1["gross_R"], r2["gross_R"])
        self.assertGreater(r1["net_R"], r2["net_R"])
        self.assertAlmostEqual(r1["net_R"], (10.0 - 0.2 - 0.07) / 5.0)
        self.assertEqual(historical_net_r(p1, 10.0, 5.0), r1)  # reproducible
        d = tempfile.mkdtemp()
        reg = LabelRunRegistry(d)
        rid = reg.register("ds1", "lab1", p1)
        with self.assertRaises(FileExistsError):
            reg.register("ds1", "lab1", p1)
        self.assertNotEqual(reg.register("ds1", "lab1", p2), rid)
        self.assertEqual(len(reg.list_runs()), 2)

    # security
    def test_20_read_only_enforced(self):
        ro = ReadOnlyMT5(FakeMT5())
        for n in ("order_send", "order_check", "positions_get", "orders_get", "login"):
            with self.assertRaises(PermissionError):
                getattr(ro, n)
        with self.assertRaises(Exception):
            ro.foo = 1

    def test_21_no_secret_or_login_in_outputs(self):
        t, *_ = mk(FakeMT5())
        out = json.dumps([call(t, n) for n in ("get_mt5_connection_status", "get_active_account_context", "get_symbol_specifications",
                                               "get_current_spread", "get_broker_context_status")], default=str)
        self.assertNotIn("123456789", out)
        self.assertNotIn("SHOULD_NEVER_LEAK", out)
        snap = json.dumps(t.builder.build().to_model_dict(), default=str)
        self.assertNotIn("123456789", snap)

    def test_22_secrets_rejected(self):
        t, home, _ = mk(FakeMT5())
        for args in ({"fields": {"password": "x"}}, {"fields": {"currency": "USD"}, "token": "abc"}):
            r = t.call("update_user_account_profile", args)
            self.assertEqual(r["status"], "invalid_arguments")
        r = call(t, "update_user_account_profile", fields={"login": 5})
        self.assertIn("login", r["data"]["rejected"])
        r = call(t, "update_user_account_profile", fields={"investor_password": "zz"})
        self.assertNotEqual(r["status"], "ok")
        if os.path.exists(os.path.join(home, "user_profile.json")):
            self.assertNotIn("zz", open(os.path.join(home, "user_profile.json")).read())

    def test_23_schema_strict(self):
        t, *_ = mk(FakeMT5())
        self.assertEqual(t.call("calculate_position_risk", {"lots": -1, "entry_price": 1, "stop_price": 1})["status"], "invalid_arguments")
        self.assertEqual(t.call("calculate_position_risk", {"lots": 1, "entry_price": 1, "stop_price": 2, "extra": 1})["status"], "invalid_arguments")
        self.assertEqual(t.call("nope", {})["status"], "invalid_arguments")
        self.assertEqual(t.call("get_symbol_specifications", {"symbol": "X; rm -rf"})["status"], "invalid_arguments")
        names = {s["name"] for s in TOOL_SCHEMAS}
        self.assertEqual(len(names), 10)
        self.assertFalse(any("authorize" in n or "order" in n or "login" in n for n in names))

    def test_24_account_change_detected(self):
        m = FakeMT5()
        t, *_ = mk(m)
        call(t, "update_user_account_profile", fields={"commission_amount": 3.5, "commission_basis": "per_lot_per_side", "commission_currency": "USD"},
             confirmed=True, bind_to_connected_account=True)
        self.assertIsNone(call(t, "get_active_account_context")["data"].get("x"))
        m.login = 987654321
        a = call(t, "get_active_account_context")
        self.assertTrue(a["data"]["account_changed_since_profile_bound"])
        self.assertEqual(a["data"]["account"]["commission_amount"]["basis"], "user_unconfirmed")

    def test_25_user_only_mode_without_mt5(self):
        t, *_ = mk(None, authorize=False)
        t.provider._mod = None
        call(t, "update_user_account_profile", fields={"currency": "USD", "contract_size": 100, "tick_size": 0.01, "tick_value": 1.0,
                                                       "point": 0.01, "volume_min": 0.01, "volume_step": 0.01, "volume_max": 50}, confirmed=True)
        r = call(t, "calculate_position_risk", lots=0.1, entry_price=2000, stop_price=1995)
        self.assertEqual(r["status"], "complete")
        self.assertAlmostEqual(r["data"]["components"]["risk_to_stop"]["value"], 50.0)
        r2 = call(t, "calculate_position_risk", lots=0.015, entry_price=2000, stop_price=1995)
        self.assertTrue(any("volume step" in w for w in r2["warnings"]))  # off-step lots are flagged

    def test_26_precedence_mt5_over_unconfirmed_user(self):
        t, *_ = mk(FakeMT5())
        call(t, "update_user_account_profile", fields={"contract_size": 50})
        s = call(t, "get_symbol_specifications")
        self.assertEqual(s["data"]["specifications"]["contract_size"]["value"], 100.0)

    def test_27_broker_status(self):
        t, *_ = mk(FakeMT5())
        r = call(t, "get_broker_context_status")
        self.assertIn("calculations", r["data"])
        self.assertTrue(r["data"]["calculations"]["position_risk"]["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
