import json, os, sys, unittest
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage5 import s5config as S, audits
from stage5.analysis import analyze, plan_trade, management_playbook
from stage5.probabilities import Estimator, metrics
from stage5.render_sft import load_snapshots, render
from stage5.source_quality import evaluate_source
from stage5.evaluate import score_decisions, score_sizing

D = S.DIR_DATASETS


def rd(name, n=None):
    out = []
    for l in open(os.path.join(D, name + ".jsonl")):
        out.append(json.loads(l))
        if n and len(out) >= n:
            break
    return out


class TestDatasets(unittest.TestCase):
    def test_audit_all_pass(self):
        o = audits.run()
        failed = {k: v for k, v in o["checks"].items() if not v["pass"]}
        self.assertEqual(failed, {}, f"Stage 5 audit must pass every gate: {failed}")
        c = o["checks"]["management_action_coverage"]["counts"]
        required = ("HOLD", "CLOSE_NOW", "TAKE_PARTIAL_50", "TAKE_PARTIAL_33", "MOVE_STOP_TO_BREAK_EVEN",
                    "TAKE_PARTIAL_AND_MOVE_STOP_TO_BREAK_EVEN", "TRAIL_STOP_ATR", "TRAIL_STOP_STRUCTURE")
        for action in required:
            self.assertGreaterEqual(c.get(action, 0), 30, action)

    def test_all_nine_task_files_and_splits_present(self):
        for t in audits.TASKS:
            recs = rd(t)
            self.assertTrue(len(recs) > 500, t)
            self.assertEqual({r["split"] for r in recs}, {"train", "val", "test"}, t)

    def test_no_trade_is_present_and_substantial(self):
        c = pd.Series([r["target"]["decision"] for r in rd("entry_exit_planning")]).value_counts(normalize=True)
        self.assertGreater(c["NO_TRADE"], 0.5)
        self.assertGreater(c.get("LONG", 0), 0.02)
        self.assertGreater(c.get("SHORT", 0), 0.02)

    def test_sizing_numeric_accuracy_vs_independent_formula(self):
        n = ok = 0
        for r in rd("position_sizing"):
            res = r["target"]["tool_results"][0]
            d = res["data"]
            if res["status"] not in ("complete", "complete_with_gaps") or d["binding_constraint"] != "policy_risk_pct" or d["reductions"]:
                continue
            a = r["target"]["tool_calls"][0]["arguments"]
            st, pol = a["account_state"], a["policy"]
            base = st["equity"] if pol["base"] == "equity" else st["balance"]
            risk = base * pol["risk_pct"] / 100
            dist = abs(a["entry_price"] - a["stop_price"])
            lots = np.floor(risk / (dist * 100) / 0.01 + 1e-9) * 0.01
            n += 1
            ok += abs(lots - d["lots"]) < 1e-9 and d["risk_money"] <= risk + 1e-6
        self.assertGreater(n, 200)
        self.assertEqual(ok, n)

    def test_incomplete_cases_report_zero_lots_not_a_guess(self):
        seen = 0
        for r in rd("position_sizing") + rd("risk_management"):
            res = r["target"]["tool_results"][0]
            if res["status"] == "incomplete":
                seen += 1
                self.assertEqual(res["data"]["lots"], 0.0)
                self.assertIn("incomplete", r["target"]["final_text"].lower())
        self.assertGreater(seen, 20)

    def test_final_text_numbers_come_from_tool_results(self):
        for r in rd("position_sizing", 400):
            res = r["target"]["tool_results"][0]
            if res["status"] in ("complete", "complete_with_gaps"):
                self.assertIn(str(res["data"]["lots"]), r["target"]["final_text"])
                self.assertIn(str(res["data"]["risk_money"]), r["target"]["final_text"])

    def test_render_structure(self):
        snaps = load_snapshots(D)
        for t in audits.TASKS:
            r = rd(t, 3)[0]
            x = render(r, snaps)
            roles = [m["role"] for m in x["messages"]]
            self.assertEqual(roles[0], "system")
            self.assertEqual(roles[-1], "assistant")
            if t in ("risk_management", "position_sizing", "portfolio_risk", "integrated_trading_tasks") and r["target"].get("tool_calls"):
                self.assertIn("tool", roles)
                self.assertIsNotNone(x["tools"])

    def test_probability_estimator_shrinks_to_base_when_sparse(self):
        df = pd.DataFrame({"h4_trend_state": [1] * 3, "h1_trend_state": [1] * 3, "trend_state": [1] * 3, "session_code": [1] * 3, "atr_ratio": [1.0] * 3, "t_x": [1, 1, 1]})
        big = pd.DataFrame({"h4_trend_state": [-1] * 1000, "h1_trend_state": [-1] * 1000, "trend_state": [-1] * 1000, "session_code": [2] * 1000, "atr_ratio": [1.0] * 1000, "t_x": [0] * 500 + [1] * 500})
        est = Estimator(pd.concat([df, big]), "t_x", np.array([0.9, 1.1]))
        p, n = est.predict(df.iloc[:1])
        self.assertLess(p[0], 0.9)          # 3 of 3 successes must not give ~1.0
        self.assertGreater(p[0], est.base)

    def test_metrics_perfect_and_calibrated(self):
        y = np.array([0, 1] * 500)
        m = metrics(y, np.full(1000, 0.5))
        self.assertAlmostEqual(m["brier"], 0.25)
        self.assertAlmostEqual(m["ece"], 0.0, places=6)

    def test_scorers(self):
        s = score_decisions(["LONG", "SHORT", "NO_TRADE", "NO_TRADE"], ["LONG", "NO_TRADE", "NO_TRADE", "NO_TRADE"])
        self.assertEqual(s["per_class"]["LONG"]["precision"], 1.0)
        self.assertEqual(s["confusion_matrix(rows=true,cols=pred)"]["matrix"][1], [0, 0, 1])
        self.assertEqual(score_sizing([0.1, 0.2], [0.1, 0.21])["exact_match_rate"], 0.5)

    def test_playbook_rules(self):
        base = dict(side="long", entry_price=2000.0, current_sl=1995.0, remaining_size=1.0, unrealized_r=0.2, partial_exits=[], break_even_active=False, price=2001.0,
                    steps_held=5, risk_distance=5.0, atr_now=3.0, atr_ratio_now=1.0, structure_invalidated=False)
        self.assertEqual(management_playbook(base)["action"], "HOLD")
        self.assertEqual(management_playbook(dict(base, unrealized_r=1.1))["action"], "TAKE_PARTIAL_AND_MOVE_STOP_TO_BREAK_EVEN")
        self.assertEqual(management_playbook(dict(base, structure_invalidated=True))["action"], "CLOSE_NOW")
        mv = management_playbook(dict(base, unrealized_r=1.6, break_even_active=True, current_sl=2000.5, price=2008.0, partial_exits=[{}, {}], remaining_size=0.34))
        self.assertEqual(mv["action"], "TRAIL_STOP_ATR")
        self.assertGreaterEqual(mv["new_stop"], 2000.5)         # never loosens

    def test_playbook_new_actions(self):
        base = dict(side="long", entry_price=2000.0, current_sl=2000.5, remaining_size=1.0, unrealized_r=1.2, partial_exits=[], break_even_active=True, price=2006.0,
                    steps_held=9, risk_distance=5.0, atr_now=3.0, atr_ratio_now=1.0, structure_invalidated=False)
        self.assertEqual(management_playbook(base)["action"], "TAKE_PARTIAL_50")
        tr = management_playbook(dict(base, unrealized_r=1.6, price=2008.0))      # protected, no partial yet, >=1.5R -> trail, not partial
        self.assertIn(tr["action"], ("TRAIL_STOP_ATR", "TRAIL_STOP_STRUCTURE"))
        self.assertEqual(management_playbook(dict(base, current_sl=1995.0, break_even_active=False, partial_exits=[{}], remaining_size=0.5))["action"], "MOVE_STOP_TO_BREAK_EVEN")
        self.assertEqual(management_playbook(dict(base, partial_exits=[{}], remaining_size=0.5, unrealized_r=1.7, price=2008.5))["action"], "TAKE_PARTIAL_33")
        t = management_playbook(dict(base, partial_exits=[{}, {}], remaining_size=0.34, unrealized_r=1.8, price=2009.0, current_sl=2000.5,
                                     m15_last_swing_low=2007.0))
        self.assertEqual(t["action"], "TRAIL_STOP_STRUCTURE")
        self.assertGreater(t["new_stop"], 2004.5)
        s = management_playbook(dict(base, side="short", entry_price=2000.0, current_sl=2005.0, price=1994.0, partial_exits=[{}, {}], remaining_size=0.34, unrealized_r=1.6))
        self.assertEqual(s["action"], "TRAIL_STOP_ATR"); self.assertLess(s["new_stop"], 2005.0)

    def test_analysis_blocks_on_poor_execution(self):
        snap = {k: None for k in S.SNAPSHOT_COLS}
        snap.update(atr_14=2.0, close=2000.0, spread=60, spread_known=1, h4_trend_state=1, h1_trend_state=1, d1_trend_state=1, trend_state=1, session_london=1, atr_ratio=1.0)
        a = analyze(snap)
        self.assertIsNone(a["setup"]["side"])
        self.assertTrue(any("poor_execution" in f for f in a["no_trade_flags"]))

    def test_plan_geometry(self):
        snap = {k: None for k in S.SNAPSHOT_COLS}
        snap.update(atr_14=2.0, close=2000.0, last_swing_low=1990.0, last_swing_high=2010.0, fvg_bull_in_zone=1)
        a = analyze(snap)
        for side in ("long", "short"):
            p = plan_trade(snap, side, a)
            sg = 1 if side == "long" else -1
            self.assertAlmostEqual(abs(p["entry_price"] - p["stop_loss"]), 3.0, places=1)
            self.assertAlmostEqual(sg * (p["take_profit_2"] - p["entry_price"]), 6.0, places=1)

    def test_source_quality_rejects_bad_series(self):
        df = pd.DataFrame({"time": ["2024-01-01", "2023-01-01", "2023-01-01"], "v": [1, np.nan, 3]})
        r = evaluate_source(df)
        self.assertEqual(r["verdict"], "REJECT_UNTIL_FIXED")
        self.assertIn("not monotonic", r["blocking_problems"])


if __name__ == "__main__":
    unittest.main()
