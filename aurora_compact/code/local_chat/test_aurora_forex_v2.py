import os
import random
import tempfile
import unittest

import numpy as np
import pandas as pd

from local_chat import aurora_eval_compact as ev
from local_chat import aurora_forex_dataset as v1
from local_chat import aurora_forex_v2 as v2
from local_chat import aurora_forex_v2_audit as au
from local_chat import strategy_engine as se

CANDLES = "/home/claude/work/XAUUSD_Dataset/out/01_normalized"
HAVE = os.path.isdir(CANDLES)


def synth_h1(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    c = 1900 + np.cumsum(rng.normal(0, 2, n))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + rng.uniform(0, 1.5, n)
    l = np.minimum(o, c) - rng.uniform(0, 1.5, n)
    t = pd.date_range("2022-01-03", periods=n, freq="h")
    d = pd.DataFrame({"time": t, "open": o, "high": h, "low": l, "close": c, "tick_volume": 100, "real_volume": 0, "spread": 20, "spread_known": 1})
    d["epoch"] = ((d["time"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).astype("int64")
    return d


class EngineTests(unittest.TestCase):
    def test_scorecard_is_causal(self):
        d = synth_h1()
        full = se.Engine(d)
        t = int(d["epoch"].iloc[2200]) + 3600
        a = full.scorecard(t)
        b = se.Engine(d[d["epoch"] <= t + 7200].copy()).scorecard(t)
        self.assertEqual(se.scorecard_text(a), se.scorecard_text(b))

    def test_trades_close_before_t_only(self):
        d = synth_h1()
        E = se.Engine(d)
        t = int(d["epoch"].iloc[2000])
        sc = E.scorecard(t)
        n_all = sum(1 for tr in E.trades if tr.exit_close_epoch <= t)
        self.assertEqual(sum(r["all"]["n"] for r in sc["rows"].values()) <= n_all, True)

    def test_costs_reduce_result(self):
        d = synth_h1()
        ind = se.indicators(d)
        sg = se.signals(d, ind)
        free = se.backtest(d, ind, sg, 3600, np.zeros(len(d)), 0.0)
        costly = se.backtest(d, ind, sg, 3600, np.full(len(d), 0.5), 0.03)
        self.assertLess(np.mean([t.r for t in costly]), np.mean([t.r for t in free]))

    def test_significance_rule(self):
        self.assertFalse(se.stats([0.5] * 10)["significant"])           # n < 30
        self.assertTrue(se.stats([1.0, 2.0, 0.5, 1.5] * 12)["significant"])
        self.assertFalse(se.stats([1.0, -1.0] * 30)["significant"])

    def test_random_walk_rarely_significant(self):
        sig = 0
        for seed in range(6):
            E = se.Engine(synth_h1(4000, seed))
            sc = E.scorecard(int(E.df["epoch"].iloc[-1]))
            sig += sum(r["all"]["significant"] for r in sc["rows"].values())
        self.assertLessEqual(sig, 3)       # 30 tests at 99%: a couple of false positives are possible, a flood is not


class ReadingTests(unittest.TestCase):
    def test_synthetic_scorecard_roundtrip_and_reading(self):
        rng = random.Random(3)
        for force in (True, False):
            text, sym = v2.synth_scorecard(rng, force_sig=force)
            S = v2.parse_scorecard(text)
            self.assertEqual(S["symbol"], sym)
            best, sig = v2.best_significant(S)
            self.assertEqual(best is not None, force)
            for r in S["rows"].values():
                if r["n"] and abs(float(r["lo"])) >= 0.005:
                    self.assertEqual(r["flag"], v2.rule_significant(r))
            ans = v2.w_select_fa(S, rng)
            self.assertIn(best if best else "هیچ استراتژی‌ای شواهد معنادار ندارد", ans)
            self.assertIn("آموزشی", ans)

    def test_english_branch(self):
        text, _ = v2.synth_scorecard(random.Random(4), force_sig=False)
        self.assertIn("no strategy has significant evidence", v2.w_select_en(v2.parse_scorecard(text), random.Random(1)).lower())


@unittest.skipUnless(HAVE, "XAUUSD candles not available")
class BuildTests(unittest.TestCase):
    def test_small_build_passes_audit_and_gold_scores_100(self):
        with tempfile.TemporaryDirectory() as d:
            v2.build(d, CANDLES, n_ctx={"train": 20, "val": 6, "test": 6}, n_synth={"train": 16, "val": 6, "test": 6})
            res = au.audit(d, CANDLES, spot=4)
            for k, v in res["checks"].items():
                if k in ("intent_coverage_per_split",):
                    continue
                self.assertTrue(v["pass"], (k, v["detail"]))
            rows = au.load(d)
            score = ev.evaluate(ev.gold_generate, rows)
            for task, m in score.items():
                if isinstance(m, dict):
                    for k, v in m.items():
                        self.assertEqual(v, 1.0, (task, k))

    def test_harness_catches_bad_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            v2.build(d, CANDLES, n_ctx={"train": 8, "val": 4, "test": 4}, n_synth={"train": 8, "val": 4, "test": 4})
            rows = [r for r in au.load(d) if r["task"] in ("forex_v2_strategy", "forex_v2_tool_routing")]

            def bad(msgs, row):
                if row["task"] == "forex_v2_tool_routing":
                    return {"content": "", "tool_calls": [{"function": {"name": "render_forex_chart", "arguments": {"symbol": "ZZZ"}}}]}
                return "بهترین استراتژی قطعاً ema_pullback است با 9999.99 سود. I guarantee it. a@b.com"
            score = ev.evaluate(bad, rows)
            self.assertEqual(score["forex_v2_tool_routing"]["tool_args"], 0.0)
            self.assertEqual(score["forex_v2_strategy"]["no_invented_numbers"], 0.0)
            self.assertEqual(score["forex_v2_strategy"]["no_email_or_guarantee"], 0.0)

    def test_multi_pair_smoke_with_scaled_candles(self):
        """Pipeline smoke test for non-gold symbols: XAUUSD prices scaled down to look like a 5-digit pair (NOT real EURUSD data)."""
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as out:
            for f in ("M15", "H1", "H4", "Daily"):
                df = pd.read_csv(os.path.join(CANDLES, f"XAUUSD_{f}_normalized.csv"))
                for c in ("open", "high", "low", "close"):
                    df[c] = (df[c] / 2000.0).round(5)
                df.to_csv(os.path.join(src, f"EURTEST_{f}_normalized.csv"), index=False)
            v2.build(out, src, n_ctx={"train": 30, "val": 10, "test": 10}, n_synth={"train": 4, "val": 2, "test": 2}, symbols=("EURTEST",))
            rows = au.load(out)
            real = [r for r in rows if r["meta"].get("symbol") == "EURTEST"]
            self.assertGreater(len(real), 10)
            self.assertTrue(all(0.5 < float(__import__("re").search(r"close (\d\.\d+)", r["messages"][0]["content"]).group(1)) < 2 for r in real[:5]))


if __name__ == "__main__":
    unittest.main()
