import os, sys, unittest
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage5 import s5config as S, splits
from stage5.simulator import POLICIES, make_context, simulate_trade

LAB = os.path.join(S.DIR_LABELS, f"labels_{S.LABEL_CONFIG_VERSION}.csv")


class TestSim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lab = pd.read_csv(LAB, parse_dates=["time", "entry_time", "bar_close_time"])
        cls.ctx = make_context()
        cls.prof = S.make_profile(S.PRIMARY_SCENARIO)
        cls.v = cls.lab[cls.lab.valid].sample(300, random_state=11)

    def run_one(self, r, side, pol, **kw):
        return simulate_trade(self.ctx, side, int(r.idx), POLICIES[pol], self.prof, r.entry_price, r.sl_dist, int(r.entry_time.value), **kw)

    def test_fixed_policy_reproduces_labels(self):
        n = bad = 0
        for _, r in self.v.iterrows():
            for side in ("long", "short"):
                o = r[f"{side}_outcome"]
                if o not in ("tp", "sl"):
                    continue
                t = self.run_one(r, side, "fixed")
                n += 1
                if abs(t["net_r"] - r[f"{side}_net_r_{S.PRIMARY_SCENARIO}"]) > 1e-6:
                    bad += 1
        self.assertGreater(n, 100)
        self.assertLessEqual(bad / n, 0.01, f"{bad}/{n} mismatches between M5 simulator and M15+M5-resolved labels")

    def test_no_lookahead_truncation(self):
        """Records up to step t are identical when the world is cut right after t."""
        for _, r in self.v.head(40).iterrows():
            for pol in ("atr_trail", "structure_trail", "partial_be", "volatility_exit"):
                full = self.run_one(r, "long", pol)
                for rec in full["records"][:4]:
                    cut = self.run_one(r, "long", pol, upto_step=rec["step"] + 1)
                    same = [x for x in cut["records"] if x["step"] == rec["step"]]
                    if not same:               # trade ended before that step in the truncated world only if it also ended in the full one
                        continue
                    a, b = dict(same[0]), dict(rec)
                    for kx in ("partial_exits",):
                        a.pop(kx), b.pop(kx)
                    self.assertEqual(a, b, (pol, rec["step"]))

    def test_record_fields_complete(self):
        r = self.v.iloc[0]
        t = self.run_one(r, "long", "partial_be", record_every=1)
        need = {"entry_price", "entry_time_ns", "initial_sl", "initial_tp", "current_sl", "current_tp", "remaining_size", "unrealized_r",
                "mfe_r", "mae_r", "trailing_active", "break_even_active", "partial_exits", "costs_r_so_far"}
        self.assertTrue(t["records"])
        self.assertTrue(need <= set(t["records"][0]))
        self.assertIn("exit_reason", t)

    def test_stops_only_tighten_and_size_nonincreasing(self):
        for _, r in self.v.head(80).iterrows():
            for side in ("long", "short"):
                for pol in ("break_even", "atr_trail", "structure_trail", "partial_be", "multi_tp"):
                    t = self.run_one(r, side, pol, record_every=1)
                    sgn = 1 if side == "long" else -1
                    sls = [x["current_sl"] for x in t["records"]]
                    rem = [x["remaining_size"] for x in t["records"]]
                    for a, b in zip(sls, sls[1:]):
                        self.assertGreaterEqual(sgn * (b - a), -1e-9)
                    for a, b in zip(rem, rem[1:]):
                        self.assertLessEqual(b, a + 1e-12)

    def test_break_even_never_worse_than_costs_after_activation(self):
        for _, r in self.v.head(120).iterrows():
            t = self.run_one(r, "long", "break_even")
            if t["exit_reason"] == "break_even_stop":
                self.assertGreater(t["net_r"], -0.3)

    def test_partial_accounting(self):
        for _, r in self.v.iterrows():
            t = self.run_one(r, "long", "multi_tp")
            tot = sum(e["fraction"] for e in t["events"])
            self.assertAlmostEqual(tot, 1.0, places=5)

    def test_ambiguity_worst_case(self):
        n = 0
        for _, r in self.v.iterrows():
            t = self.run_one(r, "short", "fixed")
            if t["ambiguous"]:
                n += 1
                self.assertEqual(t["exit_reason"], "stop_loss")
        self.assertGreaterEqual(n, 0)

    def test_all_policies_finite(self):
        for _, r in self.v.head(60).iterrows():
            for pol in POLICIES:
                for side in ("long", "short"):
                    t = self.run_one(r, side, pol)
                    self.assertTrue(np.isfinite(t["net_r"]))
                    self.assertLessEqual(t["bars_held_m5"], S.MGMT_HORIZON_M5)


class TestSplits(unittest.TestCase):
    def test_split_audit_passes(self):
        lab = pd.read_csv(LAB, parse_dates=["time", "entry_time", "bar_close_time"])
        a = splits.audit(lab, splits.assign_splits(lab))
        self.assertTrue(a["all_pass"], a["checks"])


if __name__ == "__main__":
    unittest.main()
