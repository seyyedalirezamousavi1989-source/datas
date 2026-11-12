import os, sys, unittest
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage5 import s5config as S
from stage5.data import load_m15, load_m5
from stage5.labeling import net_r_vector
from broker_context.cost_profiles import historical_net_r

LAB = os.path.join(S.DIR_LABELS, f"labels_{S.LABEL_CONFIG_VERSION}.csv")


def ref_side(d, m5, i, side, k=1.5, rr=2.0, H=32):
    """Scalar brute-force reference implementation (independent of the vectorized code)."""
    entry = d.open.iloc[i + 1]
    a = d.atr_14.iloc[i]
    sl = k * a
    sgn = 1 if side == "long" else -1
    slp, tpp = entry - sgn * sl, entry + sgn * rr * sl
    for j in range(i + 1, i + 1 + H):
        hi, lo, op = d.high.iloc[j], d.low.iloc[j], d.open.iloc[j]
        s = (lo <= slp) if side == "long" else (hi >= slp)
        t = (hi >= tpp) if side == "long" else (lo <= tpp)
        if s and t:
            sub = m5[(m5.time >= d.time.iloc[j]) & (m5.time < d.time.iloc[j] + pd.Timedelta(minutes=15))]
            for _, r in sub.iterrows():
                s5 = (r.low <= slp) if side == "long" else (r.high >= slp)
                t5 = (r.high >= tpp) if side == "long" else (r.low <= tpp)
                if s5 and t5:
                    return "sl", j - i
                if s5:
                    return "sl", j - i
                if t5:
                    return "tp", j - i
            return "sl", j - i
        if s:
            return "sl", j - i
        if t:
            return "tp", j - i
    return "timeout", H


class TestLabeling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lab = pd.read_csv(LAB, parse_dates=["time", "entry_time", "bar_close_time"])
        cls.d = load_m15()
        cls.m5 = load_m5()

    def test_bruteforce_matches_vectorized(self):
        rng = np.random.default_rng(7)
        v = self.lab[self.lab.valid]
        sample = rng.choice(v.idx.values, 250, replace=False)
        for i in sample:
            for side in ("long", "short"):
                o, bars = ref_side(self.d, self.m5, int(i), side)
                row = self.lab.iloc[int(i)]
                self.assertEqual(row[f"{side}_outcome"], o, (i, side))
                if o != "timeout":
                    self.assertEqual(int(row[f"{side}_bars_to_exit"]), bars, (i, side))

    def test_entry_is_next_open_and_causal_atr(self):
        r = self.lab[self.lab.valid].sample(200, random_state=1)
        for _, x in r.iterrows():
            i = int(x.idx)
            self.assertEqual(x.entry_time, self.d.time.iloc[i] + pd.Timedelta(minutes=15))
            self.assertAlmostEqual(x.entry_price, self.d.open.iloc[i + 1], places=4)
            self.assertAlmostEqual(x.atr_14, self.d.atr_14.iloc[i], places=4)
            self.assertAlmostEqual(x.sl_dist, 1.5 * self.d.atr_14.iloc[i], places=4)

    def test_label_rule_consistency(self):
        v = self.lab[self.lab.valid]
        pf = S.PRIMARY_SCENARIO
        L = v[v.label == "LONG"]
        self.assertTrue((L.long_outcome == "tp").all())
        self.assertTrue((L[f"long_net_r_{pf}"] > 0).all())
        Sx = v[v.label == "SHORT"]
        self.assertTrue((Sx.short_outcome == "tp").all())
        self.assertFalse(((v.long_outcome == "tp") & (v.short_outcome == "tp")).any())
        self.assertEqual(set(v.label.unique()), {"LONG", "SHORT", "NO_TRADE"})

    def test_net_r_vector_equals_scalar_profile(self):
        for nm in S.COST_SCENARIOS:
            p = S.make_profile(nm)
            rng = np.random.default_rng(3)
            g = rng.normal(0, 4, 200)
            sl = rng.uniform(1, 8, 200)
            sp = np.where(rng.random(200) < 0.2, 0.0, rng.integers(3, 40, 200).astype(float))
            nr, _ = net_r_vector(p, g, sl, sp)
            for j in range(200):
                self.assertAlmostEqual(nr[j], historical_net_r(p, float(g[j]), float(sl[j]), float(sp[j]))["net_R"], places=9)

    def test_cost_scenarios_ordered(self):
        v = self.lab[(self.lab.valid) & (self.lab.long_outcome == "tp")]
        a, b, c = (v[f"long_net_r_{n}"] for n in ("A_spread_only", "B_moderate_assumed", "C_stress_assumed"))
        self.assertTrue((a > b).all() and (b > c).all())

    def test_ambiguity_worst_case(self):
        a = self.lab[self.lab.long_ambiguity_resolution == "same_m5_bar"]
        if len(a):
            self.assertTrue((a.long_outcome == "sl").all())
            self.assertTrue((a.long_ambiguous == 1).all())

    def test_no_weekend_windows_in_valid(self):
        v = self.lab[self.lab.valid]
        self.assertFalse(v.exclusion_reason.fillna("").ne("").any())
        self.assertTrue((self.lab.exclusion_reason == "horizon_crosses_weekend").sum() > 0)


if __name__ == "__main__":
    unittest.main()
