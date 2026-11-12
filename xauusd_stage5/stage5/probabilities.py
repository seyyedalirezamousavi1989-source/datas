"""Statistically grounded, calibrated scenario probabilities.

* Targets are measured historical outcomes (label file) - never model confidence.
* Estimator: hierarchical empirical-Bayes (Beta shrinkage) over causal context keys; fit on TRAIN only.
* TRAIN rows receive blocked out-of-fold estimates (6 chronological folds, +-24h purge around the held-out block);
  VAL/TEST rows receive the full-train estimate.
* VALIDATION decides, per target, whether the context model has skill over the base rate (day-block bootstrap of the Brier difference).
  If not, the base rate is used and reported as "no validated skill".  TEST is evaluated once, never used for any choice.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import s5config as S
from .data import load_m15
from .splits import assign_splits

M_SHRINK = 100.0
LEVELS = [["h4_trend_state", "h1_trend_state"], ["h4_trend_state", "h1_trend_state", "trend_state", "session_code"],
          ["h4_trend_state", "h1_trend_state", "trend_state", "session_code", "vol_bin"]]
FEATS = ["time", "trend_state", "h1_trend_state", "h4_trend_state", "atr_ratio", "session_asia", "session_london", "session_ny", "session_london_ny_overlap"]


def load_frame():
    lab = pd.read_csv(os.path.join(S.DIR_LABELS, f"labels_{S.LABEL_CONFIG_VERSION}.csv"), parse_dates=["time", "entry_time", "bar_close_time"])
    sp = assign_splits(lab)
    f = load_m15(FEATS)
    f["idx"] = np.arange(len(f))
    m = lab.merge(sp[["idx", "split"]], on="idx").merge(f.drop(columns=["time", "trend_state"]), on="idx")
    m = m[m.valid].reset_index(drop=True)
    m["session_code"] = (m.session_asia * 1 + m.session_london * 2 + m.session_ny * 4).astype(int)
    for c in ("h1_trend_state", "h4_trend_state"):
        m[c] = m[c].fillna(0)
    return add_targets(m)


def add_targets(m):
    m["t_long_tp"] = (m.long_outcome == "tp").astype(int)
    m["t_short_tp"] = (m.short_outcome == "tp").astype(int)
    m["t_bull_continuation"] = (m.first_touch_dir == 1).astype(int)
    m["t_bear_continuation"] = (m.first_touch_dir == -1).astype(int)
    m["t_range"] = ((m.up_exc_atr < S.SCENARIO["move_atr"]) & (m.dn_exc_atr < S.SCENARIO["move_atr"])).astype(int)
    trend = m.trend_state
    m["t_trend_continuation"] = (((trend == 1) & (m.first_touch_dir == 1)) | ((trend == -1) & (m.first_touch_dir == -1))).astype(int)
    m["t_reversal"] = (((trend == 1) & (m.first_touch_dir == -1)) | ((trend == -1) & (m.first_touch_dir == 1))).astype(int)
    m["t_no_trade"] = (m.label == "NO_TRADE").astype(int)
    return m


TARGETS = ["t_long_tp", "t_short_tp", "t_bull_continuation", "t_bear_continuation", "t_range", "t_trend_continuation", "t_reversal", "t_no_trade"]
CONDITIONAL = {"t_trend_continuation": lambda d: d.trend_state != 0, "t_reversal": lambda d: d.trend_state != 0}


class Estimator:
    def __init__(self, fit_df, target, vol_edges):
        self.target, self.edges = target, vol_edges
        d = self._prep(fit_df)
        if target in CONDITIONAL:
            d = d[CONDITIONAL[target](d)]
        self.base, self.n = float(d[target].mean()), len(d)
        self.tables = []
        for cols in LEVELS:
            g = d.groupby(cols)[target].agg(["sum", "count"])
            self.tables.append((cols, g))

    def _prep(self, df):
        d = df.copy()
        d["vol_bin"] = np.digitize(d.atr_ratio.fillna(1.0), self.edges)
        return d

    def predict(self, df):
        d = self._prep(df)
        p = np.full(len(d), self.base)
        support = np.full(len(d), self.n)
        for cols, g in self.tables:
            idx = pd.MultiIndex.from_frame(d[cols])
            s = g["sum"].reindex(idx).values
            c = g["count"].reindex(idx).values
            has = ~np.isnan(c)
            newp = np.where(has, (np.nan_to_num(s) + M_SHRINK * p) / (np.nan_to_num(c) + M_SHRINK), p)
            p = newp
            support = np.where(has, c, support)
        return p, support


def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    brier = float(np.mean((p - y) ** 2))
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    bins = np.minimum((p * 10).astype(int), 9)
    ece, rel = 0.0, []
    for b in range(10):
        m = bins == b
        if m.sum():
            ece += m.sum() / len(y) * abs(p[m].mean() - y[m].mean())
            rel.append({"bin": b, "n": int(m.sum()), "mean_pred": round(float(p[m].mean()), 4), "observed": round(float(y[m].mean()), 4)})
    return {"brier": brier, "log_loss": ll, "ece": float(ece), "reliability": rel, "n": int(len(y))}


def block_bootstrap_diff(day, y, p_model, p_base, reps=400, seed=5):
    """CI of Brier(base) - Brier(model) (positive = model better) with whole-day blocks."""
    rng = np.random.default_rng(seed)
    d = (p_base - y) ** 2 - (p_model - y) ** 2
    df = pd.DataFrame({"day": day, "d": d}).groupby("day")["d"].agg(["sum", "count"])
    s, c = df["sum"].values, df["count"].values
    n = len(s)
    out = []
    for _ in range(reps):
        k = rng.integers(0, n, n)
        out.append(s[k].sum() / c[k].sum())
    return float(np.mean(d)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def fit_all():
    m = load_frame()
    tr, va, te = (m[m.split == s] for s in ("train", "val", "test"))
    edges = np.quantile(tr.atr_ratio.dropna(), [1 / 3, 2 / 3])
    # blocked OOF on train
    folds = np.array_split(np.arange(len(tr)), 6)
    cols_out = {}
    report = {"method": __doc__, "shrinkage_m": M_SHRINK, "levels": LEVELS, "vol_edges": edges.tolist(), "targets": {}}
    for tg in TARGETS:
        oof_p, oof_n = np.zeros(len(tr)), np.zeros(len(tr))
        for f in folds:
            held = tr.iloc[f]
            lo, hi = held.bar_close_time.min() - pd.Timedelta(hours=24), held.bar_close_time.max() + pd.Timedelta(hours=24)
            fit = tr[(tr.bar_close_time < lo) | (tr.bar_close_time > hi)]
            p, n = Estimator(fit, tg, edges).predict(held)
            oof_p[f], oof_n[f] = p, n
        est = Estimator(tr, tg, edges)
        pv, nv = est.predict(va)
        pt, nt = est.predict(te)
        base = est.base
        mask = CONDITIONAL[tg](va) if tg in CONDITIONAL else np.ones(len(va), bool)
        yv = va[tg].values[mask]
        mv, mb = metrics(yv, pv[mask]), metrics(yv, np.full(mask.sum(), base))
        diff, lo_, hi_ = block_bootstrap_diff(va.bar_close_time.dt.date.values[mask], yv, pv[mask], np.full(mask.sum(), base))
        skill = lo_ > 0
        # TEST is reported once for the frozen decision; it is never used to choose
        maskt = CONDITIONAL[tg](te) if tg in CONDITIONAL else np.ones(len(te), bool)
        yt = te[tg].values[maskt]
        use = "context_model" if skill else "base_rate"
        pt_use = pt[maskt] if skill else np.full(maskt.sum(), base)
        report["targets"][tg] = {"train_base_rate": base, "train_n": est.n, "val": {"model": mv, "base_rate": mb, "brier_gain_vs_base": diff,
                                 "brier_gain_ci95": [lo_, hi_]}, "decision_on_validation": use,
                                 "test_frozen": {"chosen": metrics(yt, pt_use), "base_rate": metrics(yt, np.full(maskt.sum(), base))},
                                 "note": "gain>0 with CI lower bound>0 required to use the context model"}
        def final(p, base_=base, skill_=skill):
            return p if skill_ else np.full(len(p), base_)
        cols_out[tg] = (final(oof_p), oof_n, final(pv), nv, final(pt), nt, use)
    # assemble per-row probabilities
    rows = []
    for name, df, k in (("train", tr, 0), ("val", va, 2), ("test", te, 4)):
        d = pd.DataFrame({"idx": df.idx.values, "split": name})
        for tg in TARGETS:
            d["p_" + tg[2:]] = cols_out[tg][k]
            d["n_" + tg[2:]] = cols_out[tg][k + 1]
        rows.append(d)
    prob = pd.concat(rows).sort_values("idx")
    # movement / time-to-target quantiles (train, by coarse context)
    mv = {}
    for key, g in tr.groupby(["h4_trend_state", "h1_trend_state"]):
        mv[str(key)] = {"n": int(len(g)), "up_exc_atr_q": g.up_exc_atr.quantile([.1, .25, .5, .75, .9]).round(3).tolist(),
                        "dn_exc_atr_q": g.dn_exc_atr.quantile([.1, .25, .5, .75, .9]).round(3).tolist(),
                        "bars_to_tp_median_long": float(g[g.long_outcome == "tp"].long_bars_to_exit.median()) if (g.long_outcome == "tp").any() else None}
    report["movement_quantiles_train_by_h4_h1_trend"] = mv
    report["decisions"] = {tg: cols_out[tg][6] for tg in TARGETS}
    return prob, report


def run():
    prob, rep = fit_all()
    os.makedirs(S.DIR_REPORTS5, exist_ok=True)
    prob.to_csv(os.path.join(S.DIR_LABELS, "probabilities_v1.csv"), index=False, float_format="%.6f")
    with open(os.path.join(S.DIR_REPORTS5, "scenario_calibration_report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=str)
    for tg, r in rep["targets"].items():
        v = r["val"]
        print(f"{tg:24s} base={r['train_base_rate']:.3f} valBrier model={v['model']['brier']:.5f} base={v['base_rate']['brier']:.5f} gainCI={v['brier_gain_ci95'][0]:+.5f}..{v['brier_gain_ci95'][1]:+.5f} -> {r['decision_on_validation']}")
    return prob, rep


if __name__ == "__main__":
    run()
