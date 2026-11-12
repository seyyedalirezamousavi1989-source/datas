"""Evaluation harness.  No model exists yet: this module (a) measures the deterministic PROCEDURE and simple BASELINES on
chronological splits and (b) provides scorers for real model predictions (``score_*``).
Nothing here is tuned: every rule/threshold was fixed before looking at validation or test results.
Costs: three explicit, unverified assumption scenarios (A spread-only, B moderate, C stress)."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from . import s5config as S
from .analysis import analyze, plan_trade
from .data import load_m15
from .probabilities import metrics as prob_metrics
from .simulator import POLICIES, make_context, rescore_net_r, simulate_entry_order, simulate_trade
from .splits import assign_splits

CLASSES = ["LONG", "SHORT", "NO_TRADE"]
SC_NAMES = list(S.COST_SCENARIOS)


def score_decisions(y_true, y_pred):
    p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=0)
    return {"per_class": {c: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4), "f1": round(float(f[i]), 4), "support": int(sup[i])} for i, c in enumerate(CLASSES)},
            "macro_f1": round(float(np.mean(f)), 4), "accuracy": round(float(np.mean(np.array(y_true) == np.array(y_pred))), 4),
            "confusion_matrix(rows=true,cols=pred)": {"labels": CLASSES, "matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist()}}


def score_probabilities(y, p):
    return prob_metrics(np.asarray(y, float), np.asarray(p, float))


def score_sizing(pred_lots, ref_lots, tol=1e-9):
    a, b = np.asarray(pred_lots, float), np.asarray(ref_lots, float)
    return {"n": int(len(a)), "exact_match_rate": float(np.mean(np.abs(a - b) <= tol)), "max_abs_error_lots": float(np.max(np.abs(a - b))) if len(a) else None}


def _dd(r):
    c = np.cumsum(r)
    return float(np.max(np.maximum.accumulate(c) - c)) if len(c) else 0.0


def _mean_ci(day, x, reps=400, seed=1):
    df = pd.DataFrame({"d": day, "x": x}).groupby("d")["x"].agg(["sum", "count"])
    s, c = df["sum"].values, df["count"].values
    rng = np.random.default_rng(seed)
    bs = [s[k].sum() / c[k].sum() for k in (rng.integers(0, len(s), len(s)) for _ in range(reps))]
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def procedure_frame():
    lab = pd.read_csv(os.path.join(S.DIR_LABELS, f"labels_{S.LABEL_CONFIG_VERSION}.csv"), parse_dates=["time", "entry_time", "bar_close_time"])
    sp = assign_splits(lab)
    d = load_m15()
    d["idx"] = np.arange(len(d))
    lab = lab.merge(sp[["idx", "split"]], on="idx")
    lab = lab[lab.valid & lab.split.isin(["train", "val", "test"])].reset_index(drop=True)
    feat = d.set_index("idx").loc[lab.idx]
    recs = feat.to_dict("records")
    out = [analyze(r) for r in recs]
    lab["proc_side"] = [o["setup"]["side"] or "none" for o in out]
    lab["proc_quality"] = [o["setup"]["quality"] for o in out]
    lab["regime"] = [o["regime"] for o in out]
    lab["session_name"] = [o["session"] for o in out]
    lab["vol_state"] = [o["volatility"] for o in out]
    lab["proc_decision"] = lab.proc_side.map({"long": "LONG", "short": "SHORT", "none": "NO_TRADE"})
    lab["h4_trend"] = feat.h4_trend_state.fillna(0).values
    return lab, feat, recs, out


def decision_eval(lab):
    res = {}
    rng = np.random.default_rng(0)
    tr_freq = lab[lab.split == "train"].label.value_counts(normalize=True).reindex(CLASSES).fillna(0).values
    for sp in ("train", "val", "test"):
        x = lab[lab.split == sp]
        y = x.label.values
        base = {"procedure": x.proc_decision.values, "always_NO_TRADE": np.full(len(x), "NO_TRADE"), "always_LONG": np.full(len(x), "LONG"),
                "h4_trend_follower": np.where(x.h4_trend.values > 0, "LONG", np.where(x.h4_trend.values < 0, "SHORT", "NO_TRADE")),
                "random_train_frequencies": rng.choice(CLASSES, len(x), p=tr_freq)}
        res[sp] = {k: score_decisions(y, v) for k, v in base.items()}
    return res


def trading_eval(lab):
    """Cost-adjusted outcome of the next-open market entry (fixed 1.5ATR/2R) on the rows each rule trades."""
    res = {}
    for sp in ("train", "val", "test"):
        x = lab[lab.split == sp]
        def block(mask_long, mask_short, name):
            out = {}
            n = int(mask_long.sum() + mask_short.sum())
            for sc in SC_NAMES:
                r = np.concatenate([x[mask_long][f"long_net_r_{sc}"].values, x[mask_short][f"short_net_r_{sc}"].values])
                day = np.concatenate([x[mask_long].bar_close_time.dt.date.values, x[mask_short].bar_close_time.dt.date.values])
                out[sc] = {"mean_net_r": round(float(r.mean()), 4) if n else None, "ci95_day_block": _mean_ci(day, r) if n > 30 else None,
                           "win_rate": round(float((r > 0).mean()), 4) if n else None, "max_drawdown_r_chronological": round(_dd(r[np.argsort(np.concatenate([x[mask_long].bar_close_time.values, x[mask_short].bar_close_time.values]))]), 2) if n else None}
            out["n_trades"] = n
            return out
        L, Sx = x.proc_side == "long", x.proc_side == "short"
        every = np.ones(len(x), bool)
        rng = np.random.default_rng(3)
        coin = rng.random(len(x)) < 0.5
        res[sp] = {"procedure": block(L, Sx, "procedure"), "always_long_every_bar": block(every, np.zeros(len(x), bool), "al"),
                   "always_short_every_bar": block(np.zeros(len(x), bool), every, "as"),
                   "random_side_every_bar": block(coin, ~coin, "rnd"),
                   "random_side_same_count_as_procedure": block(*(lambda m: (m & coin, m & ~coin))((L | Sx).values), "rnd2"),
                   "h4_trend_every_bar": block((x.h4_trend > 0).values, (x.h4_trend < 0).values, "h4")}
        tp = x[(L | Sx)]
        side_tp = np.where(tp.proc_side == "long", tp.long_outcome == "tp", tp.short_outcome == "tp")
        res[sp]["procedure"]["tp_before_sl_rate"] = round(float(side_tp.mean()), 4) if len(tp) else None
        res[sp]["note"] = "tp_before_sl base rate for a random side is ~0.27 (2R target vs 1R stop random-walk expectation ~1/3 minus timeouts)"
    return res


def stability(lab):
    out = {}
    t = lab[(lab.proc_side != "none") & lab.split.isin(["train", "val", "test"])].copy()
    t["net_B"] = np.where(t.proc_side == "long", t.long_net_r_B_moderate_assumed, t.short_net_r_B_moderate_assumed)
    t["tp"] = np.where(t.proc_side == "long", t.long_outcome == "tp", t.short_outcome == "tp")
    t["year"] = t.bar_close_time.dt.year
    for key in ("year", "session_name", "vol_state", "regime"):
        g = t.groupby(key).agg(n=("net_B", "size"), mean_net_r_B=("net_B", "mean"), tp_rate=("tp", "mean")).round(4)
        out[key] = g.reset_index().to_dict("records")
    return out


def management_eval(lab, feat, ctx, n_per_split=1200, seed=11):
    profs = {k: S.make_profile(k) for k in SC_NAMES}
    prim = profs[S.PRIMARY_SCENARIO]
    res, entry_res = {}, {}
    for sp in ("train", "val", "test"):
        x = lab[(lab.split == sp) & (lab.proc_side != "none")]
        x = x.sample(min(n_per_split, len(x)), random_state=seed).sort_values("bar_close_time")
        pol_rows = {p: [] for p in POLICIES}
        ent_rows = {k: [] for k in ("market", "limit", "stop", "rule_selected")}
        for _, r in x.iterrows():
            side = r.proc_side
            sl_dist = float(r.sl_dist)
            for pol, cfg in POLICIES.items():
                t = simulate_trade(ctx, side, int(r.idx), cfg, prim, float(r.entry_price), sl_dist, int(r.entry_time.value))
                row = {"time": r.bar_close_time, "gross_r": t["gross_r"], "mfe": t["mfe_r"], "mae": t["mae_r"], "amb": t["ambiguous"], "exit": t["exit_reason"], "held": t["bars_held_m5"]}
                for sc, pf in profs.items():
                    row[sc] = rescore_net_r(t, pf, float(r.spread_pts_decision), sl_dist)
                pol_rows[pol].append(row)
            snap = feat.loc[int(r.idx)].to_dict()
            an = analyze(snap)
            plan = plan_trade(snap, side, an)
            sel = plan["order_type"]
            for ot in ("market", "limit", "stop"):
                pl = plan_trade(snap, side, an)
                # level for forced types (same rules, forced order type)
                atr, close = r.atr_14, snap["close"]
                sg = 1 if side == "long" else -1
                lvl = None if ot == "market" else (close - sg * 0.5 * atr if ot == "limit" else close + sg * 0.25 * atr)
                t = simulate_entry_order(ctx, side, ot, lvl, int(r.bar_close_time.value), int(r.idx), POLICIES["fixed"], prim, 1.5, atr)
                if t.get("filled"):
                    v = rescore_net_r(t, prim, float(r.spread_pts_decision), 1.5 * atr)
                else:
                    v = 0.0
                ent_rows[ot].append((t.get("filled", False), v))
                if ot == sel:
                    ent_rows["rule_selected"].append((t.get("filled", False), v))
        res[sp] = {}
        for pol, rows in pol_rows.items():
            df = pd.DataFrame(rows)
            capt = df[df.mfe > 0.5]
            res[sp][pol] = {"n": len(df), **{f"mean_net_r_{sc}": round(float(df[sc].mean()), 4) for sc in SC_NAMES},
                            "win_rate_B": round(float((df[S.PRIMARY_SCENARIO] > 0).mean()), 4), "max_drawdown_r_B": round(_dd(df[S.PRIMARY_SCENARIO].values), 2),
                            "mean_mfe_r": round(float(df.mfe.mean()), 3), "mean_mae_r": round(float(df.mae.mean()), 3),
                            "mfe_capture_mean(mfe>0.5R)": round(float((capt.gross_r / capt.mfe).mean()), 3) if len(capt) else None,
                            "ambiguous_share": round(float(df.amb.mean()), 4), "exit_mix": df.exit.value_counts(normalize=True).round(3).to_dict(),
                            "mean_bars_held_m5": round(float(df.held.mean()), 1)}
        entry_res[sp] = {k: {"fill_rate": round(float(np.mean([f for f, _ in v])), 4), "mean_net_r_per_setup_unfilled_as_0": round(float(np.mean([r for _, r in v])), 4),
                             "mean_net_r_filled": round(float(np.mean([r for f, r in v if f])), 4) if any(f for f, _ in v) else None} for k, v in ent_rows.items()}
    return res, entry_res


def run(n_per_split=1200):
    os.makedirs(S.DIR_REPORTS5, exist_ok=True)
    lab, feat, recs, out = procedure_frame()
    lab[["idx", "split", "proc_side", "proc_quality", "regime", "session_name", "vol_state"]].to_csv(os.path.join(S.DIR_LABELS, "procedure_decisions_v1.csv"), index=False)
    rep = {"config": {"label": S.LABEL, "costs": {k: S.make_profile(k).as_dict() for k in SC_NAMES}, "all_cost_assumptions_unverified": True},
           "decision_classification_vs_hindsight_labels": decision_eval(lab), "cost_adjusted_trading_outcomes": trading_eval(lab),
           "stability_of_procedure_trades_all_splits": stability(lab)}
    ctx = make_context()
    rep["management_policies"], rep["entry_order_types"] = management_eval(lab, feat, ctx, n_per_split)
    rep["disclaimer"] = ("Historical measurements under assumed costs; they do not demonstrate profitability or predictive accuracy and say nothing about future results. "
                         "The test split was evaluated once with rules frozen before the test was read.")
    json.dump(rep, open(os.path.join(S.DIR_REPORTS5, "task_evaluation_report.json"), "w"), indent=1, default=str)
    return rep


if __name__ == "__main__":
    r = run()
    for sp in ("val", "test"):
        print(sp, "procedure", json.dumps(r["cost_adjusted_trading_outcomes"][sp]["procedure"]["B_moderate_assumed"]),
              "n", r["cost_adjusted_trading_outcomes"][sp]["procedure"]["n_trades"], "random", r["cost_adjusted_trading_outcomes"][sp]["random_side_same_count_as_procedure"]["B_moderate_assumed"]["mean_net_r"])
