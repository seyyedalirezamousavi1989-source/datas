"""Feature quality & redundancy report (M15 decision frame, ~260 columns).
Selection evidence comes from TRAIN; VALIDATION only confirms; TEST is never read."""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from . import s5config as S
from .data import load_m15
from .splits import assign_splits

NON_FEATURE = {"time", "bar_close_time"}
TIMESTR = lambda c: c.endswith("_bar_time")  # noqa: E731


def auc(y, x):
    m = np.isfinite(x)
    y, x = y[m], x[m]
    if len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
        return np.nan
    r = pd.Series(x).rank().values
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def psi(a, b, bins=10):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 100 or len(b) < 100 or len(np.unique(a)) < 3:
        return np.nan
    e = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(e) < 3:
        return np.nan
    e[0], e[-1] = -np.inf, np.inf
    pa = np.histogram(a, e)[0] / len(a) + 1e-6
    pb = np.histogram(b, e)[0] / len(b) + 1e-6
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def run():
    lab = pd.read_csv(os.path.join(S.DIR_LABELS, f"labels_{S.LABEL_CONFIG_VERSION}.csv"), parse_dates=["time", "entry_time", "bar_close_time"])
    sp = assign_splits(lab)
    d = load_m15()
    d["idx"] = np.arange(len(d))
    m = lab.merge(sp[["idx", "split"]], on="idx", suffixes=("", "_s")).merge(d.drop(columns=["time"]), on="idx", suffixes=("", "_f"))
    m = m[m.valid]
    tr, va = m[m.split == "train"], m[m.split == "val"]          # test split is intentionally not touched
    cols = [c for c in d.columns if c not in NON_FEATURE | {"idx"} and not TIMESTR(c) and pd.api.types.is_numeric_dtype(d[c])]
    cols = [c for c in cols if c in tr.columns or c + "_f" in tr.columns]
    get = lambda df, c: df[c + "_f"] if c + "_f" in df.columns else df[c]  # noqa: E731
    yd_tr = tr.first_touch_dir.values
    yd_va = va.first_touch_dir.values
    mtr, mva = yd_tr != 0, yd_va != 0
    ent = tr.t_ = None
    rows = []
    sample = tr.sample(min(40000, len(tr)), random_state=0).sort_index()
    X = pd.DataFrame({c: get(sample, c).values for c in cols})
    dup = {}
    seen = {}
    for c in cols:
        key = pd.util.hash_pandas_object(X[c].round(6), index=False).sum(), X[c].isna().sum()
        if key in seen and np.allclose(X[c].fillna(-9e9), X[seen[key]].fillna(-9e9)):
            dup[c] = seen[key]
        else:
            seen.setdefault(key, c)
    corr = X.drop(columns=[c for c in X if X[c].nunique() < 2]).corr(method="spearman")
    high = {}
    for c in corr.columns:
        peers = [p for p in corr.columns if p != c and abs(corr.at[c, p]) >= 0.95]
        high[c] = peers
    years = tr.bar_close_time.dt.year
    for c in cols:
        xt, xv = get(tr, c).values.astype(float), get(va, c).values.astype(float)
        a_tr = auc((yd_tr[mtr] == 1).astype(int), xt[mtr])
        a_va = auc((yd_va[mva] == 1).astype(int), xv[mva])
        nt_tr = auc((tr.label.values == "NO_TRADE").astype(int), xt)
        nt_va = auc((va.label.values == "NO_TRADE").astype(int), xv)
        yr_means = pd.Series(xt).groupby(years.values).mean()
        stab = float(yr_means.std() / (np.nanstd(xt) + 1e-12)) if len(yr_means) > 1 else np.nan
        flags = []
        miss_tr = float(np.isnan(xt).mean())
        if np.nanstd(xt) == 0 or len(np.unique(xt[np.isfinite(xt)])) <= 1:
            flags.append("constant")
        if c in dup:
            flags.append(f"duplicate_of:{dup[c]}")
        if miss_tr > 0.9:
            flags.append("very_sparse(event/zone-conditional; NaN means 'none active')")
        if len(high.get(c, [])) > 0:
            flags.append(f"corr>=0.95_with:{len(high[c])}")
        for nm, a in (("dir", a_tr), ("no_trade", nt_tr)):
            if np.isfinite(a) and abs(a - 0.5) > 0.15:
                flags.append(f"strong_univariate_{nm}_auc_review_for_leakage")
        p = psi(xt, xv)
        if np.isfinite(p) and p > 0.25:
            flags.append("distribution_shift_train_to_val(PSI>0.25)")
        if c == "real_volume":
            flags.append("all_zero_real_volume: broker supplies no exchange volume")
        if c.startswith("tick_volume") or c.startswith("tv_"):
            flags.append("TICK volume (broker ticks), not exchange volume")
        rows.append({"feature": c, "missing_train": round(miss_tr, 4), "missing_val": round(float(np.isnan(xv).mean()), 4),
                     "n_unique_train": int(len(np.unique(xt[np.isfinite(xt)]))), "dir_auc_train": a_tr, "dir_auc_val": a_va,
                     "notrade_auc_train": nt_tr, "notrade_auc_val": nt_va, "yearly_mean_instability": stab, "psi_train_val": p,
                     "in_model_snapshot": c in S.SNAPSHOT_COLS, "flags": ";".join(flags)})
    rep = pd.DataFrame(rows)
    rep["dir_confirmed"] = (rep.dir_auc_train.sub(.5).abs() > 0.01) & (np.sign(rep.dir_auc_train - .5) == np.sign(rep.dir_auc_val - .5)) & \
                           (rep.dir_auc_val.sub(.5).abs() >= 0.5 * rep.dir_auc_train.sub(.5).abs())
    rep["notrade_confirmed"] = (rep.notrade_auc_train.sub(.5).abs() > 0.01) & (np.sign(rep.notrade_auc_train - .5) == np.sign(rep.notrade_auc_val - .5)) & \
                               (rep.notrade_auc_val.sub(.5).abs() >= 0.5 * rep.notrade_auc_train.sub(.5).abs())
    os.makedirs(S.DIR_REPORTS5, exist_ok=True)
    rep.to_csv(os.path.join(S.DIR_REPORTS5, "feature_quality_report.csv"), index=False, float_format="%.5f")
    # incremental value by feature group (train fit -> val AUC; confirmatory only)
    groups = {"G0_vol_session": ["atr_ratio", "rv_ratio", "atr_pct", "bb_width", "tod_sin", "tod_cos", "session_asia", "session_london", "session_ny", "dow", "spread"],
              "G1_trend_momentum": ["ret_3", "ret_10", "ret_20", "rsi_14", "dist_ema_20_atr", "dist_ema_50_atr", "dist_ema_200_atr", "ema_stack", "macd_hist_atr", "bb_pctb", "donchian_pos_20", "donchian_pos_100"],
              "G2_smc_structure": ["trend_state", "struct_last_event", "bars_since_struct_event", "pd_pos", "pd_zone", "disp_strength_atr", "dist_last_swing_high_atr", "dist_last_swing_low_atr", "bsl_sweep", "ssl_sweep", "dist_prev_day_high_atr", "dist_prev_day_low_atr", "fvg_bull_dist_atr", "fvg_bear_dist_atr", "ob_bull_dist_atr", "ob_bear_dist_atr"],
              "G3_mtf": [c for c in cols if c.startswith(("h1_", "h4_", "d1_", "m30_")) and not TIMESTR(c) and "close" not in c and "swing" not in c]}
    inc = {}
    acc = []
    tgt = {"direction(up-first vs down-first)": ((yd_tr != 0), (yd_tr == 1).astype(int), (yd_va != 0), (yd_va == 1).astype(int)),
           "no_trade": (np.ones(len(tr), bool), (tr.label.values == "NO_TRADE").astype(int), np.ones(len(va), bool), (va.label.values == "NO_TRADE").astype(int))}
    for tn, (mt, yt, mv, yv) in tgt.items():
        acc = []
        res = {}
        for gname, gcols in groups.items():
            acc += [c for c in gcols if c in cols]
            Xt = np.column_stack([get(tr, c).values for c in acc])[mt]
            Xv = np.column_stack([get(va, c).values for c in acc])[mv]
            clf = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.06, max_depth=4, random_state=0)
            clf.fit(Xt, yt[mt])
            res[gname] = {"n_features_cumulative": len(acc), "val_auc": float(roc_auc_score(yv[mv], clf.predict_proba(Xv)[:, 1])),
                          "train_auc": float(roc_auc_score(yt[mt], clf.predict_proba(Xt)[:, 1]))}
        inc[tn] = res
    summary = {"rows_train": int(len(tr)), "rows_val": int(len(va)), "n_features": len(cols), "duplicates": dup,
               "n_constant": int(rep["flags"].str.contains("constant").sum()), "n_very_sparse": int(rep["flags"].str.contains("very_sparse").sum()),
               "n_high_corr_features": int(sum(1 for v in high.values() if v)),
               "n_dir_confirmed": int(rep.dir_confirmed.sum()), "n_notrade_confirmed": int(rep.notrade_confirmed.sum()),
               "n_strong_univariate_review": int(rep["flags"].str.contains("strong_univariate").sum()),
               "incremental_value_by_group_cumulative": inc,
               "policy": "Correlated features are NOT dropped solely for correlation; the model snapshot uses a curated subset (S.SNAPSHOT_COLS) chosen for distinct analytical purpose. "
                         "No feature was selected with validation or test data; validation only confirms train-side evidence.",
               "inference_availability": "All columns are computed from candles closed at bar_close_time; HTF columns use nominal-close merge_asof (Stage 4). "
                                         "real_volume is all zero; tick_volume is a broker tick count.",
               "test_split_used": False}
    top = rep.assign(score=(rep.dir_auc_val - .5).abs()).sort_values("score", ascending=False).head(10)[["feature", "dir_auc_train", "dir_auc_val"]]
    summary["top10_univariate_direction_auc_val"] = top.round(4).to_dict("records")
    json.dump(summary, open(os.path.join(S.DIR_REPORTS5, "feature_quality_summary.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("policy", "inference_availability", "duplicates")}, indent=1, default=str)[:3500])


if __name__ == "__main__":
    run()


def price_level_flags():
    """Adds (does not replace) a classification of absolute-price-level columns: |spearman with close| >= 0.98 on TRAIN.
    These are non-stationary reference levels, not learnable features; the model should use their ATR-normalised distances.
    Also reports the G3 ablation with those columns removed (train fit -> val confirmation; test untouched)."""
    lab = pd.read_csv(os.path.join(S.DIR_LABELS, f"labels_{S.LABEL_CONFIG_VERSION}.csv"), parse_dates=["bar_close_time", "entry_time"])
    sp = assign_splits(lab)
    d = load_m15()
    d["idx"] = np.arange(len(d))
    m = lab[["idx", "valid", "label", "first_touch_dir"]].merge(sp[["idx", "split"]], on="idx").merge(d.drop(columns=["time"]), on="idx")
    m = m[m.valid]
    tr, va = m[m.split == "train"], m[m.split == "val"]
    cols = [c for c in d.columns if c not in NON_FEATURE | {"idx", "close"} and not TIMESTR(c) and pd.api.types.is_numeric_dtype(d[c])]
    smp = tr.sample(30000, random_state=0)
    cc = {c: abs(smp[c].corr(smp["close"], method="spearman")) for c in cols + ["open", "high", "low"] if c in smp}
    level = sorted(c for c, v in cc.items() if v >= 0.98)
    rep = pd.DataFrame({"feature": list(cc), "abs_spearman_with_close_train": list(cc.values())})
    rep["absolute_price_level"] = rep.feature.isin(level)
    rep.to_csv(os.path.join(S.DIR_REPORTS5, "feature_price_level_flags.csv"), index=False, float_format="%.4f")
    g3 = [c for c in cols if c.startswith(("h1_", "h4_", "d1_", "m30_")) and "close" not in c and "swing" not in c]
    base = [c for c in ["atr_ratio", "rv_ratio", "atr_pct", "bb_width", "tod_sin", "tod_cos", "session_asia", "session_london", "session_ny", "dow", "spread", "ret_3", "ret_10", "ret_20", "rsi_14",
                        "dist_ema_20_atr", "dist_ema_50_atr", "dist_ema_200_atr", "ema_stack", "macd_hist_atr", "bb_pctb", "donchian_pos_20", "donchian_pos_100", "trend_state", "struct_last_event",
                        "bars_since_struct_event", "pd_pos", "pd_zone", "disp_strength_atr", "dist_last_swing_high_atr", "dist_last_swing_low_atr", "bsl_sweep", "ssl_sweep",
                        "dist_prev_day_high_atr", "dist_prev_day_low_atr", "fvg_bull_dist_atr", "fvg_bear_dist_atr", "ob_bull_dist_atr", "ob_bear_dist_atr"] if c in cols]
    out = {"n_absolute_price_level_columns": len(level), "examples": level[:25], "ablation": {}}
    for name, extra in (("with_price_levels", g3), ("without_price_levels", [c for c in g3 if c not in level])):
        acc = base + extra
        res = {}
        for tn, (ft, fv) in {"direction": ((tr.first_touch_dir != 0, (tr.first_touch_dir == 1).astype(int)), (va.first_touch_dir != 0, (va.first_touch_dir == 1).astype(int))),
                             "no_trade": ((np.ones(len(tr), bool), (tr.label == "NO_TRADE").astype(int)), (np.ones(len(va), bool), (va.label == "NO_TRADE").astype(int)))}.items():
            (mt, yt), (mv, yv) = (np.asarray(ft[0]), np.asarray(ft[1])), (np.asarray(fv[0]), np.asarray(fv[1]))
            clf = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.06, max_depth=4, random_state=0)
            Xt, Xv = tr[acc].values[mt], va[acc].values[mv]
            clf.fit(Xt, yt[mt])
            res[tn] = {"train_auc": round(float(roc_auc_score(yt[mt], clf.predict_proba(Xt)[:, 1])), 4), "val_auc": round(float(roc_auc_score(yv[mv], clf.predict_proba(Xv)[:, 1])), 4)}
        out["ablation"][name] = {"n_features": len(acc), **res}
    out["note"] = "diagnostic only; test split not read; no feature set was chosen from this"
    json.dump(out, open(os.path.join(S.DIR_REPORTS5, "feature_price_level_diagnostic.json"), "w"), indent=1)
    return out
