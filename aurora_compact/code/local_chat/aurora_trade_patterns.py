"""Trade pattern mining for the v3 dataset: label every historical setup trade with its outcome, then look for conditions
(setup family x side x up to 2 context features) whose train-period results are positive AND confirmed on later data.

Honest design:
* every trade is a mechanical, cost-aware simulation (strategy_engine.backtest, entry at the next open, stop-first on ties);
* features use only bars closed at the signal bar;
* patterns are mined on TRAIN only; Benjamini-Hochberg over every pattern tested; VAL confirms; TEST is reported, never used for selection;
* tiers: medium = BH-significant on train and positive on val, weak = nominal p<0.05 on train and positive on val, none = otherwise.
A tier is a statement about history, not a prediction.  Result tables are also written, so the claim 'tier predicts test performance' can be checked.
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from . import strategy_engine as se
    from . import edge_lab as el
except ImportError:                       # pragma: no cover
    import strategy_engine as se
    import edge_lab as el

SPLITS = {"train": ("2021-09-01", "2024-06-29"), "val": ("2024-07-02", "2025-06-29"), "test": ("2025-07-02", "2030-01-01")}
MIN_TRAIN_N, MIN_VAL_N, FDR_Q = 60, 20, 0.10
HOUR_STRATS = {f"hour_long_{h}": {"hour": h, "side": 1, "rr": 1.5} for h in (20, 21, 22)}
FA = {"ema_pullback": "پولبک به EMA در روند", "donchian_breakout": "شکست کانال دونچین", "range_reversion": "بازگشت به میانگین در رنج",
      "sweep_reversal": "بازگشت پس از جاروی نقدینگی", "squeeze_breakout": "شکست پس از فشردگی",
      "hour_long_20": "خرید ساعت ۲۰ سرور", "hour_long_21": "خرید ساعت ۲۱ سرور", "hour_long_22": "خرید ساعت ۲۲ سرور"}


def split_of(epoch):
    t = pd.Timestamp(epoch, unit="s")
    for k, (a, b) in SPLITS.items():
        if pd.Timestamp(a) <= t <= pd.Timestamp(b) + pd.Timedelta(days=1):
            return k
    return None


def features(df, ind, i, side):
    """Bucketed causal features at signal bar i (bar i is closed)."""
    ep = int(df["epoch"].iloc[i])
    hour = (ep // 3600) % 24
    dow = int(((ep // 86400) + 4) % 7)            # 0=Mon
    adx, rk, rsi = ind["adx"].iloc[i], ind["atr_rank"].iloc[i], ind["rsi"].iloc[i]
    atr = ind["atr"].iloc[i]
    c = df["close"].iloc[i]
    trend = np.sign(ind["ema50"].iloc[i] - ind["ema200"].iloc[i]) * side
    ext = (c - ind["ema20"].iloc[i]) / atr * side if atr > 0 else 0.0
    rs = (rsi - 50) * side
    return {
        "hblk": f"h{hour // 4 * 4:02d}-{hour // 4 * 4 + 3:02d}",
        "dow": ["Mon", "Tue", "Wed", "Thu", "Fri"][dow] if dow < 5 else "Weekend",
        "reg": "adx<20" if adx < 20 else "adx20-25" if adx < 25 else "adx>25",
        "vol": "atr_low" if rk < 1 / 3 else "atr_mid" if rk < 2 / 3 else "atr_high",
        "trend": "with_trend" if trend > 0 else "against_trend",
        "ext": "below_ema20" if ext < -1 else "near_below" if ext < 0 else "near_above" if ext < 1 else "far_above_ema20",
        "rsi": "rsi_against" if rs < -10 else "rsi_neutral" if rs <= 10 else "rsi_with",
    }


FEATS = ["hblk", "dow", "reg", "vol", "trend", "ext", "rsi"]
FEAT_FA = {"hblk": "بازهٔ ساعت سرور", "dow": "روز هفته", "reg": "رژیم ADX", "vol": "نوسان نسبی", "trend": "جهت نسبت به روند EMA50/200", "ext": "فاصله از EMA20", "rsi": "RSI نسبت به جهت"}
VAL_FA = {"atr_low": "نوسان پایین", "atr_mid": "نوسان متوسط", "atr_high": "نوسان بالا", "with_trend": "هم‌جهت با روند", "against_trend": "خلاف روند",
          "below_ema20": "زیر EMA20 بیش از ۱ ATR", "near_below": "کمی زیر EMA20", "near_above": "کمی بالای EMA20", "far_above_ema20": "بالای EMA20 بیش از ۱ ATR",
          "rsi_against": "RSI خلاف جهت", "rsi_neutral": "RSI خنثی", "rsi_with": "RSI هم‌جهت", "adx<20": "ADX<20 (رنج)", "adx20-25": "ADX بین ۲۰ و ۲۵", "adx>25": "ADX>25 (روندی)"}


def trades_table(df: pd.DataFrame) -> pd.DataFrame:
    """All mechanical trades of all setup families with outcome (R after costs) and features."""
    ind = se.indicators(df)
    sp = df["spread_price"].values
    sigs = se.signals(df, ind)
    cache = el.Cache(df)
    for name, p in HOUR_STRATS.items():
        sigs[name] = el.f_hour_drift(cache, p)
    ok = (ind["atr"] > 0) & ind[["atr", "adx", "rsi", "ema200", "atr_rank"]].notna().all(axis=1)
    sigs = {k: (np.where(ok.values, s, 0), d, rr) for k, (s, d, rr) in sigs.items()}
    tr = se.backtest(df, ind, sigs, el.BAR_SECONDS, sp, 0.03)
    rows = []
    for t in tr:
        i = t.entry_i - 1
        f = features(df, ind, i, t.side)
        rows.append({"strategy": t.strategy, "side": t.side, "signal_i": i, "entry_i": t.entry_i, "exit_i": t.exit_i, "entry_epoch": int(df["epoch"].iloc[t.entry_i]),
                     "exit_close_epoch": t.exit_close_epoch, "r": t.r, "dist": float(sigs[t.strategy][1][i]), "rr": float(sigs[t.strategy][2]),
                     "atr": float(ind["atr"].iloc[i]), **f})
    d = pd.DataFrame(rows)
    d["split"] = [split_of(e) for e in d["entry_epoch"]]
    return d[d["split"].notna()].reset_index(drop=True)


def _p_one_sided(r):
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        return 1.0
    t = r.mean() / (r.std(ddof=1) / math.sqrt(n))
    return 0.5 * math.erfc(t / math.sqrt(2))


def conditions():
    yield ()
    for f in FEATS:
        yield (f,)
    for a, b in itertools.combinations(FEATS, 2):
        yield (a, b)


def mine(T: pd.DataFrame) -> pd.DataFrame:
    """Return one row per tested pattern with train/val/test stats and tier."""
    out = []
    for strat in sorted(T["strategy"].unique()):
        S = T[T["strategy"] == strat]
        for side_name, sv in (("long", 1), ("short", -1)):
            Ss = S if sv is None else S[S["side"] == sv]
            if len(Ss) < MIN_TRAIN_N:
                continue
            for cond in conditions():
                if not cond:
                    groups = [((), Ss)]
                else:
                    groups = [(tuple(zip(cond, k if isinstance(k, tuple) else (k,))), g) for k, g in Ss.groupby(list(cond))]
                for key, g in groups:
                    tr = g[g["split"] == "train"]["r"].values
                    if len(tr) < MIN_TRAIN_N:
                        continue
                    va, te = g[g["split"] == "val"]["r"].values, g[g["split"] == "test"]["r"].values
                    out.append({"strategy": strat, "side": side_name, "cond": tuple(key), "n_tr": len(tr), "mean_tr": float(tr.mean()), "win_tr": float((tr > 0).mean()),
                                "p_tr": _p_one_sided(tr), "n_va": len(va), "mean_va": float(va.mean()) if len(va) else np.nan,
                                "n_te": len(te), "mean_te": float(te.mean()) if len(te) else np.nan, "win_te": float((te > 0).mean()) if len(te) else np.nan})
    P = pd.DataFrame(out)
    if P.empty:
        return P
    m = len(P)
    order = P["p_tr"].rank(method="first").astype(int)
    P["bh_threshold"] = order / m * FDR_Q
    srt = P.sort_values("p_tr").reset_index()
    kmax = max([i for i, row in srt.iterrows() if row["p_tr"] <= (i + 1) / m * FDR_Q] or [-1])
    P["bh_pass"] = False
    if kmax >= 0:
        P.loc[srt["index"][: kmax + 1], "bh_pass"] = True
    val_ok = (P["n_va"] >= MIN_VAL_N) & (P["mean_va"] > 0)
    P["tier"] = np.where(P["bh_pass"] & (P["mean_tr"] > 0) & val_ok, "medium", np.where((P["p_tr"] <= 0.05) & (P["mean_tr"] > 0) & val_ok, "weak", "none"))
    P["n_tested"] = m
    return P


def tier_check(P: pd.DataFrame) -> dict:
    """Does the tier predict held-out TEST results? (each pattern's test trades, pooled by tier, trade-weighted)."""
    res = {}
    for t, g in P.groupby("tier"):
        g = g[g["n_te"] > 0]
        n = int(g["n_te"].sum())
        res[t] = {"patterns": int(len(g)), "test_trades": n, "test_mean_R": float((g["mean_te"] * g["n_te"]).sum() / n) if n else None}
    return res
