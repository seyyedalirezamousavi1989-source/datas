"""Strategy evidence engine: causal rule-based strategies, cost-aware R-multiple backtest, and an "as of t" scorecard.

Purpose: the fine-tuned model must NOT guess which strategy works on a pair; this module computes the evidence with code and the model
only reads it.  Nothing here is an optimised trading system: parameters are fixed (no tuning), five strategies are tested (a multiple-
testing warning is part of the scorecard) and significance uses a 99% bootstrap CI on the mean R after spread + slippage.

Causality: a signal at bar i uses only data <= i, the entry is the OPEN of bar i+1, and the scorecard at time t uses only trades that were
fully closed at t (exit bar close <= t).  Same-bar stop/target ties resolve to the stop (conservative).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

STRATEGIES = ("ema_pullback", "donchian_breakout", "range_reversion", "sweep_reversal", "squeeze_breakout")
STRATEGY_FA = {
    "ema_pullback": "پولبک به EMA در روند",
    "donchian_breakout": "شکست کانال 20 کندلی",
    "range_reversion": "بازگشت به میانگین در رنج",
    "sweep_reversal": "برگشت بعد از sweep نقدینگی",
    "squeeze_breakout": "شکست فشردگی بولینگر",
}
STRATEGY_RULE_FA = {
    "ema_pullback": "روند صعودی/نزولی با EMA50 و EMA200، لمس EMA20 و کندل برگشتی هم‌جهت",
    "donchian_breakout": "close بیرون از بالاترین/پایین‌ترین 20 کندل قبل با ADX بالای 20",
    "range_reversion": "ADX زیر 20 و RSI در اشباع به‌همراه خروج close از باند بولینگر",
    "sweep_reversal": "سایه‌ای که کف/سقف 20 کندل قبل را برمی‌دارد ولی close داخل برمی‌گردد",
    "squeeze_breakout": "فشردگی باند بولینگر (20% پایین 100 کندل) و سپس close بیرون از باند",
}
MAX_HOLD = 48
STOP_ATR = 1.5
MIN_TRADES_SIGNIFICANT = 30
CI_LEVEL = 0.99


# ------------------------------------------------------------------------------------------------ indicators
def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _adx(df, n=14):
    up, dn = df["high"].diff(), -df["low"].diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = _atr(df, n)
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * pd.Series(mdm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def _rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def indicators(df):
    o = pd.DataFrame(index=df.index)
    o["atr"] = _atr(df)
    o["adx"] = _adx(df)
    o["rsi"] = _rsi(df["close"])
    o["ema20"], o["ema50"], o["ema200"] = (_ema(df["close"], n) for n in (20, 50, 200))
    mid = df["close"].rolling(20).mean()
    sd = df["close"].rolling(20).std(ddof=0)
    o["bb_mid"], o["bb_up"], o["bb_lo"] = mid, mid + 2 * sd, mid - 2 * sd
    o["bw"] = (o["bb_up"] - o["bb_lo"]) / mid
    o["bw_rank"] = o["bw"].rolling(100).apply(lambda x: (x[:-1] < x[-1]).mean() if len(x) == 100 else np.nan, raw=True)
    o["atr_rank"] = o["atr"].rolling(200).apply(lambda x: (x[:-1] < x[-1]).mean() if len(x) == 200 else np.nan, raw=True)
    return o


# ------------------------------------------------------------------------------------------------ signals
def signals(df, ind):
    """dict name -> (side array in {-1,0,1}, stop-distance array in price units, rr). Each value at bar i uses data <= i only."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    atr = ind["atr"]
    out = {}
    # 1 ema_pullback
    up = (ind["ema50"] > ind["ema200"]) & (c > ind["ema200"]) & (l <= ind["ema20"]) & (c > ind["ema20"]) & (c > o)
    dn = (ind["ema50"] < ind["ema200"]) & (c < ind["ema200"]) & (h >= ind["ema20"]) & (c < ind["ema20"]) & (c < o)
    out["ema_pullback"] = (np.where(up, 1, np.where(dn, -1, 0)), (STOP_ATR * atr).values, 2.0)
    # 2 donchian_breakout
    hh, ll = h.shift(1).rolling(20).max(), l.shift(1).rolling(20).min()
    up = (c > hh) & (ind["adx"] > 20)
    dn = (c < ll) & (ind["adx"] > 20)
    out["donchian_breakout"] = (np.where(up, 1, np.where(dn, -1, 0)), (STOP_ATR * atr).values, 2.0)
    # 3 range_reversion
    rng = ind["adx"] < 20
    up = rng & (ind["rsi"] < 30) & (c < ind["bb_lo"])
    dn = rng & (ind["rsi"] > 70) & (c > ind["bb_up"])
    out["range_reversion"] = (np.where(up, 1, np.where(dn, -1, 0)), (STOP_ATR * atr).values, 1.5)
    # 4 sweep_reversal (stop beyond the sweep wick)
    pl, ph = l.shift(1).rolling(20).min(), h.shift(1).rolling(20).max()
    up = (l < pl) & (c > pl) & (c > o)
    dn = (h > ph) & (c < ph) & (c < o)
    wick_dn = (c - l) + 0.2 * atr
    wick_up = (h - c) + 0.2 * atr
    dist = np.where(up, wick_dn, np.where(dn, wick_up, STOP_ATR * atr))
    dist = np.clip(dist, 0.5 * atr.values, 3.0 * atr.values)
    out["sweep_reversal"] = (np.where(up, 1, np.where(dn, -1, 0)), dist, 2.0)
    # 5 squeeze_breakout
    sq = ind["bw_rank"].shift(1) <= 0.2
    up = sq & (c > ind["bb_up"])
    dn = sq & (c < ind["bb_lo"])
    out["squeeze_breakout"] = (np.where(up, 1, np.where(dn, -1, 0)), (STOP_ATR * atr).values, 2.0)
    ok = (ind["atr"] > 0) & ind[["atr", "adx", "rsi", "ema200", "bb_mid"]].notna().all(axis=1)
    for k, (s, d, rr) in out.items():
        out[k] = (np.where(ok.values, s, 0), d, rr)
    return out


# ------------------------------------------------------------------------------------------------ backtest
@dataclass
class Trade:
    strategy: str
    entry_i: int
    exit_i: int
    side: int
    r: float
    adx: float
    exit_close_epoch: int     # close time of the exit bar (seconds, same clock as df['epoch'])


def backtest(df, ind, sigs, bar_seconds, spread_price=None, slip_atr=0.03):
    """Run every strategy; one position at a time per strategy. Costs: spread + slippage in price units, converted to R."""
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    ep = df["epoch"].values
    atr = ind["atr"].values
    adx = ind["adx"].values
    sp = spread_price if spread_price is not None else np.zeros(len(df))
    n = len(df)
    trades: List[Trade] = []
    for name, (side_arr, dist_arr, rr) in sigs.items():
        i = 0
        while i < n - 2:
            s = int(side_arr[i])
            if s == 0:
                i += 1
                continue
            e = i + 1
            entry, dist = o[e], float(dist_arr[i])
            if not dist > 0:
                i += 1
                continue
            stop = entry - s * dist
            tgt = entry + s * rr * dist
            exit_px, exit_j = None, None
            for j in range(e, min(n, e + MAX_HOLD)):
                hit_stop = l[j] <= stop if s == 1 else h[j] >= stop
                hit_tgt = h[j] >= tgt if s == 1 else l[j] <= tgt
                if hit_stop:
                    exit_px = stop if (j == e or (o[j] > stop if s == 1 else o[j] < stop)) else o[j]
                    exit_j = j
                    break
                if hit_tgt:
                    exit_px, exit_j = tgt, j
                    break
            if exit_px is None:
                exit_j = min(n - 1, e + MAX_HOLD - 1)
                exit_px = c[exit_j]
                if exit_j >= n - 1 and exit_j - e < MAX_HOLD - 1:
                    break                       # trade still open at the end of the data: not counted
            gross = s * (exit_px - entry) / dist
            cost = (float(sp[e]) + slip_atr * atr[i]) / dist
            trades.append(Trade(name, e, exit_j, s, float(gross - cost), float(adx[i]), int(ep[exit_j] + bar_seconds)))
            i = exit_j + 1
    return trades


# ------------------------------------------------------------------------------------------------ statistics
def bootstrap_ci(r, level=CI_LEVEL, reps=2000):
    r = np.asarray(r, float)
    if len(r) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(len(r) * 7919 + int(abs(r.sum()) * 1000) % 100003)
    means = rng.choice(r, size=(reps, len(r)), replace=True).mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def stats(rs):
    rs = np.asarray(rs, float)
    n = len(rs)
    if n == 0:
        return {"n": 0, "mean": None, "win": None, "pf": None, "lo": None, "hi": None, "significant": False}
    lo, hi = bootstrap_ci(rs)
    gp, gl = rs[rs > 0].sum(), -rs[rs < 0].sum()
    return {"n": n, "mean": float(rs.mean()), "win": float((rs > 0).mean()), "pf": (float(gp / gl) if gl > 0 else None), "lo": lo, "hi": hi,
            "significant": bool(n >= MIN_TRADES_SIGNIFICANT and lo > 0)}


def regime_label(adx):
    return "روندی" if adx > 25 else "رنج" if adx < 20 else "میانی"


def vol_label(rank):
    return "نوسان بالا" if rank >= 0.8 else "نوسان پایین" if rank <= 0.2 else "نوسان عادی"


class Engine:
    """Precomputes indicators/signals/trades on a full H1 history; ``scorecard(t_epoch)`` is then cheap and causal."""

    def __init__(self, df_h1, spread_price=None, bar_seconds=3600):
        self.df = df_h1.reset_index(drop=True)
        self.bar_seconds = bar_seconds
        self.ind = indicators(self.df)
        self.sigs = signals(self.df, self.ind)
        sp = spread_price
        if sp is None and "spread" in self.df:
            known = self.df.get("spread_known", pd.Series(1, index=self.df.index)).values
            sp = np.where(known > 0, self.df["spread"].values * 0.01, 0.25)
        self.trades = backtest(self.df, self.ind, self.sigs, bar_seconds, sp)
        self.ep = self.df["epoch"].values

    def last_closed(self, t_epoch):
        i = int(np.searchsorted(self.ep + self.bar_seconds, t_epoch, side="right")) - 1
        return i

    def scorecard(self, t_epoch, window_months=24):
        i = self.last_closed(t_epoch)
        if i < 250:
            return None
        t0 = t_epoch - window_months * 30 * 86400
        adx_now, rank_now = float(self.ind["adx"].iloc[i]), float(self.ind["atr_rank"].iloc[i])
        reg = regime_label(adx_now)
        rows = {}
        for s in STRATEGIES:
            tr = [t for t in self.trades if t.strategy == s and t.exit_close_epoch <= t_epoch and self.ep[t.entry_i] >= t0]
            allr = stats([t.r for t in tr])
            same = stats([t.r for t in tr if regime_label(t.adx) == reg])
            sg = int(self.sigs[s][0][i])
            rows[s] = {"all": allr, "regime": same, "signal": sg, "stop": float(self.sigs[s][1][i]), "rr": self.sigs[s][2]}
        win = min(window_months, max(1, int((t_epoch - self.ep[0]) / (30 * 86400))))
        return {"i": i, "t_epoch": t_epoch, "regime": reg, "adx": adx_now, "vol": vol_label(rank_now), "window_months": win, "rows": rows,
                "n_tested": len(STRATEGIES), "close": float(self.df["close"].iloc[i]), "atr": float(self.ind["atr"].iloc[i])}


def _sgn(x, nd=2):
    return f"{x:+.{nd}f}"


def scorecard_text(sc, symbol="XAUUSD", digits=2):
    """Persian block appended to the analysis context. Every number is formatted here, so the dataset can reuse the exact strings."""
    L = [f"کارت‌امتیاز استراتژی‌ها ({symbol} H1؛ معاملات کاملاً بسته‌شده تا همین لحظه، {sc['window_months']} ماه اخیر، با هزینهٔ اسپرد و اسلیپیج؛ پارامترها ثابت و بهینه‌نشده):"]
    for s in STRATEGIES:
        r = sc["rows"][s]
        a = r["all"]
        if a["n"] == 0:
            L.append(f"- {s}: بدون معامله در بازه")
            continue
        pf = "-" if a["pf"] is None else f"{a['pf']:.2f}"
        L.append(f"- {s}: n={a['n']} | میانگین {_sgn(a['mean'])}R | CI99% [{_sgn(a['lo'])}، {_sgn(a['hi'])}] | نرخ برد {a['win'] * 100:.0f}% | PF {pf} | معنادار: {'بله' if a['significant'] else 'خیر'}")
    L.append(f"رژیم فعلی: {sc['regime']} (ADX {sc['adx']:.0f}) و {sc['vol']}.")
    reg_lines = []
    for s in STRATEGIES:
        g = sc["rows"][s]["regime"]
        if g["n"]:
            reg_lines.append(f"{s}: n={g['n']} میانگین {_sgn(g['mean'])}R")
    L.append("همین رژیم: " + ("؛ ".join(reg_lines) if reg_lines else "معاملهٔ کافی نیست") + ".")
    sig = [f"{s}={'خرید' if sc['rows'][s]['signal'] > 0 else 'فروش'} (فاصلهٔ استاپ {sc['rows'][s]['stop']:.{digits}f}، RR {sc['rows'][s]['rr']:g})"
           for s in STRATEGIES if sc["rows"][s]["signal"] != 0]
    L.append("سیگنال روی آخرین کندل بستهٔ H1: " + ("، ".join(sig) if sig else "هیچ‌کدام فعال نیست") + ".")
    L.append(f"هشدار: {sc['n_tested']} استراتژی آزموده شده (خطر چندآزمونی)؛ «معنادار» یعنی n حداقل {MIN_TRADES_SIGNIFICANT} و حد پایین CI99% مثبت؛ گذشته تضمین آینده نیست.")
    return "\n".join(L)
