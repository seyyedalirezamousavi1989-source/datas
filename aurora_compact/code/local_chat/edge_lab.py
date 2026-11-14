"""Edge lab: an honest, reproducible search for tradable edge on candle data (any symbols, CSV or MetaTrader 5).

It is the research harness an agent calls; the language model never decides whether an edge exists, this module does:

  1. pair screen      cost (median spread / ATR), history length and gaps decide which symbols are worth searching
  2. strategy space   ~270 fixed configurations in 7 families (trend pullback, channel breakout, range reversion, sweep reversal,
                      squeeze breakout, hour-of-day drift, volatility-gated momentum); every configuration is a registered TRIAL
  3. walk-forward     the whole selection procedure (pick the best config on the past, trade it on the next unseen 6 months) is replayed
                      chronologically with purge/embargo; only these out-of-sample trades count as evidence
  4. multiple testing Hansen's SPA / White's reality check (stationary bootstrap over ALL trials of a symbol) answers: is the best of K
                      trials better than the best you would expect by chance?
  5. stress           2x costs, cross-symbol replication, share of the family that is positive
  6. sealed holdout   the last N months are never used by 1-5; one look per finalist, logged in the registry; a second look is refused

Verdicts: ``no_edge`` (default), ``candidate`` (passes 1-5 on development data, still unproven) and ``holdout_passed`` (survived the one look
at sealed data; the next step is a forward/paper test, not capital).  A validated edge is not a promise of profit.

    python -m local_chat.edge_lab screen  --candles candles --symbols EURUSD,GBPUSD,XAUUSD
    python -m local_chat.edge_lab search  --candles candles --symbols EURUSD,GBPUSD,XAUUSD --out lab_out
    python -m local_chat.edge_lab holdout --candles candles --out lab_out --finalist <id>        (once per finalist)
    (--mt5 instead of --candles reads history directly from the MetaTrader 5 terminal of LocalChat)
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from . import strategy_engine as se
except ImportError:                       # pragma: no cover
    import strategy_engine as se

BAR_SECONDS = 3600
DAY = 86400
DEFAULT_SYMBOLS = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "EURJPY", "GBPJPY"]
SCREEN_MAX_COST_ATR = 0.15        # median spread / ATR(H1): above this, costs eat most of any plausible edge
SCREEN_MIN_BARS = 20000
HOLDOUT_MONTHS = 12
INIT_TRAIN_MONTHS = 18
TEST_MONTHS = 6
EMBARGO_BARS = 48
MIN_TRADES_SELECT = 30
SPA_ALPHA = 0.05
FOLD_POSITIVE_MIN = 0.6
BREADTH_MIN = 0.55


# ============================================================================================ data
def frame_from_columns(cols: dict, point: float = 0.0, name: str = "") -> pd.DataFrame:
    """Normalize raw columns (time epoch seconds, open..close, spread in points, tick_volume) to the lab frame."""
    d = pd.DataFrame({k: cols[k] for k in ("time", "open", "high", "low", "close") if k in cols})
    d["epoch"] = d["time"].astype("int64")
    sp = np.asarray(cols.get("spread", np.zeros(len(d))), float)
    d["spread_price"] = sp * point if point else sp
    d = d.drop_duplicates("epoch").sort_values("epoch").reset_index(drop=True)
    return d


def load_csv(candles_dir: str, symbol: str, tf: str = "H1", point: Optional[float] = None) -> pd.DataFrame:
    f = "Daily" if tf == "D1" else tf
    d = pd.read_csv(os.path.join(candles_dir, f"{symbol}_{f}_normalized.csv"), parse_dates=["time"])
    d["epoch"] = ((d["time"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).astype("int64")
    pt = point if point is not None else (0.01 if symbol.startswith(("XAU", "BTC")) else 0.001 if symbol.endswith("JPY") else 0.00001)
    if "spread" in d:
        known = d.get("spread_known", pd.Series(1, index=d.index)).values
        d["spread_price"] = np.where(known > 0, d["spread"].values * pt, np.nan)
    else:
        d["spread_price"] = np.nan
    med = np.nanmedian(d["spread_price"]) if np.isfinite(d["spread_price"]).any() else 0.0
    d["spread_price"] = d["spread_price"].fillna(med)
    return d.sort_values("epoch").reset_index(drop=True)


def load_mt5(symbol: str, bars: int = 60000, suffix: str = "") -> pd.DataFrame:      # pragma: no cover - needs a terminal
    try:
        from . import mt5_data
    except ImportError:
        import mt5_data
    h = mt5_data.fetch_history(symbol, {"H1": bars}, suffix=suffix)
    col = h["frames"]["H1"]
    return frame_from_columns(col, point=h["info"]["point"], name=symbol)


# ============================================================================================ pair screen
def screen_symbol(df: pd.DataFrame) -> dict:
    ind = se.indicators(df)
    atr = ind["atr"].replace(0, np.nan)
    cost = float(np.nanmedian(df["spread_price"] / atr))
    dt = np.diff(df["epoch"].values)
    big = int((dt > 4 * BAR_SECONDS).sum() - (dt > 40 * BAR_SECONDS).sum())          # ignore weekends (<= 3 days) via threshold below
    weekend = int(((dt > 4 * BAR_SECONDS) & (dt <= 3 * DAY)).sum())
    gaps = int((dt > 3 * DAY).sum())
    years = (df["epoch"].iloc[-1] - df["epoch"].iloc[0]) / (365.25 * DAY)
    ok = cost <= SCREEN_MAX_COST_ATR and len(df) >= SCREEN_MIN_BARS and gaps <= 20
    reason = [] if ok else [r for r, c in (("cost/ATR too high", cost > SCREEN_MAX_COST_ATR), ("too few bars", len(df) < SCREEN_MIN_BARS), ("many data gaps", gaps > 20)) if c]
    return {"bars": int(len(df)), "years": round(years, 2), "median_spread_over_atr": round(cost, 4), "multi_day_gaps": gaps, "pass": bool(ok), "reasons": reason}


# ============================================================================================ strategy space
class Cache:
    """Indicator cache so that 270 configurations do not recompute the same EMA/rolling series."""

    def __init__(self, df):
        self.df, self.ind, self._m = df, se.indicators(df), {}
        self.hour = ((df["epoch"].values // 3600) % 24).astype(int)

    def get(self, key, fn):
        if key not in self._m:
            self._m[key] = fn()
        return self._m[key]

    def ema(self, n):
        return self.get(("ema", n), lambda: self.df["close"].ewm(span=n, adjust=False).mean())


def _pack(up, dn, dist, rr, ok=None):
    side = np.where(up, 1, np.where(dn, -1, 0))
    if ok is not None:
        side = np.where(ok, side, 0)
    return side, np.asarray(dist, float), float(rr)


def f_ema_pullback(c, p):
    o, h, l, cl, atr = c.df["open"], c.df["high"], c.df["low"], c.df["close"], c.ind["atr"]
    et, ep = c.ema(p["trend"]), c.ema(p["pull"])
    up = (cl > et) & (ep > et) & (l <= ep) & (cl > ep) & (cl > o)
    dn = (cl < et) & (ep < et) & (h >= ep) & (cl < ep) & (cl < o)
    return _pack(up, dn, p["stop"] * atr, p["rr"])


def f_donchian(c, p):
    h, l, cl, atr, adx = c.df["high"], c.df["low"], c.df["close"], c.ind["atr"], c.ind["adx"]
    hh, ll = h.shift(1).rolling(p["n"]).max(), l.shift(1).rolling(p["n"]).min()
    return _pack((cl > hh) & (adx > p["adx"]), (cl < ll) & (adx > p["adx"]), p["stop"] * atr, p["rr"])


def f_range_rev(c, p):
    cl, atr, adx, rsi = c.df["close"], c.ind["atr"], c.ind["adx"], c.ind["rsi"]
    mid = c.get(("bbm",), lambda: cl.rolling(20).mean())
    sd = c.get(("bbs",), lambda: cl.rolling(20).std(ddof=0))
    up = (adx < p["adx_max"]) & (rsi < p["rsi"]) & (cl < mid - p["bb"] * sd)
    dn = (adx < p["adx_max"]) & (rsi > 100 - p["rsi"]) & (cl > mid + p["bb"] * sd)
    return _pack(up, dn, 1.5 * atr, p["rr"])


def f_sweep(c, p):
    o, h, l, cl, atr = c.df["open"], c.df["high"], c.df["low"], c.df["close"], c.ind["atr"]
    pl, ph = l.shift(1).rolling(p["lb"]).min(), h.shift(1).rolling(p["lb"]).max()
    up, dn = (l < pl) & (cl > pl) & (cl > o), (h > ph) & (cl < ph) & (cl < o)
    dist = np.where(up, (cl - l) + 0.2 * atr, np.where(dn, (h - cl) + 0.2 * atr, 1.5 * atr))
    return _pack(up, dn, np.clip(dist, 0.5 * atr.values, p["cap"] * atr.values), p["rr"])


def f_squeeze(c, p):
    cl, atr, bw = c.df["close"], c.ind["atr"], c.ind["bw_rank"]
    mid = c.get(("bbm",), lambda: cl.rolling(20).mean())
    sd = c.get(("bbs",), lambda: cl.rolling(20).std(ddof=0))
    sq = bw.shift(1) <= p["rank"]
    return _pack(sq & (cl > mid + 2 * sd), sq & (cl < mid - 2 * sd), p["stop"] * atr, p["rr"])


def f_hour_drift(c, p):
    """Fixed-direction trade at one server-time hour (session/seasonality effect). No indicator involved."""
    atr = c.ind["atr"]
    on = c.hour == p["hour"]
    return _pack(on & (p["side"] > 0), on & (p["side"] < 0), 1.5 * atr, p["rr"])


def f_vol_mom(c, p):
    cl, atr = c.df["close"], c.ind["atr"]
    ret = cl - cl.shift(p["lb"])
    hi = c.ind["atr_rank"] >= p["rank"]
    return _pack(hi & (ret > 0), hi & (ret < 0), 1.5 * atr, p["rr"])


FAMILIES: Dict[str, Tuple[Callable, dict]] = {
    "ema_pullback": (f_ema_pullback, {"trend": [100, 200], "pull": [10, 20, 30], "stop": [1.0, 1.5, 2.0], "rr": [1.5, 2.0, 3.0]}),
    "donchian": (f_donchian, {"n": [10, 20, 40, 80], "adx": [0, 20, 25], "stop": [1.5, 2.0], "rr": [2.0, 3.0]}),
    "range_rev": (f_range_rev, {"rsi": [25, 30, 35], "adx_max": [18, 22], "bb": [2.0, 2.5], "rr": [1.0, 1.5]}),
    "sweep": (f_sweep, {"lb": [10, 20, 40], "cap": [2.0, 3.0], "rr": [1.5, 2.0, 3.0]}),
    "squeeze": (f_squeeze, {"rank": [0.1, 0.2, 0.3], "stop": [1.5, 2.0], "rr": [2.0, 3.0]}),
    "hour_drift": (f_hour_drift, {"hour": list(range(24)), "side": [1, -1], "rr": [1.5]}),
    "vol_mom": (f_vol_mom, {"lb": [12, 24, 48], "rank": [0.5, 0.8], "rr": [2.0, 3.0]}),
}


def config_id(family: str, params: dict) -> str:
    return family + ":" + ",".join(f"{k}={params[k]}" for k in sorted(params))


def grid() -> List[Tuple[str, dict]]:
    out = []
    for fam, (_, space) in FAMILIES.items():
        keys = list(space)
        for vals in itertools.product(*[space[k] for k in keys]):
            out.append((fam, dict(zip(keys, vals))))
    return out


# ============================================================================================ trade generation
@dataclass
class Book:
    symbol: str
    df: pd.DataFrame
    cache: Cache
    ids: List[str]
    fam: List[str]
    entry_ep: List[np.ndarray]      # per config: entry epoch of each trade
    exit_ep: List[np.ndarray]
    r: List[np.ndarray]
    side: List[np.ndarray]


def build_book(symbol: str, df: pd.DataFrame, cost_mult: float = 1.0, configs=None) -> Book:
    cache = Cache(df)
    sp = df["spread_price"].values * cost_mult
    ids, fams, E, X, R, S = [], [], [], [], [], []
    ep = df["epoch"].values
    for fam, p in (configs or grid()):
        fn = FAMILIES[fam][0]
        sg = {"x": fn(cache, p)}
        tr = se.backtest(df, cache.ind, sg, BAR_SECONDS, sp, 0.03 * cost_mult)
        ids.append(config_id(fam, p))
        fams.append(fam)
        E.append(np.array([ep[t.entry_i] for t in tr], dtype=np.int64))
        X.append(np.array([t.exit_close_epoch for t in tr], dtype=np.int64))
        R.append(np.array([t.r for t in tr], float))
        S.append(np.array([t.side for t in tr], int))
    return Book(symbol, df, cache, ids, fams, E, X, R, S)


def timing_null(df, cache, fam, params, t0, t1, shifts=40, seed=3):
    """Drift/beta control: replay the same signal circularly shifted in time (keeps side mix, clustering, stops) and
    compare mean R inside [t0, t1). Returns the share of shifted replays that did at least as well (a p-value)."""
    fn = FAMILIES[fam][0]
    side, dist, rr = fn(cache, params)
    ep, sp = df["epoch"].values, df["spread_price"].values
    rng = np.random.default_rng(seed)

    def mean_r(sd):
        tr = se.backtest(df, cache.ind, {"x": (sd, dist, rr)}, BAR_SECONDS, sp, 0.03)
        r = [t.r for t in tr if t0 <= ep[t.entry_i] and t.exit_close_epoch < t1]
        return float(np.mean(r)) if len(r) >= 10 else float("nan")

    obs = mean_r(side)
    n = len(side)
    nulls = []
    for _ in range(shifts):
        k = int(rng.integers(500, n - 500))
        nulls.append(mean_r(np.roll(side, k)))
    nulls = np.array([x for x in nulls if np.isfinite(x)])
    p = float((np.sum(nulls >= obs) + 1) / (len(nulls) + 1)) if len(nulls) else 1.0
    return {"observed": obs, "null_mean": float(nulls.mean()) if len(nulls) else None, "null_p": p}


# ============================================================================================ statistics
def boot_ci(r, level=0.95, reps=1500, seed=11):
    r = np.asarray(r, float)
    if len(r) < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed + len(r))
    m = rng.choice(r, size=(reps, len(r)), replace=True).mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(m, a)), float(np.quantile(m, 1 - a))


def summarize(r, level=0.95):
    r = np.asarray(r, float)
    n = len(r)
    if n == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None, "win": None, "t": None}
    lo, hi = boot_ci(r, level)
    t = float(r.mean() / (r.std(ddof=1) / math.sqrt(n))) if n > 2 and r.std(ddof=1) > 0 else 0.0
    return {"n": int(n), "mean": float(r.mean()), "lo": lo, "hi": hi, "win": float((r > 0).mean()), "t": t}


def daily_matrix(book: Book, t0: int, t1: int) -> np.ndarray:
    """K x T matrix of summed R by exit day inside [t0, t1) (zeros on idle days)."""
    d0, d1 = t0 // DAY, t1 // DAY
    T = max(1, d1 - d0)
    M = np.zeros((len(book.ids), T))
    for k in range(len(book.ids)):
        m = (book.exit_ep[k] >= t0) & (book.exit_ep[k] < t1)
        if m.any():
            M[k] += np.bincount((book.exit_ep[k][m] // DAY - d0).astype(int), weights=book.r[k][m], minlength=T)[:T]
    return M


def spa_pvalue(D: np.ndarray, reps=400, block=10, seed=5) -> dict:
    """Hansen (2005) SPA_c with a stationary bootstrap; benchmark = not trading. D: K x T daily R."""
    K, T = D.shape
    active = D.std(axis=1) > 0
    if active.sum() == 0 or T < 60:
        return {"p": 1.0, "k": int(K), "best_t": 0.0}
    D = D[active]
    K = D.shape[0]
    mean, sd = D.mean(axis=1), D.std(axis=1, ddof=1) + 1e-12
    tstat = math.sqrt(T) * mean / sd
    stat = max(tstat.max(), 0.0)
    mu = np.where(tstat <= -math.sqrt(2 * math.log(math.log(max(T, 16)))), mean, 0.0)
    rng = np.random.default_rng(seed)
    idx = np.zeros((reps, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, T, reps)
    jump = rng.random((reps, T)) < 1.0 / block
    new = rng.integers(0, T, (reps, T))
    for j in range(1, T):
        idx[:, j] = np.where(jump[:, j], new[:, j], (idx[:, j - 1] + 1) % T)
    cnt = 0
    for b in range(reps):
        mb = D[:, idx[b]].mean(axis=1)
        z = math.sqrt(T) * (mb - mean + mu) / sd
        cnt += max(z.max(), 0.0) >= stat
    return {"p": float((cnt + 1) / (reps + 1)), "k": int(K), "best_t": float(stat)}


# ============================================================================================ walk-forward of the selection procedure
def month_epochs(start_ep: int, months: float) -> int:
    return int(months * 30.4375 * DAY)


def folds(dev_start: int, dev_end: int):
    t = dev_start + month_epochs(dev_start, INIT_TRAIN_MONTHS)
    out = []
    while t + month_epochs(t, TEST_MONTHS) * 0.5 < dev_end:
        te = min(dev_end, t + month_epochs(t, TEST_MONTHS))
        out.append((t, te))
        t = te
    return out


def walk_forward(book: Book, dev_start: int, dev_end: int) -> dict:
    fl = folds(dev_start, dev_end)
    emb = EMBARGO_BARS * BAR_SECONDS
    oos_r, per_fold, avg_r = [], [], []
    for (ts, te) in fl:
        best, best_t = None, -1e9
        for k in range(len(book.ids)):
            m = (book.exit_ep[k] <= ts - emb) & (book.entry_ep[k] >= dev_start)
            r = book.r[k][m]
            if len(r) < MIN_TRADES_SELECT or r.std(ddof=1) == 0:
                continue
            t = r.mean() / (r.std(ddof=1) / math.sqrt(len(r)))
            if t > best_t:
                best, best_t = k, t
        if best is None:
            per_fold.append({"test": [ts, te], "chosen": None, "n": 0, "mean": None})
            continue
        m = (book.entry_ep[best] >= ts) & (book.entry_ep[best] < te)
        r = book.r[best][m]
        oos_r.append(r)
        per_fold.append({"test": [int(ts), int(te)], "chosen": book.ids[best], "in_sample_t": round(float(best_t), 2), "n": int(len(r)),
                         "mean": (float(r.mean()) if len(r) else None)})
        allr = [book.r[k][(book.entry_ep[k] >= ts) & (book.entry_ep[k] < te)] for k in range(len(book.ids))]
        avg_r.append(np.mean([x.mean() for x in allr if len(x)]) if any(len(x) for x in allr) else 0.0)
    pooled = np.concatenate(oos_r) if oos_r else np.array([])
    fm = [f["mean"] for f in per_fold if f["mean"] is not None]
    return {"folds": per_fold, "pooled": summarize(pooled), "fold_positive_share": (float(np.mean([m > 0 for m in fm])) if fm else 0.0),
            "avg_config_mean_oos": (float(np.mean(avg_r)) if avg_r else None), "pooled_r": pooled}


# ============================================================================================ registry (trial counting + holdout looks)
class Registry:
    def __init__(self, path: Optional[str]):
        self.path = path
        self.events: List[dict] = []
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.events = [json.loads(l) for l in f if l.strip()]

    def log(self, ev: dict):
        ev = dict(ev, ts=time.time())
        self.events.append(ev)
        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    def n_trials(self, symbol=None):
        return sum(e.get("trials", 0) for e in self.events if e["type"] == "search" and (symbol is None or e.get("symbol") == symbol))

    def holdout_looks(self, finalist):
        return [e for e in self.events if e["type"] == "holdout" and e["finalist"] == finalist]


# ============================================================================================ main search
def split_points(df: pd.DataFrame, holdout_months=HOLDOUT_MONTHS):
    last = int(df["epoch"].iloc[-1]) + BAR_SECONDS
    return int(df["epoch"].iloc[0]), last - month_epochs(last, holdout_months), last


def search(frames: Dict[str, pd.DataFrame], registry: Optional[Registry] = None, holdout_months=HOLDOUT_MONTHS, spa_reps=400, progress=None) -> dict:
    registry = registry or Registry(None)
    scr = {s: screen_symbol(d) for s, d in frames.items()}
    universe = [s for s in frames if scr[s]["pass"]]
    res = {"screen": scr, "universe": universe, "symbols": {}, "holdout_months": holdout_months, "config_count": len(grid()), "version": 1}
    books = {}
    for s in universe:
        df = frames[s]
        t_start, dev_end, last = split_points(df, holdout_months)
        if progress:
            progress(f"{s}: building {len(grid())} trials")
        book = build_book(s, df)
        books[s] = book
        registry.log({"type": "search", "symbol": s, "trials": len(book.ids), "dev_end": dev_end})
        wf = walk_forward(book, t_start, dev_end)
        D = daily_matrix(book, t_start, dev_end)
        spa = spa_pvalue(D, reps=spa_reps)
        dev_stats = []
        for k in range(len(book.ids)):
            m = (book.entry_ep[k] >= t_start) & (book.exit_ep[k] < dev_end)
            st = summarize(book.r[k][m], 0.95)
            st["id"], st["family"] = book.ids[k], book.fam[k]
            dev_stats.append(st)
        ok = [x for x in dev_stats if x["n"] >= MIN_TRADES_SELECT]
        top = sorted(ok, key=lambda x: -x["t"])[:3]
        breadth = {}
        for fam in FAMILIES:
            fm = [x["mean"] for x in ok if x["family"] == fam]
            breadth[fam] = float(np.mean([m > 0 for m in fm])) if fm else None
        res["symbols"][s] = {"dev": [t_start, dev_end], "wf": {k: v for k, v in wf.items() if k != "pooled_r"}, "spa": spa, "top": top, "breadth": breadth,
                             "n_trials_registry": registry.n_trials(s), "_pooled_r": wf["pooled_r"]}
    # stress + replication for finalists
    for s in universe:
        info = res["symbols"][s]
        info["finalists"] = []
        for x in info["top"]:
            fam, params = _parse_id(x["id"])
            cost2 = build_book(s, frames[s], cost_mult=2.0, configs=[(fam, params)])
            t_start, dev_end = info["dev"]
            m = (cost2.entry_ep[0] >= t_start) & (cost2.exit_ep[0] < dev_end)
            st2 = summarize(cost2.r[0][m])
            rep = []
            for s2 in universe:
                if s2 == s:
                    continue
                b2 = books[s2]
                k = b2.ids.index(x["id"])
                a, bnd = res["symbols"][s2]["dev"]
                mm = (b2.entry_ep[k] >= a) & (b2.exit_ep[k] < bnd)
                rep.append({"symbol": s2, **{kk: v for kk, v in summarize(b2.r[k][mm]).items() if kk in ("n", "mean", "lo")}})
            pos = sum(1 for r in rep if r["n"] >= MIN_TRADES_SELECT and r["mean"] is not None and r["mean"] > 0)
            tn = timing_null(frames[s], books[s].cache, fam, params, t_start, dev_end)
            finalist = {"timing_null": tn, "id": x["id"], "dev": {k: x[k] for k in ("n", "mean", "lo", "hi", "t")}, "cost2x": {k: st2[k] for k in ("n", "mean", "lo")}, "replication": rep,
                        "replication_positive": pos, "replication_of": len(rep), "family_breadth": info["breadth"].get(fam)}
            finalist["gates"] = _gates(info, finalist)
            info["finalists"].append(finalist)
        info["verdict"] = "candidate" if any(f["gates"]["all"] for f in info["finalists"]) else "no_edge"
        info.pop("_pooled_r", None)
    res["overall"] = "candidate" if any(i["verdict"] == "candidate" for i in res["symbols"].values()) else "no_edge"
    res["digest"] = digest(res)
    return res


def _parse_id(cid: str):
    fam, rest = cid.split(":", 1)
    space = FAMILIES[fam][1]
    p = {}
    for kv in rest.split(","):
        k, v = kv.split("=")
        sample = space[k][0]
        p[k] = type(sample)(float(v)) if isinstance(sample, (int, float)) and not isinstance(sample, bool) else v
        if isinstance(sample, int):
            p[k] = int(float(v))
    return fam, p


def _gates(info: dict, f: dict) -> dict:
    wf = info["wf"]
    g = {"spa_p_below_alpha": info["spa"]["p"] <= SPA_ALPHA,
         "walk_forward_oos_positive_ci": bool(wf["pooled"]["n"] >= MIN_TRADES_SELECT and wf["pooled"]["lo"] is not None and wf["pooled"]["lo"] > 0),
         "folds_positive": wf["fold_positive_share"] >= FOLD_POSITIVE_MIN,
         "survives_2x_costs": bool(f["cost2x"]["n"] >= MIN_TRADES_SELECT and f["cost2x"]["mean"] is not None and f["cost2x"]["mean"] > 0),
         "beats_timing_null": f["timing_null"]["null_p"] <= 0.05,
         "replicates_elsewhere": (f["replication_of"] == 0) or (f["replication_positive"] >= math.ceil(f["replication_of"] / 2)),
         "family_breadth": (f["family_breadth"] or 0) >= BREADTH_MIN}
    g["all"] = all(g.values())
    return g


# ============================================================================================ sealed holdout
def holdout(frames: Dict[str, pd.DataFrame], search_result: dict, finalist: str, registry: Registry, allow_repeat=False) -> dict:
    """One look at the sealed window for one finalist on every symbol where it was searched. Refused if already looked at."""
    prev = registry.holdout_looks(finalist)
    if prev and not allow_repeat:
        raise RuntimeError(f"holdout already used for {finalist}; a second look would no longer be out-of-sample")
    fam, params = _parse_id(finalist)
    out = {}
    pooled = []
    for s in search_result["universe"]:
        df = frames[s]
        _, dev_end, last = split_points(df, search_result["holdout_months"])
        book = build_book(s, df, configs=[(fam, params)])
        m = (book.entry_ep[0] >= dev_end) & (book.entry_ep[0] < last)
        r = book.r[0][m]
        out[s] = summarize(r)
        pooled.append(r)
    allr = np.concatenate(pooled) if pooled else np.array([])
    pooled_s = summarize(allr)
    passed = bool(pooled_s["n"] >= MIN_TRADES_SELECT and pooled_s["lo"] is not None and pooled_s["lo"] > 0)
    registry.log({"type": "holdout", "finalist": finalist, "passed": passed, "pooled": pooled_s})
    return {"finalist": finalist, "per_symbol": out, "pooled": pooled_s, "passed": passed,
            "verdict": "holdout_passed" if passed else "no_edge",
            "note": "The sealed window has now been looked at once for this finalist. Next: forward/paper test; do not search again on this window."}


# ============================================================================================ reporting
def _d(ep):
    return time.strftime("%Y-%m-%d", time.gmtime(ep))


def digest(res: dict, max_chars: int = 1800) -> str:
    """Short Persian text for the agent / the fine-tuned model."""
    L = [f"گزارش آزمایشگاه لبه ({res['config_count']} پیکربندی ثابت در ۷ خانواده؛ نتیجهٔ کلی: {'نامزد (اثبات‌نشده)' if res['overall'] == 'candidate' else 'لبه‌ای پیدا نشد'})"]
    excl = [f"{s}({'، '.join(v['reasons'])})" for s, v in res["screen"].items() if not v["pass"]]
    if excl:
        L.append("کنار گذاشته در غربال: " + "؛ ".join(excl))
    for s, i in res["symbols"].items():
        p = i["wf"]["pooled"]
        mean = "-" if p["mean"] is None else f"{p['mean']:+.2f}R"
        lo = "-" if p["lo"] is None else f"{p['lo']:+.2f}"
        L.append(f"- {s}: SPA p={i['spa']['p']:.2f} (K={i['spa']['k']}) | walk-forward بیرون‌نمونه n={p['n']} میانگین {mean} حد پایین CI95% {lo} | فولدهای مثبت {i['wf']['fold_positive_share'] * 100:.0f}% | حکم: {i['verdict']}")
        for f in i["finalists"][:1]:
            bad = [k for k, v in f["gates"].items() if k != "all" and not v]
            L.append(f"  بهترین بر پایهٔ توسعه: {f['id']} (t={f['dev']['t']:.2f}) | گیت‌های ردشده: {', '.join(bad) if bad else 'هیچ'}")
    L.append("پنجرهٔ مهرشده فقط یک‌بار برای هر نامزد باز می‌شود؛ حتی نامزد قبول‌شده سودآوری آینده را تضمین نمی‌کند.")
    t = "\n".join(L)
    return t if len(t) <= max_chars else t[:max_chars - 1] + "…"


def save(res: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "latest_report.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with open(os.path.join(out_dir, "latest_digest.txt"), "w", encoding="utf-8") as f:
        f.write(res["digest"])


# ============================================================================================ CLI
def _frames(a):
    syms = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    out = {}
    for s in syms:
        try:
            out[s] = load_mt5(s, a.bars) if a.mt5 else load_csv(a.candles, s)
        except Exception as e:  # noqa: BLE001
            print(f"skip {s}: {e}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["screen", "search", "holdout"])
    ap.add_argument("--candles", default="candles")
    ap.add_argument("--mt5", action="store_true")
    ap.add_argument("--bars", type=int, default=60000)
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--out", default="lab_out")
    ap.add_argument("--finalist")
    ap.add_argument("--holdout-months", type=int, default=HOLDOUT_MONTHS)
    a = ap.parse_args(argv)
    frames = _frames(a)
    if a.cmd == "screen":
        print(json.dumps({s: screen_symbol(d) for s, d in frames.items()}, ensure_ascii=False, indent=1))
        return 0
    reg = Registry(os.path.join(a.out, "registry.jsonl"))
    if a.cmd == "search":
        res = search(frames, reg, a.holdout_months, progress=print)
        save(res, a.out)
        print(res["digest"])
        return 0
    with open(os.path.join(a.out, "latest_report.json"), encoding="utf-8") as f:
        res = json.load(f)
    h = holdout(frames, res, a.finalist, reg)
    with open(os.path.join(a.out, f"holdout_{hashlib.sha1(a.finalist.encode()).hexdigest()[:8]}.json"), "w", encoding="utf-8") as f:
        json.dump(h, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(h, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
