"""LONG / SHORT / NO_TRADE labeling (triple barrier) + scenario outcomes.

Rules (all documented in the run record):
* decision = M15 bar close; ATR14, spread known then. Entry = OPEN of the next M15 bar (must be exactly +15 min).
* SL = sl_atr_mult x ATR14, TP = rr x SL, horizon = horizon_bars M15 bars (entry bar included). Bid-chart prices.
* LONG: long TP hit before SL.  SHORT: short TP before SL.  Otherwise NO_TRADE (SL first, timeout, or both-fail).
* Same-bar double touch -> M5 replay; if the SAME M5 bar still touches both -> worst case (SL first), ambiguous=1.
* SL gaps (bar opens beyond the stop) fill at the open (worse than the stop); TP gaps fill at the TP level (conservative).
* Costs: CostProfile (spread from entry-decision bar, commission, slippage) -> net R.  Windows crossing a weekend are excluded.
"""
from __future__ import annotations

import json
import os
import time as _time

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

from broker_context.cost_profiles import LabelRunRegistry

from . import s5config as S
from .data import load_m15, load_m5

WEEKEND_GAP_MIN = 600


def net_r_vector(profile, gross_move, sl_dist, spread_points):
    """Vectorized twin of broker_context.cost_profiles.historical_net_r (equality is unit-tested)."""
    c = profile.contract_assumptions
    pt, cs = c["point"], c["contract_size"]
    sp_pts = np.where((spread_points > 0) & np.isfinite(spread_points), spread_points, np.nan)
    if profile.spread["source"] == "bar_spread":
        fb = profile.spread.get("fallback_points")
        sp_pts = np.where(np.isnan(sp_pts), fb, sp_pts)
    else:
        sp_pts = np.full_like(gross_move, profile.spread["fixed_points"], dtype=float)
    sp_price = sp_pts * pt
    cm = profile.commission
    com = 0.0 if cm.get("model") == "none" else cm["amount"] * (2 if cm["basis"] == "per_lot_per_side" else 1) / cs
    slip = profile.slippage["value"] * 2
    net = gross_move - sp_price - com - slip
    return net / sl_dist, gross_move / sl_dist


def _side(open_, high, low, close, entry, sl_dist, rr, side, H, win_start):
    """first-touch analysis for one side on windows [N,H]."""
    hw = swv(high, H)[win_start]
    lw = swv(low, H)[win_start]
    ow = swv(open_, H)[win_start]
    cw = swv(close, H)[win_start]
    n = len(entry)
    sgn = 1 if side == "long" else -1
    sl_px = entry - sgn * sl_dist
    tp_px = entry + sgn * rr * sl_dist
    if side == "long":
        sl_m, tp_m = lw <= sl_px[:, None], hw >= tp_px[:, None]
    else:
        sl_m, tp_m = hw >= sl_px[:, None], lw <= tp_px[:, None]
    fs = np.where(sl_m.any(1), sl_m.argmax(1), H)
    ft = np.where(tp_m.any(1), tp_m.argmax(1), H)
    outcome = np.full(n, "timeout", dtype=object)
    outcome[(ft < fs)] = "tp"
    outcome[(fs < ft)] = "sl"
    amb = (fs == ft) & (fs < H)
    outcome[amb] = "ambiguous"
    return dict(hw=hw, lw=lw, ow=ow, cw=cw, sl_px=sl_px, tp_px=tp_px, fs=fs, ft=ft, outcome=outcome, amb=amb)


def _resolve_m5(m5_time_ns, m5_o, m5_h, m5_l, bar_start_ns, side, sl_px, tp_px):
    """-> ('tp'|'sl', same_m5_bar_flag). Worst case = SL if the same M5 bar touches both."""
    a = np.searchsorted(m5_time_ns, bar_start_ns)
    b = np.searchsorted(m5_time_ns, bar_start_ns + 15 * 60 * 10**9)
    if b <= a:
        return "sl", True, "no_m5_data"
    for k in range(a, b):
        if side == "long":
            s, t = m5_l[k] <= sl_px, m5_h[k] >= tp_px
        else:
            s, t = m5_h[k] >= sl_px, m5_l[k] <= tp_px
        if s and t:
            return "sl", True, "same_m5_bar"
        if s:
            return "sl", False, "m5"
        if t:
            return "tp", False, "m5"
    return "sl", True, "m5_no_touch_inconsistent"


def build_labels(cfg=None, verbose=True):
    cfg = dict(S.LABEL, **(cfg or {}))
    H, k, rr = cfg["horizon_bars"], cfg["sl_atr_mult"], cfg["rr"]
    cols = ["time", "open", "high", "low", "close", "spread", "spread_known", "atr_14", "prev_day_high", "prev_day_low", "trend_state"]
    d = load_m15(cols)
    m5 = load_m5()
    m5_ns = m5["time"].values.astype("datetime64[ns]").astype(np.int64)
    m5o, m5h, m5l = m5["open"].values, m5["high"].values, m5["low"].values
    n = len(d)
    t = d["time"].values.astype("datetime64[ns]")
    o, h, l, c = (d[x].values.astype(float) for x in ("open", "high", "low", "close"))
    atr, spr = d["atr_14"].values.astype(float), d["spread"].values.astype(float)
    N = n - H - 1                                      # decision indices 0..N-1 have a full window
    idx = np.arange(N)
    entry_idx = idx + 1
    entry = o[entry_idx]
    sl_dist = k * atr[:N]
    adj = (t[entry_idx] - t[idx]) == np.timedelta64(15, "m")
    gaps = np.diff(t.astype("datetime64[m]").astype(np.int64))
    big = np.concatenate([[0], np.cumsum(gaps > WEEKEND_GAP_MIN)])
    crosses_weekend = big[idx + H] != big[idx + 1]
    ok_atr = np.isfinite(atr[:N]) & (atr[:N] > 0)
    sl_dist_safe = np.where(ok_atr, sl_dist, np.nan)
    prof_ids = {nm: S.make_profile(nm) for nm in S.COST_SCENARIOS}
    out = pd.DataFrame({"time": t[:N], "idx": idx, "bar_close_time": t[:N] + np.timedelta64(15, "m"), "entry_time": t[entry_idx],
                        "entry_price": entry, "atr_14": atr[:N], "sl_dist": sl_dist_safe, "spread_pts_decision": spr[:N],
                        "spread_known_decision": d["spread_known"].values[:N], "trend_state": d["trend_state"].values[:N]})
    spread_entry = spr[entry_idx]
    for side in ("long", "short"):
        r = _side(o, h, l, c, entry, sl_dist_safe, rr, side, H, slice(1, N + 1))
        sgn = 1 if side == "long" else -1
        outcome = r["outcome"].copy()
        fs, ft = r["fs"].copy(), r["ft"].copy()
        how = np.full(N, "", dtype=object)
        flag = np.zeros(N, dtype=int)
        for i in np.where(r["amb"] & ok_atr)[0]:
            bar = int(i + 1 + fs[i])
            res, same, why = _resolve_m5(m5_ns, m5o, m5h, m5l, t[bar].astype("datetime64[ns]").astype(np.int64), side, r["sl_px"][i], r["tp_px"][i])
            outcome[i] = res
            how[i], flag[i] = why, int(same)
            if res == "tp":
                ft[i], fs[i] = fs[i], H
            else:
                fs[i], ft[i] = fs[i], H
        ex = np.where(outcome == "tp", ft, np.where(outcome == "sl", fs, H - 1)).astype(int)
        ex = np.minimum(ex, H - 1)
        rows = np.arange(N)
        gap_open = r["ow"][rows, ex]
        gross = np.where(outcome == "tp", rr * sl_dist_safe, 0.0)
        sl_fill = np.where(sgn * (gap_open - entry) < -sl_dist_safe, sgn * (gap_open - entry), -sl_dist_safe)   # gap -> worse fill
        gross = np.where(outcome == "sl", sl_fill, gross)
        gross = np.where(outcome == "timeout", sgn * (r["cw"][rows, H - 1] - entry), gross)
        gross = np.where(outcome == "ambiguous", -sl_dist_safe, gross)
        # MFE / MAE (bar-granular, inclusive of the exit bar)
        pos = np.arange(H)[None, :] <= ex[:, None]
        if side == "long":
            fav = np.where(pos, r["hw"], -np.inf).max(1) - entry
            adv = entry - np.where(pos, r["lw"], np.inf).min(1)
        else:
            fav = entry - np.where(pos, r["lw"], np.inf).min(1)
            adv = np.where(pos, r["hw"], -np.inf).max(1) - entry
        out[f"{side}_outcome"] = outcome
        out[f"{side}_bars_to_exit"] = ex + 1
        out[f"{side}_gross_r"] = gross / sl_dist_safe
        for nm, pf in prof_ids.items():
            nr, _ = net_r_vector(pf, gross, sl_dist_safe, spread_entry if False else spr[:N])
            out[f"{side}_net_r_{nm}"] = nr
        out[f"{side}_mfe_r"] = fav / sl_dist_safe
        out[f"{side}_mae_r"] = adv / sl_dist_safe
        out[f"{side}_ambiguous"] = flag
        out[f"{side}_ambiguity_resolution"] = how
        out[f"{side}_sl_gap"] = (outcome == "sl") & (sgn * (gap_open - entry) < -sl_dist_safe)
    # exclusions (causal / structural reasons)
    reason = np.full(N, "", dtype=object)
    reason[idx < cfg["warmup_rows"]] = "warmup"
    reason[~ok_atr & (reason == "")] = "atr_unavailable"
    reason[~adj & (reason == "")] = "no_adjacent_entry_bar"
    reason[crosses_weekend & (reason == "")] = "horizon_crosses_weekend"
    out["valid"] = reason == ""
    out["exclusion_reason"] = reason
    pf = S.PRIMARY_SCENARIO
    lw = (out["long_outcome"] == "tp") & (out[f"long_net_r_{pf}"] >= cfg["min_net_r"])
    sw = (out["short_outcome"] == "tp") & (out[f"short_net_r_{pf}"] >= cfg["min_net_r"])
    label = np.where(lw & ~sw, "LONG", np.where(sw & ~lw, "SHORT", "NO_TRADE"))
    ntr = np.full(N, "", dtype=object)
    wide = (out["spread_known_decision"] == 1) & (out["spread_pts_decision"] * 0.01 > cfg["max_spread_atr"] * out["atr_14"])
    ntr[label == "NO_TRADE"] = "no_barrier_edge"
    ntr[(label == "NO_TRADE") & wide.values] = "spread_too_wide"
    ntr[lw.values & sw.values] = "conflicting_outcomes"
    label = np.where(wide.values & (label != "NO_TRADE"), "NO_TRADE", label)          # execution filter wins
    ntr[wide.values & (np.where(lw & ~sw, "LONG", np.where(sw & ~lw, "SHORT", "NO_TRADE")) != "NO_TRADE")] = "spread_too_wide"
    out["label"] = label
    out["no_trade_reason"] = ntr
    # scenario outcomes over the same horizon (hindsight; evaluation / probability targets only)
    hw, lw_ = swv(h, H)[1:N + 1], swv(l, H)[1:N + 1]
    cw = swv(c, H)[1:N + 1]
    a = sl_dist_safe / k
    out["up_exc_atr"] = (hw.max(1) - entry) / a
    out["dn_exc_atr"] = (entry - lw_.min(1)) / a
    out["end_ret_atr"] = (cw[:, -1] - entry) / a
    mv = S.SCENARIO["move_atr"] * a
    up_m, dn_m = hw >= (entry + mv)[:, None], lw_ <= (entry - mv)[:, None]
    fu = np.where(up_m.any(1), up_m.argmax(1), H)
    fd = np.where(dn_m.any(1), dn_m.argmax(1), H)
    first = np.where(fu < fd, 1, np.where(fd < fu, -1, 0))
    out["first_touch_dir"] = first                       # +1 up first, -1 down first, 0 none/same bar
    out["first_touch_bars"] = np.where(first == 1, fu + 1, np.where(first == -1, fd + 1, 0))
    for nm, col in (("pdh", "prev_day_high"), ("pdl", "prev_day_low")):
        lvl = d[col].values[:N]
        out[f"reach_{nm}"] = (hw.max(1) >= lvl) if nm == "pdh" else (lw_.min(1) <= lvl)
        out[f"{nm}_above_entry" if nm == "pdh" else f"{nm}_below_entry"] = (lvl > entry) if nm == "pdh" else (lvl < entry)
    out["time"] = pd.to_datetime(out["time"])
    out["entry_time"] = pd.to_datetime(out["entry_time"])
    out["bar_close_time"] = pd.to_datetime(out["bar_close_time"])
    return out, cfg


def run(verbose=True):
    t0 = _time.time()
    lab, cfg = build_labels()
    prof = S.make_profile(S.PRIMARY_SCENARIO)
    os.makedirs(S.DIR_LABELS, exist_ok=True)
    reg = LabelRunRegistry(S.RUNS_DIR)
    extra = {"label_parameters": cfg, "scenario": S.SCENARIO, "other_cost_scenarios": {k: S.make_profile(k).profile_id for k in S.COST_SCENARIOS},
             "rows": int(len(lab)), "valid_rows": int(lab["valid"].sum()),
             "label_distribution_valid": lab[lab.valid]["label"].value_counts().to_dict(),
             "ambiguous_long": int(lab["long_ambiguous"].sum()), "ambiguous_short": int(lab["short_ambiguous"].sum()),
             "primary_cost_scenario": S.PRIMARY_SCENARIO, "all_cost_assumptions_unverified": True}
    try:
        rid = reg.register(S.DATASET_VERSION, S.LABEL_CONFIG_VERSION, prof, extra)
    except FileExistsError:
        rid = reg.run_id(S.DATASET_VERSION, S.LABEL_CONFIG_VERSION, prof)
        print("run exists (never rewritten); labels CSV regenerated deterministically for", rid)
    path = os.path.join(S.DIR_LABELS, f"labels_{S.LABEL_CONFIG_VERSION}.csv")
    lab.to_csv(path, index=False, float_format="%.6f")
    if verbose:
        print(rid, f"{len(lab)} rows, {time_str(t0)}", json.dumps(extra["label_distribution_valid"]))
    return lab


def time_str(t0):
    return f"{_time.time() - t0:.0f}s"


if __name__ == "__main__":
    run()
