"""Chronological trade-path reconstruction & management simulator (M5 replay).

Every record at step t uses only information at the CLOSE of M5 bar t (and M15 bars closed by then).
Intrabar order is unknown: if one M5 bar touches the stop and any target the worst case (stop first)
is used and flagged ``ambiguous``.  Stops change only at bar close and apply from the NEXT bar.
Market-type exits decided at a bar close (time / volatility / structure invalidation) fill at the NEXT M5 open.
Costs follow the historical CostProfile: entry-side commission+slippage on the full size, exit-side
commission+slippage and the spread pro rata for every closed fraction, so a single full exit equals historical_net_r.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from broker_context.cost_profiles import historical_net_r

from . import s5config as S
from .data import load_m5, load_m15

POLICIES: Dict[str, Dict[str, Any]] = {
    "fixed": dict(tp_r=2.0),
    "break_even": dict(tp_r=2.0, be_at_r=1.0, be_offset_r=0.1),
    "atr_trail": dict(tp_r=3.0, trail_start_r=1.0, trail_atr_mult=1.5),
    "structure_trail": dict(tp_r=3.0, trail_start_r=1.0, structure_trail=True, struct_buffer_atr=0.1),
    "partial_be": dict(tp_r=2.0, partials=[(1.0, 0.5)], be_after_first_partial=True),
    "multi_tp": dict(tp_r=3.0, partials=[(1.0, 1 / 3), (2.0, 1 / 3)]),
    "time_exit": dict(tp_r=2.0, max_hold_m5=48),
    "volatility_exit": dict(tp_r=2.0, vol_exit_atr_ratio=1.6),
    "structure_invalidation": dict(tp_r=2.0, structure_invalidation=True),
}


@dataclass
class SimContext:
    m5_t: np.ndarray      # int64 ns open times
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    spread: np.ndarray
    m15_idx: np.ndarray   # for each M5 bar: index of the latest M15 bar CLOSED at this M5 bar's close
    m15: pd.DataFrame     # M15 columns (atr_14, atr_ratio, last_swing_high, last_swing_low, close, time)


def make_context() -> SimContext:
    m5 = load_m5()
    m15 = load_m15(["time", "close", "atr_14", "atr_ratio", "last_swing_high", "last_swing_low", "trend_state", "spread"])
    m5_t = m5["time"].values.astype("datetime64[ns]").astype(np.int64)
    m15_close = (m15["time"].values.astype("datetime64[ns]") + np.timedelta64(15, "m")).astype(np.int64)
    m5_close = m5_t + 5 * 60 * 10**9
    j = np.searchsorted(m15_close, m5_close, side="right") - 1
    return SimContext(m5_t, m5["open"].values, m5["high"].values, m5["low"].values, m5["close"].values,
                      m5["spread"].values.astype(float), j, m15)


def cost_parts(profile, spread_points: float) -> Dict[str, float]:
    r = historical_net_r(profile, 0.0, 1.0, spread_points=spread_points, atr=1.0)
    return {"spread": r["spread_price"], "commission": r["commission_price"], "slippage": r["slippage_price"]}


def simulate_trade(ctx: SimContext, side: str, m15_idx: int, policy: Dict[str, Any], profile, entry_price: float, sl_dist: float,
                   entry_ns: int, record_every: int = 12, max_steps: int = S.MGMT_HORIZON_M5, upto_step: Optional[int] = None) -> Dict[str, Any]:
    """``upto_step`` truncates the world after that many M5 bars (used by the no-lookahead test)."""
    sgn = 1 if side == "long" else -1
    start = int(np.searchsorted(ctx.m5_t, entry_ns))
    assert ctx.m5_t[start] == entry_ns, "entry time must align with an M5 bar open"
    end = min(start + max_steps, len(ctx.m5_t))
    if upto_step is not None:
        end = min(end, start + upto_step)
    cp = cost_parts(profile, float(ctx.m15["spread"].iat[m15_idx]))   # spread of the decision bar (known at decision time)
    cost_entry = (cp["commission"] + cp["slippage"]) / 2
    cost_exit_unit = cp["spread"] + (cp["commission"] + cp["slippage"]) / 2
    sl0 = entry_price - sgn * sl_dist
    sl = sl0
    tp_r = policy.get("tp_r")
    tp = entry_price + sgn * tp_r * sl_dist if tp_r else None
    levels = [(entry_price + sgn * r * sl_dist, frac, r) for r, frac in policy.get("partials", [])]
    remaining, realized_move, cost_paid = 1.0, 0.0, cost_entry
    events: List[Dict[str, Any]] = []
    be_active = trail_active = False
    mfe = mae = 0.0
    best_close = entry_price
    inv_level = ctx.m15["last_swing_low"].iat[m15_idx] if side == "long" else ctx.m15["last_swing_high"].iat[m15_idx]
    inv_level = float(inv_level) if np.isfinite(inv_level) else None
    ambiguous = False
    records: List[Dict[str, Any]] = []
    exit_reason: Optional[str] = None
    pending_market_exit: Optional[str] = None

    def close(frac, price, reason, k):
        nonlocal remaining, realized_move, cost_paid
        frac = min(frac, remaining)
        realized_move += frac * sgn * (price - entry_price)
        cost_paid += frac * cost_exit_unit
        remaining = round(remaining - frac, 12)
        events.append({"step": k - start, "reason": reason, "fraction": round(frac, 6), "price": float(price)})

    for k in range(start, end):
        o, h, l, c = ctx.o[k], ctx.h[k], ctx.l[k], ctx.c[k]
        if pending_market_exit is not None:
            close(remaining, o, pending_market_exit, k)
            exit_reason = pending_market_exit
            break
        adverse, favor = (l, h) if side == "long" else (h, l)
        hit_sl = sgn * (adverse - sl) <= 0
        hit_levels = [lv for lv in levels if sgn * (favor - lv[0]) >= 0]
        hit_tp = tp is not None and sgn * (favor - tp) >= 0
        if hit_sl and (hit_levels or hit_tp):
            ambiguous = True                         # worst-case policy: stop first
        if hit_sl:
            px = o if sgn * (o - sl) <= 0 else sl    # gap through the stop fills at the open
            close(remaining, px, "stop_loss" if not (be_active or trail_active) else ("trailing_stop" if trail_active else "break_even_stop"), k)
            exit_reason = events[-1]["reason"]
            break
        for lv in sorted(hit_levels, key=lambda x: x[2]):
            close(lv[1], lv[0], f"partial_tp_{lv[2]:g}R", k)
            levels.remove(lv)
            if policy.get("be_after_first_partial"):
                new = entry_price + sgn * policy.get("be_offset_r", 0.1) * sl_dist
                sl = max(sl, new) if side == "long" else min(sl, new)
                be_active = True
        if hit_tp:
            close(remaining, tp, "take_profit", k)
            exit_reason = "take_profit"
            break
        # ---- bar close: update state with information available NOW
        mfe = max(mfe, sgn * (favor - entry_price) / sl_dist)
        mae = max(mae, sgn * (entry_price - adverse) / sl_dist)
        best_close = max(best_close, c) if side == "long" else min(best_close, c)
        unreal_move = realized_move + remaining * sgn * (c - entry_price)
        unreal_r = unreal_move / sl_dist
        step = k - start
        j = int(ctx.m15_idx[k])
        atr_now = float(ctx.m15["atr_14"].iat[j])
        ratio = float(ctx.m15["atr_ratio"].iat[j])
        r_close = sgn * (c - entry_price) / sl_dist
        # stop management (applies from the next bar)
        if policy.get("be_at_r") is not None and not be_active and r_close >= policy["be_at_r"]:
            new = entry_price + sgn * policy.get("be_offset_r", 0.0) * sl_dist
            sl = max(sl, new) if side == "long" else min(sl, new)
            be_active = True
        if policy.get("trail_start_r") is not None and (trail_active or r_close >= policy["trail_start_r"]):
            if policy.get("structure_trail"):
                lvl = ctx.m15["last_swing_low"].iat[j] if side == "long" else ctx.m15["last_swing_high"].iat[j]
                if np.isfinite(lvl):
                    cand = lvl - sgn * policy.get("struct_buffer_atr", 0.1) * atr_now
                    if sgn * (cand - entry_price) > 0 or trail_active:
                        new = cand
                        sl2 = max(sl, new) if side == "long" else min(sl, new)
                        trail_active = trail_active or sl2 != sl
                        sl = sl2
            else:
                cand = best_close - sgn * policy["trail_atr_mult"] * atr_now
                sl2 = max(sl, cand) if side == "long" else min(sl, cand)
                trail_active = trail_active or sl2 != sl
                sl = sl2
        # market-type exits decided on this close -> executed at next open
        if policy.get("max_hold_m5") and step + 1 >= policy["max_hold_m5"]:
            pending_market_exit = "time_exit"
        elif policy.get("vol_exit_atr_ratio") and ratio > policy["vol_exit_atr_ratio"] and unreal_r > 0:
            pending_market_exit = "volatility_exit"
        elif policy.get("structure_invalidation") and inv_level is not None and sgn * (ctx.m15["close"].iat[j] - inv_level) < 0 and j > m15_idx:
            pending_market_exit = "structure_invalidation_exit"
        m15_close_now = float(ctx.m15["close"].iat[j])
        struct_inv = bool(inv_level is not None and j > m15_idx and sgn * (m15_close_now - inv_level) < 0)
        if step % record_every == 0 or pending_market_exit:
            records.append({"risk_distance": sl_dist, "steps_held": step + 1, "structure_invalidated": struct_inv, "m15_close": m15_close_now,
                            "m15_trend_state": int(ctx.m15["trend_state"].iat[j]), "m15_last_swing_high": float(ctx.m15["last_swing_high"].iat[j]),
                            "m15_last_swing_low": float(ctx.m15["last_swing_low"].iat[j]), "m15_bar_index": j,"step": step, "time_ns": int(ctx.m5_t[k]) + 5 * 60 * 10**9, "price": float(c),
                            "entry_price": entry_price, "entry_time_ns": entry_ns, "side": side,
                            "initial_sl": sl0, "initial_tp": entry_price + sgn * tp_r * sl_dist if tp_r else None,
                            "current_sl": float(sl), "current_tp": None if tp is None else float(tp), "remaining_size": remaining,
                            "unrealized_r": round(unreal_r, 4), "mfe_r": round(mfe, 4), "mae_r": round(mae, 4),
                            "trailing_active": trail_active, "break_even_active": be_active, "partial_exits": list(events),
                            "costs_r_so_far": round(cost_paid / sl_dist, 4), "atr_now": atr_now, "atr_ratio_now": ratio})
    if remaining > 1e-9:                      # horizon reached (or truncated world): close at last known close
        kk = end - 1
        close(remaining, ctx.c[kk], "horizon_close" if upto_step is None else "truncated_world", kk)
        exit_reason = events[-1]["reason"]
    gross_r = realized_move / sl_dist
    net_r = (realized_move - cost_paid) / sl_dist
    return {"side": side, "exit_reason": exit_reason, "gross_r": gross_r, "net_r": net_r, "costs_r": cost_paid / sl_dist,
            "mfe_r": mfe, "mae_r": mae, "bars_held_m5": events[-1]["step"] + 1, "ambiguous": ambiguous, "events": events,
            "records": records, "be_active": be_active, "trail_active": trail_active,
            "mfe_capture": (gross_r / mfe) if mfe > 0 else None}


def rescore_net_r(trade: Dict[str, Any], profile, spread_points: float, sl_dist: float) -> float:
    """Net R of an already simulated trade under another cost profile (total round-trip cost is independent of exit fractions)."""
    cp = cost_parts(profile, spread_points)
    return trade["gross_r"] - (cp["spread"] + cp["commission"] + cp["slippage"]) / sl_dist


def simulate_entry_order(ctx: SimContext, side: str, order_type: str, level: Optional[float], decision_close_ns: int, m15_idx: int, policy: Dict[str, Any],
                         profile, sl_atr: float, atr: float, valid_m5: int = 24, record_every: int = 12) -> Dict[str, Any]:
    """Market / Limit / Stop entry from the decision close.  Conservative conventions:
    * market: fills at the next M5 open; limit/stop: working order for ``valid_m5`` M5 bars, gap fills at the (better/worse) open.
    * if the fill bar itself reaches the stop, the trade is stopped out in that bar (worst case, flagged).
    * management starts at the bar after the fill."""
    sgn = 1 if side == "long" else -1
    start = int(np.searchsorted(ctx.m5_t, decision_close_ns))
    if start >= len(ctx.m5_t):
        return {"filled": False, "reason": "no_data"}
    sl_dist = sl_atr * atr
    if order_type == "market":
        entry = float(ctx.o[start])
        fill_k = start - 1                                  # management starts at `start`
    else:
        fill_k, entry = None, None
        for k in range(start, min(start + valid_m5, len(ctx.m5_t))):
            o, h, l = ctx.o[k], ctx.h[k], ctx.l[k]
            if order_type == "limit":
                hit = (l <= level) if side == "long" else (h >= level)
                if hit:
                    entry = float(min(o, level) if side == "long" else max(o, level))
            else:  # stop
                hit = (h >= level) if side == "long" else (l <= level)
                if hit:
                    entry = float(max(o, level) if side == "long" else min(o, level))
            if hit:
                fill_k = k
                break
        if fill_k is None:
            return {"filled": False, "reason": "not_filled_within_validity"}
        sl = entry - sgn * sl_dist
        adverse = ctx.l[fill_k] if side == "long" else ctx.h[fill_k]
        if sgn * (adverse - sl) <= 0:
            cp = cost_parts(profile, float(ctx.m15["spread"].iat[m15_idx]))
            gross = -1.0
            net = gross - (cp["spread"] + cp["commission"] + cp["slippage"]) / sl_dist
            return {"filled": True, "entry_price": entry, "exit_reason": "stopped_in_fill_bar", "gross_r": gross, "net_r": net, "mfe_r": 0.0, "mae_r": 1.0,
                    "bars_held_m5": 1, "ambiguous": True, "records": [], "events": [], "costs_r": (net - gross) * -1, "side": side}
    nxt = fill_k + 1
    if nxt >= len(ctx.m5_t):
        return {"filled": False, "reason": "no_data"}
    t = simulate_trade(ctx, side, m15_idx, policy, profile, entry if order_type != "market" else entry, sl_dist, int(ctx.m5_t[nxt]), record_every)
    t["filled"], t["entry_price"] = True, entry
    return t
