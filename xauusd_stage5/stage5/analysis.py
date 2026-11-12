"""Deterministic, CAUSAL market-analysis, setup-qualification and trade-planning rules.

Pure functions of a single snapshot (features known at the decision bar close).  They define the analytical
PROCEDURE the model is taught (decisions are NOT derived from hindsight labels).  Their historical outcomes are
reported separately (evaluate.py) - the procedure is not claimed to be profitable.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

SC = {1: "bullish", -1: "bearish", 0: "undefined"}
STRUCT = {0: "none", 1: "BOS bullish", 2: "BOS bearish", 3: "CHOCH bullish", 4: "CHOCH bearish"}
SPREAD_WIDE_ATR, VOL_EXTREME = 0.20, 1.8
PLAN = dict(sl_atr=1.5, rr1=1.0, rr2=2.0, limit_pullback_atr=0.5, stop_offset_atr=0.25, limit_valid_m15_bars=8)


def _f(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _b(s, k, default="unavailable"):
    v = _f(s.get(k))
    return default if v is None else SC.get(int(v), default)


def analyze(s: Dict[str, Any]) -> Dict[str, Any]:
    g = lambda k: _f(s.get(k))  # noqa: E731
    atr, close = g("atr_14"), g("close")
    bias = {"M15": _b(s, "trend_state"), "M30": _b(s, "m30_trend_state"), "H1": _b(s, "h1_trend_state"), "H4": _b(s, "h4_trend_state"), "D1": _b(s, "d1_trend_state")}
    ema = {"M15": g("ema_stack"), "H1": g("h1_ema_stack"), "H4": g("h4_ema_stack")}
    h1, h4, d1 = (g("h1_trend_state") or 0), (g("h4_trend_state") or 0), (g("d1_trend_state") or 0)
    m15 = g("trend_state") or 0
    votes = [m15, h1, h4, d1]
    n_bull, n_bear = sum(v > 0 for v in votes), sum(v < 0 for v in votes)
    if h4 != 0 and h4 == h1 and d1 != -h4:
        conf = "bullish_confluence" if h4 > 0 else "bearish_confluence"
    elif h4 != 0 and h1 != 0 and h4 != h1:
        conf = "htf_conflict"
    else:
        conf = "mixed"
    ratio = g("atr_ratio")
    vol = "unavailable" if ratio is None else ("expansion" if ratio > 1.2 else ("contraction" if ratio < 0.85 else "normal"))
    d50 = g("dist_ema_50_atr")
    stack = g("ema_stack") or 0
    bars_ev = g("bars_since_struct_event")
    ev = int(g("struct_last_event") or 0)
    choch_recent = ev in (3, 4) and bars_ev is not None and bars_ev <= 12
    sweeps = {"buy_side_swept_now": bool(g("bsl_sweep")), "sell_side_swept_now": bool(g("ssl_sweep"))}
    if choch_recent or sweeps["buy_side_swept_now"] or sweeps["sell_side_swept_now"]:
        regime = "reversal_candidate"
    elif stack != 0 and stack == m15 and d50 is not None and abs(d50) > 0.5:
        regime = "trend_up" if m15 > 0 else "trend_down"
    elif vol == "contraction":
        regime = "consolidation"
    else:
        regime = "range"
    session = ("london_ny_overlap" if g("session_london_ny_overlap") else "london" if g("session_london") else "new_york" if g("session_ny") else "asia" if g("session_asia") else "off_session")
    spread_atr = None
    if g("spread_known") and atr:
        spread_atr = (g("spread") or 0) * 0.01 / atr
    pd_zone = {1: "premium", -1: "discount", 0: "equilibrium/undefined"}.get(int(g("pd_zone") or 0))
    zones = {"bull_fvg_in_zone": bool(g("fvg_bull_in_zone")), "bear_fvg_in_zone": bool(g("fvg_bear_in_zone")),
             "bull_ob_in_zone": bool(g("ob_bull_in_zone")), "bear_ob_in_zone": bool(g("ob_bear_in_zone")),
             "bull_fvg_dist_atr": g("fvg_bull_dist_atr"), "bear_fvg_dist_atr": g("fvg_bear_dist_atr"),
             "bull_ob_dist_atr": g("ob_bull_dist_atr"), "bear_ob_dist_atr": g("ob_bear_dist_atr")}
    # ---------------- evidence
    bull, bear = [], []
    if conf == "bullish_confluence":
        bull.append("H4 and H1 structure bullish (D1 not opposing)")
    if conf == "bearish_confluence":
        bear.append("H4 and H1 structure bearish (D1 not opposing)")
    if m15 > 0:
        bull.append("M15 structure bullish")
    if m15 < 0:
        bear.append("M15 structure bearish")
    if sweeps["sell_side_swept_now"]:
        bull.append("sell-side liquidity swept and closed back above")
    if sweeps["buy_side_swept_now"]:
        bear.append("buy-side liquidity swept and closed back below")
    if pd_zone == "discount":
        bull.append("price in discount of the last swing range")
    if pd_zone == "premium":
        bear.append("price in premium of the last swing range")
    if zones["bull_fvg_in_zone"] or zones["bull_ob_in_zone"]:
        bull.append("price inside a bullish FVG/Order Block zone")
    if zones["bear_fvg_in_zone"] or zones["bear_ob_in_zone"]:
        bear.append("price inside a bearish FVG/Order Block zone")
    if ev in (1, 3) and bars_ev is not None and bars_ev <= 12:
        bull.append(f"recent {STRUCT[ev]} ({int(bars_ev)} bars ago)")
    if ev in (2, 4) and bars_ev is not None and bars_ev <= 12:
        bear.append(f"recent {STRUCT[ev]} ({int(bars_ev)} bars ago)")
    if (g("disp_dir") or 0) > 0:
        bull.append("bullish displacement candle")
    if (g("disp_dir") or 0) < 0:
        bear.append("bearish displacement candle")
    # ---------------- no-trade flags
    nt: List[str] = []
    if atr is None or close is None:
        nt.append("insufficient_data: ATR/close unavailable")
    if spread_atr is not None and spread_atr > SPREAD_WIDE_ATR:
        nt.append(f"poor_execution: spread/ATR {spread_atr:.2f} > {SPREAD_WIDE_ATR}")
    if ratio is not None and ratio > VOL_EXTREME:
        nt.append(f"abnormal_volatility: atr_ratio {ratio:.2f} > {VOL_EXTREME}")
    if conf == "htf_conflict":
        nt.append("htf_conflict: H4 and H1 structure disagree")
    if session == "off_session":
        nt.append("off_session: no defined liquid session")
    # ---------------- setup qualification (direction from causal evidence only)
    side, quality = None, 0
    if not nt:
        if conf == "bullish_confluence" and len(bull) >= 3 and len(bull) > len(bear):
            side, quality = "long", len(bull)
        elif conf == "bearish_confluence" and len(bear) >= 3 and len(bear) > len(bull):
            side, quality = "short", len(bear)
        if side == "long" and (g("dist_last_swing_high_atr") is not None and 0 <= g("dist_last_swing_high_atr") < 0.3):
            nt.append("opposing_level_too_close: swing high within 0.3 ATR"); side = None
        if side == "short" and (g("dist_last_swing_low_atr") is not None and -0.3 < g("dist_last_swing_low_atr") <= 0):
            nt.append("opposing_level_too_close: swing low within 0.3 ATR"); side = None
    if side is None and not nt:
        nt.append("no_confluent_setup: evidence below threshold")
    contra = bear if side == "long" else bull if side == "short" else []
    return {"bias": bias, "htf_confluence": conf, "votes": {"bullish": n_bull, "bearish": n_bear}, "regime": regime, "volatility": vol,
            "atr_ratio": ratio, "session": session, "spread_atr": None if spread_atr is None else round(spread_atr, 4),
            "premium_discount": pd_zone, "last_structure_event": STRUCT[ev], "bars_since_structure_event": bars_ev, "liquidity_sweeps": sweeps,
            "zones": zones, "evidence": {"bullish": bull, "bearish": bear}, "no_trade_flags": nt,
            "setup": {"side": side, "quality": quality, "supporting": bull if side == "long" else bear if side == "short" else [], "contradicting": contra}}


def plan_trade(s: Dict[str, Any], side: str, an: Dict[str, Any]) -> Dict[str, Any]:
    """Numeric plan from the snapshot ONLY.  Reference price = decision close; real fills are next-bar prices."""
    atr, close = _f(s["atr_14"]), _f(s["close"])
    sgn = 1 if side == "long" else -1
    z = an["zones"]
    d_fvg, d_ob = (z["bull_fvg_dist_atr"], z["bull_ob_dist_atr"]) if side == "long" else (z["bear_fvg_dist_atr"], z["bear_ob_dist_atr"])
    in_zone = (z["bull_fvg_in_zone"] or z["bull_ob_in_zone"]) if side == "long" else (z["bear_fvg_in_zone"] or z["bear_ob_in_zone"])
    near = [d for d in (d_fvg, d_ob) if d is not None and 0 < d <= 1.0]
    breakout = bool(_f(s.get("brk_high_20")) if side == "long" else _f(s.get("brk_low_20"))) and (_f(s.get("disp_dir")) or 0) == sgn
    if in_zone or an["liquidity_sweeps"]["sell_side_swept_now" if side == "long" else "buy_side_swept_now"]:
        otype, entry, why = "market", close, "price already reacting at a zone/liquidity sweep: enter at the next open"
    elif breakout:
        otype, entry, why = "stop", close + sgn * PLAN["stop_offset_atr"] * atr, "displacement breakout: enter only if price extends further"
    elif near:
        otype, entry, why = "limit", close - sgn * min(near) * atr, "pullback into the nearest facing zone edge"
    else:
        otype, entry, why = "limit", close - sgn * PLAN["limit_pullback_atr"] * atr, "no zone nearby: 0.5 ATR pullback limit"
    sl_dist = PLAN["sl_atr"] * atr
    sl = entry - sgn * sl_dist
    tp1, tp2 = entry + sgn * PLAN["rr1"] * sl_dist, entry + sgn * PLAN["rr2"] * sl_dist
    struct_lvl = _f(s.get("last_swing_low")) if side == "long" else _f(s.get("last_swing_high"))
    inval = struct_lvl
    r = lambda x: None if x is None else round(float(x), 2)  # noqa: E731
    plan = {"side": side, "order_type": otype, "order_rationale": why, "reference_price": r(close), "entry_price": r(entry),
            "stop_loss": r(sl), "stop_distance_atr": PLAN["sl_atr"], "stop_distance_price": round(sl_dist, 2),
            "take_profit_1": r(tp1), "take_profit_2": r(tp2), "risk_reward_tp1": PLAN["rr1"], "risk_reward_tp2": PLAN["rr2"],
            "stop_method": "ATR-based (1.5 x ATR14); structure level shown for reference", "structure_reference_level": r(struct_lvl),
            "structure_invalidation_close": r(inval),
            "confirmation": [f"M15 bar closes {'above' if side == 'long' else 'below'} the decision close with no opposing displacement",
                             "spread/ATR stays below 0.20"],
            "invalidation": [f"M15 close {'below' if side == 'long' else 'above'} {r(inval)} (last confirmed swing) before entry",
                             "H4/H1 structure turns against the trade"],
            "cancel_conditions": [f"unfilled after {PLAN['limit_valid_m15_bars']} M15 bars" if otype != "market" else "next bar gaps beyond 0.5 ATR from reference",
                                  "ATR ratio rises above 1.8", "spread/ATR above 0.35"]}
    plan["staged_entry"] = ({"tranches": [{"fraction": 0.5, "type": "market"}, {"fraction": 0.5, "type": "limit", "price": r(close - sgn * 0.5 * atr)}],
                             "note": "only when setup quality >= 4; total risk unchanged (each tranche carries half the risk budget)"}
                            if an["setup"]["quality"] >= 4 and otype == "market" else None)
    return plan


def management_playbook(state: Dict[str, Any]) -> Dict[str, Any]:
    """State-only action recommendation for an OPEN trade (no future information).
    Optional keys m15_last_swing_high / m15_last_swing_low enable structure trailing.  Stops are only ever tightened."""
    r, sgn = state["unrealized_r"], 1 if state["side"] == "long" else -1
    n_part = len(state["partial_exits"])
    entry, rd, cur = state["entry_price"], state["risk_distance"], state["current_sl"]
    tighten = lambda new: (max(new, cur) if sgn > 0 else min(new, cur))  # noqa: E731
    be_new = round(tighten(entry + sgn * 0.1 * rd), 2)
    be_improves = be_new != round(cur, 2)
    protected = bool(state["break_even_active"] or state.get("trailing_active"))

    def trail():
        atr_stop = tighten(state["price"] - sgn * 1.5 * state["atr_now"])
        lvl = state.get("m15_last_swing_low") if sgn > 0 else state.get("m15_last_swing_high")
        st_stop = None
        if lvl is not None and lvl == lvl:
            c = lvl - sgn * 0.1 * state["atr_now"]
            if sgn * (c - state["price"]) < 0 and sgn * (c - entry) > 0:
                st_stop = tighten(c)
        best, kind = atr_stop, "TRAIL_STOP_ATR"
        if st_stop is not None and sgn * (st_stop - cur) > 0:          # a valid confirmed swing that tightens the stop is preferred over the ATR trail
            best, kind = st_stop, "TRAIL_STOP_STRUCTURE"
        if round(best, 2) == round(cur, 2):
            return None
        why = (f"+{r:.2f}R with the stop protecting capital: trail 1.5 ATR behind price" if kind == "TRAIL_STOP_ATR"
               else f"+{r:.2f}R with the stop protecting capital: trail behind the last confirmed M15 swing (structure-based trail)")
        return {"action": kind, "reason": [why], "new_stop": round(best, 2)}

    if state.get("structure_invalidated"):
        return {"action": "CLOSE_NOW", "reason": ["M15 closed beyond the structure invalidation level"], "new_stop": None}
    if state["atr_ratio_now"] is not None and state["atr_ratio_now"] > 1.6 and r > 0:
        return {"action": "CLOSE_NOW", "reason": [f"abnormal volatility (ATR ratio {state['atr_ratio_now']:.2f}) while in profit: secure the open gain"], "new_stop": None}
    if state.get("trailing_active") and r >= 1.5:
        t = trail()
        if t:
            return t
    if state["remaining_size"] >= 0.999 and r >= 1.0 and n_part == 0 and (not protected or r < 1.5):
        if not be_improves:
            return {"action": "TAKE_PARTIAL_50", "reason": [f"+{r:.2f}R reached; the stop already protects capital: realize half"], "new_stop": None}
        return {"action": "TAKE_PARTIAL_AND_MOVE_STOP_TO_BREAK_EVEN", "reason": [f"+{r:.2f}R reached: realize half, move stop to break-even + 0.1R"], "new_stop": be_new}
    if n_part == 1 and r >= 1.5 and state["remaining_size"] >= 0.45:
        return {"action": "TAKE_PARTIAL_33", "reason": [f"+{r:.2f}R: approaching the second target stage, realize another third"], "new_stop": None}
    if n_part >= 1 and not protected and be_improves:
        return {"action": "MOVE_STOP_TO_BREAK_EVEN", "reason": [f"partial profit already taken (+{r:.2f}R): protect the remainder at break-even + 0.1R"], "new_stop": be_new}
    if r >= 1.5 and protected:
        t = trail()
        if t:
            return t
    if state["steps_held"] >= 48 and r < 0.5:
        return {"action": "CLOSE_NOW", "reason": ["time stop: 48 M5 bars without reaching +0.5R"], "new_stop": None}
    return {"action": "HOLD", "reason": ["no management rule triggered; stop and target unchanged"], "new_stop": None}
