"""Deterministic Smart-Money-Concepts structures computed from closed candles (no model, no future data).

The model never "finds" an order block or a liquidity pool by eye: this module computes them and the model only
interprets the printed result (same philosophy as the rest of forex_analysis.py).  Definitions are deliberately
simple and stated here so the dataset, the tests and the runtime agree.

* swing high/low   fractal pivot (``left``/``right`` bars), confirmed only ``right`` bars later (no look-ahead)
* BOS / CHoCH      a CLOSE beyond the last confirmed swing; BOS continues the current structural trend, CHoCH is the
                   first break against it
* liquidity pool   EQH / EQL: >= 2 swing highs (lows) within 0.10 x ATR; single unswept swing extremes also count
* sweep            a bar whose wick takes a swing extreme but CLOSES back inside (within the last ``sweep_bars`` bars)
* FVG              3-candle imbalance (bull: low[i] > high[i-2]; bear: high[i] < low[i-2]), size >= 0.25 x ATR,
                   still not fully filled
* order block      last opposite-colour candle at/before the origin of an impulse that produced a BOS/CHoCH;
                   needs displacement (a body >= 1.0 x ATR or a leg >= 2 x ATR); zone = that candle's high-low
* premium/discount position of the close inside the last confirmed swing range (50 % = equilibrium)

These are discretionary-trading constructs.  Detecting them does NOT imply an edge; the dataset says so explicitly.
"""
from __future__ import annotations

try:
    from . import forex_analysis as fa
except ImportError:                       # pragma: no cover - flat import
    import forex_analysis as fa

EQ_TOL_ATR = 0.10
FVG_MIN_ATR = 0.25
DISPLACEMENT_BODY_ATR = 1.0
DISPLACEMENT_LEG_ATR = 2.0


def _confirmed_swings(h, l, left, right):
    sh, sl = fa.swing_points(h, l, left, right)
    return [(i, p, i + right) for i, p in sh], [(i, p, i + right) for i, p in sl]


def structure_events(o, h, l, c, left=3, right=3):
    """All BOS/CHoCH events (oldest first) and the structural trend after the last one."""
    shs, sls = _confirmed_swings(h, l, left, right)
    events, trend = [], None
    broken_h, broken_l = set(), set()
    for i in range(len(c)):
        known_h = [s for s in shs if s[2] <= i and s[0] not in broken_h]
        known_l = [s for s in sls if s[2] <= i and s[0] not in broken_l]
        if known_h and c[i] > known_h[-1][1]:
            sw = known_h[-1]
            events.append({"idx": i, "type": "BOS" if trend in (None, "up") else "CHoCH", "dir": "up",
                           "level": sw[1], "swing_idx": sw[0]})
            trend = "up"
            broken_h.update(s[0] for s in known_h if s[1] <= c[i])
        elif known_l and c[i] < known_l[-1][1]:
            sw = known_l[-1]
            events.append({"idx": i, "type": "BOS" if trend in (None, "down") else "CHoCH", "dir": "down",
                           "level": sw[1], "swing_idx": sw[0]})
            trend = "down"
            broken_l.update(s[0] for s in known_l if s[1] >= c[i])
    return events, trend


def liquidity(h, l, c, atr_value, left=3, right=3, lookback=120, sweep_bars=20):
    n = len(c)
    start = max(0, n - lookback)
    shs, sls = _confirmed_swings(h, l, left, right)
    shs = [s for s in shs if s[0] >= start]
    sls = [s for s in sls if s[0] >= start]
    tol = EQ_TOL_ATR * atr_value if atr_value else 0.0
    price = c[-1]

    def unswept_high(s):
        return all(h[j] <= s[1] + tol for j in range(s[0] + 1, n))

    def unswept_low(s):
        return all(l[j] >= s[1] - tol for j in range(s[0] + 1, n))

    def pools(points, unswept, pick):
        live = sorted((s for s in points if unswept(s)), key=lambda s: s[1])
        groups = []
        for s in live:
            if groups and s[1] - groups[-1][-1][1] <= tol:
                groups[-1].append(s)
            else:
                groups.append([s])
        return [{"price": pick(g)[1], "touches": len(g), "kind": "EQ" if len(g) >= 2 else "swing",
                 "last_idx": max(x[0] for x in g)} for g in groups]

    highs = [p for p in pools(shs, unswept_high, lambda g: g[-1]) if p["price"] > price]
    lows = [p for p in pools(sls, unswept_low, lambda g: g[0]) if p["price"] < price]
    highs.sort(key=lambda p: p["price"])
    lows.sort(key=lambda p: -p["price"])
    sweeps = []
    for j in range(max(0, n - sweep_bars), n):
        for s in shs:
            if s[0] + right < j and h[j] > s[1] + tol and c[j] < s[1] and all(h[k] <= s[1] + tol for k in range(s[0] + 1, j)):
                sweeps.append({"side": "high", "level": s[1], "idx": j})
        for s in sls:
            if s[0] + right < j and l[j] < s[1] - tol and c[j] > s[1] and all(l[k] >= s[1] - tol for k in range(s[0] + 1, j)):
                sweeps.append({"side": "low", "level": s[1], "idx": j})
    sweeps.sort(key=lambda s: (-s["idx"], -s["level"] if s["side"] == "high" else s["level"]))   # most recent, most extreme
    return {"buy_side": highs[:2], "sell_side": lows[:2], "last_sweep": sweeps[0] if sweeps else None}


def fvgs(o, h, l, c, atr_value, lookback=150, limit=2):
    n = len(c)
    out = []
    min_size = FVG_MIN_ATR * atr_value if atr_value else 0.0
    for i in range(max(2, n - lookback), n):
        if l[i] > h[i - 2] and l[i] - h[i - 2] >= min_size:
            top, bottom, direction = l[i], h[i - 2], "up"
        elif h[i] < l[i - 2] and l[i - 2] - h[i] >= min_size:
            top, bottom, direction = l[i - 2], h[i], "down"
        else:
            continue
        later_low = min(l[i + 1:], default=top)
        later_high = max(h[i + 1:], default=bottom)
        if direction == "up":
            if later_low <= bottom:
                continue                                  # fully filled
            filled = max(0.0, top - later_low) / (top - bottom)
        else:
            if later_high >= top:
                continue
            filled = max(0.0, later_high - bottom) / (top - bottom)
        out.append({"dir": direction, "top": top, "bottom": bottom, "idx": i, "filled": min(1.0, filled)})
    price = c[-1]
    out.sort(key=lambda f: min(abs(price - f["top"]), abs(price - f["bottom"])))
    return out[:limit]


def order_blocks(o, h, l, c, atr_value, events, lookback=200, limit=2):
    n = len(c)
    found = []
    for ev in reversed(events):
        if len(found) >= 4 or ev["idx"] < n - lookback:
            break
        e, s = ev["idx"], ev["swing_idx"]
        if ev["dir"] == "up":
            origin = min(range(max(0, s - 3), e + 1), key=lambda j: l[j])      # lowest bar of the leg
            body = max((abs(c[j] - o[j]) for j in range(origin, e + 1)), default=0.0)
            leg = max(h[origin:e + 1]) - l[origin]
            if not atr_value or (body < DISPLACEMENT_BODY_ATR * atr_value and leg < DISPLACEMENT_LEG_ATR * atr_value):
                continue
            k = next((j for j in range(origin, max(-1, origin - 6), -1) if c[j] < o[j]), None)
            if k is None:
                continue
            zone_lo, zone_hi, direction = l[k], h[k], "up"
            broken = any(c[j] < zone_lo for j in range(e + 1, n))
            touched = any(l[j] <= zone_hi for j in range(e + 1, n))
        else:
            origin = max(range(max(0, s - 3), e + 1), key=lambda j: h[j])
            body = max((abs(c[j] - o[j]) for j in range(origin, e + 1)), default=0.0)
            leg = h[origin] - min(l[origin:e + 1])
            if not atr_value or (body < DISPLACEMENT_BODY_ATR * atr_value and leg < DISPLACEMENT_LEG_ATR * atr_value):
                continue
            k = next((j for j in range(origin, max(-1, origin - 6), -1) if c[j] > o[j]), None)
            if k is None:
                continue
            zone_lo, zone_hi, direction = l[k], h[k], "down"
            broken = any(c[j] > zone_hi for j in range(e + 1, n))
            touched = any(h[j] >= zone_lo for j in range(e + 1, n))
        if broken:
            continue
        found.append({"dir": direction, "low": zone_lo, "high": zone_hi, "idx": k, "event_idx": e,
                      "event": ev["type"], "touched": touched})
    price = c[-1]
    found.sort(key=lambda z: min(abs(price - z["low"]), abs(price - z["high"])))
    seen, out = set(), []
    for z in found:
        key = (z["dir"], round(z["low"], 6), round(z["high"], 6))
        if key not in seen:
            seen.add(key)
            out.append(z)
    return out[:limit]


def premium_discount(h, l, c, left=3, right=3):
    shs, sls = _confirmed_swings(h, l, left, right)
    if not shs or not sls:
        return None
    hi, lo = shs[-1][1], sls[-1][1]
    if hi <= lo:
        return None
    price = c[-1]
    pos = (price - lo) / (hi - lo)
    zone = "خارج از رنج" if pos < 0 or pos > 1 else ("Premium" if pos > 0.5 else "Discount" if pos < 0.5 else "تعادل")
    return {"high": hi, "low": lo, "equilibrium": (hi + lo) / 2.0, "position": pos, "zone": zone}


def smc_report(o, h, l, c, atr_value, left=3, right=3):
    """Everything above for one timeframe; arguments are equally long lists of CLOSED candles, oldest first."""
    if len(c) < 30 or not atr_value:
        return None
    events, trend = structure_events(o, h, l, c, left, right)
    last = events[-1] if events else None
    return {
        "bars": len(c), "trend": trend,
        "last_event": None if not last else {**last, "bars_ago": len(c) - 1 - last["idx"]},
        "liquidity": liquidity(h, l, c, atr_value, left, right),
        "fvg": fvgs(o, h, l, c, atr_value),
        "order_blocks": order_blocks(o, h, l, c, atr_value, events),
        "range": premium_discount(h, l, c, left, right),
    }


# ------------------------------------------------------------------------------------------ text for the model
def _p(value, digits):
    return f"{value:.{digits}f}"


def smc_lines(report, digits, bars_total=None):
    """Compact Persian text lines (Latin terms) for one timeframe; empty list when there is nothing."""
    if not report:
        return []
    n = report["bars"]
    lines = []
    ev = report["last_event"]
    trend_fa = {"up": "صعودی", "down": "نزولی", None: "نامشخص"}[report["trend"]]
    if ev:
        dir_fa = "صعودی" if ev["dir"] == "up" else "نزولی"
        lines.append(f"  SMC ساختار: آخرین رویداد {ev['type']} {dir_fa} (close بیرون از {_p(ev['level'], digits)}، "
                     f"{ev['bars_ago']} کندل پیش) | روند ساختاری: {trend_fa}")
    else:
        lines.append("  SMC ساختار: رویداد BOS/CHoCH در داده نیست | روند ساختاری: نامشخص")
    liq = report["liquidity"]
    parts = []
    if liq["buy_side"]:
        b = liq["buy_side"][0]
        parts.append(f"بالا (BSL) {_p(b['price'], digits)} {'EQH' if b['kind'] == 'EQ' else 'swing'}×{b['touches']}")
    if liq["sell_side"]:
        s = liq["sell_side"][0]
        parts.append(f"پایین (SSL) {_p(s['price'], digits)} {'EQL' if s['kind'] == 'EQ' else 'swing'}×{s['touches']}")
    sw = liq["last_sweep"]
    if sw:
        side = "سقف" if sw["side"] == "high" else "کف"
        parts.append(f"آخرین sweep: {side} {_p(sw['level'], digits)} ({n - 1 - sw['idx']} کندل پیش، close برگشته داخل)")
    lines.append("  نقدینگی ناشکسته: " + (" | ".join(parts) if parts else "مورد مشخصی در پنجرهٔ ۱۲۰ کندل نیست"))
    fv = report["fvg"]
    lines.append("  FVG باز: " + ("؛ ".join(
        f"{'صعودی' if f['dir'] == 'up' else 'نزولی'} {_p(f['bottom'], digits)}–{_p(f['top'], digits)} "
        f"(پر شده {f['filled'] * 100:.0f}%)" for f in fv) if fv else "مورد مشخصی نیست"))
    ob = report["order_blocks"]
    lines.append("  OB معتبر: " + ("؛ ".join(
        f"{'صعودی' if z['dir'] == 'up' else 'نزولی'} {_p(z['low'], digits)}–{_p(z['high'], digits)} "
        f"({'لمس‌شده' if z['touched'] else 'دست‌نخورده'}، منشأ {z['event']})" for z in ob) if ob else "مورد مشخصی نیست"))
    rg = report["range"]
    if rg:
        lines.append(f"  رنج ساختاری: کف {_p(rg['low'], digits)} – سقف {_p(rg['high'], digits)} | تعادل {_p(rg['equilibrium'], digits)} "
                     f"| قیمت در {rg['zone']} ({rg['position'] * 100:.0f}%)")
    return lines
