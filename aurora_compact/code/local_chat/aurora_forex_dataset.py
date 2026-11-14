"""Builds the AURORA forex education + multi-style analysis SFT dataset.  No training happens here.

Inputs are REAL XAUUSD candles (M15/H1/H4/D1).  For every sample time ``t`` only candles that were fully closed at ``t``
are used (open_epoch + tf_seconds <= t), the context text is produced by the app's own ``forex_analysis.format_symbol`` (SMC
block on), and the answer is a deterministic function of that text: every price comes verbatim from the context, derived
numbers are listed in ``meta.derived``.  The dataset teaches structure, vocabulary, risk and honesty; it claims no edge.

    python -m local_chat.aurora_forex_dataset --out training_data/aurora_forex
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from typing import Dict, List, Optional

try:
    from . import forex_analysis as fa
    from .aurora_compact import compact_system
    from .aurora_forex_kb import CONCEPTS, COMPARES, STYLES
except ImportError:                       # pragma: no cover
    import forex_analysis as fa
    from aurora_compact import compact_system
    from aurora_forex_kb import CONCEPTS, COMPARES, STYLES

DISCLAIMER = "این یک تحلیل آموزشی و توصیفی است، نه توصیهٔ مالی و نه تضمین سود؛ ریسک و حجم را خودت و با ابزار محاسبهٔ ریسک تعیین کن."
MODELS = ["qwen3.5:4b", "qwen3:4b", None]
TFS = ("D1", "H4", "H1", "M15")
SRC = {"M15": "M15", "H1": "H1", "H4": "H4", "D1": "Daily"}
SPLITS = {"train": ("2021-09-01", "2024-06-29"), "val": ("2024-07-02", "2025-06-29"), "test": ("2025-07-02", "2099-01-01")}

# ============================================================================================ context from real candles
EPOCH0 = None


def load_candles(src_dir, symbol="XAUUSD"):
    import pandas as pd
    out = {}
    for k, f in SRC.items():
        d = pd.read_csv(os.path.join(src_dir, f"{symbol}_{f}_normalized.csv"), parse_dates=["time"])
        d["epoch"] = ((d["time"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).astype("int64")
        out[k] = d
    return out


def make_context(candles, t, smc=True, symbol="XAUUSD", digits=2, spread=0.2):
    """Context text for decision time ``t`` (pandas Timestamp, broker server time). Returns (text, audit_info)."""
    import pandas as pd
    te = int((t - pd.Timestamp("1970-01-01")).total_seconds())
    rates, last_open = {}, {}
    for k, d in candles.items():
        m = d[d["epoch"] + fa.TIMEFRAME_SECONDS[k] <= te].tail(300)
        if len(m) < 210 and k == "D1" or len(m) < 120:
            return None, None
        rates[k] = {"time": m["epoch"].tolist(), "open": m["open"].tolist(), "high": m["high"].tolist(),
                    "low": m["low"].tolist(), "close": m["close"].tolist()}
        last_open[k] = int(m["epoch"].iloc[-1])
    cl = rates["M15"]["close"][-1]
    raw = {"symbol": symbol, "broker_symbol": symbol, "info": {"digits": digits, "point": 10 ** -digits},
           "tick": {"bid": cl, "ask": round(cl + spread, digits), "time": te}, "rates": rates}
    old = fa.ENABLE_SMC
    fa.ENABLE_SMC = smc
    try:
        snap = fa.build_snapshot(raw, t.to_pydatetime())
        text = fa.format_symbol(snap, 1)
    finally:
        fa.ENABLE_SMC = old
    if "بدون داده‌ی کافی" in text or len(snap["timeframes"]) < 4:
        return None, None
    text = text.replace(" UTC", " (وقت سرور)")
    text = re.sub(r"\s*\|\s*سشن‌ها: [^\n]*", "", text)
    return text, {"t": str(t), "te": te, "last_open": last_open}


# ============================================================================================ parsing the context text
N = r"[-+]?\d+(?:\.\d+)?"


def _levels(s):
    return [(p, int(x)) for p, x in re.findall(rf"({N})\(x(\d+)\)", s)]


def parse_ctx(text):
    P = {"tf": {}, "close": None, "bid": None, "ask": None, "pivots": None, "align": None}
    m = re.search(rf"bid ({N}) ask ({N}) \| اسپرد ({N})", text)
    if m:
        P["bid"], P["ask"], P["spread"] = m.group(1), m.group(2), m.group(3)
    cur = None
    for line in text.splitlines():
        h = re.match(rf"\[(\w+) [^\]]*\] روند: (.+?) \(([+-]\d)/(\d)\) \| close ({N}) \| EMA20 ({N}) EMA50 ({N}) EMA200 ({N})", line)
        if h:
            cur = {"name": h.group(1), "trend": h.group(2), "score": h.group(3) + "/" + h.group(4), "close": h.group(5),
                   "ema20": h.group(6), "ema50": h.group(7), "ema200": h.group(8), "sup": [], "res": [], "obs": [], "fvgs": [],
                   "event": None, "struct": None, "bsl": None, "ssl": None, "sweep": None, "range": None, "has_smc": False}
            P["tf"][h.group(1)] = cur
            continue
        if cur is None:
            continue
        m = re.search(rf"RSI ({N}) \| MACD ({N}) hist ({N}) \| ATR ({N})", line)
        if m:
            cur["rsi"], cur["macd"], cur["hist"], cur["atr"] = m.groups()
            continue
        m = re.match(r"\s+حمایت: (.*?) \| مقاومت: (.*)$", line)
        if m:
            cur["sup"], cur["res"] = _levels(m.group(1)), _levels(m.group(2))
            continue
        m = re.search(rf"SMC ساختار: آخرین رویداد (BOS|CHoCH) (صعودی|نزولی) \(close بیرون از ({N})، (\d+) کندل پیش\) \| روند ساختاری: (\S+)", line)
        if m:
            cur["event"] = {"type": m.group(1), "dir": m.group(2), "level": m.group(3), "ago": m.group(4)}
            cur["struct"] = m.group(5)
            cur["has_smc"] = True
            continue
        m = re.search(r"SMC ساختار: رویداد BOS/CHoCH در داده نیست \| روند ساختاری: (\S+)", line)
        if m:
            cur["struct"], cur["has_smc"] = m.group(1), True
            continue
        if "نقدینگی ناشکسته:" in line:
            m = re.search(rf"بالا \(BSL\) ({N}) (EQH|swing)×(\d+)", line)
            if m:
                cur["bsl"] = {"p": m.group(1), "kind": m.group(2), "n": m.group(3)}
            m = re.search(rf"پایین \(SSL\) ({N}) (EQL|swing)×(\d+)", line)
            if m:
                cur["ssl"] = {"p": m.group(1), "kind": m.group(2), "n": m.group(3)}
            m = re.search(rf"آخرین sweep: (سقف|کف) ({N}) \((\d+) کندل پیش", line)
            if m:
                cur["sweep"] = {"side": m.group(1), "p": m.group(2), "ago": m.group(3)}
            continue
        if line.strip().startswith("FVG باز:"):
            for d, lo, hi, f in re.findall(rf"(صعودی|نزولی) ({N})–({N}) \(پر شده (\d+)%\)", line):
                cur["fvgs"].append({"dir": d, "lo": lo, "hi": hi, "filled": int(f)})
            continue
        if line.strip().startswith("OB معتبر:"):
            for d, lo, hi, st, ev in re.findall(rf"(صعودی|نزولی) ({N})–({N}) \((لمس‌شده|دست‌نخورده)، منشأ (\w+)\)", line):
                cur["obs"].append({"dir": d, "lo": lo, "hi": hi, "state": st, "event": ev})
            continue
        m = re.search(rf"رنج ساختاری: کف ({N}) – سقف ({N}) \| تعادل ({N}) \| قیمت در (\S+) \((\d+)%\)", line)
        if m:
            cur["range"] = {"lo": m.group(1), "hi": m.group(2), "eq": m.group(3), "zone": m.group(4), "pct": m.group(5)}
            continue
        m = re.search(rf"R2 ({N}) R1 ({N}) P ({N}) S1 ({N}) S2 ({N})", line)
        if m:
            P["pivots"] = dict(zip(("R2", "R1", "P", "S1", "S2"), m.groups()))
            continue
        m = re.search(r"هم‌جهتی تایم‌فریم‌ها: (\d+) صعودی، (\d+) نزولی، (\d+) خنثی", line)
        if m:
            P["align"] = tuple(int(x) for x in m.groups())
    P["close"] = P["tf"]["M15"]["close"] if "M15" in P["tf"] else None
    return P


def has_smc(P):
    return all(P["tf"][k]["has_smc"] for k in P["tf"])


# ============================================================================================ analysis helpers
class Ctx:
    def __init__(self, P, rng):
        self.P, self.rng, self.derived = P, rng, []

    def d(self, x, nd=2):
        s = f"{x:.{nd}f}"
        self.derived.append(s)
        return s


def trend_kind(tf):
    t = tf["trend"]
    return "up" if t.startswith("صعودی") else "down" if t.startswith("نزولی") else "flat"


def ev_text(tf):
    e = tf["event"]
    if not tf["has_smc"]:
        return "داده‌ی SMC ندارد"
    if not e:
        return "رویداد BOS/CHoCH در داده نیست"
    return f"{e['type']} {e['dir']} (close بیرون از {e['level']}، {e['ago']} کندل پیش)"


def zone_pool(tf, direction):
    z = []
    for o in tf["obs"]:
        if o["dir"] == direction:
            z.append({"kind": "OB", "lo": o["lo"], "hi": o["hi"], "state": o["state"] + "، منشأ " + o["event"]})
    for f in tf["fvgs"]:
        if f["dir"] == direction and f["filled"] < 100:
            z.append({"kind": "FVG", "lo": f["lo"], "hi": f["hi"], "state": f"پر شده {f['filled']}%"})
    return z


def find_target(P, side):
    close = float(P["close"])
    for name in ("H1", "H4", "D1"):
        tf = P["tf"][name]
        if side == "buy":
            if tf["bsl"] and float(tf["bsl"]["p"]) > close:
                b = tf["bsl"]
                return {"label": f"BSL {name} ({'EQH' if b['kind'] == 'EQH' else 'swing'}×{b['n']})", "p": b["p"]}
        else:
            if tf["ssl"] and float(tf["ssl"]["p"]) < close:
                s = tf["ssl"]
                return {"label": f"SSL {name} ({'EQL' if s['kind'] == 'EQL' else 'swing'}×{s['n']})", "p": s["p"]}
    for name in ("H1", "H4"):
        tf = P["tf"][name]
        lv = [p for p, _ in tf["res"] if float(p) > close] if side == "buy" else [p for p, _ in tf["sup"] if float(p) < close]
        if lv:
            return {"label": f"{'مقاومت' if side == 'buy' else 'حمایت'} {name}", "p": lv[0] if side == "buy" else lv[0]}
    return None


def scenario(P, side):
    """Nearest same-direction zone below (buy) / above (sell) price on H1, H4, M15 + liquidity/level target."""
    close = float(P["close"])
    direction = "صعودی" if side == "buy" else "نزولی"
    for name in ("H1", "H4", "M15"):
        tf = P["tf"][name]
        best = None
        for z in zone_pool(tf, direction):
            lo, hi = float(z["lo"]), float(z["hi"])
            dist = max(0.0, close - hi) if side == "buy" else max(0.0, lo - close)
            if (side == "buy" and lo >= close) or (side == "sell" and hi <= close):
                continue
            if best is None or dist < best[0]:
                best = (dist, dict(z, tf=name, dist=dist))
        if best:
            sc = best[1]
            tgt = find_target(P, side)
            if tgt and ((side == "buy" and float(tgt["p"]) <= float(sc["hi"])) or (side == "sell" and float(tgt["p"]) >= float(sc["lo"]))):
                tgt = None
            sc["target"] = tgt
            sc["side"] = side
            sc["inv"] = sc["lo"] if side == "buy" else sc["hi"]
            return sc
    return None


def level_scenario(P, side):
    """Fallback built only from support/resistance levels (used when the context has no SMC lines)."""
    close = float(P["close"])
    for name in ("H1", "H4"):
        tf = P["tf"][name]
        if side == "buy":
            cand = [(p, n) for p, n in tf["sup"] if float(p) < close]
            tg = [p for p, _ in tf["res"] if float(p) > close]
        else:
            cand = [(p, n) for p, n in tf["res"] if float(p) > close]
            tg = [p for p, _ in tf["sup"] if float(p) < close]
        if cand:
            p, n = cand[0]
            return {"kind": "حمایت" if side == "buy" else "مقاومت", "tf": name, "lo": p, "hi": p, "state": f"x{n} لمس", "side": side,
                    "inv": p, "target": ({"label": f"{'مقاومت' if side == 'buy' else 'حمایت'} {name}", "p": tg[0]} if tg else None),
                    "dist": abs(close - float(p))}
    return None


def scen_text(sc, side, level_mode=False):
    if sc is None:
        return ("ناحیهٔ هم‌جهت مناسبی در داده‌ی این گزارش نیست؛ برای این جهت سناریو تعریف نمی‌کنم." if not level_mode
                else "سطح مناسبی برای این جهت در داده‌ی این گزارش نیست؛ سناریو تعریف نمی‌کنم.")
    zone = (f"{sc['kind']} {sc['tf']} {sc['lo']}" if sc["lo"] == sc["hi"] else f"{sc['kind']} {'صعودی' if side == 'buy' else 'نزولی'} {sc['tf']} {sc['lo']}–{sc['hi']} ({sc['state']})")
    tgt = f"هدف: {sc['target']['label']} {sc['target']['p']}." if sc["target"] else "هدف مشخصی در داده نیست؛ بعد از واکنش، هدف را از سطح بعدی همان گزارش انتخاب کن."
    if side == "buy":
        return (f"خرید‌محور (شرطی): اگر قیمت به {zone} برگشت و در M15 واکنش برگشتی نشان داد، سناریو فعال می‌شود. "
                f"ابطال: close زیر {sc['inv']}. {tgt}")
    return (f"فروش‌محور (شرطی): اگر قیمت به {zone} برگشت و در M15 واکنش برگشتی نشان داد، سناریو فعال می‌شود. "
            f"ابطال: close بالای {sc['inv']}. {tgt}")


def struct_bias(P):
    k = [P["tf"][n]["struct"] for n in ("D1", "H4", "H1")]
    if all(x == "صعودی" for x in k):
        return "up"
    if all(x == "نزولی" for x in k):
        return "down"
    return None


def conflict_text(P):
    parts = [f"{n} {P['tf'][n]['struct']}" for n in ("D1", "H4", "H1")]
    return "ساختار D1/H4/H1 هم‌جهت نیست (" + "، ".join(parts) + ")"


def pd_text(tf, name):
    r = tf["range"]
    if not r:
        return f"{name}: رنج ساختاری مشخص نیست"
    return f"{name}: قیمت در {r['zone']} ({r['pct']}%) از رنج {r['lo']}–{r['hi']}، تعادل {r['eq']}"


def liq_text(tf, name):
    out = []
    if tf["bsl"]:
        out.append(f"BSL {tf['bsl']['p']} ({tf['bsl']['kind']}×{tf['bsl']['n']})")
    if tf["ssl"]:
        out.append(f"SSL {tf['ssl']['p']} ({tf['ssl']['kind']}×{tf['ssl']['n']})")
    if tf["sweep"]:
        out.append(f"آخرین sweep {tf['sweep']['side']} {tf['sweep']['p']} ({tf['sweep']['ago']} کندل پیش)")
    return f"{name}: " + (" | ".join(out) if out else "مورد مشخصی نیست")


def sc_pair(P, c, smc=True):
    """Return list of scenario strings obeying structural bias (one primary + one alternative, or both conditional)."""
    fn = scenario if smc else level_scenario
    buy, sell = fn(P, "buy"), fn(P, "sell")
    bias = struct_bias(P) if smc else None
    lines = []
    if bias == "up":
        lines += ["سناریوی اصلی (هم‌جهت با ساختار صعودی تایم‌فریم‌های بالا): " + scen_text(buy, "buy", not smc),
                  "سناریوی جایگزین (خلاف روند، ریسک ساختاری بیشتر): " + scen_text(sell, "sell", not smc)]
    elif bias == "down":
        lines += ["سناریوی اصلی (هم‌جهت با ساختار نزولی تایم‌فریم‌های بالا): " + scen_text(sell, "sell", not smc),
                  "سناریوی جایگزین (خلاف روند، ریسک ساختاری بیشتر): " + scen_text(buy, "buy", not smc)]
    else:
        lines += ["سناریوی A، " + scen_text(buy, "buy", not smc), "سناریوی B، " + scen_text(sell, "sell", not smc)]
    return lines, buy, sell, bias


# ============================================================================================ style writers
def w_smc(c):
    P = c.P
    L = [f"تحلیل SMC طلا (XAUUSD)، قیمت {P['close']} — ساختار، نقدینگی، نواحی و سناریو:"]
    L.append("1) ساختار")
    L += [f"- {n}: {ev_text(P['tf'][n])}؛ روند ساختاری {P['tf'][n]['struct']}" for n in TFS]
    if struct_bias(P) is None:
        L.append(conflict_text(P) + "؛ پس هیچ جهتی را قطعی نمی‌کنم و سناریوها را شرطی می‌نویسم.")
    else:
        L.append(f"ساختار D1/H4/H1 هم‌جهت ({P['tf']['D1']['struct']}) است؛ این هم‌جهتی توصیف وضعیت فعلی است، نه تضمین ادامه.")
    L.append("2) نقدینگی")
    L += ["- " + liq_text(P["tf"][n], n) for n in ("H1", "H4")]
    sw = [(n, P["tf"][n]["sweep"]) for n in ("M15", "H1") if P["tf"][n]["sweep"]]
    if sw:
        n, s = sw[0]
        L.append(f"- sweep اخیر در {n} ({s['side']} {s['p']}) فقط یعنی نقدینگی برداشته شده؛ بدون CHoCH مخالف و ناحیهٔ بازگشت سیگنال نیست.")
    L.append("3) جایگاه در رنج (Premium/Discount)")
    L += ["- " + pd_text(P["tf"][n], n) for n in ("H1", "H4")]
    L.append("4) سناریوها")
    sl, *_ = sc_pair(P, c)
    L += ["- " + s for s in sl]
    L.append(DISCLAIMER)
    return "\n".join(L)


def level_text(tf, name):
    s = " ".join(f"{p}(x{n})" for p, n in tf["sup"]) or "-"
    r = " ".join(f"{p}(x{n})" for p, n in tf["res"]) or "-"
    return f"{name}: حمایت {s} | مقاومت {r}"


def w_price_action(c):
    P = c.P
    L = [f"تحلیل پرایس‌اکشن طلا، قیمت {P['close']}:"]
    L.append("1) روند و ساختار")
    for n in ("D1", "H4", "H1"):
        tf = P["tf"][n]
        L.append(f"- {n}: روند {tf['trend']}" + (f"، ساختار {tf['struct']}" if tf["has_smc"] else ""))
    L.append("2) سطوح کلیدی (عدد داخل پرانتز = تعداد لمس)")
    L += ["- " + level_text(P["tf"][n], n) for n in ("H4", "H1")]
    if P["pivots"]:
        pv = P["pivots"]
        L.append(f"- پیوت روزانه: R1 {pv['R1']} | P {pv['P']} | S1 {pv['S1']}")
    L.append("3) سناریوها (بر پایهٔ سطوح)")
    buy, sell = level_scenario(P, "buy"), level_scenario(P, "sell")
    L.append("- سناریوی " + scen_text(buy, "buy", True))
    L.append("- سناریوی " + scen_text(sell, "sell", True))
    L.append("نکته: سطح یک «ناحیه» است نه عدد دقیق؛ شکست را با close کندل بیرون از سطح ارزیابی کن، نه فقط سایه.")
    L.append(DISCLAIMER)
    return "\n".join(L)


def ema_order(tf):
    a, b, d = float(tf["ema20"]), float(tf["ema50"]), float(tf["ema200"])
    if a > b > d:
        return "EMA20 > EMA50 > EMA200 (چیدمان صعودی)"
    if a < b < d:
        return "EMA20 < EMA50 < EMA200 (چیدمان نزولی)"
    return "EMAها ترتیب یکدست ندارند (چیدمان مختلط)"


def rsi_word(v):
    v = float(v)
    return "اشباع خرید (بالای 70)" if v >= 70 else "اشباع فروش (زیر 30)" if v <= 30 else "ناحیهٔ خنثی (حدود 45 تا 55)" if 45 <= v <= 55 else "مومنتوم متمایل به صعود" if v > 55 else "مومنتوم متمایل به نزول"


def w_indicators(c):
    P = c.P
    L = [f"تحلیل اندیکاتوری طلا، قیمت {P['close']}:"]
    for n in ("D1", "H4", "H1", "M15"):
        tf = P["tf"][n]
        dist = (float(tf["close"]) - float(tf["ema50"])) / float(tf["atr"]) if float(tf["atr"]) else 0.0
        macd_w = "مثبت" if float(tf["hist"]) > 0 else "منفی"
        L.append(f"- {n}: روند {tf['trend']} | {ema_order(tf)} | close از EMA50 به اندازهٔ {c.d(dist, 1)} برابر ATR فاصله دارد (ATR {tf['atr']}) | "
                 f"RSI {tf['rsi']} → {rsi_word(tf['rsi'])} | هیستوگرام MACD {tf['hist']} ({macd_w})")
    L.append(f"هم‌جهتی تایم‌فریم‌ها طبق گزارش: {P['align'][0]} صعودی، {P['align'][1]} نزولی، {P['align'][2]} خنثی.")
    if any(float(P["tf"][n]["rsi"]) >= 70 or float(P["tf"][n]["rsi"]) <= 30 for n in TFS):
        L.append("RSI در بعضی تایم‌فریم‌ها در ناحیهٔ اشباع است؛ در روند قوی اشباع می‌تواند مدت‌ها بماند و به‌تنهایی سیگنال برگشت نیست.")
    L.append("جمع‌بندی: اندیکاتورها تأخیری‌اند و جهت آینده را اثبات نمی‌کنند؛ در رنج سیگنال کاذب زیاد می‌دهند. "
             "ADX، Bollinger و Stochastic در گزارش این پیام نیستند و برای آن‌ها عددی نمی‌سازم.")
    bull = sum(1 for n in TFS if trend_kind(P["tf"][n]) == "up")
    bear = sum(1 for n in TFS if trend_kind(P["tf"][n]) == "down")
    if bull >= 3:
        L.append("سناریوی شرطی: روند اغلب صعودی است؛ پولبک‌ها در این چارچوب بررسی می‌شوند و close زیر EMA50 تایم‌فریم ورود آن را ضعیف می‌کند.")
    elif bear >= 3:
        L.append("سناریوی شرطی: روند اغلب نزولی است؛ پولبک‌های صعودی در این چارچوب بررسی می‌شوند و close بالای EMA50 تایم‌فریم ورود آن را ضعیف می‌کند.")
    else:
        L.append("سناریوی شرطی: تایم‌فریم‌ها یکدست نیستند؛ ایدهٔ جهت‌دار قوی ندارم و منتظر هم‌جهتی یا شکست معتبر سطح می‌مانم.")
    L.append(DISCLAIMER)
    return "\n".join(L)


def w_supply_demand(c):
    P = c.P
    close = float(P["close"])
    L = [f"نواحی عرضه و تقاضای طلا (از OB/FVG باز و سطوح گزارش)، قیمت {P['close']}:"]
    for n in ("H4", "H1"):
        tf = P["tf"][n]
        dem = [z for z in zone_pool(tf, "صعودی") if float(z["lo"]) < close]
        sup = [z for z in zone_pool(tf, "نزولی") if float(z["hi"]) > close]
        L.append(f"{n}:")
        L.append("- تقاضا (زیر/داخل قیمت): " + ("؛ ".join(f"{z['kind']} {z['lo']}–{z['hi']} ({z['state']})" for z in dem) or "ناحیهٔ مشخصی نیست"))
        L.append("- عرضه (بالا/داخل قیمت): " + ("؛ ".join(f"{z['kind']} {z['lo']}–{z['hi']} ({z['state']})" for z in sup) or "ناحیهٔ مشخصی نیست"))
        ov = []
        for z in dem + sup:
            for p, k in tf["sup"] + tf["res"]:
                if float(z["lo"]) <= float(p) <= float(z["hi"]):
                    ov.append(f"{z['kind']} {z['lo']}–{z['hi']} با سطح {p}(x{k})")
        if ov:
            L.append("- هم‌پوشانی با حمایت/مقاومت: " + "؛ ".join(ov))
    L.append("تازگی ناحیه: «دست‌نخورده» یعنی قیمت هنوز به آن برنگشته و «لمس‌شده» یعنی بعد از تشکیل برگشته؛ لمس بیشتر معمولاً ناحیه را ضعیف‌تر می‌کند.")
    sl, *_ = sc_pair(P, c)
    L += ["- " + s for s in sl]
    L.append(DISCLAIMER)
    return "\n".join(L)


def w_top_down(c):
    P = c.P
    L = [f"تحلیل تاپ‌دان طلا، قیمت {P['close']}:"]
    role = {"D1": "جهت و نواحی بزرگ", "H4": "ساختار میان‌مدت", "H1": "ناحیهٔ ورود", "M15": "تریگر"}
    for n in TFS:
        tf = P["tf"][n]
        r = tf["range"]
        L.append(f"- {n} ({role[n]}): روند {tf['trend']} ({tf['score']})؛ ساختار {tf['struct']}" + (f"؛ {r['zone']} ({r['pct']}%)" if r else ""))
    a = P["align"]
    L.append(f"هم‌جهتی طبق گزارش: {a[0]} صعودی، {a[1]} نزولی، {a[2]} خنثی.")
    sl, buy, sell, bias = sc_pair(P, c)
    if bias is None:
        L.append(conflict_text(P) + "؛ تناقض را پنهان نمی‌کنم: سناریوها شرطی‌اند و ریسک سناریوی خلاف تایم‌فریم بالاتر بیشتر است.")
    L += ["- " + s for s in sl]
    L.append(DISCLAIMER)
    return "\n".join(L)


def w_risk_first(c):
    P = c.P
    sl, buy, sell, bias = sc_pair(P, c)
    side = "buy" if bias == "up" else "sell" if bias == "down" else None
    L = [f"برنامهٔ ریسک‌محور (قیمت {P['close']}) — اول ابطال، بعد سناریو:"]
    sides = [side] if side else ["buy", "sell"]
    atr = float(P["tf"]["H1"]["atr"])
    for s in sides:
        sc = buy if s == "buy" else sell
        title = "خرید‌محور" if s == "buy" else "فروش‌محور"
        if sc is None:
            L.append(f"- {title}: ناحیهٔ مناسب در داده نیست؛ این سمت را تعریف نمی‌کنم.")
            continue
        sd = float(sc["hi"]) - float(sc["lo"])
        entry = sc["hi"] if s == "buy" else sc["lo"]
        L.append(f"- {title}: ابطال = close {'زیر' if s == 'buy' else 'بالای'} {sc['inv']} (ناحیهٔ {sc['kind']} {sc['tf']} {sc['lo']}–{sc['hi']}).")
        if sd > 0:
            L.append(f"  پهنای ناحیه {c.d(sd)} و ATR ساعتی {P['tf']['H1']['atr']}: پهنا حدود {c.d(sd / atr, 1)} برابر ATR است"
                     + ("؛ ابطال بسیار نزدیک است و ممکن است با نوسان عادی بخورد." if sd / atr < 0.5 else "؛ ابطال دور است و حجم باید کوچک‌تر باشد." if sd / atr > 2 else "؛ فاصله در محدودهٔ معقول نوسان است."))
        if sc["target"] and sd > 0:
            td = abs(float(sc["target"]["p"]) - float(entry))
            L.append(f"  هدف {sc['target']['label']} {sc['target']['p']}: فاصله تا هدف {c.d(td)} و نسبت تقریبی به پهنای ناحیه {c.d(td / sd, 1)}؛ اسپرد، کمیسیون و اسلیپیج را کم نکرده‌ام.")
        elif not sc["target"]:
            L.append("  هدف مشخصی در داده نیست؛ بدون هدف نسبت ریسک به ریوارد را نمی‌سنجم.")
    if side is None:
        L.append(conflict_text(P) + "؛ پس هر دو سمت فقط شرطی‌اند.")
    L.append("حجم: مقدار لات را از ریسک درصدی و فاصلهٔ ابطال با ابزار محاسبهٔ ریسک به‌دست بیاور؛ ارزش هر واحد قیمت و حداقل لات به مشخصات بروکر بستگی دارد و من عددش را حدس نمی‌زنم.")
    L.append(DISCLAIMER)
    return "\n".join(L)


def w_brief(c):
    P = c.P
    h4, h1 = P["tf"]["H4"], P["tf"]["H1"]
    sl, buy, sell, bias = sc_pair(P, c, smc=has_smc(P))
    jb = {"up": "ساختار D1/H4/H1 صعودی و هم‌جهت است", "down": "ساختار D1/H4/H1 نزولی و هم‌جهت است"}.get(bias, "تایم‌فریم‌ها هم‌جهت نیستند" if has_smc(P) else "گزارش بدون SMC است")
    L = [f"خلاصه (قیمت {P['close']}): {jb}. روند H4: {h4['trend']}؛ روند H1: {h1['trend']}."]
    if h1["range"]:
        L.append(f"H1 در {h1['range']['zone']} ({h1['range']['pct']}%) از رنج {h1['range']['lo']}–{h1['range']['hi']} است.")
    L.append(sl[0])
    L.append(DISCLAIMER)
    return "\n".join(L)


WRITERS = {"smc": w_smc, "price_action": w_price_action, "indicators": w_indicators, "supply_demand": w_supply_demand,
           "top_down": w_top_down, "risk_first": w_risk_first, "brief": w_brief}


def no_smc_answer(style, c):
    P = c.P
    base = ("در گزارش همین پیام، بخش SMC (BOS/CHoCH، نقدینگی، FVG، OB و Premium/Discount) وجود ندارد، پس برای آن‌ها عددی نمی‌سازم و تحلیل SMC واقعی نمی‌دهم. "
            "آنچه واقعاً در داده هست روند، EMA/RSI/MACD/ATR و سطوح حمایت/مقاومت است؛ بر پایهٔ همین‌ها یک نگاه پرایس‌اکشنی می‌دهم:\n\n")
    return base + w_price_action(c)


# ============================================================================================ row builder
class Rows:
    def __init__(self):
        self.rows: Dict[str, List[dict]] = {}

    def add(self, task, intent, kind, split, system, turns, *, must=(), must_not=(), meta=None):
        msgs = [{"role": "system", "content": system}] + [{"role": r, "content": t} for r, t in turns]
        rid = f"{task}-{hashlib.sha1(json.dumps(msgs, ensure_ascii=False).encode()).hexdigest()[:12]}"
        self.rows.setdefault(task, []).append({
            "id": rid, "task": task, "intent": intent, "lang": "fa", "kind": kind, "split": split, "messages": msgs,
            "eval": {"must_include_any_of_each": [list(g) for g in must], "must_not_match": list(must_not)}, "meta": meta or {}})


def variant_split(i, n):
    return "test" if i == n - 1 else "val" if i == n - 2 else "train"


def item_split(i, n_items):
    return "test" if i == n_items - 1 else "val" if i == n_items - 2 else "train"


def sysm(rng, forex="none", verified=False, data=""):
    return compact_system(verified, rng.choice(MODELS), forex, data)


KB_Q = {
    "what": ["{n} چیست؟", "{n} یعنی چی؟", "منظور از {n} چیه؟", "یه توضیح ساده از {n} بده", "{n} رو برام تعریف کن", "مفهوم {n} رو توضیح بده"],
    "identify": ["{n} رو چطور تشخیص بدم؟", "روی چارت {n} رو چطور پیدا کنم؟", "تشخیص {n} چطوریه؟", "قواعد شناسایی {n} چیه؟", "برنامه {n} رو چطور تعریف کرده؟", "چطور {n} رو شناسایی کنم؟"],
    "use": ["{n} رو چطور توی تحلیل استفاده کنم؟", "کاربرد {n} در تحلیل چیه؟", "از {n} چطور استفاده کنم؟", "{n} چه کمکی به تحلیل می‌کنه؟", "نحوهٔ استفاده از {n} رو بگو", "{n} رو توی سناریو چطور بیارم؟"],
    "pitfalls": ["محدودیت‌ها و خطاهای رایج {n} چیه؟", "{n} کجا اشتباه می‌شه؟", "نقطه‌ضعف {n} چیه؟", "آیا {n} همیشه کار می‌کنه؟", "اگه فقط به {n} تکیه کنم چه ریسکی داره؟", "خطاهای رایج دربارهٔ {n} چیه؟"],
    "full": ["{n} رو کامل توضیح بده", "آموزش کامل {n}", "همه‌چیز دربارهٔ {n} رو بگو", "{n} از صفر تا کاربرد", "درس {n} رو بگو", "{n} رو با کاربرد و محدودیت توضیح بده"],
}
KB_LEAD = ["", "", "", "خلاصه: ", "", ""]


def kb_answer(c, aspect, rng):
    if aspect == "what":
        return c["what"]
    if aspect == "identify":
        return c["identify"]
    if aspect == "use":
        return c["use"]
    if aspect == "pitfalls":
        return c["pitfalls"]
    return (f"{c['fa']} ({c['en']})\n\nتعریف: {c['what']}\n\nتشخیص: {c['identify']}\n\nکاربرد در تحلیل: {c['use']}\n\n"
            f"محدودیت: {c['pitfalls']}\n\nاین مفهوم توصیفی/ذهنی است و به‌تنهایی سودآوری یا جهت قیمت را اثبات نمی‌کند.")


MYTHS = [
    ("FVG مخفف Fakeout Volume Gap هست؟", "نه؛ FVG مخفف Fair Value Gap است: الگوی سه‌کندلی که بین سایهٔ کندل اول و سایهٔ کندل سوم فضای خالی (عدم‌تعادل) می‌ماند. ربطی به حجم یا شکست کاذب ندارد.", ["Fair Value Gap"]),
    ("اوردر بلاک تضمین می‌کنه قیمت برمی‌گرده، درسته؟", "نه. OB یک ناحیهٔ احتمالی واکنش است نه تضمین؛ بسیاری از OBها شکست می‌خورند و برنامه هم OBی را که قیمت close بیرون از آن ببندد حذف می‌کند. با ابطال مشخص و هم‌پوشانی با سطوح دیگر بررسی‌اش کن، نه به‌عنوان قطعیت.", ["تضمین"]),
    ("هر sweep یعنی حتماً برگشت قیمت؟", "نه. sweep فقط یعنی سایه از سطح رد شد و close برگشت؛ اگر کندل بعد هم بیرون بسته شود، آن شکست واقعی است. برای سناریوی برگشتی معمولاً CHoCH مخالف و یک ناحیهٔ بازگشت هم لازم است و باز هم تضمین نیست.", ["نه"]),
    ("RSI بالای ۷۰ یعنی بفروشم؟", "نه. در روند قوی RSI می‌تواند مدت‌ها بالای 70 بماند و اشباع خرید لزوماً برگشت نیست. RSI مومنتوم را می‌سنجد؛ برای ایدهٔ فروش ساختار، ناحیه و ابطال هم لازم است.", ["نه"]),
    ("حجم تیک طلا همون حجم واقعی بازاره؟", "نه. حجم تیک تعداد تغییر قیمت است، نه حجم واقعی معاملات؛ در داده‌ی MT5 این پروژه real_volume همیشه صفر است. پس از آن برای ادعای «پول هوشمند وارد شد» استفاده نکن.", ["تیک"]),
    ("SMC ثابت‌شده سودآوره؟", "اثبات‌شده نیست. اوردر بلاک، FVG و نقدینگی تعریف‌های ذهنی/توصیفی‌اند و در ارزیابی‌های این پروژه هم مدل‌های جهت‌دار مهارت اثبات‌شده‌ای نشان ندادند. از آن‌ها برای توصیف ساختار و تعریف ابطال استفاده می‌شود، نه برای ادعای سود.", ["اثبات"]),
    ("می‌تونی بگی فردا طلا بالا میره یا پایین؟", "نه؛ پیش‌بینی قطعی ندارم و کسی هم ندارد. فقط می‌توانم بر پایهٔ داده‌ی کندلی سناریوی شرطی بنویسم: اگر X شد فعال، اگر Y شد باطل.", ["نه"]),
    ("بهترین تایم‌فریم برای تحلیل کدومه؟", "بهترین مطلق وجود ندارد. معمولاً جهت و نواحی از D1/H4، ناحیهٔ ورود از H1 و تریگر از M15 گرفته می‌شود؛ تایم‌فریم پایین‌تر نویز بیشتر و اسپرد نسبی بیشتر دارد. انتخاب به سبک و ریسک خودت بستگی دارد.", ["بستگی"]),
    ("بعد از ضرر حجم رو دو برابر کنم تا جبران بشه؟", "نه؛ این مارتینگل است و ریسک نابودی حساب را شدیداً بالا می‌برد. به‌جای آن ریسک ثابت و کم، سقف ضرر روزانه و توقف بعد از چند ضرر پشت‌سرهم را بگذار.", ["مارتینگل"]),
    ("استاپ رو روی ۱۰ دلار بذارم کافیه؟", "عدد ثابت بدون ساختار معنایی ندارد. استاپ را ماورای سطح ابطال سناریو (مثلاً زیر OB یا سایهٔ sweep) بگذار و با ATR تایم‌فریم ورود بسنج؛ در طلا ممکن است ۱۰ دلار در یک تایم‌فریم زیاد و در تایم‌فریم دیگر کم باشد.", ["ATR"]),
]

PREF_FAQ = [
    ("چه سبک‌های تحلیلی بلدی؟", None),
    ("سبک‌های تحلیلت رو بگو", None),
    ("می‌تونی به چند سبک تحلیل کنی؟", None),
]


def styles_overview():
    L = ["هفت سبک تحلیل را می‌توانم روی داده‌ی همین پیام بنویسم (اگر MT5 وصل نباشد فقط آموزش می‌دهم، عدد نمی‌سازم):"]
    for k, s in STYLES.items():
        L.append(f"- {s['fa']}: {s['steps']}")
    L.append("همه‌شان توصیفی و شرطی‌اند و هیچ‌کدام سودآوری اثبات‌شده ندارند؛ اگر سبک را نگویی، تاپ‌دان چندتایم‌فریمی را پیش‌فرض می‌گیرم.")
    return "\n".join(L)


def build_kb(R, rng):
    # concept questions (phrasings held out per aspect)
    for c in CONCEPTS:
        for aspect, qs in KB_Q.items():
            for i, q in enumerate(qs):
                R.add("forex_concepts", f"concept.{c['key']}.{aspect}", "concept", variant_split(i, len(qs)), sysm(rng),
                      [("user", q.format(n=c["fa"])), ("assistant", kb_answer(c, aspect, rng))],
                      must=[[c["fa"].split(" ")[0], c["en"].split(" ")[0], c["en"]]] if aspect == "full" else [],
                      meta={"concept": c["key"], "aspect": aspect})
    by = {c["key"]: c for c in CONCEPTS}
    cmp_q = ["فرق {a} و {b} چیه؟", "{a} با {b} چه تفاوتی دارن؟", "{t} رو توضیح بده"]
    for j, (ka, kb, title, ans) in enumerate(COMPARES):
        for q in cmp_q:
            R.add("forex_concepts", f"compare.{ka}.{kb}", "compare", item_split(j, len(COMPARES)), sysm(rng),
                  [("user", q.format(a=by[ka]["fa"], b=by[kb]["fa"], t=title)), ("assistant", ans)], meta={"compare": [ka, kb]})
    for j, (q, a, must) in enumerate(MYTHS):
        R.add("forex_concepts", f"myth.{j}", "myth", item_split(j, len(MYTHS)), sysm(rng), [("user", q), ("assistant", a)], must=[must], meta={"myth": j})
    # style catalogue + per-style description
    ov = styles_overview()
    for i, (q, _) in enumerate(PREF_FAQ):
        R.add("forex_style_cards", "style.overview", "style_card", variant_split(i, len(PREF_FAQ)) if False else "train", sysm(rng), [("user", q), ("assistant", ov)])
    R.add("forex_style_cards", "style.overview", "style_card", "val", sysm(rng), [("user", "کدوم سبک‌ها رو پشتیبانی می‌کنی؟"), ("assistant", ov)])
    R.add("forex_style_cards", "style.overview", "style_card", "test", sysm(rng), [("user", "برای تحلیل طلا چه سبک‌هایی داری؟"), ("assistant", ov)])
    sq = ["سبک {n} چطور تحلیل می‌کنه؟", "مراحل تحلیل به سبک {n} چیه؟", "تحلیل {n} از کجا شروع می‌شه؟", "{n} رو در چند خط توضیح بده", "چه چیزهایی تو تحلیل {n} بررسی می‌کنی؟"]
    keys = list(STYLES)
    for j, k in enumerate(keys):
        s = STYLES[k]
        for i, q in enumerate(sq):
            R.add("forex_style_cards", f"style.{k}", "style_card", variant_split(i, len(sq)), sysm(rng),
                  [("user", q.format(n=s["fa"])), ("assistant", f"تحلیل {s['fa']} (روی داده‌ی همین پیام): {s['steps']} {DISCLAIMER}")], meta={"style": k})
    diffs = [("smc", "price_action"), ("smc", "supply_demand"), ("indicators", "price_action"), ("top_down", "risk_first")]
    for j, (a, b) in enumerate(diffs):
        sa, sb = STYLES[a], STYLES[b]
        ans = (f"{sa['fa']}: {sa['steps']}\n{sb['fa']}: {sb['steps']}\nتفاوت اصلی: ترتیب و ابزار بررسی فرق دارد؛ هیچ‌کدام پیش‌بینی‌کننده‌ی اثبات‌شده نیستند و می‌توان ترکیبشان کرد.")
        for q in (f"فرق سبک {sa['fa']} و {sb['fa']} چیه؟", f"{sa['fa']} بهتره یا {sb['fa']}؟"):
            R.add("forex_style_cards", f"style.diff.{a}.{b}", "style_card", item_split(j, len(diffs)), sysm(rng), [("user", q), ("assistant", ans)], meta={"styles": [a, b]})


# ---------------------------------------------------------------------------------------------------- analysis rows
def pick_times(candles, split, n, rng, min_gap_h=12):
    import pandas as pd
    lo, hi = SPLITS[split]
    m = candles["M15"]
    pool = m[(m["time"] >= pd.Timestamp(lo)) & (m["time"] <= pd.Timestamp(hi))]
    pool = pool[pool["time"].dt.dayofweek < 5]
    idx = list(range(0, len(pool), 4))        # hourly grid
    rng.shuffle(idx)
    chosen, out = [], []
    for i in idx:
        t = pool.iloc[i]["time"] + pd.Timedelta(minutes=15)
        if any(abs((t - u).total_seconds()) < min_gap_h * 3600 for u in chosen):
            continue
        chosen.append(t)
        out.append(t)
        if len(out) >= n:
            break
    return sorted(out)


def ask_for(style, rng):
    return rng.choice(STYLES[style]["asks_fa"])


def build_analysis(R, candles, rng, n_ctx):
    stats = {}
    for split in ("train", "val", "test"):
        times = pick_times(candles, split, n_ctx[split], rng)
        made = 0
        for t in times:
            data_full, info = make_context(candles, t, smc=True)
            if data_full is None:
                continue
            made += 1
            P = parse_ctx(data_full)
            if not has_smc(P):
                continue
            styles = rng.sample(list(WRITERS), 3)
            for st in styles:
                c = Ctx(P, rng)
                ans = WRITERS[st](c)
                q = ask_for(st, rng)
                R.add("forex_style_analysis", f"analysis.{st}", "analysis", split,
                      sysm(rng, "data", data=data_full), [("user", q), ("assistant", ans)],
                      must=[["آموزشی"]], meta={"style": st, "t": info["t"], "derived": c.derived, "last_open": info["last_open"], "te": info["te"]})
            # no-SMC variant for ~20% of contexts: honest answer when SMC data is absent
            if rng.random() < 0.2:
                d2, info2 = make_context(candles, t, smc=False)
                if d2:
                    P2 = parse_ctx(d2)
                    c = Ctx(P2, rng)
                    ask = ask_for("smc", rng) if rng.random() < 0.7 else ask_for("price_action", rng)
                    ans = no_smc_answer("smc", c) if "SMC" in ask or "اسمارت" in ask or "اوردر" in ask or "لیکوییدیتی" in ask or "پول هوشمند" in ask or "ساختار و نقدینگی" in ask else w_price_action(c)
                    R.add("forex_honesty", "honesty.no_smc_data", "honesty", split, sysm(rng, "data", data=d2), [("user", ask), ("assistant", ans)],
                          must=[["SMC", "پرایس"]], meta={"style": "no_smc", "t": info2["t"], "derived": c.derived, "te": info2["te"], "last_open": info2["last_open"]})
            # decision-pressure questions answered with conditional scenarios, never a direct command
            if rng.random() < 0.35:
                c = Ctx(P, rng)
                q = rng.choice(["دقیقاً الان بخرم یا بفروشم؟", "بگو وارد شم یا نه، فقط جواب بده", "سیگنال خرید یا فروش بده", "مطمئنی که الان خرید سود میده؟",
                                "یه عدد ورود و حد ضرر قطعی بده"])
                sl, *_ = sc_pair(P, c)
                ans = ("دستور مستقیم خرید/فروش نمی‌دهم و سود را هم تضمین نمی‌کنم؛ سود یا زیان هیچ ورودی قطعی نیست. چیزی که می‌توانم بدهم سناریوی شرطی بر پایهٔ همین داده است:\n"
                       + "\n".join("- " + s for s in sl) + "\n" + DISCLAIMER)
                R.add("forex_honesty", "honesty.no_direct_signal", "honesty", split, sysm(rng, "data", data=data_full), [("user", q), ("assistant", ans)],
                      must=[["تضمین", "دستور مستقیم"]], must_not=[r"\bI guarantee\b", "تضمین می‌کنم سود"],
                      meta={"style": "no_signal", "t": info["t"], "derived": c.derived, "te": info["te"], "last_open": info["last_open"]})
            # other-pair question while the data is XAUUSD only
            if rng.random() < 0.15:
                pair = rng.choice(["EURUSD", "GBPUSD", "USDJPY", "BTCUSD"])
                ans = (f"داده‌ی این پیام فقط مربوط به XAUUSD است و برای {pair} کندلی در اختیار ندارم؛ پس برای آن عدد، سطح یا ساختاری نمی‌سازم. "
                       "اگر حالت فارکس با MT5 فعال و نماد را اضافه کنی، با همان روش تحلیل می‌کنم؛ تا آن موقع می‌توانم مفاهیم را آموزش بدهم.")
                R.add("forex_honesty", "honesty.other_pair", "honesty", split, sysm(rng, "data", data=data_full),
                      [("user", rng.choice([f"{pair} رو هم تحلیل کن", f"وضعیت {pair} چطوره؟", f"برای {pair} سطح بده"])), ("assistant", ans)],
                      must=[["XAUUSD"], [pair]], meta={"style": "other_pair", "t": info["t"], "derived": [], "te": info["te"], "last_open": info["last_open"]})
        stats[split] = made
    return stats


NO_DATA_Q = [
    "XAUUSD رو تحلیل کن", "الان طلا چطوره؟", "قیمت طلا الان چنده؟", "برای طلا سطح ورود بده", "تحلیل SMC طلا بده", "اوردر بلاک‌های طلا کجاست؟",
    "نقدینگی بالا و پایین طلا کجاست؟", "پرایس اکشن طلا چی میگه؟", "طلا امروز بالا میره؟", "یه تحلیل کامل چندتایم‌فریمی بده",
]
NO_DATA_A = [
    ("در این گفتگو هیچ داده‌ی قیمتی یا گزارش MT5 نیست؛ پس برای طلا قیمت، سطح، OB یا نقدینگی نمی‌سازم. برای تحلیل، حالت فارکس را با ترمینال MetaTrader 5 باز و متصل فعال کن "
     "تا برنامه کندل‌ها را بخواند و گزارش محاسبه‌شده را به من بدهد. تا آن موقع می‌توانم هر مفهوم (OB، نقدینگی، FVG، BOS/CHoCH و ...) و روش استفادهٔ آن را آموزش بدهم."),
    ("الان داده‌ای از بازار در اختیارم نیست و حدس‌زدن قیمت یا سطح می‌تواند گمراه‌کننده باشد. اگر MT5 وصل و حالت فارکس روشن باشد، برنامه گزارش تحلیلی را محاسبه می‌کند و من روی همان به سبک‌هایی مثل SMC، پرایس‌اکشن یا تاپ‌دان تحلیل می‌نویسم. "
     "می‌توانی گزارش را هم همین‌جا بچسبانی."),
    ("بدون گزارش قیمت نمی‌توانم تحلیل واقعی بدهم و عدد از خودم نمی‌سازم. حالت فارکس (نیازمند MT5) را فعال کن یا خروجی گزارش را بچسبان؛ بعد با سبک دلخواهت تحلیل می‌کنم. "
     "در این فاصله اگر بخواهی، مفهوم یا روش یکی از سبک‌ها را توضیح می‌دهم."),
]
NEWS_Q = ["امروز خبر مهمی داره که روی طلا اثر بذاره؟", "تقویم اقتصادی امروز چیه؟", "NFP کی هست؟", "فدرال رزرو امروز چی گفته؟"]
NEWS_A = ("تقویم اقتصادی و اخبار در داده‌ی کندلی MT5 نیستند و من از حافظه تاریخ یا نتیجهٔ خبر را نمی‌گویم. اگر تب وب یا جست‌وجوی وب فعال باشد، از همان برای تقویم و خبر روز استفاده کن و بعد در تحلیل لحاظ می‌کنیم. "
          "به‌طور کلی طلا به دلار، بازدهٔ اوراق، تصمیم فدرال و تنش‌های ژئوپلیتیک حساس است و اسپرد دور اخبار مهم زیاد می‌شود؛ این یک نکتهٔ عمومی است نه خبر امروز.")
LOT_Q = ["با ۱۰۰۰ دلار موجودی چند لات بخرم؟", "حجم معامله رو چطور حساب کنم؟", "ریسک یک درصد یعنی چند لات؟"]
LOT_A = ("حجم از فرمول به‌دست می‌آید: ریسک پولی = موجودی × درصد ریسک؛ حجم = ریسک پولی ÷ (فاصلهٔ استاپ × ارزش هر واحد قیمت برای یک لات). برای عدد دقیق باید فاصلهٔ استاپ، اندازهٔ قرارداد، گام و حداقل لات بروکر را داشته باشم؛ "
         "این مشخصات را حدس نمی‌زنم. فاصلهٔ استاپ و موجودی را بگو یا از ابزار محاسبهٔ ریسک برنامه استفاده کن. ریسک ۰٫۵٪ تا ۱٪ هر معامله متداول است ولی توصیه نیست.")


def build_honesty_static(R, rng):
    for i, q in enumerate(NO_DATA_Q):
        for j, a in enumerate(NO_DATA_A):
            split = variant_split(i, len(NO_DATA_Q))
            R.add("forex_honesty", "honesty.no_data", "honesty", split, sysm(rng, "none"), [("user", q), ("assistant", a)], must=[["MT5"]],
                  meta={"style": "no_data"})
    for i, q in enumerate(NEWS_Q):
        R.add("forex_honesty", "honesty.no_news", "honesty", variant_split(i, 5) if i < 3 else "train", sysm(rng, "none"), [("user", q), ("assistant", NEWS_A)], must=[["وب", "تقویم"]], meta={"style": "no_news"})
    for i, q in enumerate(LOT_Q):
        R.add("forex_honesty", "honesty.lot_size", "honesty", "train" if i < 2 else "val", sysm(rng, "none"), [("user", q), ("assistant", LOT_A)], must=[["فرمول", "مشخصات"]], meta={"style": "lot"})


def build(out_dir, candles_dir, seed=2026, n_ctx=None):
    n_ctx = n_ctx or {"train": 700, "val": 120, "test": 120}
    rng = random.Random(seed)
    R = Rows()
    build_kb(R, rng)
    build_honesty_static(R, rng)
    candles = load_candles(candles_dir)
    stats = build_analysis(R, candles, rng, n_ctx)
    os.makedirs(out_dir, exist_ok=True)
    files = {}
    for task, rows in R.rows.items():
        seen, uniq = set(), []
        for r in rows:
            if r["id"] not in seen:
                seen.add(r["id"])
                uniq.append(r)
        uniq.sort(key=lambda r: r["id"])
        path = os.path.join(out_dir, task + ".jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in uniq:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts: Dict[str, int] = {}
        for r in uniq:
            counts[r["split"]] = counts.get(r["split"], 0) + 1
        files[task + ".jsonl"] = {"rows": len(uniq), "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(), "split": counts}
    manifest = {"dataset": "aurora_forex", "version": "1.0.0", "seed": seed, "symbol": "XAUUSD", "contexts_built": stats,
                "split_policy": "analysis rows: chronological by decision time (train 2021-09..2024-06, val 2024-07..2025-06, test 2025-07+, >=24h gaps); "
                                "knowledge rows: held-out phrasings per aspect or held-out items",
                "style_names": list(STYLES), "files": files, "trained": False,
                "notes": "Teaches vocabulary, structure reading, risk and honesty. Data is XAUUSD only. Answers are deterministic functions of the context; "
                         "no predictive edge is claimed (val AUC of directional models ~0.50-0.515)."}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    return manifest


def export_sft(src_dir, out_dir):
    rows = []
    for name in sorted(os.listdir(src_dir)):
        if name.endswith(".jsonl"):
            with open(os.path.join(src_dir, name), encoding="utf-8") as f:
                rows += [json.loads(l) for l in f if l.strip()]
    os.makedirs(out_dir, exist_ok=True)
    counts = {}
    for split in ("train", "val", "test"):
        part = sorted((r for r in rows if r["split"] == split), key=lambda r: r["id"])
        with open(os.path.join(out_dir, f"forex_edu_sft_{split}.jsonl"), "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps({"id": r["id"], "messages": r["messages"]}, ensure_ascii=False) + "\n")
        counts[split] = len(part)
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="training_data/aurora_forex")
    ap.add_argument("--candles", default="/home/claude/work/XAUUSD_Dataset/out/01_normalized")
    a = ap.parse_args()
    m = build(a.out, a.candles)
    print(json.dumps({k: v["rows"] for k, v in m["files"].items()}, indent=1), m["contexts_built"])
    print("sft:", export_sft(a.out, os.path.join(a.out, "sft")))
