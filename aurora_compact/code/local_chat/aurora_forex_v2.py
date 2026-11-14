"""AURORA forex dataset v2: strategy-evidence reading, reasoned analysis, multi-turn follow-ups, tool routing, English subset.

What is new compared with ``aurora_forex_dataset`` (v1):
* every analysis context also carries the strategy scorecard computed by ``strategy_engine`` (causal, cost-aware, 99% bootstrap CI);
  the model learns to READ it, name a strategy only when the evidence is significant, and otherwise say that no strategy is proven;
* synthetic scorecards pasted by the user (clearly user-supplied) teach the "significant" branch, which real XAUUSD history never reaches;
* reasoned analyses (evidence -> conflicts -> strategy evidence -> conditional scenarios -> risk), with 2-turn follow-ups;
* tool-routing rows (get_market_analysis / render_forex_chart / no tool) in an optional file;
* an English subset for strategy selection.
Answers are deterministic functions of the context; no predictive edge is claimed.  No training happens here.

    python -m local_chat.aurora_forex_v2 --out training_data/aurora_forex_v2
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
    from . import strategy_engine as se
    from . import aurora_forex_dataset as v1
    from .aurora_compact import compact_system
    from .aurora_forex_kb import STYLES
except ImportError:                       # pragma: no cover
    import strategy_engine as se
    import aurora_forex_dataset as v1
    from aurora_compact import compact_system
    from aurora_forex_kb import STYLES

DISCLAIMER = v1.DISCLAIMER
DISCLAIMER_EN = "Educational and descriptive only: not financial advice and no guarantee of profit. Define risk and invalidation before any entry."
FA_NAME = se.STRATEGY_FA

# ============================================================================================ scorecard parsing
SC_HEAD = re.compile(r"کارت‌امتیاز استراتژی‌ها \((\S+) H1؛ .*?، (\d+) ماه اخیر")
SC_ROW = re.compile(r"^- (\w+): n=(\d+) \| میانگین ([-+]\d+\.\d+)R \| CI99% \[([-+]\d+\.\d+)، ([-+]\d+\.\d+)\] \| نرخ برد (\d+)% \| PF (\S+) \| معنادار: (بله|خیر)$")
SC_NONE = re.compile(r"^- (\w+): بدون معامله در بازه$")
SC_REG = re.compile(r"رژیم فعلی: (\S+) \(ADX (\d+)\) و (.+?)\.")
SC_SAME = re.compile(r"^همین رژیم: (.*)\.$")
SC_SAME_ITEM = re.compile(r"(\w+): n=(\d+) میانگین ([-+]\d+\.\d+)R")
SC_SIG = re.compile(r"(\w+)=(خرید|فروش) \(فاصلهٔ استاپ ([\d.]+)، RR ([\d.]+)\)")
SC_WARN = re.compile(r"(\d+) استراتژی آزموده")


def parse_scorecard(text):
    S = {"rows": {}, "symbol": None, "window": None, "regime": None, "adx": None, "vol": None, "same": {}, "signals": {}, "tested": None}
    for line in text.splitlines():
        m = SC_HEAD.search(line)
        if m:
            S["symbol"], S["window"] = m.group(1), m.group(2)
            continue
        m = SC_ROW.match(line)
        if m:
            k = m.group(1)
            S["rows"][k] = {"n": int(m.group(2)), "mean": m.group(3), "lo": m.group(4), "hi": m.group(5), "win": m.group(6), "pf": m.group(7), "flag": m.group(8) == "بله"}
            continue
        m = SC_NONE.match(line)
        if m:
            S["rows"][m.group(1)] = {"n": 0, "mean": None, "lo": None, "hi": None, "win": None, "pf": None, "flag": False}
            continue
        m = SC_REG.search(line)
        if m:
            S["regime"], S["adx"], S["vol"] = m.group(1), m.group(2), m.group(3)
            continue
        m = SC_SAME.match(line)
        if m:
            S["same"] = {k: {"n": int(n), "mean": mean} for k, n, mean in SC_SAME_ITEM.findall(m.group(1))}
            continue
        if line.startswith("سیگنال روی آخرین کندل"):
            S["signals"] = {k: {"side": sd, "stop": st, "rr": rr} for k, sd, st, rr in SC_SIG.findall(line)}
            continue
        m = SC_WARN.search(line)
        if m and line.startswith("هشدار"):
            S["tested"] = m.group(1)
    return S


def rule_significant(row):
    """Re-derive the significance flag from the printed numbers (used by the audit and the writers)."""
    return bool(row["n"] >= se.MIN_TRADES_SIGNIFICANT and row["lo"] is not None and float(row["lo"]) > 0)


def best_significant(S):
    sig = [k for k in se.STRATEGIES if k in S["rows"] and S["rows"][k]["flag"]]
    if not sig:
        return None, []
    sig.sort(key=lambda k: (float(S["rows"][k]["lo"]), float(S["rows"][k]["mean"])), reverse=True)
    return sig[0], sig


def best_point_estimate(S):
    c = [k for k in se.STRATEGIES if k in S["rows"] and S["rows"][k]["n"] >= se.MIN_TRADES_SIGNIFICANT]
    if not c:
        return None
    return max(c, key=lambda k: float(S["rows"][k]["mean"]))


def row_text(S, k):
    r = S["rows"][k]
    return f"میانگین {r['mean']}R با n={r['n']} و CI99% [{r['lo']}، {r['hi']}]"


# ============================================================================================ strategy selection writers
def w_select_fa(S, rng):
    sym = S["symbol"]
    best, sig = best_significant(S)
    L = []
    if best is None:
        L.append(pick(rng, [
            f"نتیجهٔ کارت‌امتیاز {sym}: هیچ استراتژی‌ای شواهد معنادار ندارد؛ در همه‌شان یا n کمتر از 30 است یا بازهٔ CI99% صفر را می‌پوشاند. پس «بهترین استراتژی» اثبات‌شده‌ای برای این لحظه معرفی نمی‌کنم.",
            f"با توجه به آمار، الان هیچ استراتژی دارای شواهد بیرون‌نمونهٔ معنادار برای {sym} نیست (حد پایین CI99% مثبت نیست یا n کم است)؛ انتخاب یکی از آن‌ها فقط حدس می‌شود.",
        ]))
        bp = best_point_estimate(S)
        if bp:
            L.append(f"بالاترین برآورد نقطه‌ای: {bp} ({FA_NAME[bp]}) با {row_text(S, bp)}؛ چون CI صفر را می‌پوشاند این عدد قابل‌اعتماد نیست.")
        else:
            L.append("هیچ استراتژی‌ای معاملهٔ کافی (حداقل 30) در این بازه ندارد، پس حتی برآورد نقطه‌ای معنایی ندارد.")
    else:
        L.append(f"استراتژی دارای شواهد برای {sym}: {best} ({FA_NAME[best]}) با {row_text(S, best)}؛ قاعده: {se.STRATEGY_RULE_FA[best]}.")
        if len(sig) > 1:
            L.append("موارد معنادار دیگر: " + "؛ ".join(f"{k} ({row_text(S, k)})" for k in sig[1:]) + ".")
        g = S["same"].get(best)
        if g and g["n"] >= se.MIN_TRADES_SIGNIFICANT and float(g["mean"]) > 0:
            L.append(f"در رژیم فعلی ({S['regime']}) هم میانگین مثبت است (n={g['n']}، میانگین {g['mean']}R).")
        elif g and g["n"] >= se.MIN_TRADES_SIGNIFICANT:
            L.append(f"اما در رژیم فعلی ({S['regime']}) میانگین این استراتژی {g['mean']}R با n={g['n']} است؛ شواهد کلی با رژیم فعلی همسو نیست، پس احتیاط یا صبر.")
        else:
            L.append(f"در رژیم فعلی ({S['regime']}) معاملهٔ کافی برای تأیید همسویی ندارد.")
        s = S["signals"].get(best)
        if s:
            L.append(f"الان روی آخرین کندل بستهٔ H1 سیگنال {s['side']} دارد (فاصلهٔ استاپ {s['stop']}، RR {s['rr']}). ورود فرضی در open کندل بعد است و فقط وقتی معنا دارد که با ساختار تایم‌فریم بالا و سطح ابطال هم‌خوان باشد.")
        else:
            L.append("الان سیگنال فعالی ندارد؛ صبر برای سیگنال بخشی از خود استراتژی است.")
    L.append(f"رژیم فعلی بازار: {S['regime']} (ADX {S['adx']}) و {S['vol']}.")
    others = [f"{k}={v['side']}" for k, v in S["signals"].items() if k != best]
    if others:
        L.append("سیگنال‌های فعال دیگر (توصیفی، بدون شواهد معنادار): " + "، ".join(others) + ".")
    if best is None:
        L.append("پیشنهاد عملی: تصمیم را روی سناریوی ساختاری شرطی با سطح ابطال و ریسک کوچک بگذار؛ اگر استراتژی‌ای را می‌خواهی امتحان کنی، اول در حساب دمو و با حجم کوچک پایش کن.")
    L.append(f"هشدار چندآزمونی: {S['tested']} استراتژی آزموده شده؛ حتی یک نتیجهٔ معنادار ممکن است تصادفی باشد و گذشته آینده را تضمین نمی‌کند.")
    L.append(DISCLAIMER)
    return "\n".join(L)


def w_select_en(S, rng):
    sym = S["symbol"]
    best, sig = best_significant(S)
    L = []
    if best is None:
        L.append(f"Scorecard result for {sym}: no strategy has significant evidence (n is below 30 or the 99% CI covers zero), so I won't name a proven best strategy for this moment.")
        bp = best_point_estimate(S)
        if bp:
            r = S["rows"][bp]
            L.append(f"Highest point estimate: {bp} with mean {r['mean']}R, n={r['n']}, 99% CI [{r['lo']}, {r['hi']}]; the CI covers zero, so don't rely on it.")
        else:
            L.append("No strategy has enough trades (at least 30) in this window.")
    else:
        r = S["rows"][best]
        L.append(f"Strategy with evidence for {sym}: {best} with mean {r['mean']}R, n={r['n']}, 99% CI [{r['lo']}, {r['hi']}].")
        g = S["same"].get(best)
        if g and g["n"] >= se.MIN_TRADES_SIGNIFICANT:
            L.append(f"In the current regime ({S['regime']}) its mean is {g['mean']}R with n={g['n']}" + ("; consistent with the overall result." if float(g["mean"]) > 0 else "; this does not support the overall result, so be cautious."))
        else:
            L.append(f"There are too few trades in the current regime ({S['regime']}) to confirm the fit.")
        s = S["signals"].get(best)
        L.append(f"It currently signals {'buy' if s['side'] == 'خرید' else 'sell'} on the last closed H1 bar (stop distance {s['stop']}, RR {s['rr']})." if s else "It has no active signal right now; waiting is part of the strategy.")
    L.append(f"Current regime: {S['regime']} (ADX {S['adx']}).")
    L.append(f"Multiple-testing warning: {S['tested']} strategies were tested, so even one significant result may be chance; the past does not guarantee the future.")
    L.append(DISCLAIMER_EN)
    return "\n".join(L)


def pick(rng, opts):
    return opts[rng.randrange(len(opts))]


SEL_Q_FA = ["بهترین استراتژی برای {s} الان چیه؟", "کدوم استراتژی الان روی {s} جواب می‌ده؟", "استراتژی مناسب این لحظه رو پیشنهاد بده", "کارت‌امتیاز استراتژی‌ها رو بخون و بگو کدوم؟",
            "با توجه به آمار، الان با چه استراتژی‌ای وارد شم؟", "شواهد کدوم استراتژی قوی‌تره؟", "الان رژیم بازار چیه و چه استراتژی‌ای می‌خوره؟", "استراتژی برتر این لحظه برای {s}؟"]
SEL_Q_EN = ["What's the best strategy for {s} right now?", "Which strategy has real evidence on {s} at the moment?", "Read the strategy scorecard and tell me what to use",
            "Should I trade any of these strategies now?", "Which strategy fits the current regime?", "Pick the strongest strategy right now"]
REASON_Q = ["تحلیل کامل با استدلال بده", "{s} رو با استدلال قدم‌به‌قدم تحلیل کن", "چرا و بر چه اساسی؟ تحلیل مفصل {s}", "تحلیل بده و دلیل هر نتیجه‌گیری رو هم بگو",
            "یه تحلیل مستدل از {s} با سناریو و ریسک بنویس", "وضعیت {s} رو با شواهد و استدلال بگو", "تحلیل ساختاری و آماری {s} رو با هم بده", "تحلیل چندلایهٔ {s}"]


# ============================================================================================ reasoned analysis
def decide(P):
    class C:  # sc_pair only needs the object for signature compatibility
        pass
    lines, buy, sell, bias = v1.sc_pair(P, C(), True)
    return {"lines": lines, "buy": buy, "sell": sell, "bias": bias}


def zone_reasoning(P, bias):
    r = P["tf"]["H1"]["range"]
    if not r or bias is None:
        return None
    z = r["zone"]
    if bias == "up" and z.startswith("Premium"):
        return f"با این‌که ساختار صعودی است، قیمت H1 در Premium ({r['pct']}%) است؛ خرید همین‌جا ارزان نیست و بهتر است منتظر برگشت به Discount یا ناحیهٔ هم‌جهت بمانی."
    if bias == "up" and z.startswith("Discount"):
        return f"قیمت H1 در Discount ({r['pct']}%) است و برای سناریوی خرید جایگاه بهتری دارد."
    if bias == "down" and z.startswith("Discount"):
        return f"با این‌که ساختار نزولی است، قیمت H1 در Discount ({r['pct']}%) است؛ فروش همین‌جا ارزان است و بهتر است منتظر برگشت به Premium یا ناحیهٔ هم‌جهت بمانی."
    if bias == "down" and z.startswith("Premium"):
        return f"قیمت H1 در Premium ({r['pct']}%) است و برای سناریوی فروش جایگاه بهتری دارد."
    return f"قیمت H1 نزدیک تعادل ({r['pct']}%) است و مزیت جایگاهی مشخصی ندارد."


def w_reasoned(P, S):
    D = decide(P)
    bias = D["bias"]
    L = [f"تحلیل مستدل {S['symbol']} (قیمت {P['close']}):"]
    L.append("1) شواهد ساختاری")
    for n in ("D1", "H4", "H1"):
        tf = P["tf"][n]
        L.append(f"- {n}: {v1.ev_text(tf)}؛ ساختار {tf['struct']} و روند {tf['trend']}")
    L.append("2) نقدینگی و جایگاه")
    L.append("- " + v1.liq_text(P["tf"]["H1"], "H1"))
    L.append("- " + v1.pd_text(P["tf"]["H1"], "H1"))
    L.append("3) استدلال")
    if bias is None:
        L.append(f"چون {v1.conflict_text(P)}، جهت قطعی نمی‌دهم؛ هر دو سناریو شرطی‌اند و سناریوی خلاف تایم‌فریم بالا ریسک ساختاری بیشتری دارد.")
    else:
        word = "صعودی" if bias == "up" else "نزولی"
        L.append(f"چون ساختار D1/H4/H1 هم‌جهت ({word}) است، سناریوی هم‌جهت را اصلی می‌گیرم؛ این هم‌جهتی توصیف وضعیت فعلی است نه تضمین ادامه.")
    zr = zone_reasoning(P, bias)
    if zr:
        L.append(zr)
    L.append("4) شواهد استراتژی‌ها")
    best, _ = best_significant(S)
    if best is None:
        L.append("کارت‌امتیاز هیچ استراتژی معناداری نشان نمی‌دهد؛ پس سناریوی ساختاری زیر توصیفی است و پشتوانهٔ آماری ندارد.")
    else:
        L.append(f"{best} شواهد معنادار دارد ({row_text(S, best)})؛ با این حال شواهد کلی است و با ساختار بالا باید هم‌خوان شود.")
    L.append("5) سناریوها")
    L += ["- " + s for s in D["lines"]]
    L.append("6) ریسک")
    prim = D["buy"] if bias == "up" else D["sell"] if bias == "down" else None
    if prim:
        L.append(f"سطح ابطال سناریوی اصلی {prim['inv']} است؛ با شکست آن (close) ایده باطل می‌شود. حجم را ابزار محاسبهٔ ریسک از فاصلهٔ ابطال و درصد ریسک تعیین کند.")
    else:
        L.append("بدون سناریوی اصلی، ریسک را کم نگه دار و برای هر سمت فقط بعد از تأیید در M15 و با ابطال مشخص عمل کن.")
    conf = "ساختاری متوسط ولی آماری ندارد" if (bias and best is None) else "ساختاری پایین به‌خاطر تناقض و آماری ندارد" if best is None else "ساختاری و آماری محدود"
    L.append(f"جمع‌بندی: اطمینان {conf}.")
    L.append(DISCLAIMER)
    return "\n".join(L), D


def followups(P, S, D):
    out = []
    bias = D["bias"]
    prim = D["buy"] if bias == "up" else D["sell"] if bias == "down" else None
    # why this scenario
    if bias:
        word = "صعودی" if bias == "up" else "نزولی"
        a = (f"چون ساختار D1/H4/H1 هم‌جهت ({word}) است و نزدیک‌ترین ناحیهٔ هم‌جهت "
             + (f"{prim['kind']} {prim['tf']} {prim['lo']}–{prim['hi']} ({prim['state']}) است." if prim else "در داده نیست؛ پس سناریو تعریف نشد.")
             + " سناریوی خلاف روند را فقط جایگزین گذاشتم چون ریسک ساختاری بیشتری دارد. این استدلال توصیفی است و پیش‌بینی نیست.")
    else:
        a = f"سناریوی اصلی نگذاشتم چون {v1.conflict_text(P)}؛ وقتی تایم‌فریم‌ها هم‌جهت نیستند، دو سناریوی شرطی منصفانه‌تر از یک جهت قطعی است."
    out.append((["چرا همین سناریو رو اصلی گذاشتی؟", "دلیل انتخاب سناریوی اصلی چی بود؟", "چرا جهت قطعی ندادی؟" if not bias else "این سناریو رو چطور توجیه می‌کنی؟"], a))
    # H4 flips
    h4 = P["tf"]["H4"]
    r = h4["range"]
    if h4["struct"] == "صعودی" and r:
        a2 = (f"ساختار H4 صعودی است؛ برگشتش یعنی close کندل H4 زیر آخرین کف ساختاری {r['lo']}. اگر این رخ بدهد سناریوی خرید‌محور تضعیف می‌شود و باید سناریوی فروش‌محور را بررسی کنی: "
              + v1.scen_text(D["sell"], "sell") + " این شرط، تأیید نشده و فقط یک «اگر» است.")
    elif h4["struct"] == "نزولی" and r:
        a2 = (f"ساختار H4 نزولی است؛ برگشتش یعنی close کندل H4 بالای آخرین سقف ساختاری {r['hi']}. اگر این رخ بدهد سناریوی فروش‌محور تضعیف می‌شود و باید سناریوی خرید‌محور را بررسی کنی: "
              + v1.scen_text(D["buy"], "buy") + " این شرط، تأیید نشده و فقط یک «اگر» است.")
    else:
        a2 = "ساختار H4 در این گزارش مشخص نیست، پس برای برگشت آن سطح مشخصی نمی‌سازم؛ تا ساختار روشن نشود جهت قطعی نمی‌گیرم."
    out.append((["اگه H4 برگرده چی؟", "اگر ساختار H4 عوض شد چه کار کنم؟", "شرط برگشت H4 چیه؟"], a2))
    # exact invalidation
    if prim:
        a3 = (f"ابطال سناریوی اصلی: close {'زیر' if bias == 'up' else 'بالای'} {prim['inv']} (ناحیهٔ {prim['kind']} {prim['tf']} {prim['lo']}–{prim['hi']}). "
              "فقط سایه‌ی بیرون‌زده کافی نیست؛ بسته‌شدن کندل بیرون از ناحیه ملاک است. حجم را از فاصلهٔ همین ابطال و ریسک درصدی حساب کن.")
    else:
        a3 = "چون سناریوی اصلی ندارم، ابطال هر سمت را جدا می‌گویم: " + " ".join(D["lines"])
    out.append((["ابطال دقیقاً کجاست؟", "کجا بفهمم ایده باطل شد؟", "حد ابطال رو بگو"], a3))
    return out


# ============================================================================================ synthetic scorecards (user-pasted)
PAIRS = [("XAUUSD", 2), ("EURUSD", 5), ("GBPUSD", 5), ("USDJPY", 3), ("GBPJPY", 3), ("AUDUSD", 5), ("USDCAD", 5), ("XAGUSD", 3)]


def synth_scorecard(rng, force_sig=None):
    sym, dg = rng.choice(PAIRS)
    rows, best = {}, rng.choice(se.STRATEGIES)
    plant = rng.random() < 0.5 if force_sig is None else force_sig
    for s in se.STRATEGIES:
        n = rng.choice([0, 12, 25, 40, 80, 150, 300])
        if s == best and plant:
            n = rng.choice([60, 120, 250, 400])
            mean = 2.58 * 1.45 / (n ** 0.5) + rng.uniform(0.03, 0.2)       # planted so that the 99% CI lower bound is above zero
        else:
            mean = rng.gauss(0, 0.08)
        if n == 0:
            rows[s] = {"n": 0, "mean": None, "lo": None, "hi": None, "significant": False, "win": None, "pf": None}
            continue
        hw = 2.58 * 1.45 / (n ** 0.5)
        lo, hi = round(mean - hw, 2), round(mean + hw, 2)
        mean = round(mean, 2)
        win = rng.uniform(0.3, 0.55)
        pf = max(0.2, 1 + mean / 0.9)
        rows[s] = {"n": n, "mean": mean, "lo": lo, "hi": hi, "win": win, "pf": pf, "significant": bool(n >= se.MIN_TRADES_SIGNIFICANT and lo > 0)}
    reg = rng.choice(["روندی", "رنج", "میانی"])
    for s in rows:
        rows[s] = {"all": rows[s], "regime": se.stats([]) if rng.random() < 0.3 else {"n": rng.choice([8, 20, 35, 60, 90]), "mean": round(rng.gauss(0.03, 0.15), 2), "win": 0.4, "pf": 1.0, "lo": 0, "hi": 0, "significant": False},
                   "signal": rng.choice([0, 0, 0, 1, -1]), "stop": rng.uniform(0.0007, 0.003) if dg == 5 else rng.uniform(0.15, 0.4) if dg == 3 else rng.uniform(4, 12), "rr": rng.choice([1.5, 2.0])}
    sc = {"window_months": rng.choice([6, 12, 18, 24]), "regime": reg, "adx": rng.choice([14, 18, 23, 27, 33]), "vol": rng.choice(["نوسان بالا", "نوسان عادی", "نوسان پایین"]),
          "rows": rows, "n_tested": 5}
    return se.scorecard_text(sc, sym, dg), sym


# ============================================================================================ rows
class Rows(v1.Rows):
    def add_msgs(self, task, intent, kind, split, messages, *, lang="fa", meta=None, ev=None):
        rid = f"{task}-{hashlib.sha1(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()[:12]}"
        self.rows.setdefault(task, []).append({"id": rid, "task": task, "intent": intent, "lang": lang, "kind": kind, "split": split, "messages": messages,
                                               "eval": ev or {"must_include_any_of_each": [], "must_not_match": []}, "meta": meta or {}})


def sys_with(rng, forex, data=""):
    return v1.sysm(rng, forex, data=data)


def split_phr(i, n):
    return v1.variant_split(i, n)


def build_selection_synth(R, rng, n_per_split):
    qs = SEL_Q_FA
    for split, n in n_per_split.items():
        r2 = random.Random({"train": 101, "val": 202, "test": 303}[split])
        for k in range(n):
            plant = (k % 2 == 0)
            text, sym = synth_scorecard(r2, force_sig=plant)
            S = parse_scorecard(text)
            en = (k % 5 == 4)
            q = (r2.choice(SEL_Q_EN) if en else r2.choice(qs)).format(s=sym)
            ans = w_select_en(S, r2) if en else w_select_fa(S, r2)
            sys_ = sys_with(r2, "none")
            R.add("forex_v2_strategy", "strategy.synthetic" + (".en" if en else ""), "synthetic_scorecard", split, sys_, [("user", text + "\n\n" + q), ("assistant", ans)],
                  must=[["Educational" if en else "آموزشی"]], meta={"synthetic": True, "significant_present": best_significant(S)[0] is not None, "lang": "en" if en else "fa", "derived": []})
            R.rows["forex_v2_strategy"][-1]["lang"] = "en" if en else "fa"


SYMBOL_DIGITS = {"XAUUSD": 2, "XAGUSD": 3, "BTCUSD": 2, "USDJPY": 3, "EURJPY": 3, "GBPJPY": 3, "AUDJPY": 3, "CADJPY": 3, "CHFJPY": 3}
SYMBOL_SPREAD_POINTS = {"XAUUSD": 20, "XAGUSD": 30, "BTCUSD": 2000}


def symbol_digits(sym):
    return SYMBOL_DIGITS.get(sym, 5)


def build_real(R, candles, E, rng, n_ctx, symbol="XAUUSD"):
    dg = symbol_digits(symbol)
    spread = SYMBOL_SPREAD_POINTS.get(symbol, 12) * 10 ** -dg
    stats = {}
    for split in ("train", "val", "test"):
        times = v1.pick_times(candles, split, n_ctx[split], rng)
        made = 0
        for t in times:
            data, info = v1.make_context(candles, t, smc=True, symbol=symbol, digits=dg, spread=spread)
            if data is None:
                continue
            sc = E.scorecard(info["te"])
            if sc is None:
                continue
            P = v1.parse_ctx(data)
            if not v1.has_smc(P):
                continue
            made += 1
            full = data + "\n\n" + se.scorecard_text(sc, symbol, dg)
            S = parse_scorecard(full)
            meta = {"t": info["t"], "te": info["te"], "last_open": info["last_open"], "derived": [], "sc_i": sc["i"], "symbol": symbol}
            # strategy selection (fa; en for 1 in 4)
            en = rng.random() < 0.25
            q = (rng.choice(SEL_Q_EN) if en else rng.choice(SEL_Q_FA)).format(s=symbol)
            ans = w_select_en(S, rng) if en else w_select_fa(S, rng)
            R.add("forex_v2_strategy", "strategy.real" + (".en" if en else ""), "strategy_selection", split, sys_with(rng, "data", full), [("user", q), ("assistant", ans)],
                  must=[["Educational" if en else "آموزشی"]], meta=dict(meta, lang="en" if en else "fa"))
            R.rows["forex_v2_strategy"][-1]["lang"] = "en" if en else "fa"
            # reasoned analysis (+ follow-ups for 40%)
            if rng.random() < 0.7:
                ans, D = w_reasoned(P, S)
                q = rng.choice(REASON_Q).format(s=symbol)
                turns = [("user", q), ("assistant", ans)]
                kind = "reasoned_analysis"
                if rng.random() < 0.4:
                    fq, fa_ = rng.choice(followups(P, S, D))
                    turns += [("user", rng.choice(fq)), ("assistant", fa_)]
                    kind = "reasoned_followup"
                R.add("forex_v2_reasoning", "reasoning." + kind, kind, split, sys_with(rng, "data", full), turns,
                      must=[] if kind == "reasoned_followup" else [["آموزشی"]], meta=meta)
        stats[split] = made
    return stats


TOOL_PAIRS = [("طلا", "XAUUSD"), ("XAUUSD", "XAUUSD"), ("یورو دلار", "EURUSD"), ("EURUSD", "EURUSD"), ("پوند دلار", "GBPUSD"), ("دلار ین", "USDJPY"), ("پوند ین", "GBPJPY"),
              ("نقره", "XAGUSD"), ("بیت‌کوین", "BTCUSD"), ("دلار فرانک", "USDCHF"), ("دلار کانادا", "USDCAD"), ("دلار استرالیا", "AUDUSD")]
TF_WORDS = [("ساعتی", "H1"), ("چهارساعته", "H4"), ("۱۵ دقیقه", "M15"), ("روزانه", "D1")]
ANALYSIS_T = ["تحلیل {p} رو بده", "{p} الان چطوره؟", "سطح ورود و ابطال برای {p} بگو", "سناریوی {p} رو بنویس", "بهترین استراتژی {p} الان چیه؟", "{p} رو با SMC تحلیل کن", "وضعیت {p} رو بررسی کن", "آیا {p} فرصت معامله داره؟"]
CHART_T = [("چارت {p} رو نشون بده", {}), ("نمودار {p} با FVG بکش", {"overlays": ["fvg"]}), ("اسکرین‌شات چارت {p} با پس‌زمینه سفید", {"theme": "light"}),
           ("نمودار {p} با سقف و کف", {"overlays": ["swings"]}), ("فلش سناریوی محتمل رو روی چارت {p} بذار", {"overlays": ["scenario"]}), ("یه تصویر از نمودار {p} بده", {}),
           ("چارت {p} با FVG و سقف و کف", {"overlays": ["fvg", "swings"]}), ("نمودار {p} رو سفید کن", {"theme": "light"})]
CONCEPT_T = ["order block چیه؟", "FVG یعنی چی؟", "لیکوییدیتی رو توضیح بده", "فرق BOS و CHoCH چیه؟", "ریسک به ریوارد یعنی چی؟", "Premium و Discount چیه؟"]


def build_tools(R, rng):
    def call(name, args):
        return {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}]}
    for i, t in enumerate(ANALYSIS_T):
        for j, (p, sym) in enumerate(TOOL_PAIRS):
            sp = split_phr(i, len(ANALYSIS_T))
            tf = TF_WORDS[(i + j) % len(TF_WORDS)] if (i + j) % 3 == 0 else None
            q = t.format(p=p) + (f" روی {tf[0]}" if tf else "")
            args = {"symbols": [sym]}
            if tf:
                args["timeframe"] = tf[1]
            R.add_msgs("forex_v2_tool_routing", "tool.get_market_analysis", "tool_call", sp,
                       [{"role": "system", "content": sys_with(rng, "none")}, {"role": "user", "content": q}, call("get_market_analysis", args)],
                       meta={"expect_tool": "get_market_analysis", "args": args})
    for i, (t, extra) in enumerate(CHART_T):
        for j, (p, sym) in enumerate(TOOL_PAIRS):
            sp = split_phr(i, len(CHART_T))
            args = dict({"symbol": sym}, **extra)
            R.add_msgs("forex_v2_tool_routing", "tool.render_forex_chart", "tool_call", sp,
                       [{"role": "system", "content": sys_with(rng, "none")}, {"role": "user", "content": t.format(p=p)}, call("render_forex_chart", args)],
                       meta={"expect_tool": "render_forex_chart", "args": args})
    for i, p in enumerate(TOOL_PAIRS):
        a, b = p, TOOL_PAIRS[(i + 3) % len(TOOL_PAIRS)]
        if a[1] == b[1]:
            continue
        args = {"symbols": [a[1], b[1]]}
        R.add_msgs("forex_v2_tool_routing", "tool.get_market_analysis.multi", "tool_call", "train" if i % 6 < 4 else "val" if i % 6 == 4 else "test",
                   [{"role": "system", "content": sys_with(rng, "none")}, {"role": "user", "content": f"{a[0]} و {b[0]} رو با هم تحلیل کن"}, call("get_market_analysis", args)],
                   meta={"expect_tool": "get_market_analysis", "args": args})


def build(out_dir, candles_dir, seed=2027, n_ctx=None, n_synth=None, symbols=("XAUUSD",)):
    n_ctx = n_ctx or {"train": 520, "val": 90, "test": 90}
    if len(symbols) > 1:                       # keep the total size comparable: split the budget across symbols
        n_ctx = {k: max(30, v // len(symbols)) for k, v in n_ctx.items()}
    n_synth = n_synth or {"train": 260, "val": 40, "test": 40}
    rng = random.Random(seed)
    R = Rows()
    stats = {}
    for sym in symbols:
        candles = v1.load_candles(candles_dir, sym)
        E = se.Engine(candles["H1"])
        stats[sym] = build_real(R, candles, E, rng, n_ctx, sym)
    build_selection_synth(R, rng, n_synth)
    build_tools(R, rng)
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
        files[task + ".jsonl"] = {"rows": len(uniq), "split": counts, "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    man = {"dataset": "aurora_forex_v2", "version": "2.0.0", "seed": seed, "symbols": list(symbols),
           "contexts_built": stats, "files": files, "trained": False,
           "strategy_engine": {"strategies": list(se.STRATEGIES), "max_hold_bars": se.MAX_HOLD, "stop_atr": se.STOP_ATR, "ci": se.CI_LEVEL,
                               "min_trades_significant": se.MIN_TRADES_SIGNIFICANT},
           "notes": "Real XAUUSD history shows NO significant strategy at any sampled time, so the 'significant' branch is taught with user-pasted synthetic scorecards "
                    "(meta.synthetic=true). forex_v2_tool_routing.jsonl is optional (tool_calls format) and is not merged by default."}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    return man


def export_sft(src_dir, out_dir, include_tools=False):
    rows = []
    for name in sorted(os.listdir(src_dir)):
        if name.endswith(".jsonl") and (include_tools or "tool_routing" not in name):
            with open(os.path.join(src_dir, name), encoding="utf-8") as f:
                rows += [json.loads(l) for l in f if l.strip()]
    os.makedirs(out_dir, exist_ok=True)
    c = {}
    for sp in ("train", "val", "test"):
        part = sorted((r for r in rows if r["split"] == sp), key=lambda r: r["id"])
        with open(os.path.join(out_dir, f"forex_v2_sft_{sp}.jsonl"), "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps({"id": r["id"], "messages": r["messages"]}, ensure_ascii=False) + "\n")
        c[sp] = len(part)
    return c


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="training_data/aurora_forex_v2")
    ap.add_argument("--candles", default="/home/claude/work/XAUUSD_Dataset/out/01_normalized")
    ap.add_argument("--symbols", default="XAUUSD", help="comma separated; needs {SYMBOL}_{M15,H1,H4,Daily}_normalized.csv in --candles")
    a = ap.parse_args()
    m = build(a.out, a.candles, symbols=tuple(x.strip().upper() for x in a.symbols.split(",") if x.strip()))
    print(json.dumps({k: v["rows"] for k, v in m["files"].items()}, indent=1), m["contexts_built"])
    print("sft:", export_sft(a.out, os.path.join(a.out, "sft")))
