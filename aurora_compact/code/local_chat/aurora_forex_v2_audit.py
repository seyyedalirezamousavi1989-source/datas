"""Audit for AURORA forex dataset v2.   python -m local_chat.aurora_forex_v2_audit training_data/aurora_forex_v2 [candles_dir]"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

try:
    from . import aurora_forex_v2 as v2
    from . import aurora_forex_dataset as v1
    from . import aurora_forex_audit as a1
    from . import strategy_engine as se
    from . import forex_analysis as fa
except ImportError:                       # pragma: no cover
    import aurora_forex_v2 as v2
    import aurora_forex_dataset as v1
    import aurora_forex_audit as a1
    import strategy_engine as se
    import forex_analysis as fa

TOOLS = {"get_market_analysis": {"symbols", "timeframe"}, "render_forex_chart": {"symbol", "timeframe", "overlays", "fvg_scope", "theme", "annotations"}}
TF_ENUM = {"M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"}
SCORECARD_MARK = "کارت‌امتیاز استراتژی‌ها ("


def load(d):
    rows = []
    for n in sorted(os.listdir(d)):
        if n.endswith(".jsonl"):
            with open(os.path.join(d, n), encoding="utf-8") as f:
                rows += [json.loads(l) for l in f if l.strip()]
    return rows


def scorecard_source(r):
    """Text that holds the scorecard: system data block (real rows) or the first user message (synthetic rows)."""
    sysm = r["messages"][0]["content"]
    if SCORECARD_MARK in sysm:
        return sysm.split(SCORECARD_MARK, 1)[1]
    for m in r["messages"][1:]:
        if m["role"] == "user" and SCORECARD_MARK in m["content"]:
            return m["content"].split(SCORECARD_MARK, 1)[1]
    return None


def allowed_numbers(r):
    parts = []
    sysm = r["messages"][0]["content"]
    if "[داده‌ی تحلیل" in sysm:
        parts.append(sysm.split("[داده‌ی تحلیل", 1)[1])
    parts += [m["content"] for m in r["messages"][1:] if m["role"] == "user"]
    out = set(a1.CONST) | {x.lstrip("+") for x in r["meta"].get("derived", [])}
    for p in parts:
        out |= a1.numbers(p)
    return out


def audit(d, candles_dir=None, spot=12):
    rows = load(d)
    res = {}

    def check(name, ok, detail=""):
        res[name] = {"pass": bool(ok), "detail": detail}

    text_rows = [r for r in rows if r["task"] != "forex_v2_tool_routing"]
    # 1 numbers
    bad = []
    for r in text_rows:
        ok = allowed_numbers(r)
        for m in r["messages"]:
            if m["role"] == "assistant":
                extra = sorted(x for x in a1.numbers(m["content"]) if x not in ok)
                if extra:
                    bad.append((r["id"], extra[:4]))
    check("numbers_from_context_or_user_message", not bad, str(bad[:3]))
    # 2 look-ahead flags + chronological gaps
    la = [r["id"] for r in rows if r["meta"].get("last_open") and any(o + fa.TIMEFRAME_SECONDS[tf] > r["meta"]["te"] for tf, o in r["meta"]["last_open"].items())]
    check("no_lookahead_candles", not la, str(la[:3]))
    ts = defaultdict(list)
    for r in rows:
        if r["meta"].get("t"):
            ts[(r["meta"].get("symbol"), r["split"])].append(datetime.fromisoformat(r["meta"]["t"]))
    gap_ok, det = True, {}
    for sym in {k[0] for k in ts}:
        g = {sp: (min(ts[(sym, sp)]), max(ts[(sym, sp)])) for sp in ("train", "val", "test") if ts.get((sym, sp))}
        if len(g) == 3:
            gap_ok &= (g["val"][0] - g["train"][1]).total_seconds() >= 86400 and (g["test"][0] - g["val"][1]).total_seconds() >= 86400
        det[sym] = {k: [str(x) for x in v] for k, v in g.items()}
    check("chronological_split_with_gap", gap_ok and bool(det), str(det))
    # 3 scorecard flag consistency and reading correctness
    flag_bad, read_bad, n_sc = [], [], 0
    for r in rows:
        if r["task"] != "forex_v2_strategy":
            continue
        src = scorecard_source(r)
        if src is None:
            flag_bad.append(r["id"])
            continue
        n_sc += 1
        S = v2.parse_scorecard(SCORECARD_MARK + src)
        for k, row in S["rows"].items():
            if row["n"] and abs(float(row["lo"])) >= 0.005 and row["flag"] != v2.rule_significant(row):
                flag_bad.append((r["id"], k))
        best, sig = v2.best_significant(S)
        ans = r["messages"][-1]["content"]
        en = r["meta"].get("lang") == "en"
        if best is None:
            ok = ("No strategy has significant evidence" in ans) if en and False else (("no strategy has significant evidence" in ans.lower()) if en else ("هیچ استراتژی‌ای شواهد معنادار ندارد" in ans or "هیچ استراتژی دارای شواهد" in ans))
        else:
            ok = best in ans and ("هیچ استراتژی‌ای شواهد معنادار ندارد" not in ans) and ("no strategy has significant evidence" not in ans.lower())
        if not ok:
            read_bad.append((r["id"], best))
    check("scorecard_flag_matches_rule", not flag_bad, str(flag_bad[:3]))
    check("strategy_reading_correct", not read_bad and n_sc > 0, str(read_bad[:3]))
    syn = [r for r in rows if r["meta"].get("synthetic")]
    sig_share = sum(1 for r in syn if r["meta"]["significant_present"]) / max(1, len(syn))
    check("synthetic_covers_both_branches", 0.3 <= sig_share <= 0.7, f"significant share={sig_share:.2f} of {len(syn)}")
    real_sig = sum(1 for r in rows if r["intent"].startswith("strategy.real") and "هیچ استراتژی‌ای شواهد معنادار ندارد" not in r["messages"][-1]["content"] and r["meta"].get("lang") == "fa")
    # 4 secrets / guarantees / disclaimers
    leaks = [(r["id"], b.pattern) for r in rows for m in r["messages"] if isinstance(m["content"], str) and m["role"] != "user" for b in a1.BAD_RE if b.search(m["content"])]
    check("no_secrets_email_guarantees", not leaks, str(leaks[:3]))
    nodis = [r["id"] for r in rows if r["task"] in ("forex_v2_strategy", "forex_v2_reasoning")
             and not any(x in m["content"] for m in r["messages"] if m["role"] == "assistant" for x in ("آموزشی", "Educational"))]
    check("analysis_has_disclaimer", not nodis, str(nodis[:3]))
    # 5 turns structure
    badt = [r["id"] for r in text_rows if [m["role"] for m in r["messages"]][1:] != (["user", "assistant"] * 4)[:len(r["messages"]) - 1]]
    check("roles_alternate", not badt, str(badt[:3]))
    # 6 tool rows
    tb = []
    for r in rows:
        if r["task"] != "forex_v2_tool_routing":
            continue
        last = r["messages"][-1]
        tc = last.get("tool_calls")
        if not tc or len(tc) != 1:
            tb.append(r["id"])
            continue
        fn = tc[0]["function"]
        args = fn["arguments"]
        if fn["name"] not in TOOLS or not set(args) <= TOOLS[fn["name"]]:
            tb.append(r["id"])
        elif args.get("timeframe") and args["timeframe"] not in TF_ENUM:
            tb.append(r["id"])
        elif fn["name"] == "get_market_analysis" and not (1 <= len(args["symbols"]) <= 3 and all(s == s.upper() for s in args["symbols"])):
            tb.append(r["id"])
        elif fn["name"] == "render_forex_chart" and args["symbol"] != args["symbol"].upper():
            tb.append(r["id"])
    check("tool_calls_valid", not tb, str(tb[:3]))
    # 7 coverage / diversity / length
    cov = Counter((r["intent"].split(".")[0] + "." + r["intent"].split(".")[1], r["split"]) for r in rows)
    miss = [(i, s) for i in {k[0] for k in cov} for s in ("train", "val", "test") if cov[(i, s)] < 3]
    check("intent_coverage_per_split", not miss, str(miss[:5]))
    ans = [m["content"] for r in rows if r["task"] == "forex_v2_reasoning" for m in r["messages"] if m["role"] == "assistant" and m["content"].startswith("تحلیل مستدل")]
    dup = 1 - len(set(ans)) / max(1, len(ans))
    check("reasoning_diversity", dup < 0.05, f"dup={dup:.3f}")
    lens = [sum(len(m["content"] or "") for m in r["messages"]) for r in rows]
    check("max_row_chars_under_14000", max(lens) < 14000, f"max={max(lens)}")
    # 8 causal spot check: rebuild the scorecard from candles truncated at t and compare with the stored text
    if candles_dir:
        import pandas as pd
        rr = random.Random(5)
        pool = [r for r in rows if r["task"] == "forex_v2_strategy" and r["intent"].startswith("strategy.real")]
        mism = []
        cache = {}
        for r in rr.sample(pool, min(spot, len(pool))):
            sym = r["meta"].get("symbol", "XAUUSD")
            if sym not in cache:
                cache[sym] = v1.load_candles(candles_dir, sym)["H1"]
            h1 = cache[sym]
            te = r["meta"]["te"]
            E = se.Engine(h1[h1["epoch"] <= te + 7200].copy())
            sc = E.scorecard(te)
            txt = se.scorecard_text(sc, sym, v2.symbol_digits(sym))
            if SCORECARD_MARK + scorecard_source(r) != txt and txt not in r["messages"][0]["content"]:
                mism.append(r["id"])
        check("scorecard_reproducible_from_truncated_history", not mism, str(mism[:3]))
    return {"rows": len(rows), "by_task_split": {f"{t}/{s}": c for (t, s), c in sorted(Counter((r["task"], r["split"]) for r in rows).items())},
            "checks": res, "failed": [k for k, v in res.items() if not v["pass"]]}


if __name__ == "__main__":
    out = audit(sys.argv[1] if len(sys.argv) > 1 else "training_data/aurora_forex_v2", sys.argv[2] if len(sys.argv) > 2 else None)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    sys.exit(1 if out["failed"] else 0)
