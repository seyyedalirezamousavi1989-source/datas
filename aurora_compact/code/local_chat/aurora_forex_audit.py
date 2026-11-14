"""Audit for the AURORA forex education/analysis dataset (read-only checks; exit code 1 on any failure).

    python -m local_chat.aurora_forex_audit training_data/aurora_forex
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict

try:
    from . import forex_analysis as fa
except ImportError:                       # pragma: no cover
    import forex_analysis as fa

NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?")
CONST = {str(x) for x in (0, 1, 2, 3, 4, 5, 6, 7, 10, 12, 14, 20, 25, 26, 30, 45, 50, 55, 70, 100, 120, 200, 300, 9, 24, 15, 0.1, 0.25, 0.5, 1.0, 2.0, 99)}
BAD = [r"\bguaranteed (?:profit|returns)\b(?<!no guaranteed profit)", r"تضمین می‌کنم", r"\bI guarantee\b", "@", "api_key", "password",
       "رمز عبور:", "verification code is"]
BAD_RE = [re.compile(b, re.I) for b in BAD]
STYLES = ("smc", "price_action", "indicators", "supply_demand", "top_down", "risk_first", "brief")


def load(d):
    rows = []
    for n in sorted(os.listdir(d)):
        if n.endswith(".jsonl") and not n.startswith("forex_edu_sft"):
            with open(os.path.join(d, n), encoding="utf-8") as f:
                rows += [json.loads(l) for l in f if l.strip()]
    return rows


def numbers(text):
    return {x.lstrip("+") for x in NUM.findall(text)}


def audit(d):
    rows = load(d)
    res = {}

    def check(name, ok, detail=""):
        res[name] = {"pass": bool(ok), "detail": detail}

    # 1. numbers in analysis answers come from the context (or derived / constants)
    bad_nums = []
    for r in rows:
        if r["task"] not in ("forex_style_analysis", "forex_honesty"):
            continue
        sysm, ans = r["messages"][0]["content"], r["messages"][-1]["content"]
        if "[داده‌ی تحلیل" not in sysm:
            continue
        data = sysm.split("[داده‌ی تحلیل", 1)[1]
        ok = numbers(data) | {x.lstrip("+") for x in r["meta"].get("derived", [])} | CONST
        extra = sorted(x for x in numbers(ans) if x not in ok)
        if extra:
            bad_nums.append((r["id"], extra[:5]))
    check("numbers_from_context", not bad_nums, str(bad_nums[:3]))

    # 2. no look-ahead: every candle used was closed at decision time
    la = []
    for r in rows:
        lo = r["meta"].get("last_open")
        if lo:
            for tf, o in lo.items():
                if o + fa.TIMEFRAME_SECONDS[tf] > r["meta"]["te"]:
                    la.append(r["id"])
    check("no_lookahead", not la, str(la[:3]))

    # 3. chronological split with gaps
    from datetime import datetime
    ts = defaultdict(list)
    for r in rows:
        if r["meta"].get("t"):
            ts[r["split"]].append(datetime.fromisoformat(r["meta"]["t"]))
    rngs = {k: (min(v), max(v)) for k, v in ts.items()}
    order_ok = ("train" in rngs and "val" in rngs and "test" in rngs and (rngs["val"][0] - rngs["train"][1]).total_seconds() >= 86400
                and (rngs["test"][0] - rngs["val"][1]).total_seconds() >= 86400)
    check("chronological_split_with_gap", order_ok, str({k: [str(x) for x in v] for k, v in rngs.items()}))

    # 4. context cleanliness (no UTC / session claims; real 2020+ prices)
    dirty = [r["id"] for r in rows if "[داده‌ی تحلیل" in r["messages"][0]["content"] and (" UTC" in r["messages"][0]["content"] or "سشن‌ها:" in r["messages"][0]["content"])]
    check("context_no_unverified_session_or_utc", not dirty, str(dirty[:3]))

    # 5. leakage / secrets / absolutes
    leaks = []
    for r in rows:
        for m in r["messages"]:
            if m["role"] == "assistant" or m["role"] == "system":
                for b in BAD_RE:
                    if b.search(m["content"]):
                        leaks.append((r["id"], b.pattern))
    check("no_secrets_email_guarantees", not leaks, str(leaks[:3]))

    # 6. all analysis rows carry the educational disclaimer
    nod = [r["id"] for r in rows if r["task"] == "forex_style_analysis" and "آموزشی" not in r["messages"][-1]["content"]]
    check("analysis_has_disclaimer", not nod, str(nod[:3]))

    # 7. split hygiene: the same user question never appears in two splits for knowledge tasks
    q2s = defaultdict(set)
    for r in rows:
        if r["task"] in ("forex_concepts", "forex_style_cards"):
            q2s[r["messages"][1]["content"]].add(r["split"])
    leak_q = [q for q, s in q2s.items() if len(s) > 1]
    check("no_question_in_two_splits", not leak_q, str(leak_q[:3]))

    # 8. style / split coverage
    cov = Counter((r["meta"].get("style"), r["split"]) for r in rows if r["task"] == "forex_style_analysis")
    missing = [(s, sp) for s in STYLES for sp in ("train", "val", "test") if cov[(s, sp)] < 5]
    check("style_coverage_min5_per_split", not missing, f"missing={missing}")

    # 9. answers are not all identical per style (anti-canned)
    dup = {}
    for s in STYLES:
        a = [r["messages"][-1]["content"] for r in rows if r["task"] == "forex_style_analysis" and r["meta"].get("style") == s]
        dup[s] = round(1 - len(set(a)) / max(1, len(a)), 3)
    check("analysis_diversity", all(v < 0.05 for v in dup.values()), str(dup))

    # 10. honesty answers actually refuse invented data
    h = [r for r in rows if r["task"] == "forex_honesty" and r["intent"] == "honesty.no_data"]
    check("no_data_rows_have_no_numbers", all(not [x for x in numbers(r["messages"][-1]["content"]) if x not in CONST] for r in h), "")

    # 11. context length
    lens = [len(r["messages"][0]["content"]) + len(r["messages"][-1]["content"]) for r in rows]
    check("max_row_chars_under_12000", max(lens) < 12000, f"max={max(lens)}")
    check("system_prompt_compact", all(len(r["messages"][0]["content"].split("\n\n[داده‌ی تحلیل")[0]) < 700 for r in rows), "chars of fixed part < 700")

    # 12. eval rules satisfied by their own gold answer
    fails = []
    for r in rows:
        a = r["messages"][-1]["content"]
        for g in r["eval"]["must_include_any_of_each"]:
            if g and not any(x.lower() in a.lower() for x in g):
                fails.append((r["id"], g))
        for p in r["eval"]["must_not_match"]:
            if re.search(p, a, re.I):
                fails.append((r["id"], p))
    check("gold_satisfies_eval_rules", not fails, str(fails[:3]))

    summary = {"rows": len(rows), "by_task_split": {f"{t}/{s}": c for (t, s), c in sorted(Counter((r["task"], r["split"]) for r in rows).items())},
               "checks": res, "failed": [k for k, v in res.items() if not v["pass"]]}
    return summary


if __name__ == "__main__":
    out = audit(sys.argv[1] if len(sys.argv) > 1 else "training_data/aurora_forex")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    sys.exit(1 if out["failed"] else 0)
