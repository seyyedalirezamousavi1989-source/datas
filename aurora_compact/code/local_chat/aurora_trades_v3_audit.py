"""Audit for aurora_trades_v3 (exit 1 on failure).   python -m local_chat.aurora_trades_v3_audit training_data/aurora_trades_v3"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter

try:
    from . import aurora_forex_audit as a1
    from . import forex_analysis as fa
except ImportError:                       # pragma: no cover
    import aurora_forex_audit as a1
    import forex_analysis as fa

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
CONST = a1.CONST | {str(x) for x in (8, 36, 585, 69, 516, 21, 22, 11)}


def load(d):
    rows = []
    for n in sorted(os.listdir(d)):
        if n.endswith(".jsonl") and n.startswith("trades_v3") and "sft" not in n:
            with open(os.path.join(d, n), encoding="utf-8") as f:
                rows += [json.loads(l) for l in f if l.strip()]
    return rows


def audit(d):
    rows, res = load(d), {}

    def check(k, ok, det=""):
        res[k] = {"pass": bool(ok), "detail": det}

    bad = []
    for r in rows:
        sysm, user, ans = r["messages"][0]["content"], r["messages"][1]["content"], r["messages"][-1]["content"]
        ok = a1.numbers(sysm) | a1.numbers(user) | {x.lstrip("+") for x in r["meta"].get("derived", [])} | CONST
        extra = sorted(x for x in a1.numbers(ans) if x not in ok)
        if extra:
            bad.append((r["id"], extra[:4]))
    check("numbers_from_context_or_user", not bad, str(bad[:3]))
    la = [r["id"] for r in rows for tf, o in (r["meta"].get("last_open") or {}).items() if o + fa.TIMEFRAME_SECONDS[tf] > r["meta"]["te"]]
    check("no_lookahead_candles", not la, str(la[:3]))
    # outcomes must never be visible in the system context or in eval-row prompts
    leak = [r["id"] for r in rows if r["task"] == "trades_v3_eval" and re.search(r"برنده شد|ضرر خورد|بسته شد|سود [+-]", r["messages"][0]["content"] + r["messages"][1]["content"])]
    check("no_outcome_in_eval_prompts", not leak, str(leak[:3]))
    from datetime import datetime
    ts = {}
    for r in rows:
        if r["meta"].get("t"):
            ts.setdefault(r["split"], []).append(datetime.fromisoformat(r["meta"]["t"]))
    rg = {k: (min(v), max(v)) for k, v in ts.items()}
    check("chronological_split", all(k in rg for k in ("train", "val", "test")) and rg["val"][0] > rg["train"][1] and rg["test"][0] > rg["val"][1], str({k: [str(x) for x in v] for k, v in rg.items()}))
    lk = [r["id"] for r in rows if EMAIL.search(json.dumps(r, ensure_ascii=False)) or re.search(r"تضمین می‌کنم|I guarantee|api_key|password", r["messages"][-1]["content"], re.I)]
    check("no_secrets_email_guarantees", not lk, str(lk[:3]))
    nod = [r["id"] for r in rows if "آموزشی" not in r["messages"][-1]["content"]]
    check("all_have_disclaimer", not nod, str(nod[:3]))
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
    ev = Counter(r["meta"].get("tier") for r in rows if r["task"] == "trades_v3_eval")
    rv = Counter((r["meta"].get("won"), r["meta"].get("tier")) for r in rows if r["task"] == "trades_v3_review")
    check("both_outcomes_present", {True, False} <= {k[0] for k in rv} and len(ev) >= 2, f"eval tiers={dict(ev)} review={dict(rv)}")
    # a win must never be called proof; a loss never proof of a bad decision
    mis = [r["id"] for r in rows if r["task"] == "trades_v3_review" and re.search(r"ثابت می‌کند که تصمیم (درست|بد)|یعنی الگو کار می‌کند", r["messages"][-1]["content"])]
    check("no_outcome_proves_decision", not mis, str(mis[:3]))
    check("max_row_chars_under_14000", max(len(json.dumps(r, ensure_ascii=False)) for r in rows) < 14000)
    return {"rows": len(rows), "by_task_split": {f"{t}/{s}": c for (t, s), c in sorted(Counter((r["task"], r["split"]) for r in rows).items())}, "checks": res,
            "failed": [k for k, v in res.items() if not v["pass"]]}


if __name__ == "__main__":
    o = audit(sys.argv[1] if len(sys.argv) > 1 else "training_data/aurora_trades_v3")
    print(json.dumps(o, ensure_ascii=False, indent=1))
    sys.exit(1 if o["failed"] else 0)
