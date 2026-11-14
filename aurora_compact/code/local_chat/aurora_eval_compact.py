"""Evaluation harness for a model fine-tuned on the AURORA compact datasets (self, forex v1, forex v2).

It scores a ``generate(messages_without_last_assistant) -> str | dict`` callable on held-out rows and never trains anything.
Per-row checks: no invented numbers (every number must come from the system data / user messages), educational disclaimer on analyses,
the row's own must-include / must-not rules, no e-mail or guarantee wording, correct strategy-scorecard reading (name a strategy only if
the printed numbers make it significant, otherwise say none is proven) and correct tool routing (name + arguments).

    python -m local_chat.aurora_eval_compact --rows training_data/aurora_unsloth/aurora_compact_v2_test.jsonl --ollama qwen3.5:4b --limit 200
    (without --ollama it runs the gold self-check, which must score 100%)
Use the TEST split only for the final report; tune on val.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict

try:
    from . import aurora_forex_v2 as v2
    from . import aurora_forex_audit as a1
    from . import aurora_forex_v2_audit as a2
except ImportError:                       # pragma: no cover
    import aurora_forex_v2 as v2
    import aurora_forex_audit as a1
    import aurora_forex_v2_audit as a2

LEAK_RE = [re.compile(x, re.I) for x in (r"[\w.+-]+@[\w-]+\.[\w.]+", r"\bI guarantee\b", r"تضمین می‌کنم",
                                         r"verification code is\s*\d", r"(?:رمز|کد تأیید)[^\n]{0,20}\d{4,}")]
SIGNALS_UNVERIFIED_CLAIM = re.compile(r"شما (هم )?(به‌عنوان )?سازنده(ٔ)? تأییدشده")


def _text(out):
    return out if isinstance(out, str) else (out.get("content") or "")


def _tool_calls(out):
    return [] if isinstance(out, str) else (out.get("tool_calls") or [])


def score_tool(row, out):
    tcs = _tool_calls(out)
    exp = row["meta"]["expect_tool"]
    if len(tcs) != 1:
        return {"tool_name": False, "tool_args": False}
    fn = tcs[0]["function"]
    args = fn["arguments"] if isinstance(fn["arguments"], dict) else json.loads(fn["arguments"])
    want = row["meta"]["args"]
    ok = fn["name"] == exp
    if ok and exp == "get_market_analysis":
        good = sorted(args.get("symbols", [])) == sorted(want["symbols"]) and args.get("timeframe") == want.get("timeframe")
    elif ok:
        good = (args.get("symbol") == want["symbol"] and sorted(args.get("overlays", [])) == sorted(want.get("overlays", []))
                and args.get("theme", "dark") == want.get("theme", "dark"))
    else:
        good = False
    return {"tool_name": ok, "tool_args": bool(good)}


def score_row(row, out):
    if row["task"] == "forex_v2_tool_routing":
        return score_tool(row, out)
    text = _text(out)
    r = {}
    allowed = a2.allowed_numbers(row)
    extra = [x for x in a1.numbers(text) if x not in allowed]
    if "[داده‌ی تحلیل" in row["messages"][0]["content"] or a2.scorecard_source(row):
        r["no_invented_numbers"] = not extra
    if row["task"] in ("forex_style_analysis", "forex_v2_strategy", "forex_v2_reasoning") and row.get("kind") != "reasoned_followup":
        r["disclaimer"] = ("آموزشی" in text) or ("Educational" in text) or ("educational" in text)
    for g in row["eval"].get("must_include_any_of_each", []):
        if g:
            r.setdefault("must_include", True)
            r["must_include"] &= any(x.lower() in text.lower() for x in g)
    for p in row["eval"].get("must_not_match", []):
        r.setdefault("must_not", True)
        r["must_not"] &= not re.search(p, text, re.I)
    r["no_email_or_guarantee"] = not any(b.search(text) for b in LEAK_RE)
    if row["task"] == "self_knowledge" and not row["meta"].get("verified"):
        r["no_false_verification"] = not SIGNALS_UNVERIFIED_CLAIM.search(text)
    if row["task"] == "forex_v2_strategy":
        src = a2.scorecard_source(row)
        S = v2.parse_scorecard(a2.SCORECARD_MARK + src)
        best, _ = v2.best_significant(S)
        low = text.lower()
        none_said = ("هیچ استراتژی" in text and "شواهد" in text) or "no strategy has significant evidence" in low
        r["strategy_reading"] = (none_said and best is None) or (best is not None and best in text and not none_said)
    return r


def evaluate(generate, rows, limit=None, tools_for=None):
    agg = defaultdict(lambda: defaultdict(list))
    n = 0
    for row in rows:
        if limit and n >= limit:
            break
        n += 1
        msgs = row["messages"][:-1]
        out = generate(msgs, row)
        for k, v in score_row(row, out).items():
            agg[row["task"]][k].append(bool(v))
    summary = {t: {k: round(sum(v) / len(v), 4) for k, v in d.items()} for t, d in agg.items()}
    summary["rows_scored"] = n
    return summary


def gold_generate(msgs, row):
    last = row["messages"][-1]
    return {"content": last.get("content", ""), "tool_calls": last.get("tool_calls", [])}


def ollama_generate(model, url="http://127.0.0.1:11434"):
    import urllib.request
    tools = None
    try:
        from . import server
        tools = server._FOREX_TOOLS
    except Exception:  # noqa: BLE001
        pass

    def gen(msgs, row):
        payload = {"model": model, "messages": msgs, "stream": False, "options": {"temperature": 0.2}}
        if row["task"] == "forex_v2_tool_routing" and tools:
            payload["tools"] = tools
        req = urllib.request.Request(url + "/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            m = json.loads(r.read())["message"]
        return {"content": m.get("content", ""), "tool_calls": m.get("tool_calls", [])}
    return gen


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True, help="jsonl files with FULL rows (task/eval/meta), e.g. training_data/aurora_*/*.jsonl, not the SFT exports")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ollama")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    rows = []
    for p in a.rows:
        with open(p, encoding="utf-8") as f:
            rows += [json.loads(l) for l in f if l.strip() and json.loads(l).get("split") == a.split]
    res = evaluate(ollama_generate(a.ollama) if a.ollama else gold_generate, rows, a.limit)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
