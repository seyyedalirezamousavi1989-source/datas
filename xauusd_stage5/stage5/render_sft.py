"""Expand task records (+ snapshots by reference) into chat-format SFT examples for Qwen3.5 (OpenAI-style tool calling).
Only `input` and `target` are used: hindsight outcomes (outcomes/*.jsonl) and `direction_labels` never enter a prompt.

    python -m stage5.render_sft --split train --out sft_train.jsonl            # all tasks except direction_labels
    python -m stage5.render_sft --split val --tasks market_analysis integrated_trading_tasks --limit 200 --out sample.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re

from broker_context.tools import TOOL_SCHEMAS

from . import s5config as S
from .build_datasets import SYSTEM_PROMPT

SNAPSHOT_TASKS = {"market_analysis", "scenario_forecasting", "entry_exit_planning", "integrated_trading_tasks"}
TOOL_TASKS = {"risk_management", "position_sizing", "portfolio_risk", "integrated_trading_tasks"}
FORBIDDEN_IN_PROMPT = re.compile(r"hindsight|first_touch|up_exc_atr|dn_exc_atr|end_ret_atr|\"label\"|long_outcome|short_outcome|net_r_if|final_net_r|outcome\"", re.I)


def load_snapshots(d):
    out = {}
    for line in open(os.path.join(d, "snapshots.jsonl")):
        r = json.loads(line)
        out[r["snapshot_id"]] = r
    return out


def render(rec, snaps):
    task = rec["task"]
    user = {"task": task, "decision_time_bar_close": rec["timestamp"], "time_note": "server-clock timestamps; timezone unverified; all values known at that bar close"}
    if task in SNAPSHOT_TASKS:
        user["market_snapshot"] = snaps[rec["snapshot_id"]]["data"]
    user["input"] = rec["input"]
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": json.dumps(user, ensure_ascii=False, separators=(",", ":"))}]
    t = rec["target"]
    if task in TOOL_TASKS and t.get("tool_calls"):
        calls = [{"id": f"call_{i}", "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], separators=(",", ":"))}} for i, c in enumerate(t["tool_calls"])]
        msgs.append({"role": "assistant", "content": "", "tool_calls": calls})
        for i, r in enumerate(t["tool_results"]):
            msgs.append({"role": "tool", "tool_call_id": f"call_{i}", "content": json.dumps(r, separators=(",", ":"))})
        final = {k: v for k, v in t.items() if k not in ("tool_calls", "tool_results")}
        msgs.append({"role": "assistant", "content": json.dumps(final, ensure_ascii=False, separators=(",", ":"))})
    else:
        msgs.append({"role": "assistant", "content": json.dumps(t, ensure_ascii=False, separators=(",", ":"))})
    assert not FORBIDDEN_IN_PROMPT.search(msgs[1]["content"]), "hindsight field in prompt"
    return {"id": rec["record_id"], "split": rec["split"], "messages": msgs, "tools": TOOL_SCHEMAS if task in TOOL_TASKS else None}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=S.DIR_DATASETS)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    snaps = load_snapshots(a.dir)
    tasks = a.tasks or ["market_analysis", "scenario_forecasting", "entry_exit_planning", "risk_management", "position_sizing", "trade_management",
                        "portfolio_risk", "post_trade_review", "integrated_trading_tasks"]
    n = 0
    with open(a.out, "w") as out:
        for t in tasks:
            k = 0
            for line in open(os.path.join(a.dir, t + ".jsonl")):
                r = json.loads(line)
                if r["split"] != a.split:
                    continue
                out.write(json.dumps(render(r, snaps), ensure_ascii=False) + "\n")
                n += 1
                k += 1
                if a.limit and k >= a.limit:
                    break
    print(f"wrote {n} examples to {a.out}")


if __name__ == "__main__":
    main()
