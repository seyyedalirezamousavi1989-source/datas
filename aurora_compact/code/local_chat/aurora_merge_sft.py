"""Merge the compact SFT sets into Unsloth/TRL-ready files.

    # v1: self-knowledge + forex education/analysis
    python -m local_chat.aurora_merge_sft training_data/aurora_self training_data/aurora_forex training_data/aurora_unsloth
    # v2: v1 + strategy-evidence / reasoning / follow-ups (add --tools to include the tool-calling routing rows)
    python -m local_chat.aurora_merge_sft training_data/aurora_self training_data/aurora_forex training_data/aurora_unsloth \
        --v2 training_data/aurora_forex_v2 [--tools]

v1 writes aurora_compact_{train,val,test}.jsonl; v2 writes aurora_compact_v2_{train,val,test}.jsonl (each line {"id", "messages"}).  Deterministic
shuffle by id hash.  The test file is for the final evaluation only.  The older long-prompt identity set (training_data/aurora_identity) is NOT
merged on purpose: it uses a different system prompt and mixing two prompt formats would defeat the context saving.
Tool rows carry an assistant message with "tool_calls"; use a chat template that supports it (Qwen3 does) or leave --tools off.
"""
import argparse
import hashlib
import json
import os


def _read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _stringify_tool_args(messages):
    """HF `datasets` unifies the schema of nested dicts (arguments with different keys would be padded with nulls), so tool-call arguments are
    stored as JSON strings in the merged files; the Qwen chat template accepts string arguments."""
    out = []
    for m in messages:
        if m.get("tool_calls"):
            m = dict(m, tool_calls=[{"type": "function", "function": {"name": tc["function"]["name"],
                                                                      "arguments": json.dumps(tc["function"]["arguments"], ensure_ascii=False)}} for tc in m["tool_calls"]])
        out.append(m)
    return out


def main(self_dir, forex_dir, out_dir, v2_dir=None, tools=False):
    os.makedirs(out_dir, exist_ok=True)
    counts = {}
    for sp in ("train", "val", "test"):
        rows = _read(os.path.join(self_dir, "sft", f"self_sft_{sp}.jsonl")) + _read(os.path.join(forex_dir, "sft", f"forex_edu_sft_{sp}.jsonl"))
        name = "aurora_compact"
        if v2_dir:
            name = "aurora_compact_v2"
            rows += _read(os.path.join(v2_dir, "sft", f"forex_v2_sft_{sp}.jsonl"))
            if tools:
                with_tools = _read(os.path.join(v2_dir, "forex_v2_tool_routing.jsonl"))
                rows += [{"id": r["id"], "messages": _stringify_tool_args(r["messages"])} for r in with_tools if r["split"] == sp]
        rows.sort(key=lambda r: hashlib.sha1(r["id"].encode()).hexdigest())
        with open(os.path.join(out_dir, f"{name}_{sp}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[sp] = len(rows)
    print(counts)
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("self_dir")
    ap.add_argument("forex_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--v2")
    ap.add_argument("--tools", action="store_true")
    a = ap.parse_args()
    main(a.self_dir, a.forex_dir, a.out_dir, a.v2, a.tools)
