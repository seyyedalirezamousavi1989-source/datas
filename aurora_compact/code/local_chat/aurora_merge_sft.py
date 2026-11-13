"""Merge the compact self-knowledge set and the forex education/analysis set into Unsloth/TRL-ready files.

    python -m local_chat.aurora_merge_sft training_data/aurora_self training_data/aurora_forex training_data/aurora_unsloth

Output: aurora_compact_{train,val,test}.jsonl, each line {"id", "messages"}.  Deterministic shuffle (by id hash).  The test file is for
the final evaluation only; do not tune on it.  The older long-prompt identity set (training_data/aurora_identity) is NOT merged on purpose:
it uses a different (long) system prompt, and mixing two prompt formats would defeat the context saving.
"""
import hashlib
import json
import os
import sys


def main(self_dir, forex_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    counts = {}
    for sp in ("train", "val", "test"):
        rows = []
        for path in (os.path.join(self_dir, "sft", f"self_sft_{sp}.jsonl"), os.path.join(forex_dir, "sft", f"forex_edu_sft_{sp}.jsonl")):
            with open(path, encoding="utf-8") as f:
                rows += [json.loads(l) for l in f if l.strip()]
        rows.sort(key=lambda r: hashlib.sha1(r["id"].encode()).hexdigest())
        with open(os.path.join(out_dir, f"aurora_compact_{sp}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[sp] = len(rows)
    print(counts)
    return counts


if __name__ == "__main__":
    main(*sys.argv[1:4])
