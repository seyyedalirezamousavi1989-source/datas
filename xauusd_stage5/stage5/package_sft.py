"""Build deterministic, reproducible train/validation SFT archives.

Run after stage5.build_datasets and stage5.audits:
    python -m stage5.package_sft
    python out/09_sft/join_and_verify.py
    python -m stage5.preflight_sft

The test split is intentionally excluded. No training is performed.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from . import s5config as S
from .render_sft import load_snapshots, render

ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = Path(S.DIR_DATASETS)
SFT_DIR = ROOT / "out" / "09_sft"
IDENTITY_DIR = ROOT / "aurora_identity_dataset" / "sft"
TASKS = (
    "market_analysis", "scenario_forecasting", "entry_exit_planning",
    "risk_management", "position_sizing", "trade_management",
    "portfolio_risk", "post_trade_review", "integrated_trading_tasks",
)
SEED = 3407
PART_BYTES = 30 * 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{path.name}:{line_no}: blank line")
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path.name}:{line_no}: row must be an object")
            rows.append(item)
    return rows


def collect(split: str, snapshots: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        path = TASK_DIR / f"{task}.jsonl"
        for record in load_jsonl(path):
            if record.get("split") != split:
                continue
            example = render(record, snapshots)
            example["source"] = "xauusd_stage5"
            rows.append(example)

    identity_path = IDENTITY_DIR / f"identity_sft_{split}.jsonl"
    for identity in load_jsonl(identity_path):
        if not isinstance(identity.get("messages"), list) or not identity["messages"]:
            raise ValueError(f"{identity_path.name}: invalid messages")
        rows.append({
            "id": identity["id"],
            "split": split,
            "messages": identity["messages"],
            "tools": identity.get("tools"),
            "source": "aurora_identity",
        })

    ids = [row.get("id") for row in rows]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError(f"{split}: missing example id")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{split}: duplicate example IDs detected")
    if any(row.get("split") != split for row in rows):
        raise ValueError(f"{split}: split contamination detected")
    return rows


def write_split(split: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw_path = SFT_DIR / f"aurora_forex_combined_{split}.jsonl"
    gzip_path = SFT_DIR / f"aurora_forex_combined_{split}.jsonl.gz"
    part_prefix = f"aurora_forex_combined_{split}.jsonl.gz.part"
    for old in SFT_DIR.glob(part_prefix + "*"):
        old.unlink()
    if gzip_path.exists():
        gzip_path.unlink()

    digest = hashlib.sha256()
    byte_count = 0
    source_counts: dict[str, int] = {}
    with raw_path.open("wb") as raw_out, gzip_path.open("wb") as compressed_out:
        with gzip.GzipFile(filename="", mode="wb", fileobj=compressed_out, mtime=0, compresslevel=6) as gz:
            for row in rows:
                line = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                raw_out.write(line)
                gz.write(line)
                digest.update(line)
                byte_count += len(line)
                source = str(row.get("source", "unspecified"))
                source_counts[source] = source_counts.get(source, 0) + 1

    parts = 0
    with gzip_path.open("rb") as stream:
        while True:
            block = stream.read(PART_BYTES)
            if not block:
                break
            (SFT_DIR / f"{part_prefix}{parts:02d}").write_bytes(block)
            parts += 1
    gzip_path.unlink()
    return {
        "sha256": digest.hexdigest(),
        "rows": len(rows),
        "bytes": byte_count,
        "sources": source_counts,
        "compressed_parts": parts,
    }


def main() -> None:
    SFT_DIR.mkdir(parents=True, exist_ok=True)
    task_manifest_path = TASK_DIR / "manifest.json"
    identity_manifest_path = ROOT / "aurora_identity_dataset" / "manifest.json"
    task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
    identity_manifest = json.loads(identity_manifest_path.read_text(encoding="utf-8"))
    if task_manifest.get("stage5_version") != S.STAGE5_VERSION:
        raise RuntimeError(
            f"Task dataset version {task_manifest.get('stage5_version')} does not match code {S.STAGE5_VERSION}; rebuild task datasets first"
        )
    # Verify every source JSONL against the just-written task manifest before rendering it.
    for name in (*TASKS, "snapshots"):
        source_path = TASK_DIR / f"{name}.jsonl"
        meta = task_manifest.get("files", {}).get(source_path.name)
        if not meta:
            raise RuntimeError(f"Task manifest is missing {source_path.name}")
        if source_path.stat().st_size != int(meta["bytes"]) or sha256_file(source_path) != meta["sha256"]:
            raise RuntimeError(f"Source task dataset integrity failure: {source_path.name}; rebuild Stage 5")
    snapshots = load_snapshots(str(TASK_DIR))
    files = {}
    for split in ("train", "val"):
        rows = collect(split, snapshots)
        random.Random(SEED).shuffle(rows)
        files[f"aurora_forex_combined_{split}.jsonl"] = write_split(split, rows)

    manifest = {
        "dataset": "aurora_forex_combined_sft",
        "sources": {
            "forex_stage5": "XAUUSD Stage 5 task datasets rendered by stage5.render_sft; train/val only",
            "aurora_identity": "AURORA identity/awareness SFT examples; train/val only",
        },
        "source_stage5_version": S.STAGE5_VERSION,
        "source_stage5_schema_version": task_manifest.get("schema_version"),
        "source_task_manifest_sha256": sha256_file(task_manifest_path),
        "source_identity_manifest_sha256": sha256_file(identity_manifest_path),
        "seed_shuffle": SEED,
        "format": "OpenAI-style messages (+tools for tool tasks); fields: id, messages, split, tools?, source",
        "trained": False,
        "files": files,
        "notes": "XAUUSD only. Test split is excluded. This is procedural/tool-use training data, not evidence of predictive edge or profitability.",
    }
    (SFT_DIR / "SFT_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train": files["aurora_forex_combined_train.jsonl"], "val": files["aurora_forex_combined_val.jsonl"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
