"""Pre-training gate for the combined Stage 5/AURORA SFT JSONL files.

Run after out/09_sft/join_and_verify.py:
    python -m stage5.preflight_sft --tokenizer <local-Qwen3.5-4B-tokenizer-dir>
The tokenizer is optional for structural checks, but required to certify sequence lengths.
This script never edits or resamples training data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SFT_DIR = ROOT / "out" / "09_sft"
MANIFEST = SFT_DIR / "SFT_MANIFEST.json"
EXPECTED_FILES = ("aurora_forex_combined_train.jsonl", "aurora_forex_combined_val.jsonl")
ALLOWED_ROLES = {"system", "user", "assistant", "tool"}


class PreflightError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_message(example: dict[str, Any], line_no: int, path: Path) -> None:
    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        raise PreflightError(f"{path.name}:{line_no}: messages must contain system, user and assistant")
    if messages[0].get("role") != "system" or messages[1].get("role") != "user":
        raise PreflightError(f"{path.name}:{line_no}: first messages must be system then user")
    if messages[-1].get("role") != "assistant":
        raise PreflightError(f"{path.name}:{line_no}: final message must be assistant")
    known_call_ids: set[str] = set()
    pending_call_ids: set[str] = set()
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in ALLOWED_ROLES:
            raise PreflightError(f"{path.name}:{line_no}: invalid message at index {index}")
        role = message["role"]
        if role in {"system", "user", "tool"} and not isinstance(message.get("content"), str):
            raise PreflightError(f"{path.name}:{line_no}: {role} content must be a string")
        if role == "assistant":
            calls = message.get("tool_calls", [])
            if calls:
                if not isinstance(calls, list):
                    raise PreflightError(f"{path.name}:{line_no}: tool_calls must be a list")
                for call in calls:
                    call_id = call.get("id")
                    fn = call.get("function", {})
                    if not call_id or call_id in known_call_ids or not fn.get("name"):
                        raise PreflightError(f"{path.name}:{line_no}: malformed or duplicate tool call")
                    try:
                        json.loads(fn.get("arguments", ""))
                    except (TypeError, json.JSONDecodeError) as exc:
                        raise PreflightError(f"{path.name}:{line_no}: invalid tool arguments JSON") from exc
                    known_call_ids.add(call_id)
                    pending_call_ids.add(call_id)
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not call_id or call_id not in pending_call_ids:
                raise PreflightError(f"{path.name}:{line_no}: tool result has no matching preceding call")
            pending_call_ids.remove(call_id)
    if pending_call_ids:
        raise PreflightError(f"{path.name}:{line_no}: tool calls lack tool results: {sorted(pending_call_ids)}")
    if not isinstance(example.get("id"), str) or not example["id"]:
        raise PreflightError(f"{path.name}:{line_no}: missing string example id")
    if example.get("split") not in {"train", "val"}:
        raise PreflightError(f"{path.name}:{line_no}: unexpected split; test data must not enter SFT")
    if "tools" in example and example["tools"] is not None and not isinstance(example["tools"], list):
        raise PreflightError(f"{path.name}:{line_no}: tools must be a list or null")


def scan_file(path: Path, metadata: dict[str, Any], tokenizer: Any | None, max_length: int) -> dict[str, Any]:
    if not path.is_file():
        raise PreflightError(f"missing required file: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != metadata["sha256"]:
        raise PreflightError(f"{path.name}: SHA256 mismatch ({actual_hash})")
    actual_bytes = path.stat().st_size
    if actual_bytes != metadata["bytes"]:
        raise PreflightError(f"{path.name}: byte count mismatch ({actual_bytes} != {metadata['bytes']})")

    rows = 0
    ids: set[str] = set()
    sources: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    lengths: list[int] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, raw in enumerate(stream, 1):
            if not raw.strip():
                raise PreflightError(f"{path.name}:{line_no}: blank JSONL line")
            try:
                example = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise PreflightError(f"{path.name}:{line_no}: invalid JSON") from exc
            if not isinstance(example, dict):
                raise PreflightError(f"{path.name}:{line_no}: each row must be an object")
            _validate_message(example, line_no, path)
            if example["id"] in ids:
                raise PreflightError(f"{path.name}:{line_no}: duplicate id {example['id']}")
            ids.add(example["id"])
            sources[str(example.get("source", "unspecified"))] += 1
            user_text = example["messages"][1].get("content", "")
            try:
                user_payload = json.loads(user_text)
                tasks[str(user_payload.get("task", "identity_or_other"))] += 1
            except (TypeError, json.JSONDecodeError):
                tasks["identity_or_other"] += 1
            if tokenizer is not None:
                try:
                    token_ids = tokenizer.apply_chat_template(
                        example["messages"], tools=example.get("tools") or None,
                        tokenize=True, add_generation_prompt=False,
                    )
                except Exception as exc:
                    raise PreflightError(
                        f"{path.name}:{line_no}: tokenizer/chat-template failure: {exc}"
                    ) from exc
                lengths.append(len(token_ids))
                if len(token_ids) > max_length:
                    raise PreflightError(
                        f"{path.name}:{line_no}: {len(token_ids)} tokens exceeds max_seq_length={max_length}"
                    )
            rows += 1

    if rows != int(metadata["rows"]):
        raise PreflightError(f"{path.name}: row count mismatch ({rows} != {metadata['rows']})")
    report: dict[str, Any] = {
        "file": path.name, "rows": rows, "bytes": actual_bytes, "sha256": actual_hash,
        "sources": dict(sources), "tasks": dict(tasks),
    }
    if lengths:
        ordered = sorted(lengths)
        def percentile(p: float) -> int:
            return ordered[min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1)]
        report["tokens"] = {
            "count": len(lengths), "max": ordered[-1], "p50": percentile(.50),
            "p90": percentile(.90), "p95": percentile(.95), "p99": percentile(.99),
            "over_limit": sum(length > max_length for length in lengths),
            "max_seq_length": max_length,
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Stage 5/AURORA SFT files before Unsloth training.")
    parser.add_argument("--tokenizer", help="Local tokenizer directory for the exact training model.")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--report", default=str(SFT_DIR / "preflight_report.json"))
    args = parser.parse_args(argv)
    if args.max_seq_length < 128:
        parser.error("--max-seq-length must be at least 128")

    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("trained") is not False:
            raise PreflightError("manifest trained flag must remain false before training")
        tokenizer = None
        if args.tokenizer:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise PreflightError("install transformers to run tokenizer-aware preflight") from exc
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

        reports = []
        seen_ids: set[str] = set()
        for filename in EXPECTED_FILES:
            meta = manifest.get("files", {}).get(filename)
            if not meta:
                raise PreflightError(f"manifest missing metadata for {filename}")
            report = scan_file(SFT_DIR / filename, meta, tokenizer, args.max_seq_length)
            # Cross-file uniqueness is checked by scanning IDs again; this keeps memory bounded
            # per file while still preventing train/validation contamination.
            with (SFT_DIR / filename).open("r", encoding="utf-8") as stream:
                for line_no, raw in enumerate(stream, 1):
                    row_id = json.loads(raw)["id"]
                    if row_id in seen_ids:
                        raise PreflightError(f"duplicate id across train/val at {filename}:{line_no}: {row_id}")
                    seen_ids.add(row_id)
            reports.append(report)

        report_doc = {
            "status": "PASS", "model_training_performed": False,
            "dataset": manifest.get("dataset"), "tokenizer_checked": tokenizer is not None,
            "files": reports,
            "limitations": [
                "This validates file integrity and SFT serialization, not predictive edge or profitability.",
                "Tokenizer results apply only to the exact tokenizer directory supplied.",
                "Keep the test split sealed; it is intentionally absent from the SFT manifest.",
            ],
        }
        Path(args.report).write_text(json.dumps(report_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report_doc, ensure_ascii=False, indent=2))
        return 0
    except (OSError, KeyError, TypeError, json.JSONDecodeError, PreflightError) as exc:
        print(f"SFT PREFLIGHT FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
