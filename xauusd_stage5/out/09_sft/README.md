# Full SFT set for Unsloth (Qwen) — XAUUSD Stage 5 + AURORA identity

The checked-in gzip parts are generated artifacts. After rebuilding Stage 5 datasets, regenerate them from source rather than reusing stale chunks:

```bash
python -m stage5.package_sft
python out/09_sft/join_and_verify.py
python -m stage5.preflight_sft
python -m stage5.preflight_sft --tokenizer PATH_TO_EXACT_QWEN35_4B_TOKENIZER --max-seq-length 4096
```

The packager deterministically combines the nine XAUUSD task datasets with the AURORA identity train/validation examples. It excludes the test split, records source-manifest hashes and writes gzip parts for reproducible distribution. The join step reconstructs the full JSONL files and verifies SHA256 and row counts. The preflight step checks message/tool-call structure, unique IDs, split separation, source freshness and manifest integrity.

Do not start training unless:
- `bash run_stage5.sh` completes successfully;
- every Stage 5 audit and unit test passes, including at least 30 naturally generated examples for each supported management action;
- both structural and exact-tokenizer preflight pass;
- the exact Qwen3.5-4B model identifier, tokenizer/chat template and current Unsloth API are verified.

The current checked-in combined SFT files may reflect the previous Stage 5.0.0 build until regenerated. Never infer training readiness from their old row counts or hashes.

Limits: XAUUSD only; directional features showed no validated skill (validation AUC ~0.50–0.515), so this teaches structured analysis, risk discipline and tool use—not proven prediction or profitability. No model training has been executed.
