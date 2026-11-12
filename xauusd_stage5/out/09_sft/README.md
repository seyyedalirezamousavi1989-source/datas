# Full SFT set for Unsloth (Qwen) — XAUUSD Stage 5 + AURORA identity

`python join_and_verify.py` rebuilds `aurora_forex_combined_train.jsonl` (38,735 rows) and `..._val.jsonl` (10,093 rows) and checks SHA-256 against `SFT_MANIFEST.json`.
Each file is a single gzip (30 MB / 8 MB), so you can also `gunzip` it directly. The test split is deliberately not included.

Limits: XAUUSD only; directional features showed no validated skill (val AUC ~0.50-0.515), so this teaches structured analysis, risk and tool use, not proven prediction.
Training instructions: `../../docs/UNSLOTH_QWEN3_5_4B_TRAINING.md` (not executed).
