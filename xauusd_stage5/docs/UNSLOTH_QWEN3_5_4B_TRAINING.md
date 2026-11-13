# Training instructions: Qwen3.5-4B + QLoRA with Unsloth

**Status: NOT TRAINING-READY YET.** No model training has been run and no model weights have been changed. This guide is a cautious workflow, not a claim that the current dataset has predictive edge or that the exact Qwen3.5-4B model/API is supported by every current Unsloth release.

Verify the exact model repository, tokenizer, chat template, Unsloth support, package versions and training API against current upstream documentation before starting.

## 0. Mandatory gates

1. Ensure the Stage 4 source outputs exist at the paths configured in `stage5/s5config.py` (at minimum the M15 multi-timeframe file and M5 normalized candles). If they are missing, restore/extract the Stage 4 packages first.
2. Rebuild Stage 5 task datasets, run all audits/tests, package SFT, join gzip parts and run structural preflight:
   `bash run_stage5.sh`
   This pipeline now includes the Stage 5.1 multi-policy management sampler, SFT packaging, gzip/hash verification and stale-source preflight.
3. Inspect `out/08_stage5_reports/leakage_and_integrity_audit.json`. Every check must pass, especially `management_action_coverage`; all eight supported management actions must have at least 30 naturally generated records. If this gate fails, do not train and do not lower the threshold or fabricate/duplicate examples.
4. Run tokenizer-aware preflight with the exact local tokenizer intended for training:
   `python -m stage5.preflight_sft --tokenizer PATH_TO_LOCAL_QWEN35_4B_TOKENIZER --max-seq-length 4096`
   Review p50/p90/p95/p99/max token counts. The run must pass without silent truncation.
5. Verify the exact Qwen3.5-4B model repository, tokenizer/chat template, current Unsloth support, package versions and training API. The model identifier in older instructions was not verified.
6. Keep the test split sealed. It was already scored once for frozen baseline reporting, so it is not a pristine untouched holdout for further tuning. Prefer a later, genuinely unseen period after 2026-11-10 or a properly designed walk-forward evaluation for final generalization assessment.

## 1. SFT data

The current row counts are written to `out/09_sft/SFT_MANIFEST.json` after every deterministic package build. Do not rely on the older checked-in counts: Stage 5.1 increases naturally sampled management examples, so the regenerated train/validation row counts and hashes will differ.

The test split is intentionally excluded. The files combine XAUUSD Stage 5 task examples and AURORA identity/awareness examples. The preflight checks the manifest SHA256/byte/row counts, JSONL validity, required message structure, tool-call/result linkage, duplicate IDs and train/validation separation. It does not prove label correctness or trading skill.

Use the exact tokenizer and chat template for the model being trained. The preflight reports token percentiles and fails if any rendered example exceeds the selected maximum length. Increase the context length or deliberately redesign/re-render overly long examples; do not silently truncate tool results or assistant targets.

Do not rebalance LONG/SHORT/NO_TRADE or rare management actions by copying records. Any re-sampling must be justified by a documented, outcome-blind method and re-audited for leakage.

## 2. Suggested experimental starting point (not validated)

Only after all gates pass and model compatibility is confirmed, a small pilot could start with:
```
load_in_4bit=True
max_seq_length=4096  # only if tokenizer-aware preflight confirms it is sufficient
LoRA: r=16, alpha=32, dropout=0.0
learning_rate=1e-4, cosine schedule, warmup_ratio=0.03
epochs=1, per_device_train_batch_size=2, gradient_accumulation_steps=16
weight_decay=0.01, seed=3407
```

These are experimental values, not optimized recommendations. Confirm target modules against the actual model architecture; do not assume generic `q,k,v,o,gate,up,down` names are valid. Ensure loss is applied only to intended assistant target tokens. Tool results are context/input, not model-generated target text. Use the model's native chat template and only enable reasoning/thinking formats if deliberately represented in the data.

Run a small smoke-training job first, verify loss decreases without NaNs, inspect decoded generations and tool calls, then decide whether a longer run is warranted. Save the base model identifier, exact package versions, tokenizer revision, config, seed and dataset hashes with every run.

## 3. Evaluation protocol

Evaluate on validation without changing the frozen test split:
- Imitation: precision/recall/F1/confusion against procedural targets, clearly distinguished from hindsight direction labels.
- Probabilities: Brier score, log-loss and calibration against base-rate baselines; do not present model confidence as a calibrated probability.
- Tool tasks: schema validity, argument correctness, status handling and numerical agreement with deterministic tool results.
- Trade management: action agreement, net R, drawdown and MFE capture on validation paths, with rare-action coverage reported separately.
- Robustness: year/session/volatility regime and explicit cost scenarios.

Any claim of predictive edge requires independent, leakage-controlled out-of-sample evaluation. The existing reports do not establish profitability or a validated directional edge.

## 4. Runtime and safety

Expose the `broker_context.cli tool-schemas` definitions to the model runtime only when the corresponding tools are implemented and available. Require tools for broker-specific sizing, costs, spread and account context. Missing, stale or incomplete broker data must cause a safe abstention rather than invented values.

Live MT5 integration is read-only in this project. Live order execution is out of scope. Never allow a fine-tuned language model to bypass deterministic risk checks.
