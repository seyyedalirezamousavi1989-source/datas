# Training instructions: Qwen3.5-4B + QLoRA with Unsloth (NOT executed — no training was run in this project)

Verify package names, model id and API against the current Unsloth / Qwen documentation before running; they change quickly.
The model id below is the expected naming and was **not verified** in this environment.

## 0. Gate before any training
1. Run the audit: `python3 -m stage5.audits` → all checks must pass. 2. Keep the **test split sealed**: do not read test metrics to choose anything.
3. The only genuinely untouched future holdout is data after 2026-11-10 (the current test split was scored once for the frozen baselines).

## 1. Data
```
python3 -m stage5.render_sft --split train --out sft_train.jsonl
python3 -m stage5.render_sft --split val   --out sft_val.jsonl
```
Format: OpenAI-style `messages` (+ `tools` for tool tasks). Check the token length distribution with the Qwen tokenizer; the integrated and portfolio
records are the longest (expect roughly 2–4k tokens); set `max_seq_length` to cover the 99th percentile (start with 4096) rather than truncating tool results.
Mixture: use the files as they are (natural proportions); do **not** upsample LONG/SHORT planning examples — NO_TRADE is intended to dominate.

## 2. Suggested starting configuration (not tuned)
```
load_in_4bit=True, max_seq_length=4096
LoRA: r=16, alpha=32, dropout=0.0, target_modules=[q,k,v,o,gate,up,down]
lr=1e-4 (cosine, warmup 3%), epochs=1-2, per_device_batch=2, grad_accum=16, weight_decay=0.01, optim=adamw_8bit, seed=3407
loss only on assistant tokens (train_on_responses_only); tool results are inputs, not targets
eval on sft_val.jsonl every ~200 steps; early stop on validation loss; keep the best checkpoint
```
Use the model's own chat template (tool-calling variant). Do not enable "thinking" traces unless you add them to the data deliberately.

## 3. Evaluation protocol (after training; uses stage5/evaluate.py scorers)
- decisions: precision/recall/F1/confusion vs the **procedure** targets (imitation) and vs hindsight labels (`direction_labels`); compare with the baselines in `task_evaluation_report.json`;
- probabilities: Brier/log-loss/calibration against the base-rate baseline (`score_probabilities`); a model must not report confidence as probability;
- tool tasks: tool-call validity (schema), argument exactness, and `score_sizing` against the deterministic reference; final numbers must equal tool results;
- management: action agreement with the playbook and net R / drawdown / MFE capture of the actions on val paths;
- stability by year, session, volatility regime; costs under scenarios A/B/C.
Failure criteria worth enforcing: any invented lot size/tick value, any martingale/loss-recovery language, any ignored `incomplete` tool status.

## 4. Runtime
Expose `broker_context.cli tool-schemas` (10 tools) to the model runtime. The model must call tools for sizing, costs, spread and account context;
account facts come from the live read-only MT5 provider (after you authorize it and pass `live_integration_check.py` on Windows) or from the user profile.
Live trading execution is out of scope.
