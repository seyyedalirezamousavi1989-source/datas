# XAUUSD Stage 5 — Final Report (implemented / tested / partial / unavailable)

Date: 2026-11-13. No model was trained or modified. No profitability or predictive-accuracy claim is made.\n\n## 0. Current rebuild status (important)\nThe code has been updated to Stage 5.1.0: management states are now sampled from all nine simulator policy paths, with dense M5 recording around multi-target and trailing policies; the sampler selects naturally observed playbook actions only and does not fabricate or duplicate labels. A reproducible SFT packager and stale-package preflight gate were added. **The checked-in `out/07_task_datasets`, reports and `out/09_sft` artifacts are still the previous 5.0.0 build until `bash run_stage5.sh` completes successfully. Do not train from the old checked-in SFT files.**

## 1. Implemented and tested
| Area | Status | Evidence |
|---|---|---|
| Triple-barrier labeling (SL 1.5×ATR14, TP 2R, 32 M15 bars, next-bar-open entry, M5 replay for same-bar double touch, worst-case SL-first with `ambiguous` flag) | Implemented, tested | `tests/test_stage5_labeling.py` |
| Cost scenarios A/B/C (all **assumed, unverified**), hashed `CostProfile`, non-overwriting `LabelRunRegistry` | Implemented, tested | labeling tests |
| Chronological splits with purge/embargo + split audit | Implemented, audited | `audits.py` |
| Trade-management simulator (9 policies) reproducing labels (≥99%) | Implemented, tested | `test_stage5_simulator.py` |
| Deterministic risk engine (fixed fractional, daily/weekly/drawdown, aggregate risk, margin, volatility reduction; martingale/recovery keys forbidden) | Implemented, tested | `test_risk_engine.py` |
| Broker/Account/MT5 context manager (read-only, provenance, precedence, TTL, 10 tools) | Implemented, tested **with mocks only** | `test_broker_context.py` |
| Calibrated scenario probabilities (Beta shrinkage, blocked OOF, val bootstrap) | Implemented | `scenario_calibration_report.json` |
| Feature-quality / redundancy report, price-level diagnostic | Implemented | `out/08_stage5_reports/` |
| 9 task JSONL datasets + `direction_labels.jsonl` (LONG/SHORT/NO_TRADE preserved) + outcomes + manifest | Built | `out/07_task_datasets/` |
| Leakage / integrity audit (23 checks) | Previous 5.0.0 artifact: 22 pass, 1 fail; 5.1.0 rebuild must pass all | `leakage_and_integrity_audit.json` |
| Evaluation with baselines | Implemented | `task_evaluation_report.json` |
| Unsloth instructions | Written, **nothing executed** | `docs/UNSLOTH_QWEN3_5_4B_TRAINING.md` (model id unverified) |

Full test suite result above applies to the prior published build. The 5.1.0 code now requires the audit to pass every check and requires all eight management actions to have at least 30 naturally generated examples; rerun the full suite after rebuilding.

## 2. Partial / known shortfalls
- The published 5.0.0 artifact has a documented management-action coverage failure: `TAKE_PARTIAL_33` = 27 and `TRAIL_STOP_STRUCTURE` = 9 (minimum 30). The 5.1.0 generator now samples all simulator policies and records the relevant paths more densely. This is a **code-level correction, not yet a verified data-level pass**; rebuild and inspect the new audit counts before training.
- Management/review targets come from a causal rule-based playbook, not from hindsight-optimal or human-expert decisions. The dataset teaches the procedure, not proven edge.
- Dataset is a deterministic subsample of M15 bars (8,626 snapshots), not every bar.

## 3. Honest findings (no skill claims)
- Directional features show **no validated skill**: validation AUC ≈ 0.50–0.515 with a large train/val gap (overfit).
- The causal procedure's net R ≈ 0 with confidence intervals including 0.
- NO_TRADE AUC ≈ 0.61 on validation, but the context model **did not beat the base rate on the test split**.
- Hypothesis "price-level columns cause the overfit" was **tested and refuted** (identical ablation AUC).
- Results are in-sample-period, single-instrument, ~6 years of candles; statistical power is limited.

## 4. Test-split status
The test split (≥ 2025-07-01) is **no longer strictly untouched**: it was scored once for frozen baselines/probabilities. Nothing was tuned on it. The only truly sealed holdout is **data after 2026-11-10** (not in the dataset); reserve it for final evaluation.

## 5. Unavailable / not done
- External data sources (news, macro, COT, etc.): none ingested; `data_source_quality_report.json` documents this.
- Live MT5 integration: **not verified** against a real terminal. `live_integration_check.py` is provided for the user's Windows machine and was not run.
- Real broker specs/commission/contract details: never invented; cost scenarios are labeled assumptions.
- Timezone of server-clock timestamps unverified; tick volume only (`real_volume` is zero).
- Model training, evaluation of a trained model: out of scope.

## 6. Safety
No passwords/tokens/secrets in data (audited). No martingale/recovery language (audited). Stops only tighten in management targets (audited). Position risk never exceeds policy (audited). Live execution out of scope.

## 7. Reproduction
`bash run_stage5.sh` (see `docs/STAGE5_SYSTEM.md`). It now rebuilds labels/probabilities/task datasets, audits and tests, packages SFT train/validation, verifies gzip-part joins and runs structural preflight. Use tokenizer-aware preflight before training. Deterministic seeds are recorded in the manifests.
