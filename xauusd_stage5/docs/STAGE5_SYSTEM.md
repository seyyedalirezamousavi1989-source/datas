# Stage 5 — XAUUSD Trading Intelligence dataset system (architecture, decisions, reproduction)

Governing scope text: "Final Scope Extension: Complete XAUUSD Trading Intelligence System" (the user stated it supersedes the earlier "Critical Extension").
It was integrated into the existing implementation (Stage 4 features → broker_context → stage5), with no second pipeline.

## Reconciliation with the implementation state (decisions)
| Item | State found | Decision |
|---|---|---|
| LONG/SHORT/NO_TRADE labeling | only a *proposal* (Stage 5 labeling proposal) existed | implemented in `stage5/labeling.py` exactly per the proposal (triple barrier, next-open entry, M5 ambiguity replay, worst-case policy) |
| Broker/Account/MT5 context, cost engine | implemented (`broker_context/`) | reused by labeling (CostProfile), risk engine and datasets; no second cost engine |
| Position sizing / margin / portfolio limits | missing | added to `broker_context/risk_engine.py` + 2 tools (`calculate_position_size`, `validate_portfolio_risk`); account state is caller-supplied (MT5 `account_info` balance/equity/margin readable via `fetch_account_state`; open positions are NOT read) |
| Trade simulator | missing | `stage5/simulator.py` (M5 replay, 9 policies, market/limit/stop entries) |
| External sources (calendar, DXY, yields, silver) | none available, no network | evaluation framework + recorded decision "not integrated"; nothing fabricated; news tasks excluded |
| Historical vs runtime assumptions | — | historical cost scenarios A/B/C are **unverified assumptions** in `s5config.py`; runtime context never rewrites labels |
| Direction supervision | hindsight triple-barrier labels | kept in `direction_labels.jsonl`; planning/integrated targets use a **causal procedure** (`analysis.py`) so no target is a function of the future |

## Modules
`stage5/s5config.py` (all parameters, versions) · `labeling.py` · `splits.py` · `simulator.py` · `probabilities.py` · `feature_report.py` · `source_quality.py` ·
`analysis.py` (causal analysis, setup qualification, plans, management playbook) · `build_datasets.py` · `render_sft.py` · `package_sft.py` · `preflight_sft.py` · `audits.py` · `evaluate.py`.

## Reproduce (from the project root, after Stage 4 outputs exist)
```
python3 -m stage5.labeling            # out/06_labels/labels_label-v1.csv  (+ immutable run record in out/06_labels/runs)
python3 -m stage5.probabilities       # calibrated probabilities + scenario_calibration_report.json
python3 -m stage5.feature_report      # feature_quality_report.csv / summary (train selection, val confirmation, test never read)
python3 -m stage5.evaluate            # baselines + procedure + management comparison (rules fixed beforehand)
python3 -m stage5.build_datasets      # out/07_task_datasets/*.jsonl  (deterministic, seed 2026)
python3 -m stage5.audits              # leakage + integrity + split audit -> out/08_stage5_reports/leakage_and_integrity_audit.json
python3 -m unittest discover -s tests
```
Labels are never overwritten: `LabelRunRegistry` refuses to rewrite a run directory (`out/06_labels/runs/<run_id>`).

## Task datasets (see `out/07_task_datasets/manifest.json` for counts and hashes)
`snapshots.jsonl` holds one causal snapshot per decision; task files hold only `input` + `target` and reference `snapshot_id` (no duplication).
Hindsight facts live in `outcomes/` and are never rendered. `render_sft.py` expands records into chat/tool-call SFT examples.

| task | target is produced by |
|---|---|
| market_analysis | causal rules on the snapshot |
| scenario_forecasting | train-fit empirical probabilities (OOF for train rows), evidence, confirmation/invalidation levels |
| entry_exit_planning | causal setup qualification + plan (Market/Limit/Stop, ATR stop, TP1/TP2, cancel rules) or NO_TRADE with reasons |
| risk_management / position_sizing / portfolio_risk | real calls to the deterministic tools in a fixed-clock training environment (known / unknown commission / unknown contract); final text copies tool numbers |
| trade_management | state-only playbook on causal states from all nine independently simulated policy paths; denser M5 sampling around multi-target and trailing transitions |
| post_trade_review | rule-based review of completed simulated trades with what-if net R per policy |
| integrated_trading_tasks | consistent chain: analysis → probabilities → decision → plan → sizing tool |

## Honest status of the data
Directional probabilities equal base rates (no validated skill); the rule-based procedure shows net R ≈ 0 with confidence intervals containing 0.
The datasets teach an analytical *procedure*, tool use and risk discipline — not a demonstrated edge.
\n## Stage 5.1 management coverage and SFT packaging\nThe 5.1.0 generator samples at most one naturally observed state per available playbook action per simulated entry, across all simulator policy paths. It does not synthesize labels or duplicate rows to satisfy coverage thresholds. The audit and dataset tests require at least 30 examples for each supported action. `package_sft.py` deterministically combines the newly rendered Forex task examples with the AURORA identity train/validation examples, excludes test data, writes gzip parts and records source-manifest hashes. `preflight_sft.py` rejects stale packages, verifies hashes/row counts/schema/tool-call linkage and can measure exact tokenizer lengths. A successful structural preflight without a tokenizer does not certify sequence-length safety; run it again with the exact local tokenizer before training.\n