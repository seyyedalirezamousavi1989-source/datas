# XAUUSD Stage 5 - Labeling Methodology PROPOSAL (not implemented)

Status: **proposal only.** No labels were created, no label statistics, no backtest, no profitability claim. Everything marked *parameter* is a starting point to be reviewed and fixed **before** any label is generated,
and tuned (if at all) on the validation period only.

## 1. Principles
1. A label for bar *t* may use only information up to the close of *t* for the decision, and prices from *t+1* onward for the outcome.
2. The outcome is simulated with explicit barriers and **explicit costs**; zero cost is never assumed.
3. Ambiguity is flagged, not hidden.
4. Train/validation/test are chronological, with purging and embargo.
5. The test period is touched once, at the very end.

## 2. Decision and entry
* Decision at the close of bar *t* (features in the `*_smc` / `*_mtf` rows).
* Market entry at the **open of bar t+1**. Long fills at ask = open + spread, short fills at bid = open (assuming the candles are BID candles - unverified, see manifest).
* Gap handling: if the entry open is already beyond a barrier (weekend / break gap), the trade exits at that open price plus slippage; this is a realistic loss, not an error.

## 3. Barriers (triple-barrier style, per direction)
* Stop distance `SL = k_sl x ATR14[t]` (ATR frozen at the decision bar, never re-read later). *Parameter:* k_sl in {1.0, 1.5, 2.0}, headline 1.5.
  Rationale: displacement candles in this dataset are defined as >= 1.5 x ATR of the previous bar, so a stop tighter than ~1 ATR is inside ordinary Gold noise.
* Floor: `SL >= m x (spread + slippage)` with *parameter* m = 5, so cost never dominates the trade.
* Target `TP = RR x SL`; *parameter* RR in {1.0, 1.5, 2.0}.
* Vertical barrier (max holding) `H` bars. *Parameter (starting point):* M5 48 (4 h), M15 32 (8 h), M30 24 (12 h), H1 24 (1 day), H4 12 (2 days), Daily 10 (2 weeks).
* Horizons that cross a weekend or a multi-day holiday gap: primary rule = **exclude the sample** (label undefined), alternative = keep and flag `crosses_gap`.

## 4. Costs (all explicit, all scenario-based)
* **Spread:** use the bar's spread only when `spread_known == 1`; otherwise fall back to the median known spread of the previous 500 bars of the same timeframe and set `spread_imputed = 1`.
  Spread is in MT5 points; converting to USD needs the point size (assumed 0.01 USD, **unverified** - confirm with the broker contract specification).
  Measured known-spread medians change a lot by year (table below), so a single constant spread would be wrong.
* **Commission:** not in the data and account-dependent. Define scenarios (none / low / base / high) and report results for each; fix the headline value only after the real account terms are known.
* **Slippage:** model as a fraction of ATR14 (scenarios, e.g. 0.00 / 0.02 / 0.05 x ATR) and widen it for bars flagged as displacement or high `atr_ratio`.
* Costs are charged on entry and exit; net R = (gross outcome - costs) / SL.

Measured volatility and spread context (from the generated technical files):

| TF | year | median ATR14 (USD) | median known spread (points) | spread==0 share |
|---|---|---|---|---|
| M15 | 2020 | 2.32 | 5 | 0.0% |
| M15 | 2021 | 1.93 | 6 | 0.0% |
| M15 | 2022 | 2.13 | 8 | 15.9% |
| M15 | 2023 | 1.92 | 17 | 0.0% |
| M15 | 2024 | 2.81 | 15 | 0.0% |
| M15 | 2025 | 4.99 | 12 | 0.0% |
| M15 | 2026 | 9.06 | 11 | 0.0% |
| H1 | 2020 | 5.06 | 5 | 0.0% |
| H1 | 2021 | 4.13 | 5 | 0.0% |
| H1 | 2022 | 4.52 | 8 | 22.3% |
| H1 | 2023 | 4.15 | 16 | 0.1% |
| H1 | 2024 | 5.96 | 14 | 0.0% |
| H1 | 2025 | 10.14 | 12 | 0.0% |
| H1 | 2026 | 18.84 | 11 | 0.0% |
| H4 | 2020 | 10.22 | 5 | 0.0% |
| H4 | 2021 | 8.50 | 5 | 0.0% |
| H4 | 2022 | 9.28 | 8 | 27.0% |
| H4 | 2023 | 8.58 | 16 | 0.5% |
| H4 | 2024 | 12.16 | 14 | 0.0% |
| H4 | 2025 | 20.21 | 11 | 0.0% |
| H4 | 2026 | 38.75 | 10 | 0.0% |
| Daily | 2020 | 25.87 | 5 | 0.0% |
| Daily | 2021 | 23.33 | 5 | 0.0% |
| Daily | 2022 | 24.74 | 6 | 30.6% |
| Daily | 2023 | 23.55 | 15 | 2.3% |
| Daily | 2024 | 33.21 | 12 | 0.0% |
| Daily | 2025 | 52.05 | 9 | 0.0% |
| Daily | 2026 | 106.01 | 9 | 0.0% |

(full table: `05_reports/XAUUSD_atr_spread_by_year_and_timeframe.csv`)

## 5. Label definition
For each bar and each direction the path is simulated until a barrier is hit:
* TP first -> win (net R > 0 after costs);  SL first -> loss;  neither by H -> timeout with mark-to-market net R.
* `LONG` if the long outcome is a net win and the short outcome is not;  `SHORT` symmetrically;  `NO_TRADE` otherwise (both lose, timeout, two-sided whipsaw, ambiguous, or net R below a minimum edge threshold `theta` - *parameter*, default 0).
* Keep auxiliary columns: net R long/short, bars to exit, exit reason, `ambiguous`, `spread_imputed`, `crosses_gap`.

## 6. Ambiguous candles
When one candle touches both SL and TP:
1. replay the same interval on the lower timeframe (M5 for M15 and above; the M5 bar is the finest available);
2. if still ambiguous (both inside one M5 bar) assume the **worst case (SL first)** and set `ambiguous = 1`;
3. report how many samples are ambiguous per timeframe and test the model with and without them.

## 7. Chronological splits, purging, embargo
Data spans 2020-11-11 to 2026-11-10. Proposed split (by calendar, identical dates for all timeframes):
* Train: 2020-11-11 - 2024-06-30;  Validation: 2024-07-01 - 2025-06-30;  Test: 2025-07-01 - 2026-11-10.
* **Purge:** drop every training sample whose outcome window [t+1, t+H] reaches into the validation (or test) period.
* **Embargo:** additionally drop `E = max(H, ceil(0.01 x N_train))` bars after each boundary when a later segment is used for fitting (cross-validation folds).
* Gold changed regime (price roughly doubled+, strong trend phases): evaluate by sub-period and prefer walk-forward / rolling-origin validation over a single split.
* Normalisation statistics (if any) must be fitted on the training segment only.

## 8. Overlapping samples
Consecutive labels share most of their outcome window. Use average-uniqueness sample weights (or a stride of about H/2) and never shuffle across time.

## 9. Evaluation (when a model exists)
Report net R expectancy per trade, hit rate against the break-even rate implied by RR and costs, number of trades, drawdown of the simulated R stream, and results per year, per session window and per volatility regime.
Always compare with the trivial baselines (always NO_TRADE, random direction, simple trend rule) under the same cost model. Accuracy alone is not a valid objective.

## 10. Decisions needed from you before implementation
1. Primary decision timeframe(s) (e.g. M15 or H1) and which higher timeframes are context only.
2. Broker contract specification: point size, commission, typical slippage, whether candles are bid.
3. Horizon set and RR/SL grid above, and the weekend/holiday-gap rule.
4. Whether ambiguous samples should be dropped or kept with worst-case labels.
5. Final split dates.

## 11. Deliverables of the labeling stage (planned)
`XAUUSD_<TF>_labels.csv` keyed by `time` (+ auxiliary columns), `XAUUSD_Stage5_Labels_Manifest.json`, a validation report (label counts, ambiguity share, leakage checks, purge/embargo audit) and the reproducible script.
