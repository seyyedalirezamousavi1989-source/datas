# XAUUSD (Gold) AI Training Dataset - Stage 4 (technical + SMC + multi-timeframe features)

**Dataset version:** 1.0.0  |  **Generated (UTC):** 2026-11-11 04:09:06  |  **Instrument:** XAUUSD (Gold), broker feed: Alpari (per task description), MetaTrader 5 candles

## 1. Status in one paragraph
The feature dataset was built from the raw MT5 candles only and validated with 264 automated checks
(all passed). **No training labels (LONG/SHORT/NO TRADE) exist yet and no model was trained**; this is deliberate.
A separate labeling proposal is in `XAUUSD_Stage5_Labeling_Proposal.md`. The dataset is *not yet ready for model training* until the labeling stage and the open issues in section 12 are resolved.

## 2. Layout
```
source/XAUUSD.zip                  original archive, byte-identical copy (sha256 7458e5329bc827ff77aa9ec68d4be504451206226a74fd2cfa21f9e7f5516aa6)
scripts/                           reproducible pipeline (config.py, s01..s06, features_tech.py, features_smc.py, run_all.sh)
out/01_normalized/                 cleaned raw candles, 6 timeframes
out/02_technical/                  raw + time + technical/price-action features, 6 timeframes
out/03_smc/                        technical + SMC features (self-contained superset of 02), 6 timeframes
out/04_multitimeframe/             SMC dataset + closed higher-timeframe context, 5 lower timeframes (Daily is the top)
out/05_reports/                    data-quality, validation, gap, profile reports (JSON + CSV)
out/XAUUSD_Stage4_SMC_Manifest.json   machine-readable description of everything (columns, dtypes, definitions, results, limitations)
```
Each file is a plain CSV (comma separated, header row, `\n` line endings, floats rounded to 6 decimals, empty cell = undefined/NaN).
Rows of all stage files of one timeframe are identical in number and order (1:1 on `time`).

## 3. Source data (measured)
| timeframe | rows | first bar (open) | last bar (open) |
|---|---|---|---|
| M5 | 422,274 | 2020-11-11 01:05:00 | 2026-11-10 23:50:00 |
| M15 | 141,767 | 2020-11-11 01:00:00 | 2026-11-10 23:45:00 |
| M30 | 70,896 | 2020-11-11 01:00:00 | 2026-11-10 23:30:00 |
| H1 | 35,464 | 2020-11-11 01:00:00 | 2026-11-10 23:00:00 |
| H4 | 9,278 | 2020-11-11 00:00:00 | 2026-11-10 20:00:00 |
| Daily | 1,548 | 2020-11-11 00:00:00 | 2026-11-10 00:00:00 |

* Tab-separated MT5 export, columns `<DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>` (Daily has no `<TIME>`). Prices have 2 decimals.
* Timestamps are bar **open** times (confirmed: M15/M30/H1/H4/Daily bars equal the aggregation of M5 bars labelled by open time, 100% of covered bars).
* Weekends have no bars; there is a daily break; no candle is missing inside sessions (MT5 creates no bar without ticks).

## 4. Cleaning and validation (stage 1)
Normalised columns: `time, open, high, low, close, tick_volume, real_volume, spread, spread_known`. Original values were never altered; rows would only be dropped for exact duplicate timestamps or invalid OHLC (none occurred).

| TF | duplicates | invalid dropped | off-grid timestamps | spread==0 rows | gaps: weekend / overnight / multi-day |
|---|---|---|---|---|---|
| M5 | 0 | 0 | 0 | 7,561 (1.79%) | 300 / 1230 / 17 |
| M15 | 0 | 0 | 0 | 3,759 (2.65%) | 300 / 1230 / 17 |
| M30 | 0 | 0 | 0 | 2,285 (3.22%) | 300 / 1230 / 17 |
| H1 | 0 | 0 | 0 | 1,324 (3.73%) | 300 / 1230 / 17 |
| H4 | 0 | 0 | 0 | 425 (4.58%) | 300 / 7 / 17 |
| Daily | 0 | 0 | 0 | 85 (5.49%) | 300 / 0 / 17 |

* `real_volume` is 0 in every row: documented, **not** interpreted as traded volume, not used.
* `spread == 0` is treated as **unknown**, not as zero cost (`spread_known` flag). Spread is in MT5 points; point size is not stated in the files.
* Timeframe consistency (M5 aggregated vs. exported higher timeframes) matched on every covered bar: see `05_reports/XAUUSD_timeframe_consistency.csv`.

## 5. Timestamp / timezone status: UNVERIFIED
All timestamps are exactly as exported (naive broker-server clock). Empirical pattern: on 104 of 104 trading dates inside the weeks where US and EU daylight-saving differ, the last H1 bar of the day opens at 22:00 instead of 23:00,
and this never happens outside those weeks (0 of 1444). That is consistent with a server clock following EU DST (EET/EEST) but it is **an inference, not a verification**.
Therefore no UTC conversion was done and session features are fixed server-hour windows (assumption-based): asia (1, 10), london (10, 19), ny (15, 24), overlap (15, 19).
A measured server-hour volatility profile is in `05_reports/XAUUSD_H1_hourly_profile_server_time.csv`.

## 6. Technical and price-action features (110 columns incl. source columns in `02_technical`)
Candle body/range/wicks/direction/close position and ratios, gap to previous close, returns and log returns (1,3,5,10,20), True Range, Wilder ATR 14/50, ATR% and ATR ratio, realised volatility 20/50,
Wilder RSI 14, EMA 20/50/200 with ATR-normalised distances, spreads and slopes plus an EMA stack flag, MACD 12/26/9 (+ATR-normalised histogram), Bollinger 20/2 (population std),
previous 20/50/100-bar highs/lows (current bar excluded) with ATR distances, Donchian position and 20-bar breakout flags, tick-volume statistics (SMA, relative, z-score, change, ticks per range),
signed close streak, previous day/week levels, day/week open, running day high/low, distance to round numbers (50/100 USD), and time-of-day features.
**Every column has an exact definition in the manifest (`column_catalog`) - generated from the same code that computes it.**
Indicators with `min_periods` are NaN while warming up (first 199 rows for EMA200).

## 7. SMC rules (all parameters in `scripts/config.py`; echoed in the manifest)
| concept | rule (summary) |
|---|---|
| Swing high/low | bar i is a swing high if high[i] > max(high[i-5..i-1]) and high[i] >= max(high[i+1..i+5]); mirror for lows. Published only at bar i+5. |
| BOS | close beyond the last confirmed, not-yet-broken swing level in the direction of the current state (state 0 -> first break is a BOS). Each level breaks once. |
| CHOCH | close beyond the last confirmed swing level **against** the current state (bullish CHOCH when state is bearish and vice versa). |
| Liquidity sweep | wick beyond an active confirmed swing pool (<= 500 bars old, not yet exceeded) with the close back on the original side; the pool is consumed on first excursion. If the close is beyond the level it is a break, not a sweep. |
| Equal highs / lows | a new confirmed swing within 0.1 x ATR14 of an active unswept swing of the same side. |
| FVG | 3-candle gap (low[t] > high[t-2] bullish; high[t] < low[t-2] bearish), size >= 0.1 x ATR14. Touch = later bar overlaps the zone; invalidated = later **close** beyond the far edge; expires after 300 bars; max 15 zones per side. |
| Order Block (heuristic) | at a BOS/CHOCH close: last opposite-colour candle within 15 bars, only if a same-direction displacement candle lies in the leg; zone = that candle's range. Touch/invalidation/expiry as FVG; max 10 per side. |
| Premium / discount | position of close inside [last confirmed swing low, last confirmed swing high]; premium > 0.55, discount < 0.45, equilibrium band between; undefined if the range is invalid. |
| Displacement | range >= 1.5 x ATR14 of the previous bar and body/range >= 0.6. |

Order Blocks are heuristic interpretations, not objectively verified facts; a frequent pattern is **not** assumed to be profitable.
Measured event counts (from the files):

| event | M5 | M15 | M30 | H1 | H4 | Daily |
|---|---|---|---|---|---|---|
| swing_high_confirmed | 25,474 | 8,626 | 4,284 | 2,175 | 569 | 82 |
| swing_low_confirmed | 25,655 | 8,632 | 4,341 | 2,227 | 584 | 86 |
| bos_bull | 5,127 | 1,783 | 928 | 466 | 134 | 28 |
| bos_bear | 4,572 | 1,523 | 733 | 363 | 108 | 14 |
| choch_bull | 5,085 | 1,741 | 848 | 398 | 108 | 11 |
| choch_bear | 5,086 | 1,740 | 848 | 399 | 108 | 12 |
| bsl_sweep | 10,251 | 3,487 | 1,704 | 828 | 204 | 26 |
| ssl_sweep | 10,384 | 3,436 | 1,770 | 880 | 216 | 23 |
| eqh_formed | 1,244 | 374 | 171 | 86 | 22 | 3 |
| eql_formed | 1,219 | 372 | 182 | 83 | 17 | 5 |
| fvg_bull_formed | 35,550 | 11,697 | 5,834 | 2,939 | 871 | 186 |
| fvg_bear_formed | 33,160 | 10,485 | 5,088 | 2,466 | 693 | 110 |
| ob_bull_formed | 3,813 | 1,463 | 789 | 427 | 121 | 16 |
| ob_bear_formed | 3,900 | 1,529 | 756 | 389 | 119 | 14 |
| displacement_candles | 29,154 | 10,434 | 5,146 | 2,660 | 742 | 119 |

## 8. Gold-specific considerations
Gold shows sudden range expansion, regime changes in volatility, strong reactions at round numbers and prior day/week extremes, frequent sweeps/false breaks and news-driven spikes.
The dataset therefore (a) normalises distances by ATR instead of fixed price/pip units (no Forex pip assumptions were reused), (b) provides volatility-regime features (ATR ratio, realised-vol ratio, displacement strength),
(c) provides psychological-level (50/100 USD) distances, (d) provides sweep/equal-level features and (e) keeps server-hour features so session dependence can be learned or examined.
**No news or economic-calendar data is included and none was invented.**

## 9. Multi-timeframe alignment
Lower-timeframe rows are known at `open + bar length`. A higher-timeframe candle is attached to a lower-timeframe row only if its nominal close (`open + its length`) is <= that instant
(`merge_asof`, backward, exact matches allowed because both candles are complete at that instant). Unfinished H1/H4/Daily candles are never exposed. Columns added per higher timeframe
(prefix `m15_ m30_ h1_ h4_ d1_`): `bar_time`, `age_min` and 16 context columns (close, direction, atr_14, rsi_14, dist_ema_50_atr, ema_stack, macd_hist_atr, trend_state, struct_last_event, pd_pos, last_swing_high, last_swing_low, fvg_bull_dist_atr, fvg_bear_dist_atr, ob_bull_dist_atr, ob_bear_dist_atr). Rows before the first closed higher-timeframe candle have empty values.
Independent brute-force lookups on 3,000 sampled rows per (lower, higher) pair and a full-file inequality check passed.

## 10. Files
| file | rows | columns | size |
|---|---|---|---|
| `01_normalized/XAUUSD_M5_normalized.csv` | 422,274 | 9 | 26.3 MB |
| `01_normalized/XAUUSD_M15_normalized.csv` | 141,767 | 9 | 8.9 MB |
| `01_normalized/XAUUSD_M30_normalized.csv` | 70,896 | 9 | 4.5 MB |
| `01_normalized/XAUUSD_H1_normalized.csv` | 35,464 | 9 | 2.2 MB |
| `01_normalized/XAUUSD_H4_normalized.csv` | 9,278 | 9 | 0.6 MB |
| `01_normalized/XAUUSD_Daily_normalized.csv` | 1,548 | 9 | 0.1 MB |
| `02_technical/XAUUSD_M5_technical.csv` | 422,274 | 110 | 370.9 MB |
| `02_technical/XAUUSD_M15_technical.csv` | 141,767 | 110 | 124.5 MB |
| `02_technical/XAUUSD_M30_technical.csv` | 70,896 | 110 | 62.3 MB |
| `02_technical/XAUUSD_H1_technical.csv` | 35,464 | 110 | 31.1 MB |
| `02_technical/XAUUSD_H4_technical.csv` | 9,278 | 110 | 8.1 MB |
| `02_technical/XAUUSD_Daily_technical.csv` | 1,548 | 110 | 1.3 MB |
| `03_smc/XAUUSD_M5_smc.csv` | 422,274 | 188 | 490.6 MB |
| `03_smc/XAUUSD_M15_smc.csv` | 141,767 | 188 | 165.1 MB |
| `03_smc/XAUUSD_M30_smc.csv` | 70,896 | 188 | 82.5 MB |
| `03_smc/XAUUSD_H1_smc.csv` | 35,464 | 188 | 41.3 MB |
| `03_smc/XAUUSD_H4_smc.csv` | 9,278 | 188 | 10.8 MB |
| `03_smc/XAUUSD_Daily_smc.csv` | 1,548 | 188 | 1.8 MB |
| `04_multitimeframe/XAUUSD_M5_mtf.csv` | 422,274 | 278 | 782.5 MB |
| `04_multitimeframe/XAUUSD_M15_mtf.csv` | 141,767 | 260 | 243.6 MB |
| `04_multitimeframe/XAUUSD_M30_mtf.csv` | 70,896 | 242 | 112.0 MB |
| `04_multitimeframe/XAUUSD_H1_mtf.csv` | 35,464 | 224 | 51.1 MB |
| `04_multitimeframe/XAUUSD_H4_mtf.csv` | 9,278 | 206 | 12.1 MB |

ZIP packages (`packages/`):

| package | size |
|---|---|
| `pkg_01_normalized_all.zip` | 11.0 MB |
| `pkg_02_technical_H1_H4_Daily.zip` | 14.4 MB |
| `pkg_02_technical_M15_M30.zip` | 65.9 MB |
| `pkg_02_technical_M5.zip` | 129.7 MB |
| `pkg_03_smc_H1_H4_Daily.zip` | 17.5 MB |
| `pkg_03_smc_M15_M30.zip` | 79.9 MB |
| `pkg_03_smc_M5.zip` | 157.3 MB |
| `pkg_04_mtf_H1_H4_Daily.zip` | 18.0 MB |
| `pkg_04_mtf_M15_M30.zip` | 91.4 MB |
| `pkg_04_mtf_M5.zip` | 179.9 MB |

## 11. Reproduce
```
python3 scripts/s01_clean_validate.py          # reads source/XAUUSD.zip read-only
python3 scripts/s02_build_features.py          # technical + SMC (all timeframes, or pass names e.g. H1 M15)
python3 scripts/s03_mtf_join.py                # multi-timeframe join
python3 scripts/s04_validate_outputs.py        # 100% of the checks quoted here
```
Needs only Python 3, pandas and numpy (tested with pandas 3.0.5, numpy 2.5.3); no network, no compiled extras. The SMC pass is a plain Python loop (about 2 minutes for M5).

## 12. Known limitations / open issues
* Raw price-level columns (open/high/low/close, EMAs, Bollinger bands, previous highs/lows, levels, SMC zone prices) are non-stationary (gold traded from ~1,615 to ~5,598 USD in the file). Prefer the ATR-normalised/relative columns for modelling.
* Order Blocks are a heuristic interpretation; zones are capped in number and expire after a fixed number of bars, so very old zones are forgotten by design.
* Swings need 5 closed bars on each side, so structure information lags price by 5 bars.
* Tick volume is broker-specific and not comparable with exchange volume.
* Spread==0 rows are unknown; zero-spread share is not uniform across years (see reports).
* Only one broker feed; no tick data; no order-book data.
* Daily candles and day/week levels use the server calendar date, which in US/EU DST-mismatch weeks is shifted by one hour relative to the New York-based market break.
* The data starts on 2020-11-11, so there is no pre-history for indicator warm-up.
* Exact historical broker-server timezone and DST rules (empirical pattern is consistent with EET/EEST but unverified).
* Meaning of the MT5 spread field per bar (open/last/max) and the point size.
* Commission and slippage for the intended trading account are unknown.
* No external price source was available, so absolute price accuracy cannot be cross-checked (only internal consistency).
* No economic-calendar/news data is included; none was invented.

## 13. How to use the features safely
* Row `time` is the **open** time; features are final at `bar_close_time`. Build labels from the *next* bar onward.
* Drop at least the first 500 rows of every timeframe (warm-up of EMA200 and the SMC state).
* Do not feed raw price levels directly to a model; use the ATR-normalised and relative columns.
* Treat `spread == 0` as unknown and never as free execution.
* Split chronologically with purging/embargo (see the labeling proposal).

## 14. Not done (by design)
No LONG/SHORT/NO TRADE labels, no backtest, no performance claim, no model training.
