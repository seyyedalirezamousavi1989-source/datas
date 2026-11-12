"""Stage 5 configuration (single source of truth; echoed into every report/run record)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out")
M15_MTF = os.path.join(OUT, "04_multitimeframe", "XAUUSD_M15_mtf.csv")
M5_NORM = os.path.join(OUT, "01_normalized", "XAUUSD_M5_normalized.csv")
DIR_LABELS = os.path.join(OUT, "06_labels")
DIR_DATASETS = os.path.join(OUT, "07_task_datasets")
DIR_REPORTS5 = os.path.join(OUT, "08_stage5_reports")
RUNS_DIR = os.path.join(OUT, "06_labels", "runs")

STAGE5_VERSION = "5.0.0"
DATASET_VERSION = "1.0.0"
LABEL_CONFIG_VERSION = "label-v1"

DECISION_TF_MIN = 15
LABEL = dict(sl_atr_mult=1.5, rr=2.0, horizon_bars=32, min_net_r=0.0, max_spread_atr=0.35, warmup_rows=500)
SCENARIO = dict(move_atr=1.5, zone_levels=("prev_day_high", "prev_day_low"))
MGMT_HORIZON_M5 = 288       # 24 h of M5 bars hard cap for management replay

# chronological split (decision time = bar close of the decision candle)
SPLIT = dict(train_end="2024-06-30 23:59:59", val_start="2024-07-01 00:00:00", val_end="2025-06-30 23:59:59",
             test_start="2025-07-01 00:00:00", purge_bars=96, embargo_bars=96)

# ---- HISTORICAL cost scenarios: explicit UNVERIFIED assumptions (never claimed to be the user's account)
_CONTRACT = {"point": 0.01, "contract_size": 100.0, "account_currency": "USD", "profit_currency": "USD",
             "verified": False, "note": "XAUUSD 100 oz/lot, 2 digits assumed for historical labeling; unverified"}
_BROKER = {"broker": "Alpari (data source only)", "account_label": "unspecified", "verified": False}
_SPREAD = {"source": "bar_spread", "fallback_points": 12}
COST_SCENARIOS = {
    "A_spread_only": dict(commission={"model": "none"}, slippage={"model": "fixed_price", "value": 0.0}),
    "B_moderate_assumed": dict(commission={"amount": 3.5, "currency": "USD", "basis": "per_lot_per_side"},
                               slippage={"model": "fixed_price", "value": 0.05}),
    "C_stress_assumed": dict(commission={"amount": 7.0, "currency": "USD", "basis": "per_lot_per_side"},
                             slippage={"model": "fixed_price", "value": 0.15}),
}
PRIMARY_SCENARIO = "B_moderate_assumed"


def make_profile(name):
    from broker_context.cost_profiles import CostProfile
    sc = COST_SCENARIOS[name]
    return CostProfile(name=name, config_version=LABEL_CONFIG_VERSION, broker_assumptions=_BROKER, contract_assumptions=_CONTRACT,
                       spread=_SPREAD, commission=sc["commission"], slippage=sc["slippage"])


SNAPSHOT_COLS = [  # the ONLY columns that may appear in model-visible inputs (causal, known at bar close)
    "time", "bar_close_time", "close", "atr_14", "atr_ratio", "rv_ratio", "spread", "spread_known", "server_hour", "dow",
    "session_asia", "session_london", "session_ny", "session_london_ny_overlap", "ret_3", "ret_10", "ret_20", "rsi_14",
    "dist_ema_20_atr", "dist_ema_50_atr", "dist_ema_200_atr", "ema_stack", "macd_hist_atr", "bb_pctb", "bb_width", "trend_state",
    "struct_last_event", "bars_since_struct_event", "bos_bull", "bos_bear", "choch_bull", "choch_bear", "bsl_sweep", "ssl_sweep",
    "disp_dir", "disp_strength_atr", "pd_pos", "pd_zone", "last_swing_high", "last_swing_low", "dist_last_swing_high_atr",
    "dist_last_swing_low_atr", "dist_prev_day_high_atr", "dist_prev_day_low_atr", "dist_prev_week_high_atr", "dist_prev_week_low_atr",
    "dist_day_open_atr", "dist_round_50_atr", "donchian_pos_20", "brk_high_20", "brk_low_20",
    "fvg_bull_in_zone", "fvg_bull_dist_atr", "fvg_bear_in_zone", "fvg_bear_dist_atr", "ob_bull_in_zone", "ob_bull_dist_atr",
    "ob_bear_in_zone", "ob_bear_dist_atr", "bsl_n_active", "ssl_n_active", "tv_rel_20",
    "prev_day_high", "prev_day_low",
] + [f"{p}_{c}" for p in ("m30", "h1", "h4", "d1") for c in ("bar_time", "trend_state", "struct_last_event", "ema_stack", "pd_pos", "rsi_14", "dist_ema_50_atr")]
