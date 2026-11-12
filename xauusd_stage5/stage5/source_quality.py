"""External data-source quality evaluation framework + the recorded decision for this build.

No external dataset was supplied and this environment has no general internet access, so NO external source could be
verified.  Rather than fabricating or silently omitting, every candidate is recorded with the checks it must pass and is NOT integrated.
``evaluate_source`` runs the mandatory checks on a user-supplied CSV so the same evaluation can be repeated when data exists."""
from __future__ import annotations

import json
import os
from typing import Any, Dict

import numpy as np
import pandas as pd

from . import s5config as S

CANDIDATES = {
    "economic_calendar_actuals": dict(purpose="event risk windows (NFP, CPI, FOMC) and surprise = actual - forecast",
        needs=["event timestamp in UTC with documented timezone", "actual AND consensus forecast AND first-release (not revised) values", "release-time availability (point-in-time)", "licence permitting model training"]),
    "us_dollar_index_dxy": dict(purpose="gold/USD inverse relationship regime context",
        needs=["intraday history aligned to broker server clock (timezone offset documented)", "same bar-close availability at inference", "licence; note DXY is itself ~58% EURUSD (check redundancy with FX data)"]),
    "us_treasury_yields_10y_real": dict(purpose="real-rate driver of gold",
        needs=["intraday or at least daily point-in-time series (FRED daily is end-of-day: only usable lagged)", "release timing", "licence"]),
    "silver_xagusd": dict(purpose="metals co-movement / confirmation",
        needs=["same-broker XAGUSD candles preferred (same clock, same costs, available at inference through MT5)", "measure incremental value vs XAUUSD own features"]),
    "tick_volume_existing": dict(purpose="already in the data (tick_volume); evaluated in the feature report", status="USED_AS_IS",
        notes="broker tick count, NOT centralized exchange volume; real_volume is all zero"),
}
PROTOCOL = ["historical availability over the full 2020-11..2026-11 span", "timestamp + timezone reliability (DST handling, release-time vs bar-close)", "missingness and revision policy (point-in-time)",
            "availability at actual inference time (latency, API access, licence)", "incremental value: train-fit HistGradientBoosting ablation, validation-confirmed (see feature_report.py)", "redundancy vs existing features (correlation + ablation)"]


def evaluate_source(df: pd.DataFrame, time_col="time", tz_note=None, max_lag_minutes=None) -> Dict[str, Any]:
    """Mandatory structural checks for a candidate external series (value columns = all numeric columns)."""
    t = pd.to_datetime(df[time_col], errors="coerce", utc=False)
    out: Dict[str, Any] = {"rows": int(len(df)), "unparseable_timestamps": int(t.isna().sum()), "monotonic": bool(t.dropna().is_monotonic_increasing),
                           "duplicate_timestamps": int(t.duplicated().sum()), "timezone_documented": bool(tz_note), "timezone_note": tz_note}
    vals = df.select_dtypes("number")
    out["missing_share_by_column"] = vals.isna().mean().round(4).to_dict()
    gaps = t.dropna().diff().dt.total_seconds().div(60)
    out["median_gap_minutes"] = None if gaps.empty else float(gaps.median())
    out["span"] = [str(t.min()), str(t.max())]
    out["covers_project_span"] = bool(t.min() <= pd.Timestamp("2020-11-11") and t.max() >= pd.Timestamp("2026-11-09"))
    out["max_publication_lag_minutes_declared"] = max_lag_minutes
    problems = [k for k, bad in (("unparseable timestamps", out["unparseable_timestamps"] > 0), ("not monotonic", not out["monotonic"]), ("duplicate timestamps", out["duplicate_timestamps"] > 0),
                                 ("timezone not documented", not out["timezone_documented"]), ("does not cover project span", not out["covers_project_span"]),
                                 ("publication lag unknown (point-in-time availability unverifiable)", max_lag_minutes is None)) if bad]
    out["blocking_problems"] = problems
    out["verdict"] = "REJECT_UNTIL_FIXED" if problems else "ELIGIBLE_FOR_INCREMENTAL_VALUE_TEST"
    return out


def run():
    rep = {"summary": "No external source was integrated. None could be verified in this environment (no supplied data; no general internet access). Nothing was fabricated.",
           "protocol": PROTOCOL, "candidates": {k: dict(v, decision=v.get("status", "NOT_INTEGRATED_UNVERIFIED"), incremental_value="unmeasured") for k, v in CANDIDATES.items()},
           "how_to_add": "Provide a CSV (time, value columns) and run stage5.source_quality.evaluate_source(df, tz_note=..., max_lag_minutes=...); then add features through the existing feature pipeline and rerun feature_report.py (train selection, validation confirmation).",
           "news_tasks": "News/event-based tasks are excluded: no verified historical event data exists in the project; models must not invent news."}
    os.makedirs(S.DIR_REPORTS5, exist_ok=True)
    json.dump(rep, open(os.path.join(S.DIR_REPORTS5, "data_source_quality_report.json"), "w"), indent=1)
    return rep


if __name__ == "__main__":
    run(); print("written")
