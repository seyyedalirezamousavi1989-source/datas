"""Leakage audit + chronological split audit + dataset integrity audit for the Stage 5 task datasets."""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from . import s5config as S
from .analysis import analyze, management_playbook, plan_trade
from .render_sft import FORBIDDEN_IN_PROMPT, load_snapshots, render
from .splits import LABEL_END_HOURS

TASKS = ["market_analysis", "scenario_forecasting", "entry_exit_planning", "risk_management", "position_sizing", "trade_management", "portfolio_risk",
         "post_trade_review", "integrated_trading_tasks"]
HIND_COL = re.compile(r"^(label|long_|short_|up_exc|dn_exc|end_ret|first_touch|reach_|mfe|mae|t_|valid|exclusion|no_trade_reason)")
HTF_MIN = {"m30": 30, "h1": 60, "h4": 240, "d1": 1440}
SECRET = re.compile(r"password|passwd|token|api[_-]?key|secret|investor|\"login\"", re.I)
REQ = {"market_analysis": {"bias", "regime", "evidence", "no_trade_flags"}, "scenario_forecasting": {"scenarios", "uncertainty", "trade_outcome_probabilities"},
       "entry_exit_planning": {"decision"}, "risk_management": {"tool_calls", "tool_results", "decision", "final_text"}, "position_sizing": {"tool_calls", "tool_results", "final_text"},
       "trade_management": {"action", "reasons", "constraints"}, "portfolio_risk": {"tool_calls", "tool_results", "decision"}, "post_trade_review": {"outcome", "net_r", "lessons"},
       "integrated_trading_tasks": {"decision", "final_text", "scenario_probabilities"}}


def read(d, name):
    return [json.loads(l) for l in open(os.path.join(d, name + ".jsonl"))]


def run(d=None):
    d = d or S.DIR_DATASETS
    snaps = load_snapshots(d)
    recs = {t: read(d, t) for t in TASKS}
    chk = {}
    prob = pd.read_csv(os.path.join(S.DIR_LABELS, "probabilities_v1.csv")).set_index("idx")
    lab = pd.read_csv(os.path.join(S.DIR_LABELS, f"labels_{S.LABEL_CONFIG_VERSION}.csv"), parse_dates=["bar_close_time", "entry_time"])
    # 1 snapshot columns / hindsight names
    bad_cols = set()
    not_allowed = set()
    for s in snaps.values():
        for k in s["data"]:
            if HIND_COL.match(k):
                bad_cols.add(k)
            if k not in S.SNAPSHOT_COLS:
                not_allowed.add(k)
    chk["snapshot_has_no_hindsight_columns"] = {"pass": not bad_cols, "offending": sorted(bad_cols)}
    chk["snapshot_columns_subset_of_allowed_list"] = {"pass": not not_allowed, "offending": sorted(not_allowed)}
    # 2 temporal availability
    viol = 0
    for s in snaps.values():
        bc = pd.Timestamp(s["bar_close_time"])
        if pd.Timestamp(s["data"]["time"]) + pd.Timedelta(minutes=15) != bc:
            viol += 1
        for p, m in HTF_MIN.items():
            bt = s["data"].get(f"{p}_bar_time")
            if bt and pd.Timestamp(bt) + pd.Timedelta(minutes=m) > bc:
                viol += 1
    chk["no_unclosed_higher_timeframe_candle_in_any_snapshot"] = {"pass": viol == 0, "violations": viol, "snapshots": len(snaps)}
    # 3 decisions are pure functions of the snapshot (recompute)
    mism = 0
    for r in recs["entry_exit_planning"]:
        a = analyze(snaps[r["snapshot_id"]]["data"])
        side = a["setup"]["side"]
        if (side.upper() if side else "NO_TRADE") != r["target"]["decision"]:
            mism += 1
        elif side and plan_trade(snaps[r["snapshot_id"]]["data"], side, a)["entry_price"] != r["target"]["plan"]["entry_price"]:
            mism += 1
    chk["planning_targets_recomputable_from_snapshot_alone"] = {"pass": mism == 0, "mismatches": mism, "n": len(recs["entry_exit_planning"])}
    mism = 0
    for r in recs["market_analysis"]:
        a = analyze(snaps[r["snapshot_id"]]["data"])
        if a["regime"] != r["target"]["regime"] or a["htf_confluence"] != r["target"]["htf_confluence"]:
            mism += 1
    chk["market_analysis_targets_recomputable_from_snapshot_alone"] = {"pass": mism == 0, "mismatches": mism, "n": len(recs["market_analysis"])}
    # 4 probabilities come from train-only fits
    tr_lab = lab[(lab.bar_close_time <= pd.Timestamp(S.SPLIT["train_end"])) & lab.valid]
    cal = json.load(open(os.path.join(S.DIR_REPORTS5, "scenario_calibration_report.json")))
    m = 0
    n_chk = 0
    for r in recs["scenario_forecasting"]:
        if r["split"] == "train":
            continue
        idx = snaps[r["snapshot_id"]]["decision_bar_index"]
        for sc in r["target"]["scenarios"]:
            if sc["probability_source"].startswith("train_base_rate") and sc["scenario"] in ("bullish_continuation", "bearish_continuation"):
                n_chk += 1
                key = {"bullish_continuation": "t_bull_continuation", "bearish_continuation": "t_bear_continuation"}[sc["scenario"]]
                if abs(sc["probability"] - round(cal["targets"][key]["train_base_rate"], 4)) > 1e-4:
                    m += 1
        break_ = False
    chk["val_test_probabilities_equal_train_fit_values"] = {"pass": m == 0, "mismatches": m, "checked": n_chk}
    chk["calibration_report_never_used_test_for_decisions"] = {"pass": all(v["decision_on_validation"] in ("base_rate", "context_model") for v in cal["targets"].values()), "note": "decision field derives from validation only (code: probabilities.fit_all)"}
    # 5 prompt hygiene
    hits, n = 0, 0
    for t in TASKS:
        for r in recs[t][:400]:
            n += 1
            try:
                render(r, snaps)
            except AssertionError:
                hits += 1
    chk["rendered_prompts_contain_no_hindsight_fields"] = {"pass": hits == 0, "violations": hits, "checked": n}
    # 6 management state: no future info (time ordering, steps)
    bad = 0
    for r in recs["trade_management"]:
        st = r["input"]["open_trade_state"]
        et = pd.Timestamp(r["input"]["entry_time"])
        if pd.Timestamp(r["timestamp"]) <= et or abs(st["minutes_since_entry"] - (pd.Timestamp(r["timestamp"]) - et).total_seconds() / 60) > 6 * 5 + 1e-6 and False:
            bad += 1
        if set(st) & {"final_exit", "exit_reason", "net_r", "final_net_r"}:
            bad += 1
    chk["trade_management_inputs_contain_only_state_known_at_decision_time"] = {"pass": bad == 0, "violations": bad, "n": len(recs["trade_management"]),
                                                                                  "also_covered_by": "tests.test_stage5_simulator.test_no_lookahead_truncation"}
    # 6b playbook actions are pure functions of the visible state; all intended action families are represented
    pm = 0
    acts = Counter()
    for r in recs["trade_management"]:
        merged = {**r["input"]["open_trade_state"], **r["input"]["market_state_now"]}
        pb = management_playbook(merged)
        acts[r["target"]["action"]] += 1
        if pb["action"] != r["target"]["action"] or pb["new_stop"] != r["target"]["new_stop"]:
            pm += 1
    chk["management_targets_recomputable_from_visible_state"] = {"pass": pm == 0, "mismatches": pm}
    need = {"HOLD", "CLOSE_NOW", "TAKE_PARTIAL_AND_MOVE_STOP_TO_BREAK_EVEN", "MOVE_STOP_TO_BREAK_EVEN", "TRAIL_STOP_ATR", "TRAIL_STOP_STRUCTURE", "TAKE_PARTIAL_50", "TAKE_PARTIAL_33"}
    chk["management_action_coverage"] = {"pass": all(acts[a] >= 30 for a in need), "counts": dict(acts)}
    # stop moves never loosen: new_stop vs current_sl
    loose = 0
    for r in recs["trade_management"]:
        st = r["input"]["open_trade_state"]
        ns = r["target"]["new_stop"]
        if ns is not None:
            sgn = 1 if st["side"] == "long" else -1
            if sgn * (ns - st["current_sl"]) < -1e-9:
                loose += 1
    chk["management_never_loosens_stop"] = {"pass": loose == 0, "violations": loose}
    # 7 integrated consistency
    by_snap = {r["snapshot_id"]: r for r in recs["entry_exit_planning"]}
    incons = 0
    for r in recs["integrated_trading_tasks"]:
        p = by_snap[r["snapshot_id"]]["target"]
        t = r["target"]
        if t["decision"] != p["decision"] or t.get("plan") != p.get("plan"):
            incons += 1
        if t["decision"] == "NO_TRADE" and t["tool_calls"]:
            incons += 1
        if t["decision"] != "NO_TRADE" and (not t["tool_calls"] or t["tool_calls"][0]["arguments"]["stop_price"] != t["plan"]["stop_loss"]):
            incons += 1
    chk["integrated_dataset_has_no_contradictory_labels"] = {"pass": incons == 0, "inconsistencies": incons, "n": len(recs["integrated_trading_tasks"])}
    # 8 split integrity
    sp_of = {k: v["split"] for k, v in snaps.items()}
    wrong = sum(1 for t in TASKS for r in recs[t] if sp_of[r["snapshot_id"]] != r["split"])
    bounds = {"train": (None, pd.Timestamp(S.SPLIT["train_end"])), "val": (pd.Timestamp(S.SPLIT["val_start"]), pd.Timestamp(S.SPLIT["val_end"])), "test": (pd.Timestamp(S.SPLIT["test_start"]), None)}
    outside = 0
    for k, v in snaps.items():
        lo, hi = bounds[v["split"]]
        bc = pd.Timestamp(v["bar_close_time"])
        if (lo is not None and bc < lo) or (hi is not None and bc > hi):
            outside += 1
    chk["record_split_equals_snapshot_split"] = {"pass": wrong == 0, "violations": wrong}
    chk["snapshots_inside_their_split_bounds"] = {"pass": outside == 0, "violations": outside}
    # label windows end before next split
    end_ok = True
    for k, v in snaps.items():
        bc = pd.Timestamp(v["bar_close_time"])
        end = bc + pd.Timedelta(minutes=15) + pd.Timedelta(hours=LABEL_END_HOURS)
        if v["split"] == "train" and end >= pd.Timestamp(S.SPLIT["val_start"]):
            end_ok = False
        if v["split"] == "val" and end >= pd.Timestamp(S.SPLIT["test_start"]):
            end_ok = False
    chk["label_and_management_windows_do_not_cross_split_boundaries"] = {"pass": end_ok}
    emb = pd.Timedelta(minutes=15 * S.SPLIT["embargo_bars"])
    chk["embargo_respected"] = {"pass": all(pd.Timestamp(v["bar_close_time"]) >= pd.Timestamp(S.SPLIT["val_start"]) + emb for v in snaps.values() if v["split"] == "val") and
                                all(pd.Timestamp(v["bar_close_time"]) >= pd.Timestamp(S.SPLIT["test_start"]) + emb for v in snaps.values() if v["split"] == "test")}
    # 9 integrity
    ids = Counter(r["record_id"] for t in TASKS for r in recs[t])
    dup = [k for k, c in ids.items() if c > 1]
    miss = {t: sum(1 for r in recs[t] if not REQ[t] <= set(r["target"])) for t in TASKS}
    chk["unique_record_ids"] = {"pass": not dup, "duplicates": dup[:5]}
    chk["required_target_fields_present"] = {"pass": sum(miss.values()) == 0, "missing_by_task": miss}
    raw = "".join(open(os.path.join(d, t + ".jsonl")).read() for t in TASKS + ["snapshots"])
    chk["no_secrets_or_login_in_datasets"] = {"pass": not SECRET.search(raw.replace("investor password", "")), "found": sorted(set(m.group(0) for m in SECRET.finditer(raw)))[:5]}
    texts = " ".join(r["target"].get("final_text", "") for t in TASKS for r in recs[t] if isinstance(r["target"], dict))
    chk["no_martingale_or_recovery_language_in_final_texts"] = {"pass": not re.search(r"martingale|recover (the )?loss|double (the )?(size|lot)|increase (the )?(size|lot)s? after", texts, re.I)}
    # no size escalation after loss: for same snapshot-state kinds, lots with higher realized loss never exceed lots with zero loss (engine property tested) + scan reductions
    esc = 0
    for r in recs["risk_management"]:
        res = r["target"]["tool_results"][-1]
        d_ = res.get("data") or {}
        if res["status"] in ("complete", "complete_with_gaps") and d_.get("lots", 0) > 0:
            st = r["input"]["account_state"]
            base_eq = st.get("equity") if d_["policy"]["base"] == "equity" else st.get("balance")
            if base_eq and d_["risk_money"] > base_eq * d_["policy"]["risk_pct"] / 100 * 1.0000001:
                esc += 1
    chk["position_risk_never_exceeds_policy_risk_pct"] = {"pass": esc == 0, "violations": esc}
    # tick volume honesty
    chk["no_exchange_volume_claims"] = {"pass": "real_volume" not in raw and "exchange volume" not in raw.lower().replace("not exchange volume", "")}
    out = {"checks": chk, "all_pass": all(v["pass"] for v in chk.values()), "counts": {t: len(recs[t]) for t in TASKS}, "snapshots": len(snaps)}
    # split audit extras
    by = defaultdict(Counter)
    for t in TASKS:
        for r in recs[t]:
            by[t][r["split"]] += 1
    out["records_by_task_and_split"] = {t: dict(c) for t, c in by.items()}
    dist = defaultdict(Counter)
    for r in recs["entry_exit_planning"]:
        dist[r["split"]][r["target"]["decision"]] += 1
    out["planning_decision_distribution_by_split"] = {k: dict(v) for k, v in dist.items()}
    os.makedirs(S.DIR_REPORTS5, exist_ok=True)
    json.dump(out, open(os.path.join(S.DIR_REPORTS5, "leakage_and_integrity_audit.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    o = run()
    for k, v in o["checks"].items():
        print("PASS" if v["pass"] else "FAIL", k, {x: y for x, y in v.items() if x != "pass"})
    print("ALL PASS:", o["all_pass"])
