"""Builds the unified multi-task datasets (JSONL) from validated Stage 4 features + Stage 5 labels/probabilities/simulator.

Layout (out/07_task_datasets):
  snapshots.jsonl            one causal market snapshot per decision (referenced by snapshot_id - never duplicated in task files)
  market_analysis / scenario_forecasting / entry_exit_planning / risk_management / position_sizing / trade_management /
  portfolio_risk / post_trade_review / integrated_trading_tasks .jsonl     (input + target only)
  direction_labels.jsonl     the preserved LONG/SHORT/NO_TRADE triple-barrier labels (hindsight supervision, separate by design)
  outcomes/*.jsonl           hindsight outcomes for EVALUATION ONLY (never rendered into prompts)
  manifest.json
Every tool-using target contains the real tool calls and the real tool results produced by broker_context (fixed clock),
so numbers in the final answer are copied from deterministic tools, never invented.
"""
from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

from broker_context.tools import TOOL_SCHEMAS, create_tools

from . import s5config as S
from .analysis import PLAN, analyze, management_playbook, plan_trade
from .evaluate import procedure_frame
from .simulator import POLICIES, make_context, rescore_net_r, simulate_trade

SCHEMA_VERSION = "5.1.0"
SYSTEM_PROMPT = (
    "You are an XAUUSD (gold) trading-analysis assistant. You analyze only the data in the prompt. Never invent prices, probabilities, lot sizes, tick values, "
    "margin or costs: call the provided tools for every exact financial value and report tool results exactly; if a tool returns incomplete, partial or stale, say what "
    "is missing. Probabilities must come from the supplied calibrated estimates; say so when they equal base rates. NO_TRADE is a valid and often correct decision. "
    "Never propose martingale, loss recovery or larger size after losses. Live order execution is out of scope.")


class FixedClock:
    def __call__(self):
        return dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)


def _env(kind: str):
    home = tempfile.mkdtemp(prefix="train_env_")
    t = create_tools(home, None, FixedClock())
    spec = {"currency": "USD", "leverage": 500, "symbol": "XAUUSD", "digits": 2, "point": 0.01, "tick_size": 0.01, "tick_value": 1.0, "contract_size": 100.0,
            "volume_min": 0.01, "volume_max": 100.0, "volume_step": 0.01, "currency_base": "XAU", "currency_profit": "USD", "calc_mode": "cfd",
            "spread_points": 12, "swap_long": -20.0, "swap_short": 5.0, "swap_mode": "points"}
    if kind == "unknown_contract":
        for k in ("contract_size", "tick_size", "tick_value"):
            spec.pop(k)
    t.call("update_user_account_profile", {"fields": spec, "confirmed": True})
    if kind != "unknown_commission":
        t.call("update_user_account_profile", {"fields": {"commission_model": "per_lot", "commission_amount": 3.5, "commission_currency": "USD",
                                                          "commission_basis": "per_lot_per_side"}, "confirmed": True})
    return t


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (isinstance(o, float) and math.isnan(o)) or (isinstance(o, np.floating) and np.isnan(o)) else round(float(o), 4)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, dt.datetime)):
        return o.strftime("%Y-%m-%d %H:%M:%S")
    return o


def snap_of(row: dict) -> dict:
    d = {}
    for c in S.SNAPSHOT_COLS:
        v = row.get(c)
        d[c] = None if (v is None or (isinstance(v, float) and math.isnan(v))) else v
    d["time"] = str(row["time"])
    return clean(d)


def sid_of(bar_close) -> str:
    return "XAUUSD_M15_" + pd.Timestamp(bar_close).strftime("%Y%m%dT%H%M")


def select(lab: pd.DataFrame, an: list) -> np.ndarray:
    ev = np.array([bool(a["liquidity_sweeps"]["buy_side_swept_now"] or a["liquidity_sweeps"]["sell_side_swept_now"] or a["last_structure_event"] in ("BOS bullish", "BOS bearish", "CHOCH bullish", "CHOCH bearish") and (a["bars_since_structure_event"] or 99) <= 1) for a in an])
    setup = np.array([a["setup"]["side"] is not None for a in an])
    i = lab.idx.values
    return ((i % 24 == 0) | (ev & (i % 10 == 0)) | (setup & (i % 8 == 0)))


def _scenario(snap, an, probs, calib, moves, lab_row):
    p = lambda k: None if probs.get("p_" + k) is None or (isinstance(probs.get("p_" + k), float) and math.isnan(probs.get("p_" + k))) else round(float(probs["p_" + k]), 4)  # noqa: E731
    n = lambda k: int(probs["n_" + k])  # noqa: E731
    dec = calib["decisions"]
    src = lambda t: "context_model_validated" if dec["t_" + t] == "context_model" else "train_base_rate(no validated contextual skill)"  # noqa: E731
    ev = an["evidence"]
    hi, lo = snap.get("last_swing_high"), snap.get("last_swing_low")
    mv = S.SCENARIO["move_atr"]
    sc = [
        {"scenario": "bullish_continuation", "probability": p("bull_continuation"), "probability_source": src("bull_continuation"), "n_support": n("bull_continuation"),
         "definition": f"price reaches +{mv} ATR before -{mv} ATR within {S.LABEL['horizon_bars']} M15 bars", "supporting_evidence": ev["bullish"], "contradicting_evidence": ev["bearish"],
         "confirmation": f"M15 closes above the last confirmed swing high {hi}" if hi else "no confirmed swing high available", "invalidation": f"M15 closes below the last confirmed swing low {lo}" if lo else "n/a"},
        {"scenario": "bearish_continuation", "probability": p("bear_continuation"), "probability_source": src("bear_continuation"), "n_support": n("bear_continuation"),
         "definition": f"price reaches -{mv} ATR before +{mv} ATR within {S.LABEL['horizon_bars']} M15 bars", "supporting_evidence": ev["bearish"], "contradicting_evidence": ev["bullish"],
         "confirmation": f"M15 closes below the last confirmed swing low {lo}" if lo else "n/a", "invalidation": f"M15 closes above the last confirmed swing high {hi}" if hi else "n/a"},
        {"scenario": "range_or_consolidation", "probability": p("range"), "probability_source": src("range"), "n_support": n("range"),
         "definition": f"neither +{mv} ATR nor -{mv} ATR is reached within {S.LABEL['horizon_bars']} bars",
         "supporting_evidence": [f"volatility state: {an['volatility']}", f"regime: {an['regime']}"], "contradicting_evidence": [f"displacement/sweep/structure break present"] if an["regime"] == "reversal_candidate" else [],
         "confirmation": "price stays between the last confirmed swing high and low", "invalidation": "M15 close beyond either swing"},
        {"scenario": "potential_reversal", "probability": p("reversal") if an["bias"]["M15"] in ("bullish", "bearish") else None, "probability_source": src("reversal"), "n_support": n("reversal"),
         "definition": "first +/-1.5 ATR excursion goes against the current M15 structure state (only defined when M15 state is bullish/bearish)",
         "supporting_evidence": [e for e in (f"last structure event: {an['last_structure_event']}",) if an["regime"] == "reversal_candidate"] + (["buy-side sweep"] if an["liquidity_sweeps"]["buy_side_swept_now"] else []) + (["sell-side sweep"] if an["liquidity_sweeps"]["sell_side_swept_now"] else []),
         "contradicting_evidence": [f"HTF confluence: {an['htf_confluence']}"], "confirmation": "CHOCH against the current structure on M15 close", "invalidation": "new BOS in the direction of the current structure"},
        {"scenario": "no_trade_or_insufficient_information", "probability": p("no_trade"), "probability_source": src("no_trade"), "n_support": n("no_trade"),
         "definition": "triple-barrier labeling would give neither a LONG nor a SHORT win (1.5 ATR stop, 2R target, 32 bars, assumed costs)",
         "supporting_evidence": an["no_trade_flags"] or ["no specific blocking flag; edge not demonstrated"], "contradicting_evidence": [], "confirmation": "flags persist", "invalidation": "flags clear and confluent setup forms"},
    ]
    hk = str((float(snap.get("h4_trend_state") or 0), float(snap.get("h1_trend_state") or 0)))
    m = moves.get(hk)
    atr, close = snap["atr_14"], snap["close"]
    expect = None
    if m:
        q = lambda arr: {"p10": arr[0], "p25": arr[1], "median": arr[2], "p75": arr[3], "p90": arr[4]}  # noqa: E731
        expect = {"context": f"H4/H1 trend states {hk} (training set)", "n_train": m["n"], "max_up_excursion_atr": q(m["up_exc_atr_q"]), "max_down_excursion_atr": q(m["dn_exc_atr_q"]),
                  "max_up_price_median": round(close + atr * m["up_exc_atr_q"][2], 2), "max_down_price_median": round(close - atr * m["dn_exc_atr_q"][2], 2),
                  "median_bars_to_tp_long": m["bars_to_tp_median_long"], "horizon_bars": S.LABEL["horizon_bars"]}
    return {"scenarios": sc, "trade_outcome_probabilities": {
                "long_tp_before_sl": {"p": p("long_tp"), "source": src("long_tp"), "n_support": n("long_tp")},
                "short_tp_before_sl": {"p": p("short_tp"), "source": src("short_tp"), "n_support": n("short_tp")},
                "definition": f"SL {S.LABEL['sl_atr_mult']} ATR, TP {S.LABEL['rr']}R, {S.LABEL['horizon_bars']} bars, next-bar-open entry"},
            "trend_continuation_probability": {"p": p("trend_continuation"), "source": src("trend_continuation")} if an["bias"]["M15"] in ("bullish", "bearish") else None,
            "expected_price_movement": expect,
            "uncertainty": ["Probabilities are historical frequencies fit on the training split and checked on validation; they are not model confidence.",
                            "Directional probabilities equal base rates unless marked context_model_validated; the validation study found no demonstrated directional skill from these features."
                            if dec["t_bull_continuation"] == "base_rate" else "Context model validated for the directional targets.",
                            "Costs and contract specs behind trade-outcome probabilities are unverified historical assumptions."]}


def _risk_scenario(rng, kind, ref_price, atr_val):
    bal = float(rng.choice([1000, 2500, 5000, 10000, 25000, 100000]))
    eq = round(bal * float(rng.uniform(0.93, 1.03)), 2)
    st = {"balance": bal, "equity": eq, "margin_used": 0.0, "peak_equity": round(max(bal, eq) * float(rng.uniform(1.0, 1.06)), 2)}
    pol = {"risk_pct": float(rng.choice([0.25, 0.5, 1.0])), "base": str(rng.choice(["equity", "balance"]))}
    if kind in ("risk", "portfolio"):
        pol["max_daily_loss_pct"] = float(rng.choice([1.0, 2.0, 3.0]))
        pol["max_weekly_loss_pct"] = float(rng.choice([3.0, 4.0, 6.0]))
        pol["max_drawdown_pct"] = float(rng.choice([10.0, 15.0, 20.0]))
        pol["max_positions"] = int(rng.choice([2, 3, 4]))
        pol["max_same_direction_positions"] = int(rng.choice([2, 3]))
        st["realized_loss_today"] = round(bal * float(rng.choice([0, 0, 0, 0, 0.003, 0.008, 0.015, 0.025])), 2)
        st["realized_loss_week"] = round(max(st["realized_loss_today"], bal * float(rng.choice([0, 0, 0.005, 0.015, 0.03]))), 2)
        st["margin_used"] = round(eq * float(rng.choice([0, 0, 0.01, 0.03, 0.1])), 2)
    if kind == "portfolio":
        n = int(rng.choice([0, 0, 0, 1, 1, 1, 2, 2, 3, 4]))
        st["open_positions"] = []
        for _ in range(n):
            side = str(rng.choice(["long", "short"]))
            sd = float(rng.uniform(0.8, 2.5)) * atr_val
            ent = ref_price + float(rng.uniform(-3, 3)) * atr_val
            st["open_positions"].append({"side": side, "lots": float(rng.choice([0.01, 0.01, 0.02, 0.03, 0.05, 0.1])), "entry": round(ent, 2),
                                         "stop": round(ent - sd if side == "long" else ent + sd, 2)})
    if float(rng.random()) < 0.06:
        st.pop("equity" if pol["base"] == "equity" else "balance")
    return st, pol


def _fmt_size(res):
    d = res["data"] or {}
    if res["status"] in ("complete", "complete_with_gaps"):
        s = (f"Tool result ({res['status']}): {d['lots']} lots, monetary risk {d['risk_money']} {d['currency']} ({d['risk_pct_of_base']}% of {d['policy']['base']}), "
             f"binding constraint: {d['binding_constraint']}.")
        if d.get("reductions"):
            s += " Reductions applied: " + "; ".join(d["reductions"]) + "."
        if d["margin"].get("required") is not None:
            s += f" Margin required {round(d['margin']['required'], 2)} {d['currency']}."
        if res["warnings"]:
            s += " Warnings: " + "; ".join(res["warnings"]) + "."
        return s
    reason = (d or {}).get("reason") or "; ".join(res.get("warnings", []))
    miss = res.get("missing_fields") or []
    return f"Tool result ({res['status']}): no position size can be approved. {reason}" + (f" Missing: {', '.join(miss)}." if miss else "") + " I will not estimate a size without the tool."


def build(out_dir=None, seed=2026, max_trades=2400):
    out_dir = out_dir or S.DIR_DATASETS
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(os.path.join(out_dir, "outcomes"))
    lab, feat, recs, ans = procedure_frame()
    probs = pd.read_csv(os.path.join(S.DIR_LABELS, "probabilities_v1.csv")).set_index("idx")
    calib = json.load(open(os.path.join(S.DIR_REPORTS5, "scenario_calibration_report.json")))
    moves = {k: v for k, v in calib["movement_quantiles_train_by_h4_h1_trend"].items()}
    sel = select(lab, ans)
    lab = lab[sel].reset_index(drop=True)
    recs = [r for r, s_ in zip(recs, sel) if s_]
    ans = [a for a, s_ in zip(ans, sel) if s_]
    run_id = f"{S.DATASET_VERSION}__{S.LABEL_CONFIG_VERSION}__{S.make_profile(S.PRIMARY_SCENARIO).profile_id[:12]}"
    prov = {"dataset_version": S.DATASET_VERSION, "stage5_version": S.STAGE5_VERSION, "label_config_version": S.LABEL_CONFIG_VERSION, "label_run_id": run_id,
            "historical_cost_profile": S.PRIMARY_SCENARIO, "cost_profile_id": S.make_profile(S.PRIMARY_SCENARIO).profile_id, "schema_version": SCHEMA_VERSION,
            "source": "Alpari MT5 candles (tick volume only), server-clock timestamps (timezone unverified)"}
    files = {k: open(os.path.join(out_dir, k + ".jsonl"), "w") for k in ["snapshots", "market_analysis", "scenario_forecasting", "entry_exit_planning", "risk_management",
                                                                          "position_sizing", "trade_management", "portfolio_risk", "post_trade_review", "integrated_trading_tasks", "direction_labels"]}
    outc = {k: open(os.path.join(out_dir, "outcomes", k + ".jsonl"), "w") for k in ["snapshot_outcomes", "management_outcomes", "post_trade_outcomes"]}
    W = lambda f, o: f.write(json.dumps(clean(o), ensure_ascii=False, separators=(",", ":")) + "\n")  # noqa: E731
    envs = {"known": _env("known"), "unknown_commission": _env("unknown_commission"), "unknown_contract": _env("unknown_contract")}
    ctx, prim = make_context(), S.make_profile(S.PRIMARY_SCENARIO)
    counts = {}
    bump = lambda k, split: counts.setdefault(k, {}).update({split: counts.get(k, {}).get(split, 0) + 1})  # noqa: E731
    elig = {sp_: sum(1 for rw, aa in zip(lab.to_dict("records"), ans) if rw["split"] == sp_ and aa["setup"]["side"] and rw["idx"] % 2 == 0) for sp_ in ("train", "val", "test")}
    keep_p = {sp_: min(1.0, max_trades / max(n_, 1)) for sp_, n_ in elig.items()}      # uniform subsample across the whole period (no chronological truncation)
    for i, (r, row, a) in enumerate(zip(recs, lab.to_dict("records"), ans)):
        snap = snap_of(r)
        a = analyze(snap)          # targets are functions of exactly what the model sees (4-dp rounded snapshot)
        sid = sid_of(row["bar_close_time"])
        split = row["split"]
        rng = np.random.default_rng(seed + int(row["idx"]))
        P = probs.loc[int(row["idx"])].to_dict()
        base = {"snapshot_id": sid, "split": split, "timestamp": str(row["bar_close_time"]), "provenance": prov}
        mk = lambda task, suffix="": {"record_id": f"{task}:{sid}{suffix}", "task": task, **base}  # noqa: E731
        W(files["snapshots"], {"snapshot_id": sid, "split": split, "bar_close_time": str(row["bar_close_time"]), "decision_bar_index": int(row["idx"]), "data": snap, "provenance": prov})
        # ---- 1 market analysis
        mana = {k: a[k] for k in ("bias", "htf_confluence", "regime", "volatility", "atr_ratio", "session", "spread_atr", "premium_discount", "last_structure_event",
                                  "bars_since_structure_event", "liquidity_sweeps", "zones", "evidence", "no_trade_flags")}
        mana["levels"] = {k: snap.get(k) for k in ("last_swing_high", "last_swing_low", "prev_day_high", "prev_day_low")}
        mana["no_trade_recommended_by_flags"] = bool(a["no_trade_flags"])
        rec = mk("market_analysis")
        rec.update(input={"instruction": "Analyze the market structure, regime, volatility, liquidity, zones and multi-timeframe confluence. State bullish/bearish evidence and conditions where no trade is appropriate."}, target=mana)
        W(files["market_analysis"], rec); bump("market_analysis", split)
        # ---- 2 scenario forecasting
        scen = _scenario(snap, a, P, calib, moves, row)
        rec = mk("scenario_forecasting")
        rec.update(input={"instruction": f"List the plausible scenarios for the next {S.LABEL['horizon_bars']} M15 bars with calibrated probabilities (only those supplied by the calibration), evidence, confirmation and invalidation."}, target=scen)
        W(files["scenario_forecasting"], rec); bump("scenario_forecasting", split)
        # ---- 3 entry / exit planning (causal procedure)
        side = a["setup"]["side"]
        if side:
            plan = plan_trade(snap, side, a)
            decision = {"decision": side.upper(), "setup_quality": a["setup"]["quality"], "supporting_evidence": a["setup"]["supporting"], "contradicting_evidence": a["setup"]["contradicting"],
                        "plan": plan, "why_not_every_move": "A trade is proposed only with HTF confluence, >=3 aligned evidence items, no blocking execution/volatility flag and a clear opposing level distance."}
        else:
            decision = {"decision": "NO_TRADE", "reasons": a["no_trade_flags"], "what_would_change_the_decision": [
                "H4 and H1 structure agree with no D1 opposition", "at least 3 aligned evidence items and more than the opposing side", "spread/ATR <= 0.20 and ATR ratio <= 1.8", "inside a defined liquid session"]}
        rec = mk("entry_exit_planning")
        rec.update(input={"instruction": "Decide LONG, SHORT or NO_TRADE. If trading, give order type (Market/Limit/Stop), entry, stop-loss, take-profits, risk/reward, confirmation, invalidation and cancellation conditions. Do not compute lot size."}, target=decision)
        W(files["entry_exit_planning"], rec); bump("entry_exit_planning", split)
        # ---- preserved hindsight labels
        W(files["direction_labels"], {**mk("direction_labels"), "label": row["label"], "no_trade_reason": row["no_trade_reason"] if isinstance(row["no_trade_reason"], str) else None,
                                      "long": {"outcome": row["long_outcome"], **{f"net_r_{s_}": row[f"long_net_r_{s_}"] for s_ in S.COST_SCENARIOS}},
                                      "short": {"outcome": row["short_outcome"], **{f"net_r_{s_}": row[f"short_net_r_{s_}"] for s_ in S.COST_SCENARIOS}},
                                      "note": "hindsight supervision (triple barrier); intentionally NOT used as the planning target"})
        W(outc["snapshot_outcomes"], {"snapshot_id": sid, "split": split, "label": row["label"], "long_outcome": row["long_outcome"], "short_outcome": row["short_outcome"],
                                      "up_exc_atr": row["up_exc_atr"], "dn_exc_atr": row["dn_exc_atr"], "first_touch_dir": row["first_touch_dir"], "end_ret_atr": row["end_ret_atr"],
                                      "procedure_decision": row["proc_decision"]})
        # ---- 4/5/8 account-dependent tasks
        spread_price = round((snap["spread"] or 0) * 0.01, 4) if snap.get("spread_known") else None
        atr, close = snap["atr_14"], snap["close"]
        if side:
            ent, stp, sd = decision["plan"]["entry_price"], decision["plan"]["stop_loss"], side
        else:
            sd = "long" if rng.random() < 0.5 else "short"
            ent, stp = close, round(close - (1 if sd == "long" else -1) * 1.5 * atr, 2)
        for task, kind, key in (("position_sizing", "size", 3), ("risk_management", "risk", 3), ("portfolio_risk", "portfolio", 4)):
            if int(row["idx"]) % key != 0 and task != "portfolio_risk":
                continue
            if task == "portfolio_risk" and int(row["idx"]) % 4 != 0:
                continue
            envk = "known" if rng.random() < 0.7 else ("unknown_commission" if rng.random() < 0.8 else "unknown_contract")
            T = envs[envk]
            st, pol = _risk_scenario(rng, kind, ent, atr)
            args = {"side": sd, "entry_price": ent, "stop_price": stp, "account_state": st, "policy": pol, "atr": atr, "atr_ratio": snap["atr_ratio"], "spread_price": spread_price}
            args = {k: v for k, v in args.items() if v is not None}
            calls, results = [], []
            if task == "portfolio_risk":
                c1 = {"account_state": st, "policy": pol}
                calls.append({"name": "validate_portfolio_risk", "arguments": c1}); results.append(T.call("validate_portfolio_risk", c1))
            calls.append({"name": "calculate_position_size", "arguments": args}); results.append(T.call("calculate_position_size", args))
            sz = results[-1]
            if task == "position_sizing" and sz["status"] in ("complete", "complete_with_gaps") and rng.random() < 0.5:
                c3 = {"side": sd, "lots": sz["data"]["lots"], "entry_price": ent, "stop_price": stp, "slippage_price_per_side": 0.05}
                calls.append({"name": "calculate_trade_costs", "arguments": c3}); results.append(T.call("calculate_trade_costs", c3))
            final = _fmt_size(sz)
            if task == "portfolio_risk":
                pv = results[0]
                dec = "BLOCK_NEW_TRADE" if (pv["status"] == "violation" or sz["status"] in ("blocked", "invalid")) else ("INCOMPLETE" if "incomplete" in (pv["status"], sz["status"]) else "ALLOW")
                final = f"Portfolio check: {pv['status']}" + (f" (violations: {', '.join(pv['data']['violations'])})" if pv["data"] and pv["data"].get("violations") else "") + ". " + final + f" Decision: {dec}."
            else:
                dec = {"complete": "APPROVE", "complete_with_gaps": "APPROVE_WITH_UNVERIFIED_ITEMS", "blocked": "BLOCK", "incomplete": "INCOMPLETE", "invalid": "INVALID"}[sz["status"]]
                if sz["status"] in ("complete", "complete_with_gaps") and (sz["data"]["reductions"] or sz["data"]["binding_constraint"] != "policy_risk_pct"):
                    dec = "APPROVE_REDUCED"
            rec = mk(task)
            instr = {"position_sizing": "Compute the position size for the proposed trade with the deterministic tools and report it.",
                     "risk_management": "Apply the stated risk policy and account state to the proposed trade; use the tools; explain any reduction or block.",
                     "portfolio_risk": "Check the open portfolio against the policy and decide whether a new trade may be added; use the tools."}[task]
            rec.update(input={"instruction": instr, "proposed_trade": {"side": sd, "entry_price": ent, "stop_price": stp, "source": "plan" if side else "user_request"},
                              "account_state": st, "risk_policy": pol, "account_context": {"environment": envk, "note": "training account context; at runtime tools read the live/user profile"}},
                       target={"tool_calls": calls, "tool_results": results, "decision": dec, "final_text": final})
            W(files[task], rec); bump(task, split)
            if task == "risk_management" and side:
                pass
        # ---- 9 integrated tasks (consistent chain)
        rec = mk("integrated_trading_tasks")
        T = envs["known"]
        calls, results, final_trade = [], [], None
        if side:
            st, pol = _risk_scenario(np.random.default_rng(seed + 7 * int(row["idx"])), "risk", ent, atr)
            args = {"side": side, "entry_price": ent, "stop_price": stp, "account_state": st, "policy": pol, "atr": atr, "atr_ratio": snap["atr_ratio"], "spread_price": spread_price}
            args = {k: v for k, v in args.items() if v is not None}
            calls.append({"name": "calculate_position_size", "arguments": args}); results.append(T.call("calculate_position_size", args))
            final_trade = {"size_status": results[0]["status"], "lots": (results[0]["data"] or {}).get("lots")}
            final_txt = (f"{side.upper()} setup (quality {a['setup']['quality']}); {decision['plan']['order_type']} entry {decision['plan']['entry_price']}, stop {decision['plan']['stop_loss']}, "
                         f"TP1 {decision['plan']['take_profit_1']}, TP2 {decision['plan']['take_profit_2']}. " + _fmt_size(results[0]))
            acc_in = {"account_state": st, "risk_policy": pol}
        else:
            final_txt = "NO_TRADE: " + "; ".join(a["no_trade_flags"])
            acc_in = None
        rec.update(input={"instruction": "Full workflow: analyze, compare scenarios with the supplied calibrated probabilities, decide LONG/SHORT/NO_TRADE, plan the trade, size it with tools and state invalidation.", **({"account": acc_in} if acc_in else {"account": {"note": "not needed unless a trade is proposed"}})},
                   target={"analysis_summary": {k: mana[k] for k in ("htf_confluence", "regime", "volatility", "session", "premium_discount")},
                           "scenario_probabilities": {s_["scenario"]: s_["probability"] for s_ in scen["scenarios"]}, "probability_note": scen["uncertainty"][1],
                           "decision": decision["decision"], "plan": decision.get("plan"), "reasons_if_no_trade": decision.get("reasons"),
                           "tool_calls": calls, "tool_results": results, "final_text": final_txt})
        W(files["integrated_trading_tasks"], rec); bump("integrated_trading_tasks", split)
        # ---- 6/7 trade management + post-trade review (simulated paths, setups only)
        if side and int(row["idx"]) % 2 == 0 and np.random.default_rng(seed * 3 + int(row["idx"])).random() < keep_p[split]:
            sl_dist = float(row["sl_dist"])
            alt = {}
            for pol_name, cfg in POLICIES.items():
                # Dense, still-causal sampling is essential around partial exits and trailing transitions;
                # sparse sampling can skip the short-lived states that trigger those playbook actions.
                interval = 1 if pol_name in ("multi_tp", "structure_trail", "atr_trail") else 6
                t = simulate_trade(ctx, side, int(row["idx"]), cfg, prim, float(row["entry_price"]), sl_dist, int(row["entry_time"].value), record_every=interval)
                alt[pol_name] = t
            review_policy = str(rng.choice(["fixed", "break_even", "atr_trail", "structure_trail", "partial_be", "multi_tp"]))
            base_t = alt[review_policy]  # preserve randomized policy selection for post-trade review
            # Build candidates from every independently simulated policy path. Each target is still computed
            # only from that candidate's visible state; no future outcome enters action selection.
            def _mk(st_):
                state = {k: st_[k] for k in ("side", "entry_price", "initial_sl", "initial_tp", "current_sl", "current_tp", "remaining_size", "unrealized_r", "mfe_r", "mae_r", "trailing_active",
                                             "break_even_active", "partial_exits", "costs_r_so_far", "price", "steps_held", "risk_distance", "atr_now", "atr_ratio_now", "structure_invalidated")}
                mkt = {"m15_close": st_["m15_close"], "m15_trend_state": st_["m15_trend_state"], "m15_last_swing_high": st_["m15_last_swing_high"], "m15_last_swing_low": st_["m15_last_swing_low"]}
                return state, mkt, management_playbook(clean({**state, **mkt}))
            cands = []
            for pol_name, path_result in alt.items():
                for st_ in path_result["records"]:
                    if st_["step"] < 1:
                        continue
                    state, mkt, pb = _mk(st_)
                    cands.append((pol_name, path_result, st_, state, mkt, pb))
            by_act = {}
            for candidate in cands:
                by_act.setdefault(candidate[5]["action"], []).append(candidate)
            # One example per naturally observed action per entry snapshot. No synthetic labels or duplication.
            order = ["TAKE_PARTIAL_33", "TRAIL_STOP_STRUCTURE", "TRAIL_STOP_ATR", "TAKE_PARTIAL_50", "MOVE_STOP_TO_BREAK_EVEN", "TAKE_PARTIAL_AND_MOVE_STOP_TO_BREAK_EVEN", "CLOSE_NOW", "HOLD"]
            chosen_states = []
            for act in order:
                if act in by_act:
                    options = by_act[act]
                    chosen_states.append(options[int(rng.integers(len(options)))])
            for pol_name, path_result, st_, state, mkt, pb in sorted(chosen_states, key=lambda item: (item[2]["step"], item[0])):
                record_id = f"trade_management:{sid}:{pol_name}:s{st_['step']}"
                mrec = {"record_id": record_id, "task": "trade_management", "snapshot_id": sid, "split": split,
                        "timestamp": str(pd.Timestamp(st_["time_ns"])), "provenance": {**prov, "management_simulation_policy": pol_name},
                        "input": {"instruction": "Decide how to manage this open trade now using only the state shown (no future information).",
                                  "entry_snapshot_id": sid, "entry_time": str(row["entry_time"]), "simulation_policy": pol_name,
                                  "open_trade_state": {**state, "minutes_since_entry": st_["steps_held"] * 5},
                                  "market_state_now": mkt},
                        "target": {"action": pb["action"], "new_stop": pb["new_stop"], "reasons": pb["reason"], "constraints": ["stops are only tightened, never widened", "no size increase", "no averaging down"]}}
                W(files["trade_management"], mrec); bump("trade_management", split)
                fin = path_result["net_r"]
                now_net = st_["unrealized_r"] - st_["costs_r_so_far"]
                W(outc["management_outcomes"], {"record_id": record_id, "split": split, "simulation_policy": pol_name,
                                                "final_net_r_if_hold_same_policy": round(fin, 4), "net_r_if_closed_now_approx": round(now_net, 4),
                                                "hindsight_hold_better": bool(fin > now_net + 0.15)})
            # post-trade review (information complete after exit)
            ex_t = base_t
            alt_net = {k: round(rescore_net_r(v, prim, float(row["spread_pts_decision"]), sl_dist), 3) for k, v in alt.items()}
            flags = []
            if ex_t["net_r"] < 0 and ex_t["mfe_r"] >= 1.0:
                flags.append("gave back >= 1R of open profit: a break-even/partial rule would have changed the outcome")
            if ex_t["net_r"] > 0 and ex_t["mae_r"] > 0.8:
                flags.append("winner came within 0.2R of the stop: stop placement was tight relative to noise")
            if ex_t["ambiguous"]:
                flags.append("intrabar ambiguity resolved by the worst-case (stop-first) policy")
            if a["setup"]["contradicting"]:
                flags.append("entry had contradicting evidence: " + "; ".join(a["setup"]["contradicting"]))
            cost_share = round(ex_t["costs_r"] / max(abs(ex_t["gross_r"]), 1e-9), 3)
            review = {"outcome": "win" if ex_t["net_r"] > 0.05 else ("loss" if ex_t["net_r"] < -0.05 else "scratch"), "exit_reason": ex_t["exit_reason"], "net_r": round(ex_t["net_r"], 3),
                      "gross_r": round(ex_t["gross_r"], 3), "cost_r": round(ex_t["costs_r"], 3), "cost_share_of_gross_move": cost_share, "mfe_r": round(ex_t["mfe_r"], 3), "mae_r": round(ex_t["mae_r"], 3),
                      "mfe_capture": None if not ex_t["mfe_r"] > 0.5 else round(ex_t["gross_r"] / ex_t["mfe_r"], 3), "process_flags": flags, "what_if_net_r_by_management_policy": alt_net,
                      "lessons": ["judge the decision by process quality and the stated risk, not by this single outcome", "a loss never justifies a larger next position or loss recovery sizing",
                                  "policy differences on one trade are noise; use the aggregate evaluation report"]}
            prec = {"record_id": f"post_trade_review:{sid}", "task": "post_trade_review", "snapshot_id": sid, "split": split, "timestamp": str(pd.Timestamp(ex_t["events"][-1]["step"] * 5 * 60 * 10**9 + int(row["entry_time"].value))), "provenance": prov,
                    "input": {"instruction": "Review this completed trade: classify the outcome, assess stop placement, costs and management, and state process lessons.",
                              "trade": {"side": side, "entry_time": str(row["entry_time"]), "entry_price": round(float(row["entry_price"]), 2), "initial_sl": round(ex_t["records"][0]["initial_sl"], 2) if ex_t["records"] else None,
                                        "initial_tp": round(ex_t["records"][0]["initial_tp"], 2) if ex_t["records"] else None, "bars_held_m5": ex_t["bars_held_m5"], "exit_events": ex_t["events"],
                                        "managed_policy": {"name": chosen, **POLICIES[chosen]}, "entry_snapshot_id": sid}},
                    "target": review}
            W(files["post_trade_review"], prec); bump("post_trade_review", split)
            W(outc["post_trade_outcomes"], {"record_id": prec["record_id"], "split": split, "net_r_by_policy": alt_net})
    for f in list(files.values()) + list(outc.values()):
        f.close()
    # manifest
    man = {"stage5_version": S.STAGE5_VERSION, "schema_version": SCHEMA_VERSION, "provenance": prov, "counts_by_task_and_split": counts, "snapshots": int(len(lab)),
           "split_config": S.SPLIT, "management_state_sampling": "one naturally observed state per available playbook action per simulated trade, sampled across all nine independent management-policy paths; dense M5 sampling for multi_tp/structure_trail/atr_trail; action is a pure function of visible state; trade subsample uniform over each split", "sampling": "idx%24==0 | event&idx%10==0 | setup&idx%8==0 (deterministic)", "tool_schemas": [t["name"] for t in TOOL_SCHEMAS],
           "files": {}}
    for fn in sorted(os.listdir(out_dir)):
        p = os.path.join(out_dir, fn)
        if os.path.isfile(p):
            man["files"][fn] = {"bytes": os.path.getsize(p), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
    json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    return man


if __name__ == "__main__":
    m = build()
    print(json.dumps(m["counts_by_task_and_split"], indent=0)); print({k: round(v["bytes"] / 1e6, 1) for k, v in m["files"].items()})
