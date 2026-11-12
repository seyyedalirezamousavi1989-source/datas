"""Deterministic position sizing, margin and portfolio risk-limit validation.

Reuses the SAME validated Snapshot and cost engine (no second cost engine).  Never guesses: missing balance/equity/margin
inputs give status 'incomplete'.  No loss-recovery, martingale or size escalation exists or can be configured: sizes only
shrink or stay equal as constraints bind, and a policy containing escalation keys is rejected.
Limitations: account state (balance/equity/open positions) is caller-supplied (user) - MT5 can supply balance/equity/margin
(see MT5Provider.fetch_account_state) but open positions are not read.  Margin formula implemented for MT5 calc modes
forex / cfd / cfd_leverage with the margin currency equal to the account currency or a USD-quoted metal in a USD account.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .context_snapshot import Snapshot
from .cost_engine import calculate_position_risk, value_per_price_unit
from .context_validator import validate

FORBIDDEN_POLICY_KEYS = ("martingale", "recovery", "double", "escalat", "scale_up_after_loss", "loss_multiplier", "gale")


@dataclass
class RiskPolicy:
    risk_pct: float = 0.5                      # % of base capital risked per trade (user-configurable; not claimed optimal)
    base: str = "equity"                       # "balance" | "equity"
    max_daily_loss_pct: float = 2.0
    max_weekly_loss_pct: float = 4.0
    max_drawdown_pct: float = 10.0             # from peak equity; at/above -> no new risk
    max_aggregate_open_risk_pct: float = 2.0
    max_positions: int = 3
    max_same_direction_positions: int = 2      # XAUUSD positions are the same exposure: direction-stacking is limited
    max_margin_usage_pct: float = 30.0         # margin used / equity after the new trade
    min_margin_level_pct: float = 300.0
    vol_reduce_atr_ratio: float = 1.5          # reduce size above this atr_ratio
    vol_reduce_factor: float = 0.5
    wide_spread_atr: float = 0.20              # poor execution: spread/ATR above this -> reduce
    wide_spread_factor: float = 0.5
    block_spread_atr: float = 0.35             # ... above this -> block
    def validate(self) -> None:
        for k in asdict(self):
            if any(f in k.lower() for f in FORBIDDEN_POLICY_KEYS):
                raise ValueError(f"policy key {k!r} is forbidden (no martingale / loss recovery / escalation)")
        if self.base not in ("balance", "equity"):
            raise ValueError("base must be 'balance' or 'equity'")
        if not 0 < self.risk_pct <= 5:
            raise ValueError("risk_pct must be in (0, 5]")
        for k in ("vol_reduce_factor", "wide_spread_factor"):
            if not 0 < getattr(self, k) <= 1:
                raise ValueError(f"{k} must be in (0, 1]: size may only be reduced, never increased")
        for k in ("max_daily_loss_pct", "max_weekly_loss_pct", "max_drawdown_pct", "max_aggregate_open_risk_pct", "max_margin_usage_pct"):
            if getattr(self, k) <= 0:
                raise ValueError(f"{k} must be > 0")

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "RiskPolicy":
        d = dict(d or {})
        for k in d:
            if any(f in k.lower() for f in FORBIDDEN_POLICY_KEYS):
                raise ValueError(f"policy key {k!r} is forbidden (no martingale / loss recovery / escalation)")
        unknown = set(d) - set(RiskPolicy.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown policy keys: {sorted(unknown)}")
        p = RiskPolicy(**d)
        p.validate()
        return p


@dataclass
class AccountState:
    balance: Optional[float] = None
    equity: Optional[float] = None
    margin_used: Optional[float] = None
    peak_equity: Optional[float] = None
    realized_loss_today: float = 0.0           # positive number = loss
    realized_loss_week: float = 0.0
    open_positions: List[Dict[str, Any]] = field(default_factory=list)   # {side, lots, entry, stop, risk_money?}

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "AccountState":
        d = dict(d or {})
        unknown = set(d) - set(AccountState.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown account-state keys: {sorted(unknown)}")
        return AccountState(**d)


def _round_down(x: float, step: float, vmin: float) -> float:
    k = math.floor((x - vmin) / step + 1e-9)
    return round(vmin + max(k, 0) * step, 10) if x >= vmin - 1e-12 else 0.0


def margin_required(snap: Snapshot, lots: float, price: float, leverage: Optional[float], fx_margin_to_account: Optional[float] = None):
    s, a = snap.symbol, snap.account
    mode, cs, mc, acc = s.value("calc_mode"), s.value("contract_size"), s.value("currency_margin"), a.value("currency")
    miss = [n for n, v in (("calc_mode", mode), ("contract_size", cs), ("leverage", leverage), ("account_currency", acc)) if v in (None, 0)]
    if miss:
        return None, miss, []
    if mode not in ("forex", "cfd", "cfd_leverage"):
        return None, [], [f"margin formula for MT5 calc mode {mode!r} is not implemented"]
    notes = []
    base_val = lots * cs * price / leverage if mode != "forex" or True else None
    # gold is quoted in USD per oz; the formula yields USD. Convert if the account currency differs.
    if str(acc).upper() != "USD":
        if not fx_margin_to_account:
            return None, ["fx_rate_usd_to_account"], []
        base_val *= fx_margin_to_account
        notes.append("margin converted with a supplied USD->account FX rate (estimate)")
    return base_val, [], notes


def calculate_position_size(snap: Snapshot, policy: RiskPolicy, state: AccountState, side: str, entry_price: float, stop_price: float,
                            atr_ratio: Optional[float] = None, spread_price: Optional[float] = None, atr: Optional[float] = None,
                            leverage: Optional[float] = None, fx_rate_profit_to_account: Optional[float] = None,
                            fx_rate_margin_to_account: Optional[float] = None) -> Dict[str, Any]:
    policy.validate()
    checks: List[Dict[str, Any]] = []
    reductions: List[str] = []
    warnings: List[str] = []
    base_val = state.equity if policy.base == "equity" else state.balance
    missing = [n for n, v in (("balance" if policy.base == "balance" else "equity", base_val),) if v is None]
    res: Dict[str, Any] = {"what": "position_size", "policy": asdict(policy), "context_id": snap.id, "context_stale": snap.is_stale}

    def done(status, lots=0.0, **kw):
        res.update(status=status, lots=lots, checks=checks, reductions=reductions, warnings=warnings, missing_fields=sorted(set(missing)), **kw)
        return res

    if side not in ("long", "short"):
        raise ValueError("side must be long/short")
    if (side == "long" and stop_price >= entry_price) or (side == "short" and stop_price <= entry_price):
        missing.append("valid_stop_side")
        return done("invalid", reason="stop must be below entry for long and above for short")
    if missing:
        return done("incomplete", reason="account state missing: " + ", ".join(missing))
    leverage = leverage or snap.account.value("leverage")
    rep = validate(snap, "position_risk")
    vpp, vb, vmiss, vnotes = value_per_price_unit(snap, fx_rate_profit_to_account)
    if vpp is None or rep.blocking_missing or rep.blocking_conflicts:
        missing += vmiss + rep.blocking_missing + rep.blocking_conflicts
        return done("incomplete", reason="contract specification / account currency unavailable or conflicting")
    vmin, vstep, vmax = snap.symbol.value("volume_min"), snap.symbol.value("volume_step"), snap.symbol.value("volume_max")
    if None in (vmin, vstep):
        missing += [n for n, v in (("volume_min", vmin), ("volume_step", vstep)) if v is None]
        return done("incomplete", reason="volume limits unknown; cannot round lots")
    dist = abs(entry_price - stop_price)
    risk_per_lot = dist * vpp
    # ---- hard blocks
    if state.peak_equity and state.equity is not None:
        dd = (state.peak_equity - state.equity) / state.peak_equity * 100
        ok = dd < policy.max_drawdown_pct
        checks.append({"name": "max_drawdown", "ok": ok, "value_pct": round(dd, 4), "limit_pct": policy.max_drawdown_pct})
        if not ok:
            return done("blocked", reason="max drawdown reached: no new risk")
    else:
        warnings.append("peak_equity not provided: max drawdown not checked")
    if spread_price is not None and atr:
        sp_atr = spread_price / atr
        if sp_atr > policy.block_spread_atr:
            checks.append({"name": "execution_quality", "ok": False, "spread_atr": round(sp_atr, 4)})
            return done("blocked", reason="spread too wide relative to ATR")
    # ---- risk budget (only reductions below)
    factor = 1.0
    if atr_ratio is not None and atr_ratio > policy.vol_reduce_atr_ratio:
        factor *= policy.vol_reduce_factor
        reductions.append(f"abnormal volatility (atr_ratio {atr_ratio:.2f} > {policy.vol_reduce_atr_ratio}): x{policy.vol_reduce_factor}")
    if spread_price is not None and atr and spread_price / atr > policy.wide_spread_atr:
        factor *= policy.wide_spread_factor
        reductions.append(f"poor execution (spread/ATR {spread_price / atr:.2f}): x{policy.wide_spread_factor}")
    risk_money = base_val * policy.risk_pct / 100 * factor
    # remaining budgets
    open_risk = 0.0
    for p in state.open_positions:
        r = p.get("risk_money")
        if r is None:
            r = abs(p["entry"] - p["stop"]) * vpp * p["lots"] if all(k in p for k in ("entry", "stop", "lots")) else None
        if r is None:
            missing.append("open_position_risk")
            return done("incomplete", reason="an open position lacks entry/stop/lots or risk_money")
        open_risk += r
    budgets = {
        "daily_loss": policy.max_daily_loss_pct / 100 * base_val - state.realized_loss_today - open_risk,
        "weekly_loss": policy.max_weekly_loss_pct / 100 * base_val - state.realized_loss_week - open_risk,
        "aggregate_open_risk": policy.max_aggregate_open_risk_pct / 100 * base_val - open_risk,
    }
    n_open = len(state.open_positions)
    same = sum(1 for p in state.open_positions if p.get("side") == side)
    for nm, lim, cur in (("max_positions", policy.max_positions, n_open), ("max_same_direction_positions", policy.max_same_direction_positions, same)):
        ok = cur < lim
        checks.append({"name": nm, "ok": ok, "current": cur, "limit": lim})
        if not ok:
            return done("blocked", reason=f"{nm} reached")
    binding = "policy_risk_pct"
    allowed = risk_money
    for nm, b in budgets.items():
        checks.append({"name": nm + "_budget", "remaining_money": round(b, 4)})
        if b < allowed:
            allowed, binding = max(b, 0.0), nm
    if allowed <= 0:
        return done("blocked", reason=f"no remaining {binding} budget", binding_constraint=binding)
    lots_raw = allowed / risk_per_lot
    lots = _round_down(lots_raw, vstep, vmin)
    if vmax is not None:
        lots = min(lots, vmax)
    if lots < vmin - 1e-12 or lots <= 0:
        return done("blocked", reason=f"risk budget {allowed:.2f} {snap.account.value('currency')} is below the risk of the minimum lot ({vmin}); trade not allowed within policy",
                    binding_constraint=binding, min_lot_risk=round(vmin * risk_per_lot, 4))
    # ---- margin
    mreq, mmiss, mnotes = margin_required(snap, lots, entry_price, leverage, fx_rate_margin_to_account)
    margin_info: Dict[str, Any] = {"required": mreq, "notes": mnotes}
    if mreq is None:
        missing += mmiss
        warnings.append("margin could not be computed: " + (", ".join(mmiss) or "unsupported calc mode") + "; margin safety NOT verified")
    elif state.margin_used is not None:
        while True:
            used_after = state.margin_used + mreq
            usage = used_after / state.equity * 100 if state.equity else float("inf")
            level = state.equity / used_after * 100 if used_after > 0 else float("inf")
            ok = usage <= policy.max_margin_usage_pct and level >= policy.min_margin_level_pct
            if ok or lots <= vmin + 1e-12:
                break
            lots = round(max(lots - vstep, vmin), 10)
            binding = "margin_safety"
            mreq, _, _ = margin_required(snap, lots, entry_price, leverage, fx_rate_margin_to_account)
        checks.append({"name": "margin_safety", "ok": ok, "usage_pct_after": round(usage, 4), "margin_level_pct_after": round(level, 2),
                       "limits": {"max_usage_pct": policy.max_margin_usage_pct, "min_level_pct": policy.min_margin_level_pct}})
        margin_info["required"] = mreq
        if not ok:
            return done("blocked", reason="margin safety limits cannot be met even at the minimum lot", binding_constraint="margin_safety")
    else:
        warnings.append("margin_used not provided: margin safety NOT verified")
    actual_risk = lots * risk_per_lot
    # cost-inclusive risk through the shared cost engine (single source of truth)
    pr = calculate_position_risk(snap, lots, entry_price, stop_price, fx_rate_profit_to_account)
    status = "complete" if pr["status"] == "complete" and mreq is not None and state.margin_used is not None and vb.value != "estimated" else "complete_with_gaps"
    if missing and status == "complete":
        status = "complete_with_gaps"
    return done(status, lots=lots, risk_money=round(actual_risk, 4), risk_pct_of_base=round(actual_risk / base_val * 100, 4),
                requested_risk_money=round(risk_money, 4), binding_constraint=binding, stop_distance_price=dist,
                risk_per_lot=round(risk_per_lot, 6), margin=margin_info, currency=snap.account.value("currency"),
                risk_engine_note="lots only ever shrink to satisfy constraints; no escalation after losses exists")


def validate_portfolio(snap: Snapshot, policy: RiskPolicy, state: AccountState, fx_rate_profit_to_account: Optional[float] = None) -> Dict[str, Any]:
    policy.validate()
    out: Dict[str, Any] = {"what": "portfolio_risk", "checks": [], "warnings": [], "context_id": snap.id}
    base_val = state.equity if policy.base == "equity" else state.balance
    if base_val is None:
        return dict(out, status="incomplete", missing_fields=[policy.base], violations=[])
    vpp, _, vmiss, _ = value_per_price_unit(snap, fx_rate_profit_to_account)
    viol: List[str] = []
    open_risk = 0.0
    for p in state.open_positions:
        r = p.get("risk_money")
        if r is None and vpp and all(k in p for k in ("entry", "stop", "lots")):
            r = abs(p["entry"] - p["stop"]) * vpp * p["lots"]
        if r is None:
            return dict(out, status="incomplete", missing_fields=["open_position_risk"] + vmiss, violations=[])
        open_risk += r
    longs = sum(p["lots"] for p in state.open_positions if p.get("side") == "long" and "lots" in p)
    shorts = sum(p["lots"] for p in state.open_positions if p.get("side") == "short" and "lots" in p)
    chk = [("open_risk_pct", open_risk / base_val * 100, policy.max_aggregate_open_risk_pct),
           ("daily_loss_pct", (state.realized_loss_today + open_risk) / base_val * 100, policy.max_daily_loss_pct),
           ("weekly_loss_pct", (state.realized_loss_week + open_risk) / base_val * 100, policy.max_weekly_loss_pct)]
    if state.peak_equity and state.equity is not None:
        chk.append(("drawdown_pct", (state.peak_equity - state.equity) / state.peak_equity * 100, policy.max_drawdown_pct))
    for nm, v, lim in chk:
        ok = v <= lim
        out["checks"].append({"name": nm, "value": round(v, 4), "limit": lim, "ok": ok})
        if not ok:
            viol.append(nm)
    n = len(state.open_positions)
    if n > policy.max_positions:
        viol.append("max_positions")
    out["checks"].append({"name": "positions", "value": n, "limit": policy.max_positions, "ok": n <= policy.max_positions})
    out["exposure"] = {"long_lots": longs, "short_lots": shorts, "net_lots": round(longs - shorts, 8),
                       "note": "all XAUUSD positions are one instrument: same-direction positions are fully correlated exposure; opposite positions are hedged/netted exposure"}
    same = max(sum(1 for p in state.open_positions if p.get("side") == s) for s in ("long", "short")) if n else 0
    if same > policy.max_same_direction_positions:
        viol.append("max_same_direction_positions")
    out["violations"], out["open_risk_money"] = viol, round(open_risk, 4)
    out["status"] = "violation" if viol else "ok"
    return out
