"""The small, strictly-validated tool surface exposed to the model.

Every tool returns {status, data, source, retrieved_at, is_stale, missing_fields, warnings}.
There is deliberately NO tool that authorizes MT5, collects credentials, or trades: MT5 access is enabled by
the user outside the model (cli.py), and this stage is read-only.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from .account_provider import BrokerDocumentationProvider
from .context_models import ToolResult, find_secret_keys, iso, SystemClock
from .context_snapshot import BrokerContextConfig, SnapshotBuilder
from .context_validator import validate
from .cost_engine import calculate_position_risk, calculate_trade_costs
from .mt5_provider import AuthorizationStore, MT5Provider
from .user_profile_provider import ACCOUNT_SCOPE, SYMBOL_SCOPE, UserProfileStore

_NUM = {"type": "number"}
_SYMBOL = {"type": "string", "pattern": r"^[A-Za-z0-9._#@\-]{1,32}$", "description": "Broker symbol name or 'XAUUSD' (prefix/suffix variants are resolved)"}
_COST_COMMON = {
    "symbol": _SYMBOL, "lots": {"type": "number", "exclusiveMinimum": 0}, "entry_price": {"type": "number", "exclusiveMinimum": 0},
    "stop_price": {"type": "number", "exclusiveMinimum": 0},
    "fx_rate_profit_to_account": {"type": "number", "exclusiveMinimum": 0},
}
TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {"name": "get_mt5_connection_status", "description": "Is the user-authorized, read-only MT5 terminal initialized, connected and logged in?",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "get_active_account_context", "description": "Resolved account properties with source/basis/timestamp per field (login withheld).",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "get_symbol_specifications", "description": "Resolved contract specification of the tradable gold symbol.",
     "parameters": {"type": "object", "properties": {"symbol": _SYMBOL}, "additionalProperties": False}},
    {"name": "get_current_spread", "description": "Current spread (points and price units) and its source.",
     "parameters": {"type": "object", "properties": {"symbol": _SYMBOL}, "additionalProperties": False}},
    {"name": "get_broker_context_status", "description": "Overall health: MT5 status, user profile, conflicts, staleness, which calculations are possible.",
     "parameters": {"type": "object", "properties": {"symbol": _SYMBOL}, "additionalProperties": False}},
    {"name": "update_user_account_profile",
     "description": "Store user-provided account/symbol facts (source=user). 'confirmed' only if the user explicitly confirmed the value. "
                    "An account type label (ECN/Standard) never defines a commission amount.",
     "parameters": {"type": "object", "required": ["fields"], "additionalProperties": False, "properties": {
         "fields": {"type": "object", "description": "name -> value; allowed: " + ", ".join(ACCOUNT_SCOPE + SYMBOL_SCOPE)},
         "confirmed": {"type": "boolean"}, "bind_to_connected_account": {"type": "boolean"},
         "acknowledge_conflicts": {"type": "object", "description": "field -> 'mt5' | 'user' (user's explicit choice)"}}}},
    {"name": "calculate_trade_costs",
     "description": "Spread, commission, slippage (scenario), swap and net results from the validated runtime context; incomplete if inputs are missing.",
     "parameters": {"type": "object", "required": ["side", "lots", "entry_price"], "additionalProperties": False, "properties": dict(
         _COST_COMMON, side={"type": "string", "enum": ["long", "short"]}, exit_price={"type": "number", "exclusiveMinimum": 0},
         slippage_price_per_side={"type": "number", "minimum": 0}, nights={"type": "integer", "minimum": 0},
         triple_rollover_nights={"type": "integer", "minimum": 0}, fx_rate_commission_to_account={"type": "number", "exclusiveMinimum": 0},
         scenario={"type": "object", "description": "explicit what-if values for UNKNOWN components only"})}},
    {"name": "calculate_position_risk", "description": "Monetary risk to the stop for a lot size using validated contract specs.",
     "parameters": {"type": "object", "required": ["lots", "entry_price", "stop_price"], "additionalProperties": False, "properties": dict(_COST_COMMON)}},
]
_STATE = {"type": "object", "additionalProperties": False, "properties": {
    "balance": _NUM, "equity": _NUM, "margin_used": {"type": "number", "minimum": 0}, "peak_equity": _NUM,
    "realized_loss_today": {"type": "number", "minimum": 0}, "realized_loss_week": {"type": "number", "minimum": 0},
    "open_positions": {"type": "array", "description": "list of {side, lots, entry, stop, risk_money?}"}}}
_POLICY = {"type": "object", "description": "RiskPolicy overrides; martingale/recovery/escalation keys are rejected"}
TOOL_SCHEMAS += [
    {"name": "calculate_position_size", "description": "Deterministic lot size from risk policy, stop distance, validated contract specs, loss limits and margin.",
     "parameters": {"type": "object", "required": ["side", "entry_price", "stop_price", "account_state"], "additionalProperties": False, "properties": {
         "symbol": _SYMBOL, "side": {"type": "string", "enum": ["long", "short"]}, "entry_price": {"type": "number", "exclusiveMinimum": 0},
         "stop_price": {"type": "number", "exclusiveMinimum": 0}, "account_state": _STATE, "policy": _POLICY,
         "atr": {"type": "number", "exclusiveMinimum": 0}, "atr_ratio": {"type": "number", "exclusiveMinimum": 0},
         "spread_price": {"type": "number", "minimum": 0}, "leverage": {"type": "number", "exclusiveMinimum": 0},
         "fx_rate_profit_to_account": {"type": "number", "exclusiveMinimum": 0}, "fx_rate_margin_to_account": {"type": "number", "exclusiveMinimum": 0}}}},
    {"name": "validate_portfolio_risk", "description": "Check open risk, daily/weekly loss, drawdown, positions and same-instrument exposure against the policy.",
     "parameters": {"type": "object", "required": ["account_state"], "additionalProperties": False, "properties": {
         "symbol": _SYMBOL, "account_state": _STATE, "policy": _POLICY, "fx_rate_profit_to_account": {"type": "number", "exclusiveMinimum": 0}}}},
]
SCHEMA_BY_NAME = {s["name"]: s["parameters"] for s in TOOL_SCHEMAS}


def validate_args(schema: Dict[str, Any], value: Any, path: str = "args") -> List[str]:
    errs: List[str] = []
    t = schema.get("type")
    if t == "object":
        if not isinstance(value, dict):
            return [f"{path}: expected object"]
        props = schema.get("properties", {})
        for r in schema.get("required", []):
            if r not in value:
                errs.append(f"{path}.{r}: required")
        if schema.get("additionalProperties") is False:
            for k in value:
                if k not in props:
                    errs.append(f"{path}.{k}: unknown argument")
        for k, v in value.items():
            if k in props:
                errs += validate_args(props[k], v, f"{path}.{k}")
        return errs
    if t == "array":
        if not isinstance(value, list):
            return [f"{path}: expected array"]
        if len(value) > 50:
            return [f"{path}: too many items"]
        for n, v in enumerate(value):
            if not isinstance(v, dict) or set(v) - {"side", "lots", "entry", "stop", "risk_money"}:
                errs.append(f"{path}[{n}]: must be an object with only side, lots, entry, stop, risk_money")
        return errs
    if t == "string":
        if not isinstance(value, str):
            return [f"{path}: expected string"]
        if "pattern" in schema and not re.fullmatch(schema["pattern"], value):
            errs.append(f"{path}: does not match {schema['pattern']}")
    elif t in ("number", "integer"):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or (t == "integer" and int(value) != value):
            return [f"{path}: expected {t}"]
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{path}: must be >= {schema['minimum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errs.append(f"{path}: must be > {schema['exclusiveMinimum']}")
    elif t == "boolean":
        if not isinstance(value, bool):
            return [f"{path}: expected boolean"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: must be one of {schema['enum']}")
    return errs


class BrokerContextTools:
    def __init__(self, builder: SnapshotBuilder, provider: MT5Provider, store: UserProfileStore, clock=None):
        self.builder, self.provider, self.store, self.clock = builder, provider, store, clock or SystemClock()

    # --------------------------------------------------------------- dispatch
    def call(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        args = arguments or {}
        now = iso(self.clock())
        if name not in SCHEMA_BY_NAME:
            return ToolResult("invalid_arguments", None, "tools", now, warnings=[f"unknown tool {name!r}"]).to_dict()
        secrets = find_secret_keys(args)
        if secrets:
            return ToolResult("invalid_arguments", None, "tools", now,
                              warnings=["credential-like arguments are never accepted: " + ", ".join(secrets)]).to_dict()
        errs = validate_args(SCHEMA_BY_NAME[name], args)
        if errs:
            return ToolResult("invalid_arguments", None, "tools", now, warnings=errs).to_dict()
        try:
            return getattr(self, "_" + name)(**args).to_dict()
        except Exception as e:  # noqa: BLE001  never crash the application
            return ToolResult("error", None, "tools", now, warnings=[f"{type(e).__name__}: {e}"]).to_dict()

    # ------------------------------------------------------------------ tools
    def _get_mt5_connection_status(self) -> ToolResult:
        st = self.provider.connection_status()
        return ToolResult(st["status"], st, "mt5", st["retrieved_at"], False, [], [st["error"]] if st.get("error") else [])

    def _get_active_account_context(self) -> ToolResult:
        s = self.builder.build()
        srcs = sorted({s.account.get(n).source.value for n in s.account.names if s.account.known(n)})
        st = s.staleness["account"]
        return ToolResult("ok" if s.account.known("currency") and not s.unresolved_material_conflicts else "partial",
                          {"account": s.account.to_dict(), "conflicts": [c.to_dict() for c in s.conflicts if c.scope == "account"],
                           "account_changed_since_profile_bound": s.account_changed, "applies_to": "current account, not historical market facts"},
                          "+".join(srcs) or "unknown", s.created_at, bool(st["used"] and st["stale"]), s.account.missing(), s.warnings)

    def _get_symbol_specifications(self, symbol: Optional[str] = None) -> ToolResult:
        s = self.builder.build(symbol)
        st = s.staleness["symbol_specification"]
        status = "ambiguous" if s.symbol_candidates and not s.symbol.known("contract_size") and not s.symbol.known("point") else \
                 ("ok" if not s.symbol.missing(("point", "tick_size", "tick_value", "contract_size")) else "partial")
        return ToolResult(status, {"symbol": s.symbol_name, "specifications": s.symbol.to_dict(),
                                   "candidates": s.symbol_candidates, "conflicts": [c.to_dict() for c in s.conflicts if c.scope == "symbol"]},
                          "mt5_live" if s.mt5_status.get("status") == "ok" else "resolved", s.created_at,
                          bool(st["used"] and st["stale"]), s.symbol.missing(), s.warnings)

    def _get_current_spread(self, symbol: Optional[str] = None) -> ToolResult:
        s = self.builder.build(symbol)
        q = s.staleness["quote"]
        sp, pp = s.symbol.get("spread_points"), s.symbol.get("spread_price")
        if not (sp.known or pp.known):
            return ToolResult("unavailable", {"symbol": s.symbol_name}, "none", s.created_at, False, ["spread_points", "spread_price"],
                              s.warnings + ["spread could not be retrieved; use an explicitly labelled scenario"])
        return ToolResult("ok" if pp.basis.value in ("verified_live", "user_confirmed") else "partial",
                          {"symbol": s.symbol_name, "spread_points": sp.to_dict(), "spread_price": pp.to_dict(), "quote": s.quote},
                          pp.source.value, pp.retrieved_at, bool(q["used"] and q["stale"]) or pp.note is not None and "STALE" in (pp.note or ""), [], s.warnings)

    def _get_broker_context_status(self, symbol: Optional[str] = None) -> ToolResult:
        s = self.builder.build(symbol)
        checks = {c: validate(s, c) for c in ("position_risk", "trade_costs")}
        data = {"mt5": s.mt5_status, "user_profile": self.store.summary(), "conflicts": [c.to_dict() for c in s.conflicts],
                "unresolved_material_conflicts": [c.field for c in s.unresolved_material_conflicts], "staleness": s.staleness,
                "calculations": {k: {"ok": v.ok, "blocking_missing": v.blocking_missing, "optional_missing": v.optional_missing,
                                     "blocking_conflicts": v.blocking_conflicts} for k, v in checks.items()},
                "snapshot_id": s.id}
        return ToolResult("ok" if checks["position_risk"].ok else "partial", data, "context_resolver", s.created_at, s.is_stale,
                          sorted(set(checks["trade_costs"].blocking_missing + checks["trade_costs"].optional_missing)), s.warnings)

    def _update_user_account_profile(self, fields: Dict[str, Any], confirmed: bool = False, bind_to_connected_account: bool = False,
                                     acknowledge_conflicts: Optional[Dict[str, str]] = None) -> ToolResult:
        res = self.store.update(fields, confirmed=confirmed)
        warn = list(res["warnings"])
        for f, choice in (acknowledge_conflicts or {}).items():
            try:
                self.store.acknowledge_conflict(f, choice)
            except ValueError as e:
                res["rejected"][f"acknowledge:{f}"] = str(e)
        if bind_to_connected_account:
            self.provider.connection_status()  # refreshes the (hashed) login reference
            ref = self.provider.last_login_ref
            if ref:
                self.store.bind_account(ref)
            else:
                warn.append("no MT5 account connected; profile not bound to an account")
        snap = self.builder.build()
        data = {"changes": res["changes"], "rejected": res["rejected"], "profile": self.store.summary(),
                "conflicts_after_update": [c.to_dict() for c in snap.conflicts], "mt5_status": snap.mt5_status.get("status")}
        still_unknown = [n for n in ("commission_amount", "commission_basis", "commission_currency") if not snap.account.known(n)]
        if still_unknown:
            warn.append("still unknown after this update: " + ", ".join(still_unknown))
        return ToolResult("ok" if not res["rejected"] else "partial", data, "user", iso(self.clock()), False, still_unknown, warn + snap.warnings)

    def _wrap_engine(self, snap, res: Dict[str, Any]) -> ToolResult:
        return ToolResult(res["status"], res, "runtime_context", snap.created_at, snap.is_stale, res.get("missing_fields", []), res.get("warnings", []))

    def _calculate_trade_costs(self, side: str, lots: float, entry_price: float, symbol: Optional[str] = None, stop_price: Optional[float] = None,
                               exit_price: Optional[float] = None, slippage_price_per_side: Optional[float] = None, nights: int = 0,
                               triple_rollover_nights: int = 0, fx_rate_profit_to_account: Optional[float] = None,
                               fx_rate_commission_to_account: Optional[float] = None, scenario: Optional[Dict[str, Any]] = None) -> ToolResult:
        snap = self.builder.build(symbol)
        return self._wrap_engine(snap, calculate_trade_costs(snap, side, lots, entry_price, stop_price, exit_price, slippage_price_per_side, nights,
                                                             triple_rollover_nights, fx_rate_profit_to_account, fx_rate_commission_to_account, scenario))

    def _calculate_position_risk(self, lots: float, entry_price: float, stop_price: float, symbol: Optional[str] = None,
                                 fx_rate_profit_to_account: Optional[float] = None) -> ToolResult:
        snap = self.builder.build(symbol)
        return self._wrap_engine(snap, calculate_position_risk(snap, lots, entry_price, stop_price, fx_rate_profit_to_account))


def _risk_tool(self, kind, args):
    from .risk_engine import AccountState, RiskPolicy, calculate_position_size, validate_portfolio
    now = iso(self.clock())
    try:
        policy = RiskPolicy.from_dict(args.get("policy"))
        raw = dict(args["account_state"])
        ops = raw.get("open_positions") or []
        if not isinstance(ops, list):
            return ToolResult("invalid_arguments", None, "tools", now, warnings=["account_state.open_positions must be an array"])
        raw["open_positions"] = ops
        state = AccountState.from_dict(raw)
    except (ValueError, TypeError) as e:
        return ToolResult("invalid_arguments", None, "tools", now, warnings=[str(e)])
    snap = self.builder.build(args.get("symbol"))
    if kind == "size":
        res = calculate_position_size(snap, policy, state, args["side"], args["entry_price"], args["stop_price"], args.get("atr_ratio"),
                                      args.get("spread_price"), args.get("atr"), args.get("leverage"), args.get("fx_rate_profit_to_account"),
                                      args.get("fx_rate_margin_to_account"))
    else:
        res = validate_portfolio(snap, policy, state, args.get("fx_rate_profit_to_account"))
    return ToolResult(res["status"], res, "runtime_context+user_account_state", snap.created_at, snap.is_stale, res.get("missing_fields", []), res.get("warnings", []))


BrokerContextTools._calculate_position_size = lambda self, **a: _risk_tool(self, "size", a)
BrokerContextTools._validate_portfolio_risk = lambda self, **a: _risk_tool(self, "portfolio", a)


def create_tools(home: Optional[str] = None, mt5_module: Any = None, clock=None, config: Optional[BrokerContextConfig] = None,
                 authorization: Optional[AuthorizationStore] = None, docs: Optional[BrokerDocumentationProvider] = None) -> BrokerContextTools:
    home = home or os.environ.get("BROKER_CONTEXT_HOME") or os.path.join(os.path.expanduser("~"), ".broker_context")
    clock = clock or SystemClock()
    auth = authorization or AuthorizationStore(os.path.join(home, "mt5_authorization.json"), clock)
    store = UserProfileStore(os.path.join(home, "user_profile.json"), clock)
    provider = MT5Provider(auth, mt5_module, clock)
    builder = SnapshotBuilder(provider, store, docs, clock, config)
    return BrokerContextTools(builder, provider, store, clock)
