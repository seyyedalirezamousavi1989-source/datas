"""Dynamic cost & risk engine. Takes a validated runtime Snapshot - no fixed broker values anywhere.

Money is produced ONLY when the required specification is known. Otherwise the result says
"incomplete" and lists the missing inputs. Every component is labelled with its basis
(verified_live / user_confirmed / broker_documentation / user_unconfirmed / estimated / unknown).

Conventions (also in docs): prices are BID-chart prices; a round trip crosses the spread once;
slippage is a per-side price scenario supplied by the caller (never inferred); swap component is a COST
(positive = you pay, negative = you receive); vpp = money per 1.0 price unit per 1 lot in ACCOUNT currency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .context_models import BASIS_RANK, TRUSTED_BASES, Basis, Field, weakest
from .context_snapshot import Snapshot
from .context_validator import ValidationReport, validate


@dataclass
class Component:
    name: str
    value: Optional[float]
    currency: Optional[str]
    basis: Basis
    formula: str
    notes: List[str] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"value": None if self.value is None else round(self.value, 8), "currency": self.currency,
                "basis": self.basis.value, "formula": self.formula, "notes": self.notes, "detail": self.detail}


def value_per_price_unit(snap: Snapshot, fx_profit_to_account: Optional[float] = None
                         ) -> Tuple[Optional[float], Basis, List[str], List[str]]:
    """Money (account currency) per 1.0 price unit per 1 lot."""
    s, a = snap.symbol, snap.account
    tv, ts = s.get("tick_value"), s.get("tick_size")
    if tv.known and ts.known:
        return (float(tv.value) / float(ts.value), weakest(tv.basis, ts.basis), [],
                ["vpp = tick_value / tick_size (MT5 tick_value is expressed in the account currency)"])
    cs, cp, acc = s.get("contract_size"), s.get("currency_profit"), a.get("currency")
    if cs.known and cp.known and acc.known:
        if str(cp.value).upper() == str(acc.value).upper():
            return float(cs.value), weakest(cs.basis, cp.basis, acc.basis), [], ["vpp = contract_size (profit currency == account currency)"]
        if fx_profit_to_account:
            return (float(cs.value) * float(fx_profit_to_account), Basis.ESTIMATED, [],
                    [f"vpp = contract_size x supplied FX rate {cp.value}->{acc.value}; the rate is an estimate"])
        return None, Basis.UNKNOWN, ["fx_rate_profit_to_account"], [f"profit currency {cp.value} differs from account currency {acc.value}; conversion rate required"]
    miss = [n for n in ("tick_value", "tick_size") if not s.known(n)] if not (cs.known and cp.known) else []
    if not (cs.known and cp.known):
        miss += [n for n in ("contract_size", "currency_profit") if not s.known(n)]
    if not acc.known:
        miss.append("account_currency")
    return None, Basis.UNKNOWN, sorted(set(miss)), ["cannot convert price to money without tick value/size or contract size + currencies"]


def _status(components: List[Component], snap: Snapshot, vpp_ok: bool) -> str:
    if not vpp_ok:
        return "incomplete"
    if any(c.value is None for c in components):
        return "partial"
    if snap.is_stale or any(c.basis not in TRUSTED_BASES for c in components):
        return "complete_with_estimates"
    return "complete"


def _incomplete(snap: Snapshot, rep: ValidationReport, extra_missing: List[str], what: str) -> Dict[str, Any]:
    miss = sorted(set(rep.blocking_missing + extra_missing))
    why = []
    if rep.blocking_conflicts:
        why.append("unresolved conflicts that affect the result need user confirmation: " + ", ".join(rep.blocking_conflicts))
    if miss:
        why.append("missing inputs: " + ", ".join(miss))
    return {"status": "incomplete", "what": what, "currency": snap.account.value("currency"), "components": {}, "totals": None,
            "missing_fields": miss, "blocking_conflicts": rep.blocking_conflicts, "warnings": rep.warnings + why,
            "context_id": snap.id, "context_stale": snap.is_stale,
            "note": "No monetary value was produced because required inputs are missing or in unresolved conflict."}


def _spread_component(snap: Snapshot, vpp, vb, scenario, ccy, lots) -> Component:
    f = snap.symbol.get("spread_price")
    notes: List[str] = []
    price, basis = None, Basis.UNKNOWN
    if f.known:
        price, basis = float(f.value), f.basis
        if f.note:
            notes.append(f.note)
        if price == 0:
            notes.append("zero spread reported: plausible for raw-spread accounts, otherwise a stale/invalid quote - verify")
    elif scenario and scenario.get("spread_price") is not None:
        price, basis = float(scenario["spread_price"]), Basis.ESTIMATED
        notes.append("scenario spread (no live/confirmed spread available)")
    if f.known and scenario and scenario.get("spread_price") is not None and f.basis in TRUSTED_BASES:
        notes.append("scenario spread ignored: a verified spread exists")
    pts, ts = snap.symbol.value("spread_points"), snap.symbol.value("tick_size")
    det = {"spread_price": price, "spread_points": pts, "spread_ticks": None if (price is None or not ts) else price / ts}
    if price is None:
        return Component("spread", None, ccy, Basis.UNKNOWN, "spread_price x vpp x lots", ["spread unavailable"], det)
    return Component("spread", price * vpp * lots, ccy, weakest(basis, vb), "spread_price x vpp x lots (one round-trip crossing)", notes, det)


def _commission_component(snap: Snapshot, lots, scenario, fx_c2a, ccy) -> Component:
    a = snap.account
    model, amt, cur, basis_f = a.get("commission_model"), a.get("commission_amount"), a.get("commission_currency"), a.get("commission_basis")
    notes: List[str] = []
    sc = (scenario or {}).get("commission")
    trusted_known = amt.known and basis_f.known and cur.known and weakest(amt.basis, basis_f.basis, cur.basis) in TRUSTED_BASES
    if model.known and model.value == "none":
        return Component("commission", 0.0, ccy, model.basis, "no commission (user/broker-stated model 'none')", notes)
    use_amt, use_basis_name, use_cur, b = amt.value, basis_f.value, cur.value, weakest(amt.basis, basis_f.basis, cur.basis)
    if sc and not trusted_known:
        use_amt, use_basis_name, use_cur, b = sc.get("amount"), sc.get("basis"), sc.get("currency"), Basis.ESTIMATED
        notes.append("commission taken from the supplied SCENARIO (not a verified or confirmed value)")
    elif sc and trusted_known:
        notes.append("scenario commission ignored: a confirmed commission exists")
    if use_amt is None or use_basis_name is None or use_cur is None:
        at = a.value("account_type")
        n = "commission unknown: amount/basis/currency not available from MT5 (MT5 does not expose commission) or confirmed by the user"
        if at:
            n += f". The account label {at!r} does not define a commission amount and was not used to guess one"
        return Component("commission", None, ccy, Basis.UNKNOWN, "amount x lots x sides", [n],
                         {"missing": [x for x, v in (("commission_amount", use_amt), ("commission_basis", use_basis_name), ("commission_currency", use_cur)) if v is None]})
    if use_basis_name == "per_lot_per_side":
        native, sides = float(use_amt) * lots * 2, 2
        formula = "amount x lots x 2 sides"
    elif use_basis_name == "per_lot_round_turn":
        native, sides = float(use_amt) * lots, 1
        formula = "amount x lots (amount already covers entry+exit)"
    elif use_basis_name == "flat_per_trade_round_turn":
        native, sides = float(use_amt), 1
        formula = "flat amount per round trip"
    else:
        return Component("commission", None, ccy, Basis.UNKNOWN, "unsupported basis", [f"commission basis {use_basis_name!r} is not supported"], {})
    det = {"native_amount": native, "native_currency": use_cur, "sides_charged": sides}
    if str(use_cur).upper() == str(ccy).upper():
        return Component("commission", native, ccy, b, formula, notes, det)
    if fx_c2a:
        return Component("commission", native * float(fx_c2a), ccy, Basis.ESTIMATED, formula + f" x supplied FX {use_cur}->{ccy}",
                         notes + ["converted with a caller-supplied FX rate (estimate)"], det)
    return Component("commission", None, ccy, Basis.UNKNOWN, formula, [f"commission is in {use_cur} but the account currency is {ccy}; fx_rate_commission_to_account required"],
                     dict(det, missing=["fx_rate_commission_to_account"]))


def _swap_component(snap: Snapshot, side: str, lots, nights, triple, vpp, vb, fx_p2a, ccy) -> Component:
    s = snap.symbol
    sw = s.get("swap_long" if side == "long" else "swap_short")
    mode, point = s.get("swap_mode"), s.get("point")
    eff = nights + 2 * triple
    if not sw.known or not mode.known:
        return Component("swap", None, ccy, Basis.UNKNOWN, "swap per night x effective nights", ["swap rate/mode unavailable"], {"missing": [n for n in ("swap_long" if side == "long" else "swap_short", "swap_mode") if not s.known(n)]})
    b = weakest(sw.basis, mode.basis)
    m = mode.value
    notes = [f"effective nights = nights + 2 x triple-rollover nights = {eff}", "value is a COST: positive = you pay, negative = you receive"]
    if m == "points":
        if not point.known or vpp is None:
            return Component("swap", None, ccy, Basis.UNKNOWN, "swap_points x point x vpp x lots x nights", ["point or value-per-price-unit unavailable"], {})
        per_night = float(sw.value) * float(point.value) * vpp * lots
        b = weakest(b, point.basis, vb)
    elif m == "currency_deposit":
        per_night = float(sw.value) * lots
    elif m in ("currency_symbol", "currency_margin"):
        cur = s.value("currency_profit" if m == "currency_symbol" else "currency_margin")
        if cur and str(cur).upper() == str(ccy).upper():
            per_night = float(sw.value) * lots
        elif fx_p2a:
            per_night, b = float(sw.value) * lots * float(fx_p2a), Basis.ESTIMATED
            notes.append("converted with a supplied FX rate (estimate)")
        else:
            return Component("swap", None, ccy, Basis.UNKNOWN, "swap x lots x nights", [f"swap is in {cur} (not the account currency); fx rate required"], {})
    else:
        return Component("swap", None, ccy, Basis.UNKNOWN, "unsupported swap mode", [f"swap mode {m!r} is not supported (interest-based modes need price/rate data)"], {})
    return Component("swap", -per_night * eff, ccy, b, "-(swap per night) x effective nights", notes, {"per_night_credit": per_night})


def calculate_position_risk(snap: Snapshot, lots: float, entry_price: float, stop_price: float,
                            fx_rate_profit_to_account: Optional[float] = None) -> Dict[str, Any]:
    rep = validate(snap, "position_risk", lots)
    vpp, vb, miss, notes = value_per_price_unit(snap, fx_rate_profit_to_account)
    if vpp is None or rep.blocking_conflicts or rep.blocking_missing:
        return _incomplete(snap, rep, miss, "position_risk")
    ccy = snap.account.value("currency")
    dist = abs(float(entry_price) - float(stop_price))
    pt, ts = snap.symbol.value("point"), snap.symbol.value("tick_size")
    comp = Component("risk_to_stop", dist * vpp * lots, ccy, vb, "|entry - stop| x vpp x lots (excludes costs)", notes,
                     {"stop_distance_price": dist, "stop_distance_points": None if not pt else dist / pt,
                      "stop_distance_ticks": None if not ts else dist / ts, "value_per_price_unit_per_lot": vpp})
    st = _status([comp], snap, True)
    return {"status": st, "what": "position_risk", "currency": ccy, "components": {"risk_to_stop": comp.to_dict()},
            "missing_fields": [], "warnings": rep.warnings, "context_id": snap.id, "context_stale": snap.is_stale,
            "assumptions": ["risk excludes spread, commission, slippage and swap; use calculate_trade_costs for those"]}


def calculate_trade_costs(snap: Snapshot, side: str, lots: float, entry_price: float, stop_price: Optional[float] = None,
                          exit_price: Optional[float] = None, slippage_price_per_side: Optional[float] = None,
                          nights: int = 0, triple_rollover_nights: int = 0,
                          fx_rate_profit_to_account: Optional[float] = None, fx_rate_commission_to_account: Optional[float] = None,
                          scenario: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    side = side.lower()
    if side not in ("long", "short"):
        raise ValueError("side must be 'long' or 'short'")
    calc = "swap" if nights else "trade_costs"
    rep = validate(snap, "trade_costs" if not nights else "swap", lots)
    if nights:
        rep2 = validate(snap, "trade_costs", lots)
        rep.optional_missing = sorted(set(rep.optional_missing + rep2.optional_missing))
        rep.blocking_conflicts = sorted(set(rep.blocking_conflicts + rep2.blocking_conflicts))
        rep.blocking_missing = sorted(set(rep.blocking_missing + rep2.blocking_missing))
    vpp, vb, miss, vnotes = value_per_price_unit(snap, fx_rate_profit_to_account)
    if vpp is None or rep.blocking_conflicts or rep.blocking_missing:
        return _incomplete(snap, rep, miss, "trade_costs")
    ccy = snap.account.value("currency")
    comps: List[Component] = [_spread_component(snap, vpp, vb, scenario, ccy, lots),
                              _commission_component(snap, lots, scenario, fx_rate_commission_to_account, ccy)]
    sl_sc = slippage_price_per_side if slippage_price_per_side is not None else (scenario or {}).get("slippage_price_per_side")
    if sl_sc is not None:
        comps.append(Component("slippage", float(sl_sc) * 2 * vpp * lots, ccy, Basis.ESTIMATED, "slippage_price_per_side x 2 sides x vpp x lots",
                               ["slippage is a caller-supplied scenario, not a measured value"], {"slippage_price_per_side": float(sl_sc)}))
    else:
        comps.append(Component("slippage", None, ccy, Basis.UNKNOWN, "slippage_price_per_side x 2 x vpp x lots",
                               ["slippage is not provided and cannot be derived from account/symbol specifications"], {"missing": ["slippage_price_per_side"]}))
    if nights:
        comps.append(_swap_component(snap, side, lots, nights, triple_rollover_nights, vpp, vb, fx_rate_profit_to_account, ccy))
    known = [c for c in comps if c.value is not None]
    unknown = [c.name for c in comps if c.value is None]
    total_known = sum(c.value for c in known)
    out: Dict[str, Any] = {"status": _status(comps, snap, True), "what": "trade_costs", "currency": ccy, "side": side, "lots": lots,
                           "components": {c.name: c.to_dict() for c in comps},
                           "totals": {"known_costs": round(total_known, 8), "unknown_components": unknown,
                                      "is_total_complete": not unknown,
                                      "note": "known_costs sums only components with values; it is NOT the full cost while unknown_components is non-empty"},
                           "value_per_price_unit_per_lot": {"value": vpp, "basis": vb.value, "notes": vnotes}}
    miss_out = sorted({m for c in comps for m in c.detail.get("missing", [])})
    if stop_price is not None:
        risk = abs(float(entry_price) - float(stop_price)) * vpp * lots
        out["position_risk"] = {"risk_to_stop": round(risk, 8), "basis": vb.value,
                                "worst_case_loss_at_stop": round(risk + total_known, 8) if not unknown else None,
                                "worst_case_note": "available only when every cost component is known"}
        if exit_price is not None and risk > 0:
            direction = 1 if side == "long" else -1
            gross = direction * (float(exit_price) - float(entry_price)) * vpp * lots
            net = gross - total_known
            out["net"] = {"gross_pnl": round(gross, 8), "net_pnl_after_known_costs": round(net, 8),
                          "net_R_after_known_costs": round(net / risk, 6), "R_definition": "money at risk to the stop, excluding costs",
                          "complete": not unknown}
    elif exit_price is not None:
        direction = 1 if side == "long" else -1
        gross = direction * (float(exit_price) - float(entry_price)) * vpp * lots
        out["net"] = {"gross_pnl": round(gross, 8), "net_pnl_after_known_costs": round(gross - total_known, 8), "net_R_after_known_costs": None,
                      "complete": not unknown, "note": "stop_price not given: R not defined"}
    out["missing_fields"] = miss_out
    out["warnings"] = rep.warnings + ([f"unknown cost components: {', '.join(unknown)}"] if unknown else [])
    out["assumptions"] = ["prices are bid-chart prices; one spread crossing per round trip", "slippage and scenario values are estimates",
                          "swap is a cost (positive = pay)"]
    out["context_id"], out["context_stale"] = snap.id, snap.is_stale
    return out
