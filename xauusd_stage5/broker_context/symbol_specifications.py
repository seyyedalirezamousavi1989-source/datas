"""Gold symbol discovery/validation and MT5 symbol_info -> normalized fields."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .context_models import SYMBOL_FIELDS, Basis, Field, FieldSet, Source

CORE = "XAUUSD"
CALC_MODES = {0: "forex", 1: "futures", 2: "cfd", 3: "cfd_index", 4: "cfd_leverage", 5: "forex_no_leverage",
              32: "exchange_stocks", 33: "exchange_futures", 34: "exchange_futures_forts", 35: "exchange_bonds",
              36: "exchange_stocks_moex", 37: "exchange_bonds_moex", 38: "serv_collateral"}
EXEC_MODES = {0: "request", 1: "instant", 2: "market", 3: "exchange"}
SWAP_MODES = {0: "disabled", 1: "points", 2: "currency_symbol", 3: "currency_margin", 4: "currency_deposit",
              5: "interest_current", 6: "interest_open", 7: "reopen_current", 8: "reopen_bid"}
FILLING_FLAGS = {1: "FOK", 2: "IOC"}  # symbol filling_mode is a bit mask; RETURN is not encoded in it
POSITIVE_ONLY = ("point", "tick_size", "tick_value", "contract_size", "volume_min", "volume_max", "volume_step")


def split_symbol(name: str) -> Optional[Tuple[str, str]]:
    """('prefix','suffix') if the name contains XAUUSD, else None. Case-insensitive."""
    i = name.upper().find(CORE)
    return None if i < 0 else (name[:i], name[i + len(CORE):])


@dataclass
class Candidate:
    name: str
    score: int
    reason: str


def gold_candidates(names: Sequence[str], requested: str = CORE) -> List[Candidate]:
    out: List[Candidate] = []
    for n in names:
        sp = split_symbol(n)
        if n == requested:
            out.append(Candidate(n, 100, "exact requested name"))
        elif n.upper() == requested.upper():
            out.append(Candidate(n, 95, "case-insensitive requested name"))
        elif sp is not None:
            pre, suf = sp
            if len(pre) <= 4 and len(suf) <= 6:
                out.append(Candidate(n, 60 if n.upper() != CORE else 90, f"prefix '{pre}' / suffix '{suf}' around {CORE}"))
        elif n.upper().startswith("GOLD") and len(n) <= 10:
            out.append(Candidate(n, 20, "alias 'GOLD' (needs verification)"))
    return sorted(out, key=lambda c: (-c.score, c.name))


@dataclass
class GoldCheck:
    ok: bool
    verified: bool
    reasons: List[str] = field(default_factory=list)


def validate_gold(name: str, info: Any) -> GoldCheck:
    """Is this the intended XAU/USD instrument? verified=True only with base/profit currency from MT5."""
    reasons: List[str] = []
    base = (getattr(info, "currency_base", None) or "").upper() or None
    profit = (getattr(info, "currency_profit", None) or "").upper() or None
    desc = ((getattr(info, "description", None) or "") + " " + (getattr(info, "path", None) or "")).lower()
    if base is not None and base != "XAU":
        return GoldCheck(False, False, [f"base currency is {base}, expected XAU"])
    if profit is not None and profit != "USD":
        return GoldCheck(False, False, [f"profit currency is {profit}, expected USD (a different gold cross)"])
    if base == "XAU" and profit == "USD":
        return GoldCheck(True, True, ["base currency XAU and profit currency USD reported by MT5"])
    if split_symbol(name) is None and not any(w in desc for w in ("gold", "xau")):
        return GoldCheck(False, False, ["name and description do not identify gold"])
    reasons.append("currency base/profit not reported by MT5; identified by name/description only")
    return GoldCheck(True, False, reasons)


def _mapped(table: Dict[int, str], v: Any) -> Optional[str]:
    if v is None:
        return None
    return table.get(int(v), f"unknown_code_{int(v)}")


def parse_symbol_info(name: str, info: Any, now_iso: str, quote: Optional[Dict[str, Any]] = None
                      ) -> Tuple[FieldSet, Dict[str, List[str]], List[str]]:
    """Normalize an MT5 SymbolInfo-like object. Missing/invalid attributes stay None."""
    fs = FieldSet(SYMBOL_FIELDS)
    warnings: List[str] = []
    avail = {"available": [], "unavailable": []}

    def put(fname: str, value: Any, note: Optional[str] = None, basis: Basis = Basis.VERIFIED_LIVE) -> None:
        if value is None:
            fs.set(fname, Field(None, Source.UNKNOWN, Basis.UNKNOWN, now_iso, note or "not provided by MT5"))
            avail["unavailable"].append(fname)
        else:
            fs.set(fname, Field(value, Source.MT5_LIVE, basis, now_iso, note))
            avail["available"].append(fname)

    def raw(attr: str) -> Any:
        return getattr(info, attr, None)

    put("symbol", name)
    put("digits", raw("digits"))
    put("point", raw("point"))
    put("tick_size", raw("trade_tick_size"))
    put("tick_value", raw("trade_tick_value"))
    put("contract_size", raw("trade_contract_size"))
    put("volume_min", raw("volume_min"))
    put("volume_max", raw("volume_max"))
    put("volume_step", raw("volume_step"))
    put("stops_level", raw("trade_stops_level"))
    put("freeze_level", raw("trade_freeze_level"))
    for f_ in POSITIVE_ONLY:
        v = fs.value(f_)
        if v is not None and not (float(v) > 0):
            put(f_, None, f"MT5 returned {v!r} (not a usable positive value)")
            warnings.append(f"{f_} returned by MT5 is not positive and was treated as unavailable")
    put("calc_mode", _mapped(CALC_MODES, raw("trade_calc_mode")))
    put("execution_mode", _mapped(EXEC_MODES, raw("trade_exemode")))
    fm = raw("filling_mode")
    put("filling_modes", None if fm is None else [n for b, n in FILLING_FLAGS.items() if int(fm) & b],
        "bit mask FOK=1, IOC=2; RETURN is not encoded in this property")
    put("swap_long", raw("swap_long"))
    put("swap_short", raw("swap_short"))
    put("swap_mode", _mapped(SWAP_MODES, raw("swap_mode")))
    put("swap_rollover3days", raw("swap_rollover3days"), "MT5 weekday code of the triple-swap day")
    put("sessions", None, "trading sessions are not exposed by the MetaTrader5 Python API symbol_info")
    sp = raw("spread")
    put("spread_points", sp, "symbol_info.spread (points); float spread" if raw("spread_float") else "symbol_info.spread (points); fixed spread")
    put("currency_base", raw("currency_base"))
    put("currency_profit", raw("currency_profit"))
    put("currency_margin", raw("currency_margin"))
    put("description", raw("description"))
    put("path", raw("path"))
    put("visible", raw("visible"))
    # spread in price units: prefer the live quote, else points * point
    if quote and quote.get("spread_price") is not None:
        fs.set("spread_price", Field(quote["spread_price"], Source.MT5_LIVE, Basis.VERIFIED_LIVE, quote.get("retrieved_at", now_iso), "ask - bid from symbol_info_tick"))
        avail["available"].append("spread_price")
    elif fs.known("spread_points") and fs.known("point"):
        fs.set("spread_price", Field(float(fs.value("spread_points")) * float(fs.value("point")), Source.MT5_LIVE, Basis.VERIFIED_LIVE, now_iso, "symbol_info.spread * point (no live quote)"))
        avail["available"].append("spread_price")
    else:
        put("spread_price", None, "no quote and no spread/point")
    if fs.value("visible") is False:
        warnings.append("symbol is not visible in Market Watch; quotes may be unavailable (read-only mode does not change visibility)")
    return fs, avail, warnings
