"""User-provided account/symbol information (source = user) with timestamps and verification status.

A generic label such as "ECN" or "Standard" is stored as an account_type LABEL only. It never creates
a commission amount. Passwords/tokens/secrets are rejected. The MT5 login is not accepted here.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional

from .context_models import (ACCOUNT_FIELDS, COMMISSION_BASES, COMMISSION_MODELS, SYMBOL_FIELDS, Basis, Field,
                             FieldSet, Source, SystemClock, find_secret_keys, iso)

ACCOUNT_SCOPE = ("broker_name", "server", "account_type", "currency", "leverage", "trade_mode", "margin_mode",
                 "commission_model", "commission_amount", "commission_currency", "commission_basis", "swap_free")
SYMBOL_SCOPE = ("symbol", "digits", "point", "tick_size", "tick_value", "contract_size", "volume_min", "volume_max",
                "volume_step", "stops_level", "freeze_level", "swap_long", "swap_short", "swap_mode",
                "currency_base", "currency_profit", "currency_margin", "spread_points", "calc_mode")
COMMISSION_FIELDS = ("commission_model", "commission_amount", "commission_currency", "commission_basis")


def _pos_float(v: Any) -> float:
    x = float(v)
    if not x > 0:
        raise ValueError("must be > 0")
    return x


def _nonneg_float(v: Any) -> float:
    x = float(v)
    if x < 0:
        raise ValueError("must be >= 0")
    return x


def _ccy(v: Any) -> str:
    s = str(v).strip().upper()
    if not re.fullmatch(r"[A-Z]{3}", s):
        raise ValueError("must be a 3-letter currency code")
    return s


def _text(maxlen: int = 64) -> Callable[[Any], str]:
    def f(v: Any) -> str:
        s = str(v).strip()
        if not s or len(s) > maxlen:
            raise ValueError(f"must be 1-{maxlen} characters")
        return s
    return f


def _enum(options) -> Callable[[Any], str]:
    def f(v: Any) -> str:
        s = str(v).strip().lower()
        if s not in options:
            raise ValueError(f"must be one of {list(options)}")
        return s
    return f


def _bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    raise ValueError("must be true or false")


VALIDATORS: Dict[str, Callable[[Any], Any]] = {
    "broker_name": _text(), "server": _text(), "account_type": _text(), "currency": _ccy,
    "leverage": lambda v: int(_pos_float(v)), "trade_mode": _enum(("demo", "contest", "real")),
    "margin_mode": _enum(("retail_netting", "exchange", "retail_hedging")),
    "commission_model": _enum(COMMISSION_MODELS), "commission_amount": _nonneg_float, "commission_currency": _ccy,
    "commission_basis": _enum(COMMISSION_BASES), "swap_free": _bool,
    "symbol": lambda v: _symbol(v), "digits": lambda v: _digits(v), "point": _pos_float, "tick_size": _pos_float,
    "tick_value": _pos_float, "contract_size": _pos_float, "volume_min": _pos_float, "volume_max": _pos_float,
    "volume_step": _pos_float, "stops_level": _nonneg_float, "freeze_level": _nonneg_float,
    "swap_long": float, "swap_short": float, "swap_mode": _text(32), "currency_base": _ccy, "currency_profit": _ccy,
    "currency_margin": _ccy, "spread_points": _nonneg_float,
    "calc_mode": _enum(("forex", "futures", "cfd", "cfd_index", "cfd_leverage", "forex_no_leverage")),
}


def _symbol(v: Any) -> str:
    s = str(v).strip()
    if not re.fullmatch(r"[A-Za-z0-9._#@\-]{1,32}", s):
        raise ValueError("invalid symbol name")
    return s


def _digits(v: Any) -> int:
    x = int(v)
    if not 0 <= x <= 10:
        raise ValueError("must be 0..10")
    return x


class UserProfileStore:
    def __init__(self, path: Optional[str] = None, clock=None):
        self.path, self.clock = path, clock or SystemClock()
        self.records: Dict[str, Dict[str, Any]] = {}
        self.history: List[Dict[str, Any]] = []
        self.acks: Dict[str, Dict[str, str]] = {}
        self.bound_login_ref: Optional[str] = None
        if path and os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            self.records, self.history = d.get("records", {}), d.get("history", [])
            self.acks, self.bound_login_ref = d.get("acks", {}), d.get("bound_login_ref")

    # ------------------------------------------------------------ persist
    def _save(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"records": self.records, "history": self.history, "acks": self.acks,
                       "bound_login_ref": self.bound_login_ref}, f, indent=1)
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)

    # ------------------------------------------------------------- update
    def update(self, fields: Dict[str, Any], confirmed: bool = False) -> Dict[str, Any]:
        """Store user statements. confirmed=True only if the user explicitly confirmed the value."""
        now = iso(self.clock())
        out: Dict[str, Any] = {"changes": [], "rejected": {}, "warnings": []}
        secrets = find_secret_keys(fields)
        if secrets:
            out["rejected"] = {k: "secrets/credentials are never accepted or stored" for k in secrets}
            out["warnings"].append("request contained credential-like keys and was refused in full")
            return out
        clean: Dict[str, Any] = {}
        for k, v in fields.items():
            if k == "login":
                out["rejected"][k] = "the MT5 login is read from the terminal and is not accepted from chat"
            elif k not in VALIDATORS:
                out["rejected"][k] = "unknown field"
            else:
                try:
                    clean[k] = VALIDATORS[k](v)
                except (ValueError, TypeError) as e:
                    out["rejected"][k] = f"invalid value: {e}"
        old_type = self.records.get("account_type", {}).get("value")
        for k, v in clean.items():
            prev = self.records.get(k)
            scope = "account" if k in ACCOUNT_SCOPE else "symbol"
            rec = {"value": v, "timestamp": now, "verification": "confirmed" if confirmed else "unconfirmed", "scope": scope}
            self.records[k] = rec
            self.history.append({"field": k, "old": None if prev is None else prev["value"], "new": v, "timestamp": now,
                                 "verification": rec["verification"]})
            out["changes"].append({"field": k, "old": None if prev is None else prev["value"], "new": v,
                                   "verification": rec["verification"]})
        new_type = clean.get("account_type")
        if new_type is not None and old_type is not None and str(new_type).casefold() != str(old_type).casefold():
            downgraded = []
            for cf in COMMISSION_FIELDS:
                if cf in self.records and cf not in clean and self.records[cf]["verification"] == "confirmed":
                    self.records[cf]["verification"] = "unconfirmed"
                    downgraded.append(cf)
            self.acks.pop("commission_amount", None)
            out["warnings"].append(f"account type changed {old_type!r} -> {new_type!r}; previously stored commission values were "
                                   f"marked unconfirmed ({', '.join(downgraded) or 'none stored'}). Please re-confirm the commission.")
        if new_type is not None:
            out["warnings"].append("account_type is only a label; it does not define any commission amount, spread or contract size. "
                                   "Those stay unknown until provided/confirmed or read from MT5.")
        self._save()
        return out

    def acknowledge_conflict(self, field: str, choice: str) -> None:
        if choice not in ("mt5", "user"):
            raise ValueError("choice must be 'mt5' or 'user'")
        self.acks[field] = {"choice": choice, "timestamp": iso(self.clock())}
        self._save()

    def bind_account(self, login_ref: Optional[str]) -> None:
        self.bound_login_ref = login_ref
        self._save()

    # -------------------------------------------------------------- views
    def _fieldset(self, scope: str, names, downgrade: bool = False) -> FieldSet:
        fs = FieldSet(names)
        for n in names:
            r = self.records.get(n)
            if r and r["scope"] == scope:
                ok = r["verification"] == "confirmed" and not downgrade
                fs.set(n, Field(r["value"], Source.USER, Basis.USER_CONFIRMED if ok else Basis.USER_UNCONFIRMED, r["timestamp"],
                                "user-provided" + ("" if ok else " (unconfirmed" + (" / recorded for a different account)" if downgrade else ")"))))
        return fs

    def account_fields(self, downgrade: bool = False) -> FieldSet:
        return self._fieldset("account", ACCOUNT_FIELDS, downgrade)

    def symbol_fields(self) -> FieldSet:
        return self._fieldset("symbol", SYMBOL_FIELDS)

    def summary(self) -> Dict[str, Any]:
        return {"fields": {k: {"value": v["value"], "timestamp": v["timestamp"], "verification": v["verification"]}
                           for k, v in self.records.items()},
                "unconfirmed": sorted(k for k, v in self.records.items() if v["verification"] != "confirmed"),
                "bound_to_account": self.bound_login_ref is not None}
