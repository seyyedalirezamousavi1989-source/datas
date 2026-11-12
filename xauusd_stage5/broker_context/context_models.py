"""Broker-independent normalized schema used by every other broker_context module.

Every value carries its provenance (source, basis, retrieval time). Unknown values are kept as
``None`` - never guessed - and are reported in ``missing_fields``.
"""
from __future__ import annotations

import datetime as dt
import enum
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


class Source(str, enum.Enum):
    MT5_LIVE = "mt5_live"
    USER = "user"
    BROKER_DOC = "broker_documentation"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"


class Basis(str, enum.Enum):
    """How far a value can be trusted. Used to label every monetary component."""
    VERIFIED_LIVE = "verified_live"
    USER_CONFIRMED = "user_confirmed"
    BROKER_DOC = "broker_documentation"
    USER_UNCONFIRMED = "user_unconfirmed"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"


BASIS_RANK = {Basis.VERIFIED_LIVE: 5, Basis.USER_CONFIRMED: 4, Basis.BROKER_DOC: 3,
              Basis.USER_UNCONFIRMED: 2, Basis.ESTIMATED: 1, Basis.UNKNOWN: 0}
TRUSTED_BASES = frozenset({Basis.VERIFIED_LIVE, Basis.USER_CONFIRMED, Basis.BROKER_DOC})


def weakest(*bases: Basis) -> Basis:
    return min(bases, key=lambda b: BASIS_RANK[b]) if bases else Basis.UNKNOWN


ACCOUNT_FIELDS = ("broker_name", "server", "login", "account_type", "currency", "leverage", "trade_mode",
                  "margin_mode", "commission_model", "commission_amount", "commission_currency", "commission_basis",
                  "swap_free")
SYMBOL_FIELDS = ("symbol", "digits", "point", "tick_size", "tick_value", "contract_size", "volume_min", "volume_max",
                 "volume_step", "stops_level", "freeze_level", "calc_mode", "execution_mode", "filling_modes",
                 "swap_long", "swap_short", "swap_mode", "swap_rollover3days", "sessions", "spread_points",
                 "spread_price", "currency_base", "currency_profit", "currency_margin", "description", "path", "visible")
SENSITIVE_FIELDS = frozenset({"login"})
COMMISSION_BASES = ("per_lot_per_side", "per_lot_round_turn", "flat_per_trade_round_turn")
COMMISSION_MODELS = ("none", "per_lot", "flat", "other")
# fields whose disagreement changes monetary results -> must be confirmed by the user
MATERIAL_FIELDS = frozenset({"contract_size", "tick_size", "tick_value", "point", "digits", "currency", "leverage",
                             "commission_amount", "commission_basis", "commission_currency", "commission_model",
                             "swap_long", "swap_short", "swap_mode", "currency_profit"})


def iso(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)


class SystemClock:
    def __call__(self) -> dt.datetime:
        return dt.datetime.now(dt.timezone.utc)


def age_seconds(retrieved_at: Optional[str], now: dt.datetime) -> Optional[float]:
    if not retrieved_at:
        return None
    return (now - parse_iso(retrieved_at)).total_seconds()


def values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().casefold() == b.strip().casefold()
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return sorted(map(str, a)) == sorted(map(str, b))
    return a == b


@dataclass(frozen=True)
class Field:
    value: Any = None
    source: Source = Source.UNKNOWN
    basis: Basis = Basis.UNKNOWN
    retrieved_at: Optional[str] = None
    note: Optional[str] = None

    @property
    def known(self) -> bool:
        return self.value is not None

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "source": self.source.value, "basis": self.basis.value,
                "retrieved_at": self.retrieved_at, "note": self.note}


UNKNOWN_FIELD = Field()


class FieldSet:
    """Ordered collection of Fields over a fixed list of names."""

    def __init__(self, names: Iterable[str], fields: Optional[Dict[str, Field]] = None):
        self.names = tuple(names)
        self._f: Dict[str, Field] = dict(fields or {})

    def get(self, name: str) -> Field:
        return self._f.get(name, UNKNOWN_FIELD)

    def value(self, name: str) -> Any:
        return self.get(name).value

    def known(self, name: str) -> bool:
        return self.get(name).known

    def set(self, name: str, f: Field) -> None:
        self._f[name] = f

    def missing(self, names: Optional[Iterable[str]] = None) -> List[str]:
        return [n for n in (names or self.names) if not self.known(n)]

    def copy(self) -> "FieldSet":
        return FieldSet(self.names, dict(self._f))

    def to_dict(self, redact: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for n in self.names:
            f = self.get(n)
            d = f.to_dict()
            if redact and n in SENSITIVE_FIELDS:
                d = {"value": None, "present": f.known, "redacted": True, "source": f.source.value,
                     "basis": f.basis.value, "retrieved_at": f.retrieved_at, "note": "sensitive identifier withheld"}
            out[n] = d
        return out


@dataclass
class Conflict:
    field: str
    scope: str
    mt5: Field
    user: Field
    more_recent: str
    material: bool
    explanation: str
    acknowledged: bool = False
    chosen: Optional[str] = None

    def to_dict(self, redact: bool = True) -> Dict[str, Any]:
        hide = redact and self.field in SENSITIVE_FIELDS
        m, u = self.mt5.to_dict(), self.user.to_dict()
        if hide:
            m["value"] = u["value"] = None
        return {"field": self.field, "scope": self.scope, "mt5": m, "user": u, "more_recent": self.more_recent,
                "material": self.material, "explanation": self.explanation, "acknowledged": self.acknowledged,
                "chosen": self.chosen}


@dataclass
class ToolResult:
    status: str
    data: Any = None
    source: str = "none"
    retrieved_at: Optional[str] = None
    is_stale: bool = False
    missing_fields: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "data": self.data, "source": self.source, "retrieved_at": self.retrieved_at,
                "is_stale": self.is_stale, "missing_fields": list(self.missing_fields), "warnings": list(self.warnings)}


# ---------------------------------------------------------------- redaction
_SECRET_TOKENS = {"password", "passwd", "pwd", "pass", "token", "secret", "apikey", "pin", "otp", "credential",
                  "credentials", "privatekey"}
_LOGIN_RE = re.compile(r"(?i)\b(login|account|acct)\b\s*[:=#]?\s*\d{4,}")
_PWD_RE = re.compile(r"(?i)\b(password|passwd|pwd|token|secret)\b\s*[:=]\s*\S+")


def is_secret_key(key: str) -> bool:
    toks = {t for t in re.split(r"[^a-z0-9]+", str(key).lower()) if t}
    return bool(toks & _SECRET_TOKENS) or str(key).lower().replace("_", "").replace("-", "") in _SECRET_TOKENS


def find_secret_keys(obj: Any, path: str = "") -> List[str]:
    bad: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            if is_secret_key(k):
                bad.append(p)
            bad += find_secret_keys(v, p)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad += find_secret_keys(v, f"{path}[{i}]")
    return bad


def redact_text(text: str) -> str:
    text = _PWD_RE.sub(lambda m: f"{m.group(1)}=<redacted>", str(text))
    return _LOGIN_RE.sub(lambda m: f"{m.group(1)} <redacted>", text)
