"""Timestamped runtime context = current market/terminal info + selected account context.

This is *runtime* information about the current account. It must never be mixed into historical
market facts or historical labels (see cost_profiles.py for the reproducible historical side).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .account_provider import BrokerDocumentationProvider
from .context_models import (ACCOUNT_FIELDS, SYMBOL_FIELDS, Basis, Conflict, Field, FieldSet, SystemClock, age_seconds,
                             iso)
from .context_resolver import resolve_scope
from .mt5_provider import MT5Provider
from .user_profile_provider import UserProfileStore


@dataclass
class BrokerContextConfig:
    account_ttl_s: float = 900.0        # account properties
    symbol_ttl_s: float = 21600.0       # contract specification, swaps (6 h)
    quote_ttl_s: float = 15.0           # live bid/ask/spread
    default_symbol: str = "XAUUSD"


@dataclass
class Snapshot:
    id: str
    created_at: str
    account: FieldSet
    symbol: FieldSet
    symbol_name: Optional[str]
    quote: Optional[Dict[str, Any]]
    conflicts: List[Conflict]
    mt5_status: Dict[str, Any]
    staleness: Dict[str, Dict[str, Any]]
    warnings: List[str] = field(default_factory=list)
    account_changed: bool = False
    symbol_candidates: List[str] = field(default_factory=list)
    kind: str = "runtime_context"

    @property
    def is_stale(self) -> bool:
        return any(v.get("stale") and v.get("used") for v in self.staleness.values())

    @property
    def unresolved_material_conflicts(self) -> List[Conflict]:
        return [c for c in self.conflicts if c.material and not c.acknowledged]

    def to_model_dict(self) -> Dict[str, Any]:
        """What the model may see: provenance on every field, no login, no credentials."""
        return {"kind": self.kind, "applies_to": "current account and terminal state; NOT historical market facts",
                "id": self.id, "created_at": self.created_at, "symbol": self.symbol_name,
                "account": self.account.to_dict(redact=True), "symbol_specifications": self.symbol.to_dict(redact=True),
                "quote": self.quote, "conflicts": [c.to_dict(redact=True) for c in self.conflicts],
                "staleness": self.staleness, "account_changed_since_profile_bound": self.account_changed,
                "mt5_status": {k: v for k, v in self.mt5_status.items() if k != "login"}, "warnings": self.warnings}


class SnapshotBuilder:
    def __init__(self, provider: MT5Provider, store: UserProfileStore, docs: Optional[BrokerDocumentationProvider] = None,
                 clock=None, config: Optional[BrokerContextConfig] = None):
        self.provider, self.store = provider, store
        self.docs = docs or BrokerDocumentationProvider()
        self.clock, self.cfg = clock or SystemClock(), config or BrokerContextConfig()
        self._cache: Dict[str, Any] = {}

    def _stale(self, retrieved_at, ttl, cached, used=True, source="mt5"):
        now = self.clock()
        a = age_seconds(retrieved_at, now)
        return {"retrieved_at": retrieved_at, "age_s": a, "ttl_s": ttl, "stale": a is None or a > ttl,
                "from_cache": cached, "used": used, "source": source}

    def build(self, symbol: Optional[str] = None) -> Snapshot:
        now = self.clock()
        warnings: List[str] = []
        status = self.provider.connection_status()
        # ------------------------------------------------------- account
        ar = self.provider.fetch_account()
        mt5_acc, acc_at, acc_cached, login_ref = None, None, False, None
        if ar.status in ("ok", "partial") and ar.fields is not None:
            self._cache["account"] = (ar.fields, ar.retrieved_at, ar.data.get("login_ref"))
            mt5_acc, acc_at, login_ref = ar.fields, ar.retrieved_at, ar.data.get("login_ref")
        else:
            warnings += [f"live MT5 account retrieval failed ({ar.status}): " + "; ".join(ar.warnings)]
            if "account" in self._cache:
                mt5_acc, acc_at, login_ref = self._cache["account"]
                acc_cached = True
                warnings.append(f"using cached account properties retrieved at {acc_at}")
        account_changed = bool(self.store.bound_login_ref and login_ref and self.store.bound_login_ref != login_ref)
        if account_changed:
            warnings.append("the connected MT5 account differs from the account the user profile was recorded for; "
                            "user-provided account values are treated as UNCONFIRMED until re-confirmed")
        user_acc = self.store.account_fields(downgrade=account_changed)
        # -------------------------------------------------------- symbol
        want = symbol or self.store.records.get("symbol", {}).get("value") or self.cfg.default_symbol
        sr = self.provider.fetch_symbol(want)
        mt5_sym, sym_at, sym_cached, quote, sym_name, cands = None, None, False, None, None, []
        if sr.status in ("ok", "partial") and sr.fields is not None:
            self._cache["symbol:" + want] = (sr.fields, sr.retrieved_at, sr.data)
            mt5_sym, sym_at, quote, sym_name = sr.fields, sr.retrieved_at, sr.data.get("quote"), sr.data.get("symbol")
            warnings += sr.warnings
        else:
            cands = sr.data.get("candidates", []) if isinstance(sr.data, dict) else []
            warnings += [f"live MT5 symbol retrieval failed ({sr.status}): " + "; ".join(sr.warnings)]
            if "symbol:" + want in self._cache:
                mt5_sym, sym_at, d = self._cache["symbol:" + want]
                quote, sym_name, sym_cached = d.get("quote"), d.get("symbol"), True
                warnings.append(f"using cached symbol specification retrieved at {sym_at}")
        user_sym = self.store.symbol_fields()
        sym_name = sym_name or (user_sym.value("symbol") if user_sym.known("symbol") else None)
        # ------------------------------------------------------- docs
        tmp_acc_server = (mt5_acc.value("server") if mt5_acc else None) or user_acc.value("server")
        doc_acc, doc_sym, dw = self.docs.lookup(tmp_acc_server, user_acc.value("account_type"), sym_name)
        # ------------------------------------------------------ resolve
        acc_age = age_seconds(acc_at, now)
        sym_age = age_seconds(sym_at, now)
        account, c1, w1 = resolve_scope("account", ACCOUNT_FIELDS, mt5_acc, acc_age, self.cfg.account_ttl_s, user_acc, doc_acc, self.store.acks)
        symbol_fs, c2, w2 = resolve_scope("symbol", SYMBOL_FIELDS, mt5_sym, sym_age, self.cfg.symbol_ttl_s, user_sym, doc_sym, self.store.acks)
        warnings += w1 + w2
        # quote staleness applies to spread fields independently of the (slower) contract specification
        q_age = age_seconds((quote or {}).get("retrieved_at"), now)
        if quote is not None and (q_age is None or q_age > self.cfg.quote_ttl_s):
            warnings.append(f"quote/spread is stale ({'unknown' if q_age is None else int(q_age)} s old, ttl {int(self.cfg.quote_ttl_s)} s)")
            for n in ("spread_price", "spread_points"):
                f = symbol_fs.get(n)
                if f.known and f.source.value == "mt5_live":
                    symbol_fs.set(n, Field(f.value, f.source, Basis.ESTIMATED, f.retrieved_at, "STALE live quote; refresh required"))
        staleness = {"account": self._stale(acc_at, self.cfg.account_ttl_s, acc_cached, mt5_acc is not None),
                     "symbol_specification": self._stale(sym_at, self.cfg.symbol_ttl_s, sym_cached, mt5_sym is not None),
                     "quote": self._stale((quote or {}).get("retrieved_at"), self.cfg.quote_ttl_s, sym_cached, quote is not None)}
        content = json.dumps({"a": account.to_dict(), "s": symbol_fs.to_dict(), "t": iso(now)}, sort_keys=True, default=str)
        return Snapshot(hashlib.sha256(content.encode()).hexdigest()[:16], iso(now), account, symbol_fs, sym_name, quote,
                        c1 + c2, status, staleness, warnings + (dw if doc_acc is None and False else []), account_changed, cands)
