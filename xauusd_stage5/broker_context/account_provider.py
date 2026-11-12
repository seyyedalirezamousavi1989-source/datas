"""Provider interface, shared result type, and the (optional) broker-documentation provider."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from .context_models import Basis, Field, FieldSet, Source, ACCOUNT_FIELDS, SYMBOL_FIELDS


@dataclass
class ProviderResult:
    status: str                      # ok | partial | unavailable | not_authorized | ambiguous | error
    fields: Optional[FieldSet] = None
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = "none"
    retrieved_at: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)


class AccountProvider(Protocol):
    def fetch_account(self) -> ProviderResult: ...
    def fetch_symbol(self, symbol: str) -> ProviderResult: ...


class BrokerDocumentationProvider:
    """Values copied from broker documentation. Used ONLY when broker server, account type and symbol
    match an entry exactly - a generic label (ECN/Standard) alone never selects an entry.

    File format: JSON list of {"server", "account_type", "symbol", "source_url", "documented_on",
    "account": {field: value}, "symbol_fields": {field: value}}.
    """

    def __init__(self, path: Optional[str] = None, entries: Optional[List[Dict[str, Any]]] = None):
        self.entries = entries if entries is not None else []
        if path and os.path.exists(path):
            with open(path) as f:
                self.entries = json.load(f)

    def lookup(self, server: Optional[str], account_type: Optional[str], symbol: Optional[str]):
        if not (server and account_type and symbol):
            return None, None, ["broker documentation not used: server, account type and symbol must all be established"]
        for e in self.entries:
            if (str(e.get("server", "")).casefold() == server.casefold()
                    and str(e.get("account_type", "")).casefold() == account_type.casefold()
                    and str(e.get("symbol", "")).casefold() == symbol.casefold()):
                note = f"{e.get('source_url', 'documentation')} ({e.get('documented_on', 'date unknown')})"
                acc, sym = FieldSet(ACCOUNT_FIELDS), FieldSet(SYMBOL_FIELDS)
                for k, v in (e.get("account") or {}).items():
                    if k in ACCOUNT_FIELDS:
                        acc.set(k, Field(v, Source.BROKER_DOC, Basis.BROKER_DOC, e.get("documented_on"), note))
                for k, v in (e.get("symbol_fields") or {}).items():
                    if k in SYMBOL_FIELDS:
                        sym.set(k, Field(v, Source.BROKER_DOC, Basis.BROKER_DOC, e.get("documented_on"), note))
                return acc, sym, []
        return None, None, ["no broker documentation entry matches this exact server/account type/symbol"]
