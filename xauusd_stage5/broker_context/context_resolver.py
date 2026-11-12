"""Source precedence + explicit conflict handling.

Order (per field):
 1. fresh MT5 value for properties MT5 actually exposes
 2. explicitly CONFIRMED user value (for what MT5 cannot expose/verify)
 3. broker documentation (only when server + account type + symbol matched exactly)
 4. stale MT5 value (kept, labelled ESTIMATED) -> 5. unconfirmed user statement -> 6. unknown (None)
If MT5 and the user disagree, BOTH values are preserved in a Conflict; nothing is silently overwritten.
Material conflicts stay "needs confirmation" until the user acknowledges a choice.
"""
from __future__ import annotations

import datetime as dt
from typing import Dict, Iterable, List, Optional, Tuple

from .context_models import (MATERIAL_FIELDS, UNKNOWN_FIELD, Basis, Conflict, Field, FieldSet, Source, parse_iso,
                             values_equal)


def _more_recent(a: Optional[str], b: Optional[str]) -> str:
    if not a or not b:
        return "unknown"
    ta, tb = parse_iso(a), parse_iso(b)
    return "mt5" if ta > tb else ("user" if tb > ta else "same")


def resolve_scope(scope: str, names: Iterable[str], mt5: Optional[FieldSet], mt5_age_s: Optional[float], mt5_ttl_s: float,
                  user: FieldSet, doc: Optional[FieldSet], acks: Dict[str, Dict[str, str]]
                  ) -> Tuple[FieldSet, List[Conflict], List[str]]:
    names = tuple(names)
    out = FieldSet(names)
    conflicts: List[Conflict] = []
    warnings: List[str] = []
    fresh = mt5_age_s is not None and mt5_age_s <= mt5_ttl_s
    for n in names:
        m = mt5.get(n) if mt5 else UNKNOWN_FIELD
        u = user.get(n)
        d = doc.get(n) if doc else UNKNOWN_FIELD
        ack = acks.get(n)
        conflict: Optional[Conflict] = None
        if m.known and u.known and not values_equal(m.value, u.value):
            mat = n in MATERIAL_FIELDS
            rec = _more_recent(m.retrieved_at, u.retrieved_at)
            expl = (f"{n}: MT5 reports {m.value!r} (terminal value retrieved {m.retrieved_at}) but the user stated {u.value!r} "
                    f"(recorded {u.retrieved_at}, {u.basis.value}). The {rec if rec not in ('same','unknown') else 'more recent source cannot be determined; the'} "
                    f"value is the more recent one. MT5 is a live terminal reading; the user value is a statement. "
                    + ("This affects monetary calculations - please confirm which is correct." if mat else "Not material for cost/risk calculations."))
            conflict = Conflict(n, scope, m, u, rec, mat, expl, bool(ack), ack["choice"] if ack else None)
            conflicts.append(conflict)
            if mat and not ack:
                warnings.append(f"CONFLICT needs confirmation: {n} (MT5={m.value!r} vs user={u.value!r})")
        chosen: Field
        if conflict and conflict.acknowledged and conflict.chosen == "user":
            chosen = u
        elif m.known and fresh:
            chosen = m
        elif u.known and u.basis == Basis.USER_CONFIRMED:
            chosen = u
        elif d.known:
            chosen = d
        elif m.known:
            age = "unknown age" if mt5_age_s is None else f"{int(mt5_age_s)} s old"
            chosen = Field(m.value, m.source, Basis.ESTIMATED, m.retrieved_at, f"STALE MT5 value ({age}, ttl {int(mt5_ttl_s)} s); refresh required")
            warnings.append(f"{n} comes from a stale MT5 retrieval")
        elif u.known:
            chosen = u
        else:
            chosen = UNKNOWN_FIELD
        out.set(n, chosen)
    return out, conflicts, warnings
