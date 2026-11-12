"""Is the runtime context sufficient for a given calculation? Reports gaps; never fills them."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .context_models import TRUSTED_BASES
from .context_snapshot import Snapshot


RELEVANT_FIELDS = {
    "position_risk": {"contract_size", "tick_size", "tick_value", "point", "currency", "currency_profit"},
    "trade_costs": {"contract_size", "tick_size", "tick_value", "point", "currency", "currency_profit", "commission_amount",
                    "commission_basis", "commission_currency", "commission_model"},
    "swap": {"contract_size", "tick_size", "tick_value", "point", "currency", "currency_profit", "swap_long", "swap_short", "swap_mode"},
}
RELEVANT_FIELDS["trade_costs"] |= RELEVANT_FIELDS["swap"]


@dataclass
class ValidationReport:
    ok: bool
    blocking_missing: List[str] = field(default_factory=list)   # nothing monetary can be computed without these
    optional_missing: List[str] = field(default_factory=list)   # component stays unknown/partial
    warnings: List[str] = field(default_factory=list)
    stale_sources: List[str] = field(default_factory=list)
    blocking_conflicts: List[str] = field(default_factory=list)


def validate_lots(snapshot: Snapshot, lots: float) -> List[str]:
    probs: List[str] = []
    s = snapshot.symbol
    if lots is None or lots <= 0:
        return ["lots must be > 0"]
    vmin, vmax, vstep = s.value("volume_min"), s.value("volume_max"), s.value("volume_step")
    if vmin is not None and lots < vmin - 1e-12:
        probs.append(f"lots {lots} below volume_min {vmin}")
    if vmax is not None and lots > vmax + 1e-12:
        probs.append(f"lots {lots} above volume_max {vmax}")
    if vstep is not None and vmin is not None:
        k = (lots - vmin) / vstep
        if abs(k - round(k)) > 1e-6:
            probs.append(f"lots {lots} is not on the volume step {vstep} (min {vmin})")
    return probs


def validate(snapshot: Snapshot, calc: str, lots: Optional[float] = None) -> ValidationReport:
    r = ValidationReport(ok=True)
    s, a = snapshot.symbol, snapshot.account
    if not a.known("currency"):
        r.blocking_missing.append("account currency")
    value_path = (s.known("tick_value") and s.known("tick_size")) or (s.known("contract_size") and s.known("currency_profit"))
    if not value_path:
        r.blocking_missing += [n for n in ("tick_value", "tick_size", "contract_size", "currency_profit") if not s.known(n)]
    if calc in ("trade_costs",):
        if not s.known("spread_price"):
            r.optional_missing.append("spread_price")
        none_model = a.value("commission_model") == "none"
        if not none_model:
            r.optional_missing += [n for n in ("commission_amount", "commission_basis", "commission_currency") if not a.known(n)]
    if calc == "swap":
        r.optional_missing += [n for n in ("swap_long", "swap_short", "swap_mode") if not s.known(n)]
    if lots is not None:
        r.warnings += validate_lots(snapshot, lots)
    for k, v in snapshot.staleness.items():
        if v.get("used") and v.get("stale"):
            r.stale_sources.append(k)
    if r.stale_sources:
        r.warnings.append("stale sources: " + ", ".join(r.stale_sources))
    relevant = RELEVANT_FIELDS.get(calc, RELEVANT_FIELDS["trade_costs"])
    r.blocking_conflicts = [c.field for c in snapshot.unresolved_material_conflicts if c.field in relevant]
    other = [c.field for c in snapshot.unresolved_material_conflicts if c.field not in relevant]
    if r.blocking_conflicts:
        r.warnings.append("unresolved material conflicts that affect this calculation: " + ", ".join(r.blocking_conflicts))
    if other:
        r.warnings.append("unresolved conflicts not used by this calculation: " + ", ".join(other))
    r.ok = not r.blocking_missing and not r.blocking_conflicts
    return r
