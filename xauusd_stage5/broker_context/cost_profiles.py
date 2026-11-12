"""Historical (reproducible) cost profiles and the append-only label-run registry.

Historical labels must NOT depend on whatever account happens to be connected today. A CostProfile is an
explicit, immutable, hashed set of assumptions (broker, contract, spread source, commission, slippage).
The same market sample can have different outcomes under different profiles; the profile identity travels
with every label run. Existing runs are never rewritten - a new profile creates a new run.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .context_models import SystemClock, iso


def _canon(o: Any) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class CostProfile:
    name: str
    config_version: str
    broker_assumptions: Dict[str, Any]       # e.g. {"broker": "Alpari", "account_label": "ECN", "verified": False}
    contract_assumptions: Dict[str, Any]     # point, contract_size, account_currency, profit_currency (+ "verified")
    spread: Dict[str, Any]                   # {"source": "bar_spread"|"fixed_points", "fixed_points":.., "fallback_points":..}
    commission: Dict[str, Any]               # {"model": "none"} or {"amount","currency","basis", "verified"}
    slippage: Dict[str, Any]                 # {"model": "atr_fraction"|"fixed_price", "value": ..}

    @property
    def profile_id(self) -> str:
        return hashlib.sha256(_canon(self.as_dict()).encode()).hexdigest()

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "config_version": self.config_version, "broker_assumptions": self.broker_assumptions,
                "contract_assumptions": self.contract_assumptions, "spread": self.spread, "commission": self.commission,
                "slippage": self.slippage}

    def validate(self) -> None:
        c = self.contract_assumptions
        for k in ("point", "contract_size", "account_currency", "profit_currency"):
            if c.get(k) is None:
                raise ValueError(f"contract_assumptions.{k} is required")
        if c["account_currency"] != c["profit_currency"]:
            raise ValueError("historical profiles currently require account currency == profit currency (no FX data)")
        cm = self.commission
        if cm.get("model") != "none" and any(cm.get(k) is None for k in ("amount", "currency", "basis")):
            raise ValueError("commission must be explicit (model 'none' or amount+currency+basis); zero cost is never assumed silently")
        if cm.get("model") != "none" and cm["currency"] != c["account_currency"]:
            raise ValueError("commission currency must equal account currency in historical profiles")
        if self.spread.get("source") not in ("bar_spread", "fixed_points"):
            raise ValueError("spread.source must be 'bar_spread' or 'fixed_points'")
        if self.slippage.get("model") not in ("atr_fraction", "fixed_price"):
            raise ValueError("slippage.model must be 'atr_fraction' or 'fixed_price'")

    def to_scenario(self) -> Dict[str, Any]:
        """Scenario dict for the runtime cost engine (fills only unknown/untrusted components)."""
        sc: Dict[str, Any] = {}
        sp, pt = self.spread, self.contract_assumptions["point"]
        if sp.get("source") == "fixed_points":
            sc["spread_price"] = sp["fixed_points"] * pt
        elif sp.get("fallback_points") is not None:
            sc["spread_price"] = sp["fallback_points"] * pt
        if self.commission.get("model") == "none":
            sc["commission"] = {"amount": 0.0, "currency": self.contract_assumptions["account_currency"], "basis": "per_lot_round_turn"}
        else:
            sc["commission"] = {k: self.commission[k] for k in ("amount", "currency", "basis")}
        if self.slippage["model"] == "fixed_price":
            sc["slippage_price_per_side"] = self.slippage["value"]
        return sc


def historical_net_r(profile: CostProfile, gross_move_price: float, sl_distance_price: float,
                     spread_points: Optional[float] = None, atr: Optional[float] = None) -> Dict[str, Any]:
    """Cost-adjusted outcome of ONE historical trade in price units and R. Deterministic and pure."""
    profile.validate()
    c, pt, cs = profile.contract_assumptions, profile.contract_assumptions["point"], profile.contract_assumptions["contract_size"]
    imputed = False
    if profile.spread["source"] == "bar_spread" and spread_points and spread_points > 0:
        sp_price = spread_points * pt
    else:
        pts = profile.spread.get("fixed_points") if profile.spread["source"] == "fixed_points" else profile.spread.get("fallback_points")
        if pts is None:
            raise ValueError("spread unknown for this bar and the profile has no fallback_points")
        sp_price, imputed = pts * pt, profile.spread["source"] == "bar_spread"
    cm = profile.commission
    if cm.get("model") == "none":
        com_price = 0.0
    else:
        per_lot_money = cm["amount"] * (2 if cm["basis"] == "per_lot_per_side" else 1)
        com_price = per_lot_money / cs                      # money per lot / (money per 1.0 price per lot)
    sl = profile.slippage
    if sl["model"] == "atr_fraction":
        if atr is None:
            raise ValueError("atr required for slippage.model atr_fraction")
        slip_price = sl["value"] * atr * 2
    else:
        slip_price = sl["value"] * 2
    net_move = gross_move_price - sp_price - com_price - slip_price
    return {"profile_id": profile.profile_id, "spread_price": sp_price, "spread_imputed": imputed, "commission_price": com_price,
            "slippage_price": slip_price, "net_move_price": net_move, "net_R": net_move / sl_distance_price,
            "gross_R": gross_move_price / sl_distance_price}


class LabelRunRegistry:
    """Append-only record of labeling runs (never overwrites)."""

    def __init__(self, base_dir: str, clock=None):
        self.base, self.clock = base_dir, clock or SystemClock()

    def run_id(self, dataset_version: str, label_config_version: str, profile: CostProfile) -> str:
        return f"{dataset_version}__{label_config_version}__{profile.profile_id[:12]}"

    def register(self, dataset_version: str, label_config_version: str, profile: CostProfile,
                 extra: Optional[Dict[str, Any]] = None) -> str:
        profile.validate()
        rid = self.run_id(dataset_version, label_config_version, profile)
        d = os.path.join(self.base, rid)
        if os.path.exists(d):
            raise FileExistsError(f"label run {rid} already exists and is never rewritten; choose a different profile/version")
        os.makedirs(d)
        rec = {"run_id": rid, "created_at": iso(self.clock()), "dataset_version": dataset_version,
               "label_config_version": label_config_version, "cost_profile_id": profile.profile_id,
               "cost_scenario": profile.name, "broker_assumptions": profile.broker_assumptions,
               "contract_assumptions": profile.contract_assumptions, "spread_source": profile.spread,
               "commission_assumptions": profile.commission, "slippage_assumptions": profile.slippage,
               "configuration_version": profile.config_version, "extra": extra or {},
               "note": "historical/market-derived run; independent of any live account"}
        with open(os.path.join(d, "run_record.json"), "w") as f:
            json.dump(rec, f, indent=1)
        os.chmod(os.path.join(d, "run_record.json"), 0o444)
        return rid

    def list_runs(self) -> List[str]:
        return sorted(os.listdir(self.base)) if os.path.isdir(self.base) else []

    def get(self, run_id: str) -> Dict[str, Any]:
        with open(os.path.join(self.base, run_id, "run_record.json")) as f:
            return json.load(f)
