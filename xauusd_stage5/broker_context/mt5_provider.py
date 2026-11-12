"""READ-ONLY MetaTrader 5 provider.

* Uses the official ``MetaTrader5`` Python package (Windows only) or an injected module (tests).
* Only a fixed allow-list of read functions is reachable; order_send & co. raise PermissionError.
* ``initialize()`` is called WITHOUT login/password/server: it attaches to the terminal the user has
  already logged in. This code never asks for, accepts, stores or logs a password.
* Requires explicit user authorization (AuthorizationStore) set outside the model's reach.
* Every call is wrapped: failures become structured statuses, never exceptions.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from typing import Any, Dict, List, Optional

from .account_provider import ProviderResult
from .context_models import (ACCOUNT_FIELDS, Basis, Field, FieldSet, Source, SystemClock, iso, redact_text)
from .symbol_specifications import gold_candidates, parse_symbol_info, validate_gold

ALLOWED_CALLS = frozenset({"initialize", "shutdown", "terminal_info", "account_info", "symbols_get",
                           "symbol_info", "symbol_info_tick", "last_error", "version"})
TRADE_MODES = {0: "demo", 1: "contest", 2: "real"}
MARGIN_MODES = {0: "retail_netting", 1: "exchange", 2: "retail_hedging"}
NOT_EXPOSED_BY_MT5 = ["account_type", "commission_model", "commission_amount", "commission_currency",
                      "commission_basis", "swap_free"]


class ReadOnlyMT5:
    """Proxy that exposes only read functions of the MetaTrader5 module."""

    def __init__(self, module: Any):
        object.__setattr__(self, "_m", module)

    def __getattr__(self, name: str) -> Any:
        if name not in ALLOWED_CALLS:
            raise PermissionError(f"MetaTrader5.{name} is not available: this provider is read-only")
        return getattr(self._m, name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise PermissionError("read-only proxy")


class AuthorizationStore:
    """Persistent flag: has the USER enabled the MT5 integration? Not exposed as a model tool."""

    def __init__(self, path: Optional[str] = None, clock=None):
        self.path, self.clock = path, clock or SystemClock()
        self._mem = {"authorized": False}

    def _read(self) -> Dict[str, Any]:
        if self.path and os.path.exists(self.path):
            with open(self.path) as f:
                return json.load(f)
        return self._mem

    def is_authorized(self) -> bool:
        return bool(self._read().get("authorized"))

    def authorize(self) -> None:
        self._write({"authorized": True, "scope": "read_only", "authorized_at": iso(self.clock())})

    def revoke(self) -> None:
        self._write({"authorized": False, "revoked_at": iso(self.clock())})

    def _write(self, rec: Dict[str, Any]) -> None:
        self._mem = rec
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(rec, f)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)


def make_login_ref(login: Any, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{login}".encode()).hexdigest()[:12]


class MT5Provider:
    def __init__(self, authorization: AuthorizationStore, mt5_module: Any = None, clock=None,
                 terminal_path: Optional[str] = None, salt: str = "broker-context-local"):
        self.auth, self._mod, self.clock = authorization, mt5_module, clock or SystemClock()
        self.terminal_path, self.salt = terminal_path, salt
        self._initialized = False
        self.last_login_ref: Optional[str] = None

    # ------------------------------------------------------------ plumbing
    def _module(self):
        if self._mod is None:
            try:
                self._mod = importlib.import_module("MetaTrader5")
            except ImportError:
                return None
        return ReadOnlyMT5(self._mod)

    def _err(self, m: Any) -> str:
        try:
            return redact_text(str(m.last_error()))
        except Exception as e:  # noqa: BLE001
            return redact_text(f"last_error unavailable ({type(e).__name__})")

    def _ensure(self):
        """-> (module|None, error_status|None, message)"""
        if not self.auth.is_authorized():
            return None, "not_authorized", "MT5 integration has not been authorized by the user"
        m = self._module()
        if m is None:
            return None, "unavailable", "MetaTrader5 Python package is not installed (Windows only)"
        if not self._initialized:
            try:
                ok = m.initialize(path=self.terminal_path) if self.terminal_path else m.initialize()
            except Exception as e:  # noqa: BLE001
                return None, "error", redact_text(f"initialize raised {type(e).__name__}: {e}")
            if not ok:
                return None, "unavailable", f"terminal not initialized: {self._err(m)}"
            self._initialized = True
        return m, None, ""

    def close(self) -> None:
        if self._initialized:
            try:
                self._module().shutdown()
            finally:
                self._initialized = False

    # ------------------------------------------------------------- status
    def connection_status(self) -> Dict[str, Any]:
        now = iso(self.clock())
        st = {"authorized": self.auth.is_authorized(), "package_installed": None, "initialized": False,
              "terminal_connected": None, "account_logged_in": None, "trade_allowed_in_terminal": None,
              "retrieved_at": now, "error": None, "status": "unavailable"}
        m, status, msg = self._ensure()
        st["package_installed"] = False if (status == "unavailable" and "not installed" in msg) else (None if status == "not_authorized" else True)
        if m is None:
            st["status"], st["error"] = status, msg
            return st
        st["initialized"] = True
        try:
            ti = m.terminal_info()
            st["terminal_connected"] = None if ti is None else bool(getattr(ti, "connected", False))
            st["trade_allowed_in_terminal"] = None if ti is None else getattr(ti, "trade_allowed", None)
            ai = m.account_info()
            login = None if ai is None else getattr(ai, "login", None)
            st["account_logged_in"] = bool(login)
            self.last_login_ref = make_login_ref(login, self.salt) if login else None
            st["status"] = "ok" if (st["terminal_connected"] and st["account_logged_in"]) else "partial"
            if st["status"] == "partial":
                st["error"] = "terminal is running but not connected to a trade server or no account is logged in"
        except Exception as e:  # noqa: BLE001
            st["status"], st["error"] = "error", redact_text(f"{type(e).__name__}: {e}")
        return st

    # ------------------------------------------------------------ account
    def fetch_account(self) -> ProviderResult:
        now = iso(self.clock())
        m, status, msg = self._ensure()
        if m is None:
            return ProviderResult(status, None, {"message": msg}, "mt5", now, [msg], list(ACCOUNT_FIELDS))
        try:
            ai = m.account_info()
        except Exception as e:  # noqa: BLE001
            return ProviderResult("error", None, {}, "mt5", now, [redact_text(f"account_info raised {type(e).__name__}: {e}")], list(ACCOUNT_FIELDS))
        if ai is None:
            return ProviderResult("unavailable", None, {"message": "account_info returned None"}, "mt5", now,
                                  [f"account_info returned None ({self._err(m)}); is the terminal logged in?"], list(ACCOUNT_FIELDS))
        fs = FieldSet(ACCOUNT_FIELDS)
        warnings: List[str] = []

        def put(name: str, value: Any, note: Optional[str] = None) -> None:
            if value in (None, "", 0) and name != "login":
                value = None
            if name == "login" and not value:
                value = None
            fs.set(name, Field(value, Source.MT5_LIVE, Basis.VERIFIED_LIVE, now, note) if value is not None
                   else Field(None, Source.UNKNOWN, Basis.UNKNOWN, now, "not provided by MT5"))

        g = lambda a: getattr(ai, a, None)  # noqa: E731
        put("login", g("login"), "kept in memory only; never sent to the model")
        put("server", g("server"))
        put("broker_name", g("company"))
        put("currency", g("currency"))
        put("leverage", g("leverage"))
        tm = g("trade_mode")
        put("trade_mode", None if tm is None else TRADE_MODES.get(int(tm), f"unknown_code_{int(tm)}"))
        mm = g("margin_mode")
        put("margin_mode", None if mm is None else MARGIN_MODES.get(int(mm), f"unknown_code_{int(mm)}"))
        login = fs.value("login")
        ref = make_login_ref(login, self.salt) if login else None
        self.last_login_ref = ref or self.last_login_ref
        miss = [n for n in ACCOUNT_FIELDS if not fs.known(n)]
        return ProviderResult("ok" if not [n for n in miss if n not in NOT_EXPOSED_BY_MT5] else "partial", fs,
                              {"login_ref": ref, "not_exposed_by_mt5": list(NOT_EXPOSED_BY_MT5)}, "mt5", now, warnings, miss)


    def fetch_account_state(self) -> Dict[str, Any]:
        """Read-only: balance/equity/margin from account_info (open positions are NOT read)."""
        now = iso(self.clock())
        m, status, msg = self._ensure()
        if m is None:
            return {"status": status, "retrieved_at": now, "values": {}, "warnings": [msg]}
        try:
            ai = m.account_info()
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "retrieved_at": now, "values": {}, "warnings": [redact_text(f"{type(e).__name__}: {e}")]}
        if ai is None:
            return {"status": "unavailable", "retrieved_at": now, "values": {}, "warnings": ["account_info returned None"]}
        vals = {k: getattr(ai, k, None) for k in ("balance", "equity", "margin", "margin_free", "margin_level")}
        return {"status": "ok", "retrieved_at": now, "source": "mt5_live", "values": {k: v for k, v in vals.items() if v is not None},
                "warnings": ["open positions are not read; supply them via the user profile/caller"]}

    # ------------------------------------------------------------- symbol
    def list_gold_candidates(self, requested: str = "XAUUSD"):
        m, status, msg = self._ensure()
        if m is None:
            return None, status, msg
        try:
            syms = m.symbols_get()
        except Exception as e:  # noqa: BLE001
            return None, "error", redact_text(f"symbols_get raised {type(e).__name__}: {e}")
        if not syms:
            return [], "unavailable", f"symbols_get returned nothing ({self._err(m)})"
        return gold_candidates([getattr(s, "name", "") for s in syms], requested), "ok", ""

    def fetch_quote(self, name: str) -> Dict[str, Any]:
        now = iso(self.clock())
        out: Dict[str, Any] = {"status": "unavailable", "retrieved_at": now, "source": "symbol_info_tick",
                               "bid": None, "ask": None, "spread_price": None, "tick_time_server": None, "warnings": []}
        m, status, msg = self._ensure()
        if m is None:
            out["status"], out["warnings"] = status, [msg]
            return out
        try:
            t = m.symbol_info_tick(name)
        except Exception as e:  # noqa: BLE001
            out["status"], out["warnings"] = "error", [redact_text(f"symbol_info_tick raised {type(e).__name__}: {e}")]
            return out
        if t is None:
            out["warnings"] = [f"symbol_info_tick returned None ({self._err(m)})"]
            return out
        bid, ask = getattr(t, "bid", None), getattr(t, "ask", None)
        if not bid or not ask or bid <= 0 or ask <= 0 or ask < bid:
            out["warnings"] = [f"invalid quote bid={bid} ask={ask}"]
            return out
        out.update(status="ok", bid=float(bid), ask=float(ask), spread_price=round(float(ask) - float(bid), 10),
                   tick_time_server=getattr(t, "time", None))
        out["warnings"] = ["tick_time_server is on the broker-server clock (timezone unverified); quote age is NOT derived from it"]
        return out

    def fetch_symbol(self, symbol: str = "XAUUSD") -> ProviderResult:
        now = iso(self.clock())
        cands, status, msg = self.list_gold_candidates(symbol)
        if cands is None or (not cands and status != "ok"):
            return ProviderResult(status if cands is None else "unavailable", None, {"message": msg}, "mt5", now, [msg], [])
        if not cands:
            return ProviderResult("unavailable", None, {"candidates": []}, "mt5", now, ["no gold-like symbol found on this server"], [])
        m = self._module()
        valid = []
        for c in cands:
            try:
                info = m.symbol_info(c.name)
            except Exception:  # noqa: BLE001
                info = None
            if info is None:
                continue
            chk = validate_gold(c.name, info)
            if chk.ok:
                valid.append((c, info, chk))
        if not valid:
            return ProviderResult("unavailable", None, {"candidates": [c.name for c in cands]}, "mt5", now,
                                  ["gold-like names exist but none validates as XAU/USD: " + ", ".join(c.name for c in cands)], [])
        top = valid[0][0].score
        best = [v for v in valid if v[0].score == top]
        if len(best) > 1:
            return ProviderResult("ambiguous", None, {"candidates": [v[0].name for v in best]}, "mt5", now,
                                  ["several equally good gold symbols; ask the user which one to use: " + ", ".join(v[0].name for v in best)], [])
        cand, info, chk = best[0]
        quote = self.fetch_quote(cand.name)
        fs, avail, warns = parse_symbol_info(cand.name, info, now, quote if quote["status"] == "ok" else None)
        warns = list(warns) + ([] if chk.verified else ["gold identity not verified by currency base/profit; name-only match"])
        if quote["status"] != "ok":
            warns.append("live quote unavailable: " + "; ".join(quote["warnings"]))
        miss = fs.missing()
        return ProviderResult("ok" if not miss else "partial", fs,
                              {"symbol": cand.name, "match": cand.reason, "gold_verified": chk.verified, "availability": avail, "quote": quote},
                              "mt5", now, warns, miss)
