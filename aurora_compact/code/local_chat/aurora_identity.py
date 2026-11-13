"""AURORA QUANT identity, authority, capability-status and typed-memory layer.

This module EXTENDS what LocalChat already has; it does not replace it:

* creator verification (code + PBKDF2 + attempts + SQLite)   -> stays in server.py; here we only READ its result
  (``creator_verified``) and turn it into a role.  Nothing here can grant the creator role.
* capability registry (capabilities.py)                       -> read through ``iter_capabilities``; a small JSON
  overlay adds the fields the registry lacked (status enum, version, last verification, operator overrides).
* PersistentMemory (memory.py, SQLite)                        -> same database file; ``FactStore`` adds a typed,
  versioned fact table (kind / status / supersession / evidence) next to the ``memories`` table.

Identity knowledge is NOT authentication: the model may know who the creator is, but the role of the person typing
comes only from application state.  No secret (code, e-mail, password, token) is ever placed in a prompt or dataset.
"""
from __future__ import annotations

import json
import os
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    from .dbutil import Database
    from . import capabilities as _caps
except ImportError:                      # pragma: no cover - flat import
    from dbutil import Database
    import capabilities as _caps

# --------------------------------------------------------------------------------------------- canonical identity
CANON = {
    "creator": "Alireza Mousavi",
    "creator_aliases": ("Alireza Mousavi", "علیرضا موسوی", "سید علیرضا موسوی"),
    "creator_role": "creator, founder and lead developer",
    "agent": "AURORA QUANT",
    "organization": "AURORA AI Research Group",
    "predecessor": "LocalChat",
    "mission": "specialized AI research and trading-intelligence agent, with particular emphasis on Forex and XAUUSD analysis",
}

CAPABILITY_STATUSES = ("available", "active", "disabled", "unavailable", "requires_configuration",
                       "requires_authorization", "experimental")
_STATUS_FA = {
    "available": "در دسترس", "active": "فعال", "disabled": "غیرفعال", "unavailable": "ناموجود",
    "requires_configuration": "نیازمند پیکربندی", "requires_authorization": "نیازمند مجوز",
    "experimental": "آزمایشی",
}

ROLE_CREATOR = "creator_verified"
ROLE_UNVERIFIED = "unverified"


def role_from_app_state(creator_verified: bool) -> str:
    """The ONLY way a role is produced: from the application's verification record, never from message text."""
    return ROLE_CREATOR if creator_verified else ROLE_UNVERIFIED


# --------------------------------------------------------------------------------------------- capability overlay
class CapabilityOverlay:
    """Persistent extra metadata for registry capabilities (version, status override, last verification)."""

    def __init__(self, path: Optional[os.PathLike] = None, clock=time.time, default_version: str = "unknown"):
        self.path = Path(path) if path else None
        self.clock = clock
        self.default_version = default_version
        self._mem: Dict[str, dict] = {}

    def _load(self) -> Dict[str, dict]:
        if self.path and self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {}
        return self._mem

    def _save(self, data: Dict[str, dict]) -> None:
        self._mem = data
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)

    def set_state(self, capability_id: str, *, status: Optional[str] = None, version: Optional[str] = None,
                  note: Optional[str] = None) -> None:
        """Operator/installer hook: call when a capability is installed, changed, disabled or removed."""
        if status is not None and status not in CAPABILITY_STATUSES and status != "removed":
            raise ValueError(f"unknown capability status: {status}")
        data = self._load()
        rec = dict(data.get(capability_id, {}))
        if status is not None:
            rec["status"] = status
        if version is not None:
            rec["version"] = version
        if note is not None:
            rec["note"] = note
        rec["updated_at"] = int(self.clock())
        data[capability_id] = rec
        self._save(data)

    def clear_override(self, capability_id: str) -> None:
        data = self._load()
        rec = data.get(capability_id)
        if rec:
            rec.pop("status", None)
            rec["updated_at"] = int(self.clock())
            self._save(data)

    def record_verification(self, capability_id: str, ok: bool) -> None:
        data = self._load()
        rec = dict(data.get(capability_id, {}))
        rec["last_verified_at"] = int(self.clock())
        rec["last_verified_ok"] = bool(ok)
        data[capability_id] = rec
        self._save(data)

    def get(self, capability_id: str) -> dict:
        return dict(self._load().get(capability_id, {}))


def capability_status(capability, overlay: Optional[CapabilityOverlay] = None,
                      active_ids: Iterable[str] = ()) -> str:
    """Seven-state status.  Registry ``availability()`` is the ground truth for installed/unavailable; operators may
    only make things MORE restrictive or label them experimental - an override can never turn an unavailable
    capability into an available one."""
    try:
        available = bool(capability.availability())
    except Exception:                                    # noqa: BLE001 - a broken probe means "not available"
        available = False
    rec = overlay.get(capability.id) if overlay else {}
    override = rec.get("status")
    if override == "removed":
        return "unavailable"
    if not available:
        return "requires_configuration" if override == "requires_configuration" else "unavailable"
    if override in ("disabled", "requires_configuration", "requires_authorization", "experimental"):
        return override
    if capability.id in set(active_ids):
        return "active"
    return "available"


def describe_capability(capability, overlay: Optional[CapabilityOverlay] = None, active_ids: Iterable[str] = ()) -> dict:
    rec = overlay.get(capability.id) if overlay else {}
    return {
        "id": capability.id, "name": capability.name, "description": capability.description,
        "purpose": capability.purpose, "inputs": list(capability.inputs), "outputs": list(capability.outputs),
        "dependencies": list(capability.dependencies), "operational_conditions": list(capability.activation_conditions),
        "limitations": list(capability.limitations),
        "status": capability_status(capability, overlay, active_ids),
        "version": rec.get("version") or (overlay.default_version if overlay else "unknown"),
        "last_verified_at": rec.get("last_verified_at"),
    }


def verify_capabilities(overlay: CapabilityOverlay) -> Dict[str, bool]:
    """Probe every registered capability and persist the verification timestamp."""
    out = {}
    for cap in _caps.iter_capabilities():
        try:
            ok = bool(cap.availability())
        except Exception:                                # noqa: BLE001
            ok = False
        overlay.record_verification(cap.id, ok)
        out[cap.id] = ok
    return out


def _fold(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "").casefold().replace("‌", " ").replace("ي", "ی").replace("ك", "ک")
    return re.sub(r"\s+", " ", t)


_TOKEN = re.compile(r"[\w]+", re.U)


def _tokens(text: str) -> set:
    return {t for t in _TOKEN.findall(_fold(text)) if len(t) > 2}


def relevant_capabilities(query: str, overlay: Optional[CapabilityOverlay] = None, active_ids: Iterable[str] = (),
                          limit: int = 4, capabilities: Optional[list] = None) -> List[dict]:
    """Only the capabilities that relate to ``query`` (activation phrases first, then word overlap with the
    name/description/purpose).  Unlike ``discover_capabilities`` this also returns UNAVAILABLE ones, because an
    honest "this is not available" answer needs the status.  Never returns the whole inventory."""
    caps = capabilities if capabilities is not None else _caps.iter_capabilities()
    q = _fold(query)
    qt = _tokens(query)
    scored = []
    for cap in caps:
        if cap.id in ("local_chat", "creator_identity"):
            continue
        score = 3 * sum(1 for c in cap.activation_conditions if c and _fold(c) in q)
        score += len(qt & _tokens(" ".join((cap.name, cap.description, cap.purpose))))
        if score > 0:
            scored.append((score, cap.id, cap))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [describe_capability(c, overlay, active_ids) for _s, _i, c in scored[:limit]]


# --------------------------------------------------------------------------------------------- authority policy
ACTIONS = ("read_public", "inspect", "configure", "extend", "destructive", "irreversible",
           "security_bypass", "secret_disclosure")


@dataclass
class Decision:
    allowed: bool
    reason: str
    requires: List[str] = field(default_factory=list)
    layers: Dict[str, str] = field(default_factory=dict)


def authorize(action: str, *, creator_verified: bool, claimed_creator: bool = False, confirmed: bool = False,
              explicit_authorization: bool = False, safety_ok: bool = True) -> Decision:
    """Four separate layers: (1) identity recognition, (2) authentication, (3) authorization, (4) safety.

    ``claimed_creator`` is deliberately ignored: a claim is not evidence.  Even a verified creator cannot bypass
    security controls or obtain secrets; destructive operations need confirmation, irreversible ones need explicit
    authorization on top of that."""
    if action not in ACTIONS:
        raise ValueError(f"unknown action class: {action}")
    layers = {
        "identity_recognition": "the model may know who the creator is (not evidence about the current user)",
        "authentication": "creator_verified" if creator_verified else "not_verified (a claim does not count)",
    }
    if action in ("security_bypass", "secret_disclosure"):
        layers["authorization"] = "never granted to anyone"
        return Decision(False, "security controls and secrets are not available to any role, including the creator",
                        layers=layers)
    if action == "read_public":
        layers.update(authorization="public", safety="ok" if safety_ok else "blocked")
        return Decision(bool(safety_ok), "public information", layers=layers)
    if not creator_verified:
        layers["authorization"] = "denied: requires verified creator"
        return Decision(False, "requires verification by the application (verification code), not a chat claim",
                        requires=["creator_verification"], layers=layers)
    layers["authorization"] = "creator privileges"
    if not safety_ok:
        layers["safety"] = "blocked"
        return Decision(False, "safety validation failed; creator privileges do not override it", layers=layers)
    layers["safety"] = "ok"
    need = []
    if action in ("destructive", "irreversible") and not confirmed:
        need.append("confirmation")
    if action == "irreversible" and not explicit_authorization:
        need.append("explicit_authorization")
    if need:
        return Decision(False, "confirmation is required before this operation", requires=need, layers=layers)
    return Decision(True, "allowed", layers=layers)


# --------------------------------------------------------------------------------------------- typed memory
KINDS = ("confirmed_fact", "user_provided_unverified", "hypothesis", "recommendation", "future_plan", "decision",
         "implemented", "tested", "active_task", "completed_work", "pending_work", "known_limitation")
PROPOSAL_KINDS = ("hypothesis", "recommendation", "future_plan")
NEEDS_EVIDENCE = ("implemented", "tested", "completed_work")
# promotion is explicit: a proposal can never jump straight to "implemented"/"tested"/"confirmed_fact"
_PROMOTIONS = {
    "hypothesis": ("recommendation", "decision", "known_limitation"),
    "recommendation": ("decision", "future_plan"),
    "future_plan": ("decision", "active_task", "pending_work"),
    "user_provided_unverified": ("confirmed_fact",),
    "decision": ("active_task", "pending_work", "implemented"),
    "active_task": ("completed_work", "pending_work", "implemented"),
    "pending_work": ("active_task",),
    "implemented": ("tested", "known_limitation"),
}
KIND_LABEL_FA = {
    "confirmed_fact": "حقیقت تأییدشده", "user_provided_unverified": "اظهار کاربر (تأییدنشده)", "hypothesis": "فرضیه",
    "recommendation": "پیشنهاد", "future_plan": "برنامهٔ آینده", "decision": "تصمیم", "implemented": "پیاده‌سازی‌شده",
    "tested": "پیاده‌سازی و آزمون‌شده", "active_task": "کار جاری", "completed_work": "کار انجام‌شده",
    "pending_work": "کار باقی‌مانده", "known_limitation": "محدودیت شناخته‌شده",
}
_SECRET = re.compile(
    r"(?:(?:password|passwd|api[_ -]?key|secret|token|verification code|کد تأیید|رمز(?: عبور)?)\s*(?:[:=]|\bis\b|است)\s*\S{4,})"
    r"|(?<!\d)\d{6}(?!\d)|\b(?:sk-|ghp_|github_pat_)[A-Za-z0-9_-]{8,}", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


class FactStoreError(ValueError):
    pass


class FactStore:
    """Typed, versioned facts in the SAME SQLite file as ``PersistentMemory`` (pass ``memory.path``)."""

    def __init__(self, database_path, clock=time.time):
        self.db = Database(database_path)
        self.clock = clock
        with self.db.tx() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS identity_facts (
                id TEXT PRIMARY KEY, key TEXT, content TEXT NOT NULL, kind TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active', evidence TEXT, supersedes TEXT, superseded_by TEXT,
                source TEXT, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL)""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_identity_facts_kind ON identity_facts(kind, status)")

    # -- writes
    def add(self, content: str, kind: str, *, key: Optional[str] = None, evidence: Optional[str] = None,
            source: str = "app") -> str:
        content = (content or "").strip()
        if kind not in KINDS:
            raise FactStoreError(f"unknown kind: {kind}")
        if not content or len(content) > 1500:
            raise FactStoreError("content must be 1..1500 characters")
        if _SECRET.search(content) or _EMAIL.search(content):
            raise FactStoreError("secret-looking or private contact data is never stored")
        if kind in NEEDS_EVIDENCE and not (evidence or "").strip():
            raise FactStoreError(f"'{kind}' requires evidence (what was built/run); a proposal is not an implementation")
        now = int(self.clock())
        fid = uuid.uuid4().hex[:12]
        with self.db.tx() as db:
            if key:
                row = db.execute("SELECT id FROM identity_facts WHERE key=? AND status='active'", (key,)).fetchone()
                if row:
                    db.execute("UPDATE identity_facts SET status='superseded', superseded_by=?, updated_at=? WHERE id=?",
                               (fid, now, row["id"]))
                    supersedes = row["id"]
                else:
                    supersedes = None
            else:
                supersedes = None
            db.execute("INSERT INTO identity_facts(id,key,content,kind,status,evidence,supersedes,source,created_at,updated_at)"
                       " VALUES(?,?,?,?, 'active',?,?,?,?,?)", (fid, key, content, kind, evidence, supersedes, source, now, now))
        return fid

    def supersede(self, old_id: str, content: str, kind: Optional[str] = None, *, evidence: Optional[str] = None) -> str:
        old = self.get(old_id)
        if not old or old["status"] != "active":
            raise FactStoreError("only an active fact can be superseded")
        new_kind = kind or old["kind"]
        if new_kind in NEEDS_EVIDENCE and not (evidence or "").strip():
            raise FactStoreError(f"'{new_kind}' requires evidence")
        if _SECRET.search(content or "") or _EMAIL.search(content or ""):
            raise FactStoreError("secret-looking or private contact data is never stored")
        now = int(self.clock())
        fid = uuid.uuid4().hex[:12]
        with self.db.tx() as db:
            db.execute("UPDATE identity_facts SET status='superseded', superseded_by=?, updated_at=? WHERE id=?",
                       (fid, now, old_id))
            db.execute("INSERT INTO identity_facts(id,key,content,kind,status,evidence,supersedes,source,created_at,updated_at)"
                       " VALUES(?,?,?,?, 'active',?,?,?,?,?)",
                       (fid, old["key"], content.strip(), new_kind, evidence, old_id, old["source"], now, now))
        return fid

    def promote(self, fact_id: str, new_kind: str, *, evidence: Optional[str] = None) -> str:
        old = self.get(fact_id)
        if not old or old["status"] != "active":
            raise FactStoreError("only an active fact can be promoted")
        if new_kind not in _PROMOTIONS.get(old["kind"], ()):
            raise FactStoreError(f"'{old['kind']}' cannot become '{new_kind}' directly; "
                                 f"allowed: {list(_PROMOTIONS.get(old['kind'], ()))}")
        return self.supersede(fact_id, old["content"], new_kind, evidence=evidence)

    # -- reads
    def get(self, fact_id: str) -> Optional[dict]:
        with self.db.read() as db:
            row = db.execute("SELECT * FROM identity_facts WHERE id=?", (fact_id,)).fetchone()
        return dict(row) if row else None

    def list(self, *, include_superseded: bool = False, kinds: Optional[Iterable[str]] = None) -> List[dict]:
        sql = "SELECT * FROM identity_facts"
        cond, args = [], []
        if not include_superseded:
            cond.append("status='active'")
        if kinds:
            ks = list(kinds)
            cond.append("kind IN (%s)" % ",".join("?" * len(ks)))
            args += ks
        if cond:
            sql += " WHERE " + " AND ".join(cond)
        sql += " ORDER BY rowid"
        with self.db.read() as db:
            return [dict(r) for r in db.execute(sql, args).fetchall()]

    def history(self, key: str) -> List[dict]:
        with self.db.read() as db:
            return [dict(r) for r in db.execute("SELECT * FROM identity_facts WHERE key=? ORDER BY rowid", (key,))]

    def relevant(self, query: str, limit: int = 8) -> List[dict]:
        """Identity facts always; then facts overlapping the query; open work/limitations when asked about status."""
        facts = self.list()
        qt = _tokens(query)
        status_q = bool(re.search(r"(?:وضعیت|ادامه|کجا|پیاده|انجام|باقی|status|progress|continue|resume|implemented|done|pending)",
                                   _fold(query)))
        picked, seen = [], set()

        def take(items):
            for f in items:
                if f["id"] not in seen and len(picked) < limit:
                    seen.add(f["id"])
                    picked.append(f)

        take(f for f in facts if (f["key"] or "").startswith("identity."))
        scored = sorted(((len(qt & _tokens(f["content"])), f) for f in facts), key=lambda x: -x[0])
        take(f for s, f in scored if s > 0)
        if status_q:
            take(f for f in facts if f["kind"] in ("active_task", "pending_work", "known_limitation", "completed_work"))
        return picked

    def recovery_context(self, query: str = "", limit: int = 8, max_chars: int = 1800) -> str:
        lines, total = [], 0
        for f in self.relevant(query, limit):
            line = f"- [{KIND_LABEL_FA[f['kind']]}] {f['content']}"
            if total + len(line) > max_chars:
                continue
            lines.append(line)
            total += len(line)
        return "\n".join(lines)

    def seed_canonical(self) -> int:
        """Idempotent: confirmed identity facts (and nothing else about the project) - unknown stays unknown."""
        seeds = [
            ("identity.creator", f"{CANON['creator']} is the creator, founder and lead developer of {CANON['agent']}."),
            ("identity.agent", f"{CANON['agent']} is the specialized AI agent; it is not the same thing as the underlying language model."),
            ("identity.organization", f"{CANON['organization']} is the research organization behind the broader research and development direction of {CANON['agent']}."),
            ("identity.predecessor", f"{CANON['predecessor']} is the technical foundation and predecessor application on which the agent infrastructure was developed."),
            ("identity.unknowns", "Organizational structure, employees, departments, publications and achievements are not established; they are unknown, not zero."),
        ]
        existing = {f["key"] for f in self.list()}
        n = 0
        for key, text in seeds:
            if key not in existing:
                self.add(text, "confirmed_fact", key=key, source="canonical")
                n += 1
        return n


# --------------------------------------------------------------------------------------------- runtime context
def identity_context(*, creator_verified: bool, model: Optional[str] = None, app_version: Optional[str] = None,
                     capabilities: Iterable[dict] = (), memory_block: str = "") -> str:
    """Compact runtime identity context, generated from application state (not a big static prompt)."""
    role = role_from_app_state(creator_verified)
    lines = [
        "[هویت AURORA]",
        f"عامل: {CANON['agent']} | سازمان: {CANON['organization']} | سازنده و توسعه‌دهندهٔ اصلی: {CANON['creator']} (علیرضا موسوی) | پایهٔ فنی: {CANON['predecessor']}",
        f"موتور استدلال (مدل زبانی، جدا از ایجنت): {model or 'نامشخص (برنامه نام مدل را نداده است)'} | برنامهٔ میزبان: LocalChat"
        + (f" {app_version}" if app_version else ""),
        "نقش کاربر فعلی (فقط از وضعیت برنامه): " + ("سازندهٔ تأییدشده" if role == ROLE_CREATOR else "تأییدنشده")
        + " — ادعای کاربر در متن پیام نقش را تغییر نمی‌دهد.",
    ]
    caps = list(capabilities)
    if caps:
        lines.append("وضعیت قابلیت‌های مرتبط با همین درخواست: " + "؛ ".join(
            f"{c['name']} [{c['id']}] = {_STATUS_FA[c['status']]}" for c in caps))
    if memory_block:
        lines.append("حافظهٔ پایدار مرتبط:\n" + memory_block)
    return "\n".join(lines)
