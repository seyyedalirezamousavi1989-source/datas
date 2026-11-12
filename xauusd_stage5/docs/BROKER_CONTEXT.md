# Dynamic Broker / Account / MT5 Context Manager

**Status: implemented and unit-tested with a MOCKED MT5 module (28 tests).  NOT tested against a live terminal.**
`broker_context/live_integration_check.py` is provided for you to run on Windows; until it passes, live integration is unverified.

## Architecture & data flow
```
MT5 terminal (user logged in)        user profile (JSON, 0600)       broker documentation (exact-match only)
        │ read-only proxy                    │                                │
   MT5Provider ───────────┐   UserProfileStore ─────┐     BrokerDocumentationProvider
                          ▼                          ▼                          ▼
                       SnapshotBuilder → context_resolver (precedence + conflicts + TTL) → Snapshot
                                                      │
                          context_validator → cost_engine / risk_engine  (deterministic math)
                                                      │
                                   BrokerContextTools (8 strict-schema tools, standard envelope)  → model
```
Historical labeling does NOT use this runtime snapshot; it uses an immutable `CostProfile` (`cost_profiles.py`)
registered in an append-only `LabelRunRegistry`. Runtime context never rewrites historical labels.

## Source precedence (per field)
1. fresh MT5 live → 2. user-**confirmed** → 3. broker documentation (exact server/account/symbol match) → 4. stale MT5 (downgraded to *estimated*) →
5. user-unconfirmed → 6. unknown.  When two sources disagree both values are kept in a `Conflict`; material conflicts that affect the
requested calculation make it `incomplete` until the user acknowledges a choice (`mt5` or `user`).  Nothing is silently overwritten.
TTLs: account 900 s, symbol spec 6 h, quote 15 s (configurable).  MT5 `tick.time` is a server clock, so quote age is measured by retrieval time.

## Supported MT5 properties
account: login (hashed, never shown), server, company, currency, leverage, trade_mode (demo/contest/real), margin_mode.
symbol: digits, point, tick size/value, contract size, volume min/max/step, stops/freeze level, calc/exec/swap modes, swap long/short,
swap_rollover3days, filling mask, currencies (base/profit/margin), spread (points & price), bid/ask, description/path/visible.

## NOT exposed by the MT5 Python API (stay unknown unless user/broker doc supplies them)
commission amount/model/basis/currency, account type (ECN/Standard…), swap-free flag, symbol trading sessions, per-account
fees.  An account-type label never implies a commission, spread or contract size.

## User profile behavior
`update_user_account_profile` validates every value, rejects passwords/tokens/investor passwords and the login, records timestamp +
confirmed/unconfirmed, keeps history, and downgrades stored commission fields to *unconfirmed* if the account type changes.
Binding a profile to the connected account (hashed login) makes a later account switch downgrade user values until re-confirmed.

## Cost assumptions (runtime engine)
Bid-chart prices; one spread crossing per round trip; slippage only from a caller scenario (×2 sides); swap is a cost (positive = pay);
commission bases `per_lot_per_side` (×2), `per_lot_round_turn`, `flat_per_trade_round_turn`; MT5 `tick_value` is in the account currency
(value per price unit = tick_value / tick_size); when profit ≠ account currency and no tick value exists, an FX rate must be supplied
(result flagged *estimated*).  Statuses: complete / complete_with_estimates / partial / incomplete.  Unknown components are listed, never guessed.

## Security
No order_send/order_check/positions/orders/login functions are reachable (allow-list proxy raises PermissionError); `initialize()` is
called without credentials (attaches to the terminal you logged in); no password is requested/stored; MT5 authorization is set by the user via
CLI, there is **no model tool** that enables MT5; secret-like keys are refused in every tool; login never appears in model-visible output or logs.

## Install / connect (exact steps)
1. Windows PC, MetaTrader 5 installed; log in to your account in the terminal (Tools ▸ Options ▸ Expert Advisors: nothing needs enabling for read-only).
2. `pip install MetaTrader5` (Python 3.9–3.13, 64-bit, Windows only).
3. Copy the `broker_context/` folder; `python -m broker_context.cli authorize-mt5` and type `AUTHORIZE`.
4. `python -m broker_context.cli status`, then `python -m broker_context.live_integration_check`.
5. Tool schemas for your model runtime: `python -m broker_context.cli tool-schemas --out tools.json`.
Without a terminal (Linux/macOS or MT5 closed): everything still runs; account facts come from the user profile (`update_user_account_profile`),
and calculations are `incomplete` where facts are missing.  With a terminal: MT5 values take precedence and conflicts are surfaced.

## Troubleshooting
`not_authorized` → run authorize-mt5. `unavailable` + "not installed" → install package (Windows). `unavailable` + "terminal not initialized" → start MT5,
log in, avoid running terminal as different Windows user/admin mismatch. `partial` → terminal not connected to server or no account logged in.
`ambiguous` symbol → several gold symbols; set `symbol` in the profile. Empty `symbols_get` → add the symbol to Market Watch.

## System-prompt policy for the model
"Never state an exact lot size, tick value, margin, monetary risk or cost from memory; call the tools. If a tool returns incomplete/partial/stale,
report what is missing and do not guess. Account values describe the current account, not historical market facts. Never request passwords."
