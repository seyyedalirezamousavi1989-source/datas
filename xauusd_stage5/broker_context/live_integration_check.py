"""Run THIS on your Windows machine with MetaTrader 5 open and logged in (demo account recommended first).

    pip install MetaTrader5
    python -m broker_context.cli authorize-mt5
    python -m broker_context.live_integration_check

It only READS (terminal/account/symbol/quote). It prints a PASS/FAIL list and writes live_check_report.json
(login is hashed, no passwords exist anywhere). Until this script has passed on a real terminal, the project must
describe MT5 integration as 'tested with a mocked terminal only'.
"""
import json
import sys

from .tools import create_tools


def main() -> int:
    t = create_tools()
    checks = []

    def chk(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": str(detail)[:300]})
        print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))

    st = t.call("get_mt5_connection_status")
    chk("terminal initialized and account logged in", st["status"] == "ok", st.get("warnings"))
    acc = t.call("get_active_account_context")
    chk("account currency read from MT5", acc["data"] and acc["data"]["account"]["currency"]["source"] == "mt5_live" if acc["data"] else False)
    sp = t.call("get_symbol_specifications")
    d = (sp["data"] or {}).get("specifications", {})
    chk("gold symbol resolved", sp["status"] in ("ok", "partial") and (sp["data"] or {}).get("symbol"), (sp["data"] or {}).get("candidates"))
    for f in ("point", "tick_size", "tick_value", "contract_size", "volume_min", "volume_step"):
        chk(f"symbol field {f}", d.get(f, {}).get("value") is not None)
    q = t.call("get_current_spread")
    chk("current spread read", q["status"] in ("ok", "partial"), q.get("warnings"))
    r = t.call("calculate_position_risk", lots=(d.get("volume_min") or {}).get("value") or 0.01, entry_price=2000.0, stop_price=1990.0)
    chk("position risk calculable", r["status"] in ("complete", "complete_with_estimates"), r.get("missing_fields"))
    out = json.dumps(checks, indent=1)
    open("live_check_report.json", "w").write(out)
    print("\nall passed:", all(c["pass"] for c in checks))
    return 0 if all(c["pass"] for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
