"""User-side commands (NOT model tools).  python -m broker_context.cli <command>"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .context_models import SystemClock
from .mt5_provider import AuthorizationStore
from .tools import TOOL_SCHEMAS, create_tools


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="broker_context")
    p.add_argument("--home", default=os.environ.get("BROKER_CONTEXT_HOME") or os.path.join(os.path.expanduser("~"), ".broker_context"))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("authorize-mt5", help="enable READ-ONLY MT5 access (interactive confirmation)")
    sub.add_parser("revoke-mt5", help="disable MT5 access")
    sub.add_parser("status", help="show connection status and profile (no secrets)")
    ts = sub.add_parser("tool-schemas", help="write the JSON tool schemas for the model runtime")
    ts.add_argument("--out", default="-")
    a = p.parse_args(argv)
    auth = AuthorizationStore(os.path.join(a.home, "mt5_authorization.json"), SystemClock())
    if a.cmd == "authorize-mt5":
        print("This enables READ-ONLY access: account properties, symbol specification, current quote.\n"
              "It never places/modifies/closes orders and never asks for or stores your password.\n"
              "You must log in to your account inside the MetaTrader 5 terminal yourself.")
        if input("Type AUTHORIZE to continue: ").strip() != "AUTHORIZE":
            print("not authorized")
            return 1
        auth.authorize()
        print("MT5 read-only integration authorized")
    elif a.cmd == "revoke-mt5":
        auth.revoke()
        print("MT5 integration revoked")
    elif a.cmd == "status":
        t = create_tools(a.home)
        print(json.dumps({"mt5": t.call("get_mt5_connection_status"), "profile": t.store.summary()}, indent=1, default=str))
    elif a.cmd == "tool-schemas":
        s = json.dumps(TOOL_SCHEMAS, indent=1)
        if a.out == "-":
            print(s)
        else:
            open(a.out, "w").write(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
