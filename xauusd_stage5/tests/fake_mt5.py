"""Mocked MetaTrader5 module for unit tests. NOT a live terminal."""
import datetime as dt
from types import SimpleNamespace as NS


class FakeClock:
    def __init__(self, t=None):
        self.t = t or dt.datetime(2026, 1, 5, 12, 0, tzinfo=dt.timezone.utc)

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += dt.timedelta(seconds=s)


def gold_info(**kw):
    d = dict(digits=2, point=0.01, trade_tick_size=0.01, trade_tick_value=1.0, trade_contract_size=100.0, volume_min=0.01,
             volume_max=100.0, volume_step=0.01, trade_stops_level=10, trade_freeze_level=0, trade_calc_mode=2, trade_exemode=2,
             filling_mode=1, swap_long=-20.0, swap_short=5.0, swap_mode=1, swap_rollover3days=3, spread=20, spread_float=True,
             currency_base="XAU", currency_profit="USD", currency_margin="XAU", description="Gold vs US Dollar", path="Metals\\XAUUSD",
             visible=True)
    d.update(kw)
    return NS(**d)


class FakeMT5:
    def __init__(self, symbols=None, account=True, connected=True, init_ok=True, currency="USD", login=123456789, bid=2000.00, ask=2000.20):
        self.symbols = symbols if symbols is not None else {"XAUUSD": gold_info()}
        self.has_account, self.connected, self.init_ok = account, connected, init_ok
        self.currency, self.login, self.bid, self.ask = currency, login, bid, ask
        self.calls = []
        self.tick_fail = False

    def initialize(self, *a, **k):
        self.calls.append(("initialize", a, k))
        return self.init_ok

    def shutdown(self):
        return True

    def last_error(self):
        return (1, "fake error")

    def terminal_info(self):
        return NS(connected=self.connected, trade_allowed=False)

    def account_info(self):
        if not self.has_account:
            return None
        return NS(login=self.login, server="Fake-Server", company="Fake Broker", currency=self.currency, leverage=500, trade_mode=0, margin_mode=2,
                  password="SHOULD_NEVER_LEAK")

    def symbols_get(self):
        return [NS(name=n) for n in self.symbols]

    def symbol_info(self, n):
        return self.symbols.get(n)

    def symbol_info_tick(self, n):
        if self.tick_fail:
            return None
        return NS(bid=self.bid, ask=self.ask, time=1700000000)

    # forbidden trading functions that must never be reachable
    def order_send(self, *a, **k):
        raise AssertionError("order_send must never be called")

    order_check = positions_get = orders_get = login = order_send
