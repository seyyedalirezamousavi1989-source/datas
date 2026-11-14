"""Export MT5 candles for many symbols in the format the AURORA dataset builders read.

Run on the Windows machine where MetaTrader 5 is installed, logged in and has the symbols in Market Watch:
    pip install MetaTrader5 pandas
    python tools/mt5_export_candles.py --symbols EURUSD,GBPUSD,USDJPY,XAUUSD --out candles
Output: <out>/<SYMBOL>_<M15|H1|H4|Daily>_normalized.csv with columns
    time (broker SERVER time, naive), open, high, low, close, tick_volume, real_volume, spread (points), spread_known
Then:   python -m local_chat.aurora_forex_v2 --candles <out> --symbols EURUSD,GBPUSD,USDJPY,XAUUSD --out training_data/aurora_forex_v2
The last (still forming) bar is dropped. Use --suffix if your broker names symbols like EURUSD.m (the file keeps the plain name).
"""
import argparse
import os
import sys

TF = {"M15": ("TIMEFRAME_M15", 140000), "H1": ("TIMEFRAME_H1", 45000), "H4": ("TIMEFRAME_H4", 12000), "Daily": ("TIMEFRAME_D1", 3000)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--out", default="candles")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply the default bar counts (more history, if the broker has it)")
    a = ap.parse_args()
    try:
        import MetaTrader5 as mt5
        import pandas as pd
    except ImportError:
        sys.exit("pip install MetaTrader5 pandas (Windows only)")
    if not mt5.initialize():
        sys.exit(f"MT5 initialize failed: {mt5.last_error()}")
    os.makedirs(a.out, exist_ok=True)
    for sym in [s.strip().upper() for s in a.symbols.split(",") if s.strip()]:
        broker = sym + a.suffix
        if not mt5.symbol_select(broker, True):
            print(f"skip {sym}: cannot select {broker}")
            continue
        for name, (attr, count) in TF.items():
            rates = mt5.copy_rates_from_pos(broker, getattr(mt5, attr), 0, int(count * a.scale))
            if rates is None or len(rates) < 300:
                print(f"skip {sym} {name}: too few bars ({0 if rates is None else len(rates)})")
                continue
            d = pd.DataFrame(rates).iloc[:-1]                        # drop the forming bar
            d["time"] = pd.to_datetime(d["time"], unit="s")          # server time, naive (same clock as the broker chart)
            d["spread_known"] = 1
            d = d[["time", "open", "high", "low", "close", "tick_volume", "real_volume", "spread", "spread_known"]]
            path = os.path.join(a.out, f"{sym}_{name}_normalized.csv")
            d.to_csv(path, index=False)
            print(f"{sym} {name}: {len(d)} bars {d['time'].iloc[0]} .. {d['time'].iloc[-1]} -> {path}")
    mt5.shutdown()


if __name__ == "__main__":
    main()
