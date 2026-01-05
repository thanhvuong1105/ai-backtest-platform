#!/usr/bin/env python3
"""Test single backtest speed"""

import time
from engine.backtest_engine import run_backtest

cfg = {
    "meta": {
        "symbols": ["BTCUSDT"],
        "timeframe": "1h"
    },
    "strategy": {
        "type": "ema_cross",
        "params": {
            "emaFast": 10,
            "emaSlow": 30
        }
    },
    "initial_equity": 10000,
    "range": {
        "from": "2024-01-01",
        "to": "2024-12-31"
    },
    "properties": {
        "initialCapital": 10000,
        "orderSize": {"value": 100, "type": "percent"}
    }
}

print("Testing SINGLE backtest speed...")
start = time.time()
result = run_backtest(cfg)
elapsed = time.time() - start

print(f"\n✓ Completed in {elapsed:.3f}s")
print(f"  Trades: {len(result.get('trades', []))}")
print(f"  Equity points: {len(result.get('equityCurve', []))}")
print(f"  PF: {result['summary'].get('profitFactor', 0):.2f}")

if elapsed > 2:
    print(f"\n⚠️  SLOW! Single backtest took {elapsed:.1f}s")
    print("   Possible causes:")
    print("   1. Data loading is slow (CSV on slow disk?)")
    print("   2. Too many candles (long period?)")
    print("   3. Complex indicators")
elif elapsed > 0.5:
    print(f"\n⚠️  Moderate speed ({elapsed:.3f}s)")
else:
    print(f"\n✓ Good speed!")
