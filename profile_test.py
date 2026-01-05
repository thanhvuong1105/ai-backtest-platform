import json
from engine.backtest_engine import run_backtest

with open('test_strategy.json') as f:
    cfg = json.load(f)

print("Running profiled backtest...")
result = run_backtest(cfg)
print(f"Trades: {len(result['trades'])}")
