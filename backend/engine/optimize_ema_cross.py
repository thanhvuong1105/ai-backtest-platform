import json
from backtest_engine import run_backtest

BASE_STRATEGY = {
    "meta": {
        "symbols": ["BTCUSDT"],
        "timeframe": "30m"
    },
    "indicators": {
        "emaFast": { "type": "ema", "length": 12 },
        "emaSlow": { "type": "ema", "length": 25 }
    }
}

fast_list = [8, 12, 16]
slow_list = [20, 25, 30]

results = []

for fast in fast_list:
    for slow in slow_list:
        if fast >= slow:
            continue

        strategy = json.loads(json.dumps(BASE_STRATEGY))
        strategy["indicators"]["emaFast"]["length"] = fast
        strategy["indicators"]["emaSlow"]["length"] = slow

        result = run_backtest(strategy)

        results.append({
            "emaFast": fast,
            "emaSlow": slow,
            "netProfit": result["summary"]["netProfit"],
            "trades": result["summary"]["totalTrades"]
        })

# sort by net profit
results = sorted(results, key=lambda x: x["netProfit"], reverse=True)

for r in results:
    print(r)
