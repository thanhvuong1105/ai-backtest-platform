# engine/optimize_ema.py
import json
from backtest_engine import run_backtest

def optimize_ema(
    symbol="BTCUSDT",
    timeframe="30m",
    fast_range=(5, 50),
    slow_range=(5, 100),
    fast_step=2,
    slow_step=5,
    min_trades=50
):
    results = []

    for fast in range(fast_range[0], fast_range[1] + 1, fast_step):
        for slow in range(slow_range[0], slow_range[1] + 1, slow_step):
            if fast >= slow:
                continue

            strategy = {
                "meta": {
                    "symbols": [symbol],
                    "timeframe": timeframe
                },
                "indicators": {
                    "emaFast": {"type": "ema", "length": fast},
                    "emaSlow": {"type": "ema", "length": slow}
                },
                "costs": {
                    "fee": 0.0004,
                    "slippage": 0.0001
                }
            }

            try:
                result = run_backtest(strategy)
                summary = result["summary"]

                if summary["totalTrades"] < min_trades:
                    continue

                results.append({
                    "params": {
                        "emaFast": fast,
                        "emaSlow": slow
                    },
                    "summary": summary
                })

                print(
                    f"✔ EMA {fast}/{slow} | "
                    f"PNL {summary['netProfit']:.2f} | "
                    f"DD {summary['maxDrawdownPct']:.2f}% | "
                    f"PF {summary['profitFactor']:.2f} | "
                    f"WR {summary['winrate']:.2f}%"
                )

            except Exception as e:
                print(f"✖ EMA {fast}/{slow} failed:", e)

    return results


if __name__ == "__main__":
    results = optimize_ema()

    print("\n===== TOP 5 (by Net Profit) =====")
    top = sorted(
        results,
        key=lambda x: x["summary"]["netProfit"],
        reverse=True
    )[:5]

    print(json.dumps(top, indent=2))
