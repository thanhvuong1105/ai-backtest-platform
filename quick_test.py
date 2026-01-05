#!/usr/bin/env python3
"""Quick test with just 10 backtests"""

import time

def main():
    from engine.ai_agent import ai_recommend

    cfg = {
        "symbols": ["BTCUSDT"],
        "timeframes": ["15m", "30m", "1h", "2h", "4h"],  # 5 timeframes
        "strategy": {
            "type": "ema_cross",
            "params": {
                "emaFast": [5, 8, 10, 13],      # 4 values
                "emaSlow": [20, 30, 40, 50]      # 4 values
            }
        },
        "range": {
            "from": "2024-01-01",
            "to": "2024-12-31"
        },
        "properties": {
            "initialCapital": 10000,
            "orderSize": {"value": 100, "type": "percent"}
        },
        "topN": 10
    }

    # 1 symbol × 5 TF × 4 fast × 4 slow = 80 backtests

    print("Testing with 80 backtests (OPTIMIZED)...")
    start = time.time()
    result = ai_recommend(cfg)
    elapsed = time.time() - start

    total_runs = result.get('total', 80)
    print(f"\n✅ Completed in {elapsed:.2f}s")
    print(f"Average: {elapsed/total_runs:.3f}s per backtest")
    print(f"Total runs: {total_runs}")

    if result.get('best'):
        summary = result['best'].get('summary', {})
        print(f"Best PF: {summary.get('profitFactor', 0):.2f}")

if __name__ == "__main__":
    main()
