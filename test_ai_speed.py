#!/usr/bin/env python3
"""
Quick benchmark to test AI Agent speed on Mac Mini M4
"""

import os
import time
import json

def test_small_optimization():
    """Test with a realistic small optimization"""
    print("=" * 60)
    print("AI AGENT SPEED TEST - Mac Mini M4")
    print("=" * 60)
    print(f"\nCPU Cores: {os.cpu_count()}")
    print(f"Workers used: {max(1, os.cpu_count() - 1)}")

    # Realistic test config
    cfg = {
        "symbols": ["BTCUSDT"],
        "timeframes": ["15m", "30m", "1h", "2h", "4h"],
        "strategy": {
            "type": "ema_cross",
            "params": {
                "fast_period": [5, 8, 10, 13, 15],
                "slow_period": [20, 25, 30, 35, 40]
            }
        },
        "range": {
            "from": "2024-01-01",
            "to": "2024-12-31"
        },
        "properties": {
            "initialCapital": 10000,
            "orderSize": {"value": 100, "type": "percent"},
            "commission": {"value": 0.1, "type": "percent"}
        },
        "topN": 20,
        "minTFAgree": 2,
        "filters": {
            "minPF": 1.0,
            "minTrades": 10,
            "maxDD": 50
        }
    }

    # Calculate total runs
    total_runs = (
        len(cfg["symbols"]) *
        len(cfg["timeframes"]) *
        len(cfg["strategy"]["params"]["fast_period"]) *
        len(cfg["strategy"]["params"]["slow_period"])
    )

    print(f"\n📊 Test Configuration:")
    print(f"  Symbols: {len(cfg['symbols'])}")
    print(f"  Timeframes: {len(cfg['timeframes'])}")
    print(f"  Fast periods: {len(cfg['strategy']['params']['fast_period'])}")
    print(f"  Slow periods: {len(cfg['strategy']['params']['slow_period'])}")
    print(f"  Total backtests: {total_runs}")

    print(f"\n⏳ Running AI Agent...")
    print(f"  (This will take a while - testing parallel performance)")

    # Import and run
    from engine.ai_agent import ai_recommend

    start = time.time()
    result = ai_recommend(cfg)
    elapsed = time.time() - start

    # Results
    print(f"\n✅ COMPLETED in {elapsed:.1f}s ({elapsed/60:.2f} minutes)")
    print(f"\n📈 Results:")
    print(f"  Total runs completed: {result.get('total', 0)}")
    print(f"  Passed filters: {len(result.get('top', []))}")
    print(f"  Best strategy:")
    if result.get('best'):
        best = result['best']
        print(f"    Symbol: {best.get('symbol')}")
        print(f"    Timeframe: {best.get('timeframe')}")
        print(f"    Params: {best.get('params')}")
        summary = best.get('summary', {})
        print(f"    PF: {summary.get('profitFactor', 0):.2f}")
        print(f"    WR: {summary.get('winrate', 0):.1f}%")
        print(f"    DD: {summary.get('maxDrawdownPct', 0):.1f}%")
        print(f"    Score: {summary.get('score', 0):.2f}")

    # Performance analysis
    avg_time = elapsed / total_runs
    throughput = total_runs / elapsed

    print(f"\n⚡ Performance Metrics:")
    print(f"  Avg time per backtest: {avg_time:.3f}s")
    print(f"  Throughput: {throughput:.1f} backtests/sec")
    print(f"  Parallel speedup estimate: {avg_time / (avg_time / os.cpu_count()):.1f}x")

    # Compare with expected
    workers = max(1, os.cpu_count() - 1)
    ideal_time = (total_runs / workers) * avg_time * 1.1  # 10% overhead

    print(f"\n💡 Analysis:")
    if elapsed < ideal_time * 1.5:
        print(f"  ✅ GOOD: Parallel execution is working well!")
        print(f"     Actual: {elapsed:.1f}s vs Ideal: {ideal_time:.1f}s")
    else:
        print(f"  ⚠️  SLOW: Performance below expected")
        print(f"     Actual: {elapsed:.1f}s vs Expected: {ideal_time:.1f}s")
        print(f"\n  Possible bottlenecks:")
        print(f"  1. Data loading overhead")
        print(f"  2. Process spawning overhead")
        print(f"  3. Node.js server overhead (if running via HTTP)")
        print(f"  4. Complex strategy calculations")

    if avg_time > 0.5:
        print(f"\n  💭 Single backtest is slow ({avg_time:.3f}s > 0.5s):")
        print(f"     - Check data range (too long?)")
        print(f"     - Check strategy complexity (too many indicators?)")
    else:
        print(f"\n  ✅ Single backtest speed is good ({avg_time:.3f}s)")

    print("\n" + "=" * 60)

    return result, elapsed

if __name__ == "__main__":
    result, elapsed = test_small_optimization()
