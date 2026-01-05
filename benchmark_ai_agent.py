#!/usr/bin/env python3
"""
Benchmark script to identify AI Agent bottlenecks on Mac Mini M4.
Tests: CPU usage, parallel efficiency, data loading speed.
"""

import os
import time
import json
from engine.data_loader import load_csv
from engine.run_builder import build_runs

def benchmark_data_loading():
    """Test CSV loading speed"""
    print("\n=== BENCHMARK 1: Data Loading Speed ===")
    symbols = ["BTCUSDT"]
    timeframes = ["15m", "30m", "1h", "4h"]

    start = time.time()
    for tf in timeframes:
        for _ in range(3):  # Load 3 times to test caching
            df = load_csv(symbols[0], tf)
            print(f"  {tf}: {len(df)} rows loaded")
    elapsed = time.time() - start
    print(f"✓ Data loading took: {elapsed:.2f}s")
    print(f"  (Should be <1s with caching)")
    return elapsed

def benchmark_run_builder():
    """Test run generation speed"""
    print("\n=== BENCHMARK 2: Run Builder Speed ===")

    # Small config for testing
    cfg = {
        "symbols": ["BTCUSDT"],
        "timeframes": ["15m", "30m", "1h"],
        "strategy": {
            "type": "ema_cross",
            "params": {
                "fast_period": [5, 10, 15],
                "slow_period": [20, 30, 40]
            }
        },
        "range": {
            "from": "2024-01-01",
            "to": "2024-12-31"
        }
    }

    start = time.time()
    runs = build_runs(cfg)
    elapsed = time.time() - start

    total_runs = len(runs)
    print(f"✓ Generated {total_runs} runs in {elapsed:.2f}s")
    print(f"  Expected: 3 TFs × 3 fast × 3 slow = 27 runs")
    return total_runs, elapsed

def benchmark_cpu_info():
    """Display CPU info and current usage"""
    print("\n=== SYSTEM INFO ===")
    print(f"CPU Count (logical): {os.cpu_count()}")
    # Note: psutil not available, using basic os module
    print(f"System: macOS (Mac Mini M4 expected)")

def test_parallel_scaling():
    """Test if parallel execution actually scales"""
    print("\n=== BENCHMARK 3: Parallel Scaling Test ===")
    print("Testing if ProcessPoolExecutor scales properly...")

    from concurrent.futures import ProcessPoolExecutor
    import time

    def dummy_work(n):
        """Simulate CPU-intensive work"""
        result = 0
        for i in range(1000000):
            result += i ** 2
        return result

    # Serial execution
    print("\n  Serial execution (baseline):")
    start = time.time()
    for i in range(8):
        dummy_work(i)
    serial_time = time.time() - start
    print(f"    Time: {serial_time:.2f}s")

    # Parallel execution
    print("\n  Parallel execution (8 workers):")
    start = time.time()
    with ProcessPoolExecutor(max_workers=8) as executor:
        list(executor.map(dummy_work, range(8)))
    parallel_time = time.time() - start
    print(f"    Time: {parallel_time:.2f}s")

    speedup = serial_time / parallel_time
    print(f"\n✓ Speedup: {speedup:.2f}x")

    if speedup < 2:
        print("  ⚠️  WARNING: Poor parallel scaling!")
        print("     Possible issues:")
        print("     - GIL contention (but ProcessPool should avoid this)")
        print("     - I/O bottleneck (CSV loading)")
        print("     - Overhead from process spawning")
    elif speedup >= 6:
        print("  ✓ Excellent parallel scaling!")
    else:
        print("  ✓ Good parallel scaling")

    return speedup

def estimate_ai_agent_time():
    """Estimate total AI Agent execution time"""
    print("\n=== BENCHMARK 4: AI Agent Time Estimate ===")

    # Typical optimization scenario
    num_strategies = 100  # param combinations
    num_symbols = 1
    num_timeframes = 5
    total_runs = num_strategies * num_symbols * num_timeframes

    # Benchmark single backtest
    print(f"  Estimating for {total_runs} backtests...")
    print("  (100 param combos × 1 symbol × 5 timeframes)")

    from engine.backtest_engine import run_backtest

    # Small test config
    test_cfg = {
        "meta": {
            "symbols": ["BTCUSDT"],
            "timeframe": "1h"
        },
        "strategy": {
            "type": "ema_cross",
            "params": {
                "fast_period": 10,
                "slow_period": 30
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

    print("\n  Running single backtest benchmark...")
    start = time.time()
    result = run_backtest(test_cfg)
    single_time = time.time() - start
    print(f"    Single backtest: {single_time:.3f}s")
    print(f"    Trades: {len(result.get('trades', []))}")

    # Calculate estimates
    workers = max(1, os.cpu_count() - 1)

    serial_estimate = total_runs * single_time
    parallel_estimate = (total_runs / workers) * single_time * 1.1  # 10% overhead

    print(f"\n  Time estimates:")
    print(f"    Serial (old code): {serial_estimate:.1f}s ({serial_estimate/60:.1f} min)")
    print(f"    Parallel ({workers} workers): {parallel_estimate:.1f}s ({parallel_estimate/60:.1f} min)")
    print(f"    Expected speedup: {serial_estimate/parallel_estimate:.1f}x")

    return single_time, parallel_estimate

def main():
    print("=" * 60)
    print("AI AGENT PERFORMANCE BENCHMARK - Mac Mini M4")
    print("=" * 60)

    # System info
    benchmark_cpu_info()

    # Test 1: Data loading
    data_time = benchmark_data_loading()

    # Test 2: Run builder
    num_runs, builder_time = benchmark_run_builder()

    # Test 3: Parallel scaling
    speedup = test_parallel_scaling()

    # Test 4: AI Agent estimate
    single_bt_time, estimated_time = estimate_ai_agent_time()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY & DIAGNOSIS")
    print("=" * 60)

    print(f"\n✓ CPU Cores: {os.cpu_count()}")
    print(f"✓ Parallel speedup: {speedup:.2f}x")
    print(f"✓ Single backtest: {single_bt_time:.3f}s")
    print(f"✓ Estimated AI Agent time: {estimated_time:.1f}s ({estimated_time/60:.1f} min)")

    # Diagnose bottlenecks
    print("\n📊 BOTTLENECK ANALYSIS:")

    if data_time > 2:
        print("  ⚠️  Data loading is slow (>2s)")
        print("     → Check if CSVs are on slow disk (external drive?)")
        print("     → Consider using SSD")
    else:
        print("  ✓ Data loading is fast")

    if speedup < 3:
        print("  ⚠️  Parallel execution is not scaling well")
        print("     → May be I/O bound (CSV loading in workers)")
        print("     → Check if DataFrame caching is working")
        print("     → Verify ProcessPoolExecutor is being used")
    elif speedup < 6:
        print("  ⚠️  Parallel execution scaling is moderate")
        print("     → Some overhead, but acceptable")
    else:
        print("  ✓ Parallel execution scaling is excellent")

    if single_bt_time > 0.5:
        print("  ⚠️  Single backtest is slow (>0.5s)")
        print("     → Check strategy complexity (too many indicators?)")
        print("     → Check data size (very long period?)")
    else:
        print("  ✓ Single backtest speed is good")

    if estimated_time > 300:  # >5 minutes
        print(f"\n⚠️  Total time ({estimated_time/60:.1f} min) is still slow!")
        print("   Recommendations:")
        print("   1. Reduce param combinations (use coarse grid first)")
        print("   2. Reduce timeframe count (test fewer TFs)")
        print("   3. Check if Node.js server has overhead")
        print("   4. Run Python directly (skip HTTP)")
    else:
        print(f"\n✓ Total time ({estimated_time/60:.1f} min) looks reasonable")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
