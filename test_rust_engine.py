#!/usr/bin/env python3
"""
Test script to compare Rust and Python backtest engines.
Ensures results match exactly.
"""

import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any

# Test configuration
TEST_CONFIG = {
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


def test_indicators():
    """Test indicator calculations match between Rust and Python."""
    print("\n" + "=" * 60)
    print("Testing Indicators")
    print("=" * 60)

    try:
        from engine.rust_bridge import calc_ema_rust, calc_rsi_rust, is_rust_available
        from engine.indicators import ema

        if not is_rust_available():
            print("Rust engine not available, skipping indicator tests")
            return False

        # Test data
        data = np.array([44.0, 44.5, 43.5, 44.0, 44.5, 44.0, 44.5, 44.0, 44.0, 43.5,
                        44.0, 44.5, 44.5, 45.0, 45.0, 45.5, 46.0, 45.5, 46.0, 46.5,
                        47.0, 46.5, 47.0, 47.5, 47.0, 47.5, 48.0, 47.5, 48.0, 48.5])

        # Test EMA
        print("\nTesting EMA...")
        py_ema = ema(pd.Series(data), 10).values
        rust_ema = calc_ema_rust(data, 10)

        ema_diff = np.abs(py_ema - rust_ema)
        ema_max_diff = np.nanmax(ema_diff)
        print(f"  EMA max difference: {ema_max_diff:.10f}")

        if ema_max_diff < 1e-6:
            print("  EMA: PASSED")
        else:
            print("  EMA: FAILED - Results don't match!")
            return False

        # Test RSI
        print("\nTesting RSI...")
        rust_rsi = calc_rsi_rust(data, 14)
        print(f"  RSI values (last 5): {rust_rsi[-5:]}")

        # RSI should be between 0 and 100
        valid_rsi = all(0 <= x <= 100 for x in rust_rsi if not np.isnan(x))
        if valid_rsi:
            print("  RSI: PASSED (values in valid range)")
        else:
            print("  RSI: FAILED - Values out of range!")
            return False

        return True

    except Exception as e:
        print(f"Error testing indicators: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backtest_comparison():
    """Compare Rust and Python backtest results."""
    print("\n" + "=" * 60)
    print("Testing Backtest Engine Comparison")
    print("=" * 60)

    try:
        from engine.rust_bridge import run_backtest_rust, is_rust_available
        from engine.backtest_engine import run_backtest
        from engine.data_loader import load_csv

        if not is_rust_available():
            print("Rust engine not available, skipping comparison test")
            return False

        # Load data
        print("\nLoading data...")
        df = load_csv("BTCUSDT", "1h")
        print(f"  Loaded {len(df)} candles")

        # Run Python backtest
        print("\nRunning Python backtest...")
        start = time.time()
        py_result = run_backtest(TEST_CONFIG)
        py_time = time.time() - start
        print(f"  Python time: {py_time:.3f}s")
        print(f"  Python trades: {len(py_result['trades'])}")

        # Run Rust backtest
        print("\nRunning Rust backtest...")
        start = time.time()
        rust_result = run_backtest_rust(TEST_CONFIG, df)
        rust_time = time.time() - start
        print(f"  Rust time: {rust_time:.3f}s")
        print(f"  Rust trades: {len(rust_result['trades'])}")

        # Compare results
        print("\n" + "-" * 40)
        print("Comparison Results")
        print("-" * 40)

        py_summary = py_result["summary"]
        rust_summary = rust_result["summary"]

        fields = [
            ("totalTrades", "Total Trades"),
            ("winTrades", "Win Trades"),
            ("lossTrades", "Loss Trades"),
            ("winrate", "Win Rate"),
            ("profitFactor", "Profit Factor"),
            ("netProfit", "Net Profit"),
            ("maxDrawdownPct", "Max Drawdown %"),
        ]

        all_match = True
        for key, label in fields:
            py_val = py_summary.get(key, 0)
            rust_val = rust_summary.get(key, 0)

            # Allow small floating point differences
            if isinstance(py_val, float):
                match = abs(py_val - rust_val) < 0.01
            else:
                match = py_val == rust_val

            status = "OK" if match else "DIFF"
            if not match:
                all_match = False

            print(f"  {label:20s}: Python={py_val:>10.2f}, Rust={rust_val:>10.2f} [{status}]")

        # Speed comparison
        print("\n" + "-" * 40)
        print("Performance")
        print("-" * 40)
        speedup = py_time / rust_time if rust_time > 0 else 0
        print(f"  Python: {py_time:.3f}s")
        print(f"  Rust:   {rust_time:.3f}s")
        print(f"  Speedup: {speedup:.1f}x")

        if all_match:
            print("\nRESULT: PASSED - Results match!")
            return True
        else:
            print("\nRESULT: FAILED - Results don't match!")
            return False

    except Exception as e:
        print(f"Error in backtest comparison: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_performance():
    """Test batch backtest performance."""
    print("\n" + "=" * 60)
    print("Testing Batch Performance")
    print("=" * 60)

    try:
        from engine.rust_bridge import run_batch_backtests_rust, is_rust_available
        from engine.data_loader import load_csv

        if not is_rust_available():
            print("Rust engine not available, skipping batch test")
            return False

        # Load data
        df = load_csv("BTCUSDT", "1h")

        # Create multiple configs with different parameters
        configs = []
        for fast in [5, 8, 10, 13, 15, 20]:
            for slow in [20, 30, 40, 50, 60, 80]:
                if fast < slow:
                    config = {
                        "meta": {"symbols": ["BTCUSDT"], "timeframe": "1h"},
                        "strategy": {
                            "type": "ema_cross",
                            "params": {"emaFast": float(fast), "emaSlow": float(slow)}
                        },
                        "initial_equity": 10000.0,
                        "range": {"from": "2024-01-01", "to": "2024-12-31"},
                        "properties": {
                            "initialCapital": 10000,
                            "orderSize": {"value": 100.0, "type": "percent"}
                        }
                    }
                    configs.append(config)

        print(f"\nRunning {len(configs)} backtests in parallel...")

        start = time.time()
        results = run_batch_backtests_rust(configs, df)
        elapsed = time.time() - start

        print(f"\nResults:")
        print(f"  Total backtests: {len(results)}")
        print(f"  Total time: {elapsed:.3f}s")
        print(f"  Average per backtest: {elapsed/len(results)*1000:.1f}ms")

        # Show top 5 by profit factor
        print("\nTop 5 by Profit Factor:")
        sorted_results = sorted(results, key=lambda x: x["summary"].get("profitFactor", 0), reverse=True)
        for i, r in enumerate(sorted_results[:5]):
            params = configs[results.index(r)]["strategy"]["params"]
            pf = r["summary"].get("profitFactor", 0)
            trades = r["summary"].get("totalTrades", 0)
            print(f"  {i+1}. EMA({int(params['emaFast'])}/{int(params['emaSlow'])}): PF={pf:.2f}, Trades={trades}")

        return True

    except Exception as e:
        print(f"Error in batch test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("Rust Backtest Engine Test Suite")
    print("=" * 60)

    # Check if Rust engine is available
    try:
        from engine.rust_bridge import is_rust_available
        if not is_rust_available():
            print("\nRust engine not available!")
            print("To build, run:")
            print("  ./build_rust.sh")
            print("\nOr manually:")
            print("  cd rust_engine")
            print("  maturin develop --release")
            return
    except ImportError as e:
        print(f"\nImport error: {e}")
        print("Make sure you're in the project directory")
        return

    # Run tests
    results = []

    results.append(("Indicators", test_indicators()))
    results.append(("Backtest Comparison", test_backtest_comparison()))
    results.append(("Batch Performance", test_batch_performance()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")


if __name__ == "__main__":
    main()
