# engine/rust_bridge.py
"""
Python bridge to Rust backtest engine.
Falls back to Python implementation if Rust module not available.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

# Try to import Rust module
RUST_AVAILABLE = False
try:
    import backtest_engine as rust_engine
    RUST_AVAILABLE = True
    print("Rust backtest engine loaded successfully!")
except ImportError:
    rust_engine = None
    # Silently fall back to Python


def is_rust_available() -> bool:
    """Check if Rust engine is available."""
    return RUST_AVAILABLE


def run_backtest_rust(strategy_config: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run backtest using Rust engine.

    Args:
        strategy_config: Strategy configuration dict (same format as Python engine)
        df: DataFrame with OHLCV data (columns: time, open, high, low, close)

    Returns:
        Backtest result dict with meta, summary, equityCurve, trades
    """
    if not RUST_AVAILABLE:
        raise ImportError("Rust backtest engine not available. Run 'cd rust_engine && maturin develop --release'")

    # Prepare data arrays
    times = df["time"].astype(str).tolist()
    opens = df["open"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)

    # Run Rust backtest
    result = rust_engine.run_backtest_py(
        strategy_config,
        times,
        opens,
        highs,
        lows,
        closes
    )

    return result


def run_batch_backtests_rust(
    configs: List[Dict[str, Any]],
    df: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Run multiple backtests in parallel using Rust engine.

    Args:
        configs: List of strategy configurations
        df: DataFrame with OHLCV data

    Returns:
        List of backtest results
    """
    if not RUST_AVAILABLE:
        raise ImportError("Rust backtest engine not available. Run 'cd rust_engine && maturin develop --release'")

    # Prepare data arrays
    times = df["time"].astype(str).tolist()
    opens = df["open"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)

    # Run batch backtests in parallel
    results = rust_engine.run_batch_backtests_py(
        configs,
        times,
        opens,
        highs,
        lows,
        closes
    )

    return results


def calc_ema_rust(data: np.ndarray, period: int) -> np.ndarray:
    """Calculate EMA using Rust (for testing)."""
    if not RUST_AVAILABLE:
        raise ImportError("Rust engine not available")
    return rust_engine.calc_ema_py(data.astype(np.float64), period)


def calc_rsi_rust(data: np.ndarray, period: int) -> np.ndarray:
    """Calculate RSI using Rust (for testing)."""
    if not RUST_AVAILABLE:
        raise ImportError("Rust engine not available")
    return rust_engine.calc_rsi_py(data.astype(np.float64), period)


# Hybrid backtest function - uses Rust if available, else Python
def run_backtest_hybrid(strategy_config: Dict[str, Any], df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Run backtest using Rust if available, otherwise fall back to Python.

    Args:
        strategy_config: Strategy configuration dict
        df: Optional DataFrame (if not provided, will load from config)

    Returns:
        Backtest result dict
    """
    if RUST_AVAILABLE:
        # Use Rust engine
        if df is None:
            from engine.data_loader import load_csv
            symbol = strategy_config["meta"]["symbols"][0]
            timeframe = strategy_config["meta"]["timeframe"]
            df = load_csv(symbol, timeframe)

        return run_backtest_rust(strategy_config, df)
    else:
        # Fall back to Python engine
        from engine.backtest_engine import run_backtest
        return run_backtest(strategy_config)
