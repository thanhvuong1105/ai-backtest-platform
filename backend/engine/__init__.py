"""
AI Backtest Platform Engine

This package provides the core backtest engine functionality.
Import functions directly from this module to avoid subprocess overhead.

Public API:
- run_backtest(config, progress_cb=None): Run a single backtest
- run_optimizer(config, progress_cb=None): Run optimization
- run_ai_agent(config, progress_cb=None): Run AI agent
- generate_chart_data(config): Generate chart data for visualization
- load_data(symbol, timeframe): Load OHLCV data
"""

from typing import Any, Callable, Dict, Optional

# Import core functions
from .backtest_engine import run_backtest as _run_backtest_core
from .optimizer import optimize as _optimize_core
from .ai_agent import ai_recommend as _ai_recommend_core
from .chart_data import get_chart_data as _get_chart_data_core
from .data_loader import load_csv


def run_backtest(
    config: Dict[str, Any],
    progress_cb: Optional[Callable[[str, int, int, str, Dict], None]] = None
) -> Dict[str, Any]:
    """
    Run a single backtest.

    Args:
        config: Strategy configuration dict with:
            - meta: { symbols: [str], timeframe: str }
            - strategy: { type: str, params: dict }
            - initial_equity: float (optional, default 10000)
            - properties: dict (optional)
            - costs: { fee: float, slippage: float } (optional)
            - range: { from: str, to: str } (optional)
        progress_cb: Optional callback function(job_id, progress, total, status, extra)

    Returns:
        Backtest result dict with:
            - meta: { strategyId, strategyName, symbol, timeframe }
            - summary: { totalTrades, winrate, profitFactor, ... }
            - equityCurve: [{ time, equity }, ...]
            - trades: [{ entry_time, exit_time, pnl, ... }, ...]
    """
    return _run_backtest_core(config)


def run_optimizer(
    config: Dict[str, Any],
    job_id: str = "",
    progress_cb: Optional[Callable[[str, int, int, str, Dict], None]] = None
) -> Dict[str, Any]:
    """
    Run parameter optimization.

    Args:
        config: Optimization configuration dict with:
            - symbols: [str]
            - timeframes: [str]
            - strategy: { type: str, params: dict (with arrays for grid search) }
            - filters: { minPF, minTrades, maxDD } (optional)
            - minTFAgree: int (optional, default 2)
            - stability: dict (optional)
            - topN: int (optional, default 50)
        job_id: Job ID for progress tracking
        progress_cb: Optional callback function(job_id, progress, total, status, extra)

    Returns:
        Optimization result dict with:
            - stats: { totalRuns, passedRuns, rejectedRuns }
            - best: Best strategy result or None
            - top: Top N strategies
            - all: All results
            - passed: Filtered results
    """
    return _optimize_core(config, top_n=config.get("topN", 50))


def run_ai_agent(
    config: Dict[str, Any],
    job_id: str = "",
    progress_cb: Optional[Callable[[str, int, int, str, Dict], None]] = None
) -> Dict[str, Any]:
    """
    Run AI agent optimization with intelligent parameter selection.

    Args:
        config: AI agent configuration dict (same as optimize config)
        job_id: Job ID for progress tracking
        progress_cb: Optional callback function(job_id, progress, total, status, extra)
            Called with: progress_cb(job_id, progress, total, "running", {})

    Returns:
        AI agent result dict with:
            - success: bool
            - fallback: bool (True if using fallback due to no passes)
            - message: str or None
            - best: Best strategy result
            - alternatives: Top 3 alternatives
            - comment: AI-generated explanation
            - top: Top N strategies
            - all: All results (limited)
            - total: Total number of runs
    """
    # Import here to avoid circular imports
    import json
    import sys
    import os

    # Create a wrapper that calls progress_cb if provided
    original_stdout = sys.stdout

    if progress_cb:
        class ProgressCapture:
            """Capture stdout to intercept progress updates."""
            def __init__(self, job_id, callback, original):
                self.job_id = job_id
                self.callback = callback
                self.original = original
                self.buffer = ""

            def write(self, text):
                self.buffer += text
                # Process complete lines
                while "\n" in self.buffer:
                    line, self.buffer = self.buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if "progress" in data and "total" in data:
                            self.callback(
                                self.job_id,
                                data["progress"],
                                data["total"],
                                "running",
                                {}
                            )
                    except (json.JSONDecodeError, KeyError):
                        pass

            def flush(self):
                pass

        # Enable progress output
        os.environ["AI_PROGRESS"] = "1"
        sys.stdout = ProgressCapture(job_id, progress_cb, original_stdout)

    try:
        result = _ai_recommend_core(config)
        return result
    finally:
        sys.stdout = original_stdout
        if progress_cb:
            os.environ.pop("AI_PROGRESS", None)


def generate_chart_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate chart data for visualization.

    Args:
        config: Chart data configuration dict with:
            - symbol: str
            - timeframe: str
            - strategy: { type: str, params: dict }
            - capital: { initial: float, orderPct: float } (optional)
            - risk: { pyramiding: int, commission: float, slippage: float } (optional)
            - range: { from: str, to: str } (optional)

    Returns:
        Chart data dict with:
            - symbol: str
            - timeframe: str
            - strategy: dict
            - candles: [{ time, open, high, low, close, volume }, ...]
            - indicators: { indicator_name: [{ time, value }, ...], ... }
            - markers: [{ time, position, color, shape, text }, ...]
            - trades: [{ id, entry_time, exit_time, ... }, ...]
            - summary: { totalTrades, winrate, ... }
    """
    return _get_chart_data_core(config)


def load_data(symbol: str, timeframe: str):
    """
    Load OHLCV data for a symbol and timeframe.

    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        timeframe: Timeframe string (e.g., "1h", "4h", "1d")

    Returns:
        pandas DataFrame with columns: time, open, high, low, close, volume
    """
    return load_csv(symbol, timeframe)


# Export public API
__all__ = [
    "run_backtest",
    "run_optimizer",
    "run_ai_agent",
    "generate_chart_data",
    "load_data",
]
