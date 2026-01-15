"""
AI Backtest Platform Engine

This package provides the core backtest engine functionality.
Import functions directly from this module to avoid subprocess overhead.

Public API:
- run_backtest(config, progress_cb=None): Run a single backtest
- run_optimizer(config, progress_cb=None): Run optimization
- run_ai_agent(config, progress_cb=None): Run AI agent
- run_quant_brain(config, progress_cb=None): Run Quant Brain v1
- run_quant_brain_v2(config, progress_cb=None): Run Quant Brain v2.0
- generate_chart_data(config): Generate chart data for visualization
- load_data(symbol, timeframe): Load OHLCV data

Quant Brain v2.0 Components:
- market_profile_v2: 32 indicators across 6 clusters
- brain_score: Multi-criteria BrainScore calculation
- genome_optimizer_v2: Advanced GA with adaptive mutation
- robustness_testing: Walk-Forward, Monte Carlo, Sensitivity tests
- param_memory_v2: Enhanced memory with similarity search
"""

from typing import Any, Callable, Dict, Optional

# Import core functions
from .backtest_engine import run_backtest as _run_backtest_core
from .optimizer import optimize as _optimize_core
# Use original AI agent (vectorized version disabled for accuracy)
from .ai_agent import ai_recommend as _ai_recommend_core
# Quant Brain v1 - Self-learning optimization engine
from .quant_brain import quant_brain_recommend as _quant_brain_core
# Quant Brain v2.0 - Advanced optimization with robustness testing
from .quant_brain_v2 import quant_brain_optimize_v2_sync as _quant_brain_v2_core
from .chart_data import get_chart_data as _get_chart_data_core
from .data_loader import load_csv

# Import v2 modules for direct access
from . import market_profile_v2
from . import brain_score
from . import genome_optimizer_v2
from . import robustness_testing
from . import param_memory_v2


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
    return _optimize_core(
        config,
        top_n=config.get("topN", 50),
        job_id=job_id,
        progress_cb=progress_cb
    )


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
    return _ai_recommend_core(config, job_id=job_id, progress_cb=progress_cb)


def run_quant_brain(
    config: Dict[str, Any],
    job_id: str = "",
    progress_cb: Optional[Callable[[str, int, int, str, Dict], None]] = None
) -> Dict[str, Any]:
    """
    Run Quant AI Brain v1 - Self-learning optimization engine.

    Features:
    - Long-term genome memory (ParamMemory)
    - Market regime classification
    - Evolutionary genome optimization
    - Coherence validation
    - Robustness filtering

    Args:
        config: Configuration dict (same as AI agent)
        job_id: Job ID for progress tracking
        progress_cb: Optional callback function

    Returns:
        Result dict with:
            - success: bool
            - strategy_hash: str
            - market_regime: str
            - bestGenomes: List of top genomes with explanations
            - meta: { total_tested, from_memory, robustness_passed, ... }
    """
    return _quant_brain_core(config, job_id=job_id, progress_cb=progress_cb)


def run_quant_brain_v2(
    config: Dict[str, Any],
    job_id: str = "",
    progress_cb: Optional[Callable[[str, int, int, str, Dict], None]] = None,
    skip_robustness: bool = False
) -> Dict[str, Any]:
    """
    Run Quant AI Brain v2.0 - Advanced optimization with robustness testing.

    Improvements over v1:
    - Market Profile v2: 32 indicators across 6 clusters
    - BrainScore v2: Multi-criteria scoring (Profitability, Risk, Consistency, Significance)
    - Genetic Algorithm v2: Adaptive mutation, fitness sharing, multi-point crossover
    - Robustness Testing: Walk-Forward, Monte Carlo, Sensitivity, Slippage, Noise
    - Memory System v2: Similarity-based retrieval with 32D vectors

    Args:
        config: Configuration dict with:
            - symbol: str
            - timeframe: str or list
            - strategy: { type: str, params: dict }
            - capital: { initial: float, orderPct: float }
            - risk: { pyramiding: int, commission: float }
            - range: { from: str, to: str } (optional)
        job_id: Job ID for progress tracking
        progress_cb: Optional callback function(stage, progress, message)
        skip_robustness: Skip robustness testing for speed

    Returns:
        Result dict with:
            - success: bool
            - strategy_hash: str
            - market_profile: {...} (32 indicators)
            - regime: str
            - best_genomes: List of top genomes with explanations
            - meta: { total_tested, from_memory, stable_genomes, elapsed_seconds, ... }
            - log: List of optimization log entries
    """
    return _quant_brain_v2_core(config, job_id=job_id, progress_cb=progress_cb, skip_robustness=skip_robustness)


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
    "run_quant_brain",
    "run_quant_brain_v2",
    "generate_chart_data",
    "load_data",
    # V2 modules
    "market_profile_v2",
    "brain_score",
    "genome_optimizer_v2",
    "robustness_testing",
    "param_memory_v2",
]
