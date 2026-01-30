# engine/quant_brain.py
"""
Quant AI Brain - Self-Learning Optimization Engine

Main orchestrator that combines:
1. Strategy hash verification
2. Market regime classification
3. ParamMemory (long-term genome storage)
4. Evolutionary genome optimization
5. Coherence validation
6. Robustness filtering
7. Auto-tuning (OOM prevention & performance optimization)

This is a BRAIN that learns from every market it observes.
"""

import os
import json
import sys
import time
import copy
import logging
from typing import Dict, Any, List, Callable, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Local imports
from .strategy_hash import (
    generate_strategy_hash,
    generate_genome_hash,
    flatten_params_to_genome,
    genome_to_flat_params,
    ENGINE_VERSION
)
from .param_memory import (
    store_genome_result,
    get_top_genomes_by_score,
    get_all_top_genomes,
    query_similar_genomes,
    register_strategy,
    get_memory_stats,
    get_redis,
    get_genome,  # For duplicate detection
)
from .coherence_validator import (
    validate_genome,
    validate_genome_batch,
    repair_genome,
    clamp_to_bounds,
    PARAM_BOUNDS,
    extract_param_bounds_from_config,
    # Combined strategy support
    validate_genome_combined,
    repair_genome_combined,
    clamp_to_bounds_combined,
    PARAM_BOUNDS_COMBINED,
    is_combined_strategy,
    extract_param_bounds_from_config_combined
)
from .regime_classifier import (
    MarketRegime,
    classify_regime,
    get_regime_param_ranges,
    sample_params_for_regime
)
from .genome_optimizer import (
    GenomeOptimizer,
    PhasedOptimizer,
    create_random_genome,
    # Combined strategy support
    GenomeOptimizerCombined,
    create_random_genome_combined,
    genome_to_strategy_params_combined,
    strategy_params_to_genome_combined
)
# Robustness testing - enabled for BRAIN_MODE
from .robustness_filter import quick_robustness_check, test_robustness
from .backtest_engine import run_backtest
from .data_loader import load_csv, get_preloaded, preload_all_data
from .scoring import score_strategy
from .guards import equity_smoothness

# Auto-tuning - OOM prevention & performance optimization
from .performance_monitor import (
    PerformanceMonitor,
    RuntimeMetrics,
    detect_system_resources,
    get_performance_monitor,
    start_monitoring,
    stop_monitoring
)
from .auto_tuner import (
    AutoTuner,
    AutoTuningContext,
    get_auto_tuner,
    check_and_tune
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configuration
QUANT_BRAIN_ENABLED = os.getenv("QUANT_BRAIN_ENABLED", "true").lower() == "true"
TOP_N_DEFAULT = int(os.getenv("QUANT_BRAIN_TOP_N", 50))
# MAX_WORKERS for Apple Silicon optimization
# Default: min(8, cpu_count) to avoid memory pressure on M1/M2/M3
_env_workers = os.getenv("MAX_THREAD_WORKERS", "")
if _env_workers:
    MAX_WORKERS = int(_env_workers)
else:
    MAX_WORKERS = min(8, (os.cpu_count() or 4))
PROGRESS_INTERVAL = 0.5

# ═══════════════════════════════════════════════════════
# MODE DEFINITIONS
# ═══════════════════════════════════════════════════════
# ULTRA_MODE: Maximum speed (Backtest Accelerator)
#   - Rank by FastScore = NetProfit / MaxDD
#   - Minimal coherence check (TP >= SL only)
#   - No robustness, no memory write
#   - Early termination at 50% DD
#   - Continuous loop (no phases)
#
# FAST_MODE: Quick exploration
#   - Rank by PnL (highest profit)
#   - No robustness testing
#   - No memory write
#   - Phased optimization
#
# BRAIN_MODE: Deep learning & deployment
#   - Rank by BrainScore (risk-adjusted)
#   - Robustness testing required
#   - Writes to ParamMemory
#   - Slower but more reliable

MODE_ULTRA = "ultra"
MODE_FAST = "fast"
MODE_BRAIN = "brain"
DEFAULT_MODE = MODE_BRAIN  # Always use BRAIN mode to save to Memory

# Early termination thresholds (ULTRA mode)
EARLY_STOP_DD_PCT = 70.0  # Kill backtest if DD > 70%
EARLY_STOP_EQUITY_PCT = 30.0  # Kill if equity < 30% of initial

# Data preload flag for Apple Silicon optimization
DATA_PRELOAD = os.getenv("DATA_PRELOAD", "true").lower() == "true"

# Scoring constants
DOWNSAMPLE_EQUITY_POINTS = 400
MAX_TRADES_RETURN = 1200

# Auto-tuning configuration
AUTO_TUNING_ENABLED = os.getenv("AUTO_TUNING_ENABLED", "true").lower() == "true"
AUTO_TUNING_INTERVAL = int(os.getenv("AUTO_TUNING_INTERVAL", 10))  # Check every N genomes

# Cancel check interval
CANCEL_CHECK_INTERVAL = 0.5  # Check every 0.5 seconds

# E1) No-improvement threshold (anti-noise)
NO_IMPROVEMENT_EPS = float(os.getenv("NO_IMPROVEMENT_EPS", 0.001))


# ═══════════════════════════════════════════════════════
# E2) RUN METADATA MANAGEMENT
# ═══════════════════════════════════════════════════════

def get_next_run_index(strategy_hash: str, symbol: str, timeframe: str) -> int:
    """
    Get next run_index (Source) for a new Quant Brain run.

    Logic:
    1. Use Redis INCR for atomic global counter (ensures always increasing)
    2. Scan existing genomes to find max source value (for migration/sync)
    3. If counter < max_source_in_genomes, sync counter to max_source + 1

    Key principle: Run index NEVER decreases, ALWAYS increases by 1 each run.
    This is independent of genome data - even if all genomes are deleted,
    the counter continues from where it left off.
    """
    try:
        r = get_redis()

        # Global counter key - this is the SOURCE OF TRUTH for run index
        # Using INCR ensures atomic increment even with concurrent access
        counter_key = f"quant_brain_run_counter:{strategy_hash}:{symbol}:{timeframe}"

        # Check if counter exists
        current_counter = r.get(counter_key)

        if current_counter is None:
            # First time: scan genomes to find max source for migration
            max_source_in_genomes = _scan_max_source_in_genomes(r, strategy_hash, symbol, timeframe)

            # Initialize counter to max_source (next INCR will return max_source + 1)
            r.set(counter_key, max_source_in_genomes)
            logger.info(f"Initialized run counter to {max_source_in_genomes} (from existing genomes)")

        # Atomically increment and get new value
        next_idx = r.incr(counter_key)

        # Safety check: ensure counter is always >= max source in genomes
        # This handles edge case where genomes were imported with higher source
        max_source = _scan_max_source_in_genomes(r, strategy_hash, symbol, timeframe)
        if next_idx <= max_source:
            # Sync counter to be ahead of all existing sources
            next_idx = max_source + 1
            r.set(counter_key, next_idx)
            logger.warning(f"Counter was behind max_source ({max_source}), synced to {next_idx}")

        logger.info(f"Quant Brain Run #{next_idx} starting (counter_key={counter_key})")
        return next_idx

    except Exception as e:
        logger.warning(f"Failed to get run_index: {e}")
        return 1


def _scan_max_source_in_genomes(r, strategy_hash: str, symbol: str, timeframe: str) -> int:
    """
    Scan existing genomes to find the maximum source value.

    This is used for:
    1. Initializing counter on first run (migration from old data)
    2. Safety check to ensure counter stays ahead of imported genomes
    """
    max_source = 0
    pattern = f"genome:{strategy_hash}:{symbol}:{timeframe}:*"
    cursor = 0
    scan_count = 0
    max_scans = 500  # Limit scans for performance

    while scan_count < max_scans:
        cursor, keys = r.scan(cursor, match=pattern, count=50)
        for key in keys:
            try:
                data = r.get(key)
                if data:
                    record = json.loads(data)
                    source = record.get("source", 0)
                    if isinstance(source, int) and source > max_source:
                        max_source = source
            except (json.JSONDecodeError, TypeError):
                pass
        scan_count += 1
        if cursor == 0:
            break

    return max_source


def store_run_metadata(
    run_id: str, run_index: int, strategy_hash: str, symbol: str, timeframe: str,
    run_status: str, best_new_score: float, best_old_score: float, genomes_count: int
) -> bool:
    """E2) Store run metadata in Redis for tracking."""
    try:
        r = get_redis()
        key = f"run_meta:{strategy_hash}:{symbol}:{timeframe}:{run_id}"
        record = {
            "run_id": run_id, "run_index": run_index, "strategy_hash": strategy_hash,
            "symbol": symbol, "timeframe": timeframe, "run_status": run_status,
            "best_new_score": best_new_score, "best_old_score": best_old_score,
            "genomes_count": genomes_count, "timestamp": int(time.time()),
        }
        r.setex(key, 30 * 24 * 3600, json.dumps(record))
        return True
    except Exception as e:
        logger.warning(f"Failed to store run metadata: {e}")
        return False


def check_improvement(
    new_genomes: List[Dict], leaderboard_genomes: List[Dict], score_key: str = "brainScore"
) -> Tuple[str, float, float, float]:
    """E1) Check if new genomes improve over leaderboard."""
    best_new_score = 0.0
    if new_genomes:
        scores = [
            g.get("summary", {}).get(score_key, 0) if "summary" in g
            else g.get("results", {}).get("score", 0) for g in new_genomes
        ]
        best_new_score = max(scores) if scores else 0.0

    best_old_score = 0.0
    if leaderboard_genomes:
        scores = [g.get("results", {}).get("score", 0) for g in leaderboard_genomes]
        best_old_score = max(scores) if scores else 0.0

    delta = best_new_score - best_old_score
    if best_new_score > best_old_score + NO_IMPROVEMENT_EPS:
        return "IMPROVED", best_new_score, best_old_score, delta
    return "NO_IMPROVEMENT", best_new_score, best_old_score, delta


# ═══════════════════════════════════════════════════════
# CANCELLATION TOKEN
# ═══════════════════════════════════════════════════════

class CancellationToken:
    """
    Thread-safe cancellation token for cooperative cancellation.

    Usage:
    - Create token at start of job
    - Pass to all long-running operations
    - Call check_cancelled() frequently to raise if cancelled
    """

    def __init__(self, job_id: str = ""):
        self.job_id = job_id
        self._cancelled = False
        self._last_check_time = 0
        self._check_fn = None

    def set_check_fn(self, fn):
        """Set external function to check cancel flag (e.g., Redis)."""
        self._check_fn = fn

    def cancel(self):
        """Mark as cancelled."""
        self._cancelled = True

    def is_cancelled(self) -> bool:
        """Check if cancelled (with throttled external check)."""
        if self._cancelled:
            return True

        # Throttled external check
        now = time.time()
        if self._check_fn and (now - self._last_check_time) >= CANCEL_CHECK_INTERVAL:
            self._last_check_time = now
            if self._check_fn():
                self._cancelled = True
                return True

        return False

    def check_cancelled(self):
        """Raise InterruptedError if cancelled."""
        if self.is_cancelled():
            logger.info(f"[{self.job_id}] Job cancelled by user")
            raise InterruptedError("Job canceled by user")


# ═══════════════════════════════════════════════════════
# PARAM BOUNDS HELPER
# ═══════════════════════════════════════════════════════

def get_effective_bounds(cfg: Dict[str, Any], memory_records: List[Dict] = None, strategy_type: str = "rf_st_rsi") -> Dict:
    """
    Get effective parameter bounds by merging:
    1. User-specified bounds from Dashboard config
    2. Auto-expanded bounds from memory (successful genomes)
    3. Default bounds from PARAM_BOUNDS

    Priority: User config > Memory expansion > Default

    Args:
        cfg: Dashboard config with parameter ranges
        memory_records: List of genome records from ParamMemory
        strategy_type: Strategy type (rf_st_rsi or rf_st_rsi_combined)

    Returns:
        Merged bounds dict
    """
    # Start with user-specified bounds from config (use appropriate extractor)
    if is_combined_strategy(strategy_type):
        user_bounds = extract_param_bounds_from_config_combined(cfg)
        default_bounds = PARAM_BOUNDS_COMBINED
    else:
        user_bounds = extract_param_bounds_from_config(cfg)
        default_bounds = PARAM_BOUNDS

    # If no memory records, return user bounds
    if not memory_records:
        logger.info(f"Using user-specified bounds for {strategy_type} (no memory records)")
        return user_bounds

    # Auto-expand bounds based on successful genomes from memory
    expanded_bounds = expand_bounds_from_memory(user_bounds, memory_records)

    logger.info(f"Effective bounds computed from config + {len(memory_records)} memory records")
    return expanded_bounds


def expand_bounds_from_memory(
    base_bounds: Dict,
    memory_records: List[Dict],
    expansion_threshold: float = 0.2
) -> Dict:
    """
    Expand parameter bounds based on successful genomes from memory.

    If memory contains high-scoring genomes with parameters outside user range,
    expand bounds to include them (with some margin).

    Args:
        base_bounds: Base bounds (from user config or defaults)
        memory_records: Genome records with scores
        expansion_threshold: Only expand for genomes in top 20% by score

    Returns:
        Expanded bounds dict
    """
    if not memory_records:
        return base_bounds

    # Filter to high-scoring genomes only
    scores = [r.get("results", {}).get("score", 0) for r in memory_records]
    if not scores:
        return base_bounds

    score_threshold = sorted(scores, reverse=True)[int(len(scores) * expansion_threshold)]
    high_performers = [r for r in memory_records
                      if r.get("results", {}).get("score", 0) >= score_threshold]

    if not high_performers:
        return base_bounds

    expanded = copy.deepcopy(base_bounds)

    # Track min/max values from high-performing genomes
    for record in high_performers:
        genome = record.get("genome", {})

        for block in ["entry", "sl", "tp_dual", "tp_rsi"]:
            if block not in genome or block not in expanded:
                continue

            for param, value in genome[block].items():
                if not isinstance(value, (int, float)):
                    continue

                if param not in expanded[block]:
                    continue

                current_min, current_max = expanded[block][param]

                # Expand if genome value is near or outside bounds
                margin = (current_max - current_min) * 0.1  # 10% margin

                if value < current_min:
                    new_min = max(0.5, value - margin)  # Don't go below 0.5
                    expanded[block][param] = (new_min, current_max)
                    logger.info(f"Expanded {block}.{param} min: {current_min} → {new_min}")

                if value > current_max:
                    new_max = value + margin
                    expanded[block][param] = (current_min, new_max)
                    logger.info(f"Expanded {block}.{param} max: {current_max} → {new_max}")

    return expanded


# ═══════════════════════════════════════════════════════
# SCORING FORMULA
# ═══════════════════════════════════════════════════════

def calculate_fast_score(summary: Dict[str, Any]) -> float:
    """
    Calculate FastScore for ULTRA mode.

    FastScore = NetProfit / MaxDD

    Simple risk-adjusted metric for maximum speed ranking.
    """
    net_profit = summary.get("netProfit", 0)
    max_dd = max(summary.get("maxDrawdownPct", 1), 1)  # Avoid division by zero

    if net_profit <= 0:
        return 0.0

    return round(net_profit / max_dd, 4)


def calculate_brain_score(summary: Dict[str, Any]) -> float:
    """
    Calculate Quant Brain score.

    Score = (PF × WR × Smoothness) / (MaxDD × LossStreak × UlcerIndex) × 1000

    Where:
    - PF: Profit Factor (capped at 5)
    - WR: Winrate (0-1)
    - Smoothness: Equity curve smoothness (0-1)
    - MaxDD: Max Drawdown % (min 1)
    - LossStreak: Max consecutive losses (min 1)
    - UlcerIndex: Sqrt of mean squared drawdown (min 0.1)
    """
    pf = min(summary.get("profitFactor", 1), 5)
    wr = summary.get("winrate", 0) / 100
    smooth = summary.get("smoothness", 0.5)

    max_dd = max(summary.get("maxDrawdownPct", 20), 1)
    loss_streak = max(summary.get("maxLossStreak", 3), 1)
    ulcer = max(summary.get("ulcerIndex", 5), 0.1)

    # Prevent division by zero
    denominator = max_dd * loss_streak * ulcer
    if denominator == 0:
        denominator = 1

    score = (pf * wr * smooth) / denominator * 1000
    return round(score, 4)


def calculate_ulcer_index(equity_curve: List[Dict]) -> float:
    """
    Calculate Ulcer Index from equity curve.

    Ulcer Index = sqrt(mean of squared drawdowns)
    Lower is better (smoother equity curve).
    """
    if not equity_curve or len(equity_curve) < 2:
        return 5.0

    equities = [p.get("equity", 0) for p in equity_curve]
    peak = equities[0]
    squared_drawdowns = []

    for eq in equities:
        if eq > peak:
            peak = eq
        dd_pct = (peak - eq) / peak * 100 if peak > 0 else 0
        squared_drawdowns.append(dd_pct ** 2)

    if not squared_drawdowns:
        return 5.0

    import math
    mean_sq_dd = sum(squared_drawdowns) / len(squared_drawdowns)
    return round(math.sqrt(mean_sq_dd), 4)


def calculate_max_loss_streak(trades: List[Dict]) -> int:
    """Calculate maximum consecutive losing trades."""
    if not trades:
        return 0

    max_streak = 0
    current_streak = 0

    for trade in trades:
        pnl = trade.get("pnl", 0)
        if pnl < 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return max_streak


# ═══════════════════════════════════════════════════════
# BACKTEST WRAPPER
# ═══════════════════════════════════════════════════════

def run_genome_backtest(
    genome: Dict,
    symbol: str,
    timeframe: str,
    cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run backtest for a genome and return enriched results.

    Args:
        genome: Genome to test
        symbol: Trading symbol
        timeframe: Timeframe
        cfg: Full config with properties, costs, date range

    Returns:
        Result dict with summary, equity curve, trades
    """
    # Get strategy type from config (support rf_st_rsi and rf_st_rsi_combined)
    strategy_type = cfg.get("strategy", {}).get("type", "rf_st_rsi")

    # Convert genome to flat params based on strategy type
    if is_combined_strategy(strategy_type):
        params = genome_to_strategy_params_combined(genome)
    else:
        params = genome_to_flat_params(genome)

    # Build run config (use `or {}` to handle explicit None values)
    run_cfg = {
        "meta": {
            "symbols": [symbol],
            "timeframe": timeframe
        },
        "strategy": {
            "type": strategy_type,
            "params": params
        },
        "costs": cfg.get("costs") or {},
        "range": cfg.get("range"),
        "properties": cfg.get("properties") or {},
    }

    # Set initial equity
    props = cfg.get("properties") or {}
    if props.get("initialCapital"):
        run_cfg["initial_equity"] = float(props["initialCapital"])

    # Run backtest
    result = run_backtest(run_cfg)

    # Handle None result
    if result is None:
        logger.warning(f"Backtest returned None for {symbol} {timeframe}")
        return {
            "genome": genome,
            "symbol": symbol,
            "timeframe": timeframe,
            "params": params,
            "summary": {"fastScore": 0, "brainScore": 0, "totalTrades": 0},
            "equityCurve": [],
            "trades": [],
        }

    # Enrich summary
    summary = result.get("summary", {})
    equity_curve = result.get("equityCurve", [])
    trades = result.get("trades", [])

    # Calculate additional metrics
    summary["smoothness"] = equity_smoothness(equity_curve)
    summary["ulcerIndex"] = calculate_ulcer_index(equity_curve)
    summary["maxLossStreak"] = calculate_max_loss_streak(trades)
    summary["brainScore"] = calculate_brain_score(summary)
    summary["fastScore"] = calculate_fast_score(summary)

    return {
        "genome": genome,
        "symbol": symbol,
        "timeframe": timeframe,
        "params": params,
        "summary": summary,
        "equityCurve": equity_curve,
        "trades": trades,
    }


# ═══════════════════════════════════════════════════════
# RESULT PROCESSING
# ═══════════════════════════════════════════════════════

def downsample_curve(curve: List, max_points: int = DOWNSAMPLE_EQUITY_POINTS) -> List:
    """Downsample equity curve for efficient serialization."""
    if not curve:
        return []
    n = len(curve)
    if n <= max_points:
        return curve
    step = (n - 1) // (max_points - 1) + 1
    sampled = curve[::step]
    if sampled[-1] != curve[-1]:
        sampled.append(curve[-1])
    return sampled[:max_points]


def prune_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Prune result for frontend consumption."""
    return {
        "symbol": result.get("symbol"),
        "timeframe": result.get("timeframe"),
        "params": result.get("params", {}),
        "genome": result.get("genome", {}),
        "summary": result.get("summary", {}),
        "equityCurve": downsample_curve(result.get("equityCurve", []), 600),
        "trades": result.get("trades", [])[-MAX_TRADES_RETURN:],
        "robustness_score": result.get("robustness_score", 0),
        "is_new_genome": result.get("is_new_genome", False),  # NEW flag
        "genome_hash": result.get("genome_hash", ""),
        "source": result.get("source"),  # Source: Run #X or Run #X.Y
        "is_from_selection": result.get("is_from_selection", False),
    }


def build_comment(best: Dict[str, Any], regime: MarketRegime, mode: str = MODE_FAST) -> str:
    """
    Generate AI comment explaining the selection.

    Args:
        best: Best result dict
        regime: Market regime
        mode: 'ultra', 'fast', or 'brain' mode

    Returns:
        Human-readable comment
    """
    s = best.get("summary", {})
    p = best.get("params", {})

    regime_desc = {
        MarketRegime.TRENDING_UP: "thị trường đang uptrend",
        MarketRegime.TRENDING_DOWN: "thị trường đang downtrend",
        MarketRegime.RANGING: "thị trường đi ngang",
        MarketRegime.VOLATILE: "thị trường biến động mạnh",
        MarketRegime.LOW_VOLATILITY: "thị trường biến động thấp",
    }

    regime_str = regime_desc.get(regime, "thị trường")

    if mode == MODE_ULTRA:
        # Ultra mode: Emphasize FastScore (PnL/DD)
        return (
            f"[ULTRA] FastScore {s.get('fastScore', 0):.2f} = "
            f"PnL ${s.get('netProfit', 0):,.2f} / DD {s.get('maxDrawdownPct', 0):.1f}%. "
            f"RF={p.get('rf_period', 100)}, ST={p.get('st_mult', 2.0)}"
        )
    elif mode == MODE_BRAIN:
        # Brain mode: Emphasize BrainScore and risk metrics
        robustness = best.get("robustness_score", 0)
        return (
            f"[BRAIN] Trong {regime_str}, chọn RF_Period={p.get('rf_period', 100)}, "
            f"ST_Mult={p.get('st_mult', 2.0)} vì đạt BrainScore {s.get('brainScore', 0):.2f}, "
            f"PF {s.get('profitFactor', 0):.2f}, winrate {s.get('winrate', 0):.1f}%, "
            f"max DD {s.get('maxDrawdownPct', 0):.1f}%, robustness {robustness:.2f}. "
            f"Genome đã được lưu vào memory."
        )
    else:
        # Fast mode: Emphasize PnL
        return (
            f"[FAST] Trong {regime_str}, chọn RF_Period={p.get('rf_period', 100)}, "
            f"ST_Mult={p.get('st_mult', 2.0)} vì đạt PnL ${s.get('netProfit', 0):,.2f}, "
            f"PF {s.get('profitFactor', 0):.2f}, winrate {s.get('winrate', 0):.1f}%, "
            f"max DD {s.get('maxDrawdownPct', 0):.1f}%."
        )


# ═══════════════════════════════════════════════════════
# MAIN QUANT BRAIN
# ═══════════════════════════════════════════════════════

def quant_brain_recommend(
    cfg: Dict[str, Any],
    job_id: str = "",
    progress_cb: Optional[Callable[[str, int, int, str, Dict], None]] = None
) -> Dict[str, Any]:
    """
    Quant AI Brain recommendation engine.

    Supports two modes:
    - FAST MODE: Quick exploration, rank by PnL, no robustness, no memory write
    - BRAIN MODE: Deep learning, rank by BrainScore, robustness required, writes to memory

    Flow:
    1. Generate strategy hash
    2. Classify market regime
    3. Query ParamMemory for similar genomes
    4. Seed population with top performers + regime hints
    5. Run phased genome optimization
    6. Apply coherence validation
    7. Backtest valid genomes (parallel)
    8. Apply robustness filter (BRAIN_MODE only)
    9. Score and rank results (PnL for FAST, BrainScore for BRAIN)
    10. Write to ParamMemory (BRAIN_MODE only)
    11. Return best genomes with explanations

    Args:
        cfg: Configuration dict with symbols, timeframes, strategy params
            - mode: 'fast' or 'brain' (default: 'fast')
        job_id: Job ID for progress tracking
        progress_cb: Progress callback function

    Returns:
        Result dict with best genomes and metadata
    """
    start_time = time.time()

    # Get mode from config (default to FAST for speed)
    mode = cfg.get("mode", DEFAULT_MODE).lower()
    if mode not in [MODE_ULTRA, MODE_FAST, MODE_BRAIN]:
        mode = DEFAULT_MODE

    logger.info(f"Quant Brain starting in {mode.upper()} mode")

    # ═══════════════════════════════════════════════════════════════
    # GET RUN_INDEX FOR THIS RUN (for Source column tracking)
    # ═══════════════════════════════════════════════════════════════
    # Extract symbol/timeframe early for run_index
    _symbols = cfg.get("symbols", ["BTCUSDT"])
    _timeframes = cfg.get("timeframes", ["1h"])
    _symbol = _symbols[0] if _symbols else "BTCUSDT"
    _timeframe = _timeframes[0] if _timeframes else "1h"
    strategy_type = cfg.get("strategy", {}).get("type", "rf_st_rsi")
    _strategy_hash_early = generate_strategy_hash(strategy_type, ENGINE_VERSION)

    # Get the run_index for this run (will be stored with all genomes)
    current_run_index = get_next_run_index(_strategy_hash_early, _symbol, _timeframe)
    logger.info(f"Assigned run_index={current_run_index} for this Quant Brain run")

    # ═══════════════════════════════════════════════════════════════
    # CANCELLATION TOKEN SETUP
    # ═══════════════════════════════════════════════════════════════
    cancel_token = CancellationToken(job_id)

    # Try to get check_cancel_flag from progress_store if available
    try:
        from app.services.progress_store import check_cancel_flag
        cancel_token.set_check_fn(lambda: check_cancel_flag(job_id))
        logger.info(f"[{job_id}] Cancellation token initialized with Redis check")
    except ImportError:
        logger.warning(f"[{job_id}] progress_store not available, cancellation via callback only")

    # Progress helper with cancel check
    last_emit_time = [0]

    def emit_progress(progress: int, total: int, extra: Dict = None, force: bool = False):
        # Check cancellation on every progress emit
        cancel_token.check_cancelled()

        now = time.time()
        if not force and now - last_emit_time[0] < PROGRESS_INTERVAL and progress != total:
            return
        last_emit_time[0] = now

        if progress_cb:
            progress_cb(job_id, progress, total, "running", extra or {})

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: INITIALIZATION (0% - 25%)
    # Steps: Auto-tuning, Data preload, Strategy hash, Market regime,
    #        Query memory, Expand bounds, Validate coherence
    # ═══════════════════════════════════════════════════════════════
    emit_progress(0, 100, {"phase": "initializing", "mode": mode, "phase_name": "Phase 1: Khởi tạo"}, force=True)

    # 0. AUTO-TUNING INITIALIZATION (0-3%)
    tuning_context = None
    tuning_summary = {}

    emit_progress(1, 100, {"phase": "detecting_resources", "phase_name": "Phase 1: Khởi tạo"}, force=True)

    if AUTO_TUNING_ENABLED:
        try:
            emit_progress(2, 100, {"phase": "detecting_system", "phase_name": "Phase 1: Khởi tạo"}, force=True)
            resources = detect_system_resources()

            emit_progress(3, 100, {"phase": "configuring_tuner", "phase_name": "Phase 1: Khởi tạo"}, force=True)
            tuning_context = AutoTuningContext(
                enabled=True,
                initial_params={
                    "batch_size": min(20, MAX_WORKERS * 2),
                    "max_workers": MAX_WORKERS,
                    "chunk_size": 100_000,
                    "queue_depth_limit": 100,
                }
            )
            tuning_context.__enter__()
            logger.info(f"Auto-tuning initialized: {resources.effective_cpus} CPUs, {resources.effective_memory_gb:.1f}GB RAM")
        except Exception as e:
            logger.warning(f"Failed to initialize auto-tuning: {e}")
            tuning_context = None

    emit_progress(4, 100, {"phase": "tuning_ready", "phase_name": "Phase 1: Khởi tạo"}, force=True)

    # 0.5. DATA PRELOAD (4-6%)
    emit_progress(5, 100, {"phase": "checking_preload", "phase_name": "Phase 1: Khởi tạo"}, force=True)
    if mode == MODE_ULTRA and DATA_PRELOAD:
        preloaded_count = preload_all_data()
        if preloaded_count > 0:
            logger.info(f"ULTRA mode: Preloaded {preloaded_count} datasets for max speed")
    emit_progress(6, 100, {"phase": "preload_done", "phase_name": "Phase 1: Khởi tạo"}, force=True)

    # 1. GENERATE STRATEGY HASH (7-9%)
    emit_progress(7, 100, {"phase": "generating_hash", "phase_name": "Phase 1: Khởi tạo"}, force=True)
    strategy_type = cfg.get("strategy", {}).get("type", "rf_st_rsi")
    strategy_hash = generate_strategy_hash(strategy_type, ENGINE_VERSION)

    emit_progress(8, 100, {"phase": "registering_strategy", "phase_name": "Phase 1: Khởi tạo"}, force=True)
    register_strategy(strategy_hash, strategy_type, ENGINE_VERSION)

    emit_progress(9, 100, {"phase": "strategy_hash", "hash": strategy_hash, "phase_name": "Phase 1: Khởi tạo"}, force=True)

    # 2. CLASSIFY MARKET REGIME (10-13%)
    emit_progress(10, 100, {"phase": "loading_market_data", "phase_name": "Phase 1: Khởi tạo"}, force=True)
    symbols = cfg.get("symbols", ["BTCUSDT"])
    timeframes = cfg.get("timeframes", ["1h"])
    symbol = symbols[0] if symbols else "BTCUSDT"
    timeframe = timeframes[0] if timeframes else "1h"

    emit_progress(11, 100, {"phase": "classifying_regime", "phase_name": "Phase 1: Khởi tạo"}, force=True)
    try:
        df = get_preloaded(symbol, timeframe)
        if df is None:
            df = load_csv(symbol, timeframe)
        regime_result = classify_regime(df)
        regime = regime_result["regime"]
        market_profile = regime_result["profile"]
    except Exception as e:
        logger.warning(f"Failed to classify regime: {e}")
        regime = MarketRegime.RANGING
        market_profile = {}

    emit_progress(13, 100, {"phase": "regime_classified", "regime": regime.value, "phase_name": "Phase 1: Khởi tạo"}, force=True)

    # 3. QUERY PARAMMEMORY FOR SEEDS (14-17%)
    emit_progress(14, 100, {"phase": "querying_memory", "phase_name": "Phase 1: Khởi tạo"}, force=True)
    seed_genomes = []
    memory_records = []

    # CHECK: User selected specific genomes from BXH?
    selected_genomes_for_seed = cfg.get("selectedGenomes", [])
    use_selected_for_seed = len(selected_genomes_for_seed) > 0

    if use_selected_for_seed:
        # USE ONLY SELECTED GENOMES from UI
        emit_progress(15, 100, {"phase": "loading_selected_genomes", "phase_name": "Phase 1: Khởi tạo"}, force=True)
        for sg in selected_genomes_for_seed:
            if "genome" in sg:
                seed_genomes.append(sg["genome"])
                memory_records.append(sg)
        logger.info(f"Using {len(seed_genomes)} SELECTED genomes from UI (ignoring Redis BXH)")
        emit_progress(16, 100, {"phase": "selected_genomes_loaded", "phase_name": "Phase 1: Khởi tạo"}, force=True)
    else:
        # NO SELECTION: Load ALL genomes from BXH (Redis)
        try:
            emit_progress(15, 100, {"phase": "loading_all_bxh_genomes", "phase_name": "Phase 1: Khởi tạo"}, force=True)

            # Load ALL top genomes from BXH (no limit)
            top_all = get_all_top_genomes(strategy_hash, symbols, timeframes, limit_per_combo=100)
            memory_records.extend(top_all)
            seed_genomes.extend([g["genome"] for g in top_all if "genome" in g])

            emit_progress(16, 100, {"phase": "loading_similar_genomes", "phase_name": "Phase 1: Khởi tạo"}, force=True)
            # Also load similar genomes for diversity
            similar = query_similar_genomes(
                strategy_hash, symbol, timeframe, market_profile, top_n=20
            )
            for s in similar:
                if "genome" in s and s["genome"] not in seed_genomes:
                    memory_records.append(s)
                    seed_genomes.append(s["genome"])

            logger.info(f"Loaded ALL {len(seed_genomes)} genomes from BXH (no selection)")
        except Exception as e:
            logger.warning(f"Failed to load from ParamMemory: {e}")

    emit_progress(17, 100, {"phase": "memory_loaded", "count": len(seed_genomes), "phase_name": "Phase 1: Khởi tạo"}, force=True)

    # Track seed genome hashes for NEW marking later
    seed_genome_hashes = set()
    for sg in seed_genomes:
        seed_genome_hashes.add(generate_genome_hash(sg))
    logger.info(f"Tracking {len(seed_genome_hashes)} seed genome hashes for NEW marking")

    # 3.5. AUTO-EXPAND BOUNDS FROM MEMORY (18-20%)
    emit_progress(18, 100, {"phase": "expanding_bounds", "phase_name": "Phase 1: Khởi tạo"}, force=True)
    # Get effective bounds - auto-expands if memory has good genomes outside user range
    effective_bounds = get_effective_bounds(cfg, memory_records, strategy_type)

    # Log if bounds were expanded - use appropriate default bounds
    use_combined = is_combined_strategy(strategy_type)
    if use_combined:
        default_bounds = PARAM_BOUNDS_COMBINED
        block_list = ["entry", "sl_long", "sl_short", "tp_dual_long", "tp_dual_short", "tp_rsi_long", "tp_rsi_short"]
    else:
        default_bounds = PARAM_BOUNDS
        block_list = ["entry", "sl", "tp_dual", "tp_rsi"]

    bounds_expanded = False
    expansion_details = []

    for block in block_list:
        if block not in effective_bounds or block not in default_bounds:
            continue
        for param in effective_bounds[block]:
            if param not in default_bounds[block]:
                continue
            eff_min, eff_max = effective_bounds[block][param]
            def_min, def_max = default_bounds[block][param]
            if eff_min < def_min or eff_max > def_max:
                bounds_expanded = True
                expansion_details.append(f"{block}.{param}: [{def_min},{def_max}] → [{eff_min:.2f},{eff_max:.2f}]")

    if bounds_expanded:
        logger.info(f"Bounds auto-expanded from memory: {expansion_details}")

    emit_progress(19, 100, {"phase": "sampling_regime_params", "phase_name": "Phase 1: Khởi tạo"}, force=True)

    # Add samples based on strategy type
    if is_combined_strategy(strategy_type):
        # Combined strategy: Use combined genome generator WITH UI bounds
        logger.info("Using combined genome generator for rf_st_rsi_combined with effective_bounds")
        from .genome_optimizer import create_random_genome_combined_with_bounds
        combined_samples = [create_random_genome_combined_with_bounds(effective_bounds) for _ in range(10)]
        seed_genomes.extend(combined_samples)
    else:
        # Standard strategy: Use regime-aware random samples
        regime_samples = sample_params_for_regime(regime, n_samples=10)
        seed_genomes.extend(regime_samples)

    emit_progress(20, 100, {
        "phase": "seeds_loaded",
        "count": len(seed_genomes),
        "bounds_expanded": bounds_expanded,
        "phase_name": "Phase 1: Khởi tạo"
    }, force=True)

    # 4. VALIDATE COHERENCE (21-24%)
    emit_progress(21, 100, {"phase": "validating_coherence", "phase_name": "Phase 1: Khởi tạo"}, force=True)

    # use_combined already determined above when computing effective_bounds

    if mode == MODE_ULTRA:
        # ULTRA: Minimal filter - only check TP_mult >= SL_mult
        valid_seeds = []
        invalid_seeds = []
        for genome in seed_genomes:
            if use_combined:
                # Combined strategy: check RR for both Long and Short
                tp_dual_rr_l = genome.get("tp_dual_long", {}).get("rr_mult", 1.0)
                tp_dual_rr_s = genome.get("tp_dual_short", {}).get("rr_mult", 0.75)
                tp_rsi_rr_l = genome.get("tp_rsi_long", {}).get("rr_mult", 1.0)
                tp_rsi_rr_s = genome.get("tp_rsi_short", {}).get("rr_mult", 0.75)
                if tp_dual_rr_l >= 0.5 and tp_dual_rr_s >= 0.1 and tp_rsi_rr_l >= 0.5 and tp_rsi_rr_s >= 0.1:
                    valid_seeds.append(genome)
                else:
                    invalid_seeds.append(genome)
            else:
                tp_dual_rr = genome.get("tp_dual", {}).get("rr_mult", 1.0)
                tp_rsi_rr = genome.get("tp_rsi", {}).get("rr_mult", 1.0)
                # Simple check: RR should be positive
                if tp_dual_rr >= 0.5 and tp_rsi_rr >= 0.5:
                    valid_seeds.append(genome)
                else:
                    invalid_seeds.append(genome)
        logger.info(f"ULTRA mode: minimal coherence check - {len(valid_seeds)} valid")
    else:
        # FAST/BRAIN: Full coherence validation
        if use_combined:
            # Use combined validator
            valid_seeds = []
            invalid_seeds = []
            for genome in seed_genomes:
                is_valid, violations = validate_genome_combined(genome)
                if is_valid:
                    valid_seeds.append(genome)
                else:
                    invalid_seeds.append({"genome": genome, "violations": violations})
            logger.info(f"Combined strategy coherence: {len(valid_seeds)} valid, {len(invalid_seeds)} invalid")
        else:
            valid_seeds, invalid_seeds = validate_genome_batch(seed_genomes)

    emit_progress(23, 100, {
        "phase": "coherence_validated",
        "valid": len(valid_seeds),
        "invalid": len(invalid_seeds),
        "phase_name": "Phase 1: Khởi tạo"
    }, force=True)

    # End of Phase 1
    emit_progress(25, 100, {"phase": "phase1_complete", "phase_name": "Phase 1: Khởi tạo"}, force=True)

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: GENOME OPTIMIZATION (25% - 50%)
    # Steps: Create fitness function, Run evolutionary optimizer
    # ═══════════════════════════════════════════════════════════════
    emit_progress(26, 100, {"phase": "creating_fitness", "phase_name": "Phase 2: Tối ưu hóa"}, force=True)

    def fitness_fn(genome: Dict) -> float:
        """Fitness function for genome evaluation."""
        try:
            result = run_genome_backtest(genome, symbol, timeframe, cfg)
            summary = result.get("summary", {})

            if mode == MODE_ULTRA:
                # ULTRA: Use FastScore = NetProfit / MaxDD
                score = summary.get("fastScore", 0)
            else:
                # FAST/BRAIN: Use BrainScore
                score = summary.get("brainScore", 0)

            return score if score else 0.0
        except Exception as e:
            import traceback
            logger.warning(f"Fitness backtest failed: {e}\n{traceback.format_exc()}")
            return float("-inf")

    emit_progress(27, 100, {"phase": "preparing_optimizer", "phase_name": "Phase 2: Tối ưu hóa Genome"}, force=True)

    num_combos = len(symbols) * len(timeframes)

    emit_progress(28, 100, {"phase": "calculating_params", "phase_name": "Phase 2: Tối ưu hóa Genome"}, force=True)
    if mode == MODE_ULTRA:
        # ULTRA MODE: Maximum speed for Apple Silicon
        # No phased optimization - continuous loop
        # Optimized for M1/M2/M3 chips with limited RAM
        generations_per_phase = 2  # Minimal generations
        population_size = 50  # Increased from 40 for better exploration
        top_genome_limit = 10  # Return only top 10 genomes
        logger.info(f"ULTRA mode: Apple Silicon optimized - pop={population_size}, top={top_genome_limit}")
    elif num_combos <= 1:
        # Single combo: BALANCED MODE (optimized for long date ranges)
        # 50 genomes × 6 generations = 300 backtests per run
        generations_per_phase = 3
        population_size = 50  # Increased from 30 for better exploration with BXH seeds
        top_genome_limit = 10  # Return top 10 genomes
        logger.info(f"BRAIN mode: Balanced - pop={population_size}, gen={generations_per_phase}")
    elif num_combos <= 3:
        # 2-3 combos: balanced quality
        generations_per_phase = 3
        population_size = 40  # Increased from 25
        top_genome_limit = 10
    else:
        # 4+ combos: speed priority
        generations_per_phase = 2
        population_size = 30  # Increased from 20
        top_genome_limit = 8

    logger.info(
        f"Adaptive params: {num_combos} combos → "
        f"generations={generations_per_phase}, population={population_size}, "
        f"top_genomes={top_genome_limit}"
    )

    emit_progress(29, 100, {"phase": "creating_optimizer", "phase_name": "Phase 2: Tối ưu hóa Genome"}, force=True)

    # Use appropriate optimizer based on strategy type
    if use_combined:
        # Combined strategy: Use GenomeOptimizerCombined with effective bounds from UI
        logger.info("Using GenomeOptimizerCombined for rf_st_rsi_combined strategy")
        logger.info(f"Using effective bounds for combined: {effective_bounds}")
        optimizer = GenomeOptimizerCombined(
            fitness_fn=fitness_fn,
            population_size=population_size,
            generations=generations_per_phase * 2,  # Total generations across all phases (reduced from *4 to *2 for faster execution)
            param_bounds=effective_bounds,  # Use UI bounds instead of fixed PARAM_BOUNDS_COMBINED
            cancel_check_fn=cancel_token.is_cancelled
        )
    else:
        # Standard strategy: Use PhasedOptimizer
        optimizer = PhasedOptimizer(
            fitness_fn=fitness_fn,
            generations_per_phase=generations_per_phase,
            population_size=population_size,
            param_bounds=effective_bounds,  # Pass expanded bounds
            cancel_check_fn=cancel_token.is_cancelled  # Pass cancel check function
        )

    emit_progress(30, 100, {"phase": "starting_evolution", "phase_name": "Phase 2: Tối ưu hóa Genome"}, force=True)

    def opt_progress(gen, total_gen, best_score):
        # Phase 2: Optimization (25% - 50%) -> maps gen/total_gen to 30-49%
        # gen starts at 1, total_gen is max generations
        progress = 30 + int((gen / total_gen) * 19)  # 30 to 49
        emit_progress(progress, 100, {
            "phase": "optimizing",
            "generation": gen,
            "total_generations": total_gen,
            "best_score": best_score,
            "phase_name": "Phase 2: Tối ưu hóa Genome"
        }, force=True)

    best_genome, best_score, top_genomes = optimizer.optimize(
        seed_genomes=valid_seeds,
        regime=regime,
        progress_cb=opt_progress
    )

    # End of Phase 2
    emit_progress(50, 100, {"phase": "phase2_complete", "best_score": best_score, "phase_name": "Phase 2: Tối ưu hóa Genome"}, force=True)

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: BACKTEST TOP GENOMES (50% - 75%)
    # Steps: Parallel backtest of top genomes across symbols/timeframes
    # ═══════════════════════════════════════════════════════════════
    emit_progress(51, 100, {"phase": "preparing_backtest", "phase_name": "Phase 3: Backtest"}, force=True)
    emit_progress(52, 100, {"phase": "backtesting_top", "phase_name": "Phase 3: Backtest"}, force=True)

    # Limit top genomes for faster execution
    limited_genomes = top_genomes[:top_genome_limit]

    # Track which genomes are newly discovered vs from seed
    seed_genome_hashes = set()
    for seed_genome in seed_genomes:
        try:
            seed_hash = generate_genome_hash(seed_genome)
            seed_genome_hashes.add(seed_hash)
        except Exception:
            pass

    results = []
    total_backtest = len(limited_genomes) * len(symbols) * len(timeframes)
    completed = 0

    # Get dynamic worker count from auto-tuner if available
    effective_workers = MAX_WORKERS
    if tuning_context:
        tuning_params = tuning_context.get_params()
        effective_workers = tuning_params.get("max_workers", MAX_WORKERS)
        logger.info(f"Using auto-tuned workers: {effective_workers}")

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = []
        cancelled = False

        for genome in limited_genomes:
            # Check cancel before submitting more work
            if cancel_token.is_cancelled():
                cancelled = True
                logger.info(f"[{job_id}] Backtest cancelled before completion")
                break

            for sym in symbols:
                for tf in timeframes:
                    future = executor.submit(
                        run_genome_backtest, genome, sym, tf, cfg
                    )
                    futures.append(future)

        for future in as_completed(futures):
            # Check cancel on each completion
            if cancel_token.is_cancelled():
                cancelled = True
                # Cancel remaining futures
                for f in futures:
                    f.cancel()
                logger.info(f"[{job_id}] Backtest cancelled, stopping iteration")
                break

            completed += 1
            # Phase 3: Backtest (50% - 75%) -> maps completed/total to 53-74%
            progress = 53 + int((completed / total_backtest) * 21)  # 53 to 74

            # Auto-tuning check every N completions
            if tuning_context and completed % AUTO_TUNING_INTERVAL == 0:
                tuning_context.generation_end()  # Mark completion for throughput tracking
                decision = tuning_context.check()
                if decision:
                    logger.info(f"[AUTO-TUNE] Applied: {decision.action.value} (reason: {decision.reason.value})")
                tuning_context.generation_start()  # Start next batch

            emit_progress(progress, 100, {
                "phase": "backtesting",
                "completed": completed,
                "total": total_backtest,
                "phase_name": "Phase 3: Backtest"
            }, force=True)

            try:
                result = future.result()
                if result.get("summary", {}).get("totalTrades", 0) > 0:
                    # Mark if this genome is newly discovered (not from seed)
                    genome_hash = generate_genome_hash(result.get("genome", {}))
                    result["is_newly_discovered"] = genome_hash not in seed_genome_hashes
                    results.append(result)
            except Exception as e:
                logger.debug(f"Backtest failed: {e}")

    # End of Phase 3
    emit_progress(75, 100, {"phase": "phase3_complete", "results": len(results), "phase_name": "Phase 3: Backtest"}, force=True)

    # ═══════════════════════════════════════════════════════════════
    # PHASE 4: ROBUSTNESS + STORE MEMORY (75% - 100%)
    # Steps: Robustness filter, Rank results, Store to memory, Build response
    # ═══════════════════════════════════════════════════════════════
    emit_progress(76, 100, {"phase": "preparing_robustness", "phase_name": "Phase 4: Lưu kết quả"}, force=True)
    robust_results = []
    fragile_results = []

    if mode == MODE_BRAIN:
        # BRAIN MODE: Run full robustness testing
        emit_progress(77, 100, {"phase": "robustness_testing", "phase_name": "Phase 4: Lưu kết quả"}, force=True)

        for idx, result in enumerate(results):
            # Check cancel before each robustness test
            if cancel_token.is_cancelled():
                logger.info(f"[{job_id}] Robustness testing cancelled at genome {idx}/{len(results)}")
                break

            genome = result.get("genome", {})
            base_score = result.get("summary", {}).get("brainScore", 0)

            try:
                passed, stability_score, details = test_robustness(
                    genome,
                    fitness_fn=lambda g: run_genome_backtest(g, symbol, timeframe, cfg).get("summary", {}).get("brainScore", 0),
                    base_score=base_score
                )

                result["robustness_passed"] = passed
                result["robustness_score"] = stability_score
                result["robustness_details"] = details

                if passed:
                    robust_results.append(result)
                else:
                    fragile_results.append(result)
            except Exception as e:
                logger.debug(f"Robustness test failed for genome {idx}: {e}")
                result["robustness_passed"] = False
                result["robustness_score"] = 0
                fragile_results.append(result)

        emit_progress(80, 100, {
            "phase": "robustness_complete",
            "robust": len(robust_results),
            "fragile": len(fragile_results),
            "phase_name": "Phase 4: Lưu kết quả"
        }, force=True)

        logger.info(f"Robustness filter: {len(robust_results)} robust, {len(fragile_results)} fragile")
    else:
        # ULTRA/FAST MODE: Skip robustness, mark all as passed
        emit_progress(78, 100, {"phase": "skipping_robustness", "phase_name": "Phase 4: Lưu kết quả"}, force=True)

        for result in results:
            result["robustness_passed"] = True
            result["robustness_score"] = 1.0

        robust_results = results

        emit_progress(80, 100, {
            "phase": "robustness_skipped",
            "robust": len(robust_results),
            "fragile": 0,
            "phase_name": "Phase 4: Lưu kết quả"
        }, force=True)

    # ═══════════════════════════════════════════════════════
    # 9. SORT AND RANK (mode-dependent)
    # ═══════════════════════════════════════════════════════
    emit_progress(81, 100, {"phase": "ranking_results", "phase_name": "Phase 4: Lưu kết quả"}, force=True)
    # Prefer robust results, fallback to all results
    final_results = robust_results if robust_results else results

    if mode == MODE_ULTRA:
        # ULTRA MODE: Rank by FastScore = NetProfit / MaxDD
        final_results.sort(
            key=lambda x: x.get("summary", {}).get("fastScore", 0),
            reverse=True
        )
        logger.info("Ranking by FastScore (ULTRA mode)")
    elif mode == MODE_BRAIN:
        # BRAIN MODE: Rank by BrainScore (risk-adjusted performance)
        final_results.sort(
            key=lambda x: x.get("summary", {}).get("brainScore", 0),
            reverse=True
        )
        logger.info("Ranking by BrainScore (BRAIN mode)")
    else:
        # FAST MODE: Rank by PnL (highest profit)
        final_results.sort(
            key=lambda x: x.get("summary", {}).get("netProfit", 0),
            reverse=True
        )
        logger.info("Ranking by PnL (FAST mode)")

    # ═══════════════════════════════════════════════════════
    # 9.1 DEDUPLICATE RESULTS (remove duplicate genomes)
    # ═══════════════════════════════════════════════════════
    # Deduplicate by BOTH genome_hash AND metrics_hash
    # This ensures no two results have identical params OR identical performance
    seen_hashes_for_dedup = set()
    seen_metrics_hashes = set()
    deduplicated_results = []
    duplicate_by_genome = 0
    duplicate_by_metrics = 0

    # seed_genome_hashes was created earlier after loading seeds

    for result in final_results:
        genome = result.get("genome", {})
        summary = result.get("summary", {})

        # Hash based on genome params
        genome_hash = generate_genome_hash(genome)

        # Hash based on key metrics (PF, WR, DD, Trades, PNL)
        metrics_key = (
            round(summary.get("profitFactor", 0), 2),
            round(summary.get("winrate", 0), 1),
            round(summary.get("maxDrawdownPct", 0), 2),
            summary.get("totalTrades", 0),
            round(summary.get("netProfit", 0), 0)
        )
        metrics_hash = hash(metrics_key)

        # Check both genome and metrics uniqueness
        is_genome_dup = genome_hash in seen_hashes_for_dedup
        is_metrics_dup = metrics_hash in seen_metrics_hashes

        if not is_genome_dup and not is_metrics_dup:
            seen_hashes_for_dedup.add(genome_hash)
            seen_metrics_hashes.add(metrics_hash)
            result["genome_hash"] = genome_hash

            # Mark as NEW if not from seed genomes (BXH)
            result["is_new_genome"] = genome_hash not in seed_genome_hashes

            deduplicated_results.append(result)
        elif is_genome_dup:
            duplicate_by_genome += 1
        else:
            duplicate_by_metrics += 1

    total_duplicates = duplicate_by_genome + duplicate_by_metrics
    if total_duplicates > 0:
        logger.info(
            f"Deduplication: removed {total_duplicates} duplicates "
            f"(by genome: {duplicate_by_genome}, by metrics: {duplicate_by_metrics}), "
            f"kept {len(deduplicated_results)} unique"
        )

    # Replace final_results with deduplicated version
    final_results = deduplicated_results

    top_n = cfg.get("topN", TOP_N_DEFAULT)
    best = final_results[0] if final_results else None
    top = final_results[:top_n]

    # ═══════════════════════════════════════════════════════
    # 9.5 ENSURE AT LEAST 1 NEWLY-DISCOVERED GENOME IN TOP
    # ═══════════════════════════════════════════════════════
    # Check if any newly-discovered genomes are already in top
    newly_discovered_in_top = any(r.get("is_newly_discovered", False) for r in top)

    if not newly_discovered_in_top and len(final_results) > len(top):
        # Find the best newly-discovered genome not yet in top
        best_new = None
        best_new_idx = -1
        for idx, result in enumerate(final_results[len(top):], start=len(top)):
            if result.get("is_newly_discovered", False):
                best_new = result
                best_new_idx = idx
                break

        if best_new is not None:
            # Replace lowest-scoring item in top with this newly-discovered genome
            if top:
                top[-1] = best_new
                logger.info(
                    f"Ensured at least 1 newly-discovered genome in top: "
                    f"replaced position {len(top)} with newly-discovered "
                    f"(score={best_new.get('summary', {}).get('brainScore', 0):.2f})"
                )

    # ═══════════════════════════════════════════════════════
    # 10. PREPARE FOR MEMORY WRITE (will be done AFTER merge)
    # ═══════════════════════════════════════════════════════
    stored_count = 0
    TOP_N_TO_STORE = 5  # Chỉ lưu Top 5 genomes mỗi lần chạy
    MIN_RANGE_MONTHS = 11  # Tối thiểu 11 tháng để được lưu vào Memory

    # Check backtest range duration
    def calculate_range_months(range_from: str, range_to: str) -> float:
        """Calculate number of months between two dates."""
        from datetime import datetime
        try:
            if not range_from or not range_to:
                return 0
            date_from = datetime.strptime(range_from, "%Y-%m-%d")
            date_to = datetime.strptime(range_to, "%Y-%m-%d")
            delta = date_to - date_from
            return delta.days / 30.44  # Average days per month
        except Exception:
            return 0

    range_from = cfg.get("range", {}).get("from", "")
    range_to = cfg.get("range", {}).get("to", "")
    range_months = calculate_range_months(range_from, range_to)

    emit_progress(82, 100, {"phase": "checking_memory_save", "phase_name": "Phase 4: Lưu kết quả"}, force=True)
    # NOTE: Actual Memory write is done AFTER merge (see section 11.2 below)

    # ═══════════════════════════════════════════════════════
    # 11. BUILD RESPONSE
    # ═══════════════════════════════════════════════════════
    emit_progress(91, 100, {"phase": "building_response", "phase_name": "Phase 4: Lưu kết quả"}, force=True)
    elapsed = time.time() - start_time

    emit_progress(93, 100, {"phase": "cleanup_tuning", "phase_name": "Phase 4: Lưu kết quả"}, force=True)
    # Cleanup auto-tuning and get summary
    if tuning_context:
        try:
            tuning_summary = tuning_context.tuner.get_summary() if tuning_context.tuner else {}
            tuning_context.__exit__(None, None, None)
            logger.info(f"Auto-tuning session complete: {tuning_summary.get('total_adjustments', 0)} adjustments made")
        except Exception as e:
            logger.warning(f"Failed to cleanup auto-tuning: {e}")

    emit_progress(95, 100, {"phase": "preparing_stats", "phase_name": "Phase 4: Lưu kết quả"}, force=True)
    emit_progress(97, 100, {"phase": "finalizing", "phase_name": "Phase 4: Lưu kết quả"}, force=True)
    emit_progress(99, 100, {"phase": "almost_done", "phase_name": "Phase 4: Lưu kết quả"}, force=True)
    emit_progress(100, 100, {"phase": "complete", "mode": mode, "phase_name": "Phase 4: Hoàn thành"}, force=True)

    # Memory stats
    try:
        memory_stats = get_memory_stats(strategy_hash)
    except:
        memory_stats = {}

    # ═══════════════════════════════════════════════════════
    # 11.1 MERGE NEW RESULTS WITH SELECTED GENOMES (KEEP BEST)
    # ═══════════════════════════════════════════════════════
    # Priority: Use selectedGenomes from UI if provided, otherwise use leaderboard from Redis

    # Check if user provided selected genomes from MemoryPage
    selected_genomes_from_ui = cfg.get("selectedGenomes", [])
    use_selected_only = len(selected_genomes_from_ui) > 0

    # Get genomes to merge with (either from UI selection or Redis leaderboard)
    merge_genomes = []
    if use_selected_only:
        # Use only the genomes user selected in MemoryPage
        merge_genomes = selected_genomes_from_ui
        logger.info(f"Using {len(merge_genomes)} selected genomes from UI for merge (ignoring Redis leaderboard)")
    else:
        # Fallback: Load from Redis leaderboard
        try:
            merge_genomes = get_top_genomes_by_score(
                strategy_hash, symbol, timeframe, limit=20
            )
            logger.info(f"Loaded {len(merge_genomes)} genomes from Redis leaderboard for merge")
        except Exception as e:
            logger.warning(f"Failed to load leaderboard for merge: {e}")

    # Check if new results improve over merge_genomes
    run_status, best_new_score, best_old_score, score_delta = check_improvement(
        top, merge_genomes, score_key="brainScore" if mode == MODE_BRAIN else "netProfit"
    )

    # ═══════════════════════════════════════════════════════
    # SMART MERGE: Only keep results BETTER than selection
    # If not better → return selection with updated Source
    # ═══════════════════════════════════════════════════════
    merged_top = []
    score_key = "brainScore" if mode == MODE_BRAIN else "netProfit"

    if use_selected_only and selected_genomes_from_ui:
        # User selected specific genomes - compare each new result against selection
        logger.info(f"Comparing {len(top)} new results against {len(selected_genomes_from_ui)} selected genomes")

        # Get best score from selected genomes
        selected_scores = []
        for sg in selected_genomes_from_ui:
            sg_results = sg.get("results", {})
            if score_key == "brainScore":
                selected_scores.append(sg_results.get("score", 0))
            else:
                selected_scores.append(sg_results.get("net_profit", 0))
        best_selected_score = max(selected_scores) if selected_scores else 0

        # Filter new results - only keep if BETTER than best selected
        better_results = []
        for result in top:
            result_score = result.get("summary", {}).get(score_key, 0)
            if result_score > best_selected_score:
                better_results.append(result)

        if better_results:
            # Have better results - use them
            merged_top = better_results
            run_status = "IMPROVED"
            logger.info(f"Found {len(better_results)} results BETTER than selection (best_new={better_results[0].get('summary', {}).get(score_key, 0):.4f} > best_selected={best_selected_score:.4f})")
        else:
            # No better results - return selected genomes with updated Source
            run_status = "NO_IMPROVEMENT"
            logger.info(f"No results better than selection (best_new={best_new_score:.4f} <= best_selected={best_selected_score:.4f})")
            logger.info(f"Returning {len(selected_genomes_from_ui)} selected genomes with updated Source: Run #{current_run_index}")

            # Convert selected genomes to result format with updated source
            for sg in selected_genomes_from_ui:
                sg_results = sg.get("results", {})
                old_source = sg.get("source", 0)

                # Create new source format: "Run #OldSource.CurrentRun"
                new_source = f"{old_source}.{current_run_index}"

                mg_result = {
                    "genome": sg.get("genome", {}),
                    "genome_hash": sg.get("genome_hash", ""),
                    "symbol": sg.get("symbol", symbol),
                    "timeframe": sg.get("timeframe", timeframe),
                    "summary": {
                        "profitFactor": sg_results.get("pf", 0),
                        "winrate": sg_results.get("winrate", 0),
                        "maxDrawdownPct": sg_results.get("max_dd", 0),
                        "netProfit": sg_results.get("net_profit", 0),
                        "netProfitPct": sg_results.get("net_profit_pct", 0),
                        "totalTrades": sg_results.get("total_trades", 0),
                        "brainScore": sg_results.get("score", 0),
                    },
                    "equityCurve": sg.get("equity_curve", []),
                    "robustness_score": sg_results.get("robustness_score", 0),
                    "is_from_leaderboard": False,
                    "is_from_selection": True,
                    "is_newly_discovered": False,
                    "is_new_genome": False,
                    "source": new_source,  # Updated source: Run #10.14
                    "original_source": old_source,  # Keep original for reference
                }
                merged_top.append(mg_result)

    else:
        # No selection - use standard merge logic
        merged_top = list(top)

        if merge_genomes and run_status == "NO_IMPROVEMENT":
            # Add merge_genomes that aren't already in results
            for mg in merge_genomes:
                mg_genome_hash = mg.get("genome_hash", "")
                already_exists = any(
                    r.get("genome_hash") == mg_genome_hash or
                    r.get("genome", {}) == mg.get("genome", {})
                    for r in merged_top
                )

                if not already_exists:
                    results_data = mg.get("results", {})
                    mg_result = {
                        "genome": mg.get("genome", {}),
                        "genome_hash": mg_genome_hash,
                        "symbol": mg.get("symbol", symbol),
                        "timeframe": mg.get("timeframe", timeframe),
                        "summary": {
                            "profitFactor": results_data.get("pf", 0),
                            "winrate": results_data.get("winrate", 0),
                            "maxDrawdownPct": results_data.get("max_dd", 0),
                            "netProfit": results_data.get("net_profit", 0),
                            "netProfitPct": results_data.get("net_profit_pct", 0),
                            "totalTrades": results_data.get("total_trades", 0),
                            "brainScore": results_data.get("score", 0),
                        },
                        "equityCurve": mg.get("equity_curve", []),
                        "robustness_score": results_data.get("robustness_score", 0),
                        "is_from_leaderboard": True,
                        "is_from_selection": False,
                        "is_newly_discovered": False,
                    }
                    merged_top.append(mg_result)

    # Re-sort merged results by score
    if mode == MODE_BRAIN:
        merged_top.sort(
            key=lambda x: x.get("summary", {}).get("brainScore", 0),
            reverse=True
        )
    else:
        merged_top.sort(
            key=lambda x: x.get("summary", {}).get("netProfit", 0),
            reverse=True
        )

    # Keep top N
    merged_top = merged_top[:max(len(top), 10)]
    logger.info(f"Final merged results: {len(merged_top)}, run_status={run_status}, use_selected={use_selected_only}")

    # Use merged_top for final results
    final_top = merged_top
    final_best = final_top[0] if final_top else best

    # ═══════════════════════════════════════════════════════
    # 11.2 WRITE TOP 5 TO PARAMMEMORY (BRAIN_MODE only)
    # ═══════════════════════════════════════════════════════
    # NOTE: This is done AFTER merge so we store the ACTUAL top results
    # that will be returned to the UI (not the pre-merge results)
    if mode == MODE_BRAIN:
        # Check if range is >= 11 months before saving to Memory
        if range_months < MIN_RANGE_MONTHS:
            emit_progress(85, 100, {
                "phase": "skipping_memory_short_range",
                "range_months": round(range_months, 1),
                "min_required": MIN_RANGE_MONTHS,
                "phase_name": "Phase 4: Lưu kết quả"
            }, force=True)
            logger.info(
                f"BRAIN mode: Skipping Memory write - Range {range_months:.1f} months < {MIN_RANGE_MONTHS} months required"
            )
        else:
            # ALWAYS save Top 5 by PF to Memory regardless of run_status
            # Even if NO_IMPROVEMENT by brainScore, there may be good genomes by PF
            if run_status == "NO_IMPROVEMENT":
                logger.info(
                    f"BRAIN mode: run_status=NO_IMPROVEMENT but still saving Top 5 by PF to Memory"
                )
            # BRAIN MODE: Write TOP 5 genomes from ALL BACKTEST results
            # IMPORTANT: Use `results` (all backtest results) NOT `top` (which is filtered by robustness)
            # This ensures we store TOP 5 by PF from ALL genomes tested, regardless of robustness filter
            emit_progress(83, 100, {"phase": "storing_memory", "phase_name": "Phase 4: Lưu kết quả"}, force=True)

            # DEBUG: Log results before sorting
            logger.info(f"[MEMORY DEBUG] results has {len(results)} total backtest results (ALL genomes)")
            logger.info(f"[MEMORY DEBUG] top has {len(top)} results (after robustness filter)")
            logger.info(f"[MEMORY DEBUG] final_top has {len(final_top)} results (after merge)")

            # Use `results` (ALL backtest results) sorted by PF for storage
            # This ensures we store TOP 5 by PF from actual backtest results, NOT filtered by robustness
            sorted_by_pf = sorted(
                [r for r in results if r.get("summary", {}).get("profitFactor", 0) > 0],
                key=lambda x: x.get("summary", {}).get("profitFactor", 0),
                reverse=True
            )

            # Log top 10 by PF for debugging
            for i, r in enumerate(sorted_by_pf[:10]):
                pf = r.get("summary", {}).get("profitFactor", 0)
                gh = generate_genome_hash(r.get("genome", {}))[:8]
                logger.info(f"[MEMORY DEBUG] sorted_by_pf[{i}]: PF={pf:.2f}, genome_hash={gh}")

            logger.info(f"[MEMORY DEBUG] sorted_by_pf has {len(sorted_by_pf)} results with PF > 0 (from ALL results)")

            # Take Top 5 unique genomes by Profit Factor (from ALL backtest results)
            seen_hashes = set()
            genomes_to_store = []

            for result in sorted_by_pf:
                genome_hash = generate_genome_hash(result.get("genome", {}))
                if genome_hash not in seen_hashes:
                    seen_hashes.add(genome_hash)
                    genomes_to_store.append(result)
                    pf = result.get("summary", {}).get("profitFactor", 0)
                    logger.info(f"[MEMORY DEBUG] Added to store: genome_hash={genome_hash[:8]}, PF={pf:.2f}")
                    if len(genomes_to_store) >= TOP_N_TO_STORE:
                        break
                else:
                    logger.info(f"[MEMORY DEBUG] Skipped duplicate genome_hash={genome_hash[:8]}")

            # Log top PF values for debugging
            top_pf_values = [r.get("summary", {}).get("profitFactor", 0) for r in genomes_to_store]
            logger.info(
                f"BRAIN mode: Storing TOP {len(genomes_to_store)} genomes to Memory "
                f"(ranked by Profit Factor: {top_pf_values}, range={range_months:.1f} months)"
            )

            # Track which genomes are duplicates (already in BXH Memory)
            duplicate_genomes = []

            for result in genomes_to_store:
                try:
                    genome = result.get("genome", {})
                    genome_hash = generate_genome_hash(genome)
                    result_symbol = result.get("symbol", symbol)
                    result_timeframe = result.get("timeframe", timeframe)

                    summary = result.get("summary", {})

                    # ═══════════════════════════════════════════════════════
                    # DUPLICATE DETECTION: Check if genome already exists in BXH Memory
                    # ═══════════════════════════════════════════════════════
                    existing_genome = get_genome(strategy_hash, result_symbol, result_timeframe, genome_hash)

                    if existing_genome:
                        # DUPLICATE: Genome already exists in BXH Memory
                        old_source = existing_genome.get("source", 0)

                        # Create new source format: "OldSource.CurrentRun" (e.g., "21.22")
                        # Handle both integer and string sources
                        if isinstance(old_source, str) and "." in old_source:
                            # Already has compound source (e.g., "20.21"), append new run
                            new_source = f"{old_source}.{current_run_index}"
                        else:
                            # First duplicate: create compound source
                            new_source = f"{old_source}.{current_run_index}"

                        logger.info(
                            f"[MEMORY] Duplicate found: genome_hash={genome_hash[:8]}, "
                            f"old_source={old_source} -> new_source={new_source}"
                        )

                        # Mark as duplicate for NEW badge
                        result["is_duplicate_in_memory"] = True
                        result["source"] = new_source
                        result["original_source"] = old_source
                        duplicate_genomes.append({
                            "genome_hash": genome_hash,
                            "old_source": old_source,
                            "new_source": new_source,
                        })

                        # Update existing record with new source
                        existing_genome["source"] = new_source
                        existing_genome["timestamp"] = int(time.time())
                        existing_genome["test_count"] = existing_genome.get("test_count", 1) + 1

                        if store_genome_result(existing_genome):
                            stored_count += 1
                            logger.info(f"Updated duplicate genome {genome_hash[:8]} with Source={new_source}")
                    else:
                        # NEW GENOME: Not in BXH Memory, store as new
                        # Downsample equity curve for storage (100 points max)
                        equity_curve = result.get("equityCurve", [])
                        equity_curve_downsampled = downsample_curve(equity_curve, 100)

                        record = {
                            "strategy_hash": strategy_hash,
                            "symbol": result_symbol,
                            "timeframe": result_timeframe,
                            "genome_hash": genome_hash,
                            "market_profile": market_profile,
                            "genome": genome,
                            "results": {
                                "pf": summary.get("profitFactor", 0),
                                "winrate": summary.get("winrate", 0),
                                "max_dd": summary.get("maxDrawdownPct", 0),
                                "net_profit": summary.get("netProfit", 0),
                                "net_profit_pct": summary.get("netProfitPct", 0),
                                "total_trades": summary.get("totalTrades", 0),
                                "score": summary.get("brainScore", 0),
                                "ulcer_index": summary.get("ulcerIndex", 0),
                                "loss_streak": summary.get("maxLossStreak", 0),
                                "robustness_score": result.get("robustness_score", 0),
                            },
                            # Store downsampled equity curve for Memory page visualization
                            "equity_curve": equity_curve_downsampled,
                            # Store backtest period (from cfg.range.from/to)
                            "backtest_start": cfg.get("range", {}).get("from") or summary.get("startDate"),
                            "backtest_end": cfg.get("range", {}).get("to") or summary.get("endDate"),
                            "timestamp": int(time.time()),
                            "test_count": 1,
                            # Store run_index (Source column) for tracking which run discovered this genome
                            "source": current_run_index,
                        }

                        if store_genome_result(record):
                            stored_count += 1
                            logger.info(f"Stored NEW genome {genome_hash[:8]} with PF={summary.get('profitFactor', 0):.2f}, Source=Run #{current_run_index}")
                except Exception as e:
                    logger.debug(f"Failed to store genome: {e}")

            # Log duplicate summary
            if duplicate_genomes:
                logger.info(
                    f"[MEMORY] Found {len(duplicate_genomes)} duplicate genomes in BXH Memory, "
                    f"updated their Source to compound format"
                )

            emit_progress(90, 100, {"phase": "memory_stored", "count": stored_count, "phase_name": "Phase 4: Lưu kết quả"}, force=True)
            logger.info(f"Stored {stored_count}/{len(genomes_to_store)} genomes to ParamMemory")
    else:
        # ULTRA/FAST MODE: Skip memory write
        emit_progress(85, 100, {"phase": "skipping_memory_write", "phase_name": "Phase 4: Lưu kết quả"}, force=True)
        logger.info(f"{mode.upper()} mode: Skipping ParamMemory write")

    # Build comment (with mode)
    comment = build_comment(final_best, regime, mode) if final_best else "No valid genome found"

    # Prune for response
    pruned_top = [prune_result(r) for r in final_top]
    pruned_best = prune_result(final_best) if final_best else None
    pruned_all = [prune_result(r) for r in results[:100]]

    # Mode-specific ranking info
    if mode == MODE_ULTRA:
        ranking_by = "fastScore"
    elif mode == MODE_BRAIN:
        ranking_by = "brainScore"
    else:
        ranking_by = "netProfit"

    return {
        "success": bool(final_best and (robust_results or merge_genomes)),
        "fallback": bool(final_best and not robust_results),
        "message": None if final_best else "No genome passed all filters",
        "mode": mode,
        "ranking_by": ranking_by,
        "strategy_hash": strategy_hash,
        "strategyType": strategy_type,  # For frontend to know which Memory to add to
        "market_regime": regime.value,
        "best": pruned_best,
        "run_status": run_status,  # "IMPROVED" or "NO_IMPROVEMENT"
        "used_selected_genomes": use_selected_only,  # True if using UI selection instead of Redis leaderboard
        "score_comparison": {
            "best_new_score": round(best_new_score, 4),
            "best_old_score": round(best_old_score, 4),
            "delta": round(score_delta, 4),
        },
        "bestGenomes": [
            {
                **prune_result(r),
                "comment": build_comment(r, regime, mode),
                "is_from_leaderboard": r.get("is_from_leaderboard", False),
                "is_from_selection": r.get("is_from_selection", False),
                "is_new_genome": r.get("is_new_genome", False),  # NEW flag for UI
            }
            for r in final_top[:5]
        ],
        "alternatives": [prune_result(r) for r in top[1:4]],
        "comment": comment,
        "top": pruned_top,
        "all": pruned_all,
        "total": len(results),
        "meta": {
            "mode": mode,
            "ranking_by": ranking_by,
            "total_tested": len(results),
            "from_memory": len(seed_genomes),
            "newly_discovered": len(results) - len(seed_genomes),
            "coherence_rejected": len(invalid_seeds),
            "robustness_passed": len(robust_results),
            "robustness_rejected": len(fragile_results),
            "stored_to_memory": stored_count,
            "elapsed_seconds": round(elapsed, 2),
            # ═══════════════════════════════════════════════════════
            # KPI: Newly-discovered genomes in Top BXH
            # ═══════════════════════════════════════════════════════
            "newly_discovered_in_top": sum(
                1 for r in final_top if r.get("is_newly_discovered", False)
            ),
            "newly_discovered_in_top_percent": round(
                (sum(1 for r in final_top if r.get("is_newly_discovered", False)) / len(final_top) * 100)
                if final_top else 0,
                1
            ),
            # ═══════════════════════════════════════════════════════
            # KPI: Merge info (selected genomes or leaderboard)
            # ═══════════════════════════════════════════════════════
            "used_selected_genomes": use_selected_only,
            "selected_genomes_count": len(selected_genomes_from_ui),
            "merge_genomes_count": len(merge_genomes),
            "from_leaderboard_in_top": sum(
                1 for r in final_top if r.get("is_from_leaderboard", False)
            ),
            "from_selection_in_top": sum(
                1 for r in final_top if r.get("is_from_selection", False)
            ),
            "memory_stats": memory_stats,
            "range": cfg.get("range", {}),  # Backtest range for frontend
            "bounds_expanded": bounds_expanded,
            "expansion_details": expansion_details if bounds_expanded else [],
            "adaptive_params": {
                "num_combos": num_combos,
                "generations_per_phase": generations_per_phase,
                "population_size": population_size,
                "top_genome_limit": top_genome_limit,
            },
            # Auto-tuning summary
            "auto_tuning": {
                "enabled": AUTO_TUNING_ENABLED,
                "adjustments_made": tuning_summary.get("total_adjustments", 0) if tuning_summary else 0,
                "final_params": tuning_summary.get("current_params", {}) if tuning_summary else {},
                "recent_decisions": tuning_summary.get("recent_decisions", []) if tuning_summary else [],
            } if AUTO_TUNING_ENABLED else None,
        }
    }


# ═══════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps({"error": "Config JSON required on stdin"}))
        sys.exit(1)

    cfg = json.loads(raw)
    output = quant_brain_recommend(cfg)
    print(json.dumps(output))
