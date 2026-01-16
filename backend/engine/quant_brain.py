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

This is a BRAIN that learns from every market it observes.
"""

import os
import json
import sys
import time
import copy
import logging
from typing import Dict, Any, List, Callable, Optional
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
    get_memory_stats
)
from .coherence_validator import (
    validate_genome,
    validate_genome_batch,
    repair_genome,
    clamp_to_bounds,
    PARAM_BOUNDS,
    extract_param_bounds_from_config
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
    create_random_genome
)
# Robustness testing - enabled for BRAIN_MODE
from .robustness_filter import quick_robustness_check, test_robustness
from .backtest_engine import run_backtest
from .data_loader import load_csv, get_preloaded, preload_all_data
from .scoring import score_strategy
from .guards import equity_smoothness

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
DEFAULT_MODE = MODE_FAST

# Early termination thresholds (ULTRA mode)
EARLY_STOP_DD_PCT = 70.0  # Kill backtest if DD > 70%
EARLY_STOP_EQUITY_PCT = 30.0  # Kill if equity < 30% of initial

# Data preload flag for Apple Silicon optimization
DATA_PRELOAD = os.getenv("DATA_PRELOAD", "true").lower() == "true"

# Scoring constants
DOWNSAMPLE_EQUITY_POINTS = 400
MAX_TRADES_RETURN = 1200


# ═══════════════════════════════════════════════════════
# PARAM BOUNDS HELPER
# ═══════════════════════════════════════════════════════

def get_effective_bounds(cfg: Dict[str, Any], memory_records: List[Dict] = None) -> Dict:
    """
    Get effective parameter bounds by merging:
    1. User-specified bounds from Dashboard config
    2. Auto-expanded bounds from memory (successful genomes)
    3. Default bounds from PARAM_BOUNDS

    Priority: User config > Memory expansion > Default

    Args:
        cfg: Dashboard config with parameter ranges
        memory_records: List of genome records from ParamMemory

    Returns:
        Merged bounds dict
    """
    # Start with user-specified bounds from config
    user_bounds = extract_param_bounds_from_config(cfg)

    # If no memory records, return user bounds
    if not memory_records:
        logger.info("Using user-specified bounds (no memory records)")
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
    # Convert genome to flat params
    params = genome_to_flat_params(genome)

    # Build run config (use `or {}` to handle explicit None values)
    run_cfg = {
        "meta": {
            "symbols": [symbol],
            "timeframe": timeframe
        },
        "strategy": {
            "type": "rf_st_rsi",
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

    # Progress helper
    last_emit_time = [0]

    def emit_progress(progress: int, total: int, extra: Dict = None, force: bool = False):
        now = time.time()
        if not force and now - last_emit_time[0] < PROGRESS_INTERVAL and progress != total:
            return
        last_emit_time[0] = now

        if progress_cb:
            progress_cb(job_id, progress, total, "running", extra or {})

    emit_progress(0, 100, {"phase": "initializing", "mode": mode}, force=True)

    # ═══════════════════════════════════════════════════════
    # 0. DATA PRELOAD (ULTRA mode optimization)
    # ═══════════════════════════════════════════════════════
    if mode == MODE_ULTRA and DATA_PRELOAD:
        # Preload all data into memory for maximum speed
        preloaded_count = preload_all_data()
        if preloaded_count > 0:
            logger.info(f"ULTRA mode: Preloaded {preloaded_count} datasets for max speed")
        emit_progress(2, 100, {"phase": "data_preloaded", "count": preloaded_count})

    # ═══════════════════════════════════════════════════════
    # 1. GENERATE STRATEGY HASH
    # ═══════════════════════════════════════════════════════
    strategy_type = cfg.get("strategy", {}).get("type", "rf_st_rsi")
    strategy_hash = generate_strategy_hash(strategy_type, ENGINE_VERSION)

    # Register strategy
    register_strategy(strategy_hash, strategy_type, ENGINE_VERSION)

    emit_progress(5, 100, {"phase": "strategy_hash", "hash": strategy_hash})

    # ═══════════════════════════════════════════════════════
    # 2. CLASSIFY MARKET REGIME
    # ═══════════════════════════════════════════════════════
    symbols = cfg.get("symbols", ["BTCUSDT"])
    timeframes = cfg.get("timeframes", ["1h"])
    symbol = symbols[0] if symbols else "BTCUSDT"
    timeframe = timeframes[0] if timeframes else "1h"

    # Load data for regime classification (use preloaded if available)
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

    emit_progress(10, 100, {"phase": "regime_classified", "regime": regime.value})

    # ═══════════════════════════════════════════════════════
    # 3. QUERY PARAMMEMORY FOR SEEDS
    # ═══════════════════════════════════════════════════════
    seed_genomes = []
    memory_records = []  # Keep full records for bounds expansion

    try:
        # Get similar genomes from memory
        similar = query_similar_genomes(
            strategy_hash, symbol, timeframe, market_profile, top_n=20
        )
        memory_records.extend(similar)
        seed_genomes.extend([g["genome"] for g in similar if "genome" in g])

        # Get top performers across all symbols/timeframes
        top_all = get_all_top_genomes(strategy_hash, symbols, timeframes, limit_per_combo=10)
        memory_records.extend(top_all)
        seed_genomes.extend([g["genome"] for g in top_all if "genome" in g])

        logger.info(f"Loaded {len(seed_genomes)} seed genomes from memory")
    except Exception as e:
        logger.warning(f"Failed to load from ParamMemory: {e}")

    # ═══════════════════════════════════════════════════════
    # 3.5. AUTO-EXPAND BOUNDS FROM MEMORY
    # ═══════════════════════════════════════════════════════
    # Get effective bounds - auto-expands if memory has good genomes outside user range
    effective_bounds = get_effective_bounds(cfg, memory_records)

    # Log if bounds were expanded
    default_bounds = PARAM_BOUNDS
    bounds_expanded = False
    expansion_details = []

    for block in ["entry", "sl", "tp_dual", "tp_rsi"]:
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

    # Add regime-aware random samples
    regime_samples = sample_params_for_regime(regime, n_samples=10)
    seed_genomes.extend(regime_samples)

    emit_progress(15, 100, {
        "phase": "seeds_loaded",
        "count": len(seed_genomes),
        "bounds_expanded": bounds_expanded
    })

    # ═══════════════════════════════════════════════════════
    # 4. VALIDATE COHERENCE (mode-dependent)
    # ═══════════════════════════════════════════════════════
    if mode == MODE_ULTRA:
        # ULTRA: Minimal filter - only check TP_mult >= SL_mult
        valid_seeds = []
        invalid_seeds = []
        for genome in seed_genomes:
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
        valid_seeds, invalid_seeds = validate_genome_batch(seed_genomes)

    emit_progress(20, 100, {
        "phase": "coherence_validated",
        "valid": len(valid_seeds),
        "invalid": len(invalid_seeds)
    })

    # ═══════════════════════════════════════════════════════
    # 5. CREATE FITNESS FUNCTION (mode-dependent)
    # ═══════════════════════════════════════════════════════
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

    # ═══════════════════════════════════════════════════════
    # 6. RUN GENOME OPTIMIZATION (ADAPTIVE PARAMS)
    # ═══════════════════════════════════════════════════════
    emit_progress(25, 100, {"phase": "optimization_starting"})

    num_combos = len(symbols) * len(timeframes)

    if mode == MODE_ULTRA:
        # ULTRA MODE: Maximum speed for Apple Silicon
        # No phased optimization - continuous loop
        # Optimized for M1/M2/M3 chips with limited RAM
        generations_per_phase = 2  # Minimal generations
        population_size = 40  # More genomes per batch
        top_genome_limit = 10  # Return only top 10 genomes
        logger.info(f"ULTRA mode: Apple Silicon optimized - pop={population_size}, top={top_genome_limit}")
    elif num_combos <= 1:
        # Single combo: BALANCED MODE (optimized for long date ranges)
        # Reduced from 100×5 to 30×3 for faster completion
        # 30 genomes × 3 generations × 4 phases = 360 backtests
        generations_per_phase = 3
        population_size = 30
        top_genome_limit = 10  # Return top 10 genomes
        logger.info(f"BRAIN mode: Balanced - pop={population_size}, gen={generations_per_phase}")
    elif num_combos <= 3:
        # 2-3 combos: balanced quality
        generations_per_phase = 3
        population_size = 25
        top_genome_limit = 10
    else:
        # 4+ combos: speed priority
        generations_per_phase = 2
        population_size = 20
        top_genome_limit = 8

    logger.info(
        f"Adaptive params: {num_combos} combos → "
        f"generations={generations_per_phase}, population={population_size}, "
        f"top_genomes={top_genome_limit}"
    )

    optimizer = PhasedOptimizer(
        fitness_fn=fitness_fn,
        generations_per_phase=generations_per_phase,
        population_size=population_size,
        param_bounds=effective_bounds  # Pass expanded bounds
    )

    def opt_progress(gen, total_gen, best_score):
        progress = 25 + int(gen / total_gen * 40)
        emit_progress(progress, 100, {
            "phase": "optimizing",
            "generation": gen,
            "best_score": best_score
        })

    best_genome, best_score, top_genomes = optimizer.optimize(
        seed_genomes=valid_seeds,
        regime=regime,
        progress_cb=opt_progress
    )

    emit_progress(65, 100, {"phase": "optimization_complete", "best_score": best_score})

    # ═══════════════════════════════════════════════════════
    # 7. BACKTEST TOP GENOMES (PARALLEL) - Limited for speed
    # ═══════════════════════════════════════════════════════
    emit_progress(70, 100, {"phase": "backtesting_top"})

    # Limit top genomes for faster execution
    limited_genomes = top_genomes[:top_genome_limit]

    results = []
    total_backtest = len(limited_genomes) * len(symbols) * len(timeframes)
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []

        for genome in limited_genomes:
            for sym in symbols:
                for tf in timeframes:
                    future = executor.submit(
                        run_genome_backtest, genome, sym, tf, cfg
                    )
                    futures.append(future)

        for future in as_completed(futures):
            completed += 1
            progress = 70 + int(completed / total_backtest * 15)
            emit_progress(progress, 100, {"phase": "backtesting", "completed": completed})

            try:
                result = future.result()
                if result.get("summary", {}).get("totalTrades", 0) > 0:
                    results.append(result)
            except Exception as e:
                logger.debug(f"Backtest failed: {e}")

    emit_progress(85, 100, {"phase": "backtesting_complete", "results": len(results)})

    # ═══════════════════════════════════════════════════════
    # 8. ROBUSTNESS FILTER (BRAIN_MODE only)
    # ═══════════════════════════════════════════════════════
    robust_results = []
    fragile_results = []

    if mode == MODE_BRAIN:
        # BRAIN MODE: Run full robustness testing
        emit_progress(90, 100, {"phase": "robustness_testing"})

        for idx, result in enumerate(results):
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

        emit_progress(92, 100, {
            "phase": "robustness_complete",
            "robust": len(robust_results),
            "fragile": len(fragile_results)
        })

        logger.info(f"Robustness filter: {len(robust_results)} robust, {len(fragile_results)} fragile")
    else:
        # ULTRA/FAST MODE: Skip robustness, mark all as passed
        emit_progress(90, 100, {"phase": "skipping_robustness"})

        for result in results:
            result["robustness_passed"] = True
            result["robustness_score"] = 1.0

        robust_results = results

        emit_progress(92, 100, {
            "phase": "robustness_skipped",
            "robust": len(robust_results),
            "fragile": 0
        })

    # ═══════════════════════════════════════════════════════
    # 9. SORT AND RANK (mode-dependent)
    # ═══════════════════════════════════════════════════════
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

    top_n = cfg.get("topN", TOP_N_DEFAULT)
    best = final_results[0] if final_results else None
    top = final_results[:top_n]

    # ═══════════════════════════════════════════════════════
    # 10. WRITE TO PARAMMEMORY (BRAIN_MODE only)
    # ═══════════════════════════════════════════════════════
    stored_count = 0

    if mode == MODE_BRAIN:
        # BRAIN MODE: Write robust genomes to memory
        emit_progress(95, 100, {"phase": "storing_memory"})

        # Only store genomes that passed robustness
        genomes_to_store = [r for r in results if r.get("robustness_passed", False)]

        for result in genomes_to_store:
            try:
                genome = result.get("genome", {})
                genome_hash = generate_genome_hash(genome)

                summary = result.get("summary", {})
                record = {
                    "strategy_hash": strategy_hash,
                    "symbol": result.get("symbol"),
                    "timeframe": result.get("timeframe"),
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
                    "timestamp": int(time.time()),
                    "test_count": 1
                }

                if store_genome_result(record):
                    stored_count += 1
            except Exception as e:
                logger.debug(f"Failed to store genome: {e}")

        logger.info(f"Stored {stored_count} robust genomes to ParamMemory")
    else:
        # ULTRA/FAST MODE: Skip memory write
        emit_progress(95, 100, {"phase": "skipping_memory_write"})
        logger.info(f"{mode.upper()} mode: Skipping ParamMemory write")

    # ═══════════════════════════════════════════════════════
    # 11. BUILD RESPONSE
    # ═══════════════════════════════════════════════════════
    elapsed = time.time() - start_time

    emit_progress(100, 100, {"phase": "complete", "mode": mode}, force=True)

    # Memory stats
    try:
        memory_stats = get_memory_stats(strategy_hash)
    except:
        memory_stats = {}

    # Build comment (with mode)
    comment = build_comment(best, regime, mode) if best else "No valid genome found"

    # Prune for response
    pruned_top = [prune_result(r) for r in top]
    pruned_best = prune_result(best) if best else None
    pruned_all = [prune_result(r) for r in results[:100]]

    # Mode-specific ranking info
    if mode == MODE_ULTRA:
        ranking_by = "fastScore"
    elif mode == MODE_BRAIN:
        ranking_by = "brainScore"
    else:
        ranking_by = "netProfit"

    return {
        "success": bool(best and robust_results),
        "fallback": bool(best and not robust_results),
        "message": None if best else "No genome passed all filters",
        "mode": mode,
        "ranking_by": ranking_by,
        "strategy_hash": strategy_hash,
        "market_regime": regime.value,
        "best": pruned_best,
        "bestGenomes": [
            {
                **prune_result(r),
                "comment": build_comment(r, regime, mode)
            }
            for r in top[:5]
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
            "memory_stats": memory_stats,
            "bounds_expanded": bounds_expanded,
            "expansion_details": expansion_details if bounds_expanded else [],
            "adaptive_params": {
                "num_combos": num_combos,
                "generations_per_phase": generations_per_phase,
                "population_size": population_size,
                "top_genome_limit": top_genome_limit,
            },
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
