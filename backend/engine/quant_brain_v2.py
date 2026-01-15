# engine/quant_brain_v2.py
"""
Quant Brain v2.0 - Complete AI-Powered Trading Strategy Optimization

This is the main orchestrator that integrates all v2 components:
1. Market Profile v2 (32 indicators)
2. BrainScore v2 (multi-criteria scoring)
3. Genetic Algorithm v2 (adaptive mutation, fitness sharing)
4. Robustness Testing (Walk-Forward, Monte Carlo, Sensitivity, Stress)
5. Memory System v2 (similarity-based retrieval)

Complete Workflow:
1. Load historical data
2. Calculate Market Profile v2
3. Query similar genomes from memory
4. Initialize population (memory seeds + regime-aware randoms)
5. Run genetic optimization with adaptive mutation
6. Calculate BrainScore for all candidates
7. Run robustness testing on top candidates
8. Filter by stability threshold
9. Store results to memory
10. Return best genomes with explanations

Usage:
    result = await quant_brain_optimize_v2(config, job_id, progress_callback)
"""

import os
import time
import copy
import logging
import hashlib
from typing import Dict, Any, List, Callable, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

# Import v2 components
from .market_profile_v2 import (
    calculate_market_profile_v2,
    get_simplified_profile,
    calculate_profile_similarity
)
from .brain_score import (
    calculate_brain_score,
    calculate_all_metrics,
    quick_score,
    get_pareto_front,
    get_score_breakdown
)
from .genome_optimizer_v2 import (
    GenomeOptimizerV2,
    PhasedOptimizerV2,
    calculate_genome_distance
)
from .robustness_testing import (
    comprehensive_robustness_test,
    quick_robustness_check,
    monte_carlo_simulation,
    WalkForwardResult
)
from .param_memory_v2 import (
    store_genome_result_v2,
    query_similar_genomes_v2,
    get_top_genomes_v2,
    get_best_genome_for_conditions,
    profile_from_simple
)
from .strategy_hash import generate_strategy_hash, ENGINE_VERSION
from .coherence_validator import validate_genome, repair_genome

# Import existing backtest engine
from .backtest_engine import run_backtest

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_POPULATION_SIZE = int(os.getenv("GENOME_POPULATION_SIZE", 30))
DEFAULT_GENERATIONS = int(os.getenv("GENOME_GENERATIONS", 5))
DEFAULT_STABILITY_THRESHOLD = float(os.getenv("STABILITY_THRESHOLD", 0.6))
TOP_GENOMES_RETURN = int(os.getenv("TOP_GENOMES_RETURN", 20))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN OPTIMIZATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

async def quant_brain_optimize_v2(
    config: Dict[str, Any],
    job_id: str = "",
    progress_cb: Callable = None,
    skip_robustness: bool = False
) -> Dict[str, Any]:
    """
    Main Quant Brain v2.0 optimization function.

    Args:
        config: {
            "symbol": str,
            "timeframe": str or list,
            "strategy": {"type": str, "params": {...}},
            "capital": {"initial": float, "orderPct": float},
            "risk": {"pyramiding": int, "commission": float},
            "range": {"from": str, "to": str}  # optional
        }
        job_id: Job identifier for progress tracking
        progress_cb: Callback(stage, progress, message)
        skip_robustness: Skip robustness testing for speed

    Returns:
        {
            "success": bool,
            "strategy_hash": str,
            "market_profile": {...},
            "best_genomes": [...],
            "meta": {...}
        }
    """
    start_time = time.time()
    optimization_log = []

    def log_stage(stage: str, message: str, progress: float = 0):
        """Log optimization progress."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        optimization_log.append(f"[{timestamp}] {stage}: {message}")
        logger.info(f"[{job_id}] {stage}: {message}")
        if progress_cb:
            progress_cb(stage, progress, message)

    try:
        # ═══════════════════════════════════════════════════════
        # STEP 1: Parse configuration
        # ═══════════════════════════════════════════════════════
        log_stage("INIT", "Parsing configuration", 0.05)

        symbol = config.get("symbol", "BTCUSDT")
        timeframes = config.get("timeframes", config.get("timeframe", ["30m"]))
        if isinstance(timeframes, str):
            timeframes = [timeframes]

        strategy_type = config.get("strategy", {}).get("type", "rf_st_rsi")
        strategy_params = config.get("strategy", {}).get("params", {})

        initial_capital = config.get("capital", {}).get("initial", 10000)
        commission = config.get("risk", {}).get("commission", 0.04)

        # Generate strategy hash
        strategy_hash = generate_strategy_hash(strategy_type, ENGINE_VERSION)
        log_stage("INIT", f"Strategy hash: {strategy_hash[:16]}...", 0.08)

        # ═══════════════════════════════════════════════════════
        # STEP 2: Load historical data
        # ═══════════════════════════════════════════════════════
        log_stage("DATA", "Loading historical data", 0.10)

        # Import data loader
        from .data_loader import load_data_for_symbol

        df_dict = {}
        for tf in timeframes:
            df = load_data_for_symbol(symbol, tf, config.get("range"))
            if df is not None and len(df) > 100:
                df_dict[f"{symbol}_{tf}"] = df
                log_stage("DATA", f"Loaded {symbol}_{tf}: {len(df)} bars", 0.12)
            else:
                log_stage("DATA", f"Warning: Insufficient data for {symbol}_{tf}", 0.12)

        if not df_dict:
            return {
                "success": False,
                "error": "No data available for optimization",
                "log": optimization_log
            }

        # Use primary timeframe
        primary_key = f"{symbol}_{timeframes[0]}"
        primary_df = df_dict.get(primary_key)

        # ═══════════════════════════════════════════════════════
        # STEP 3: Calculate Market Profile v2
        # ═══════════════════════════════════════════════════════
        log_stage("PROFILE", "Calculating Market Profile v2", 0.15)

        # Load BTC data for correlation (if not BTC itself)
        btc_df = None
        if symbol != "BTCUSDT":
            btc_df = load_data_for_symbol("BTCUSDT", timeframes[0], config.get("range"))

        market_profile = calculate_market_profile_v2(
            primary_df,
            btc_df=btc_df,
            lookback=min(500, len(primary_df))
        )

        regime = market_profile.get("summary", {}).get("regime", "ranging")
        log_stage("PROFILE", f"Market regime: {regime}, 32 indicators calculated", 0.18)

        # ═══════════════════════════════════════════════════════
        # STEP 4: Query similar genomes from memory
        # ═══════════════════════════════════════════════════════
        log_stage("MEMORY", "Querying similar genomes from memory", 0.20)

        similar_genomes = query_similar_genomes_v2(
            strategy_hash,
            symbol,
            timeframes[0],
            market_profile,
            top_n=30,
            similarity_threshold=0.5,
            min_robustness=0.3
        )

        seed_genomes = [g.get("genome", {}) for g in similar_genomes if g.get("genome")]
        log_stage("MEMORY", f"Found {len(seed_genomes)} similar genomes in memory", 0.23)

        # ═══════════════════════════════════════════════════════
        # STEP 5: Create fitness function
        # ═══════════════════════════════════════════════════════
        log_stage("FITNESS", "Creating fitness function", 0.25)

        def fitness_function(genome: Dict) -> float:
            """Evaluate genome fitness using backtest."""
            try:
                # Validate genome
                valid, violations = validate_genome(genome)
                if not valid:
                    return float("-inf")

                # Build params
                params = _genome_to_params(genome, strategy_params)

                # Run backtest
                result = run_backtest(
                    df=primary_df.copy(),
                    params=params,
                    initial_capital=initial_capital,
                    commission_pct=commission
                )

                if not result or not result.get("summary"):
                    return float("-inf")

                summary = result["summary"]

                # Calculate BrainScore
                metrics = {
                    "netProfitPct": summary.get("netProfitPct", 0),
                    "profitFactor": summary.get("profitFactor", 0),
                    "winrate": summary.get("winrate", 0),
                    "maxDrawdownPct": summary.get("maxDrawdownPct", 100),
                    "sharpeRatio": summary.get("sharpeRatio", 0),
                    "sortinoRatio": summary.get("sortinoRatio", 0),
                    "totalTrades": summary.get("totalTrades", 0),
                    "maxLossStreak": summary.get("maxLossStreak", 10),
                    "smoothness": summary.get("smoothness", 0),
                    "ulcerIndex": summary.get("ulcerIndex", 10),
                    "expectancy": summary.get("expectancy", 0),
                    "recoveryFactor": summary.get("recoveryFactor", 0),
                    "pValue": summary.get("pValue", 1),
                    "tStatistic": summary.get("tStatistic", 0)
                }

                brain_score = calculate_brain_score(metrics)
                return brain_score

            except Exception as e:
                logger.warning(f"Fitness evaluation error: {e}")
                return float("-inf")

        # ═══════════════════════════════════════════════════════
        # STEP 6: Run genetic optimization
        # ═══════════════════════════════════════════════════════
        log_stage("OPTIMIZE", "Running genetic algorithm v2", 0.30)

        # Determine optimization parameters
        pop_size = DEFAULT_POPULATION_SIZE
        generations = DEFAULT_GENERATIONS

        # Expand bounds from strategy params
        param_bounds = _expand_bounds_from_params(strategy_params)

        # Create optimizer
        optimizer = GenomeOptimizerV2(
            fitness_fn=fitness_function,
            population_size=pop_size,
            generations=generations,
            mutation_rate_max=0.3,
            mutation_rate_min=0.05,
            use_fitness_sharing=True,
            crossover_type="blx_alpha",
            param_bounds=param_bounds
        )

        # Progress callback for optimizer
        def opt_progress(gen, total_gen, best_score):
            progress = 0.30 + (gen / total_gen) * 0.40
            log_stage("OPTIMIZE", f"Generation {gen}/{total_gen}, best={best_score:.4f}", progress)

        # Run optimization
        from .regime_classifier import MarketRegime

        regime_enum = _get_regime_enum(regime)

        best_genome, best_score, top_genomes = optimizer.optimize(
            seed_genomes=seed_genomes,
            regime=regime_enum,
            progress_cb=opt_progress
        )

        log_stage("OPTIMIZE", f"Optimization complete, best score: {best_score:.4f}", 0.70)

        # ═══════════════════════════════════════════════════════
        # STEP 7: Run full backtest on top candidates
        # ═══════════════════════════════════════════════════════
        log_stage("BACKTEST", "Running detailed backtest on top candidates", 0.72)

        evaluated_genomes = []

        for i, genome in enumerate(top_genomes[:TOP_GENOMES_RETURN * 2]):
            try:
                params = _genome_to_params(genome, strategy_params)

                result = run_backtest(
                    df=primary_df.copy(),
                    params=params,
                    initial_capital=initial_capital,
                    commission_pct=commission
                )

                if result and result.get("summary"):
                    summary = result["summary"]

                    # Calculate full metrics
                    equity_curve = result.get("equityCurve", [])
                    trades = result.get("trades", [])

                    if equity_curve:
                        full_metrics = calculate_all_metrics(
                            equity_curve,
                            trades,
                            initial_capital
                        )
                        # Merge with summary
                        summary.update(full_metrics)

                    # Calculate BrainScore
                    brain_score = calculate_brain_score(summary)
                    summary["brainScore"] = brain_score

                    evaluated_genomes.append({
                        "genome": genome,
                        "summary": summary,
                        "trades": trades,
                        "equity_curve": equity_curve
                    })

            except Exception as e:
                logger.warning(f"Backtest failed for genome {i}: {e}")

            if progress_cb:
                progress = 0.72 + (i / len(top_genomes[:TOP_GENOMES_RETURN * 2])) * 0.08
                progress_cb("BACKTEST", progress, f"Evaluated {i + 1}/{len(top_genomes[:TOP_GENOMES_RETURN * 2])}")

        log_stage("BACKTEST", f"Evaluated {len(evaluated_genomes)} candidates", 0.80)

        # ═══════════════════════════════════════════════════════
        # STEP 8: Robustness testing
        # ═══════════════════════════════════════════════════════
        if not skip_robustness and evaluated_genomes:
            log_stage("ROBUST", "Running robustness testing", 0.82)

            for i, candidate in enumerate(evaluated_genomes):
                genome = candidate["genome"]
                trades = candidate.get("trades", [])

                try:
                    # Quick robustness check
                    passed, stability = quick_robustness_check(
                        genome,
                        fitness_function
                    )

                    # Monte Carlo if trades available
                    if trades and len(trades) >= 10:
                        mc_result = monte_carlo_simulation(
                            trades,
                            initial_capital,
                            n_simulations=100  # Reduced for speed
                        )
                        candidate["robustness"] = {
                            "stability_score": stability,
                            "passed": passed,
                            "monte_carlo_score": mc_result.score,
                            "profit_probability": mc_result.profit_probability,
                            "var_95": mc_result.var_95
                        }
                    else:
                        candidate["robustness"] = {
                            "stability_score": stability,
                            "passed": passed
                        }

                except Exception as e:
                    logger.warning(f"Robustness test failed: {e}")
                    candidate["robustness"] = {
                        "stability_score": 0,
                        "passed": False
                    }

                if progress_cb:
                    progress = 0.82 + (i / len(evaluated_genomes)) * 0.08
                    progress_cb("ROBUST", progress, f"Testing {i + 1}/{len(evaluated_genomes)}")

            log_stage("ROBUST", "Robustness testing complete", 0.90)
        else:
            log_stage("ROBUST", "Robustness testing skipped", 0.90)
            for candidate in evaluated_genomes:
                candidate["robustness"] = {"stability_score": 0.5, "passed": True}

        # ═══════════════════════════════════════════════════════
        # STEP 9: Rank and filter results
        # ═══════════════════════════════════════════════════════
        log_stage("RANK", "Ranking and filtering results", 0.92)

        # Sort by combined score (BrainScore * stability)
        def combined_rank_score(candidate):
            brain = candidate.get("summary", {}).get("brainScore", 0)
            stability = candidate.get("robustness", {}).get("stability_score", 0.5)
            return brain * (1 + stability * 0.3)

        evaluated_genomes.sort(key=combined_rank_score, reverse=True)

        # Filter by stability threshold
        stable_genomes = [
            g for g in evaluated_genomes
            if g.get("robustness", {}).get("stability_score", 0) >= DEFAULT_STABILITY_THRESHOLD * 0.7
        ]

        # Use stable if available, otherwise use top
        final_genomes = stable_genomes[:TOP_GENOMES_RETURN] if stable_genomes else evaluated_genomes[:TOP_GENOMES_RETURN]

        log_stage("RANK", f"Selected {len(final_genomes)} best genomes", 0.94)

        # ═══════════════════════════════════════════════════════
        # STEP 10: Store to memory
        # ═══════════════════════════════════════════════════════
        log_stage("MEMORY", "Storing results to memory", 0.96)

        stored_count = 0
        for candidate in final_genomes:
            genome = candidate["genome"]
            genome_hash = _generate_genome_hash(genome)

            record = {
                "strategy_hash": strategy_hash,
                "symbol": symbol,
                "timeframe": timeframes[0],
                "genome_hash": genome_hash,
                "market_profile": market_profile,
                "genome": genome,
                "results": candidate.get("summary", {}),
                "robustness": candidate.get("robustness", {}),
                "timestamp": int(time.time()),
                "test_count": 1
            }

            if store_genome_result_v2(record):
                stored_count += 1

        log_stage("MEMORY", f"Stored {stored_count} genomes to memory", 0.98)

        # ═══════════════════════════════════════════════════════
        # STEP 11: Format output
        # ═══════════════════════════════════════════════════════
        log_stage("DONE", "Formatting output", 1.0)

        elapsed_time = time.time() - start_time

        # Format best genomes for response
        best_genomes_output = []
        for i, candidate in enumerate(final_genomes):
            genome = candidate["genome"]
            summary = candidate.get("summary", {})
            robustness = candidate.get("robustness", {})

            # Generate explanation
            explanation = _generate_genome_explanation(
                genome, summary, robustness, regime
            )

            # Downsample equity curve
            equity = candidate.get("equity_curve", [])
            if len(equity) > 200:
                indices = np.linspace(0, len(equity) - 1, 200, dtype=int)
                equity = [equity[i] for i in indices]

            best_genomes_output.append({
                "rank": i + 1,
                "params": _genome_to_params(genome, {}),
                "genome": genome,
                "summary": summary,
                "robustness": robustness,
                "equityCurve": equity,
                "comment": explanation
            })

        return {
            "success": True,
            "strategy_hash": strategy_hash,
            "symbol": symbol,
            "timeframe": timeframes[0],
            "market_profile": market_profile,
            "regime": regime,
            "best_genomes": best_genomes_output,
            "meta": {
                "total_tested": len(optimizer.generation_history) * pop_size if optimizer.generation_history else 0,
                "from_memory": len(seed_genomes),
                "stable_genomes": len(stable_genomes),
                "elapsed_seconds": round(elapsed_time, 2),
                "population_size": pop_size,
                "generations": generations,
                "engine_version": ENGINE_VERSION
            },
            "log": optimization_log
        }

    except Exception as e:
        logger.error(f"Quant Brain v2 error: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "log": optimization_log
        }


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _genome_to_params(genome: Dict, base_params: Dict) -> Dict:
    """Convert genome to strategy parameters."""
    params = copy.deepcopy(base_params) if base_params else {}

    # Entry params
    entry = genome.get("entry", {})
    params["st_atrPeriod"] = entry.get("st_atrPeriod", 10)
    params["st_mult"] = entry.get("st_mult", 2.0)
    params["rf_period"] = entry.get("rf_period", 100)
    params["rf_mult"] = entry.get("rf_mult", 3.0)
    params["rsi_length"] = entry.get("rsi_length", 14)
    params["rsi_ma_length"] = entry.get("rsi_ma_length", 5)

    # SL params
    sl = genome.get("sl", {})
    params["sl_st_atrPeriod"] = sl.get("st_atrPeriod", 10)
    params["sl_st_mult"] = sl.get("st_mult", 4.0)
    params["sl_rf_period"] = sl.get("rf_period", 100)
    params["sl_rf_mult"] = sl.get("rf_mult", 7.0)

    # TP params
    tp_dual = genome.get("tp_dual", {})
    params["tp_dual_st_atrPeriod"] = tp_dual.get("st_atrPeriod", 10)
    params["tp_dual_st_mult"] = tp_dual.get("st_mult", 2.0)
    params["tp_dual_rr_mult"] = tp_dual.get("rr_mult", 1.3)

    tp_rsi = genome.get("tp_rsi", {})
    params["tp_rsi_st_atrPeriod"] = tp_rsi.get("st_atrPeriod", 10)
    params["tp_rsi_st_mult"] = tp_rsi.get("st_mult", 2.0)
    params["tp_rsi_rr_mult"] = tp_rsi.get("rr_mult", 1.3)

    # Mode
    mode = genome.get("mode", {})
    params["showDualFlip"] = mode.get("showDualFlip", True)
    params["showRSI"] = mode.get("showRSI", True)

    return params


def _expand_bounds_from_params(params: Dict) -> Dict:
    """Expand parameter bounds from strategy params."""
    bounds = {
        "entry": {
            "st_atrPeriod": (
                params.get("st_atrPeriod", {}).get("start", 1),
                params.get("st_atrPeriod", {}).get("end", 100)
            ),
            "st_mult": (
                params.get("st_mult", {}).get("start", 1),
                params.get("st_mult", {}).get("end", 30)
            ),
            "rf_period": (
                params.get("rf_period", {}).get("start", 1),
                params.get("rf_period", {}).get("end", 100)
            ),
            "rf_mult": (
                params.get("rf_mult", {}).get("start", 1),
                params.get("rf_mult", {}).get("end", 30)
            ),
            "rsi_length": (
                params.get("rsi_length", {}).get("start", 1),
                params.get("rsi_length", {}).get("end", 20)
            ),
            "rsi_ma_length": (
                params.get("rsi_ma_length", {}).get("start", 1),
                params.get("rsi_ma_length", {}).get("end", 15)
            ),
        },
        "sl": {
            "st_atrPeriod": (1, 100),
            "st_mult": (1, 30),
            "rf_period": (1, 100),
            "rf_mult": (1, 30),
        },
        "tp_dual": {
            "st_atrPeriod": (1, 100),
            "st_mult": (1, 30),
            "rr_mult": (0.1, 5),
        },
        "tp_rsi": {
            "st_atrPeriod": (1, 100),
            "st_mult": (1, 30),
            "rr_mult": (0.1, 5),
        },
    }

    return bounds


def _get_regime_enum(regime_str: str):
    """Convert regime string to enum."""
    from .regime_classifier import MarketRegime

    regime_map = {
        "trending_up": MarketRegime.TRENDING_UP,
        "trending_down": MarketRegime.TRENDING_DOWN,
        "ranging": MarketRegime.RANGING,
        "volatile": MarketRegime.VOLATILE,
        "low_volatility": MarketRegime.LOW_VOLATILITY,
    }

    return regime_map.get(regime_str, MarketRegime.RANGING)


def _generate_genome_hash(genome: Dict) -> str:
    """Generate unique hash for genome."""
    # Create deterministic string representation
    genome_str = json.dumps(genome, sort_keys=True, default=str)
    return hashlib.sha256(genome_str.encode()).hexdigest()[:32]


def _generate_genome_explanation(
    genome: Dict,
    summary: Dict,
    robustness: Dict,
    regime: str
) -> str:
    """Generate human-readable explanation for genome performance."""
    entry = genome.get("entry", {})
    sl = genome.get("sl", {})
    tp = genome.get("tp_dual", {})

    pf = summary.get("profitFactor", 0)
    wr = summary.get("winrate", 0)
    dd = summary.get("maxDrawdownPct", 0)
    brain_score = summary.get("brainScore", 0)
    stability = robustness.get("stability_score", 0)

    # Build explanation
    parts = []

    # Regime context
    regime_context = {
        "trending_up": "uptrending market",
        "trending_down": "downtrending market",
        "ranging": "ranging market",
        "volatile": "volatile market",
        "low_volatility": "quiet market"
    }
    parts.append(f"Optimized for {regime_context.get(regime, regime)}")

    # Key parameters
    rf_period = entry.get("rf_period", 100)
    st_mult = entry.get("st_mult", 2)
    rr = tp.get("rr_mult", 1.3)

    if rf_period > 80:
        parts.append(f"Long RF period ({rf_period}) for trend confirmation")
    else:
        parts.append(f"Short RF period ({rf_period}) for quick signals")

    if st_mult < 2.5:
        parts.append(f"Tight SuperTrend (×{st_mult}) for early entries")
    else:
        parts.append(f"Wide SuperTrend (×{st_mult}) to avoid whipsaws")

    # Performance highlights
    parts.append(f"Achieved PF {pf:.2f} with {wr:.1f}% win rate and {dd:.1f}% max DD")

    # Robustness
    if stability >= 0.7:
        parts.append("Highly stable under parameter perturbations")
    elif stability >= 0.5:
        parts.append("Moderately stable")
    else:
        parts.append("Some parameter sensitivity detected")

    return ". ".join(parts) + "."


# Import json for hash generation
import json


# ═══════════════════════════════════════════════════════════════════════════════
# SYNCHRONOUS WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

def quant_brain_optimize_v2_sync(
    config: Dict[str, Any],
    job_id: str = "",
    progress_cb: Callable = None,
    skip_robustness: bool = False
) -> Dict[str, Any]:
    """
    Synchronous wrapper for quant_brain_optimize_v2.

    Use this in non-async contexts.
    """
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(
        quant_brain_optimize_v2(config, job_id, progress_cb, skip_robustness)
    )
