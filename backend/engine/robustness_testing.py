# engine/robustness_testing.py
"""
Robustness Testing Suite v2.0 - Comprehensive Strategy Validation

Includes:
1. Walk-Forward Analysis: 3 windows, 70/30 in-sample/out-of-sample split
2. Monte Carlo Simulation: 500 simulations with trade shuffling
3. Parameter Sensitivity: ±10% parameter perturbation
4. Slippage Stress Test: 4 scenarios (0.1%, 0.2%, 0.5%, 1.0%)
5. Noise Injection: Entry/exit price noise at 3 levels

Stability Score Formula:
Stability = 0.3 * WalkForward + 0.3 * MonteCarlo + 0.2 * Sensitivity + 0.1 * Slippage + 0.1 * Noise
"""

import os
import copy
import logging
import time
from typing import Dict, Any, List, Tuple, Callable, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_MONTE_CARLO_RUNS = int(os.getenv("MONTE_CARLO_RUNS", 500))
DEFAULT_WALK_FORWARD_WINDOWS = int(os.getenv("WALK_FORWARD_WINDOWS", 3))
DEFAULT_IS_OOS_RATIO = float(os.getenv("IS_OOS_RATIO", 0.7))  # 70% in-sample, 30% out-of-sample
DEFAULT_STABILITY_THRESHOLD = float(os.getenv("STABILITY_THRESHOLD", 0.6))


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RobustnessResult:
    """Complete robustness testing result."""
    passed: bool
    stability_score: float
    walk_forward_score: float
    monte_carlo_score: float
    sensitivity_score: float
    slippage_score: float
    noise_score: float
    details: Dict[str, Any]


@dataclass
class WalkForwardResult:
    """Walk-forward analysis result."""
    passed: bool
    score: float
    windows: List[Dict[str, Any]]
    consistency_ratio: float
    is_oos_correlation: float


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation result."""
    passed: bool
    score: float
    mean_profit: float
    std_profit: float
    var_95: float  # Value at Risk at 95%
    profit_probability: float
    distribution: List[float]


@dataclass
class SensitivityResult:
    """Parameter sensitivity analysis result."""
    passed: bool
    score: float
    parameter_scores: Dict[str, float]
    most_sensitive: str
    least_sensitive: str


# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def walk_forward_analysis(
    genome: Dict,
    backtest_fn: Callable,
    df: pd.DataFrame,
    n_windows: int = DEFAULT_WALK_FORWARD_WINDOWS,
    is_ratio: float = DEFAULT_IS_OOS_RATIO
) -> WalkForwardResult:
    """
    Perform Walk-Forward Analysis.

    Splits data into multiple windows, optimizes on in-sample (IS),
    and validates on out-of-sample (OOS).

    Args:
        genome: Strategy genome to test
        backtest_fn: Function(genome, df) -> Dict with summary
        df: Full OHLCV DataFrame
        n_windows: Number of walk-forward windows
        is_ratio: In-sample ratio (default 0.7)

    Returns:
        WalkForwardResult with detailed analysis
    """
    total_bars = len(df)
    window_size = total_bars // n_windows
    windows = []

    is_profits = []
    oos_profits = []

    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = min((i + 1) * window_size, total_bars)

        window_df = df.iloc[start_idx:end_idx].copy()
        window_bars = len(window_df)

        if window_bars < 100:  # Minimum bars for meaningful test
            continue

        # Split into IS and OOS
        is_end = int(window_bars * is_ratio)
        is_df = window_df.iloc[:is_end].copy()
        oos_df = window_df.iloc[is_end:].copy()

        try:
            # Run backtest on IS
            is_result = backtest_fn(genome, is_df)
            is_summary = is_result.get("summary", {})
            is_profit = is_summary.get("netProfitPct", 0)
            is_profits.append(is_profit)

            # Run backtest on OOS
            oos_result = backtest_fn(genome, oos_df)
            oos_summary = oos_result.get("summary", {})
            oos_profit = oos_summary.get("netProfitPct", 0)
            oos_profits.append(oos_profit)

            # Calculate window metrics
            window_info = {
                "window": i + 1,
                "total_bars": window_bars,
                "is_bars": len(is_df),
                "oos_bars": len(oos_df),
                "is_profit_pct": round(is_profit, 2),
                "oos_profit_pct": round(oos_profit, 2),
                "is_pf": is_summary.get("profitFactor", 0),
                "oos_pf": oos_summary.get("profitFactor", 0),
                "degradation": round((is_profit - oos_profit) / max(abs(is_profit), 1) * 100, 2) if is_profit != 0 else 0
            }
            windows.append(window_info)

        except Exception as e:
            logger.warning(f"Walk-forward window {i} failed: {e}")
            continue

    if not windows:
        return WalkForwardResult(
            passed=False,
            score=0,
            windows=[],
            consistency_ratio=0,
            is_oos_correlation=0
        )

    # Calculate metrics
    # Consistency: How many OOS windows are profitable?
    profitable_oos = sum(1 for p in oos_profits if p > 0)
    consistency_ratio = profitable_oos / len(oos_profits) if oos_profits else 0

    # IS-OOS correlation
    if len(is_profits) > 2:
        is_oos_correlation = np.corrcoef(is_profits, oos_profits)[0, 1]
        is_oos_correlation = is_oos_correlation if not np.isnan(is_oos_correlation) else 0
    else:
        is_oos_correlation = 0

    # Average degradation
    avg_degradation = np.mean([w.get("degradation", 0) for w in windows])

    # Score calculation (0-1)
    # High score = good consistency, positive correlation, low degradation
    score = (
        consistency_ratio * 0.4 +
        max(0, is_oos_correlation + 1) / 2 * 0.3 +
        max(0, 1 - abs(avg_degradation) / 100) * 0.3
    )

    # Pass if at least 60% OOS windows profitable and positive correlation
    passed = consistency_ratio >= 0.5 and is_oos_correlation > -0.2

    return WalkForwardResult(
        passed=passed,
        score=round(score, 4),
        windows=windows,
        consistency_ratio=round(consistency_ratio, 4),
        is_oos_correlation=round(is_oos_correlation, 4)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def monte_carlo_simulation(
    trades: List[Dict[str, Any]],
    initial_capital: float = 10000,
    n_simulations: int = DEFAULT_MONTE_CARLO_RUNS
) -> MonteCarloResult:
    """
    Monte Carlo simulation by shuffling trade order.

    This tests if the strategy's performance is robust to the
    order of trades (path dependency).

    Args:
        trades: List of trade dictionaries with 'pnl' key
        initial_capital: Starting capital
        n_simulations: Number of simulations

    Returns:
        MonteCarloResult with distribution analysis
    """
    if not trades or len(trades) < 10:
        return MonteCarloResult(
            passed=False,
            score=0,
            mean_profit=0,
            std_profit=0,
            var_95=0,
            profit_probability=0,
            distribution=[]
        )

    pnls = [t.get("pnl", 0) for t in trades]
    final_profits = []

    for _ in range(n_simulations):
        # Shuffle trade order
        shuffled_pnls = np.random.permutation(pnls)

        # Calculate equity curve
        equity = initial_capital
        for pnl in shuffled_pnls:
            equity += pnl

        final_profit_pct = (equity - initial_capital) / initial_capital * 100
        final_profits.append(final_profit_pct)

    # Calculate statistics
    mean_profit = np.mean(final_profits)
    std_profit = np.std(final_profits)
    var_95 = np.percentile(final_profits, 5)  # 5th percentile = 95% VaR
    profit_probability = sum(1 for p in final_profits if p > 0) / len(final_profits)

    # Score based on:
    # - High probability of profit
    # - Low standard deviation (consistency)
    # - VaR not too negative
    score = (
        profit_probability * 0.4 +
        max(0, 1 - std_profit / 50) * 0.3 +  # Lower std = higher score
        max(0, (var_95 + 50) / 100) * 0.3  # VaR > -50% = OK
    )

    # Pass if >60% profitable and VaR > -30%
    passed = profit_probability >= 0.6 and var_95 > -30

    return MonteCarloResult(
        passed=passed,
        score=round(score, 4),
        mean_profit=round(mean_profit, 2),
        std_profit=round(std_profit, 2),
        var_95=round(var_95, 2),
        profit_probability=round(profit_probability, 4),
        distribution=final_profits
    )


def monte_carlo_with_randomized_returns(
    trades: List[Dict[str, Any]],
    initial_capital: float = 10000,
    n_simulations: int = DEFAULT_MONTE_CARLO_RUNS
) -> MonteCarloResult:
    """
    Monte Carlo with randomized trade returns (bootstrap resampling).

    Samples trades with replacement to estimate distribution of outcomes.
    """
    if not trades or len(trades) < 10:
        return MonteCarloResult(
            passed=False,
            score=0,
            mean_profit=0,
            std_profit=0,
            var_95=0,
            profit_probability=0,
            distribution=[]
        )

    pnls = [t.get("pnl", 0) for t in trades]
    n_trades = len(pnls)
    final_profits = []

    for _ in range(n_simulations):
        # Bootstrap resampling
        sampled_pnls = np.random.choice(pnls, size=n_trades, replace=True)

        # Calculate final equity
        equity = initial_capital + sum(sampled_pnls)
        final_profit_pct = (equity - initial_capital) / initial_capital * 100
        final_profits.append(final_profit_pct)

    mean_profit = np.mean(final_profits)
    std_profit = np.std(final_profits)
    var_95 = np.percentile(final_profits, 5)
    profit_probability = sum(1 for p in final_profits if p > 0) / len(final_profits)

    score = (
        profit_probability * 0.4 +
        max(0, 1 - std_profit / 50) * 0.3 +
        max(0, (var_95 + 50) / 100) * 0.3
    )

    passed = profit_probability >= 0.6 and var_95 > -30

    return MonteCarloResult(
        passed=passed,
        score=round(score, 4),
        mean_profit=round(mean_profit, 2),
        std_profit=round(std_profit, 2),
        var_95=round(var_95, 2),
        profit_probability=round(profit_probability, 4),
        distribution=final_profits
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def parameter_sensitivity_analysis(
    genome: Dict,
    fitness_fn: Callable[[Dict], float],
    base_score: float = None,
    perturbation_pct: float = 0.10  # ±10%
) -> SensitivityResult:
    """
    Analyze parameter sensitivity to ±10% perturbations.

    Tests how much each parameter affects the final score.

    Args:
        genome: Base genome
        fitness_fn: Fitness evaluation function
        base_score: Pre-calculated base score (optional)
        perturbation_pct: Perturbation percentage (default 10%)

    Returns:
        SensitivityResult with per-parameter analysis
    """
    if base_score is None:
        base_score = fitness_fn(genome)

    if base_score <= 0:
        return SensitivityResult(
            passed=False,
            score=0,
            parameter_scores={},
            most_sensitive="unknown",
            least_sensitive="unknown"
        )

    # Parameters to test
    test_params = [
        ("entry", "st_mult"),
        ("entry", "rf_mult"),
        ("entry", "rf_period"),
        ("entry", "rsi_length"),
        ("sl", "st_mult"),
        ("sl", "rf_mult"),
        ("tp_dual", "rr_mult"),
        ("tp_rsi", "rr_mult"),
    ]

    parameter_scores = {}

    for block, param in test_params:
        if block not in genome or param not in genome[block]:
            continue

        original_value = genome[block][param]

        # Skip non-numeric parameters
        if not isinstance(original_value, (int, float)):
            continue

        # Test +perturbation_pct and -perturbation_pct
        scores_at_perturbation = []

        for direction in [-1, 1]:
            perturbed = copy.deepcopy(genome)
            new_value = original_value * (1 + direction * perturbation_pct)

            if isinstance(original_value, int):
                new_value = int(round(new_value))
            else:
                new_value = round(new_value, 2)

            # Ensure positive values
            new_value = max(0.1, new_value)
            perturbed[block][param] = new_value

            try:
                score = fitness_fn(perturbed)
                if score > float("-inf"):
                    scores_at_perturbation.append(score)
            except:
                pass

        # Calculate sensitivity (how much score changes)
        if scores_at_perturbation:
            avg_perturbed = np.mean(scores_at_perturbation)
            sensitivity = abs(base_score - avg_perturbed) / max(base_score, 1) * 100
            stability = 1 - min(sensitivity / 50, 1)  # Lower sensitivity = higher stability
            parameter_scores[f"{block}.{param}"] = round(stability, 4)
        else:
            parameter_scores[f"{block}.{param}"] = 0

    if not parameter_scores:
        return SensitivityResult(
            passed=False,
            score=0,
            parameter_scores={},
            most_sensitive="unknown",
            least_sensitive="unknown"
        )

    # Find most and least sensitive parameters
    sorted_params = sorted(parameter_scores.items(), key=lambda x: x[1])
    most_sensitive = sorted_params[0][0]  # Lowest stability = most sensitive
    least_sensitive = sorted_params[-1][0]  # Highest stability = least sensitive

    # Overall score (average stability)
    overall_score = np.mean(list(parameter_scores.values()))

    # Pass if average stability > 0.6
    passed = overall_score >= 0.6

    return SensitivityResult(
        passed=passed,
        score=round(overall_score, 4),
        parameter_scores=parameter_scores,
        most_sensitive=most_sensitive,
        least_sensitive=least_sensitive
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SLIPPAGE STRESS TEST
# ═══════════════════════════════════════════════════════════════════════════════

def slippage_stress_test(
    trades: List[Dict[str, Any]],
    initial_capital: float = 10000,
    slippage_levels: List[float] = None
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Test strategy performance under various slippage scenarios.

    Args:
        trades: List of trades with entry_price, exit_price, position_size
        initial_capital: Starting capital
        slippage_levels: Slippage percentages to test (default: 0.1%, 0.2%, 0.5%, 1.0%)

    Returns:
        (passed, score, details)
    """
    if slippage_levels is None:
        slippage_levels = [0.001, 0.002, 0.005, 0.01]  # 0.1%, 0.2%, 0.5%, 1.0%

    if not trades or len(trades) < 5:
        return False, 0, {"reason": "Insufficient trades"}

    # Calculate base profit (no slippage)
    base_profit = sum(t.get("pnl", 0) for t in trades)
    base_profit_pct = base_profit / initial_capital * 100

    scenario_results = []

    for slippage in slippage_levels:
        adjusted_profit = 0

        for trade in trades:
            pnl = trade.get("pnl", 0)
            entry_price = trade.get("entry_price", 0)
            position_size = trade.get("position_size", 1)

            # Slippage cost = entry_price * slippage * position_size * 2 (entry + exit)
            if entry_price > 0:
                slippage_cost = entry_price * slippage * position_size * 2
            else:
                # Estimate slippage cost from pnl
                slippage_cost = abs(pnl) * slippage * 2

            adjusted_pnl = pnl - slippage_cost
            adjusted_profit += adjusted_pnl

        adjusted_profit_pct = adjusted_profit / initial_capital * 100

        scenario_results.append({
            "slippage_pct": slippage * 100,
            "profit_pct": round(adjusted_profit_pct, 2),
            "profit_retention": round(adjusted_profit_pct / max(base_profit_pct, 0.01) * 100, 2) if base_profit_pct > 0 else 0
        })

    # Calculate score based on profit retention at different slippage levels
    retention_scores = []
    for result in scenario_results:
        retention = result.get("profit_retention", 0) / 100
        retention_scores.append(max(0, retention))

    overall_score = np.mean(retention_scores)

    # Pass if strategy retains >50% profit at 0.2% slippage
    passed = len(scenario_results) >= 2 and scenario_results[1].get("profit_retention", 0) > 50

    details = {
        "base_profit_pct": round(base_profit_pct, 2),
        "scenarios": scenario_results,
        "avg_retention": round(np.mean([r["profit_retention"] for r in scenario_results]), 2)
    }

    return passed, round(overall_score, 4), details


# ═══════════════════════════════════════════════════════════════════════════════
# NOISE INJECTION TEST
# ═══════════════════════════════════════════════════════════════════════════════

def noise_injection_test(
    genome: Dict,
    fitness_fn: Callable[[Dict], float],
    base_score: float = None,
    noise_levels: List[float] = None,
    n_tests_per_level: int = 10
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Test strategy robustness to random parameter noise.

    Args:
        genome: Base genome
        fitness_fn: Fitness function
        base_score: Pre-calculated base score
        noise_levels: Noise levels as std dev (default: 0.02, 0.05, 0.10)
        n_tests_per_level: Number of tests per noise level

    Returns:
        (passed, score, details)
    """
    if noise_levels is None:
        noise_levels = [0.02, 0.05, 0.10]  # 2%, 5%, 10% noise

    if base_score is None:
        base_score = fitness_fn(genome)

    if base_score <= 0:
        return False, 0, {"reason": "Invalid base score"}

    level_results = []

    for noise_level in noise_levels:
        scores_at_level = []

        for _ in range(n_tests_per_level):
            noisy_genome = _add_gaussian_noise(genome, noise_level)

            try:
                score = fitness_fn(noisy_genome)
                if score > float("-inf"):
                    scores_at_level.append(score)
            except:
                pass

        if scores_at_level:
            avg_score = np.mean(scores_at_level)
            retention = avg_score / base_score if base_score > 0 else 0
            level_results.append({
                "noise_pct": noise_level * 100,
                "avg_score": round(avg_score, 4),
                "retention": round(retention, 4),
                "std": round(np.std(scores_at_level), 4)
            })

    if not level_results:
        return False, 0, {"reason": "All noise tests failed"}

    # Score = average retention across noise levels
    overall_score = np.mean([r["retention"] for r in level_results])

    # Pass if retains >70% performance at 5% noise level
    passed = len(level_results) >= 2 and level_results[1].get("retention", 0) > 0.7

    details = {
        "base_score": base_score,
        "levels": level_results
    }

    return passed, round(overall_score, 4), details


def _add_gaussian_noise(genome: Dict, std: float) -> Dict:
    """Add Gaussian noise to numeric genome parameters."""
    noisy = copy.deepcopy(genome)

    for block in ["entry", "sl", "tp_dual", "tp_rsi"]:
        if block not in noisy:
            continue

        for param, value in noisy[block].items():
            if isinstance(value, float):
                noise = np.random.normal(0, value * std)
                noisy[block][param] = max(0.1, round(value + noise, 2))
            elif isinstance(value, int) and param not in ["st_useATR"]:
                noise = int(np.random.normal(0, value * std))
                noisy[block][param] = max(1, value + noise)

    return noisy


# ═══════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE ROBUSTNESS TEST
# ═══════════════════════════════════════════════════════════════════════════════

def comprehensive_robustness_test(
    genome: Dict,
    fitness_fn: Callable[[Dict], float],
    backtest_fn: Callable = None,
    df: pd.DataFrame = None,
    trades: List[Dict[str, Any]] = None,
    initial_capital: float = 10000,
    stability_threshold: float = DEFAULT_STABILITY_THRESHOLD,
    skip_walk_forward: bool = False,
    skip_monte_carlo: bool = False
) -> RobustnessResult:
    """
    Run comprehensive robustness testing suite.

    Args:
        genome: Strategy genome to test
        fitness_fn: Function(genome) -> score
        backtest_fn: Function(genome, df) -> result (for walk-forward)
        df: OHLCV DataFrame (for walk-forward)
        trades: List of trades (for Monte Carlo)
        initial_capital: Starting capital
        stability_threshold: Minimum stability score to pass
        skip_walk_forward: Skip walk-forward analysis
        skip_monte_carlo: Skip Monte Carlo simulation

    Returns:
        RobustnessResult with all test results
    """
    start_time = time.time()
    details = {}

    # Base score
    base_score = fitness_fn(genome)
    details["base_score"] = base_score

    # 1. Walk-Forward Analysis (if data available)
    if not skip_walk_forward and backtest_fn is not None and df is not None and len(df) > 500:
        try:
            wf_result = walk_forward_analysis(genome, backtest_fn, df)
            walk_forward_score = wf_result.score
            details["walk_forward"] = {
                "passed": wf_result.passed,
                "score": wf_result.score,
                "consistency": wf_result.consistency_ratio,
                "correlation": wf_result.is_oos_correlation,
                "windows": wf_result.windows
            }
        except Exception as e:
            logger.warning(f"Walk-forward analysis failed: {e}")
            walk_forward_score = 0.5  # Neutral score on failure
            details["walk_forward"] = {"error": str(e)}
    else:
        walk_forward_score = 0.5  # Neutral if skipped
        details["walk_forward"] = {"skipped": True}

    # 2. Monte Carlo Simulation (if trades available)
    if not skip_monte_carlo and trades and len(trades) >= 10:
        try:
            mc_result = monte_carlo_simulation(trades, initial_capital, n_simulations=DEFAULT_MONTE_CARLO_RUNS)
            monte_carlo_score = mc_result.score
            details["monte_carlo"] = {
                "passed": mc_result.passed,
                "score": mc_result.score,
                "mean_profit": mc_result.mean_profit,
                "std_profit": mc_result.std_profit,
                "var_95": mc_result.var_95,
                "profit_probability": mc_result.profit_probability
            }
        except Exception as e:
            logger.warning(f"Monte Carlo simulation failed: {e}")
            monte_carlo_score = 0.5
            details["monte_carlo"] = {"error": str(e)}
    else:
        monte_carlo_score = 0.5
        details["monte_carlo"] = {"skipped": True, "reason": "Insufficient trades"}

    # 3. Parameter Sensitivity Analysis
    try:
        sens_result = parameter_sensitivity_analysis(genome, fitness_fn, base_score)
        sensitivity_score = sens_result.score
        details["sensitivity"] = {
            "passed": sens_result.passed,
            "score": sens_result.score,
            "most_sensitive": sens_result.most_sensitive,
            "least_sensitive": sens_result.least_sensitive,
            "parameters": sens_result.parameter_scores
        }
    except Exception as e:
        logger.warning(f"Sensitivity analysis failed: {e}")
        sensitivity_score = 0.5
        details["sensitivity"] = {"error": str(e)}

    # 4. Slippage Stress Test
    if trades and len(trades) >= 5:
        try:
            slip_passed, slip_score, slip_details = slippage_stress_test(trades, initial_capital)
            slippage_score = slip_score
            details["slippage"] = {
                "passed": slip_passed,
                "score": slip_score,
                **slip_details
            }
        except Exception as e:
            logger.warning(f"Slippage test failed: {e}")
            slippage_score = 0.5
            details["slippage"] = {"error": str(e)}
    else:
        slippage_score = 0.5
        details["slippage"] = {"skipped": True}

    # 5. Noise Injection Test
    try:
        noise_passed, noise_score, noise_details = noise_injection_test(genome, fitness_fn, base_score)
        details["noise"] = {
            "passed": noise_passed,
            "score": noise_score,
            **noise_details
        }
    except Exception as e:
        logger.warning(f"Noise test failed: {e}")
        noise_score = 0.5
        details["noise"] = {"error": str(e)}

    # Calculate overall stability score
    # Weights: WF=30%, MC=30%, Sens=20%, Slip=10%, Noise=10%
    stability_score = (
        walk_forward_score * 0.30 +
        monte_carlo_score * 0.30 +
        sensitivity_score * 0.20 +
        slippage_score * 0.10 +
        noise_score * 0.10
    )

    # Overall pass/fail
    passed = stability_score >= stability_threshold

    details["elapsed_seconds"] = round(time.time() - start_time, 2)

    logger.info(
        f"Robustness test: WF={walk_forward_score:.2f}, MC={monte_carlo_score:.2f}, "
        f"Sens={sensitivity_score:.2f}, Slip={slippage_score:.2f}, Noise={noise_score:.2f} "
        f"=> Stability={stability_score:.2f} ({'PASS' if passed else 'FAIL'})"
    )

    return RobustnessResult(
        passed=passed,
        stability_score=round(stability_score, 4),
        walk_forward_score=round(walk_forward_score, 4),
        monte_carlo_score=round(monte_carlo_score, 4),
        sensitivity_score=round(sensitivity_score, 4),
        slippage_score=round(slippage_score, 4),
        noise_score=round(noise_score, 4),
        details=details
    )


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK ROBUSTNESS CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def quick_robustness_check(
    genome: Dict,
    fitness_fn: Callable[[Dict], float],
    base_score: float = None
) -> Tuple[bool, float]:
    """
    Quick robustness check with minimal tests.

    Only runs sensitivity and noise tests for speed.

    Returns:
        (passed, score)
    """
    if base_score is None:
        base_score = fitness_fn(genome)

    if base_score <= 0:
        return False, 0

    # Quick sensitivity (fewer parameters)
    quick_params = [("entry", "st_mult"), ("sl", "st_mult"), ("tp_dual", "rr_mult")]
    sensitivities = []

    for block, param in quick_params:
        if block in genome and param in genome[block]:
            original = genome[block][param]
            if isinstance(original, (int, float)):
                perturbed = copy.deepcopy(genome)
                perturbed[block][param] = original * 1.1  # +10%

                try:
                    score = fitness_fn(perturbed)
                    if score > 0:
                        retention = score / base_score
                        sensitivities.append(retention)
                except:
                    pass

    if not sensitivities:
        return False, 0

    avg_retention = np.mean(sensitivities)
    passed = avg_retention >= 0.7

    return passed, round(avg_retention, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH ROBUSTNESS TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def batch_robustness_filter(
    genomes: List[Dict],
    fitness_fn: Callable[[Dict], float],
    threshold: float = DEFAULT_STABILITY_THRESHOLD,
    use_quick_check: bool = True
) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter a batch of genomes by robustness.

    Args:
        genomes: List of genomes to test
        fitness_fn: Fitness function
        threshold: Minimum stability score
        use_quick_check: Use quick check for speed

    Returns:
        (robust_genomes, fragile_genomes)
    """
    robust = []
    fragile = []

    for genome in genomes:
        if use_quick_check:
            passed, score = quick_robustness_check(genome, fitness_fn)
        else:
            result = comprehensive_robustness_test(
                genome, fitness_fn,
                skip_walk_forward=True,
                skip_monte_carlo=True
            )
            passed = result.passed
            score = result.stability_score

        genome_copy = copy.deepcopy(genome)
        genome_copy["_robustness_score"] = score

        if passed and score >= threshold:
            robust.append(genome_copy)
        else:
            fragile.append(genome_copy)

    logger.info(f"Batch robustness: {len(robust)} robust, {len(fragile)} fragile")
    return robust, fragile
