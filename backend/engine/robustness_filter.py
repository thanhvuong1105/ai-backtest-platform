# engine/robustness_filter.py
"""
Robustness Filter for Quant AI Brain

Tests genome stability under perturbations:
- Entry price noise: ±0.02%
- SL width: ±10%
- RR: ±0.2
- Candle shift: ±1 bar

Rejects genomes whose score collapses under noise.
"""

import os
import copy
import logging
from typing import Dict, Any, List, Tuple, Callable

import numpy as np

logger = logging.getLogger(__name__)

# Configuration
ROBUSTNESS_THRESHOLD = float(os.getenv("ROBUSTNESS_THRESHOLD", 0.5))  # Score must be >= 50% of original
PERTURBATION_RUNS = int(os.getenv("ROBUSTNESS_RUNS", 5))  # Number of perturbation tests


# ═══════════════════════════════════════════════════════
# PERTURBATION FUNCTIONS
# ═══════════════════════════════════════════════════════

def perturb_sl(genome: Dict, factor: float) -> Dict:
    """
    Perturb SL multipliers by factor.

    Args:
        genome: Original genome
        factor: Multiplier (e.g., 0.9 = -10%, 1.1 = +10%)

    Returns:
        Perturbed genome
    """
    perturbed = copy.deepcopy(genome)

    if "sl" in perturbed:
        perturbed["sl"]["st_mult"] = round(perturbed["sl"].get("st_mult", 4.0) * factor, 2)
        perturbed["sl"]["rf_mult"] = round(perturbed["sl"].get("rf_mult", 7.0) * factor, 2)

    return perturbed


def perturb_rr(genome: Dict, delta: float) -> Dict:
    """
    Perturb RR by delta.

    Args:
        genome: Original genome
        delta: Amount to add/subtract (e.g., ±0.2)

    Returns:
        Perturbed genome
    """
    perturbed = copy.deepcopy(genome)

    if "tp_dual" in perturbed:
        perturbed["tp_dual"]["rr_mult"] = max(0.5, round(perturbed["tp_dual"].get("rr_mult", 1.3) + delta, 2))

    if "tp_rsi" in perturbed:
        perturbed["tp_rsi"]["rr_mult"] = max(0.5, round(perturbed["tp_rsi"].get("rr_mult", 1.3) + delta, 2))

    return perturbed


def perturb_entry_params(genome: Dict, noise_pct: float) -> Dict:
    """
    Add noise to entry parameters.

    Args:
        genome: Original genome
        noise_pct: Noise percentage (e.g., 0.02 = 2%)

    Returns:
        Perturbed genome
    """
    perturbed = copy.deepcopy(genome)

    if "entry" in perturbed:
        entry = perturbed["entry"]

        # Perturb ST mult
        if "st_mult" in entry:
            noise = np.random.normal(0, entry["st_mult"] * noise_pct)
            entry["st_mult"] = max(0.5, round(entry["st_mult"] + noise, 2))

        # Perturb RF mult
        if "rf_mult" in entry:
            noise = np.random.normal(0, entry["rf_mult"] * noise_pct)
            entry["rf_mult"] = max(1.0, round(entry["rf_mult"] + noise, 2))

        # Perturb RF period (integer)
        if "rf_period" in entry:
            noise = int(np.random.normal(0, entry["rf_period"] * noise_pct))
            entry["rf_period"] = max(30, entry["rf_period"] + noise)

    return perturbed


def perturb_atr_periods(genome: Dict, delta: int) -> Dict:
    """
    Shift ATR periods by delta.

    Args:
        genome: Original genome
        delta: Amount to add/subtract (e.g., ±1)

    Returns:
        Perturbed genome
    """
    perturbed = copy.deepcopy(genome)

    # Entry ATR period
    if "entry" in perturbed and "st_atrPeriod" in perturbed["entry"]:
        perturbed["entry"]["st_atrPeriod"] = max(1, perturbed["entry"]["st_atrPeriod"] + delta)

    # SL ATR period
    if "sl" in perturbed and "st_atrPeriod" in perturbed["sl"]:
        perturbed["sl"]["st_atrPeriod"] = max(1, perturbed["sl"]["st_atrPeriod"] + delta)

    return perturbed


# ═══════════════════════════════════════════════════════
# ROBUSTNESS TESTING
# ═══════════════════════════════════════════════════════

def test_robustness(
    genome: Dict,
    fitness_fn: Callable[[Dict], float],
    base_score: float = None,
    threshold: float = ROBUSTNESS_THRESHOLD
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Test genome robustness under various perturbations.

    Args:
        genome: Genome to test
        fitness_fn: Function to evaluate genome fitness
        base_score: Original score (if None, will be calculated)
        threshold: Minimum ratio of perturbed/base score to pass

    Returns:
        (passed, stability_score, details)
    """
    # Calculate base score if not provided
    if base_score is None:
        base_score = fitness_fn(genome)

    if base_score <= 0:
        return False, 0.0, {"reason": "Invalid base score"}

    perturbation_scores = []
    details = {
        "base_score": base_score,
        "perturbations": [],
    }

    # Test SL perturbations (±10%)
    for factor in [0.9, 1.1]:
        perturbed = perturb_sl(genome, factor)
        score = fitness_fn(perturbed)
        perturbation_scores.append(score)
        details["perturbations"].append({
            "type": f"sl_factor_{factor}",
            "score": score,
            "ratio": score / base_score if base_score > 0 else 0,
        })

    # Test RR perturbations (±0.2)
    for delta in [-0.2, 0.2]:
        perturbed = perturb_rr(genome, delta)
        score = fitness_fn(perturbed)
        perturbation_scores.append(score)
        details["perturbations"].append({
            "type": f"rr_delta_{delta}",
            "score": score,
            "ratio": score / base_score if base_score > 0 else 0,
        })

    # Test entry noise (±5%)
    for _ in range(2):
        perturbed = perturb_entry_params(genome, 0.05)
        score = fitness_fn(perturbed)
        perturbation_scores.append(score)
        details["perturbations"].append({
            "type": "entry_noise_5pct",
            "score": score,
            "ratio": score / base_score if base_score > 0 else 0,
        })

    # Test ATR period shift (±1)
    for delta in [-1, 1]:
        perturbed = perturb_atr_periods(genome, delta)
        score = fitness_fn(perturbed)
        perturbation_scores.append(score)
        details["perturbations"].append({
            "type": f"atr_shift_{delta}",
            "score": score,
            "ratio": score / base_score if base_score > 0 else 0,
        })

    # Calculate stability score
    valid_scores = [s for s in perturbation_scores if s > 0]

    if not valid_scores:
        return False, 0.0, {**details, "reason": "All perturbations failed"}

    # Stability = min score ratio (worst case performance)
    min_ratio = min(s / base_score for s in valid_scores)
    avg_ratio = sum(s / base_score for s in valid_scores) / len(valid_scores)

    # Combined stability score (weight worst case more)
    stability_score = 0.7 * min_ratio + 0.3 * avg_ratio

    details["min_ratio"] = round(min_ratio, 4)
    details["avg_ratio"] = round(avg_ratio, 4)
    details["stability_score"] = round(stability_score, 4)

    # Pass if worst case is above threshold
    passed = min_ratio >= threshold

    return passed, stability_score, details


def batch_robustness_test(
    genomes: List[Dict],
    fitness_fn: Callable[[Dict], float],
    threshold: float = ROBUSTNESS_THRESHOLD
) -> Tuple[List[Dict], List[Dict]]:
    """
    Test robustness for a batch of genomes.

    Args:
        genomes: List of genomes to test
        fitness_fn: Fitness function
        threshold: Robustness threshold

    Returns:
        (robust_genomes, fragile_genomes)
    """
    robust = []
    fragile = []

    for genome in genomes:
        passed, stability, details = test_robustness(
            genome, fitness_fn, threshold=threshold
        )

        genome_with_robustness = copy.deepcopy(genome)
        genome_with_robustness["_robustness"] = {
            "passed": passed,
            "stability_score": stability,
            "details": details,
        }

        if passed:
            robust.append(genome_with_robustness)
        else:
            fragile.append(genome_with_robustness)

    logger.info(f"Robustness test: {len(robust)} robust, {len(fragile)} fragile")
    return robust, fragile


# ═══════════════════════════════════════════════════════
# MONTE CARLO ROBUSTNESS
# ═══════════════════════════════════════════════════════

def monte_carlo_robustness(
    genome: Dict,
    fitness_fn: Callable[[Dict], float],
    n_simulations: int = 20,
    noise_level: float = 0.05
) -> Tuple[float, float, List[float]]:
    """
    Monte Carlo simulation for robustness estimation.

    Runs multiple simulations with random perturbations.

    Args:
        genome: Genome to test
        fitness_fn: Fitness function
        n_simulations: Number of Monte Carlo runs
        noise_level: Standard deviation of perturbation noise

    Returns:
        (mean_score, std_score, all_scores)
    """
    scores = []

    for _ in range(n_simulations):
        # Apply random perturbations
        perturbed = copy.deepcopy(genome)

        # Random SL factor
        sl_factor = np.random.normal(1.0, noise_level)
        perturbed = perturb_sl(perturbed, sl_factor)

        # Random RR delta
        rr_delta = np.random.normal(0, noise_level * 2)
        perturbed = perturb_rr(perturbed, rr_delta)

        # Random entry noise
        perturbed = perturb_entry_params(perturbed, noise_level)

        # Evaluate
        score = fitness_fn(perturbed)
        if score > float("-inf"):
            scores.append(score)

    if not scores:
        return 0.0, 0.0, []

    mean_score = np.mean(scores)
    std_score = np.std(scores)

    return mean_score, std_score, scores


def calculate_sharpe_ratio(scores: List[float], risk_free: float = 0) -> float:
    """
    Calculate Sharpe ratio of score distribution.

    Higher Sharpe = more consistent performance.

    Args:
        scores: List of scores from Monte Carlo
        risk_free: Risk-free rate (default 0)

    Returns:
        Sharpe ratio
    """
    if not scores or len(scores) < 2:
        return 0.0

    mean = np.mean(scores)
    std = np.std(scores)

    if std == 0:
        return float("inf") if mean > 0 else float("-inf")

    return (mean - risk_free) / std


# ═══════════════════════════════════════════════════════
# QUICK ROBUSTNESS CHECK
# ═══════════════════════════════════════════════════════

def quick_robustness_check(
    genome: Dict,
    fitness_fn: Callable[[Dict], float],
    base_score: float = None
) -> bool:
    """
    Quick robustness check (fewer perturbations).

    Args:
        genome: Genome to check
        fitness_fn: Fitness function
        base_score: Optional pre-calculated base score

    Returns:
        True if genome is robust
    """
    if base_score is None:
        base_score = fitness_fn(genome)

    if base_score <= 0:
        return False

    # Quick test: just SL ±10%
    for factor in [0.9, 1.1]:
        perturbed = perturb_sl(genome, factor)
        score = fitness_fn(perturbed)

        if score < base_score * ROBUSTNESS_THRESHOLD:
            return False

    return True
