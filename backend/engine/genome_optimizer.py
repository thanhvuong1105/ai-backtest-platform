# engine/genome_optimizer.py
"""
Evolutionary Genome Optimizer for Quant AI Brain

Full evolutionary optimization with:
- Crossover at genome block level
- Gaussian mutation within bounds
- Tournament selection
- Phased optimization (Entry → SL → TP → Mode)
- Parallel fitness evaluation (hardware-aware)
"""

import os
import random
import copy
import logging
from typing import Dict, Any, List, Tuple, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import numpy as np

from .coherence_validator import (
    validate_genome, repair_genome, clamp_to_bounds, PARAM_BOUNDS,
    validate_genome_combined, repair_genome_combined, clamp_to_bounds_combined,
    PARAM_BOUNDS_COMBINED, is_combined_strategy
)
from .regime_classifier import MarketRegime, sample_params_for_regime

logger = logging.getLogger(__name__)

# =============================================================================
# SERVER SPECIFICATIONS (8 vCPU, 32GB RAM, 20+ Workers)
# =============================================================================
VCPU_COUNT = int(os.getenv("CPU_COUNT", 8))
MIN_WORKERS = int(os.getenv("MIN_WORKERS", 20))

# =============================================================================
# GENETIC ALGORITHM CONFIGURATION
# =============================================================================
POPULATION_SIZE = int(os.getenv("GENOME_POPULATION_SIZE", 100))  # Increased for better exploration
GENERATIONS = int(os.getenv("GENOME_GENERATIONS", 10))  # Increased for better convergence
MUTATION_RATE = float(os.getenv("GENOME_MUTATION_RATE", 0.15))
CROSSOVER_RATE = float(os.getenv("GENOME_CROSSOVER_RATE", 0.8))  # Increased crossover
TOURNAMENT_SIZE = int(os.getenv("GENOME_TOURNAMENT_SIZE", 5))
ELITE_COUNT = int(os.getenv("GENOME_ELITE_COUNT", 5))

# =============================================================================
# PARALLEL PROCESSING CONFIGURATION
# Optimized for 20+ concurrent evaluations
# =============================================================================
PARALLEL_FITNESS = os.getenv("PARALLEL_FITNESS", "true").lower() == "true"
_env_workers = os.getenv("MAX_THREAD_WORKERS", "")
if _env_workers:
    MAX_FITNESS_WORKERS = int(_env_workers)
else:
    # Use minimum 20 workers for I/O-bound fitness evaluations
    # Not limited by CPU cores since fitness evaluation is mostly I/O
    MAX_FITNESS_WORKERS = max(MIN_WORKERS, min(32, (os.cpu_count() or 8) * 3))


# ═══════════════════════════════════════════════════════
# GENOME OPERATIONS
# ═══════════════════════════════════════════════════════

def crossover(parent1: Dict, parent2: Dict) -> Dict:
    """
    Single-point crossover at genome block level.

    Randomly swaps one genome block between parents.
    """
    child = copy.deepcopy(parent1)

    # Blocks that can be swapped
    blocks = ["entry", "sl", "tp_dual", "tp_rsi", "mode"]

    # Pick random crossover point
    crossover_point = random.randint(1, len(blocks) - 1)

    # Swap blocks after crossover point
    for i in range(crossover_point, len(blocks)):
        block = blocks[i]
        child[block] = copy.deepcopy(parent2[block])

    return child


def mutate(genome: Dict, rate: float = MUTATION_RATE, bounds: Dict = None) -> Dict:
    """
    Gaussian mutation within parameter bounds.

    Each numeric parameter has `rate` chance of being mutated.
    Mutation adds Gaussian noise scaled by parameter range.

    Args:
        genome: Genome to mutate
        rate: Mutation probability per parameter
        bounds: Optional custom bounds (auto-expanded). If None, uses PARAM_BOUNDS

    Returns:
        Mutated genome
    """
    mutated = copy.deepcopy(genome)
    effective_bounds = bounds if bounds else PARAM_BOUNDS

    for block, block_bounds in effective_bounds.items():
        if block not in mutated:
            continue

        for param, (min_val, max_val) in block_bounds.items():
            if param not in mutated[block]:
                continue

            if random.random() < rate:
                current = mutated[block][param]
                param_range = max_val - min_val

                # Gaussian noise with std = 10% of range
                noise = np.random.normal(0, param_range * 0.1)
                new_val = current + noise

                # Clamp to bounds
                new_val = max(min_val, min(max_val, new_val))

                # Round appropriately
                if isinstance(current, int):
                    new_val = int(round(new_val))
                else:
                    new_val = round(new_val, 2)

                mutated[block][param] = new_val

    return mutated


def clamp_genome_to_bounds(genome: Dict, bounds: Dict = None) -> Dict:
    """
    Clamp genome parameters to specified bounds.

    Args:
        genome: Genome to clamp
        bounds: Custom bounds. If None, uses PARAM_BOUNDS

    Returns:
        Clamped genome
    """
    clamped = copy.deepcopy(genome)
    effective_bounds = bounds if bounds else PARAM_BOUNDS

    for block, block_bounds in effective_bounds.items():
        if block not in clamped:
            continue

        for param, (min_val, max_val) in block_bounds.items():
            if param not in clamped[block]:
                continue

            val = clamped[block][param]
            if isinstance(val, (int, float)):
                clamped[block][param] = max(min_val, min(max_val, val))

    return clamped


def tournament_select(
    population: List[Dict],
    scores: List[float],
    n: int,
    tournament_size: int = TOURNAMENT_SIZE
) -> List[Dict]:
    """
    Tournament selection.

    Randomly selects `tournament_size` individuals and picks the best.
    Repeats `n` times.
    """
    selected = []

    for _ in range(n):
        # Random tournament
        indices = random.sample(range(len(population)), min(tournament_size, len(population)))

        # Find best in tournament
        best_idx = max(indices, key=lambda i: scores[i])
        selected.append(copy.deepcopy(population[best_idx]))

    return selected


def create_random_genome(ui_ranges: Dict = None) -> Dict:
    """
    Create a random genome within bounds.

    Args:
        ui_ranges: Optional UI ranges from frontend with format:
            {
                "st_atr_period": {"start": 8, "end": 16},
                "st_mult": {"start": 1.5, "end": 3.0},
                ...
            }

    Returns:
        Random genome within specified or default bounds
    """
    # If ui_ranges provided, use them; otherwise use defaults
    if ui_ranges:
        logger.info("Population initialized from UI ranges")
        genome = _create_genome_from_ui_ranges(ui_ranges)
    else:
        # Default ranges (fallback)
        genome = {
            "entry": {
                "st_atrPeriod": random.randint(8, 16),
                "st_src": "hl2",
                "st_mult": round(random.uniform(1.5, 3.0), 2),
                "st_useATR": True,
                "rf_src": "close",
                "rf_period": random.randint(80, 120),
                "rf_mult": round(random.uniform(2.5, 4.0), 2),
                "rsi_length": random.randint(10, 18),
                "rsi_ma_length": random.randint(4, 8),
            },
            "sl": {
                "st_atrPeriod": 10,
                "st_src": "hl2",
                "st_mult": round(random.uniform(3.0, 5.0), 2),
                "st_useATR": True,
                "rf_period": 100,
                "rf_mult": round(random.uniform(5.0, 8.0), 2),
            },
            "tp_dual": {
                "st_atrPeriod": 10,
                "st_mult": 2.0,
                "rr_mult": round(random.uniform(1.0, 2.0), 2),
            },
            "tp_rsi": {
                "st_atrPeriod": 10,
                "st_mult": 2.0,
                "rr_mult": round(random.uniform(1.0, 2.0), 2),
            },
            "mode": {
                "showDualFlip": True,
                "showRSI": True,
            },
        }

    return repair_genome(genome)


def _create_genome_from_ui_ranges(ui_ranges: Dict) -> Dict:
    """
    Create genome from UI ranges (start-end only, no step).

    Mapping from UI param names to genome structure:
    - st_atr_period -> entry.st_atrPeriod
    - st_mult -> entry.st_mult
    - rf_period -> entry.rf_period
    - rf_mult -> entry.rf_mult
    - rsi_length -> entry.rsi_length
    - rsi_ma_length -> entry.rsi_ma_length
    - tp_rr_mult -> tp_dual.rr_mult, tp_rsi.rr_mult
    - sl_mult -> sl.st_mult, sl.rf_mult
    """
    def sample_value(param_name: str, default_min: float, default_max: float, is_int: bool = False):
        """Sample value from UI range or use default."""
        if param_name in ui_ranges:
            range_def = ui_ranges[param_name]
            start = range_def.get("start", default_min)
            end = range_def.get("end", default_max)
        else:
            start, end = default_min, default_max

        if is_int:
            return random.randint(int(start), int(end))
        else:
            return round(random.uniform(start, end), 2)

    genome = {
        "entry": {
            "st_atrPeriod": sample_value("st_atr_period", 8, 16, is_int=True),
            "st_src": "hl2",
            "st_mult": sample_value("st_mult", 1.5, 3.0),
            "st_useATR": True,
            "rf_src": "close",
            "rf_period": sample_value("rf_period", 80, 120, is_int=True),
            "rf_mult": sample_value("rf_mult", 2.5, 4.0),
            "rsi_length": sample_value("rsi_length", 10, 18, is_int=True),
            "rsi_ma_length": sample_value("rsi_ma_length", 4, 8, is_int=True),
        },
        "sl": {
            "st_atrPeriod": 10,
            "st_src": "hl2",
            "st_mult": sample_value("sl_mult", 3.0, 5.0),
            "st_useATR": True,
            "rf_period": 100,
            "rf_mult": sample_value("sl_rf_mult", 5.0, 8.0),
        },
        "tp_dual": {
            "st_atrPeriod": 10,
            "st_mult": 2.0,
            "rr_mult": sample_value("tp_rr_mult", 0.1, 5.0),
        },
        "tp_rsi": {
            "st_atrPeriod": 10,
            "st_mult": 2.0,
            "rr_mult": sample_value("tp_rr_mult", 0.1, 5.0),
        },
        "mode": {
            "showDualFlip": True,
            "showRSI": True,
        },
    }

    return genome


def validate_ui_ranges(ui_ranges: Dict) -> Tuple[bool, str]:
    """
    Validate UI ranges before population initialization.

    Args:
        ui_ranges: UI ranges dict from frontend

    Returns:
        (is_valid, error_message)
    """
    if not ui_ranges:
        return False, "Missing UI ranges - cannot initialize population"

    required_params = ["st_atr_period", "st_mult", "rf_period", "rf_mult", "rsi_length"]
    missing = []

    for param in required_params:
        if param not in ui_ranges:
            missing.append(param)
        else:
            range_def = ui_ranges[param]
            if "start" not in range_def or "end" not in range_def:
                missing.append(f"{param} (missing start/end)")

    if missing:
        return False, f"Missing UI ranges for: {', '.join(missing)}"

    return True, ""


def genome_distance(genome1: Dict, genome2: Dict) -> float:
    """
    Calculate normalized distance between two genomes.

    Compares all numeric parameters and returns a distance score (0-1).
    Distance of 0 = identical, 1.0 = completely different.

    Uses weighted Euclidean distance normalized by parameter ranges.
    """
    if not genome1 or not genome2:
        return 1.0

    total_distance = 0
    total_weight = 0

    blocks = ["entry", "sl", "tp_dual", "tp_rsi"]

    for block in blocks:
        block1 = genome1.get(block, {})
        block2 = genome2.get(block, {})

        for param in block1:
            if param not in block2:
                continue

            v1 = block1[param]
            v2 = block2[param]

            # Skip non-numeric parameters
            if not isinstance(v1, (int, float)) or not isinstance(v2, (int, float)):
                continue

            # Get parameter range from bounds
            param_bounds = PARAM_BOUNDS.get(block, {})
            if param not in param_bounds:
                continue

            min_val, max_val = param_bounds[param]
            param_range = max_val - min_val
            if param_range == 0:
                param_range = 1

            # Normalized distance for this parameter
            distance = abs(v1 - v2) / param_range
            total_distance += distance
            total_weight += 1

    if total_weight == 0:
        return 0.0

    # Average normalized distance
    avg_distance = total_distance / total_weight
    # Clamp to [0, 1]
    return min(1.0, max(0.0, avg_distance))


def filter_diverse_genomes(
    genomes: List[Dict],
    scores: List[float],
    min_distance: float = 0.15
) -> List[Dict]:
    """
    Filter genomes to ensure minimum distance between selected genomes.

    Greedily selects highest-scoring genomes while ensuring each new
    genome is at least min_distance away from all previously selected.

    Args:
        genomes: List of genomes
        scores: Corresponding fitness scores
        min_distance: Minimum distance between genomes (0-1)

    Returns:
        Filtered list of diverse genomes (in descending score order)
    """
    if not genomes:
        return []

    # Sort by score (descending)
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    selected = []
    selected_indices = []

    for idx in sorted_indices:
        genome = genomes[idx]

        # Check distance to all selected genomes
        is_diverse = True
        for selected_idx in selected_indices:
            distance = genome_distance(genome, genomes[selected_idx])
            if distance < min_distance:
                is_diverse = False
                logger.debug(
                    f"Genome {idx} too close to {selected_idx} "
                    f"(distance={distance:.3f}, threshold={min_distance})"
                )
                break

        if is_diverse:
            selected.append(genome)
            selected_indices.append(idx)

    logger.info(
        f"Diversity filter: {len(selected)}/{len(genomes)} genomes selected "
        f"(min_distance={min_distance})"
    )

    return selected


# ═══════════════════════════════════════════════════════
# EVOLUTIONARY OPTIMIZER
# ═══════════════════════════════════════════════════════

class GenomeOptimizer:
    """
    Full evolutionary genome optimizer.

    Supports:
    - Population seeding from memory
    - Regime-aware initialization
    - Phased optimization
    - Parallel fitness evaluation
    - Dynamic bounds (auto-expanded from memory)
    - Cooperative cancellation via cancel_check_fn
    """

    def __init__(
        self,
        fitness_fn: Callable[[Dict], float],
        population_size: int = POPULATION_SIZE,
        generations: int = GENERATIONS,
        mutation_rate: float = MUTATION_RATE,
        crossover_rate: float = CROSSOVER_RATE,
        elite_count: int = ELITE_COUNT,
        max_workers: int = 8,
        param_bounds: Dict = None,
        cancel_check_fn: Callable[[], bool] = None
    ):
        """
        Initialize optimizer.

        Args:
            fitness_fn: Function that takes genome and returns fitness score
            population_size: Population size
            generations: Number of generations
            mutation_rate: Probability of mutation per parameter
            crossover_rate: Probability of crossover per pair
            elite_count: Number of elites to preserve
            max_workers: Max parallel workers for fitness evaluation
            param_bounds: Optional custom bounds (auto-expanded from memory)
            cancel_check_fn: Optional function that returns True if cancelled
        """
        self.fitness_fn = fitness_fn
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_count = elite_count
        self.max_workers = max_workers
        self.param_bounds = param_bounds  # None means use PARAM_BOUNDS
        self.cancel_check_fn = cancel_check_fn

        # Tracking
        self.generation_history = []
        self.best_genome = None
        self.best_score = float("-inf")

    def _check_cancelled(self) -> bool:
        """Check if optimization should be cancelled."""
        if self.cancel_check_fn:
            return self.cancel_check_fn()
        return False

    def initialize_population(
        self,
        seed_genomes: List[Dict] = None,
        regime: MarketRegime = None
    ) -> List[Dict]:
        """
        Initialize population with seeds + regime-aware randoms.

        Args:
            seed_genomes: Genomes from memory to include
            regime: Market regime for biased random generation

        Returns:
            Initial population
        """
        population = []

        # Add seed genomes (from memory)
        if seed_genomes:
            for genome in seed_genomes[:self.population_size // 2]:
                valid, _ = validate_genome(genome)
                if valid:
                    population.append(copy.deepcopy(genome))

        # Add regime-aware randoms
        remaining = self.population_size - len(population)
        if regime and remaining > 0:
            regime_samples = sample_params_for_regime(regime, remaining // 2)
            for genome in regime_samples:
                repaired = repair_genome(genome)
                valid, _ = validate_genome(repaired)
                if valid:
                    population.append(repaired)

        # Fill rest with pure randoms
        while len(population) < self.population_size:
            genome = create_random_genome()
            population.append(genome)

        return population

    def evaluate_population(
        self,
        population: List[Dict],
        progress_cb: Callable = None
    ) -> List[float]:
        """
        Evaluate fitness of all genomes in parallel (hardware-aware).

        Uses ThreadPoolExecutor for parallel evaluation when PARALLEL_FITNESS=true.
        Falls back to sequential evaluation if parallel is disabled.

        Args:
            population: List of genomes
            progress_cb: Optional progress callback

        Returns:
            List of fitness scores
        """
        pop_size = len(population)

        # Use parallel evaluation if enabled and population is large enough
        if PARALLEL_FITNESS and pop_size >= 4:
            return self._evaluate_parallel(population, progress_cb)
        else:
            return self._evaluate_sequential(population, progress_cb)

    def _evaluate_parallel(
        self,
        population: List[Dict],
        progress_cb: Callable = None
    ) -> List[float]:
        """
        Parallel fitness evaluation using ThreadPoolExecutor.

        Achieves 4-16x speedup on multi-core systems.
        Supports cooperative cancellation.
        """
        pop_size = len(population)
        workers = min(MAX_FITNESS_WORKERS, pop_size)
        logger.info(f"Evaluating {pop_size} genomes in PARALLEL ({workers} workers)")

        scores = [float("-inf")] * pop_size
        completed = 0

        def evaluate_single(idx_genome):
            idx, genome = idx_genome
            try:
                score = self._safe_fitness(genome)
                if score is None or (isinstance(score, float) and score != score):
                    return idx, float("-inf")
                return idx, float(score)
            except Exception as e:
                logger.warning(f"Fitness evaluation failed for genome {idx}: {e}")
                return idx, float("-inf")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(evaluate_single, (i, g)): i
                for i, g in enumerate(population)
            }

            for future in as_completed(futures):
                # Check cancellation
                if self._check_cancelled():
                    logger.info("Parallel evaluation cancelled")
                    for f in futures:
                        f.cancel()
                    break

                try:
                    idx, score = future.result()
                    scores[idx] = score
                    completed += 1

                    if progress_cb and completed % 10 == 0:
                        progress_cb(completed, pop_size)
                except Exception as e:
                    logger.warning(f"Future failed: {e}")

        valid_count = sum(1 for s in scores if s > float("-inf"))
        logger.info(f"Parallel evaluation complete: {valid_count}/{pop_size} valid scores")
        return scores

    def _evaluate_sequential(
        self,
        population: List[Dict],
        progress_cb: Callable = None
    ) -> List[float]:
        """
        Sequential fitness evaluation (fallback for small populations).
        Supports cooperative cancellation.
        """
        scores = []
        pop_size = len(population)
        logger.info(f"Evaluating {pop_size} genomes sequentially")

        for i, genome in enumerate(population):
            # Check cancellation
            if self._check_cancelled():
                logger.info(f"Sequential evaluation cancelled at {i}/{pop_size}")
                # Fill remaining with -inf
                scores.extend([float("-inf")] * (pop_size - i))
                break

            try:
                score = self._safe_fitness(genome)
                if score is None or (isinstance(score, float) and score != score):
                    score = float("-inf")
                scores.append(float(score))
            except Exception as e:
                logger.warning(f"Fitness evaluation failed for genome {i}: {e}")
                scores.append(float("-inf"))

            if progress_cb and (i + 1) % 10 == 0:
                progress_cb(i + 1, pop_size)

        valid_count = sum(1 for s in scores if s > float("-inf"))
        logger.info(f"Evaluation complete: {valid_count}/{pop_size} valid scores")
        return scores

    def _safe_fitness(self, genome: Dict) -> float:
        """Safely evaluate fitness, returning -inf on error."""
        try:
            # Skip validation in _safe_fitness - already validated upstream
            # This avoids double validation which was causing all genomes to fail
            score = self.fitness_fn(genome)
            logger.info(f"Fitness returned: {score} (type: {type(score).__name__})")
            return score
        except Exception as e:
            import traceback
            logger.warning(f"Fitness error: {e}\n{traceback.format_exc()}")
            return float("-inf")

    def evolve_generation(
        self,
        population: List[Dict],
        scores: List[float]
    ) -> List[Dict]:
        """
        Evolve one generation.

        Steps:
        1. Keep elites
        2. Select parents via tournament
        3. Crossover pairs
        4. Mutate offspring (using dynamic bounds)
        5. Validate/repair

        Returns:
            Next generation population
        """
        # Sort by score
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        next_gen = []

        # Keep elites
        for i in range(min(self.elite_count, len(sorted_indices))):
            next_gen.append(copy.deepcopy(population[sorted_indices[i]]))

        # Fill rest with offspring
        while len(next_gen) < self.population_size:
            # Tournament selection
            parents = tournament_select(population, scores, 2)

            # Crossover
            if random.random() < self.crossover_rate:
                child = crossover(parents[0], parents[1])
            else:
                child = copy.deepcopy(parents[0])

            # Mutate with dynamic bounds (auto-expanded from memory)
            child = mutate(child, self.mutation_rate, bounds=self.param_bounds)

            # Repair and validate
            child = repair_genome(child)

            # Clamp to dynamic bounds if specified, else default bounds
            if self.param_bounds:
                child = clamp_genome_to_bounds(child, self.param_bounds)
            else:
                child = clamp_to_bounds(child)

            valid, _ = validate_genome(child)
            if valid:
                next_gen.append(child)

        return next_gen[:self.population_size]

    def optimize(
        self,
        seed_genomes: List[Dict] = None,
        regime: MarketRegime = None,
        progress_cb: Callable = None
    ) -> Tuple[Dict, float, List[Dict]]:
        """
        Run full evolutionary optimization.

        Args:
            seed_genomes: Initial seed genomes from memory
            regime: Market regime for biased initialization
            progress_cb: Callback(gen, total_gen, best_score)

        Returns:
            (best_genome, best_score, top_genomes)
        """
        # Check cancellation at start
        if self._check_cancelled():
            logger.info("Optimization cancelled before start")
            return None, 0, []

        # Initialize
        population = self.initialize_population(seed_genomes, regime)

        logger.info(f"Starting evolution: pop={self.population_size}, gen={self.generations}")

        for gen in range(self.generations):
            # Check cancellation before each generation
            if self._check_cancelled():
                logger.info(f"Optimization cancelled at generation {gen}/{self.generations}")
                break

            # Evaluate
            scores = self.evaluate_population(population)

            # Track best
            gen_best_idx = max(range(len(scores)), key=lambda i: scores[i])
            gen_best_score = scores[gen_best_idx]

            if gen_best_score > self.best_score:
                self.best_score = gen_best_score
                self.best_genome = copy.deepcopy(population[gen_best_idx])

            # Record history
            self.generation_history.append({
                "generation": gen,
                "best_score": gen_best_score,
                "avg_score": sum(s for s in scores if s > float("-inf")) / max(1, sum(1 for s in scores if s > float("-inf"))),
                "valid_count": sum(1 for s in scores if s > float("-inf")),
            })

            if progress_cb:
                progress_cb(gen + 1, self.generations, gen_best_score)

            # Evolve (except last generation)
            if gen < self.generations - 1:
                population = self.evolve_generation(population, scores)

        # Check cancellation before final evaluation
        if self._check_cancelled():
            logger.info("Optimization cancelled before final evaluation")
            return self.best_genome, self.best_score, [self.best_genome] if self.best_genome else []

        # Get top genomes from final population
        final_scores = self.evaluate_population(population)

        # Apply diversity filter to ensure top genomes are different from each other
        # This prevents near-duplicate genomes from appearing as Top 1, Top 2, etc.
        top_genomes = filter_diverse_genomes(population, final_scores, min_distance=0.15)

        # If diversity filter removed too many genomes, include some close ones
        # to ensure we have enough results
        if len(top_genomes) < 5:
            sorted_indices = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)
            top_genomes_unfiltered = [population[i] for i in sorted_indices[:20]]
            logger.info(f"Diversity filter resulted in only {len(top_genomes)} genomes, including unfiltered top 20")
            top_genomes = top_genomes_unfiltered

        # Fallback if no best genome found (all scores were -inf)
        if self.best_genome is None and population:
            # Use first genome from population as fallback
            self.best_genome = population[0]
            self.best_score = final_scores[0] if final_scores else 0
            logger.warning("No valid genome found during optimization, using fallback")

        return self.best_genome, self.best_score, top_genomes


# ═══════════════════════════════════════════════════════
# PHASED OPTIMIZATION
# ═══════════════════════════════════════════════════════

class PhasedOptimizer:
    """
    Phased genome optimization.

    Phase 1: Optimize Entry Genome
    Phase 2: Lock Entry → Optimize SL
    Phase 3: Lock Entry + SL → Optimize TP
    Phase 4: Optimize Mode combinations

    Supports dynamic bounds (auto-expanded from memory).
    Supports cooperative cancellation via cancel_check_fn.
    """

    def __init__(
        self,
        fitness_fn: Callable[[Dict], float],
        generations_per_phase: int = 5,
        population_size: int = 30,
        param_bounds: Dict = None,
        cancel_check_fn: Callable[[], bool] = None
    ):
        self.fitness_fn = fitness_fn
        self.generations_per_phase = generations_per_phase
        self.population_size = population_size
        self.param_bounds = param_bounds  # Auto-expanded bounds from memory
        self.cancel_check_fn = cancel_check_fn

    def _check_cancelled(self) -> bool:
        """Check if optimization should be cancelled."""
        if self.cancel_check_fn:
            return self.cancel_check_fn()
        return False

    def optimize(
        self,
        seed_genomes: List[Dict] = None,
        regime: MarketRegime = None,
        progress_cb: Callable = None
    ) -> Tuple[Dict, float, List[Dict]]:
        """
        Run phased optimization.

        Returns:
            (best_genome, best_score, top_genomes)
        """
        # Check cancellation at start
        if self._check_cancelled():
            logger.info("Phased optimization cancelled before start")
            return None, 0, []

        # Start with base genome
        if seed_genomes and len(seed_genomes) > 0:
            base_genome = copy.deepcopy(seed_genomes[0])
        else:
            base_genome = create_random_genome()

        all_tested = []

        # Phase 1: Entry
        if self._check_cancelled():
            logger.info("Phased optimization cancelled before Phase 1")
            return base_genome, 0, [base_genome]

        logger.info("Phase 1: Optimizing Entry genome")
        entry_optimizer = GenomeOptimizer(
            fitness_fn=self.fitness_fn,
            population_size=self.population_size,
            generations=self.generations_per_phase,
            param_bounds=self.param_bounds,
            cancel_check_fn=self.cancel_check_fn  # Pass cancel check
        )
        best_entry, _, top_entry = entry_optimizer.optimize(seed_genomes, regime)
        all_tested.extend(top_entry if top_entry else [])

        # Lock entry (with fallback check)
        if best_entry and "entry" in best_entry:
            base_genome["entry"] = best_entry["entry"]
        else:
            logger.warning("Phase 1 returned no valid entry genome, using default")

        # Phase 2: SL
        if self._check_cancelled():
            logger.info("Phased optimization cancelled before Phase 2")
            return base_genome, 0, all_tested or [base_genome]

        logger.info("Phase 2: Optimizing SL genome")
        sl_seeds = [self._vary_sl(base_genome) for _ in range(10)]
        sl_optimizer = GenomeOptimizer(
            fitness_fn=self.fitness_fn,
            population_size=self.population_size,
            generations=self.generations_per_phase,
            param_bounds=self.param_bounds,
            cancel_check_fn=self.cancel_check_fn  # Pass cancel check
        )
        best_sl, _, top_sl = sl_optimizer.optimize(sl_seeds, regime)
        all_tested.extend(top_sl if top_sl else [])

        # Lock SL (with fallback check)
        if best_sl and "sl" in best_sl:
            base_genome["sl"] = best_sl["sl"]
        else:
            logger.warning("Phase 2 returned no valid SL genome, using default")

        # Phase 3: TP
        if self._check_cancelled():
            logger.info("Phased optimization cancelled before Phase 3")
            return base_genome, 0, all_tested or [base_genome]

        logger.info("Phase 3: Optimizing TP genome")
        tp_seeds = [self._vary_tp(base_genome) for _ in range(10)]
        tp_optimizer = GenomeOptimizer(
            fitness_fn=self.fitness_fn,
            population_size=self.population_size,
            generations=self.generations_per_phase,
            param_bounds=self.param_bounds,
            cancel_check_fn=self.cancel_check_fn  # Pass cancel check
        )
        best_tp, _, top_tp = tp_optimizer.optimize(tp_seeds, regime)
        all_tested.extend(top_tp if top_tp else [])

        # Lock TP (with fallback check)
        if best_tp:
            if "tp_dual" in best_tp:
                base_genome["tp_dual"] = best_tp["tp_dual"]
            if "tp_rsi" in best_tp:
                base_genome["tp_rsi"] = best_tp["tp_rsi"]
        else:
            logger.warning("Phase 3 returned no valid TP genome, using default")

        # Phase 4: Mode
        if self._check_cancelled():
            logger.info("Phased optimization cancelled before Phase 4")
            return base_genome, 0, all_tested or [base_genome]

        logger.info("Phase 4: Optimizing Mode")
        mode_variants = self._generate_mode_variants(base_genome)
        mode_scores = []
        for variant in mode_variants:
            if self._check_cancelled():
                logger.info("Phased optimization cancelled during Phase 4")
                return base_genome, 0, all_tested or [base_genome]
            mode_scores.append(self.fitness_fn(variant))

        best_mode_idx = max(range(len(mode_scores)), key=lambda i: mode_scores[i])
        base_genome["mode"] = mode_variants[best_mode_idx]["mode"]

        # Final score
        if self._check_cancelled():
            logger.info("Phased optimization cancelled before final score")
            return base_genome, 0, all_tested or [base_genome]

        final_score = self.fitness_fn(base_genome)

        # Sort all tested by score
        all_tested.append(base_genome)
        all_tested = list({id(g): g for g in all_tested}.values())  # Dedupe

        return base_genome, final_score, all_tested[:20]

    def _vary_sl(self, genome: Dict) -> Dict:
        """Create variant with different SL params."""
        variant = copy.deepcopy(genome)
        variant["sl"]["st_mult"] = round(random.uniform(2.5, 6.0), 2)
        variant["sl"]["rf_mult"] = round(random.uniform(4.0, 10.0), 2)
        return repair_genome(variant)

    def _vary_tp(self, genome: Dict) -> Dict:
        """Create variant with different TP params."""
        variant = copy.deepcopy(genome)
        variant["tp_dual"]["rr_mult"] = round(random.uniform(0.8, 2.5), 2)
        variant["tp_rsi"]["rr_mult"] = round(random.uniform(0.8, 2.5), 2)
        return repair_genome(variant)

    def _generate_mode_variants(self, genome: Dict) -> List[Dict]:
        """Generate all mode combinations."""
        variants = []
        for dual in [True, False]:
            for rsi in [True, False]:
                if dual or rsi:  # At least one must be true
                    variant = copy.deepcopy(genome)
                    variant["mode"]["showDualFlip"] = dual
                    variant["mode"]["showRSI"] = rsi
                    variants.append(variant)
        return variants


# ═══════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════

def quick_optimize(
    fitness_fn: Callable[[Dict], float],
    seed_genomes: List[Dict] = None,
    regime: MarketRegime = None,
    generations: int = 5,
    population_size: int = 30
) -> Tuple[Dict, float]:
    """
    Quick optimization with reduced parameters.

    Returns:
        (best_genome, best_score)
    """
    optimizer = GenomeOptimizer(
        fitness_fn=fitness_fn,
        population_size=population_size,
        generations=generations,
        elite_count=3
    )
    best_genome, best_score, _ = optimizer.optimize(seed_genomes, regime)
    return best_genome, best_score


# ═══════════════════════════════════════════════════════
# RF + ST + RSI COMBINED STRATEGY GENOME OPERATIONS
# ═══════════════════════════════════════════════════════

def create_random_genome_combined(ui_ranges: Dict = None) -> Dict:
    """
    Create a random genome for rf_st_rsi_combined strategy.

    This strategy has separate params for Long and Short positions.
    Uses PARAM_BOUNDS_COMBINED for default ranges.

    Args:
        ui_ranges: Optional UI ranges from frontend

    Returns:
        Random genome within specified or default bounds
    """
    if ui_ranges:
        logger.info("Combined population initialized from UI ranges")
        genome = _create_genome_combined_from_ui_ranges(ui_ranges)
    else:
        # Use PARAM_BOUNDS_COMBINED for default ranges
        bounds = PARAM_BOUNDS_COMBINED
        genome = {
            "entry": {
                "st_atrPeriod": random.randint(*[int(x) for x in bounds["entry"]["st_atrPeriod"]]),
                "st_src": "hl2",
                "st_mult": round(random.uniform(*bounds["entry"]["st_mult"]), 2),
                "st_useATR": True,
                "rf_src": "close",
                "rf_period": random.randint(*[int(x) for x in bounds["entry"]["rf_period"]]),
                "rf_mult": round(random.uniform(*bounds["entry"]["rf_mult"]), 2),
                "rsi_length": random.randint(*[int(x) for x in bounds["entry"]["rsi_length"]]),
                "rsi_ma_length": random.randint(*[int(x) for x in bounds["entry"]["rsi_ma_length"]]),
                "dualFlipBarsLong": random.randint(*[int(x) for x in bounds["entry"]["dualFlipBarsLong"]]),
                "dualFlipBarsShort": random.randint(*[int(x) for x in bounds["entry"]["dualFlipBarsShort"]]),
            },
            "sl_long": {
                "st_atrPeriod": random.randint(*[int(x) for x in bounds["sl_long"]["st_atrPeriod"]]),
                "st_src": "hl2",
                "st_mult": round(random.uniform(*bounds["sl_long"]["st_mult"]), 2),
                "st_useATR": True,
                "rf_period": random.randint(*[int(x) for x in bounds["sl_long"]["rf_period"]]),
                "rf_mult": round(random.uniform(*bounds["sl_long"]["rf_mult"]), 2),
            },
            "sl_short": {
                "st_atrPeriod": random.randint(*[int(x) for x in bounds["sl_short"]["st_atrPeriod"]]),
                "st_src": "hl2",
                "st_mult": round(random.uniform(*bounds["sl_short"]["st_mult"]), 2),
                "st_useATR": True,
                "rf_period": random.randint(*[int(x) for x in bounds["sl_short"]["rf_period"]]),
                "rf_mult": round(random.uniform(*bounds["sl_short"]["rf_mult"]), 2),
            },
            "tp_dual_long": {
                "st_atrPeriod": random.randint(*[int(x) for x in bounds["tp_dual_long"]["st_atrPeriod"]]),
                "st_mult": round(random.uniform(*bounds["tp_dual_long"]["st_mult"]), 2),
                "rr_mult": round(random.uniform(*bounds["tp_dual_long"]["rr_mult"]), 2),
            },
            "tp_dual_short": {
                "st_atrPeriod": random.randint(*[int(x) for x in bounds["tp_dual_short"]["st_atrPeriod"]]),
                "st_mult": round(random.uniform(*bounds["tp_dual_short"]["st_mult"]), 2),
                "rr_mult": round(random.uniform(*bounds["tp_dual_short"]["rr_mult"]), 2),
            },
            "tp_rsi_long": {
                "st_atrPeriod": random.randint(*[int(x) for x in bounds["tp_rsi_long"]["st_atrPeriod"]]),
                "st_mult": round(random.uniform(*bounds["tp_rsi_long"]["st_mult"]), 2),
                "rr_mult": round(random.uniform(*bounds["tp_rsi_long"]["rr_mult"]), 2),
            },
            "tp_rsi_short": {
                "st_atrPeriod": random.randint(*[int(x) for x in bounds["tp_rsi_short"]["st_atrPeriod"]]),
                "st_mult": round(random.uniform(*bounds["tp_rsi_short"]["st_mult"]), 2),
                "rr_mult": round(random.uniform(*bounds["tp_rsi_short"]["rr_mult"]), 2),
            },
            "mode": {
                "enableLong": True,
                "enableShort": True,
                "showDualFlip": True,
                "showRSI": True,
            },
        }

    return repair_genome_combined(genome)


def create_random_genome_combined_with_bounds(bounds: Dict) -> Dict:
    """
    Create a random genome for rf_st_rsi_combined using specified bounds.

    Args:
        bounds: Parameter bounds dict with blocks (entry, sl_long, sl_short, etc.)

    Returns:
        Random genome within specified bounds
    """
    def get_bound(block: str, param: str, default: tuple) -> tuple:
        """Get bound from dict or return default."""
        if block in bounds and param in bounds[block]:
            return bounds[block][param]
        return default

    genome = {
        "entry": {
            "st_atrPeriod": random.randint(*[int(x) for x in get_bound("entry", "st_atrPeriod", (1, 50))]),
            "st_src": "hl2",
            "st_mult": round(random.uniform(*get_bound("entry", "st_mult", (0.5, 30.0))), 2),
            "st_useATR": True,
            "rf_src": "close",
            "rf_period": random.randint(*[int(x) for x in get_bound("entry", "rf_period", (1, 200))]),
            "rf_mult": round(random.uniform(*get_bound("entry", "rf_mult", (1.0, 30.0))), 2),
            "rsi_length": random.randint(*[int(x) for x in get_bound("entry", "rsi_length", (5, 30))]),
            "rsi_ma_length": random.randint(*[int(x) for x in get_bound("entry", "rsi_ma_length", (2, 20))]),
            "dualFlipBarsLong": random.randint(*[int(x) for x in get_bound("entry", "dualFlipBarsLong", (1, 30))]),
            "dualFlipBarsShort": random.randint(*[int(x) for x in get_bound("entry", "dualFlipBarsShort", (1, 30))]),
        },
        "sl_long": {
            "st_atrPeriod": random.randint(*[int(x) for x in get_bound("sl_long", "st_atrPeriod", (1, 30))]),
            "st_src": "hl2",
            "st_mult": round(random.uniform(*get_bound("sl_long", "st_mult", (1.0, 30.0))), 2),
            "st_useATR": True,
            "rf_period": random.randint(*[int(x) for x in get_bound("sl_long", "rf_period", (1, 200))]),
            "rf_mult": round(random.uniform(*get_bound("sl_long", "rf_mult", (1.0, 30.0))), 2),
        },
        "sl_short": {
            "st_atrPeriod": random.randint(*[int(x) for x in get_bound("sl_short", "st_atrPeriod", (1, 30))]),
            "st_src": "hl2",
            "st_mult": round(random.uniform(*get_bound("sl_short", "st_mult", (1.0, 30.0))), 2),
            "st_useATR": True,
            "rf_period": random.randint(*[int(x) for x in get_bound("sl_short", "rf_period", (1, 200))]),
            "rf_mult": round(random.uniform(*get_bound("sl_short", "rf_mult", (1.0, 30.0))), 2),
        },
        "tp_dual_long": {
            "st_atrPeriod": random.randint(*[int(x) for x in get_bound("tp_dual_long", "st_atrPeriod", (1, 30))]),
            "st_mult": round(random.uniform(*get_bound("tp_dual_long", "st_mult", (1.0, 30.0))), 2),
            "rr_mult": round(random.uniform(*get_bound("tp_dual_long", "rr_mult", (0.1, 5.0))), 2),
        },
        "tp_dual_short": {
            "st_atrPeriod": random.randint(*[int(x) for x in get_bound("tp_dual_short", "st_atrPeriod", (1, 30))]),
            "st_mult": round(random.uniform(*get_bound("tp_dual_short", "st_mult", (1.0, 30.0))), 2),
            "rr_mult": round(random.uniform(*get_bound("tp_dual_short", "rr_mult", (0.1, 5.0))), 2),
        },
        "tp_rsi_long": {
            "st_atrPeriod": random.randint(*[int(x) for x in get_bound("tp_rsi_long", "st_atrPeriod", (1, 30))]),
            "st_mult": round(random.uniform(*get_bound("tp_rsi_long", "st_mult", (1.0, 30.0))), 2),
            "rr_mult": round(random.uniform(*get_bound("tp_rsi_long", "rr_mult", (0.1, 5.0))), 2),
        },
        "tp_rsi_short": {
            "st_atrPeriod": random.randint(*[int(x) for x in get_bound("tp_rsi_short", "st_atrPeriod", (1, 30))]),
            "st_mult": round(random.uniform(*get_bound("tp_rsi_short", "st_mult", (1.0, 30.0))), 2),
            "rr_mult": round(random.uniform(*get_bound("tp_rsi_short", "rr_mult", (0.1, 5.0))), 2),
        },
        "mode": {
            "enableLong": True,
            "enableShort": True,
            "showDualFlip": True,
            "showRSI": True,
        },
    }

    return repair_genome_combined(genome)


def _create_genome_combined_from_ui_ranges(ui_ranges: Dict) -> Dict:
    """
    Create combined genome from UI ranges.

    Mapping from UI param names to genome structure for combined strategy.
    Uses PARAM_BOUNDS_COMBINED for default values when UI range not specified.
    """
    bounds = PARAM_BOUNDS_COMBINED

    def sample_value(param_name: str, block: str, param: str, is_int: bool = False):
        """Sample value from UI range or use default from PARAM_BOUNDS_COMBINED."""
        if param_name in ui_ranges:
            range_def = ui_ranges[param_name]
            start = range_def.get("start", bounds[block][param][0])
            end = range_def.get("end", bounds[block][param][1])
        else:
            start, end = bounds[block][param]

        if is_int:
            return random.randint(int(start), int(end))
        else:
            return round(random.uniform(start, end), 2)

    genome = {
        "entry": {
            "st_atrPeriod": sample_value("st_atrPeriod", "entry", "st_atrPeriod", is_int=True),
            "st_src": "hl2",
            "st_mult": sample_value("st_mult", "entry", "st_mult"),
            "st_useATR": True,
            "rf_src": "close",
            "rf_period": sample_value("rf_period", "entry", "rf_period", is_int=True),
            "rf_mult": sample_value("rf_mult", "entry", "rf_mult"),
            "rsi_length": sample_value("rsi_length", "entry", "rsi_length", is_int=True),
            "rsi_ma_length": sample_value("rsi_ma_length", "entry", "rsi_ma_length", is_int=True),
            "dualFlipBarsLong": sample_value("dualFlipBarsLong", "entry", "dualFlipBarsLong", is_int=True),
            "dualFlipBarsShort": sample_value("dualFlipBarsShort", "entry", "dualFlipBarsShort", is_int=True),
        },
        "sl_long": {
            "st_atrPeriod": sample_value("sl_long_st_atrPeriod", "sl_long", "st_atrPeriod", is_int=True),
            "st_src": "hl2",
            "st_mult": sample_value("sl_long_st_mult", "sl_long", "st_mult"),
            "st_useATR": True,
            "rf_period": sample_value("sl_long_rf_period", "sl_long", "rf_period", is_int=True),
            "rf_mult": sample_value("sl_long_rf_mult", "sl_long", "rf_mult"),
        },
        "sl_short": {
            "st_atrPeriod": sample_value("sl_short_st_atrPeriod", "sl_short", "st_atrPeriod", is_int=True),
            "st_src": "hl2",
            "st_mult": sample_value("sl_short_st_mult", "sl_short", "st_mult"),
            "st_useATR": True,
            "rf_period": sample_value("sl_short_rf_period", "sl_short", "rf_period", is_int=True),
            "rf_mult": sample_value("sl_short_rf_mult", "sl_short", "rf_mult"),
        },
        "tp_dual_long": {
            "st_atrPeriod": sample_value("tp_dual_long_st_atrPeriod", "tp_dual_long", "st_atrPeriod", is_int=True),
            "st_mult": sample_value("tp_dual_long_st_mult", "tp_dual_long", "st_mult"),
            "rr_mult": sample_value("tp_dual_long_rr_mult", "tp_dual_long", "rr_mult"),
        },
        "tp_dual_short": {
            "st_atrPeriod": sample_value("tp_dual_short_st_atrPeriod", "tp_dual_short", "st_atrPeriod", is_int=True),
            "st_mult": sample_value("tp_dual_short_st_mult", "tp_dual_short", "st_mult"),
            "rr_mult": sample_value("tp_dual_short_rr_mult", "tp_dual_short", "rr_mult"),
        },
        "tp_rsi_long": {
            "st_atrPeriod": sample_value("tp_rsi_long_st_atrPeriod", "tp_rsi_long", "st_atrPeriod", is_int=True),
            "st_mult": sample_value("tp_rsi_long_st_mult", "tp_rsi_long", "st_mult"),
            "rr_mult": sample_value("tp_rsi_long_rr_mult", "tp_rsi_long", "rr_mult"),
        },
        "tp_rsi_short": {
            "st_atrPeriod": sample_value("tp_rsi_short_st_atrPeriod", "tp_rsi_short", "st_atrPeriod", is_int=True),
            "st_mult": sample_value("tp_rsi_short_st_mult", "tp_rsi_short", "st_mult"),
            "rr_mult": sample_value("tp_rsi_short_rr_mult", "tp_rsi_short", "rr_mult"),
        },
        "mode": {
            "enableLong": True,
            "enableShort": True,
            "showDualFlip": True,
            "showRSI": True,
        },
    }

    return genome


def crossover_combined(parent1: Dict, parent2: Dict) -> Dict:
    """
    Single-point crossover at genome block level for combined strategy.

    Randomly swaps one or more genome blocks between parents.
    """
    child = copy.deepcopy(parent1)

    # Blocks that can be swapped
    blocks = ["entry", "sl_long", "sl_short", "tp_dual_long", "tp_dual_short",
              "tp_rsi_long", "tp_rsi_short", "mode"]

    # Pick random crossover point
    crossover_point = random.randint(1, len(blocks) - 1)

    # Swap blocks after crossover point
    for i in range(crossover_point, len(blocks)):
        block = blocks[i]
        if block in parent2:
            child[block] = copy.deepcopy(parent2[block])

    return child


def mutate_combined(genome: Dict, rate: float = MUTATION_RATE, bounds: Dict = None) -> Dict:
    """
    Gaussian mutation within parameter bounds for combined strategy.

    Each numeric parameter has `rate` chance of being mutated.
    Mutation adds Gaussian noise scaled by parameter range.

    Args:
        genome: Genome to mutate
        rate: Mutation probability per parameter
        bounds: Optional custom bounds. If None, uses PARAM_BOUNDS_COMBINED

    Returns:
        Mutated genome
    """
    mutated = copy.deepcopy(genome)
    effective_bounds = bounds if bounds else PARAM_BOUNDS_COMBINED

    for block, block_bounds in effective_bounds.items():
        if block not in mutated:
            continue

        for param, (min_val, max_val) in block_bounds.items():
            if param not in mutated[block]:
                continue

            if random.random() < rate:
                current = mutated[block][param]
                param_range = max_val - min_val

                # Gaussian noise with std = 10% of range
                noise = np.random.normal(0, param_range * 0.1)
                new_val = current + noise

                # Clamp to bounds
                new_val = max(min_val, min(max_val, new_val))

                # Round appropriately
                if isinstance(current, int):
                    new_val = int(round(new_val))
                else:
                    new_val = round(new_val, 2)

                mutated[block][param] = new_val

    return mutated


def genome_to_strategy_params_combined(genome: Dict) -> Dict:
    """
    Convert combined genome structure to strategy params.

    This maps the genome blocks to flat params expected by the strategy.

    Args:
        genome: Genome with entry, sl_long, sl_short, tp_dual_long, etc.

    Returns:
        Flat params dict for backtest engine
    """
    return {
        # Entry params
        "st_atrPeriod": genome["entry"].get("st_atrPeriod", 10),
        "st_src": genome["entry"].get("st_src", "hl2"),
        "st_mult": genome["entry"].get("st_mult", 2.0),
        "st_changeATR": genome["entry"].get("st_useATR", True),
        "rf_src_in": genome["entry"].get("rf_src", "close"),
        "rf_period": genome["entry"].get("rf_period", 100),
        "rf_mult": genome["entry"].get("rf_mult", 3.0),
        "lenRSI": genome["entry"].get("rsi_length", 14),
        "lenMA": genome["entry"].get("rsi_ma_length", 6),
        "dualFlipBarsLong": genome["entry"].get("dualFlipBarsLong", 8),
        "dualFlipBarsShort": genome["entry"].get("dualFlipBarsShort", 12),

        # Long SL params
        "st_sl_atrPeriod_L": genome["sl_long"].get("st_atrPeriod", 10),
        "st_sl_src_L": genome["sl_long"].get("st_src", "hl2"),
        "st_sl_mult_L": genome["sl_long"].get("st_mult", 3.0),
        "st_sl_useATR_L": genome["sl_long"].get("st_useATR", True),
        "rf_sl_period_L": genome["sl_long"].get("rf_period", 20),
        "rf_sl_mult_L": genome["sl_long"].get("rf_mult", 15.0),

        # Short SL params
        "st_sl_atrPeriod_S": genome["sl_short"].get("st_atrPeriod", 10),
        "st_sl_src_S": genome["sl_short"].get("st_src", "hl2"),
        "st_sl_mult_S": genome["sl_short"].get("st_mult", 2.0),
        "st_sl_useATR_S": genome["sl_short"].get("st_useATR", True),
        "rf_sl_period_S": genome["sl_short"].get("rf_period", 20),
        "rf_sl_mult_S": genome["sl_short"].get("rf_mult", 3.0),

        # Long TP Dual params
        "st_tp_dual_period_L": genome["tp_dual_long"].get("st_atrPeriod", 10),
        "st_tp_dual_mult_L": genome["tp_dual_long"].get("st_mult", 2.0),
        "rr_mult_dual_L": genome["tp_dual_long"].get("rr_mult", 4.0),

        # Short TP Dual params
        "st_tp_dual_period_S": genome["tp_dual_short"].get("st_atrPeriod", 10),
        "st_tp_dual_mult_S": genome["tp_dual_short"].get("st_mult", 2.0),
        "rr_mult_dual_S": genome["tp_dual_short"].get("rr_mult", 0.75),

        # Long TP RSI params
        "st_tp_rsi1_period_L": genome["tp_rsi_long"].get("st_atrPeriod", 10),
        "st_tp_rsi1_mult_L": genome["tp_rsi_long"].get("st_mult", 2.0),
        "rr_mult_rsi1_L": genome["tp_rsi_long"].get("rr_mult", 4.0),

        # Short TP RSI params
        "st_tp_rsi1_period_S": genome["tp_rsi_short"].get("st_atrPeriod", 10),
        "st_tp_rsi1_mult_S": genome["tp_rsi_short"].get("st_mult", 2.0),
        "rr_mult_rsi1_S": genome["tp_rsi_short"].get("rr_mult", 0.75),

        # Mode params
        "enableLong": genome["mode"].get("enableLong", True),
        "enableShort": genome["mode"].get("enableShort", True),
        "showEntryLong": genome["mode"].get("showDualFlip", True),
        "showEntryShort": genome["mode"].get("showDualFlip", True),
        "showEntryRSI_L": genome["mode"].get("showRSI", True),
        "showEntryRSI_S": genome["mode"].get("showRSI", True),
    }


def strategy_params_to_genome_combined(params: Dict) -> Dict:
    """
    Convert flat strategy params to combined genome structure.

    This is the inverse of genome_to_strategy_params_combined.

    Args:
        params: Flat params dict from backtest config

    Returns:
        Genome structure with entry, sl_long, sl_short, etc.
    """
    return {
        "entry": {
            "st_atrPeriod": params.get("st_atrPeriod", 10),
            "st_src": params.get("st_src", "hl2"),
            "st_mult": params.get("st_mult", 2.0),
            "st_useATR": params.get("st_changeATR", True),
            "rf_src": params.get("rf_src_in", "close"),
            "rf_period": params.get("rf_period", 100),
            "rf_mult": params.get("rf_mult", 3.0),
            "rsi_length": params.get("lenRSI", 14),
            "rsi_ma_length": params.get("lenMA", 6),
            "dualFlipBarsLong": params.get("dualFlipBarsLong", 8),
            "dualFlipBarsShort": params.get("dualFlipBarsShort", 12),
        },
        "sl_long": {
            "st_atrPeriod": params.get("st_sl_atrPeriod_L", 10),
            "st_src": params.get("st_sl_src_L", "hl2"),
            "st_mult": params.get("st_sl_mult_L", 3.0),
            "st_useATR": params.get("st_sl_useATR_L", True),
            "rf_period": params.get("rf_sl_period_L", 20),
            "rf_mult": params.get("rf_sl_mult_L", 15.0),
        },
        "sl_short": {
            "st_atrPeriod": params.get("st_sl_atrPeriod_S", 10),
            "st_src": params.get("st_sl_src_S", "hl2"),
            "st_mult": params.get("st_sl_mult_S", 2.0),
            "st_useATR": params.get("st_sl_useATR_S", True),
            "rf_period": params.get("rf_sl_period_S", 20),
            "rf_mult": params.get("rf_sl_mult_S", 3.0),
        },
        "tp_dual_long": {
            "st_atrPeriod": params.get("st_tp_dual_period_L", 10),
            "st_mult": params.get("st_tp_dual_mult_L", 2.0),
            "rr_mult": params.get("rr_mult_dual_L", 4.0),
        },
        "tp_dual_short": {
            "st_atrPeriod": params.get("st_tp_dual_period_S", 10),
            "st_mult": params.get("st_tp_dual_mult_S", 2.0),
            "rr_mult": params.get("rr_mult_dual_S", 0.75),
        },
        "tp_rsi_long": {
            "st_atrPeriod": params.get("st_tp_rsi1_period_L", 10),
            "st_mult": params.get("st_tp_rsi1_mult_L", 2.0),
            "rr_mult": params.get("rr_mult_rsi1_L", 4.0),
        },
        "tp_rsi_short": {
            "st_atrPeriod": params.get("st_tp_rsi1_period_S", 10),
            "st_mult": params.get("st_tp_rsi1_mult_S", 2.0),
            "rr_mult": params.get("rr_mult_rsi1_S", 0.75),
        },
        "mode": {
            "enableLong": params.get("enableLong", True),
            "enableShort": params.get("enableShort", True),
            "showDualFlip": params.get("showEntryLong", True) or params.get("showEntryShort", True),
            "showRSI": params.get("showEntryRSI_L", True) or params.get("showEntryRSI_S", True),
        },
    }


class GenomeOptimizerCombined:
    """
    Evolutionary genome optimizer for rf_st_rsi_combined strategy.

    Supports Long/Short separate parameters.

    NEW: Uses IntelligentSampler for smarter parameter generation:
    - Logical Constraints: Reject invalid param combinations
    - Correlated Sampling: Sample related params together
    - Archetype Templates: Use predefined trading styles
    - Bayesian Learning: Learn from past results
    """

    def __init__(
        self,
        fitness_fn: Callable[[Dict], float],
        population_size: int = POPULATION_SIZE,
        generations: int = GENERATIONS,
        mutation_rate: float = MUTATION_RATE,
        crossover_rate: float = CROSSOVER_RATE,
        elite_count: int = ELITE_COUNT,
        max_workers: int = 8,
        param_bounds: Dict = None,
        cancel_check_fn: Callable[[], bool] = None,
        use_intelligent_sampling: bool = True  # NEW: Enable intelligent sampling
    ):
        self.fitness_fn = fitness_fn
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_count = elite_count
        self.max_workers = max_workers
        self.param_bounds = param_bounds if param_bounds else PARAM_BOUNDS_COMBINED
        self.cancel_check_fn = cancel_check_fn
        self.use_intelligent_sampling = use_intelligent_sampling

        self.generation_history = []
        self.best_genome = None
        self.best_score = float("-inf")

        # NEW: Initialize IntelligentSampler
        self.intelligent_sampler = None
        if self.use_intelligent_sampling:
            try:
                from .intelligent_sampling import IntelligentSampler
                self.intelligent_sampler = IntelligentSampler(self.param_bounds)
                logger.info("[Combined] IntelligentSampler enabled for smarter param generation")
            except ImportError as e:
                logger.warning(f"[Combined] Could not import IntelligentSampler: {e}")
                self.use_intelligent_sampling = False

    def _check_cancelled(self) -> bool:
        """Check if optimization should be cancelled."""
        if self.cancel_check_fn:
            return self.cancel_check_fn()
        return False

    def initialize_population(
        self,
        seed_genomes: List[Dict] = None,
        regime: MarketRegime = None
    ) -> List[Dict]:
        """
        Initialize population with seeds + intelligent randoms.

        NEW STRATEGY (BXH-based):
        - If seed_genomes provided: Use them as base for optimization
        - Create variants via mutation and crossover from seeds
        - Fill remaining with intelligent sampling

        NO DEDUPLICATION here - allow duplicates for better evolution.
        Deduplication only happens at final_results before returning to UI.
        """
        population = []

        # Add ALL seed genomes from BXH (no limit, no dedup)
        if seed_genomes:
            for genome in seed_genomes:
                valid, _ = validate_genome_combined(genome)
                if valid:
                    population.append(copy.deepcopy(genome))
            logger.info(f"[Combined] Added {len(population)} seed genomes from BXH")

            # Create MUTATED variants from seed genomes
            if len(population) > 0 and len(population) < self.population_size:
                num_mutants = min(len(population), (self.population_size - len(population)) // 3)
                for i in range(num_mutants):
                    base_genome = copy.deepcopy(population[i % len(population)])
                    mutant = mutate_combined(base_genome, rate=0.3, bounds=self.param_bounds)
                    mutant = repair_genome_combined(mutant)
                    mutant = clamp_to_bounds_combined(mutant)
                    population.append(mutant)
                logger.info(f"[Combined] Created {num_mutants} mutated variants from BXH genomes")

            # Create CROSSOVER offspring from seed genomes
            if len(seed_genomes) >= 2 and len(population) < self.population_size:
                num_crossovers = min(len(seed_genomes), (self.population_size - len(population)) // 3)
                for i in range(num_crossovers):
                    parent1 = seed_genomes[i % len(seed_genomes)]
                    parent2 = seed_genomes[(i + 1) % len(seed_genomes)]
                    child = crossover_combined(parent1, parent2)
                    child = repair_genome_combined(child)
                    child = clamp_to_bounds_combined(child)
                    population.append(child)
                logger.info(f"[Combined] Created {num_crossovers} crossover offspring from BXH genomes")

        # Fill rest with intelligent sampling or random
        remaining = self.population_size - len(population)

        if remaining > 0:
            if self.intelligent_sampler and self.use_intelligent_sampling:
                logger.info(f"[Combined] Generating {remaining} genomes using IntelligentSampler")
                for _ in range(remaining):
                    genome = self.intelligent_sampler.sample()
                    population.append(genome)

                # Log sampling stats
                stats = self.intelligent_sampler.get_stats()
                logger.info(
                    f"[Combined] IntelligentSampler stats: "
                    f"generated={stats['generated']}, rejected={stats['rejected']}, "
                    f"rejection_rate={stats['rejection_rate']:.1%}, "
                    f"strategy_distribution={stats['strategy_distribution']}"
                )
            else:
                logger.info(f"[Combined] Generating {remaining} genomes using random sampling")
                for _ in range(remaining):
                    genome = create_random_genome_combined_with_bounds(self.param_bounds)
                    population.append(genome)

        logger.info(f"[Combined] Total population: {len(population)} (seeds={len(seed_genomes) if seed_genomes else 0})")
        return population

    def evaluate_population(
        self,
        population: List[Dict],
        progress_cb: Callable = None
    ) -> List[float]:
        """
        Evaluate fitness of all genomes in parallel.
        """
        pop_size = len(population)

        if PARALLEL_FITNESS and pop_size >= 4:
            return self._evaluate_parallel(population, progress_cb)
        else:
            return self._evaluate_sequential(population, progress_cb)

    def _evaluate_parallel(
        self,
        population: List[Dict],
        progress_cb: Callable = None
    ) -> List[float]:
        """Parallel fitness evaluation."""
        pop_size = len(population)
        workers = min(MAX_FITNESS_WORKERS, pop_size)
        logger.info(f"[Combined] Evaluating {pop_size} genomes in PARALLEL ({workers} workers)")

        scores = [float("-inf")] * pop_size
        completed = 0

        def evaluate_single(idx_genome):
            idx, genome = idx_genome
            try:
                score = self._safe_fitness(genome)
                if score is None or (isinstance(score, float) and score != score):
                    return idx, float("-inf")
                return idx, float(score)
            except Exception as e:
                logger.warning(f"Fitness evaluation failed for genome {idx}: {e}")
                return idx, float("-inf")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(evaluate_single, (i, g)): i
                for i, g in enumerate(population)
            }

            for future in as_completed(futures):
                if self._check_cancelled():
                    logger.info("Parallel evaluation cancelled")
                    for f in futures:
                        f.cancel()
                    break

                try:
                    idx, score = future.result()
                    scores[idx] = score
                    completed += 1

                    if progress_cb and completed % 10 == 0:
                        progress_cb(completed, pop_size)
                except Exception as e:
                    logger.warning(f"Future failed: {e}")

        valid_count = sum(1 for s in scores if s > float("-inf"))
        logger.info(f"[Combined] Parallel evaluation complete: {valid_count}/{pop_size} valid scores")
        return scores

    def _evaluate_sequential(
        self,
        population: List[Dict],
        progress_cb: Callable = None
    ) -> List[float]:
        """Sequential fitness evaluation."""
        scores = []
        pop_size = len(population)
        logger.info(f"[Combined] Evaluating {pop_size} genomes sequentially")

        for i, genome in enumerate(population):
            if self._check_cancelled():
                logger.info(f"Sequential evaluation cancelled at {i}/{pop_size}")
                scores.extend([float("-inf")] * (pop_size - i))
                break

            try:
                score = self._safe_fitness(genome)
                if score is None or (isinstance(score, float) and score != score):
                    score = float("-inf")
                scores.append(float(score))
            except Exception as e:
                logger.warning(f"Fitness evaluation failed for genome {i}: {e}")
                scores.append(float("-inf"))

            if progress_cb and (i + 1) % 10 == 0:
                progress_cb(i + 1, pop_size)

        valid_count = sum(1 for s in scores if s > float("-inf"))
        logger.info(f"[Combined] Evaluation complete: {valid_count}/{pop_size} valid scores")
        return scores

    def _safe_fitness(self, genome: Dict) -> float:
        """Safely evaluate fitness, returning -inf on error."""
        try:
            score = self.fitness_fn(genome)
            return score
        except Exception as e:
            import traceback
            logger.warning(f"Fitness error: {e}\n{traceback.format_exc()}")
            return float("-inf")

    def evolve_generation(
        self,
        population: List[Dict],
        scores: List[float]
    ) -> List[Dict]:
        """
        Evolve one generation using combined genome operations.

        NO DEDUPLICATION - allow duplicates for better evolution.
        Elite genomes are preserved, offspring created via crossover/mutation.
        """
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        next_gen = []
        crossover_count = 0
        intelligent_fallback_count = 0

        # Keep ALL elites (no dedup - best performers should dominate)
        for i in range(min(self.elite_count, len(sorted_indices))):
            elite = copy.deepcopy(population[sorted_indices[i]])
            next_gen.append(elite)

        # Fill rest with offspring
        while len(next_gen) < self.population_size:
            # Tournament selection
            parents = tournament_select(population, scores, 2)

            # Crossover
            if random.random() < self.crossover_rate:
                child = crossover_combined(parents[0], parents[1])
                crossover_count += 1
            else:
                child = copy.deepcopy(parents[0])

            # Mutate
            child = mutate_combined(child, self.mutation_rate, bounds=self.param_bounds)

            # Repair and validate
            child = repair_genome_combined(child)
            child = clamp_to_bounds_combined(child)

            # Validate against logical constraints
            valid_combined, _ = validate_genome_combined(child)

            # Also check logical constraints if intelligent sampling is enabled
            valid_constraints = True
            if self.intelligent_sampler and self.use_intelligent_sampling:
                try:
                    from .intelligent_sampling import ConstraintValidator
                    valid_constraints, _ = ConstraintValidator.validate(child)
                except ImportError:
                    pass

            if valid_combined and valid_constraints:
                next_gen.append(child)
            elif self.intelligent_sampler and self.use_intelligent_sampling:
                # Use IntelligentSampler as fallback for invalid offspring
                fallback = self.intelligent_sampler.sample()
                next_gen.append(fallback)
                intelligent_fallback_count += 1
            else:
                # Fallback: random genome
                genome = create_random_genome_combined_with_bounds(self.param_bounds)
                next_gen.append(genome)

        if crossover_count > 0 or intelligent_fallback_count > 0:
            logger.debug(
                f"[Combined] Evolution: crossovers={crossover_count}, "
                f"intelligent_fallbacks={intelligent_fallback_count}"
            )

        return next_gen[:self.population_size]

    def optimize(
        self,
        seed_genomes: List[Dict] = None,
        regime: MarketRegime = None,
        progress_cb: Callable = None
    ) -> Tuple[Dict, float, List[Dict]]:
        """
        Run full evolutionary optimization for combined strategy.

        NEW: Records results to IntelligentSampler for Bayesian learning.
        This allows future generations to learn from what worked well.

        Returns:
            (best_genome, best_score, top_genomes)
        """
        if self._check_cancelled():
            logger.info("[Combined] Optimization cancelled before start")
            return None, 0, []

        # Initialize
        population = self.initialize_population(seed_genomes, regime)

        logger.info(f"[Combined] Starting evolution: pop={self.population_size}, gen={self.generations}")
        if self.intelligent_sampler:
            logger.info("[Combined] IntelligentSampler enabled - Bayesian learning active")

        for gen in range(self.generations):
            if self._check_cancelled():
                logger.info(f"[Combined] Optimization cancelled at generation {gen}/{self.generations}")
                break

            # Evaluate
            scores = self.evaluate_population(population)

            # NEW: Record results for Bayesian learning
            if self.intelligent_sampler and self.use_intelligent_sampling:
                for genome, score in zip(population, scores):
                    if score > float("-inf"):
                        self.intelligent_sampler.record_result(genome, score)

            # Track best
            gen_best_idx = max(range(len(scores)), key=lambda i: scores[i])
            gen_best_score = scores[gen_best_idx]

            if gen_best_score > self.best_score:
                self.best_score = gen_best_score
                self.best_genome = copy.deepcopy(population[gen_best_idx])

            # Record history
            valid_scores = [s for s in scores if s > float("-inf")]
            self.generation_history.append({
                "generation": gen,
                "best_score": gen_best_score,
                "avg_score": sum(valid_scores) / max(1, len(valid_scores)),
                "valid_count": len(valid_scores),
            })

            if progress_cb:
                progress_cb(gen + 1, self.generations, gen_best_score)

            # Log Bayesian learning progress every few generations
            if self.intelligent_sampler and gen > 0 and gen % 3 == 0:
                stats = self.intelligent_sampler.get_stats()
                best_arch = stats.get("best_archetype", "unknown")
                best_sens = stats.get("best_sensitivity", "unknown")
                best_risk = stats.get("best_risk", "unknown")
                logger.info(
                    f"[Combined] Gen {gen} Bayesian insights: "
                    f"best_archetype={best_arch}, best_sensitivity={best_sens}, best_risk={best_risk}"
                )

            # Evolve (except last generation)
            if gen < self.generations - 1:
                population = self.evolve_generation(population, scores)

        if self._check_cancelled():
            logger.info("[Combined] Optimization cancelled before final evaluation")
            return self.best_genome, self.best_score, [self.best_genome] if self.best_genome else []

        # Get top genomes from final population
        final_scores = self.evaluate_population(population)
        sorted_indices = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)
        top_genomes = [population[i] for i in sorted_indices[:20]]

        # Fallback if no best genome found
        if self.best_genome is None and population:
            self.best_genome = population[0]
            self.best_score = final_scores[0] if final_scores else 0
            logger.warning("[Combined] No valid genome found during optimization, using fallback")

        # Log final IntelligentSampler stats
        if self.intelligent_sampler:
            final_stats = self.intelligent_sampler.get_stats()
            logger.info(
                f"[Combined] Final IntelligentSampler stats: "
                f"total_samples={final_stats['generated']}, "
                f"rejection_rate={final_stats['rejection_rate']:.1%}, "
                f"best_archetype={final_stats.get('best_archetype', 'N/A')}, "
                f"best_sensitivity={final_stats.get('best_sensitivity', 'N/A')}, "
                f"best_risk={final_stats.get('best_risk', 'N/A')}"
            )

        return self.best_genome, self.best_score, top_genomes
