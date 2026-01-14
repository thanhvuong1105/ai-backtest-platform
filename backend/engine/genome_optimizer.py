# engine/genome_optimizer.py
"""
Evolutionary Genome Optimizer for Quant AI Brain

Full evolutionary optimization with:
- Crossover at genome block level
- Gaussian mutation within bounds
- Tournament selection
- Phased optimization (Entry → SL → TP → Mode)
"""

import os
import random
import copy
import logging
from typing import Dict, Any, List, Tuple, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .coherence_validator import validate_genome, repair_genome, clamp_to_bounds, PARAM_BOUNDS
from .regime_classifier import MarketRegime, sample_params_for_regime

logger = logging.getLogger(__name__)

# Configuration
POPULATION_SIZE = int(os.getenv("GENOME_POPULATION_SIZE", 50))
GENERATIONS = int(os.getenv("GENOME_GENERATIONS", 10))
MUTATION_RATE = float(os.getenv("GENOME_MUTATION_RATE", 0.15))
CROSSOVER_RATE = float(os.getenv("GENOME_CROSSOVER_RATE", 0.7))
TOURNAMENT_SIZE = int(os.getenv("GENOME_TOURNAMENT_SIZE", 5))
ELITE_COUNT = int(os.getenv("GENOME_ELITE_COUNT", 5))


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


def create_random_genome() -> Dict:
    """Create a random genome within bounds."""
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
        param_bounds: Dict = None
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
        """
        self.fitness_fn = fitness_fn
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_count = elite_count
        self.max_workers = max_workers
        self.param_bounds = param_bounds  # None means use PARAM_BOUNDS

        # Tracking
        self.generation_history = []
        self.best_genome = None
        self.best_score = float("-inf")

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
        Evaluate fitness of all genomes in parallel.

        Args:
            population: List of genomes
            progress_cb: Optional progress callback

        Returns:
            List of fitness scores
        """
        scores = []
        logger.info(f"Evaluating {len(population)} genomes sequentially")

        for i, genome in enumerate(population):
            try:
                score = self._safe_fitness(genome)
                # Ensure score is a valid float
                if score is None or (isinstance(score, float) and score != score):  # NaN check
                    score = float("-inf")
                scores.append(float(score))
            except Exception as e:
                logger.warning(f"Fitness evaluation failed for genome {i}: {e}")
                scores.append(float("-inf"))

            if progress_cb and (i + 1) % 10 == 0:
                progress_cb(i + 1, len(population))

        valid_count = sum(1 for s in scores if s > float("-inf"))
        logger.info(f"Evaluation complete: {valid_count}/{len(population)} valid scores")
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
        # Initialize
        population = self.initialize_population(seed_genomes, regime)

        logger.info(f"Starting evolution: pop={self.population_size}, gen={self.generations}")

        for gen in range(self.generations):
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

        # Get top genomes from final population
        final_scores = self.evaluate_population(population)
        sorted_indices = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)
        top_genomes = [population[i] for i in sorted_indices[:20]]

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
    """

    def __init__(
        self,
        fitness_fn: Callable[[Dict], float],
        generations_per_phase: int = 5,
        population_size: int = 30,
        param_bounds: Dict = None
    ):
        self.fitness_fn = fitness_fn
        self.generations_per_phase = generations_per_phase
        self.population_size = population_size
        self.param_bounds = param_bounds  # Auto-expanded bounds from memory

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
        # Start with base genome
        if seed_genomes and len(seed_genomes) > 0:
            base_genome = copy.deepcopy(seed_genomes[0])
        else:
            base_genome = create_random_genome()

        all_tested = []

        # Phase 1: Entry
        logger.info("Phase 1: Optimizing Entry genome")
        entry_optimizer = GenomeOptimizer(
            fitness_fn=self.fitness_fn,
            population_size=self.population_size,
            generations=self.generations_per_phase,
            param_bounds=self.param_bounds  # Pass dynamic bounds
        )
        best_entry, _, top_entry = entry_optimizer.optimize(seed_genomes, regime)
        all_tested.extend(top_entry)

        # Lock entry (with fallback check)
        if best_entry and "entry" in best_entry:
            base_genome["entry"] = best_entry["entry"]
        else:
            logger.warning("Phase 1 returned no valid entry genome, using default")

        # Phase 2: SL
        logger.info("Phase 2: Optimizing SL genome")
        sl_seeds = [self._vary_sl(base_genome) for _ in range(10)]
        sl_optimizer = GenomeOptimizer(
            fitness_fn=self.fitness_fn,
            population_size=self.population_size,
            generations=self.generations_per_phase,
            param_bounds=self.param_bounds  # Pass dynamic bounds
        )
        best_sl, _, top_sl = sl_optimizer.optimize(sl_seeds, regime)
        all_tested.extend(top_sl)

        # Lock SL (with fallback check)
        if best_sl and "sl" in best_sl:
            base_genome["sl"] = best_sl["sl"]
        else:
            logger.warning("Phase 2 returned no valid SL genome, using default")

        # Phase 3: TP
        logger.info("Phase 3: Optimizing TP genome")
        tp_seeds = [self._vary_tp(base_genome) for _ in range(10)]
        tp_optimizer = GenomeOptimizer(
            fitness_fn=self.fitness_fn,
            population_size=self.population_size,
            generations=self.generations_per_phase,
            param_bounds=self.param_bounds  # Pass dynamic bounds
        )
        best_tp, _, top_tp = tp_optimizer.optimize(tp_seeds, regime)
        all_tested.extend(top_tp)

        # Lock TP (with fallback check)
        if best_tp:
            if "tp_dual" in best_tp:
                base_genome["tp_dual"] = best_tp["tp_dual"]
            if "tp_rsi" in best_tp:
                base_genome["tp_rsi"] = best_tp["tp_rsi"]
        else:
            logger.warning("Phase 3 returned no valid TP genome, using default")

        # Phase 4: Mode
        logger.info("Phase 4: Optimizing Mode")
        mode_variants = self._generate_mode_variants(base_genome)
        mode_scores = [self.fitness_fn(g) for g in mode_variants]
        best_mode_idx = max(range(len(mode_scores)), key=lambda i: mode_scores[i])
        base_genome["mode"] = mode_variants[best_mode_idx]["mode"]

        # Final score
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
