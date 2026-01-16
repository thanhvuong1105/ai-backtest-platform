# engine/genome_optimizer_v2.py
"""
Genetic Algorithm Optimizer v2.0 - Advanced Evolutionary Optimization

Key improvements over v1:
1. Adaptive Mutation Rate: Starts high (0.3), decreases to low (0.05) as generations progress
2. Fitness Sharing: Penalizes similar genomes to maintain diversity
3. Multi-point Crossover: Swaps multiple gene blocks randomly
4. Tournament Selection with variable pressure
5. Niching for diversity preservation
6. Stagnation detection and population restart
7. Elitism with crowding distance

Formula for adaptive mutation:
mutation_rate = max_rate - (max_rate - min_rate) * (generation / total_generations)

Fitness Sharing:
shared_fitness = raw_fitness / niche_count
where niche_count = sum(sharing_function(distance(i, j)) for all j)
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

# Configuration defaults
DEFAULT_POPULATION_SIZE = int(os.getenv("GENOME_POPULATION_SIZE", 30))
DEFAULT_GENERATIONS = int(os.getenv("GENOME_GENERATIONS", 3))
DEFAULT_MUTATION_RATE_MAX = 0.3
DEFAULT_MUTATION_RATE_MIN = 0.05
DEFAULT_CROSSOVER_RATE = 0.8
DEFAULT_TOURNAMENT_SIZE = 5
DEFAULT_ELITE_COUNT = 5
DEFAULT_NICHE_RADIUS = 0.15  # For fitness sharing


# ═══════════════════════════════════════════════════════════════════════════════
# GENOME DISTANCE & SIMILARITY
# ═══════════════════════════════════════════════════════════════════════════════

def genome_to_vector(genome: Dict) -> np.ndarray:
    """
    Convert genome to flat numeric vector for distance calculations.

    Normalizes all parameters to 0-1 range.
    """
    vector = []

    # Entry block
    entry = genome.get("entry", {})
    vector.extend([
        (entry.get("st_atrPeriod", 10) - 1) / 99,  # 1-100 -> 0-1
        (entry.get("st_mult", 2.0) - 1) / 29,  # 1-30 -> 0-1
        (entry.get("rf_period", 50) - 1) / 99,  # 1-100 -> 0-1
        (entry.get("rf_mult", 3.0) - 1) / 29,  # 1-30 -> 0-1
        (entry.get("rsi_length", 14) - 1) / 19,  # 1-20 -> 0-1
        (entry.get("rsi_ma_length", 5) - 1) / 14,  # 1-15 -> 0-1
    ])

    # SL block
    sl = genome.get("sl", {})
    vector.extend([
        (sl.get("st_atrPeriod", 10) - 1) / 99,
        (sl.get("st_mult", 4.0) - 1) / 29,
        (sl.get("rf_period", 50) - 1) / 99,
        (sl.get("rf_mult", 7.0) - 1) / 29,
    ])

    # TP Dual block
    tp_dual = genome.get("tp_dual", {})
    vector.extend([
        (tp_dual.get("st_atrPeriod", 10) - 1) / 99,
        (tp_dual.get("st_mult", 2.0) - 1) / 29,
        (tp_dual.get("rr_mult", 1.5) - 0.1) / 4.9,  # 0.1-5 -> 0-1
    ])

    # TP RSI block
    tp_rsi = genome.get("tp_rsi", {})
    vector.extend([
        (tp_rsi.get("st_atrPeriod", 10) - 1) / 99,
        (tp_rsi.get("st_mult", 2.0) - 1) / 29,
        (tp_rsi.get("rr_mult", 1.5) - 0.1) / 4.9,
    ])

    # Mode block
    mode = genome.get("mode", {})
    vector.extend([
        1.0 if mode.get("showDualFlip", True) else 0.0,
        1.0 if mode.get("showRSI", True) else 0.0,
    ])

    return np.array(vector)


def calculate_genome_distance(genome1: Dict, genome2: Dict) -> float:
    """
    Calculate normalized Euclidean distance between two genomes.

    Returns value between 0 (identical) and 1 (maximally different).
    """
    vec1 = genome_to_vector(genome1)
    vec2 = genome_to_vector(genome2)

    distance = np.linalg.norm(vec1 - vec2) / np.sqrt(len(vec1))
    return min(1.0, distance)


def sharing_function(distance: float, sigma: float = DEFAULT_NICHE_RADIUS) -> float:
    """
    Sharing function for fitness sharing.

    Returns 1 if distance < sigma, decreases to 0 as distance approaches sigma.
    """
    if distance >= sigma:
        return 0.0
    return 1.0 - (distance / sigma) ** 2


# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED CROSSOVER OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def single_point_crossover(parent1: Dict, parent2: Dict) -> Dict:
    """
    Single-point crossover at genome block level.
    """
    child = copy.deepcopy(parent1)
    blocks = ["entry", "sl", "tp_dual", "tp_rsi", "mode"]

    crossover_point = random.randint(1, len(blocks) - 1)

    for i in range(crossover_point, len(blocks)):
        block = blocks[i]
        if block in parent2:
            child[block] = copy.deepcopy(parent2[block])

    return child


def multi_point_crossover(parent1: Dict, parent2: Dict, n_points: int = 2) -> Dict:
    """
    Multi-point crossover - randomly swaps multiple genome blocks.

    Args:
        parent1: First parent genome
        parent2: Second parent genome
        n_points: Number of crossover points

    Returns:
        Child genome with genes from both parents
    """
    child = copy.deepcopy(parent1)
    blocks = ["entry", "sl", "tp_dual", "tp_rsi", "mode"]

    # Select random blocks to swap
    swap_blocks = random.sample(blocks, min(n_points, len(blocks)))

    for block in swap_blocks:
        if block in parent2:
            child[block] = copy.deepcopy(parent2[block])

    return child


def uniform_crossover(parent1: Dict, parent2: Dict) -> Dict:
    """
    Uniform crossover - each parameter has 50% chance from each parent.
    """
    child = copy.deepcopy(parent1)

    for block in ["entry", "sl", "tp_dual", "tp_rsi"]:
        if block in parent2 and block in child:
            for param in child[block]:
                if param in parent2[block] and random.random() < 0.5:
                    child[block][param] = copy.deepcopy(parent2[block][param])

    # Mode is inherited as whole block
    if random.random() < 0.5 and "mode" in parent2:
        child["mode"] = copy.deepcopy(parent2["mode"])

    return child


def blx_alpha_crossover(parent1: Dict, parent2: Dict, alpha: float = 0.5) -> Dict:
    """
    BLX-alpha crossover for numeric parameters.

    For each parameter, child value is sampled from extended range:
    [min(p1,p2) - alpha*range, max(p1,p2) + alpha*range]
    """
    child = copy.deepcopy(parent1)

    for block in ["entry", "sl", "tp_dual", "tp_rsi"]:
        if block not in parent2 or block not in child:
            continue

        for param in child[block]:
            if param not in parent2[block]:
                continue

            v1 = parent1[block][param]
            v2 = parent2[block][param]

            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                min_val = min(v1, v2)
                max_val = max(v1, v2)
                range_val = max_val - min_val

                # Extended range
                lower = min_val - alpha * range_val
                upper = max_val + alpha * range_val

                # Sample new value
                new_val = random.uniform(lower, upper)

                # Preserve type
                if isinstance(v1, int):
                    new_val = int(round(new_val))
                else:
                    new_val = round(new_val, 2)

                child[block][param] = new_val

    return child


# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE MUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def adaptive_mutate(
    genome: Dict,
    generation: int,
    total_generations: int,
    max_rate: float = DEFAULT_MUTATION_RATE_MAX,
    min_rate: float = DEFAULT_MUTATION_RATE_MIN,
    bounds: Dict = None
) -> Dict:
    """
    Gaussian mutation with adaptive rate.

    Mutation rate decreases linearly from max_rate to min_rate as generations progress.
    This allows exploration early and exploitation later.

    Formula:
    rate = max_rate - (max_rate - min_rate) * (generation / total_generations)
    """
    # Calculate adaptive rate
    progress = generation / max(total_generations, 1)
    rate = max_rate - (max_rate - min_rate) * progress

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

                # Adaptive std: starts high, decreases over generations
                std_scale = 0.15 - 0.1 * progress  # 0.15 -> 0.05
                noise = np.random.normal(0, param_range * std_scale)
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


def polynomial_mutation(
    genome: Dict,
    generation: int,
    total_generations: int,
    bounds: Dict = None,
    eta: float = 20
) -> Dict:
    """
    Polynomial mutation (SBX-style).

    More controlled mutation with eta parameter controlling distribution shape.
    Higher eta = more focused around current value.
    """
    mutated = copy.deepcopy(genome)
    effective_bounds = bounds if bounds else PARAM_BOUNDS

    # Adaptive mutation probability
    progress = generation / max(total_generations, 1)
    pm = 0.3 - 0.2 * progress  # 0.3 -> 0.1

    for block, block_bounds in effective_bounds.items():
        if block not in mutated:
            continue

        for param, (min_val, max_val) in block_bounds.items():
            if param not in mutated[block]:
                continue

            if random.random() < pm:
                y = mutated[block][param]
                delta_l = (y - min_val) / (max_val - min_val)
                delta_r = (max_val - y) / (max_val - min_val)

                u = random.random()

                if u < 0.5:
                    delta = (2 * u + (1 - 2 * u) * (1 - delta_l) ** (eta + 1)) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - delta_r) ** (eta + 1)) ** (1 / (eta + 1))

                new_val = y + delta * (max_val - min_val)
                new_val = max(min_val, min(max_val, new_val))

                if isinstance(y, int):
                    new_val = int(round(new_val))
                else:
                    new_val = round(new_val, 2)

                mutated[block][param] = new_val

    return mutated


# ═══════════════════════════════════════════════════════════════════════════════
# FITNESS SHARING
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_shared_fitness(
    population: List[Dict],
    raw_scores: List[float],
    sigma: float = DEFAULT_NICHE_RADIUS
) -> List[float]:
    """
    Calculate shared fitness scores to maintain diversity.

    Each individual's fitness is divided by its niche count,
    penalizing genomes that are too similar to others.

    Args:
        population: List of genomes
        raw_scores: Raw fitness scores
        sigma: Niche radius for sharing function

    Returns:
        List of shared fitness scores
    """
    n = len(population)
    shared_scores = []

    for i in range(n):
        if raw_scores[i] <= float("-inf"):
            shared_scores.append(float("-inf"))
            continue

        # Calculate niche count
        niche_count = 0.0
        for j in range(n):
            if raw_scores[j] <= float("-inf"):
                continue
            distance = calculate_genome_distance(population[i], population[j])
            niche_count += sharing_function(distance, sigma)

        # Shared fitness
        if niche_count > 0:
            shared_scores.append(raw_scores[i] / niche_count)
        else:
            shared_scores.append(raw_scores[i])

    return shared_scores


# ═══════════════════════════════════════════════════════════════════════════════
# SELECTION OPERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def tournament_selection(
    population: List[Dict],
    scores: List[float],
    n: int,
    tournament_size: int = DEFAULT_TOURNAMENT_SIZE
) -> List[Dict]:
    """
    Tournament selection with variable tournament size.
    """
    selected = []

    for _ in range(n):
        indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        best_idx = max(indices, key=lambda i: scores[i])
        selected.append(copy.deepcopy(population[best_idx]))

    return selected


def roulette_wheel_selection(
    population: List[Dict],
    scores: List[float],
    n: int
) -> List[Dict]:
    """
    Roulette wheel (fitness proportionate) selection.
    """
    # Shift scores to positive
    min_score = min(s for s in scores if s > float("-inf"))
    shifted_scores = [max(0, s - min_score + 1) for s in scores]
    total = sum(shifted_scores)

    if total == 0:
        return random.sample(population, n)

    probabilities = [s / total for s in shifted_scores]
    selected = []

    for _ in range(n):
        r = random.random()
        cumsum = 0
        for i, prob in enumerate(probabilities):
            cumsum += prob
            if r <= cumsum:
                selected.append(copy.deepcopy(population[i]))
                break

    # Fallback if something went wrong
    while len(selected) < n:
        selected.append(copy.deepcopy(random.choice(population)))

    return selected


def rank_selection(
    population: List[Dict],
    scores: List[float],
    n: int,
    selection_pressure: float = 2.0
) -> List[Dict]:
    """
    Rank-based selection with configurable selection pressure.

    Higher selection_pressure = more bias toward top performers.
    """
    # Sort by score
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i])

    # Calculate rank probabilities
    pop_size = len(population)
    probabilities = []
    for rank in range(pop_size):
        prob = (2 - selection_pressure + 2 * (selection_pressure - 1) * rank / (pop_size - 1)) / pop_size
        probabilities.append(max(0, prob))

    # Normalize
    total = sum(probabilities)
    if total > 0:
        probabilities = [p / total for p in probabilities]
    else:
        probabilities = [1 / pop_size] * pop_size

    selected = []
    for _ in range(n):
        r = random.random()
        cumsum = 0
        for i, prob in enumerate(probabilities):
            cumsum += prob
            if r <= cumsum:
                selected.append(copy.deepcopy(population[sorted_indices[i]]))
                break

    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# CROWDING DISTANCE (for diversity in elites)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_crowding_distances(
    population: List[Dict],
    scores: List[float]
) -> List[float]:
    """
    Calculate crowding distance for diversity preservation.

    Used to select diverse elites from Pareto front.
    """
    n = len(population)
    if n <= 2:
        return [float("inf")] * n

    distances = [0.0] * n
    vectors = [genome_to_vector(g) for g in population]

    # For each dimension
    n_dims = len(vectors[0])
    for d in range(n_dims):
        # Sort by this dimension
        sorted_indices = sorted(range(n), key=lambda i: vectors[i][d])

        # Boundary solutions get infinite distance
        distances[sorted_indices[0]] = float("inf")
        distances[sorted_indices[-1]] = float("inf")

        # Calculate distances for middle solutions
        dim_range = vectors[sorted_indices[-1]][d] - vectors[sorted_indices[0]][d]
        if dim_range > 0:
            for i in range(1, n - 1):
                distances[sorted_indices[i]] += (
                    vectors[sorted_indices[i + 1]][d] - vectors[sorted_indices[i - 1]][d]
                ) / dim_range

    return distances


def select_diverse_elites(
    population: List[Dict],
    scores: List[float],
    n_elites: int
) -> List[Dict]:
    """
    Select elites with maximum diversity using crowding distance.
    """
    if len(population) <= n_elites:
        return copy.deepcopy(population)

    # Calculate crowding distances
    crowding = calculate_crowding_distances(population, scores)

    # Combined score: fitness + diversity bonus
    combined = []
    for i in range(len(population)):
        if scores[i] > float("-inf"):
            # Normalize crowding distance
            cd = min(crowding[i], 10) / 10 if crowding[i] != float("inf") else 1.0
            combined.append((i, scores[i] * 0.8 + cd * 0.2 * max(scores)))
        else:
            combined.append((i, float("-inf")))

    # Sort by combined score
    combined.sort(key=lambda x: x[1], reverse=True)

    return [copy.deepcopy(population[i]) for i, _ in combined[:n_elites]]


# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED GENOME OPTIMIZER v2
# ═══════════════════════════════════════════════════════════════════════════════

class GenomeOptimizerV2:
    """
    Advanced Genetic Algorithm Optimizer v2.0

    Features:
    - Adaptive mutation rate (high -> low)
    - Fitness sharing for diversity
    - Multiple crossover operators
    - Stagnation detection with restart
    - Elitism with crowding distance
    """

    def __init__(
        self,
        fitness_fn: Callable[[Dict], float],
        population_size: int = DEFAULT_POPULATION_SIZE,
        generations: int = DEFAULT_GENERATIONS,
        mutation_rate_max: float = DEFAULT_MUTATION_RATE_MAX,
        mutation_rate_min: float = DEFAULT_MUTATION_RATE_MIN,
        crossover_rate: float = DEFAULT_CROSSOVER_RATE,
        elite_count: int = DEFAULT_ELITE_COUNT,
        niche_radius: float = DEFAULT_NICHE_RADIUS,
        use_fitness_sharing: bool = True,
        crossover_type: str = "multi_point",  # single, multi_point, uniform, blx_alpha
        max_workers: int = 8,
        param_bounds: Dict = None,
        stagnation_limit: int = 3  # Generations without improvement before restart
    ):
        """
        Initialize advanced optimizer.

        Args:
            fitness_fn: Fitness evaluation function
            population_size: Size of population
            generations: Number of generations
            mutation_rate_max: Starting mutation rate
            mutation_rate_min: Ending mutation rate
            crossover_rate: Probability of crossover
            elite_count: Number of elites to preserve
            niche_radius: Radius for fitness sharing
            use_fitness_sharing: Whether to use fitness sharing
            crossover_type: Type of crossover operator
            max_workers: Max parallel workers
            param_bounds: Custom parameter bounds
            stagnation_limit: Generations without improvement before restart
        """
        self.fitness_fn = fitness_fn
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate_max = mutation_rate_max
        self.mutation_rate_min = mutation_rate_min
        self.crossover_rate = crossover_rate
        self.elite_count = elite_count
        self.niche_radius = niche_radius
        self.use_fitness_sharing = use_fitness_sharing
        self.crossover_type = crossover_type
        self.max_workers = max_workers
        self.param_bounds = param_bounds
        self.stagnation_limit = stagnation_limit

        # Tracking
        self.generation_history = []
        self.best_genome = None
        self.best_score = float("-inf")
        self.stagnation_counter = 0

    def _create_random_genome(self) -> Dict:
        """Create a random genome within bounds."""
        bounds = self.param_bounds if self.param_bounds else PARAM_BOUNDS

        genome = {
            "entry": {
                "st_atrPeriod": random.randint(
                    int(bounds.get("entry", {}).get("st_atrPeriod", (8, 16))[0]),
                    int(bounds.get("entry", {}).get("st_atrPeriod", (8, 16))[1])
                ),
                "st_src": "hl2",
                "st_mult": round(random.uniform(
                    bounds.get("entry", {}).get("st_mult", (1.5, 3.0))[0],
                    bounds.get("entry", {}).get("st_mult", (1.5, 3.0))[1]
                ), 2),
                "st_useATR": True,
                "rf_src": "close",
                "rf_period": random.randint(
                    int(bounds.get("entry", {}).get("rf_period", (80, 120))[0]),
                    int(bounds.get("entry", {}).get("rf_period", (80, 120))[1])
                ),
                "rf_mult": round(random.uniform(
                    bounds.get("entry", {}).get("rf_mult", (2.5, 4.0))[0],
                    bounds.get("entry", {}).get("rf_mult", (2.5, 4.0))[1]
                ), 2),
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

    def initialize_population(
        self,
        seed_genomes: List[Dict] = None,
        regime: MarketRegime = None
    ) -> List[Dict]:
        """Initialize population with seeds + regime-aware randoms."""
        population = []

        # Add seed genomes
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
            population.append(self._create_random_genome())

        return population

    def evaluate_population(
        self,
        population: List[Dict],
        progress_cb: Callable = None
    ) -> List[float]:
        """Evaluate fitness of all genomes."""
        scores = []

        for i, genome in enumerate(population):
            try:
                score = self.fitness_fn(genome)
                if score is None or (isinstance(score, float) and np.isnan(score)):
                    score = float("-inf")
                scores.append(float(score))
            except Exception as e:
                logger.warning(f"Fitness evaluation failed for genome {i}: {e}")
                scores.append(float("-inf"))

            if progress_cb and (i + 1) % 10 == 0:
                progress_cb(i + 1, len(population))

        return scores

    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Apply crossover based on configured type."""
        if self.crossover_type == "single":
            return single_point_crossover(parent1, parent2)
        elif self.crossover_type == "multi_point":
            return multi_point_crossover(parent1, parent2, n_points=2)
        elif self.crossover_type == "uniform":
            return uniform_crossover(parent1, parent2)
        elif self.crossover_type == "blx_alpha":
            return blx_alpha_crossover(parent1, parent2, alpha=0.5)
        else:
            return multi_point_crossover(parent1, parent2)

    def evolve_generation(
        self,
        population: List[Dict],
        scores: List[float],
        generation: int
    ) -> List[Dict]:
        """
        Evolve one generation with adaptive operators.
        """
        # Apply fitness sharing if enabled
        if self.use_fitness_sharing:
            selection_scores = calculate_shared_fitness(population, scores, self.niche_radius)
        else:
            selection_scores = scores

        next_gen = []

        # Select diverse elites
        elites = select_diverse_elites(population, scores, self.elite_count)
        next_gen.extend(elites)

        # Fill rest with offspring
        while len(next_gen) < self.population_size:
            # Tournament selection on shared fitness
            parents = tournament_selection(population, selection_scores, 2)

            # Crossover
            if random.random() < self.crossover_rate:
                child = self._crossover(parents[0], parents[1])
            else:
                child = copy.deepcopy(parents[0])

            # Adaptive mutation
            child = adaptive_mutate(
                child,
                generation,
                self.generations,
                self.mutation_rate_max,
                self.mutation_rate_min,
                self.param_bounds
            )

            # Repair and validate
            child = repair_genome(child)

            valid, _ = validate_genome(child)
            if valid:
                next_gen.append(child)

        return next_gen[:self.population_size]

    def _inject_diversity(self, population: List[Dict], n: int) -> List[Dict]:
        """Inject new random individuals to escape local optima."""
        new_pop = population[:len(population) - n]

        for _ in range(n):
            new_pop.append(self._create_random_genome())

        return new_pop

    def optimize(
        self,
        seed_genomes: List[Dict] = None,
        regime: MarketRegime = None,
        progress_cb: Callable = None
    ) -> Tuple[Dict, float, List[Dict]]:
        """
        Run full evolutionary optimization.

        Returns:
            (best_genome, best_score, top_genomes)
        """
        # Initialize
        population = self.initialize_population(seed_genomes, regime)

        logger.info(
            f"Starting GA v2: pop={self.population_size}, gen={self.generations}, "
            f"mutation={self.mutation_rate_max}->{self.mutation_rate_min}, "
            f"fitness_sharing={self.use_fitness_sharing}"
        )

        for gen in range(self.generations):
            # Evaluate
            scores = self.evaluate_population(population)

            # Track best
            gen_best_idx = max(range(len(scores)), key=lambda i: scores[i])
            gen_best_score = scores[gen_best_idx]

            # Check for improvement
            if gen_best_score > self.best_score:
                improvement = gen_best_score - self.best_score
                self.best_score = gen_best_score
                self.best_genome = copy.deepcopy(population[gen_best_idx])
                self.stagnation_counter = 0
                logger.info(f"Gen {gen}: New best score {gen_best_score:.4f} (+{improvement:.4f})")
            else:
                self.stagnation_counter += 1
                logger.info(f"Gen {gen}: No improvement, stagnation={self.stagnation_counter}")

            # Record history
            valid_scores = [s for s in scores if s > float("-inf")]
            self.generation_history.append({
                "generation": gen,
                "best_score": gen_best_score,
                "avg_score": np.mean(valid_scores) if valid_scores else 0,
                "std_score": np.std(valid_scores) if valid_scores else 0,
                "valid_count": len(valid_scores),
                "mutation_rate": self.mutation_rate_max - (self.mutation_rate_max - self.mutation_rate_min) * (gen / self.generations),
                "diversity": self._calculate_diversity(population)
            })

            if progress_cb:
                progress_cb(gen + 1, self.generations, gen_best_score)

            # Stagnation handling
            if self.stagnation_counter >= self.stagnation_limit and gen < self.generations - 1:
                logger.info(f"Stagnation detected, injecting diversity")
                population = self._inject_diversity(population, self.population_size // 3)
                self.stagnation_counter = 0

            # Evolve (except last generation)
            if gen < self.generations - 1:
                population = self.evolve_generation(population, scores, gen)

        # Get top genomes from final population
        final_scores = self.evaluate_population(population)
        sorted_indices = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)
        top_genomes = [population[i] for i in sorted_indices[:20]]

        # Fallback
        if self.best_genome is None and population:
            self.best_genome = population[0]
            self.best_score = final_scores[0] if final_scores else 0

        return self.best_genome, self.best_score, top_genomes

    def _calculate_diversity(self, population: List[Dict]) -> float:
        """Calculate population diversity as average pairwise distance."""
        if len(population) <= 1:
            return 0

        distances = []
        for i in range(min(20, len(population))):  # Sample for speed
            for j in range(i + 1, min(20, len(population))):
                distances.append(calculate_genome_distance(population[i], population[j]))

        return np.mean(distances) if distances else 0


# ═══════════════════════════════════════════════════════════════════════════════
# PHASED OPTIMIZER v2
# ═══════════════════════════════════════════════════════════════════════════════

class PhasedOptimizerV2:
    """
    Phased genome optimization with advanced operators.

    Phase 1: Entry optimization (most parameters)
    Phase 2: SL optimization (locked entry)
    Phase 3: TP optimization (locked entry + SL)
    Phase 4: Mode selection (grid search)
    """

    def __init__(
        self,
        fitness_fn: Callable[[Dict], float],
        generations_per_phase: int = 5,
        population_size: int = 30,
        param_bounds: Dict = None,
        use_fitness_sharing: bool = True
    ):
        self.fitness_fn = fitness_fn
        self.generations_per_phase = generations_per_phase
        self.population_size = population_size
        self.param_bounds = param_bounds
        self.use_fitness_sharing = use_fitness_sharing

    def optimize(
        self,
        seed_genomes: List[Dict] = None,
        regime: MarketRegime = None,
        progress_cb: Callable = None
    ) -> Tuple[Dict, float, List[Dict]]:
        """Run phased optimization."""

        # Start with base genome
        if seed_genomes and len(seed_genomes) > 0:
            base_genome = copy.deepcopy(seed_genomes[0])
        else:
            base_genome = self._create_random_genome()

        all_tested = []

        # Phase 1: Entry
        logger.info("Phase 1: Optimizing Entry genome")
        entry_optimizer = GenomeOptimizerV2(
            fitness_fn=self.fitness_fn,
            population_size=self.population_size,
            generations=self.generations_per_phase,
            param_bounds=self.param_bounds,
            use_fitness_sharing=self.use_fitness_sharing,
            crossover_type="blx_alpha"  # Good for numeric optimization
        )
        best_entry, _, top_entry = entry_optimizer.optimize(seed_genomes, regime)
        all_tested.extend(top_entry)

        if best_entry and "entry" in best_entry:
            base_genome["entry"] = best_entry["entry"]

        # Phase 2: SL
        logger.info("Phase 2: Optimizing SL genome")
        sl_seeds = [self._vary_sl(base_genome) for _ in range(10)]
        sl_optimizer = GenomeOptimizerV2(
            fitness_fn=self.fitness_fn,
            population_size=self.population_size,
            generations=self.generations_per_phase,
            param_bounds=self.param_bounds,
            use_fitness_sharing=self.use_fitness_sharing,
            crossover_type="uniform"
        )
        best_sl, _, top_sl = sl_optimizer.optimize(sl_seeds, regime)
        all_tested.extend(top_sl)

        if best_sl and "sl" in best_sl:
            base_genome["sl"] = best_sl["sl"]

        # Phase 3: TP
        logger.info("Phase 3: Optimizing TP genome")
        tp_seeds = [self._vary_tp(base_genome) for _ in range(10)]
        tp_optimizer = GenomeOptimizerV2(
            fitness_fn=self.fitness_fn,
            population_size=self.population_size,
            generations=self.generations_per_phase,
            param_bounds=self.param_bounds,
            use_fitness_sharing=self.use_fitness_sharing,
            crossover_type="uniform"
        )
        best_tp, _, top_tp = tp_optimizer.optimize(tp_seeds, regime)
        all_tested.extend(top_tp)

        if best_tp:
            if "tp_dual" in best_tp:
                base_genome["tp_dual"] = best_tp["tp_dual"]
            if "tp_rsi" in best_tp:
                base_genome["tp_rsi"] = best_tp["tp_rsi"]

        # Phase 4: Mode (grid search)
        logger.info("Phase 4: Optimizing Mode")
        mode_variants = self._generate_mode_variants(base_genome)
        mode_scores = [self.fitness_fn(g) for g in mode_variants]
        best_mode_idx = max(range(len(mode_scores)), key=lambda i: mode_scores[i])
        base_genome["mode"] = mode_variants[best_mode_idx]["mode"]

        final_score = self.fitness_fn(base_genome)
        all_tested.append(base_genome)

        return base_genome, final_score, all_tested[:20]

    def _create_random_genome(self) -> Dict:
        """Create random genome."""
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

    def _vary_sl(self, genome: Dict) -> Dict:
        """Create SL variant."""
        variant = copy.deepcopy(genome)
        variant["sl"]["st_mult"] = round(random.uniform(2.5, 6.0), 2)
        variant["sl"]["rf_mult"] = round(random.uniform(4.0, 10.0), 2)
        return repair_genome(variant)

    def _vary_tp(self, genome: Dict) -> Dict:
        """Create TP variant."""
        variant = copy.deepcopy(genome)
        variant["tp_dual"]["rr_mult"] = round(random.uniform(0.8, 2.5), 2)
        variant["tp_rsi"]["rr_mult"] = round(random.uniform(0.8, 2.5), 2)
        return repair_genome(variant)

    def _generate_mode_variants(self, genome: Dict) -> List[Dict]:
        """Generate all mode combinations."""
        variants = []
        for dual in [True, False]:
            for rsi in [True, False]:
                if dual or rsi:
                    variant = copy.deepcopy(genome)
                    variant["mode"]["showDualFlip"] = dual
                    variant["mode"]["showRSI"] = rsi
                    variants.append(variant)
        return variants


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def quick_optimize_v2(
    fitness_fn: Callable[[Dict], float],
    seed_genomes: List[Dict] = None,
    regime: MarketRegime = None,
    generations: int = 5,
    population_size: int = 30,
    use_fitness_sharing: bool = True
) -> Tuple[Dict, float]:
    """
    Quick optimization with v2 features.
    """
    optimizer = GenomeOptimizerV2(
        fitness_fn=fitness_fn,
        population_size=population_size,
        generations=generations,
        elite_count=3,
        use_fitness_sharing=use_fitness_sharing
    )
    best_genome, best_score, _ = optimizer.optimize(seed_genomes, regime)
    return best_genome, best_score
