# engine/param_memory_v2.py
"""
ParamMemory v2.0 - Enhanced Long-term Genome Storage

Improvements over v1:
1. Support for Market Profile v2 (32 indicators)
2. Full robustness data storage
3. Extended performance metrics
4. Advanced similarity search using vector embeddings
5. Efficient batch operations with pipelines
6. Memory usage tracking and optimization

Data Structure v2:
{
    "strategy_hash": str,
    "symbol": str,
    "timeframe": str,
    "genome_hash": str,
    "market_profile": {
        "volatility": {...},   # 5 indicators
        "trend": {...},        # 7 indicators
        "momentum": {...},     # 8 indicators
        "volume": {...},       # 5 indicators
        "cycle": {...},        # 5 indicators
        "correlation": {...},  # 2 indicators
        "summary": {...}
    },
    "genome": {
        "entry": {...},
        "sl": {...},
        "tp_dual": {...},
        "tp_rsi": {...},
        "mode": {...}
    },
    "results": {
        "brainScore": float,
        "netProfitPct": float,
        "profitFactor": float,
        "winrate": float,
        "maxDrawdownPct": float,
        "sharpeRatio": float,
        "sortinoRatio": float,
        "totalTrades": int,
        "expectancy": float,
        ...
    },
    "robustness": {
        "stability_score": float,
        "walk_forward_score": float,
        "monte_carlo_score": float,
        "sensitivity_score": float,
        "slippage_score": float,
        "noise_score": float,
        "passed": bool
    },
    "timestamp": int,
    "test_count": int,
    "version": "2.0"
}
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import redis
from redis import ConnectionPool

logger = logging.getLogger(__name__)

# Redis connection
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
POOL_MAX_CONNECTIONS = int(os.getenv("REDIS_POOL_SIZE", 20))

# Memory limits
TOP_GENOMES_LIMIT = int(os.getenv("TOP_GENOMES_LIMIT", 1000))
SIMILARITY_CACHE_TTL = int(os.getenv("SIMILARITY_CACHE_TTL", 3600))  # 1 hour

# Key prefixes (v2)
GENOME_KEY_PREFIX = "genome:v2:"
TOP_GENOMES_KEY_PREFIX = "top_genomes:v2:"
PROFILE_VECTOR_KEY_PREFIX = "profile_vector:v2:"
MARKET_PROFILE_KEY_PREFIX = "market_profile:v2:"
STRATEGY_REGISTRY_PREFIX = "strategy_registry:v2:"

# Connection pool
_connection_pool: Optional[ConnectionPool] = None
_redis_client: Optional[redis.Redis] = None


# ═══════════════════════════════════════════════════════════════════════════════
# CONNECTION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def get_connection_pool() -> ConnectionPool:
    """Get or create Redis connection pool."""
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = ConnectionPool.from_url(
            REDIS_URL,
            max_connections=POOL_MAX_CONNECTIONS,
            decode_responses=True
        )
    return _connection_pool


def get_redis() -> redis.Redis:
    """Get Redis client with connection pooling."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(connection_pool=get_connection_pool())
    return _redis_client


# ═══════════════════════════════════════════════════════════════════════════════
# PROFILE VECTOR CONVERSION (for similarity search)
# ═══════════════════════════════════════════════════════════════════════════════

def profile_to_vector(profile: Dict[str, Any]) -> List[float]:
    """
    Convert Market Profile v2 to normalized vector for similarity calculations.

    Returns 32-element vector (normalized 0-1).
    """
    vector = []

    # Volatility cluster (5)
    vol = profile.get("volatility", {})
    vector.extend([
        min(vol.get("atr_pct", 0) / 5, 1),
        vol.get("vol_regime", 1) / 2,
        vol.get("vol_percentile", 50) / 100,
        vol.get("vol_expanding", 0),
        min(vol.get("vol_ratio", 1) / 3, 1)
    ])

    # Trend cluster (7)
    trend = profile.get("trend", {})
    vector.extend([
        min(trend.get("adx", 20) / 100, 1),
        min(trend.get("plus_di", 25) / 100, 1),
        min(trend.get("minus_di", 25) / 100, 1),
        trend.get("efficiency_ratio", 0.5),
        (trend.get("trend_direction", 0) + 1) / 2,
        trend.get("trend_strength_pct", 0) / 100,
        (trend.get("trend_angle", 0) + 90) / 180
    ])

    # Momentum cluster (8)
    mom = profile.get("momentum", {})
    vector.extend([
        mom.get("rsi_current", 50) / 100,
        mom.get("rsi_mean", 50) / 100,
        min(mom.get("rsi_std", 10) / 30, 1),
        (mom.get("rsi_slope", 0) + 5) / 10,
        mom.get("overbought_pct", 0) / 100,
        mom.get("oversold_pct", 0) / 100,
        min(max((mom.get("macd_histogram", 0) + 0.01) / 0.02, 0), 1),
        (mom.get("macd_trend", 0) + 1) / 2
    ])

    # Volume cluster (5)
    vol_cluster = profile.get("volume", {})
    vector.extend([
        min(vol_cluster.get("volume_ratio", 1) / 3, 1),
        (vol_cluster.get("volume_trend", 0) + 1) / 2,
        (vol_cluster.get("vp_correlation", 0) + 1) / 2,
        vol_cluster.get("volume_climax", 0),
        (vol_cluster.get("obv_trend", 0) + 1) / 2
    ])

    # Cycle cluster (5)
    cycle = profile.get("cycle", {})
    vector.extend([
        cycle.get("cycle_phase", 0) / 3,
        cycle.get("bb_position", 0.5),
        min(cycle.get("bb_width", 0) / 10, 1),
        cycle.get("squeeze", 0),
        min(max((cycle.get("squeeze_momentum", 0) + 5) / 10, 0), 1)
    ])

    # Correlation cluster (2)
    corr = profile.get("correlation", {})
    vector.extend([
        (corr.get("auto_correlation", 0) + 1) / 2,
        (corr.get("btc_correlation", 0) + 1) / 2
    ])

    # Ensure 32 elements
    while len(vector) < 32:
        vector.append(0.5)

    return vector[:32]


def profile_from_simple(simple_profile: Dict[str, float]) -> Dict[str, Any]:
    """
    Convert simple 5-indicator profile to v2 format for backward compatibility.

    Args:
        simple_profile: Old format with atr_pct, adx, volatility, trend_ratio, rsi_mean

    Returns:
        V2 profile format (partial, with defaults)
    """
    return {
        "volatility": {
            "atr_pct": simple_profile.get("atr_pct", 0),
            "vol_regime": 1,
            "vol_percentile": simple_profile.get("volatility", 0.5) * 100,
            "vol_expanding": 0,
            "vol_ratio": 1.0
        },
        "trend": {
            "adx": simple_profile.get("adx", 20),
            "plus_di": 25,
            "minus_di": 25,
            "efficiency_ratio": 0.5,
            "trend_direction": 1 if simple_profile.get("trend_ratio", 0.5) > 0.5 else -1,
            "trend_strength_pct": abs(simple_profile.get("trend_ratio", 0.5) - 0.5) * 100,
            "trend_angle": 0
        },
        "momentum": {
            "rsi_current": simple_profile.get("rsi_mean", 50),
            "rsi_mean": simple_profile.get("rsi_mean", 50),
            "rsi_std": 10,
            "rsi_slope": 0,
            "overbought_pct": 0,
            "oversold_pct": 0,
            "macd_histogram": 0,
            "macd_trend": 0
        },
        "volume": {
            "volume_ratio": 1.0,
            "volume_trend": 0,
            "vp_correlation": 0,
            "volume_climax": 0,
            "obv_trend": 0
        },
        "cycle": {
            "cycle_phase": 0,
            "bb_position": 0.5,
            "bb_width": 0,
            "squeeze": 0,
            "squeeze_momentum": 0
        },
        "correlation": {
            "auto_correlation": 0,
            "btc_correlation": 0
        },
        "summary": {
            "regime": "ranging",
            "trend_score": 50,
            "volatility_score": 50,
            "momentum_score": 0,
            "market_condition": "neutral"
        }
    }


def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    arr1 = np.array(vec1)
    arr2 = np.array(vec2)

    dot_product = np.dot(arr1, arr2)
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


# ═══════════════════════════════════════════════════════════════════════════════
# GENOME STORAGE v2
# ═══════════════════════════════════════════════════════════════════════════════

def store_genome_result_v2(record: Dict[str, Any]) -> bool:
    """
    Store a genome result with v2 schema.

    Args:
        record: Full genome record with v2 fields

    Returns:
        True if stored successfully
    """
    try:
        r = get_redis()

        strategy_hash = record["strategy_hash"]
        symbol = record["symbol"]
        timeframe = record["timeframe"]
        genome_hash = record["genome_hash"]

        # Get scores for ranking
        brain_score = record.get("results", {}).get("brainScore", 0)
        stability_score = record.get("robustness", {}).get("stability_score", 0)

        # Combined score for ranking (70% brain, 30% robustness)
        combined_score = brain_score * 0.7 + stability_score * 30 * 0.3

        # Ensure version tag
        record["version"] = "2.0"

        # Use pipeline for atomic operations
        pipe = r.pipeline()

        # Main genome key (permanent)
        key = f"{GENOME_KEY_PREFIX}{strategy_hash}:{symbol}:{timeframe}:{genome_hash}"
        pipe.set(key, json.dumps(record))

        # Add to sorted set (combined score)
        sorted_key = f"{TOP_GENOMES_KEY_PREFIX}{strategy_hash}:{symbol}:{timeframe}"
        pipe.zadd(sorted_key, {genome_hash: combined_score})

        # Store profile vector for similarity search
        market_profile = record.get("market_profile", {})
        vector = profile_to_vector(market_profile)
        vector_key = f"{PROFILE_VECTOR_KEY_PREFIX}{strategy_hash}:{symbol}:{timeframe}:{genome_hash}"
        pipe.set(vector_key, json.dumps(vector))

        # Trim sorted set to limit
        pipe.zremrangebyrank(sorted_key, 0, -(TOP_GENOMES_LIMIT + 1))

        pipe.execute()
        return True

    except Exception as e:
        logger.error(f"Failed to store genome v2: {e}")
        return False


def get_genome_v2(
    strategy_hash: str,
    symbol: str,
    timeframe: str,
    genome_hash: str
) -> Optional[Dict[str, Any]]:
    """Retrieve a specific genome record."""
    try:
        r = get_redis()

        # Try v2 key first
        key = f"{GENOME_KEY_PREFIX}{strategy_hash}:{symbol}:{timeframe}:{genome_hash}"
        data = r.get(key)

        if data:
            return json.loads(data)

        # Fallback to v1 key for backward compatibility
        v1_key = f"genome:{strategy_hash}:{symbol}:{timeframe}:{genome_hash}"
        data = r.get(v1_key)

        if data:
            return json.loads(data)

        return None

    except Exception as e:
        logger.error(f"Failed to get genome v2: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# TOP PERFORMERS RETRIEVAL v2
# ═══════════════════════════════════════════════════════════════════════════════

def get_top_genomes_v2(
    strategy_hash: str,
    symbol: str,
    timeframe: str,
    limit: int = 50,
    min_robustness: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Get top-performing genomes with optional robustness filter.

    Args:
        strategy_hash: Strategy hash
        symbol: Trading symbol
        timeframe: Timeframe
        limit: Maximum genomes to return
        min_robustness: Minimum robustness score (0-1)

    Returns:
        List of genome records sorted by combined score
    """
    try:
        r = get_redis()

        # Try v2 key first
        sorted_key = f"{TOP_GENOMES_KEY_PREFIX}{strategy_hash}:{symbol}:{timeframe}"
        top_hashes = r.zrevrange(sorted_key, 0, limit * 2 - 1)  # Get extra for filtering

        if not top_hashes:
            # Fallback to v1
            sorted_key = f"top_genomes:{strategy_hash}:{symbol}:{timeframe}"
            top_hashes = r.zrevrange(sorted_key, 0, limit * 2 - 1)

        # Fetch records with filtering
        results = []
        for genome_hash in top_hashes:
            record = get_genome_v2(strategy_hash, symbol, timeframe, genome_hash)
            if record:
                robustness = record.get("robustness", {}).get("stability_score", 1.0)
                if robustness >= min_robustness:
                    results.append(record)
                    if len(results) >= limit:
                        break

        return results

    except Exception as e:
        logger.error(f"Failed to get top genomes v2: {e}")
        return []


def get_top_robust_genomes(
    strategy_hash: str,
    symbol: str,
    timeframe: str,
    limit: int = 20,
    min_stability: float = 0.6
) -> List[Dict[str, Any]]:
    """
    Get top genomes filtered by robustness.

    Only returns genomes that passed robustness testing.
    """
    genomes = get_top_genomes_v2(
        strategy_hash, symbol, timeframe,
        limit=limit * 2,  # Get extra for filtering
        min_robustness=min_stability
    )

    # Filter by robustness passed flag
    robust_genomes = [
        g for g in genomes
        if g.get("robustness", {}).get("passed", False)
    ]

    return robust_genomes[:limit]


# ═══════════════════════════════════════════════════════════════════════════════
# SIMILARITY SEARCH v2
# ═══════════════════════════════════════════════════════════════════════════════

def query_similar_genomes_v2(
    strategy_hash: str,
    symbol: str,
    timeframe: str,
    market_profile: Dict[str, Any],
    top_n: int = 20,
    similarity_threshold: float = 0.7,
    min_robustness: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Find genomes with similar market profiles using v2 vectors.

    Uses cosine similarity on 32-dimensional profile vectors.

    Args:
        strategy_hash: Strategy hash
        symbol: Trading symbol
        timeframe: Timeframe
        market_profile: Current market profile (v2 format)
        top_n: Number of similar genomes to return
        similarity_threshold: Minimum similarity score (0-1)
        min_robustness: Minimum robustness score

    Returns:
        List of similar genome records
    """
    try:
        r = get_redis()

        # Convert current profile to vector
        current_vector = profile_to_vector(market_profile)

        # Get all genome hashes for this symbol/timeframe
        sorted_key = f"{TOP_GENOMES_KEY_PREFIX}{strategy_hash}:{symbol}:{timeframe}"
        all_hashes = r.zrevrange(sorted_key, 0, -1)

        if not all_hashes:
            # Fallback to v1
            sorted_key = f"top_genomes:{strategy_hash}:{symbol}:{timeframe}"
            all_hashes = r.zrevrange(sorted_key, 0, -1)

        if not all_hashes:
            return []

        # Calculate similarity for each
        candidates = []

        for genome_hash in all_hashes:
            # Get stored vector
            vector_key = f"{PROFILE_VECTOR_KEY_PREFIX}{strategy_hash}:{symbol}:{timeframe}:{genome_hash}"
            stored_vector_data = r.get(vector_key)

            if stored_vector_data:
                stored_vector = json.loads(stored_vector_data)
            else:
                # Fallback: calculate from stored profile
                record = get_genome_v2(strategy_hash, symbol, timeframe, genome_hash)
                if not record:
                    continue

                stored_profile = record.get("market_profile", {})

                # Handle old format
                if "volatility" not in stored_profile and "atr_pct" in stored_profile:
                    stored_profile = profile_from_simple(stored_profile)

                stored_vector = profile_to_vector(stored_profile)

            # Calculate similarity
            similarity = calculate_cosine_similarity(current_vector, stored_vector)

            if similarity >= similarity_threshold:
                candidates.append((genome_hash, similarity))

        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Fetch records
        results = []
        for genome_hash, similarity in candidates[:top_n * 2]:
            record = get_genome_v2(strategy_hash, symbol, timeframe, genome_hash)
            if record:
                robustness = record.get("robustness", {}).get("stability_score", 1.0)
                if robustness >= min_robustness:
                    record["_similarity"] = round(similarity, 4)
                    results.append(record)
                    if len(results) >= top_n:
                        break

        return results

    except Exception as e:
        logger.error(f"Failed to query similar genomes v2: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH OPERATIONS v2
# ═══════════════════════════════════════════════════════════════════════════════

def store_batch_genomes_v2(records: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    Store multiple genome records in batch using pipeline.

    Args:
        records: List of genome records

    Returns:
        (stored_count, failed_count)
    """
    stored = 0
    failed = 0

    try:
        r = get_redis()
        pipe = r.pipeline()

        for record in records:
            try:
                strategy_hash = record["strategy_hash"]
                symbol = record["symbol"]
                timeframe = record["timeframe"]
                genome_hash = record["genome_hash"]

                brain_score = record.get("results", {}).get("brainScore", 0)
                stability_score = record.get("robustness", {}).get("stability_score", 0)
                combined_score = brain_score * 0.7 + stability_score * 30 * 0.3

                record["version"] = "2.0"

                # Queue operations
                key = f"{GENOME_KEY_PREFIX}{strategy_hash}:{symbol}:{timeframe}:{genome_hash}"
                pipe.set(key, json.dumps(record))

                sorted_key = f"{TOP_GENOMES_KEY_PREFIX}{strategy_hash}:{symbol}:{timeframe}"
                pipe.zadd(sorted_key, {genome_hash: combined_score})

                market_profile = record.get("market_profile", {})
                vector = profile_to_vector(market_profile)
                vector_key = f"{PROFILE_VECTOR_KEY_PREFIX}{strategy_hash}:{symbol}:{timeframe}:{genome_hash}"
                pipe.set(vector_key, json.dumps(vector))

                stored += 1

            except Exception as e:
                logger.warning(f"Failed to queue genome: {e}")
                failed += 1

        # Execute all operations
        pipe.execute()

    except Exception as e:
        logger.error(f"Batch store failed: {e}")
        failed = len(records) - stored

    return stored, failed


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICS & MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def get_memory_stats_v2(strategy_hash: str) -> Dict[str, Any]:
    """
    Get memory statistics for a strategy (v2 enhanced).

    Returns:
        {
            "total_genomes": int,
            "robust_genomes": int,
            "symbols": {...},
            "timeframes": {...},
            "avg_brain_score": float,
            "avg_stability": float,
            "storage_size_mb": float
        }
    """
    try:
        r = get_redis()

        # Scan both v1 and v2 keys
        patterns = [
            f"{GENOME_KEY_PREFIX}{strategy_hash}:*",
            f"genome:{strategy_hash}:*"  # v1 fallback
        ]

        total = 0
        robust = 0
        symbols = defaultdict(int)
        timeframes = defaultdict(int)
        brain_scores = []
        stability_scores = []
        total_size = 0

        for pattern in patterns:
            cursor = 0
            while True:
                cursor, keys = r.scan(cursor, match=pattern, count=100)

                for key in keys:
                    data = r.get(key)
                    if data:
                        total += 1
                        total_size += len(data)

                        try:
                            record = json.loads(data)
                            parts = key.split(":")

                            # Extract symbol/timeframe
                            if len(parts) >= 4:
                                # Handle both v1 and v2 key formats
                                if "v2" in key:
                                    symbols[parts[3]] += 1
                                    timeframes[parts[4]] += 1
                                else:
                                    symbols[parts[2]] += 1
                                    timeframes[parts[3]] += 1

                            # Stats
                            brain_score = record.get("results", {}).get("brainScore", 0)
                            if brain_score:
                                brain_scores.append(brain_score)

                            robustness = record.get("robustness", {})
                            if robustness.get("passed", False):
                                robust += 1
                            stability = robustness.get("stability_score", 0)
                            if stability:
                                stability_scores.append(stability)

                        except json.JSONDecodeError:
                            pass

                if cursor == 0:
                    break

        return {
            "total_genomes": total,
            "robust_genomes": robust,
            "symbols": dict(symbols),
            "timeframes": dict(timeframes),
            "avg_brain_score": round(np.mean(brain_scores), 4) if brain_scores else 0,
            "avg_stability": round(np.mean(stability_scores), 4) if stability_scores else 0,
            "storage_size_mb": round(total_size / 1024 / 1024, 2)
        }

    except Exception as e:
        logger.error(f"Failed to get memory stats v2: {e}")
        return {"total_genomes": 0, "symbols": {}, "timeframes": {}}


def get_best_genome_for_conditions(
    strategy_hash: str,
    symbol: str,
    timeframe: str,
    market_profile: Dict[str, Any],
    prioritize_robustness: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Get the single best genome for current market conditions.

    Args:
        strategy_hash: Strategy hash
        symbol: Trading symbol
        timeframe: Timeframe
        market_profile: Current market profile
        prioritize_robustness: If True, prefer robust genomes

    Returns:
        Best matching genome or None
    """
    # Get similar genomes
    similar = query_similar_genomes_v2(
        strategy_hash, symbol, timeframe,
        market_profile,
        top_n=10,
        similarity_threshold=0.5,
        min_robustness=0.5 if prioritize_robustness else 0.0
    )

    if not similar:
        # Fallback to top performer
        top = get_top_genomes_v2(strategy_hash, symbol, timeframe, limit=1)
        return top[0] if top else None

    # Score by similarity * brain_score * stability
    def combined_score(genome):
        sim = genome.get("_similarity", 0.5)
        brain = genome.get("results", {}).get("brainScore", 0)
        stability = genome.get("robustness", {}).get("stability_score", 0.5)
        return sim * brain * (1 + stability * 0.5)

    similar.sort(key=combined_score, reverse=True)
    return similar[0]


def migrate_v1_to_v2(strategy_hash: str) -> Tuple[int, int]:
    """
    Migrate v1 genome records to v2 format.

    Returns:
        (migrated_count, failed_count)
    """
    try:
        r = get_redis()
        pattern = f"genome:{strategy_hash}:*"

        migrated = 0
        failed = 0

        cursor = 0
        while True:
            cursor, keys = r.scan(cursor, match=pattern, count=100)

            for key in keys:
                # Skip if already v2
                if ":v2:" in key:
                    continue

                try:
                    data = r.get(key)
                    if not data:
                        continue

                    record = json.loads(data)

                    # Upgrade market profile
                    market_profile = record.get("market_profile", {})
                    if "volatility" not in market_profile and "atr_pct" in market_profile:
                        record["market_profile"] = profile_from_simple(market_profile)

                    # Add empty robustness if missing
                    if "robustness" not in record:
                        record["robustness"] = {
                            "stability_score": 0,
                            "passed": False
                        }

                    # Ensure results has brainScore
                    results = record.get("results", {})
                    if "brainScore" not in results and "score" in results:
                        results["brainScore"] = results["score"]
                        record["results"] = results

                    # Store as v2
                    if store_genome_result_v2(record):
                        migrated += 1
                    else:
                        failed += 1

                except Exception as e:
                    logger.warning(f"Failed to migrate {key}: {e}")
                    failed += 1

            if cursor == 0:
                break

        logger.info(f"Migration complete: {migrated} migrated, {failed} failed")
        return migrated, failed

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return 0, 0


def clear_strategy_memory_v2(strategy_hash: str) -> int:
    """
    Clear all genomes for a strategy (v2 keys).

    Returns:
        Number of keys deleted
    """
    try:
        r = get_redis()
        patterns = [
            f"{GENOME_KEY_PREFIX}{strategy_hash}:*",
            f"{TOP_GENOMES_KEY_PREFIX}{strategy_hash}:*",
            f"{PROFILE_VECTOR_KEY_PREFIX}{strategy_hash}:*"
        ]

        deleted = 0

        for pattern in patterns:
            cursor = 0
            while True:
                cursor, keys = r.scan(cursor, match=pattern, count=100)
                if keys:
                    deleted += r.delete(*keys)
                if cursor == 0:
                    break

        logger.info(f"Cleared {deleted} v2 keys for strategy {strategy_hash}")
        return deleted

    except Exception as e:
        logger.error(f"Failed to clear strategy memory v2: {e}")
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

# Re-export v1 functions with same names for backward compatibility
def store_genome_result(record: Dict[str, Any]) -> bool:
    """Store genome (v2-compatible wrapper)."""
    return store_genome_result_v2(record)


def get_genome(strategy_hash: str, symbol: str, timeframe: str, genome_hash: str) -> Optional[Dict[str, Any]]:
    """Get genome (v2-compatible wrapper)."""
    return get_genome_v2(strategy_hash, symbol, timeframe, genome_hash)


def get_top_genomes_by_score(strategy_hash: str, symbol: str, timeframe: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Get top genomes (v2-compatible wrapper)."""
    return get_top_genomes_v2(strategy_hash, symbol, timeframe, limit)


def query_similar_genomes(
    strategy_hash: str,
    symbol: str,
    timeframe: str,
    market_profile: Dict[str, float],
    top_n: int = 20,
    similarity_threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """Query similar genomes (v2-compatible wrapper)."""
    # Convert simple profile to v2 if needed
    if "volatility" not in market_profile:
        market_profile = profile_from_simple(market_profile)

    return query_similar_genomes_v2(
        strategy_hash, symbol, timeframe,
        market_profile, top_n, similarity_threshold
    )


def get_memory_stats(strategy_hash: str) -> Dict[str, Any]:
    """Get memory stats (v2-compatible wrapper)."""
    return get_memory_stats_v2(strategy_hash)
