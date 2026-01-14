# engine/param_memory.py
"""
ParamMemory - Long-term genome storage for Quant AI Brain

Redis-based persistent storage for genome records with:
- Strategy hash isolation
- Market profile matching
- Score-based ranking
- Permanent storage (no TTL)
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict

import redis
from redis import ConnectionPool

logger = logging.getLogger(__name__)

# Redis connection
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
POOL_MAX_CONNECTIONS = int(os.getenv("REDIS_POOL_SIZE", 20))

# Memory limits
TOP_GENOMES_LIMIT = int(os.getenv("TOP_GENOMES_LIMIT", 1000))

# Key prefixes
GENOME_KEY_PREFIX = "genome:"
TOP_GENOMES_KEY_PREFIX = "top_genomes:"
MARKET_PROFILE_KEY_PREFIX = "market_profile:"
STRATEGY_REGISTRY_PREFIX = "strategy_registry:"

# Connection pool (shared)
_connection_pool: Optional[ConnectionPool] = None
_redis_client: Optional[redis.Redis] = None


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


# ═══════════════════════════════════════════════════════
# GENOME STORAGE
# ═══════════════════════════════════════════════════════

def store_genome_result(record: Dict[str, Any]) -> bool:
    """
    Store a genome result in Redis (permanent, no TTL).

    Args:
        record: {
            "strategy_hash": str,
            "symbol": str,
            "timeframe": str,
            "genome_hash": str,
            "market_profile": {...},
            "genome": {...},
            "results": {...},
            "timestamp": int,
            "test_count": int
        }

    Returns:
        True if stored successfully
    """
    try:
        r = get_redis()

        strategy_hash = record["strategy_hash"]
        symbol = record["symbol"]
        timeframe = record["timeframe"]
        genome_hash = record["genome_hash"]
        score = record.get("results", {}).get("score", 0)

        # Main genome key (permanent)
        key = f"{GENOME_KEY_PREFIX}{strategy_hash}:{symbol}:{timeframe}:{genome_hash}"
        r.set(key, json.dumps(record))

        # Add to sorted set for top performers
        sorted_key = f"{TOP_GENOMES_KEY_PREFIX}{strategy_hash}:{symbol}:{timeframe}"
        r.zadd(sorted_key, {genome_hash: score})

        # Trim sorted set to limit
        r.zremrangebyrank(sorted_key, 0, -(TOP_GENOMES_LIMIT + 1))

        return True

    except Exception as e:
        logger.error(f"Failed to store genome: {e}")
        return False


def get_genome(
    strategy_hash: str,
    symbol: str,
    timeframe: str,
    genome_hash: str
) -> Optional[Dict[str, Any]]:
    """
    Retrieve a specific genome record.

    Returns:
        Genome record or None if not found
    """
    try:
        r = get_redis()
        key = f"{GENOME_KEY_PREFIX}{strategy_hash}:{symbol}:{timeframe}:{genome_hash}"
        data = r.get(key)
        if data:
            return json.loads(data)
        return None
    except Exception as e:
        logger.error(f"Failed to get genome: {e}")
        return None


def update_genome_test_count(
    strategy_hash: str,
    symbol: str,
    timeframe: str,
    genome_hash: str
) -> bool:
    """
    Increment test_count for an existing genome.

    Returns:
        True if updated successfully
    """
    try:
        record = get_genome(strategy_hash, symbol, timeframe, genome_hash)
        if record:
            record["test_count"] = record.get("test_count", 0) + 1
            record["timestamp"] = int(time.time())
            return store_genome_result(record)
        return False
    except Exception as e:
        logger.error(f"Failed to update genome test count: {e}")
        return False


# ═══════════════════════════════════════════════════════
# TOP PERFORMERS RETRIEVAL
# ═══════════════════════════════════════════════════════

def get_top_genomes_by_score(
    strategy_hash: str,
    symbol: str,
    timeframe: str,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Get top-performing genomes for a symbol/timeframe.

    Args:
        strategy_hash: Strategy hash
        symbol: Trading symbol
        timeframe: Timeframe
        limit: Maximum number of genomes to return

    Returns:
        List of genome records sorted by score (descending)
    """
    try:
        r = get_redis()
        sorted_key = f"{TOP_GENOMES_KEY_PREFIX}{strategy_hash}:{symbol}:{timeframe}"

        # Get top genome hashes by score (descending)
        top_hashes = r.zrevrange(sorted_key, 0, limit - 1)

        # Fetch full records
        results = []
        for genome_hash in top_hashes:
            record = get_genome(strategy_hash, symbol, timeframe, genome_hash)
            if record:
                results.append(record)

        return results

    except Exception as e:
        logger.error(f"Failed to get top genomes: {e}")
        return []


def get_all_top_genomes(
    strategy_hash: str,
    symbols: List[str],
    timeframes: List[str],
    limit_per_combo: int = 20
) -> List[Dict[str, Any]]:
    """
    Get top genomes across multiple symbols/timeframes.

    Returns:
        List of genome records
    """
    all_genomes = []

    for symbol in symbols:
        for tf in timeframes:
            genomes = get_top_genomes_by_score(
                strategy_hash, symbol, tf, limit_per_combo
            )
            all_genomes.extend(genomes)

    # Sort all by score and dedupe
    seen = set()
    unique_genomes = []
    for g in sorted(all_genomes, key=lambda x: x.get("results", {}).get("score", 0), reverse=True):
        gh = g.get("genome_hash")
        if gh not in seen:
            seen.add(gh)
            unique_genomes.append(g)

    return unique_genomes


# ═══════════════════════════════════════════════════════
# SIMILAR GENOME SEARCH
# ═══════════════════════════════════════════════════════

def query_similar_genomes(
    strategy_hash: str,
    symbol: str,
    timeframe: str,
    market_profile: Dict[str, float],
    top_n: int = 20,
    similarity_threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Find genomes with similar market profiles.

    Uses cosine similarity on market profile vectors.

    Args:
        strategy_hash: Strategy hash
        symbol: Trading symbol
        timeframe: Timeframe
        market_profile: Current market profile
        top_n: Number of similar genomes to return
        similarity_threshold: Minimum similarity score (0-1)

    Returns:
        List of similar genome records
    """
    try:
        # Get all genomes for this symbol/timeframe
        all_genomes = get_top_genomes_by_score(
            strategy_hash, symbol, timeframe, limit=200
        )

        if not all_genomes:
            return []

        # Calculate similarity scores
        def profile_similarity(stored_profile: Dict, current_profile: Dict) -> float:
            """Calculate similarity between two market profiles."""
            keys = ["atr_pct", "adx", "volatility", "trend_ratio", "rsi_mean"]

            stored_vec = [stored_profile.get(k, 0) for k in keys]
            current_vec = [current_profile.get(k, 0) for k in keys]

            # Normalize vectors
            def normalize(vec):
                mag = sum(v**2 for v in vec) ** 0.5
                return [v / mag if mag > 0 else 0 for v in vec]

            stored_norm = normalize(stored_vec)
            current_norm = normalize(current_vec)

            # Cosine similarity
            similarity = sum(a * b for a, b in zip(stored_norm, current_norm))
            return max(0, min(1, similarity))

        # Score all genomes
        scored_genomes = []
        for genome in all_genomes:
            stored_profile = genome.get("market_profile", {})
            sim_score = profile_similarity(stored_profile, market_profile)

            if sim_score >= similarity_threshold:
                genome["_similarity"] = sim_score
                scored_genomes.append(genome)

        # Sort by similarity * performance score
        scored_genomes.sort(
            key=lambda g: g.get("_similarity", 0) * g.get("results", {}).get("score", 0),
            reverse=True
        )

        return scored_genomes[:top_n]

    except Exception as e:
        logger.error(f"Failed to query similar genomes: {e}")
        return []


# ═══════════════════════════════════════════════════════
# MARKET PROFILE CACHE
# ═══════════════════════════════════════════════════════

def cache_market_profile(
    symbol: str,
    timeframe: str,
    date: str,
    profile: Dict[str, float],
    ttl: int = 86400  # 1 day
) -> bool:
    """
    Cache market profile (with TTL for freshness).

    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        date: Date string (YYYY-MM-DD)
        profile: Market profile data
        ttl: Time-to-live in seconds

    Returns:
        True if cached successfully
    """
    try:
        r = get_redis()
        key = f"{MARKET_PROFILE_KEY_PREFIX}{symbol}:{timeframe}:{date}"
        r.setex(key, ttl, json.dumps(profile))
        return True
    except Exception as e:
        logger.error(f"Failed to cache market profile: {e}")
        return False


def get_cached_market_profile(
    symbol: str,
    timeframe: str,
    date: str
) -> Optional[Dict[str, float]]:
    """
    Get cached market profile.

    Returns:
        Market profile or None if not cached/expired
    """
    try:
        r = get_redis()
        key = f"{MARKET_PROFILE_KEY_PREFIX}{symbol}:{timeframe}:{date}"
        data = r.get(key)
        if data:
            return json.loads(data)
        return None
    except Exception as e:
        logger.error(f"Failed to get cached market profile: {e}")
        return None


# ═══════════════════════════════════════════════════════
# STRATEGY REGISTRY
# ═══════════════════════════════════════════════════════

def register_strategy(
    strategy_hash: str,
    strategy_type: str,
    engine_version: str
) -> bool:
    """
    Register a strategy hash in the registry.

    Args:
        strategy_hash: Generated hash
        strategy_type: Strategy identifier
        engine_version: Engine version

    Returns:
        True if registered successfully
    """
    try:
        r = get_redis()
        key = f"{STRATEGY_REGISTRY_PREFIX}{strategy_hash}"

        record = {
            "type": strategy_type,
            "version": engine_version,
            "created": int(time.time())
        }
        r.set(key, json.dumps(record))
        return True
    except Exception as e:
        logger.error(f"Failed to register strategy: {e}")
        return False


def get_strategy_info(strategy_hash: str) -> Optional[Dict[str, Any]]:
    """
    Get strategy info from registry.

    Returns:
        Strategy info or None if not found
    """
    try:
        r = get_redis()
        key = f"{STRATEGY_REGISTRY_PREFIX}{strategy_hash}"
        data = r.get(key)
        if data:
            return json.loads(data)
        return None
    except Exception as e:
        logger.error(f"Failed to get strategy info: {e}")
        return None


# ═══════════════════════════════════════════════════════
# BATCH OPERATIONS
# ═══════════════════════════════════════════════════════

def store_batch_genomes(records: List[Dict[str, Any]]) -> int:
    """
    Store multiple genome records in batch.

    Args:
        records: List of genome records

    Returns:
        Number of records stored successfully
    """
    stored = 0
    for record in records:
        if store_genome_result(record):
            stored += 1
    return stored


def get_memory_stats(strategy_hash: str) -> Dict[str, Any]:
    """
    Get memory statistics for a strategy.

    Returns:
        {
            "total_genomes": int,
            "symbols": {...},
            "timeframes": {...}
        }
    """
    try:
        r = get_redis()
        pattern = f"{GENOME_KEY_PREFIX}{strategy_hash}:*"

        # Count keys
        cursor = 0
        total = 0
        symbols = defaultdict(int)
        timeframes = defaultdict(int)

        while True:
            cursor, keys = r.scan(cursor, match=pattern, count=100)
            for key in keys:
                total += 1
                parts = key.split(":")
                if len(parts) >= 4:
                    symbols[parts[2]] += 1
                    timeframes[parts[3]] += 1

            if cursor == 0:
                break

        return {
            "total_genomes": total,
            "symbols": dict(symbols),
            "timeframes": dict(timeframes)
        }

    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}")
        return {"total_genomes": 0, "symbols": {}, "timeframes": {}}


def clear_strategy_memory(strategy_hash: str) -> int:
    """
    Clear all genomes for a strategy (manual cleanup).

    Args:
        strategy_hash: Strategy hash to clear

    Returns:
        Number of keys deleted
    """
    try:
        r = get_redis()
        pattern = f"{GENOME_KEY_PREFIX}{strategy_hash}:*"
        top_pattern = f"{TOP_GENOMES_KEY_PREFIX}{strategy_hash}:*"

        deleted = 0

        # Delete genome keys
        cursor = 0
        while True:
            cursor, keys = r.scan(cursor, match=pattern, count=100)
            if keys:
                deleted += r.delete(*keys)
            if cursor == 0:
                break

        # Delete sorted set keys
        cursor = 0
        while True:
            cursor, keys = r.scan(cursor, match=top_pattern, count=100)
            if keys:
                deleted += r.delete(*keys)
            if cursor == 0:
                break

        logger.info(f"Cleared {deleted} keys for strategy {strategy_hash}")
        return deleted

    except Exception as e:
        logger.error(f"Failed to clear strategy memory: {e}")
        return 0
