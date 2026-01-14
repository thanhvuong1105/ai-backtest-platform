# engine/indicator_cache.py
"""
Cached indicator calculations for performance optimization.

Uses LRU cache to avoid recalculating indicators for the same data.
Cache key: (data_hash, indicator_type, period)
"""

import os
import hashlib
from functools import lru_cache
from typing import Tuple, Optional
import pandas as pd
import numpy as np

# Cache size from environment or default
CACHE_SIZE = int(os.getenv("INDICATOR_CACHE_SIZE", 256))


def _hash_series(series: pd.Series) -> str:
    """Create a hash of series data for cache key."""
    # Use first, last values and length for quick hash
    if len(series) == 0:
        return "empty"
    data_str = f"{len(series)}_{series.iloc[0]:.6f}_{series.iloc[-1]:.6f}_{series.iloc[len(series)//2]:.6f}"
    return hashlib.md5(data_str.encode()).hexdigest()[:16]


def _series_to_tuple(series: pd.Series) -> Tuple:
    """Convert series to hashable tuple for cache key."""
    # Only use key points to create hash (faster than full conversion)
    if len(series) == 0:
        return ()
    step = max(1, len(series) // 100)  # Sample ~100 points
    return tuple(series.iloc[::step].round(6).tolist())


class IndicatorCache:
    """
    Thread-safe indicator cache using LRU.

    Caches computed indicators to avoid recalculation when running
    multiple backtests on the same data with same parameters.
    """

    def __init__(self, maxsize: int = CACHE_SIZE):
        self.maxsize = maxsize
        self._cache = {}
        self._hits = 0
        self._misses = 0

    def _make_key(self, data_id: str, indicator: str, *params) -> str:
        """Create cache key from data identifier and parameters."""
        return f"{data_id}:{indicator}:{':'.join(str(p) for p in params)}"

    def get(self, key: str) -> Optional[np.ndarray]:
        """Get cached result."""
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def set(self, key: str, value: np.ndarray) -> None:
        """Set cached result with LRU eviction."""
        if len(self._cache) >= self.maxsize:
            # Remove oldest entry (simple FIFO, not true LRU but fast)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = value

    def stats(self) -> dict:
        """Return cache statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "size": len(self._cache),
            "maxsize": self.maxsize
        }

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0


# Global cache instance
_indicator_cache = IndicatorCache()


def get_cache() -> IndicatorCache:
    """Get the global indicator cache instance."""
    return _indicator_cache


# Cached indicator functions
@lru_cache(maxsize=CACHE_SIZE)
def _ema_cached(data_tuple: Tuple, length: int) -> Tuple:
    """Cached EMA calculation."""
    series = pd.Series(data_tuple)
    result = series.ewm(span=length, adjust=False).mean()
    return tuple(result.values)


@lru_cache(maxsize=CACHE_SIZE)
def _rsi_cached(data_tuple: Tuple, length: int) -> Tuple:
    """Cached RSI calculation."""
    series = pd.Series(data_tuple)
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    gain = pd.Series(gain).rolling(length).mean()
    loss = pd.Series(loss).rolling(length).mean()

    rs = gain / loss
    result = 100 - (100 / (1 + rs))
    return tuple(result.values)


@lru_cache(maxsize=CACHE_SIZE)
def _atr_cached(high_tuple: Tuple, low_tuple: Tuple, close_tuple: Tuple, length: int) -> Tuple:
    """Cached ATR calculation."""
    high = pd.Series(high_tuple)
    low = pd.Series(low_tuple)
    close = pd.Series(close_tuple)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    result = tr.ewm(span=length, adjust=False).mean()
    return tuple(result.values)


def ema_cached(series: pd.Series, length: int) -> pd.Series:
    """
    Calculate EMA with caching.

    Args:
        series: Price series
        length: EMA period

    Returns:
        EMA series
    """
    data_tuple = tuple(series.values)
    result_tuple = _ema_cached(data_tuple, length)
    return pd.Series(result_tuple, index=series.index)


def rsi_cached(series: pd.Series, length: int = 14) -> pd.Series:
    """
    Calculate RSI with caching.

    Args:
        series: Price series
        length: RSI period

    Returns:
        RSI series
    """
    data_tuple = tuple(series.values)
    result_tuple = _rsi_cached(data_tuple, length)
    return pd.Series(result_tuple, index=series.index)


def atr_cached(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """
    Calculate ATR with caching.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        length: ATR period

    Returns:
        ATR series
    """
    result_tuple = _atr_cached(
        tuple(high.values),
        tuple(low.values),
        tuple(close.values),
        length
    )
    return pd.Series(result_tuple, index=close.index)


def clear_indicator_cache():
    """Clear all indicator caches."""
    _ema_cached.cache_clear()
    _rsi_cached.cache_clear()
    _atr_cached.cache_clear()
    _indicator_cache.clear()


def get_cache_stats() -> dict:
    """Get cache statistics for all caches."""
    return {
        "ema": _ema_cached.cache_info()._asdict(),
        "rsi": _rsi_cached.cache_info()._asdict(),
        "atr": _atr_cached.cache_info()._asdict(),
        "instance": _indicator_cache.stats()
    }
