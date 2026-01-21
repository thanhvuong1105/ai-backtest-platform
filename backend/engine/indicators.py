# engine/indicators.py
"""
Technical indicators with G1) Indicator caching for performance optimization.
"""
import os
import pandas as pd
import numpy as np

# G1) Indicator cache configuration
INDICATOR_CACHE_SIZE = int(os.getenv("INDICATOR_CACHE_SIZE", 4096))
_indicator_cache = {}
_cache_hits = 0
_cache_misses = 0


def _get_series_hash(series: pd.Series) -> str:
    """Generate a hash key for a pandas Series (fast approximation)."""
    if series is None or len(series) == 0:
        return "empty"
    first_val = series.iloc[0] if len(series) > 0 else 0
    last_val = series.iloc[-1] if len(series) > 0 else 0
    return f"{first_val:.6f}_{last_val:.6f}_{len(series)}"


def _cache_key(indicator_name: str, series_hash: str, *params) -> str:
    """Generate cache key for indicator."""
    params_str = "_".join(str(p) for p in params)
    return f"{indicator_name}:{series_hash}:{params_str}"


def get_cache_stats() -> dict:
    """Get indicator cache statistics."""
    global _cache_hits, _cache_misses
    total = _cache_hits + _cache_misses
    hit_rate = _cache_hits / total if total > 0 else 0
    return {
        "hits": _cache_hits, "misses": _cache_misses, "total": total,
        "hit_rate": round(hit_rate * 100, 2), "cache_size": len(_indicator_cache),
    }


def clear_indicator_cache():
    """Clear indicator cache."""
    global _indicator_cache, _cache_hits, _cache_misses
    _indicator_cache = {}
    _cache_hits = 0
    _cache_misses = 0


def ema(series: pd.Series, length: int):
    """Exponential Moving Average with caching."""
    global _indicator_cache, _cache_hits, _cache_misses

    series_hash = _get_series_hash(series)
    key = _cache_key("ema", series_hash, length)

    if key in _indicator_cache:
        _cache_hits += 1
        return _indicator_cache[key].copy()

    _cache_misses += 1
    result = series.ewm(span=length, adjust=False).mean()

    if len(_indicator_cache) < INDICATOR_CACHE_SIZE:
        _indicator_cache[key] = result.copy()

    return result


def rsi(series: pd.Series, length: int = 14):
    """Relative Strength Index with caching."""
    global _indicator_cache, _cache_hits, _cache_misses

    series_hash = _get_series_hash(series)
    key = _cache_key("rsi", series_hash, length)

    if key in _indicator_cache:
        _cache_hits += 1
        return _indicator_cache[key].copy()

    _cache_misses += 1

    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    gain = pd.Series(gain).rolling(length).mean()
    loss = pd.Series(loss).rolling(length).mean()

    rs = gain / loss
    result = 100 - (100 / (1 + rs))

    if len(_indicator_cache) < INDICATOR_CACHE_SIZE:
        _indicator_cache[key] = result.copy()

    return result
