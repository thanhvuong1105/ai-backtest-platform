# engine/regime_classifier.py
"""
Market Regime Classifier for Quant AI Brain

Classifies market conditions to guide parameter selection:
- TRENDING_UP: Strong upward trend
- TRENDING_DOWN: Strong downward trend
- RANGING: Sideways consolidation
- VOLATILE: High volatility
- LOW_VOLATILITY: Quiet market
"""

from enum import Enum
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"


# ═══════════════════════════════════════════════════════
# REGIME-SPECIFIC PARAM SUGGESTIONS
# ═══════════════════════════════════════════════════════

REGIME_PARAM_HINTS = {
    MarketRegime.TRENDING_UP: {
        "entry": {
            "rf_period": (100, 150),    # Longer for trend confirmation
            "st_mult": (1.5, 2.5),      # Tighter to catch trends
            "rf_mult": (2.5, 4.0),
        },
        "sl": {
            "st_mult": (3.0, 5.0),      # Reasonable SL
            "rf_mult": (5.0, 8.0),
        },
        "tp_dual": {
            "rr_mult": (1.5, 2.5),      # Higher RR in trends
        },
        "tp_rsi": {
            "rr_mult": (1.5, 2.5),
        },
    },
    MarketRegime.TRENDING_DOWN: {
        "entry": {
            "rf_period": (100, 150),
            "st_mult": (1.5, 2.5),
            "rf_mult": (2.5, 4.0),
        },
        "sl": {
            "st_mult": (3.0, 5.0),
            "rf_mult": (5.0, 8.0),
        },
        "tp_dual": {
            "rr_mult": (1.0, 1.5),      # Lower RR in downtrend (long-only)
        },
        "tp_rsi": {
            "rr_mult": (1.0, 1.5),
        },
    },
    MarketRegime.RANGING: {
        "entry": {
            "rf_period": (60, 100),     # Shorter for quicker signals
            "st_mult": (2.5, 3.5),      # Wider to avoid whipsaws
            "rf_mult": (3.0, 5.0),
        },
        "sl": {
            "st_mult": (4.0, 6.0),
            "rf_mult": (6.0, 10.0),
        },
        "tp_dual": {
            "rr_mult": (1.0, 1.3),      # Lower RR in ranges
        },
        "tp_rsi": {
            "rr_mult": (1.0, 1.3),
        },
    },
    MarketRegime.VOLATILE: {
        "entry": {
            "rf_period": (80, 120),
            "st_mult": (3.0, 4.0),      # Wide to handle volatility
            "rf_mult": (4.0, 6.0),
        },
        "sl": {
            "st_mult": (5.0, 8.0),      # Very wide SL
            "rf_mult": (8.0, 12.0),
        },
        "tp_dual": {
            "rr_mult": (2.0, 3.0),      # High RR for volatility swings
        },
        "tp_rsi": {
            "rr_mult": (2.0, 3.0),
        },
    },
    MarketRegime.LOW_VOLATILITY: {
        "entry": {
            "rf_period": (50, 80),      # Shorter periods
            "st_mult": (1.0, 2.0),      # Very tight
            "rf_mult": (2.0, 3.5),
        },
        "sl": {
            "st_mult": (2.5, 4.0),
            "rf_mult": (4.0, 6.0),
        },
        "tp_dual": {
            "rr_mult": (0.8, 1.2),      # Lower RR in quiet markets
        },
        "tp_rsi": {
            "rr_mult": (0.8, 1.2),
        },
    },
}


# ═══════════════════════════════════════════════════════
# INDICATOR CALCULATIONS
# ═══════════════════════════════════════════════════════

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average Directional Index (ADX).

    Higher ADX = stronger trend (>25 is trending)
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # +DM and -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    # ATR
    atr = calculate_atr(df, period)

    # +DI and -DI
    plus_di = 100 * (plus_dm.rolling(period).sum() / atr.rolling(period).sum())
    minus_di = 100 * (minus_dm.rolling(period).sum() / atr.rolling(period).sum())

    # DX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)

    # ADX
    adx = dx.rolling(period).mean()

    return adx


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    close = df["close"]
    delta = close.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_volatility(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate rolling volatility (std of returns)."""
    returns = df["close"].pct_change()
    volatility = returns.rolling(window=period).std() * np.sqrt(252)  # Annualized
    return volatility


# ═══════════════════════════════════════════════════════
# REGIME CLASSIFICATION
# ═══════════════════════════════════════════════════════

def classify_regime(
    df: pd.DataFrame,
    lookback: int = 100
) -> Dict[str, Any]:
    """
    Classify market regime based on recent price action.

    Args:
        df: OHLCV DataFrame
        lookback: Number of bars to analyze

    Returns:
        {
            "regime": MarketRegime,
            "profile": {
                "atr_pct": float,
                "adx": float,
                "volatility": float,
                "trend_ratio": float,
                "rsi_mean": float
            },
            "suggested_params": {...}
        }
    """
    # Use recent data
    recent = df.tail(lookback).copy()

    if len(recent) < lookback:
        # Not enough data, return default
        return {
            "regime": MarketRegime.RANGING,
            "profile": {
                "atr_pct": 0,
                "adx": 0,
                "volatility": 0,
                "trend_ratio": 0.5,
                "rsi_mean": 50,
            },
            "suggested_params": REGIME_PARAM_HINTS[MarketRegime.RANGING],
        }

    # Calculate indicators
    atr = calculate_atr(recent, 14)
    adx = calculate_adx(recent, 14)
    rsi = calculate_rsi(recent, 14)
    volatility = calculate_volatility(recent, 20)

    # Get latest values
    current_price = recent["close"].iloc[-1]
    atr_pct = (atr.iloc[-1] / current_price * 100) if current_price > 0 else 0
    adx_val = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 20
    vol_val = volatility.iloc[-1] if not pd.isna(volatility.iloc[-1]) else 0.3
    rsi_mean = rsi.tail(20).mean() if not rsi.empty else 50

    # Calculate trend ratio (up bars / total bars)
    returns = recent["close"].pct_change()
    up_bars = (returns > 0).sum()
    total_bars = len(returns.dropna())
    trend_ratio = up_bars / total_bars if total_bars > 0 else 0.5

    # Build market profile
    profile = {
        "atr_pct": round(atr_pct, 4),
        "adx": round(adx_val, 2),
        "volatility": round(vol_val, 4),
        "trend_ratio": round(trend_ratio, 4),
        "rsi_mean": round(rsi_mean, 2),
    }

    # Classify regime
    regime = _determine_regime(profile)

    return {
        "regime": regime,
        "profile": profile,
        "suggested_params": REGIME_PARAM_HINTS.get(regime, {}),
    }


def _determine_regime(profile: Dict[str, float]) -> MarketRegime:
    """
    Determine market regime from profile metrics.

    Decision tree:
    1. High volatility (>0.5 annualized) → VOLATILE
    2. Low volatility (<0.15) → LOW_VOLATILITY
    3. High ADX (>25) + trend_ratio > 0.6 → TRENDING_UP
    4. High ADX (>25) + trend_ratio < 0.4 → TRENDING_DOWN
    5. Otherwise → RANGING
    """
    adx = profile.get("adx", 20)
    volatility = profile.get("volatility", 0.3)
    trend_ratio = profile.get("trend_ratio", 0.5)

    # Check volatility first
    if volatility > 0.5:
        return MarketRegime.VOLATILE

    if volatility < 0.15:
        return MarketRegime.LOW_VOLATILITY

    # Check trend strength
    if adx > 25:
        if trend_ratio > 0.6:
            return MarketRegime.TRENDING_UP
        elif trend_ratio < 0.4:
            return MarketRegime.TRENDING_DOWN

    # Default to ranging
    return MarketRegime.RANGING


# ═══════════════════════════════════════════════════════
# PARAM RANGE GENERATION
# ═══════════════════════════════════════════════════════

def get_regime_param_ranges(regime: MarketRegime) -> Dict[str, Any]:
    """
    Get suggested parameter ranges for a regime.

    Args:
        regime: Market regime

    Returns:
        Dict with parameter ranges for each genome block
    """
    return REGIME_PARAM_HINTS.get(regime, REGIME_PARAM_HINTS[MarketRegime.RANGING])


def sample_params_for_regime(
    regime: MarketRegime,
    n_samples: int = 10
) -> List[Dict[str, Any]]:
    """
    Generate random parameter samples biased by regime.

    Args:
        regime: Market regime
        n_samples: Number of samples to generate

    Returns:
        List of genome dictionaries
    """
    hints = get_regime_param_ranges(regime)
    samples = []

    for _ in range(n_samples):
        genome = {
            "entry": {
                "st_atrPeriod": np.random.randint(8, 16),
                "st_src": "hl2",
                "st_mult": _sample_range(hints.get("entry", {}).get("st_mult", (1.5, 3.0))),
                "st_useATR": True,
                "rf_src": "close",
                "rf_period": int(_sample_range(hints.get("entry", {}).get("rf_period", (80, 120)))),
                "rf_mult": _sample_range(hints.get("entry", {}).get("rf_mult", (2.5, 4.0))),
                "rsi_length": np.random.randint(10, 18),
                "rsi_ma_length": np.random.randint(4, 8),
            },
            "sl": {
                "st_atrPeriod": 10,
                "st_src": "hl2",
                "st_mult": _sample_range(hints.get("sl", {}).get("st_mult", (3.0, 5.0))),
                "st_useATR": True,
                "rf_period": 100,
                "rf_mult": _sample_range(hints.get("sl", {}).get("rf_mult", (5.0, 8.0))),
            },
            "tp_dual": {
                "st_atrPeriod": 10,
                "st_mult": 2.0,
                "rr_mult": _sample_range(hints.get("tp_dual", {}).get("rr_mult", (1.0, 2.0))),
            },
            "tp_rsi": {
                "st_atrPeriod": 10,
                "st_mult": 2.0,
                "rr_mult": _sample_range(hints.get("tp_rsi", {}).get("rr_mult", (1.0, 2.0))),
            },
            "mode": {
                "showDualFlip": True,
                "showRSI": True,
            },
        }
        samples.append(genome)

    return samples


def _sample_range(range_tuple: tuple) -> float:
    """Sample a value uniformly from a range."""
    low, high = range_tuple
    return round(np.random.uniform(low, high), 2)


# ═══════════════════════════════════════════════════════
# MULTI-TIMEFRAME REGIME
# ═══════════════════════════════════════════════════════

def classify_multi_timeframe(
    df_dict: Dict[str, pd.DataFrame],
    symbol: str,
    timeframes: List[str]
) -> Dict[str, Any]:
    """
    Classify regime across multiple timeframes.

    Args:
        df_dict: Dict of {timeframe: DataFrame}
        symbol: Trading symbol
        timeframes: List of timeframes to analyze

    Returns:
        {
            "dominant_regime": MarketRegime,
            "by_timeframe": {tf: regime, ...},
            "consensus": bool
        }
    """
    regimes = {}

    for tf in timeframes:
        key = f"{symbol}_{tf}"
        if key in df_dict:
            result = classify_regime(df_dict[key])
            regimes[tf] = result["regime"]

    # Find dominant regime (most common)
    if not regimes:
        return {
            "dominant_regime": MarketRegime.RANGING,
            "by_timeframe": {},
            "consensus": False,
        }

    regime_counts = {}
    for regime in regimes.values():
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

    dominant = max(regime_counts.keys(), key=lambda r: regime_counts[r])
    consensus = regime_counts[dominant] >= len(regimes) * 0.6  # 60% agreement

    return {
        "dominant_regime": dominant,
        "by_timeframe": {tf: r.value for tf, r in regimes.items()},
        "consensus": consensus,
    }
