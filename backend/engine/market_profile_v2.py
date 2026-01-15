# engine/market_profile_v2.py
"""
Market Profile v2.0 - Advanced Market Analysis with 32 Indicators

This module provides comprehensive market characterization using 32 indicators
organized into 6 clusters for genetic algorithm seeding and regime classification.

Clusters:
- Volatility (5 indicators): atr_pct, vol_regime, vol_percentile, vol_expanding, vol_ratio
- Trend (7 indicators): adx, plus_di, minus_di, efficiency_ratio, trend_direction, trend_strength_pct, trend_angle
- Momentum (8 indicators): rsi_current, rsi_mean, rsi_std, rsi_slope, overbought_pct, oversold_pct, macd_histogram, macd_trend
- Volume (5 indicators): volume_ratio, volume_trend, vp_correlation, volume_climax, obv_trend
- Cycle (5 indicators): cycle_phase, bb_position, bb_width, squeeze, squeeze_momentum
- Correlation (2 indicators): auto_correlation, btc_correlation

Total: 32 indicators (NO structure analysis as per specification)
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# VOLATILITY CLUSTER (5 indicators)
# ═══════════════════════════════════════════════════════════════════════════════

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


def calculate_volatility_cluster(df: pd.DataFrame, lookback: int = 100) -> Dict[str, float]:
    """
    Calculate Volatility Cluster indicators.

    Returns:
        - atr_pct: ATR as percentage of price (normalized volatility)
        - vol_regime: Current volatility regime (0=low, 1=normal, 2=high)
        - vol_percentile: Current volatility percentile (0-100)
        - vol_expanding: Whether volatility is expanding (0 or 1)
        - vol_ratio: Current vol / Average vol ratio
    """
    recent = df.tail(lookback).copy()

    if len(recent) < 20:
        return {
            "atr_pct": 0.0,
            "vol_regime": 1,
            "vol_percentile": 50.0,
            "vol_expanding": 0,
            "vol_ratio": 1.0
        }

    # ATR as percentage of price
    atr = calculate_atr(recent, 14)
    current_price = recent["close"].iloc[-1]
    atr_pct = (atr.iloc[-1] / current_price * 100) if current_price > 0 else 0

    # Rolling volatility (std of returns)
    returns = recent["close"].pct_change()
    current_vol = returns.tail(14).std()
    historical_vol = returns.std()

    # Volatility percentile
    rolling_vol = returns.rolling(14).std()
    vol_percentile = (rolling_vol < current_vol).sum() / len(rolling_vol.dropna()) * 100

    # Volatility regime (low < 25th, high > 75th percentile)
    if vol_percentile < 25:
        vol_regime = 0  # Low volatility
    elif vol_percentile > 75:
        vol_regime = 2  # High volatility
    else:
        vol_regime = 1  # Normal volatility

    # Volatility expanding or contracting
    recent_vol = rolling_vol.tail(5).mean()
    older_vol = rolling_vol.tail(20).head(15).mean()
    vol_expanding = 1 if recent_vol > older_vol * 1.1 else 0

    # Volatility ratio
    vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0

    return {
        "atr_pct": round(atr_pct, 4),
        "vol_regime": int(vol_regime),
        "vol_percentile": round(vol_percentile, 2),
        "vol_expanding": int(vol_expanding),
        "vol_ratio": round(vol_ratio, 4)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TREND CLUSTER (7 indicators)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_adx_full(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate ADX with +DI and -DI.

    Returns:
        (adx, plus_di, minus_di)
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

    # Smoothed +DI and -DI
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() /
                     atr.ewm(span=period, adjust=False).mean())
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() /
                      atr.ewm(span=period, adjust=False).mean())

    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, adjust=False).mean()

    return adx, plus_di, minus_di


def calculate_efficiency_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate Kaufman's Efficiency Ratio.

    ER = Direction / Volatility
    - 1.0 = Perfect trend
    - 0.0 = Complete noise/chop
    """
    close = df["close"]

    direction = abs(close - close.shift(period))
    volatility = abs(close.diff()).rolling(period).sum()

    er = direction / (volatility + 1e-10)
    return er


def calculate_trend_cluster(df: pd.DataFrame, lookback: int = 100) -> Dict[str, float]:
    """
    Calculate Trend Cluster indicators.

    Returns:
        - adx: Average Directional Index (0-100)
        - plus_di: +DI value
        - minus_di: -DI value
        - efficiency_ratio: Kaufman ER (0-1)
        - trend_direction: 1=up, -1=down, 0=neutral
        - trend_strength_pct: Trend strength as percentage
        - trend_angle: Slope angle of linear regression
    """
    recent = df.tail(lookback).copy()

    if len(recent) < 30:
        return {
            "adx": 20.0,
            "plus_di": 25.0,
            "minus_di": 25.0,
            "efficiency_ratio": 0.5,
            "trend_direction": 0,
            "trend_strength_pct": 0.0,
            "trend_angle": 0.0
        }

    # ADX with +DI/-DI
    adx, plus_di, minus_di = calculate_adx_full(recent, 14)
    adx_val = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 20
    plus_di_val = plus_di.iloc[-1] if not pd.isna(plus_di.iloc[-1]) else 25
    minus_di_val = minus_di.iloc[-1] if not pd.isna(minus_di.iloc[-1]) else 25

    # Efficiency Ratio
    er = calculate_efficiency_ratio(recent, 20)
    er_val = er.iloc[-1] if not pd.isna(er.iloc[-1]) else 0.5

    # Trend direction
    close = recent["close"]
    sma_short = close.rolling(10).mean().iloc[-1]
    sma_long = close.rolling(30).mean().iloc[-1]

    if sma_short > sma_long * 1.01:
        trend_direction = 1  # Uptrend
    elif sma_short < sma_long * 0.99:
        trend_direction = -1  # Downtrend
    else:
        trend_direction = 0  # Neutral

    # Trend strength percentage
    if plus_di_val + minus_di_val > 0:
        trend_strength_pct = abs(plus_di_val - minus_di_val) / (plus_di_val + minus_di_val) * 100
    else:
        trend_strength_pct = 0

    # Trend angle (linear regression slope)
    x = np.arange(len(close.tail(20)))
    y = close.tail(20).values
    if len(x) > 1:
        slope, _ = np.polyfit(x, y, 1)
        # Normalize slope to angle degrees
        price_range = y.max() - y.min()
        if price_range > 0:
            normalized_slope = slope / price_range * 20  # Normalize by price range
            trend_angle = np.degrees(np.arctan(normalized_slope))
        else:
            trend_angle = 0
    else:
        trend_angle = 0

    return {
        "adx": round(adx_val, 2),
        "plus_di": round(plus_di_val, 2),
        "minus_di": round(minus_di_val, 2),
        "efficiency_ratio": round(min(max(er_val, 0), 1), 4),
        "trend_direction": int(trend_direction),
        "trend_strength_pct": round(trend_strength_pct, 2),
        "trend_angle": round(trend_angle, 2)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MOMENTUM CLUSTER (8 indicators)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    close = df["close"]
    delta = close.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD indicator.

    Returns:
        (macd_line, signal_line, histogram)
    """
    close = df["close"]

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_momentum_cluster(df: pd.DataFrame, lookback: int = 100) -> Dict[str, float]:
    """
    Calculate Momentum Cluster indicators.

    Returns:
        - rsi_current: Current RSI value
        - rsi_mean: Average RSI over lookback
        - rsi_std: RSI standard deviation
        - rsi_slope: RSI trend slope
        - overbought_pct: % of time RSI > 70
        - oversold_pct: % of time RSI < 30
        - macd_histogram: Current MACD histogram value
        - macd_trend: MACD trend direction (1=bullish, -1=bearish, 0=neutral)
    """
    recent = df.tail(lookback).copy()

    if len(recent) < 30:
        return {
            "rsi_current": 50.0,
            "rsi_mean": 50.0,
            "rsi_std": 10.0,
            "rsi_slope": 0.0,
            "overbought_pct": 0.0,
            "oversold_pct": 0.0,
            "macd_histogram": 0.0,
            "macd_trend": 0
        }

    # RSI calculations
    rsi = calculate_rsi(recent, 14)
    rsi_clean = rsi.dropna()

    rsi_current = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    rsi_mean = rsi_clean.mean() if len(rsi_clean) > 0 else 50
    rsi_std = rsi_clean.std() if len(rsi_clean) > 0 else 10

    # RSI slope (last 10 bars)
    rsi_recent = rsi.tail(10).dropna()
    if len(rsi_recent) > 1:
        x = np.arange(len(rsi_recent))
        slope, _ = np.polyfit(x, rsi_recent.values, 1)
        rsi_slope = slope
    else:
        rsi_slope = 0

    # Overbought/Oversold percentages
    overbought_pct = (rsi_clean > 70).sum() / len(rsi_clean) * 100 if len(rsi_clean) > 0 else 0
    oversold_pct = (rsi_clean < 30).sum() / len(rsi_clean) * 100 if len(rsi_clean) > 0 else 0

    # MACD calculations
    macd_line, signal_line, histogram = calculate_macd(recent)
    macd_histogram = histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else 0

    # MACD trend (compare recent vs older histogram)
    hist_recent = histogram.tail(5).mean()
    hist_older = histogram.tail(15).head(10).mean()

    if hist_recent > hist_older * 1.1:
        macd_trend = 1  # Bullish
    elif hist_recent < hist_older * 0.9:
        macd_trend = -1  # Bearish
    else:
        macd_trend = 0  # Neutral

    return {
        "rsi_current": round(rsi_current, 2),
        "rsi_mean": round(rsi_mean, 2),
        "rsi_std": round(rsi_std, 2),
        "rsi_slope": round(rsi_slope, 4),
        "overbought_pct": round(overbought_pct, 2),
        "oversold_pct": round(oversold_pct, 2),
        "macd_histogram": round(macd_histogram, 6),
        "macd_trend": int(macd_trend)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# VOLUME CLUSTER (5 indicators)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """Calculate On-Balance Volume."""
    close = df["close"]
    volume = df["volume"]

    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv


def calculate_volume_cluster(df: pd.DataFrame, lookback: int = 100) -> Dict[str, float]:
    """
    Calculate Volume Cluster indicators.

    Returns:
        - volume_ratio: Current volume / Average volume
        - volume_trend: Volume trend direction (1=increasing, -1=decreasing, 0=stable)
        - vp_correlation: Volume-Price correlation
        - volume_climax: Whether current volume is climax (0 or 1)
        - obv_trend: OBV trend direction (1=up, -1=down, 0=flat)
    """
    recent = df.tail(lookback).copy()

    if len(recent) < 20 or "volume" not in recent.columns:
        return {
            "volume_ratio": 1.0,
            "volume_trend": 0,
            "vp_correlation": 0.0,
            "volume_climax": 0,
            "obv_trend": 0
        }

    volume = recent["volume"]
    close = recent["close"]

    # Volume ratio
    current_vol = volume.iloc[-1]
    avg_vol = volume.mean()
    volume_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

    # Volume trend
    recent_vol_avg = volume.tail(10).mean()
    older_vol_avg = volume.tail(30).head(20).mean()

    if recent_vol_avg > older_vol_avg * 1.2:
        volume_trend = 1  # Increasing
    elif recent_vol_avg < older_vol_avg * 0.8:
        volume_trend = -1  # Decreasing
    else:
        volume_trend = 0  # Stable

    # Volume-Price correlation
    returns = close.pct_change().dropna()
    vol_clean = volume.iloc[1:len(returns) + 1]

    if len(returns) > 10 and len(vol_clean) > 10:
        vp_correlation = returns.corr(vol_clean)
        vp_correlation = vp_correlation if not pd.isna(vp_correlation) else 0
    else:
        vp_correlation = 0

    # Volume climax (> 2x average)
    volume_climax = 1 if volume_ratio > 2.0 else 0

    # OBV trend
    obv = calculate_obv(recent)
    obv_sma_short = obv.tail(10).mean()
    obv_sma_long = obv.tail(30).mean()

    if obv_sma_short > obv_sma_long * 1.05:
        obv_trend = 1  # Bullish
    elif obv_sma_short < obv_sma_long * 0.95:
        obv_trend = -1  # Bearish
    else:
        obv_trend = 0  # Flat

    return {
        "volume_ratio": round(volume_ratio, 4),
        "volume_trend": int(volume_trend),
        "vp_correlation": round(vp_correlation, 4),
        "volume_climax": int(volume_climax),
        "obv_trend": int(obv_trend)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CYCLE CLUSTER (5 indicators)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.

    Returns:
        (upper_band, middle_band, lower_band)
    """
    close = df["close"]

    middle = close.rolling(period).mean()
    std = close.rolling(period).std()

    upper = middle + std_mult * std
    lower = middle - std_mult * std

    return upper, middle, lower


def calculate_cycle_cluster(df: pd.DataFrame, lookback: int = 100) -> Dict[str, float]:
    """
    Calculate Cycle Cluster indicators.

    Returns:
        - cycle_phase: Current market cycle phase (0=accumulation, 1=markup, 2=distribution, 3=markdown)
        - bb_position: Price position within Bollinger Bands (0-1)
        - bb_width: Bollinger Band width (normalized)
        - squeeze: Whether BB is in squeeze (0 or 1)
        - squeeze_momentum: Squeeze momentum value
    """
    recent = df.tail(lookback).copy()

    if len(recent) < 30:
        return {
            "cycle_phase": 0,
            "bb_position": 0.5,
            "bb_width": 0.0,
            "squeeze": 0,
            "squeeze_momentum": 0.0
        }

    close = recent["close"]

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(recent, 20, 2.0)

    # BB position (0 = lower band, 1 = upper band)
    current_price = close.iloc[-1]
    bb_range = bb_upper.iloc[-1] - bb_lower.iloc[-1]
    if bb_range > 0:
        bb_position = (current_price - bb_lower.iloc[-1]) / bb_range
        bb_position = min(max(bb_position, 0), 1)
    else:
        bb_position = 0.5

    # BB width (normalized by price)
    bb_width = (bb_range / bb_middle.iloc[-1] * 100) if bb_middle.iloc[-1] > 0 else 0

    # Squeeze detection (BB inside Keltner Channel)
    atr = calculate_atr(recent, 20)
    kc_width = 1.5 * atr.iloc[-1]
    squeeze = 1 if bb_range < 2 * kc_width else 0

    # Squeeze momentum (momentum histogram)
    momentum = close - close.rolling(20).mean()
    squeeze_momentum = momentum.iloc[-1] / close.iloc[-1] * 100 if close.iloc[-1] > 0 else 0

    # Cycle phase detection
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()

    current_sma20 = sma_20.iloc[-1]
    current_sma50 = sma_50.iloc[-1]
    prev_sma20 = sma_20.iloc[-10] if len(sma_20) > 10 else current_sma20

    if current_price > current_sma20 > current_sma50 and current_sma20 > prev_sma20:
        cycle_phase = 1  # Markup
    elif current_price > current_sma20 and current_sma20 < prev_sma20:
        cycle_phase = 2  # Distribution
    elif current_price < current_sma20 < current_sma50 and current_sma20 < prev_sma20:
        cycle_phase = 3  # Markdown
    else:
        cycle_phase = 0  # Accumulation

    return {
        "cycle_phase": int(cycle_phase),
        "bb_position": round(bb_position, 4),
        "bb_width": round(bb_width, 4),
        "squeeze": int(squeeze),
        "squeeze_momentum": round(squeeze_momentum, 4)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CORRELATION CLUSTER (2 indicators)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_correlation_cluster(
    df: pd.DataFrame,
    btc_df: Optional[pd.DataFrame] = None,
    lookback: int = 100
) -> Dict[str, float]:
    """
    Calculate Correlation Cluster indicators.

    Returns:
        - auto_correlation: Price autocorrelation (lag 1)
        - btc_correlation: Correlation with BTC (if not BTC itself)
    """
    recent = df.tail(lookback).copy()

    if len(recent) < 30:
        return {
            "auto_correlation": 0.0,
            "btc_correlation": 0.0
        }

    returns = recent["close"].pct_change().dropna()

    # Autocorrelation (lag 1)
    if len(returns) > 10:
        auto_correlation = returns.autocorr(lag=1)
        auto_correlation = auto_correlation if not pd.isna(auto_correlation) else 0
    else:
        auto_correlation = 0

    # BTC correlation
    btc_correlation = 0.0
    if btc_df is not None and len(btc_df) >= lookback:
        btc_recent = btc_df.tail(lookback).copy()
        btc_returns = btc_recent["close"].pct_change().dropna()

        # Align the series
        min_len = min(len(returns), len(btc_returns))
        if min_len > 10:
            btc_correlation = returns.tail(min_len).corr(btc_returns.tail(min_len))
            btc_correlation = btc_correlation if not pd.isna(btc_correlation) else 0

    return {
        "auto_correlation": round(auto_correlation, 4),
        "btc_correlation": round(btc_correlation, 4)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN MARKET PROFILE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_market_profile_v2(
    df: pd.DataFrame,
    btc_df: Optional[pd.DataFrame] = None,
    lookback: int = 100
) -> Dict[str, Any]:
    """
    Calculate complete Market Profile v2.0 with 32 indicators across 6 clusters.

    Args:
        df: OHLCV DataFrame for the target symbol
        btc_df: Optional BTC DataFrame for correlation calculation
        lookback: Number of bars to analyze

    Returns:
        {
            "volatility": {...},  # 5 indicators
            "trend": {...},       # 7 indicators
            "momentum": {...},    # 8 indicators
            "volume": {...},      # 5 indicators
            "cycle": {...},       # 5 indicators
            "correlation": {...}, # 2 indicators
            "summary": {...},     # Aggregated summary
            "total_indicators": 32
        }
    """
    logger.debug(f"Calculating Market Profile v2 with lookback={lookback}")

    # Calculate all clusters
    volatility = calculate_volatility_cluster(df, lookback)
    trend = calculate_trend_cluster(df, lookback)
    momentum = calculate_momentum_cluster(df, lookback)
    volume = calculate_volume_cluster(df, lookback)
    cycle = calculate_cycle_cluster(df, lookback)
    correlation = calculate_correlation_cluster(df, btc_df, lookback)

    # Create summary for quick access to key metrics
    summary = {
        "regime": _determine_regime_from_clusters(volatility, trend, momentum),
        "trend_score": _calculate_trend_score(trend, momentum),
        "volatility_score": _calculate_volatility_score(volatility),
        "momentum_score": _calculate_momentum_score(momentum),
        "market_condition": _classify_market_condition(volatility, trend, momentum, cycle)
    }

    return {
        "volatility": volatility,
        "trend": trend,
        "momentum": momentum,
        "volume": volume,
        "cycle": cycle,
        "correlation": correlation,
        "summary": summary,
        "total_indicators": 32
    }


def _determine_regime_from_clusters(
    volatility: Dict[str, float],
    trend: Dict[str, float],
    momentum: Dict[str, float]
) -> str:
    """Determine overall market regime from cluster indicators."""
    vol_regime = volatility.get("vol_regime", 1)
    adx = trend.get("adx", 20)
    trend_direction = trend.get("trend_direction", 0)

    if vol_regime == 2:  # High volatility
        return "volatile"
    elif vol_regime == 0:  # Low volatility
        return "low_volatility"
    elif adx > 25:
        if trend_direction > 0:
            return "trending_up"
        elif trend_direction < 0:
            return "trending_down"

    return "ranging"


def _calculate_trend_score(trend: Dict[str, float], momentum: Dict[str, float]) -> float:
    """Calculate combined trend score (0-100)."""
    adx = trend.get("adx", 20)
    er = trend.get("efficiency_ratio", 0.5)
    trend_strength = trend.get("trend_strength_pct", 0)
    macd_trend = momentum.get("macd_trend", 0)

    # Weighted combination
    score = (
        adx * 0.4 +
        er * 50 * 0.3 +
        trend_strength * 0.2 +
        (macd_trend + 1) * 25 * 0.1
    )

    return round(min(max(score, 0), 100), 2)


def _calculate_volatility_score(volatility: Dict[str, float]) -> float:
    """Calculate volatility score (0-100)."""
    vol_percentile = volatility.get("vol_percentile", 50)
    vol_ratio = volatility.get("vol_ratio", 1)
    vol_expanding = volatility.get("vol_expanding", 0)

    # Higher score = more volatile
    score = vol_percentile * 0.6 + min(vol_ratio * 30, 30) * 0.3 + vol_expanding * 10 * 0.1

    return round(min(max(score, 0), 100), 2)


def _calculate_momentum_score(momentum: Dict[str, float]) -> float:
    """Calculate momentum score (-100 to 100, positive = bullish)."""
    rsi = momentum.get("rsi_current", 50)
    rsi_slope = momentum.get("rsi_slope", 0)
    macd_trend = momentum.get("macd_trend", 0)

    # RSI contribution (50 = neutral)
    rsi_score = (rsi - 50) * 2

    # MACD contribution
    macd_score = macd_trend * 20

    # RSI slope contribution
    slope_score = min(max(rsi_slope * 50, -20), 20)

    score = rsi_score * 0.5 + macd_score * 0.3 + slope_score * 0.2

    return round(min(max(score, -100), 100), 2)


def _classify_market_condition(
    volatility: Dict[str, float],
    trend: Dict[str, float],
    momentum: Dict[str, float],
    cycle: Dict[str, float]
) -> str:
    """Classify overall market condition for strategy selection."""
    cycle_phase = cycle.get("cycle_phase", 0)
    squeeze = cycle.get("squeeze", 0)
    vol_regime = volatility.get("vol_regime", 1)
    trend_direction = trend.get("trend_direction", 0)

    conditions = []

    # Cycle phase names
    phase_names = ["accumulation", "markup", "distribution", "markdown"]
    conditions.append(phase_names[cycle_phase])

    # Volatility state
    vol_states = ["quiet", "normal", "volatile"]
    conditions.append(vol_states[vol_regime])

    # Squeeze state
    if squeeze:
        conditions.append("squeeze")

    # Trend state
    if trend_direction > 0:
        conditions.append("bullish")
    elif trend_direction < 0:
        conditions.append("bearish")
    else:
        conditions.append("neutral")

    return "_".join(conditions)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPATIBILITY LAYER (for existing code)
# ═══════════════════════════════════════════════════════════════════════════════

def get_simplified_profile(df: pd.DataFrame, lookback: int = 100) -> Dict[str, float]:
    """
    Get simplified 5-indicator profile for backward compatibility.

    This maintains the original interface used by existing code.
    """
    profile = calculate_market_profile_v2(df, lookback=lookback)

    return {
        "atr_pct": profile["volatility"]["atr_pct"],
        "adx": profile["trend"]["adx"],
        "volatility": profile["volatility"]["vol_percentile"] / 100,
        "trend_ratio": 0.5 + profile["trend"]["trend_direction"] * 0.25,
        "rsi_mean": profile["momentum"]["rsi_mean"]
    }


def get_profile_vector(profile: Dict[str, Any]) -> List[float]:
    """
    Convert profile to flat vector for similarity calculations.

    Returns 32-element vector (normalized 0-1 where applicable).
    """
    vector = []

    # Volatility cluster (5)
    vol = profile.get("volatility", {})
    vector.extend([
        min(vol.get("atr_pct", 0) / 5, 1),  # Normalize ATR% to 0-1
        vol.get("vol_regime", 1) / 2,  # 0-2 → 0-1
        vol.get("vol_percentile", 50) / 100,
        vol.get("vol_expanding", 0),
        min(vol.get("vol_ratio", 1) / 3, 1)  # Normalize ratio
    ])

    # Trend cluster (7)
    trend = profile.get("trend", {})
    vector.extend([
        min(trend.get("adx", 20) / 100, 1),
        min(trend.get("plus_di", 25) / 100, 1),
        min(trend.get("minus_di", 25) / 100, 1),
        trend.get("efficiency_ratio", 0.5),
        (trend.get("trend_direction", 0) + 1) / 2,  # -1,0,1 → 0,0.5,1
        trend.get("trend_strength_pct", 0) / 100,
        (trend.get("trend_angle", 0) + 90) / 180  # -90,90 → 0,1
    ])

    # Momentum cluster (8)
    mom = profile.get("momentum", {})
    vector.extend([
        mom.get("rsi_current", 50) / 100,
        mom.get("rsi_mean", 50) / 100,
        min(mom.get("rsi_std", 10) / 30, 1),
        (mom.get("rsi_slope", 0) + 5) / 10,  # Normalize slope
        mom.get("overbought_pct", 0) / 100,
        mom.get("oversold_pct", 0) / 100,
        (mom.get("macd_histogram", 0) + 0.01) / 0.02,  # Normalize histogram
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
        (cycle.get("squeeze_momentum", 0) + 5) / 10
    ])

    # Correlation cluster (2)
    corr = profile.get("correlation", {})
    vector.extend([
        (corr.get("auto_correlation", 0) + 1) / 2,
        (corr.get("btc_correlation", 0) + 1) / 2
    ])

    return vector


def calculate_profile_similarity(profile1: Dict[str, Any], profile2: Dict[str, Any]) -> float:
    """
    Calculate cosine similarity between two market profiles.

    Returns value between 0 (completely different) and 1 (identical).
    """
    vec1 = np.array(get_profile_vector(profile1))
    vec2 = np.array(get_profile_vector(profile2))

    # Cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = dot_product / (norm1 * norm2)
    return round(max(0, min(1, similarity)), 4)
