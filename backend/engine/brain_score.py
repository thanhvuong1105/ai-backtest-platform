# engine/brain_score.py
"""
BrainScore v2.0 - Multi-Criteria Fitness Scoring System

Comprehensive scoring system for genetic algorithm optimization with:
- Profitability metrics (40% weight)
- Risk metrics (30% weight)
- Consistency metrics (20% weight)
- Statistical significance (10% weight)

Formula:
BrainScore = 0.4 * Profitability + 0.3 * Risk + 0.2 * Consistency + 0.1 * Significance

Where each component is normalized to 0-100 scale.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE METRICS CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_all_metrics(
    equity_curve: List[float],
    trades: List[Dict[str, Any]],
    initial_capital: float = 10000,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """
    Calculate all 15+ performance metrics from equity curve and trades.

    Args:
        equity_curve: List of equity values over time
        trades: List of trade dictionaries with pnl, entry_time, exit_time, etc.
        initial_capital: Starting capital
        risk_free_rate: Annual risk-free rate for Sharpe/Sortino calculation

    Returns:
        Dictionary containing all performance metrics
    """
    if not equity_curve or len(equity_curve) < 2:
        return _get_default_metrics()

    equity = np.array(equity_curve)
    final_equity = equity[-1]

    # Basic metrics
    net_profit = final_equity - initial_capital
    net_profit_pct = (net_profit / initial_capital) * 100
    total_trades = len(trades)

    # Trade analysis
    winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
    losing_trades = [t for t in trades if t.get("pnl", 0) < 0]
    num_wins = len(winning_trades)
    num_losses = len(losing_trades)

    # Win rate
    winrate = (num_wins / total_trades * 100) if total_trades > 0 else 0

    # Profit Factor
    gross_profit = sum(t.get("pnl", 0) for t in winning_trades)
    gross_loss = abs(sum(t.get("pnl", 0) for t in losing_trades))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else gross_profit if gross_profit > 0 else 0

    # Average trade metrics
    avg_win = gross_profit / num_wins if num_wins > 0 else 0
    avg_loss = gross_loss / num_losses if num_losses > 0 else 0
    avg_trade = net_profit / total_trades if total_trades > 0 else 0

    # Expectancy
    expectancy = (winrate / 100 * avg_win) - ((1 - winrate / 100) * avg_loss)

    # Drawdown analysis
    max_dd_pct, max_dd_duration, ulcer_index = _calculate_drawdown_metrics(equity)

    # Returns for risk metrics
    returns = np.diff(equity) / equity[:-1]

    # Sharpe Ratio (annualized)
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    sharpe_ratio = (np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)) if np.std(excess_returns) > 0 else 0

    # Sortino Ratio (only downside volatility)
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-10
    sortino_ratio = (np.mean(excess_returns) / downside_std * np.sqrt(252)) if downside_std > 0 else 0

    # Calmar Ratio (annual return / max DD)
    annual_return = net_profit_pct  # Simplified, assume ~1 year data
    calmar_ratio = (annual_return / max_dd_pct) if max_dd_pct > 0 else annual_return if annual_return > 0 else 0

    # Recovery Factor
    recovery_factor = (net_profit / (max_dd_pct * initial_capital / 100)) if max_dd_pct > 0 else 0

    # Loss streak analysis
    max_loss_streak, max_win_streak = _calculate_streaks(trades)

    # Equity curve smoothness
    smoothness = _calculate_smoothness(equity)

    # R-Multiple metrics
    avg_r, r_expectancy = _calculate_r_metrics(trades)

    # Statistical significance
    t_stat, p_value = _calculate_statistical_significance(trades)

    return {
        # Basic metrics
        "netProfit": round(net_profit, 2),
        "netProfitPct": round(net_profit_pct, 2),
        "totalTrades": total_trades,
        "winrate": round(winrate, 2),
        "profitFactor": round(min(profit_factor, 99), 2),  # Cap at 99

        # Trade metrics
        "avgWin": round(avg_win, 2),
        "avgLoss": round(avg_loss, 2),
        "avgTrade": round(avg_trade, 2),
        "expectancy": round(expectancy, 2),
        "avgR": round(avg_r, 2),
        "rExpectancy": round(r_expectancy, 4),

        # Risk metrics
        "maxDrawdownPct": round(max_dd_pct, 2),
        "maxDrawdownDuration": max_dd_duration,
        "ulcerIndex": round(ulcer_index, 4),
        "sharpeRatio": round(sharpe_ratio, 4),
        "sortinoRatio": round(sortino_ratio, 4),
        "calmarRatio": round(calmar_ratio, 4),
        "recoveryFactor": round(recovery_factor, 4),

        # Consistency metrics
        "maxLossStreak": max_loss_streak,
        "maxWinStreak": max_win_streak,
        "smoothness": round(smoothness, 4),

        # Statistical metrics
        "tStatistic": round(t_stat, 4),
        "pValue": round(p_value, 4),

        # Final capital
        "finalEquity": round(final_equity, 2)
    }


def _get_default_metrics() -> Dict[str, float]:
    """Return default metrics when calculation is not possible."""
    return {
        "netProfit": 0,
        "netProfitPct": 0,
        "totalTrades": 0,
        "winrate": 0,
        "profitFactor": 0,
        "avgWin": 0,
        "avgLoss": 0,
        "avgTrade": 0,
        "expectancy": 0,
        "avgR": 0,
        "rExpectancy": 0,
        "maxDrawdownPct": 0,
        "maxDrawdownDuration": 0,
        "ulcerIndex": 0,
        "sharpeRatio": 0,
        "sortinoRatio": 0,
        "calmarRatio": 0,
        "recoveryFactor": 0,
        "maxLossStreak": 0,
        "maxWinStreak": 0,
        "smoothness": 0,
        "tStatistic": 0,
        "pValue": 1,
        "finalEquity": 0
    }


def _calculate_drawdown_metrics(equity: np.ndarray) -> Tuple[float, int, float]:
    """
    Calculate drawdown-related metrics.

    Returns:
        (max_drawdown_pct, max_drawdown_duration, ulcer_index)
    """
    # Running maximum
    running_max = np.maximum.accumulate(equity)

    # Drawdown array
    drawdown = (running_max - equity) / running_max * 100

    # Max drawdown percentage
    max_dd_pct = np.max(drawdown) if len(drawdown) > 0 else 0

    # Drawdown duration (bars in drawdown)
    in_drawdown = drawdown > 0
    max_dd_duration = 0
    current_duration = 0

    for in_dd in in_drawdown:
        if in_dd:
            current_duration += 1
            max_dd_duration = max(max_dd_duration, current_duration)
        else:
            current_duration = 0

    # Ulcer Index (sqrt of mean squared drawdown)
    ulcer_index = np.sqrt(np.mean(drawdown ** 2)) if len(drawdown) > 0 else 0

    return max_dd_pct, max_dd_duration, ulcer_index


def _calculate_streaks(trades: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    Calculate max losing and winning streaks.

    Returns:
        (max_loss_streak, max_win_streak)
    """
    if not trades:
        return 0, 0

    max_loss_streak = 0
    max_win_streak = 0
    current_loss_streak = 0
    current_win_streak = 0

    for trade in trades:
        pnl = trade.get("pnl", 0)

        if pnl > 0:
            current_win_streak += 1
            current_loss_streak = 0
            max_win_streak = max(max_win_streak, current_win_streak)
        elif pnl < 0:
            current_loss_streak += 1
            current_win_streak = 0
            max_loss_streak = max(max_loss_streak, current_loss_streak)
        else:
            # Break even trade, reset both
            current_loss_streak = 0
            current_win_streak = 0

    return max_loss_streak, max_win_streak


def _calculate_smoothness(equity: np.ndarray) -> float:
    """
    Calculate equity curve smoothness (0-1, higher = smoother).

    Uses R-squared of linear regression fit.
    """
    if len(equity) < 2:
        return 0

    x = np.arange(len(equity))
    y = equity

    # Linear regression
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept

    # R-squared
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return max(0, r_squared)


def _calculate_r_metrics(trades: List[Dict[str, Any]]) -> Tuple[float, float]:
    """
    Calculate R-multiple metrics.

    R = Risk unit = Initial stop loss distance
    R-multiple = PnL / R

    Returns:
        (avg_r_multiple, r_expectancy)
    """
    if not trades:
        return 0, 0

    r_multiples = []

    for trade in trades:
        pnl = trade.get("pnl", 0)
        risk = trade.get("risk", None)  # Initial risk amount

        if risk and risk > 0:
            r_multiple = pnl / risk
            r_multiples.append(r_multiple)
        elif trade.get("entry_price") and trade.get("sl_price"):
            # Calculate R from entry and SL
            entry = trade["entry_price"]
            sl = trade["sl_price"]
            risk_pct = abs(entry - sl) / entry
            if risk_pct > 0:
                pnl_pct = pnl / trade.get("position_size", entry)
                r_multiple = pnl_pct / risk_pct
                r_multiples.append(r_multiple)

    if not r_multiples:
        return 0, 0

    avg_r = np.mean(r_multiples)
    r_expectancy = np.mean(r_multiples)  # Same as avg_r but conceptually different

    return avg_r, r_expectancy


def _calculate_statistical_significance(trades: List[Dict[str, Any]]) -> Tuple[float, float]:
    """
    Calculate statistical significance of trading results.

    Uses one-sample t-test to determine if mean PnL is significantly different from 0.

    Returns:
        (t_statistic, p_value)
    """
    if len(trades) < 10:
        return 0, 1  # Not enough trades for significance

    pnls = [t.get("pnl", 0) for t in trades]
    pnls = np.array(pnls)

    mean_pnl = np.mean(pnls)
    std_pnl = np.std(pnls, ddof=1)
    n = len(pnls)

    if std_pnl == 0:
        return 0, 1

    # t-statistic
    t_stat = mean_pnl / (std_pnl / np.sqrt(n))

    # p-value (two-tailed, approximate using normal distribution for large n)
    # For more accuracy, could use scipy.stats.t.sf but we avoid dependencies
    from math import erf

    def normal_cdf(x):
        return (1 + erf(x / np.sqrt(2))) / 2

    p_value = 2 * (1 - normal_cdf(abs(t_stat)))

    return t_stat, p_value


# ═══════════════════════════════════════════════════════════════════════════════
# BRAINSCORE CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_brain_score(
    metrics: Dict[str, float],
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate BrainScore v2.0 from performance metrics.

    Formula:
    BrainScore = w1 * Profitability + w2 * Risk + w3 * Consistency + w4 * Significance

    Default weights: 0.4, 0.3, 0.2, 0.1

    Args:
        metrics: Performance metrics dictionary
        weights: Optional custom weights for each component

    Returns:
        BrainScore value (0-100 scale)
    """
    if weights is None:
        weights = {
            "profitability": 0.4,
            "risk": 0.3,
            "consistency": 0.2,
            "significance": 0.1
        }

    # Calculate component scores
    profitability_score = _calculate_profitability_score(metrics)
    risk_score = _calculate_risk_score(metrics)
    consistency_score = _calculate_consistency_score(metrics)
    significance_score = _calculate_significance_score(metrics)

    # Weighted combination
    brain_score = (
        weights["profitability"] * profitability_score +
        weights["risk"] * risk_score +
        weights["consistency"] * consistency_score +
        weights["significance"] * significance_score
    )

    logger.debug(
        f"BrainScore components: P={profitability_score:.2f}, R={risk_score:.2f}, "
        f"C={consistency_score:.2f}, S={significance_score:.2f} => {brain_score:.2f}"
    )

    return round(brain_score, 4)


def _calculate_profitability_score(metrics: Dict[str, float]) -> float:
    """
    Calculate Profitability component score (0-100).

    Factors:
    - Net Profit % (40%)
    - Profit Factor (30%)
    - Win Rate (20%)
    - Expectancy (10%)
    """
    # Net Profit % - scale: -20% to +100%
    net_profit_pct = metrics.get("netProfitPct", 0)
    profit_score = _scale_metric(net_profit_pct, -20, 100, 0, 100)

    # Profit Factor - scale: 0 to 3
    pf = metrics.get("profitFactor", 0)
    pf_score = _scale_metric(pf, 0, 3, 0, 100)

    # Win Rate - scale: 30% to 70%
    winrate = metrics.get("winrate", 0)
    wr_score = _scale_metric(winrate, 30, 70, 0, 100)

    # Expectancy - scale: -100 to +500
    expectancy = metrics.get("expectancy", 0)
    exp_score = _scale_metric(expectancy, -100, 500, 0, 100)

    return profit_score * 0.4 + pf_score * 0.3 + wr_score * 0.2 + exp_score * 0.1


def _calculate_risk_score(metrics: Dict[str, float]) -> float:
    """
    Calculate Risk component score (0-100).
    Higher score = LOWER risk (better).

    Factors:
    - Max Drawdown (inverted) (40%)
    - Sharpe Ratio (30%)
    - Ulcer Index (inverted) (20%)
    - Recovery Factor (10%)
    """
    # Max Drawdown - scale: 50% (bad) to 5% (good), inverted
    max_dd = metrics.get("maxDrawdownPct", 50)
    dd_score = _scale_metric(max_dd, 50, 5, 0, 100)  # Inverted

    # Sharpe Ratio - scale: -1 to 3
    sharpe = metrics.get("sharpeRatio", 0)
    sharpe_score = _scale_metric(sharpe, -1, 3, 0, 100)

    # Ulcer Index - scale: 20 (bad) to 1 (good), inverted
    ulcer = metrics.get("ulcerIndex", 10)
    ulcer_score = _scale_metric(ulcer, 20, 1, 0, 100)  # Inverted

    # Recovery Factor - scale: 0 to 5
    recovery = metrics.get("recoveryFactor", 0)
    recovery_score = _scale_metric(recovery, 0, 5, 0, 100)

    return dd_score * 0.4 + sharpe_score * 0.3 + ulcer_score * 0.2 + recovery_score * 0.1


def _calculate_consistency_score(metrics: Dict[str, float]) -> float:
    """
    Calculate Consistency component score (0-100).

    Factors:
    - Equity Smoothness (50%)
    - Max Loss Streak (inverted) (30%)
    - Sortino Ratio (20%)
    """
    # Smoothness - scale: 0 to 1
    smoothness = metrics.get("smoothness", 0)
    smooth_score = _scale_metric(smoothness, 0, 1, 0, 100)

    # Max Loss Streak - scale: 10 (bad) to 2 (good), inverted
    loss_streak = metrics.get("maxLossStreak", 5)
    streak_score = _scale_metric(loss_streak, 10, 2, 0, 100)  # Inverted

    # Sortino Ratio - scale: -1 to 4
    sortino = metrics.get("sortinoRatio", 0)
    sortino_score = _scale_metric(sortino, -1, 4, 0, 100)

    return smooth_score * 0.5 + streak_score * 0.3 + sortino_score * 0.2


def _calculate_significance_score(metrics: Dict[str, float]) -> float:
    """
    Calculate Statistical Significance component score (0-100).

    Factors:
    - Total Trades (40%) - minimum sample size
    - P-Value (inverted) (40%) - statistical significance
    - T-Statistic (20%) - strength of result
    """
    # Total Trades - scale: 10 to 100
    total_trades = metrics.get("totalTrades", 0)
    trades_score = _scale_metric(total_trades, 10, 100, 0, 100)

    # P-Value - scale: 0.5 (bad) to 0.01 (good), inverted
    p_value = metrics.get("pValue", 0.5)
    p_score = _scale_metric(p_value, 0.5, 0.01, 0, 100)  # Inverted

    # T-Statistic - scale: 0 to 3
    t_stat = abs(metrics.get("tStatistic", 0))
    t_score = _scale_metric(t_stat, 0, 3, 0, 100)

    return trades_score * 0.4 + p_score * 0.4 + t_score * 0.2


def _scale_metric(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """
    Scale a metric from one range to another with clamping.

    Handles both normal and inverted scaling (when in_min > in_max).
    """
    # Handle inverted range
    if in_min > in_max:
        # Inverted: higher input = lower output
        in_min, in_max = in_max, in_min
        value = in_max - (value - in_min)

    # Clamp to input range
    value = max(in_min, min(in_max, value))

    # Linear scaling
    if in_max == in_min:
        return out_min

    scaled = (value - in_min) / (in_max - in_min) * (out_max - out_min) + out_min

    return max(out_min, min(out_max, scaled))


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK SCORE (for fast filtering)
# ═══════════════════════════════════════════════════════════════════════════════

def quick_score(summary: Dict[str, float]) -> float:
    """
    Quick score calculation for initial population filtering.

    Uses simplified formula:
    Score = (PF * WR * Smoothness) / (MaxDD * LossStreak * UlcerIndex)

    This is the original formula, kept for backward compatibility and speed.
    """
    pf = min(summary.get("profitFactor", 1), 5)
    wr = summary.get("winrate", 0) / 100
    smooth = summary.get("smoothness", 0.5)

    max_dd = max(summary.get("maxDrawdownPct", 20), 1)
    loss_streak = max(summary.get("maxLossStreak", 3), 1)
    ulcer = max(summary.get("ulcerIndex", 5), 0.1)

    if max_dd * loss_streak * ulcer == 0:
        return 0

    score = (pf * wr * smooth) / (max_dd * loss_streak * ulcer) * 1000

    return round(score, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# FITNESS FUNCTIONS FOR GENETIC ALGORITHM
# ═══════════════════════════════════════════════════════════════════════════════

def fitness_pareto_dominates(metrics1: Dict[str, float], metrics2: Dict[str, float]) -> bool:
    """
    Check if metrics1 Pareto-dominates metrics2.

    Pareto dominance: metrics1 is at least as good in all objectives,
    and strictly better in at least one.

    Objectives (all maximize except DD which is minimized):
    - Net Profit %
    - Sharpe Ratio
    - Win Rate
    - Max Drawdown (inverted)
    """
    objectives = [
        (metrics1.get("netProfitPct", 0), metrics2.get("netProfitPct", 0), True),  # maximize
        (metrics1.get("sharpeRatio", 0), metrics2.get("sharpeRatio", 0), True),  # maximize
        (metrics1.get("winrate", 0), metrics2.get("winrate", 0), True),  # maximize
        (metrics1.get("maxDrawdownPct", 100), metrics2.get("maxDrawdownPct", 100), False),  # minimize
    ]

    at_least_as_good = True
    strictly_better = False

    for v1, v2, maximize in objectives:
        if maximize:
            if v1 < v2:
                at_least_as_good = False
                break
            if v1 > v2:
                strictly_better = True
        else:  # minimize
            if v1 > v2:
                at_least_as_good = False
                break
            if v1 < v2:
                strictly_better = True

    return at_least_as_good and strictly_better


def get_pareto_front(
    results: List[Dict[str, Any]],
    metrics_key: str = "summary"
) -> List[Dict[str, Any]]:
    """
    Extract Pareto-optimal solutions from a list of results.

    Args:
        results: List of result dictionaries
        metrics_key: Key containing metrics in each result

    Returns:
        List of Pareto-optimal results
    """
    if not results:
        return []

    pareto_front = []

    for candidate in results:
        is_dominated = False
        candidate_metrics = candidate.get(metrics_key, candidate)

        # Check if candidate is dominated by any in current front
        for front_member in pareto_front:
            front_metrics = front_member.get(metrics_key, front_member)
            if fitness_pareto_dominates(front_metrics, candidate_metrics):
                is_dominated = True
                break

        if not is_dominated:
            # Remove any front members dominated by candidate
            pareto_front = [
                member for member in pareto_front
                if not fitness_pareto_dominates(
                    candidate_metrics,
                    member.get(metrics_key, member)
                )
            ]
            pareto_front.append(candidate)

    return pareto_front


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def add_brain_score_to_summary(summary: Dict[str, float]) -> Dict[str, float]:
    """
    Add BrainScore to an existing summary dictionary.

    Modifies the summary in-place and returns it.
    """
    brain_score = calculate_brain_score(summary)
    summary["brainScore"] = brain_score
    return summary


def rank_results_by_brain_score(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rank results by BrainScore (descending).

    Each result should have a 'summary' key with metrics.
    """
    for result in results:
        summary = result.get("summary", {})
        if "brainScore" not in summary:
            brain_score = calculate_brain_score(summary)
            result["summary"]["brainScore"] = brain_score

    return sorted(
        results,
        key=lambda r: r.get("summary", {}).get("brainScore", 0),
        reverse=True
    )


def get_score_breakdown(metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Get detailed breakdown of BrainScore components.

    Useful for debugging and explaining why a genome scored high/low.
    """
    return {
        "total": calculate_brain_score(metrics),
        "components": {
            "profitability": {
                "score": round(_calculate_profitability_score(metrics), 2),
                "weight": 0.4,
                "contribution": round(_calculate_profitability_score(metrics) * 0.4, 2)
            },
            "risk": {
                "score": round(_calculate_risk_score(metrics), 2),
                "weight": 0.3,
                "contribution": round(_calculate_risk_score(metrics) * 0.3, 2)
            },
            "consistency": {
                "score": round(_calculate_consistency_score(metrics), 2),
                "weight": 0.2,
                "contribution": round(_calculate_consistency_score(metrics) * 0.2, 2)
            },
            "significance": {
                "score": round(_calculate_significance_score(metrics), 2),
                "weight": 0.1,
                "contribution": round(_calculate_significance_score(metrics) * 0.1, 2)
            }
        },
        "key_metrics": {
            "netProfitPct": metrics.get("netProfitPct", 0),
            "profitFactor": metrics.get("profitFactor", 0),
            "winrate": metrics.get("winrate", 0),
            "maxDrawdownPct": metrics.get("maxDrawdownPct", 0),
            "sharpeRatio": metrics.get("sharpeRatio", 0),
            "totalTrades": metrics.get("totalTrades", 0)
        }
    }
