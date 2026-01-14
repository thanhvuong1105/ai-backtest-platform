# engine/numba_indicators.py
"""
Numba-accelerated indicator calculations for 10-50x speedup.
These functions are JIT-compiled for maximum performance.
"""

import numpy as np
from numba import njit, prange
from typing import Tuple


@njit(cache=True)
def calc_true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Calculate True Range - Numba optimized."""
    n = len(high)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    return tr


@njit(cache=True)
def rma_incremental(values: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate RMA (Wilder's Moving Average) incrementally.
    Same as Pine Script's ta.rma()
    """
    n = len(values)
    result = np.empty(n)
    alpha = 1.0 / period

    # First value
    if n == 0:
        return result

    # Initialize with SMA for first `period` values
    running_sum = 0.0
    for i in range(min(period, n)):
        running_sum += values[i]
        result[i] = running_sum / (i + 1)

    # If we have enough values, start RMA
    if n > period:
        rma = running_sum / period
        result[period - 1] = rma

        for i in range(period, n):
            rma = alpha * values[i] + (1 - alpha) * rma
            result[i] = rma

    return result


@njit(cache=True)
def ema_incremental(values: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate EMA incrementally.
    Same as Pine Script's ta.ema()
    """
    n = len(values)
    result = np.empty(n)
    multiplier = 2.0 / (period + 1)

    if n == 0:
        return result

    # Initialize with SMA for first `period` values
    running_sum = 0.0
    for i in range(min(period, n)):
        running_sum += values[i]
        result[i] = running_sum / (i + 1)

    # If we have enough values, start EMA
    if n > period:
        ema = running_sum / period
        result[period - 1] = ema

        for i in range(period, n):
            ema = values[i] * multiplier + ema * (1 - multiplier)
            result[i] = ema

    return result


@njit(cache=True)
def sma_rolling(values: np.ndarray, period: int) -> np.ndarray:
    """Calculate SMA with rolling window."""
    n = len(values)
    result = np.empty(n)

    if n == 0:
        return result

    running_sum = 0.0
    for i in range(n):
        running_sum += values[i]
        if i < period:
            result[i] = running_sum / (i + 1)
        else:
            running_sum -= values[i - period]
            result[i] = running_sum / period

    return result


@njit(cache=True)
def calc_supertrend(
    src: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    mult: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate SuperTrend indicator - Numba optimized.

    Returns:
        up, dn, trend, flip_buy, flip_sell arrays
    """
    n = len(src)
    up = np.empty(n)
    dn = np.empty(n)
    trend = np.empty(n)
    flip_buy = np.zeros(n)
    flip_sell = np.zeros(n)

    # Initialize
    up[0] = src[0] - mult * atr[0]
    dn[0] = src[0] + mult * atr[0]
    trend[0] = 1.0

    for i in range(1, n):
        # Basic bands
        basic_up = src[i] - mult * atr[i]
        basic_dn = src[i] + mult * atr[i]

        # Smoothing
        if close[i - 1] > up[i - 1]:
            up[i] = max(basic_up, up[i - 1])
        else:
            up[i] = basic_up

        if close[i - 1] < dn[i - 1]:
            dn[i] = min(basic_dn, dn[i - 1])
        else:
            dn[i] = basic_dn

        # Trend
        prev_trend = trend[i - 1]
        if prev_trend == -1.0 and close[i] > dn[i - 1]:
            trend[i] = 1.0
        elif prev_trend == 1.0 and close[i] < up[i - 1]:
            trend[i] = -1.0
        else:
            trend[i] = prev_trend

        # Flip signals
        if trend[i] == 1.0 and prev_trend == -1.0:
            flip_buy[i] = 1.0
        if trend[i] == -1.0 and prev_trend == 1.0:
            flip_sell[i] = 1.0

    return up, dn, trend, flip_buy, flip_sell


@njit(cache=True)
def calc_range_filter(
    src: np.ndarray,
    rng: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Range Filter - Numba optimized.

    Returns:
        f, up, dn, long_cond, short_cond, flip_buy, flip_sell arrays
    """
    n = len(src)
    f = np.empty(n)
    up = np.zeros(n)
    dn = np.zeros(n)
    long_cond = np.zeros(n)
    short_cond = np.zeros(n)
    flip_buy = np.zeros(n)
    flip_sell = np.zeros(n)

    # Initialize
    f[0] = src[0]

    for i in range(1, n):
        f_prev = f[i - 1]
        if src[i] > f_prev:
            f[i] = max(src[i] - rng[i], f_prev)
        else:
            f[i] = min(src[i] + rng[i], f_prev)

        # Direction
        up[i] = 1.0 if f[i] > f_prev else 0.0
        dn[i] = 1.0 if f[i] < f_prev else 0.0

        # Conditions
        long_cond[i] = 1.0 if (src[i] > f[i] and up[i] > 0) else 0.0
        short_cond[i] = 1.0 if (src[i] < f[i] and dn[i] > 0) else 0.0

        # Flip signals
        if long_cond[i] > 0 and f_prev < src[i - 1]:
            flip_buy[i] = 1.0
        if short_cond[i] > 0 and f_prev > src[i - 1]:
            flip_sell[i] = 1.0

    return f, up, dn, long_cond, short_cond, flip_buy, flip_sell


@njit(cache=True)
def calc_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Calculate RSI - Numba optimized."""
    n = len(close)
    rsi = np.full(n, 50.0)

    if n < 2:
        return rsi

    # Calculate changes
    changes = np.empty(n - 1)
    for i in range(1, n):
        changes[i - 1] = close[i] - close[i - 1]

    # Up/down components
    ups = np.where(changes > 0, changes, 0.0)
    downs = np.where(changes < 0, -changes, 0.0)

    # RMA of ups and downs
    up_rma = rma_incremental(ups, period)
    down_rma = rma_incremental(downs, period)

    # Calculate RSI
    for i in range(len(up_rma)):
        if down_rma[i] == 0:
            rsi[i + 1] = 100.0
        elif up_rma[i] == 0:
            rsi[i + 1] = 0.0
        else:
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + up_rma[i] / down_rma[i]))

    return rsi


@njit(cache=True)
def calc_smoothrng(abs_diff: np.ndarray, period: int, mult: float) -> np.ndarray:
    """
    Calculate smooth range for Range Filter.
    smoothrng = ema(ema(abs_diff, period), period * 2 - 1) * mult
    """
    ema1 = ema_incremental(abs_diff, period)
    ema2 = ema_incremental(ema1, period * 2 - 1)
    return ema2 * mult


@njit(cache=True)
def calc_rf_sl(
    src: np.ndarray,
    smrng: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Range Filter SL component - Numba optimized.

    Returns:
        filt, hband, lband, upward, downward, long_cond, short_cond, cond_ini arrays
    """
    n = len(src)
    filt = np.empty(n)
    hband = np.empty(n)
    lband = np.empty(n)
    upward = np.zeros(n)
    downward = np.zeros(n)
    long_cond = np.zeros(n)
    short_cond = np.zeros(n)
    cond_ini = np.zeros(n)

    # Initialize
    filt[0] = src[0]
    hband[0] = filt[0] + smrng[0]
    lband[0] = filt[0] - smrng[0]

    for i in range(1, n):
        filt_prev = filt[i - 1]

        # Filter calculation
        if src[i] > filt_prev:
            if src[i] - smrng[i] < filt_prev:
                filt[i] = filt_prev
            else:
                filt[i] = src[i] - smrng[i]
        else:
            if src[i] + smrng[i] > filt_prev:
                filt[i] = filt_prev
            else:
                filt[i] = src[i] + smrng[i]

        hband[i] = filt[i] + smrng[i]
        lband[i] = filt[i] - smrng[i]

        # Upward/downward counting
        if filt[i] > filt_prev:
            upward[i] = upward[i - 1] + 1
            downward[i] = 0
        elif filt[i] < filt_prev:
            downward[i] = downward[i - 1] + 1
            upward[i] = 0
        else:
            upward[i] = upward[i - 1]
            downward[i] = downward[i - 1]

        # Long/Short conditions
        src_prev = src[i - 1]
        if (src[i] > filt[i] and upward[i] > 0):
            long_cond[i] = 1.0
        if (src[i] < filt[i] and downward[i] > 0):
            short_cond[i] = 1.0

        # CondIni state
        if long_cond[i] > 0:
            cond_ini[i] = 1
        elif short_cond[i] > 0:
            cond_ini[i] = -1
        else:
            cond_ini[i] = cond_ini[i - 1]

    return filt, hband, lband, upward, downward, long_cond, short_cond, cond_ini


@njit(cache=True)
def calc_dual_flip_signal(
    rf_flip_buy: np.ndarray,
    st_flip_buy: np.ndarray,
    window: int = 8
) -> np.ndarray:
    """
    Calculate Dual Flip Long signal.
    True when RF and ST both flip buy within `window` bars of each other.
    """
    n = len(rf_flip_buy)
    result = np.zeros(n)

    bars_since_rf = 9999
    bars_since_st = 9999

    for i in range(n):
        # Update bars since
        if rf_flip_buy[i] > 0:
            bars_since_rf = 0
        else:
            bars_since_rf += 1

        if st_flip_buy[i] > 0:
            bars_since_st = 0
        else:
            bars_since_st += 1

        # Check dual flip
        if (rf_flip_buy[i] > 0 and bars_since_st <= window) or \
           (st_flip_buy[i] > 0 and bars_since_rf <= window):
            result[i] = 1.0

    return result


@njit(cache=True, parallel=True)
def batch_calc_atr(
    tr_batch: np.ndarray,
    period: int,
    use_rma: bool = True
) -> np.ndarray:
    """
    Calculate ATR for multiple series in parallel.
    tr_batch: shape (num_series, num_bars)
    Returns: shape (num_series, num_bars)
    """
    num_series, num_bars = tr_batch.shape
    result = np.empty((num_series, num_bars))

    for s in prange(num_series):
        if use_rma:
            result[s] = rma_incremental(tr_batch[s], period)
        else:
            result[s] = sma_rolling(tr_batch[s], period)

    return result


@njit(cache=True)
def run_backtest_loop(
    n: int,
    execution_start_idx: int,
    # Price arrays
    price_open: np.ndarray,
    price_close: np.ndarray,
    # Signal arrays (entry)
    dual_flip_long: np.ndarray,
    rsi_bull_div_signal: np.ndarray,
    rf_state: np.ndarray,
    rf_lb: np.ndarray,
    rf_sl_lband: np.ndarray,
    # Signal arrays (exit)
    rf_sl_flip_sell: np.ndarray,
    st_sl_trend: np.ndarray,
    st_sl_flip_sell: np.ndarray,
    rf_sl_state: np.ndarray,
    st_tp_dual_flip_sell: np.ndarray,
    st_tp_rsi_flip_sell: np.ndarray,
    # Config
    show_entry_long: bool,
    show_entry_rsi: bool,
    rr_mult_dual: float,
    rr_mult_rsi: float,
    # Cost config
    fee_rate: float,
    slippage: float,
    # Account config
    initial_equity: float,
    order_type: int,  # 0=percent, 1=fixed
    order_value: float,
    pyramiding: int,
    compound: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-optimized backtest loop.
    Returns arrays for entry_bars, exit_bars, entry_prices, exit_prices, sizes, pnls, entry_types, equity_curve

    Much faster than Python loop with pandas iloc.
    """
    # Pre-allocate for max possible trades
    max_trades = n // 2 + 1
    entry_bars = np.empty(max_trades, dtype=np.int64)
    exit_bars = np.empty(max_trades, dtype=np.int64)
    entry_prices = np.empty(max_trades, dtype=np.float64)
    exit_prices = np.empty(max_trades, dtype=np.float64)
    sizes = np.empty(max_trades, dtype=np.float64)
    pnls = np.empty(max_trades, dtype=np.float64)
    entry_types = np.empty(max_trades, dtype=np.int64)  # 0=dual_flip, 1=rsi

    # Equity curve
    equity_curve = np.empty(n, dtype=np.float64)

    # State
    equity = initial_equity
    trade_count = 0
    pending_entry = False
    pending_entry_type = -1  # 0=dual_flip, 1=rsi

    # Position state
    in_long = False
    in_dual_flip = False
    in_rsi = False
    open_trades = 0

    # Entry state for exit logic
    entry_price_dual = np.nan
    entry_price_rsi = np.nan
    rf_sl_at_entry_dual = np.nan
    rf_sl_at_entry_rsi = np.nan
    risk_r_dual = np.nan
    risk_r_rsi = np.nan
    current_entry_bar = -1
    current_entry_price = 0.0
    current_size = 0.0
    current_entry_type = -1

    for i in range(n):
        in_execution_range = i >= execution_start_idx

        # Reset pending at execution start
        if i == execution_start_idx:
            pending_entry = False

        # Execute pending entry at open
        if pending_entry and open_trades < pyramiding and in_execution_range:
            entry_price = price_open[i] + slippage

            # Calculate size
            if order_type == 0:  # percent
                base = equity if compound else initial_equity
                position_size = max((order_value / 100.0) * base / entry_price, 0)
            else:  # fixed
                position_size = max(order_value / entry_price, 0)

            entry_fee = entry_price * fee_rate * position_size
            equity -= entry_fee

            # Store entry
            current_entry_bar = i
            current_entry_price = entry_price
            current_size = position_size
            current_entry_type = pending_entry_type

            if pending_entry_type == 0:  # dual_flip
                entry_price_dual = entry_price
                rf_sl_at_entry_dual = rf_sl_lband[i - 1] if i > 0 else np.nan
                risk_r_dual = entry_price_dual - rf_sl_at_entry_dual if not np.isnan(rf_sl_at_entry_dual) else np.nan
                in_dual_flip = True
            else:  # rsi
                entry_price_rsi = entry_price
                rf_sl_at_entry_rsi = rf_sl_lband[i - 1] if i > 0 else np.nan
                risk_r_rsi = entry_price_rsi - rf_sl_at_entry_rsi if not np.isnan(rf_sl_at_entry_rsi) else np.nan
                in_rsi = True

            in_long = True
            open_trades += 1

        pending_entry = False
        pending_entry_type = -1

        # Check exit
        if in_long and open_trades > 0:
            should_exit = False
            close = price_close[i]

            # Check dual flip exit
            if in_dual_flip:
                sl_cond1 = rf_sl_flip_sell[i] > 0 and st_sl_trend[i] == -1
                sl_cond2 = st_sl_flip_sell[i] > 0 and rf_sl_state[i] == -1
                sl_cond3 = not np.isnan(rf_sl_at_entry_dual) and close <= rf_sl_at_entry_dual

                if sl_cond1 or sl_cond2 or sl_cond3:
                    should_exit = True
                elif st_tp_dual_flip_sell[i] > 0 and not np.isnan(entry_price_dual):
                    current_pnl = close - entry_price_dual
                    tp_target = risk_r_dual * rr_mult_dual
                    if not np.isnan(tp_target) and current_pnl > tp_target:
                        should_exit = True

            # Check RSI exit
            if in_rsi:
                valid_sl = (not np.isnan(rf_sl_at_entry_rsi) and
                           not np.isnan(entry_price_rsi) and
                           rf_sl_at_entry_rsi < entry_price_rsi)

                rsi_sl_cond1 = close <= rf_sl_at_entry_rsi
                rsi_sl_cond2 = ((rf_sl_flip_sell[i] > 0 and st_sl_trend[i] == -1) or
                               (st_sl_flip_sell[i] > 0 and rf_sl_state[i] == -1)) and close <= entry_price_rsi

                if valid_sl and (rsi_sl_cond1 or rsi_sl_cond2):
                    should_exit = True
                elif st_tp_rsi_flip_sell[i] > 0 and not np.isnan(entry_price_rsi):
                    tp_target = risk_r_rsi * rr_mult_rsi
                    current_pnl = close - entry_price_rsi
                    if not np.isnan(tp_target) and current_pnl > tp_target:
                        should_exit = True

            if should_exit:
                exit_price = close - slippage
                gross_pnl = (exit_price - current_entry_price) * current_size
                exit_fee = exit_price * fee_rate * current_size
                pnl = gross_pnl - exit_fee
                equity += pnl

                # Record trade
                entry_bars[trade_count] = current_entry_bar
                exit_bars[trade_count] = i
                entry_prices[trade_count] = current_entry_price
                exit_prices[trade_count] = exit_price
                sizes[trade_count] = current_size
                pnls[trade_count] = pnl
                entry_types[trade_count] = current_entry_type
                trade_count += 1

                # Reset state
                in_long = False
                in_dual_flip = False
                in_rsi = False
                open_trades -= 1
                entry_price_dual = np.nan
                entry_price_rsi = np.nan
                rf_sl_at_entry_dual = np.nan
                rf_sl_at_entry_rsi = np.nan
                risk_r_dual = np.nan
                risk_r_rsi = np.nan

        # Check entry signal
        if in_execution_range and not in_long and open_trades < pyramiding:
            # Dual flip entry
            if show_entry_long and dual_flip_long[i] > 0 and rf_state[i] == 1:
                close = price_close[i]
                risk_long = close - rf_lb[i] if not np.isnan(rf_lb[i]) else np.nan
                if not np.isnan(risk_long) and risk_long > 0:
                    pending_entry = True
                    pending_entry_type = 0

            # RSI entry (only if no dual flip entry)
            if not pending_entry and show_entry_rsi and rsi_bull_div_signal[i] > 0 and rf_state[i] == 1:
                close = price_close[i]
                risk_r = close - rf_sl_lband[i] if not np.isnan(rf_sl_lband[i]) else np.nan
                if not np.isnan(risk_r) and risk_r > 0:
                    pending_entry = True
                    pending_entry_type = 1

        # Record equity
        if in_execution_range:
            equity_curve[i] = equity
        else:
            equity_curve[i] = initial_equity

    # Close any open position at last bar
    if in_long and open_trades > 0:
        exit_price = price_close[n - 1] - slippage
        gross_pnl = (exit_price - current_entry_price) * current_size
        exit_fee = exit_price * fee_rate * current_size
        pnl = gross_pnl - exit_fee
        equity += pnl

        entry_bars[trade_count] = current_entry_bar
        exit_bars[trade_count] = n - 1
        entry_prices[trade_count] = current_entry_price
        exit_prices[trade_count] = exit_price
        sizes[trade_count] = current_size
        pnls[trade_count] = pnl
        entry_types[trade_count] = current_entry_type
        trade_count += 1

        equity_curve[n - 1] = equity

    # Return only used portion
    return (
        entry_bars[:trade_count],
        exit_bars[:trade_count],
        entry_prices[:trade_count],
        exit_prices[:trade_count],
        sizes[:trade_count],
        pnls[:trade_count],
        entry_types[:trade_count],
        equity_curve
    )
