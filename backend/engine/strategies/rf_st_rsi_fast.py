# engine/strategies/rf_st_rsi_fast.py
"""
Fast RF + ST + RSI Strategy using Numba-accelerated indicators.
10-50x faster than pure Python implementation.
"""

import os
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from collections import OrderedDict

# Import Numba-accelerated functions
try:
    from ..numba_indicators import (
        calc_true_range,
        rma_incremental,
        ema_incremental,
        sma_rolling,
        calc_supertrend,
        calc_range_filter,
        calc_rsi,
        calc_smoothrng,
        calc_rf_sl,
        calc_dual_flip_signal,
    )
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# ═══════════════════════════════════════════════════════
# INDICATOR RESULT CACHE
# Cache computed indicators to skip recalc for same data+params
# ═══════════════════════════════════════════════════════
_INDICATOR_RESULT_CACHE: OrderedDict = OrderedDict()
_CACHE_MAX_SIZE = int(os.getenv("INDICATOR_RESULT_CACHE_SIZE", 32))


def _make_cache_key(df_len: int, first_close: float, last_close: float, params: dict) -> str:
    """Create a cache key from data characteristics and indicator params only.

    IMPORTANT: Only include params that affect INDICATOR CALCULATION.
    Params like showDualFlip, showRSI, tp_*_rr_mult only affect entry/exit logic,
    not indicator values. This allows cache hits across different strategy configs.
    """
    # Only indicator-affecting params
    indicator_params = {
        # SuperTrend Entry
        "st_atrPeriod": params.get("st_atrPeriod", 10),
        "st_src": params.get("st_src", "hl2"),
        "st_mult": params.get("st_mult", 2.0),
        "st_useATR": params.get("st_useATR", True),
        # Range Filter Entry
        "rf_src": params.get("rf_src", "close"),
        "rf_period": params.get("rf_period", 100),
        "rf_mult": params.get("rf_mult", 3.0),
        # RSI
        "rsi_length": params.get("rsi_length", 14),
        "rsi_ma_length": params.get("rsi_ma_length", 6),
        # SL params
        "sl_st_atrPeriod": params.get("sl_st_atrPeriod", 10),
        "sl_st_src": params.get("sl_st_src", "hl2"),
        "sl_st_mult": params.get("sl_st_mult", 4.0),
        "sl_st_useATR": params.get("sl_st_useATR", True),
        "sl_rf_period": params.get("sl_rf_period", 100),
        "sl_rf_mult": params.get("sl_rf_mult", 7.0),
        # TP SuperTrend params
        "tp_dual_st_atrPeriod": params.get("tp_dual_st_atrPeriod", 10),
        "tp_dual_st_mult": params.get("tp_dual_st_mult", 2.0),
        "tp_rsi_st_atrPeriod": params.get("tp_rsi_st_atrPeriod", 10),
        "tp_rsi_st_mult": params.get("tp_rsi_st_mult", 2.0),
    }
    params_str = str(sorted(indicator_params.items()))
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
    return f"{df_len}_{first_close:.4f}_{last_close:.4f}_{params_hash}"


def _get_cached_indicators(key: str) -> Optional[Dict[str, np.ndarray]]:
    """Get cached indicator arrays if available."""
    if key in _INDICATOR_RESULT_CACHE:
        # Move to end (LRU)
        _INDICATOR_RESULT_CACHE.move_to_end(key)
        return _INDICATOR_RESULT_CACHE[key]
    return None


def _set_cached_indicators(key: str, indicators: Dict[str, np.ndarray]) -> None:
    """Cache indicator arrays with LRU eviction."""
    if len(_INDICATOR_RESULT_CACHE) >= _CACHE_MAX_SIZE:
        # Remove oldest
        _INDICATOR_RESULT_CACHE.popitem(last=False)
    _INDICATOR_RESULT_CACHE[key] = indicators


class RFSTRSIStrategyFast:
    """
    Fast implementation of RF + ST + RSI strategy.
    Uses Numba JIT-compiled functions for indicator calculations.
    """

    def __init__(self, df: pd.DataFrame, params: dict):
        # Store reference, avoid copy until needed
        self.df = df
        self.params = params
        self.debug = params.get("debug", False)
        self.logs: List[str] = []
        self._indicators_computed = False

        # Extract params
        self.show_entry_long = params.get("showDualFlip", True)
        self.show_entry_rsi = params.get("showRSI", True)

        # SuperTrend Entry params
        self.st_atr_period = params.get("st_atrPeriod", 10)
        self.st_src = params.get("st_src", "hl2")
        self.st_mult = params.get("st_mult", 2.0)
        self.st_use_atr = params.get("st_useATR", True)

        # Range Filter Entry params
        self.rf_src = params.get("rf_src", "close")
        self.rf_period = params.get("rf_period", 100)
        self.rf_mult = params.get("rf_mult", 3.0)

        # RSI params
        self.rsi_length = params.get("rsi_length", 14)
        self.rsi_ma_length = params.get("rsi_ma_length", 6)

        # SL params
        self.st_sl_atr_period = params.get("sl_st_atrPeriod", 10)
        self.st_sl_src = params.get("sl_st_src", "hl2")
        self.st_sl_mult = params.get("sl_st_mult", 4.0)
        self.st_sl_use_atr = params.get("sl_st_useATR", True)
        self.rf_sl_period = params.get("sl_rf_period", 100)
        self.rf_sl_mult = params.get("sl_rf_mult", 7.0)

        # TP Dual Flip params
        self.st_tp_dual_period = params.get("tp_dual_st_atrPeriod", 10)
        self.st_tp_dual_mult = params.get("tp_dual_st_mult", 2.0)
        self.rr_mult_dual = params.get("tp_dual_rr_mult", 1.3)

        # TP RSI params
        self.st_tp_rsi_period = params.get("tp_rsi_st_atrPeriod", 10)
        self.st_tp_rsi_mult = params.get("tp_rsi_st_mult", 2.0)
        self.rr_mult_rsi = params.get("tp_rsi_rr_mult", 1.3)

        # Position state
        self._init_position_state()

    def _log(self, bar_idx: int, time, msg: str):
        if self.debug:
            log_entry = f"[Bar {bar_idx}] {time}: {msg}"
            self.logs.append(log_entry)

    def _get_source_array(self, src_name: str) -> np.ndarray:
        """Get source array by name."""
        if src_name == "hl2":
            return ((self.df["high"] + self.df["low"]) / 2).values
        elif src_name == "hlc3":
            return ((self.df["high"] + self.df["low"] + self.df["close"]) / 3).values
        elif src_name == "ohlc4":
            return ((self.df["open"] + self.df["high"] + self.df["low"] + self.df["close"]) / 4).values
        else:
            return self.df["close"].values

    def prepare_indicators(self):
        """
        Calculate all indicators using Numba-accelerated functions.
        Much faster than pure Python loop.
        """
        if self._indicators_computed:
            return

        # Make a copy only when we need to modify
        self.df = self.df.copy()

        n = len(self.df)
        close = self.df["close"].values.astype(np.float64)
        high = self.df["high"].values.astype(np.float64)
        low = self.df["low"].values.astype(np.float64)

        # ═══════════════════════════════════════════════════════
        # CHECK CACHE - Skip calculation if same data+params
        # ═══════════════════════════════════════════════════════
        cache_key = _make_cache_key(n, close[0], close[-1], self.params)
        cached = _get_cached_indicators(cache_key)
        if cached is not None:
            # Restore from cache
            for col_name, arr in cached.items():
                self.df[col_name] = arr
            self._indicators_computed = True
            return

        # Source arrays
        st_src = self._get_source_array(self.st_src)
        rf_src = self._get_source_array(self.rf_src)
        st_sl_src = self._get_source_array(self.st_sl_src)

        # True Range
        tr = calc_true_range(high, low, close)

        # ═══════════════════════════════════════════════════════
        # SUPERTREND ENTRY
        # ═══════════════════════════════════════════════════════
        if self.st_use_atr:
            st_atr = rma_incremental(tr, self.st_atr_period)
        else:
            st_atr = sma_rolling(tr, self.st_atr_period)

        st_up, st_dn, st_trend, st_flip_buy, st_flip_sell = calc_supertrend(
            st_src, close, st_atr, self.st_mult
        )

        # ═══════════════════════════════════════════════════════
        # RANGE FILTER ENTRY
        # ═══════════════════════════════════════════════════════
        # Calculate abs diff for smoothrng
        abs_diff = np.zeros(n)
        abs_diff[1:] = np.abs(np.diff(rf_src))

        rf_rng = calc_smoothrng(abs_diff, self.rf_period, self.rf_mult)
        rf_f, rf_up, rf_dn, rf_long, rf_short, rf_flip_buy, rf_flip_sell = calc_range_filter(
            rf_src, rf_rng
        )

        # RF bands
        rf_hb = rf_f + rf_rng
        rf_lb = rf_f - rf_rng

        # RF state
        rf_state = np.zeros(n)
        for i in range(n):
            if rf_long[i] > 0:
                rf_state[i] = 1
            elif rf_short[i] > 0:
                rf_state[i] = -1
            elif i > 0:
                rf_state[i] = rf_state[i - 1]

        # ═══════════════════════════════════════════════════════
        # SUPERTREND SL
        # ═══════════════════════════════════════════════════════
        if self.st_sl_use_atr:
            st_sl_atr = rma_incremental(tr, self.st_sl_atr_period)
        else:
            st_sl_atr = sma_rolling(tr, self.st_sl_atr_period)

        st_sl_up, st_sl_dn, st_sl_trend, _, st_sl_flip_sell = calc_supertrend(
            st_sl_src, close, st_sl_atr, self.st_sl_mult
        )

        # ═══════════════════════════════════════════════════════
        # RANGE FILTER SL
        # ═══════════════════════════════════════════════════════
        rf_sl_smrng = calc_smoothrng(abs_diff, self.rf_sl_period, self.rf_sl_mult)
        rf_sl_filt, rf_sl_hband, rf_sl_lband, rf_sl_upward, rf_sl_downward, \
            rf_sl_long_cond, rf_sl_short_cond, rf_sl_cond_ini = calc_rf_sl(rf_src, rf_sl_smrng)

        # RF SL flip signals
        rf_sl_flip_buy = np.zeros(n)
        rf_sl_flip_sell = np.zeros(n)
        for i in range(1, n):
            if rf_sl_long_cond[i] > 0 and rf_sl_cond_ini[i - 1] == -1:
                rf_sl_flip_buy[i] = 1
            if rf_sl_short_cond[i] > 0 and rf_sl_cond_ini[i - 1] == 1:
                rf_sl_flip_sell[i] = 1

        # RF SL state
        rf_sl_state = np.zeros(n)
        for i in range(n):
            if rf_sl_long_cond[i] > 0:
                rf_sl_state[i] = 1
            elif rf_sl_short_cond[i] > 0:
                rf_sl_state[i] = -1
            elif i > 0:
                rf_sl_state[i] = rf_sl_state[i - 1]

        # ═══════════════════════════════════════════════════════
        # SUPERTREND TP DUAL
        # ═══════════════════════════════════════════════════════
        st_tp_dual_atr = rma_incremental(tr, self.st_tp_dual_period)
        st_tp_dual_up, st_tp_dual_dn, st_tp_dual_trend, _, st_tp_dual_flip_sell = calc_supertrend(
            st_src, close, st_tp_dual_atr, self.st_tp_dual_mult
        )

        # ═══════════════════════════════════════════════════════
        # SUPERTREND TP RSI
        # ═══════════════════════════════════════════════════════
        st_tp_rsi_atr = rma_incremental(tr, self.st_tp_rsi_period)
        st_tp_rsi_up, st_tp_rsi_dn, st_tp_rsi_trend, _, st_tp_rsi_flip_sell = calc_supertrend(
            st_src, close, st_tp_rsi_atr, self.st_tp_rsi_mult
        )

        # ═══════════════════════════════════════════════════════
        # RSI
        # ═══════════════════════════════════════════════════════
        rsi = calc_rsi(close, self.rsi_length)
        rsi_ma = sma_rolling(rsi, self.rsi_ma_length)

        # RSI crossover
        rsi_crossup = np.zeros(n)
        rsi_crossdown = np.zeros(n)
        for i in range(1, n):
            if rsi[i - 1] <= rsi_ma[i - 1] and rsi[i] > rsi_ma[i]:
                rsi_crossup[i] = 1
            if rsi[i - 1] >= rsi_ma[i - 1] and rsi[i] < rsi_ma[i]:
                rsi_crossdown[i] = 1

        # ═══════════════════════════════════════════════════════
        # RSI DIVERGENCE
        # ═══════════════════════════════════════════════════════
        rsi_bull_div_signal = np.zeros(n)
        in_seg_bull = False
        seg_min_rsi = np.nan
        seg_min_low = np.nan
        arr_rsi_bull = []
        arr_low_bull = []

        for i in range(1, n):
            if rsi_crossdown[i] > 0:
                in_seg_bull = True
                seg_min_rsi = rsi[i]
                seg_min_low = low[i]

            if in_seg_bull and rsi[i] < seg_min_rsi:
                seg_min_rsi = rsi[i]
                seg_min_low = low[i]

            if rsi_crossup[i] > 0 and in_seg_bull:
                arr_rsi_bull.append(seg_min_rsi)
                arr_low_bull.append(seg_min_low)

                if len(arr_rsi_bull) > 2:
                    arr_rsi_bull.pop(0)
                    arr_low_bull.pop(0)

                if len(arr_rsi_bull) == 2:
                    r1, r2 = arr_rsi_bull[0], arr_rsi_bull[1]
                    l1, l2 = arr_low_bull[0], arr_low_bull[1]
                    regular = (l2 < l1) and (r2 > r1)
                    hidden = (l2 > l1) and (r2 < r1)
                    if regular or hidden:
                        rsi_bull_div_signal[i] = 1

                in_seg_bull = False
                seg_min_rsi = np.nan
                seg_min_low = np.nan

        # ═══════════════════════════════════════════════════════
        # DUAL FLIP
        # ═══════════════════════════════════════════════════════
        dual_flip_long = calc_dual_flip_signal(rf_flip_buy, st_flip_buy, 8)

        # ═══════════════════════════════════════════════════════
        # ASSIGN TO DATAFRAME
        # ═══════════════════════════════════════════════════════
        self.df["st_trend"] = st_trend
        self.df["st_flipBuy"] = st_flip_buy
        self.df["st_flipSell"] = st_flip_sell
        self.df["rf_f"] = rf_f
        self.df["rf_up"] = rf_up
        self.df["rf_dn"] = rf_dn
        self.df["rf_long"] = rf_long
        self.df["rf_short"] = rf_short
        self.df["rf_flipBuy"] = rf_flip_buy
        self.df["rf_flipSell"] = rf_flip_sell
        self.df["rf_hb"] = rf_hb
        self.df["rf_lb"] = rf_lb
        self.df["rf_state"] = rf_state
        self.df["st_sl_trend"] = st_sl_trend
        self.df["st_sl_flipSell"] = st_sl_flip_sell
        self.df["rf_sl_filt"] = rf_sl_filt
        self.df["rf_sl_hband"] = rf_sl_hband
        self.df["rf_sl_lband"] = rf_sl_lband
        self.df["rf_sl_upward"] = rf_sl_upward
        self.df["rf_sl_downward"] = rf_sl_downward
        self.df["rf_sl_longCond"] = rf_sl_long_cond
        self.df["rf_sl_shortCond"] = rf_sl_short_cond
        self.df["rf_sl_CondIni"] = rf_sl_cond_ini
        self.df["rf_sl_flipBuy"] = rf_sl_flip_buy
        self.df["rf_sl_flipSell"] = rf_sl_flip_sell
        self.df["rf_sl_state"] = rf_sl_state
        self.df["st_tp_dual_trend"] = st_tp_dual_trend
        self.df["st_tp_dual_flipSell"] = st_tp_dual_flip_sell
        self.df["st_tp_rsi_trend"] = st_tp_rsi_trend
        self.df["st_tp_rsi_flipSell"] = st_tp_rsi_flip_sell
        self.df["rsi"] = rsi
        self.df["rsi_ma"] = rsi_ma
        self.df["rsi_crossup"] = rsi_crossup
        self.df["rsi_crossdown"] = rsi_crossdown
        self.df["rsi_bull_div_signal"] = rsi_bull_div_signal
        self.df["dual_flip_long"] = dual_flip_long

        # ═══════════════════════════════════════════════════════
        # SAVE TO CACHE - For next runs with same data+params
        # ═══════════════════════════════════════════════════════
        indicator_cols = {
            "st_trend": st_trend,
            "st_flipBuy": st_flip_buy,
            "st_flipSell": st_flip_sell,
            "rf_f": rf_f,
            "rf_up": rf_up,
            "rf_dn": rf_dn,
            "rf_long": rf_long,
            "rf_short": rf_short,
            "rf_flipBuy": rf_flip_buy,
            "rf_flipSell": rf_flip_sell,
            "rf_hb": rf_hb,
            "rf_lb": rf_lb,
            "rf_state": rf_state,
            "st_sl_trend": st_sl_trend,
            "st_sl_flipSell": st_sl_flip_sell,
            "rf_sl_filt": rf_sl_filt,
            "rf_sl_hband": rf_sl_hband,
            "rf_sl_lband": rf_sl_lband,
            "rf_sl_upward": rf_sl_upward,
            "rf_sl_downward": rf_sl_downward,
            "rf_sl_longCond": rf_sl_long_cond,
            "rf_sl_shortCond": rf_sl_short_cond,
            "rf_sl_CondIni": rf_sl_cond_ini,
            "rf_sl_flipBuy": rf_sl_flip_buy,
            "rf_sl_flipSell": rf_sl_flip_sell,
            "rf_sl_state": rf_sl_state,
            "st_tp_dual_trend": st_tp_dual_trend,
            "st_tp_dual_flipSell": st_tp_dual_flip_sell,
            "st_tp_rsi_trend": st_tp_rsi_trend,
            "st_tp_rsi_flipSell": st_tp_rsi_flip_sell,
            "rsi": rsi,
            "rsi_ma": rsi_ma,
            "rsi_crossup": rsi_crossup,
            "rsi_crossdown": rsi_crossdown,
            "rsi_bull_div_signal": rsi_bull_div_signal,
            "dual_flip_long": dual_flip_long,
        }
        _set_cached_indicators(cache_key, indicator_cols)

        self._indicators_computed = True

    def check_entry(self, i: int) -> bool:
        """Check if there's an entry signal at bar index i."""
        if i < 1:
            return False

        if self._in_long:
            return False

        row = self.df.iloc[i]

        dual_flip_long = row.get("dual_flip_long", 0) == 1
        rf_state = row.get("rf_state", 0)
        rf_lb = row.get("rf_lb", np.nan)
        rf_sl_lband = row.get("rf_sl_lband", np.nan)

        # Dual Flip Long Entry
        strategy_long = dual_flip_long and rf_state == 1

        if strategy_long and self.show_entry_long:
            close = row["close"]
            risk_long = close - rf_lb if not np.isnan(rf_lb) else np.nan

            if not np.isnan(risk_long) and risk_long > 0:
                self._entry_price_dual = close
                self._rf_sl_at_entry_dual = rf_sl_lband
                self._risk_r_dual = self._entry_price_dual - self._rf_sl_at_entry_dual
                self._in_long = True
                self._in_dual_flip_long = True
                self._entry_type = "dual_flip"
                return True

        # RSI Divergence Long Entry
        rsi_bull_div = row.get("rsi_bull_div_signal", 0) == 1
        rsi_long = rsi_bull_div and rf_state == 1

        if rsi_long and self.show_entry_rsi:
            close = row["close"]
            risk_r = close - rf_sl_lband if not np.isnan(rf_sl_lband) else np.nan

            if not np.isnan(risk_r) and risk_r > 0:
                self._entry_price_rsi = close
                self._rf_sl_at_entry_rsi = rf_sl_lband
                self._risk_r_rsi = risk_r
                self._in_long = True
                self._in_rsi_long = True
                self._entry_type = "rsi"
                return True

        return False

    def check_exit(self, i: int, position: dict) -> bool:
        """Check if there's an exit signal at bar index i."""
        if i < 1 or not self._in_long:
            return False

        row = self.df.iloc[i]
        close = row["close"]

        rf_sl_flip_sell = row.get("rf_sl_flipSell", 0) == 1
        st_sl_trend = row.get("st_sl_trend", 1)
        st_sl_flip_sell = row.get("st_sl_flipSell", 0) == 1
        rf_sl_state = row.get("rf_sl_state", 0)

        st_tp_dual_flip_sell = row.get("st_tp_dual_flipSell", 0) == 1
        st_tp_rsi_flip_sell = row.get("st_tp_rsi_flipSell", 0) == 1

        # Check exit for Dual Flip position
        if self._in_dual_flip_long:
            sl_cond1 = rf_sl_flip_sell and st_sl_trend == -1
            sl_cond2 = st_sl_flip_sell and rf_sl_state == -1
            sl_cond3 = not np.isnan(self._rf_sl_at_entry_dual) and close <= self._rf_sl_at_entry_dual

            if sl_cond1 or sl_cond2 or sl_cond3:
                self._reset_dual_flip_state()
                return True

            if st_tp_dual_flip_sell and not np.isnan(self._entry_price_dual):
                current_pnl = close - self._entry_price_dual
                tp_target = self._risk_r_dual * self.rr_mult_dual
                if not np.isnan(tp_target) and current_pnl > tp_target:
                    self._reset_dual_flip_state()
                    return True

        # Check exit for RSI position
        if self._in_rsi_long:
            valid_rsi_sl = (not np.isnan(self._rf_sl_at_entry_rsi) and
                           not np.isnan(self._entry_price_rsi) and
                           self._rf_sl_at_entry_rsi < self._entry_price_rsi)

            rsi_sl_cond1 = close <= self._rf_sl_at_entry_rsi
            rsi_sl_cond2 = ((rf_sl_flip_sell and st_sl_trend == -1) or
                           (st_sl_flip_sell and rf_sl_state == -1)) and close <= self._entry_price_rsi

            if valid_rsi_sl and (rsi_sl_cond1 or rsi_sl_cond2):
                self._reset_rsi_state()
                return True

            if st_tp_rsi_flip_sell and not np.isnan(self._entry_price_rsi):
                tp_target = self._risk_r_rsi * self.rr_mult_rsi
                current_pnl = close - self._entry_price_rsi
                if not np.isnan(tp_target) and current_pnl > tp_target:
                    self._reset_rsi_state()
                    return True

        return False

    def _reset_dual_flip_state(self):
        self._in_dual_flip_long = False
        if not self._in_rsi_long:
            self._in_long = False
        self._entry_price_dual = np.nan
        self._rf_sl_at_entry_dual = np.nan
        self._risk_r_dual = np.nan

    def _reset_rsi_state(self):
        self._in_rsi_long = False
        if not self._in_dual_flip_long:
            self._in_long = False
        self._entry_price_rsi = np.nan
        self._rf_sl_at_entry_rsi = np.nan
        self._risk_r_rsi = np.nan

    def _init_position_state(self):
        self._in_long = False
        self._in_dual_flip_long = False
        self._in_rsi_long = False
        self._entry_type = None
        self._entry_price_dual = np.nan
        self._rf_sl_at_entry_dual = np.nan
        self._risk_r_dual = np.nan
        self._entry_price_rsi = np.nan
        self._rf_sl_at_entry_rsi = np.nan
        self._risk_r_rsi = np.nan

    def get_price_arrays(self) -> Dict[str, np.ndarray]:
        """Return numpy arrays for fast backtest loop."""
        return {
            "time": self.df["time"].values,
            "open": self.df["open"].values.astype(np.float64),
            "close": self.df["close"].values.astype(np.float64),
            "high": self.df["high"].values.astype(np.float64),
            "low": self.df["low"].values.astype(np.float64),
        }

    def get_indicator_arrays(self) -> Dict[str, np.ndarray]:
        """Return indicator arrays for fast backtest loop."""
        if not self._indicators_computed:
            self.prepare_indicators()

        return {
            "dual_flip_long": self.df["dual_flip_long"].values,
            "rf_state": self.df["rf_state"].values,
            "rf_lb": self.df["rf_lb"].values,
            "rf_sl_lband": self.df["rf_sl_lband"].values,
            "rsi_bull_div_signal": self.df["rsi_bull_div_signal"].values,
            "rf_sl_flipSell": self.df["rf_sl_flipSell"].values,
            "st_sl_trend": self.df["st_sl_trend"].values,
            "st_sl_flipSell": self.df["st_sl_flipSell"].values,
            "rf_sl_state": self.df["rf_sl_state"].values,
            "st_tp_dual_flipSell": self.df["st_tp_dual_flipSell"].values,
            "st_tp_rsi_flipSell": self.df["st_tp_rsi_flipSell"].values,
        }
