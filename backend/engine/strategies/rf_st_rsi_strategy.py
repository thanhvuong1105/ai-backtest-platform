# engine/strategies/rf_st_rsi_strategy.py
"""
RF + ST + RSI Divergence Combined Strategy
Converted from Pine Script to Python - 100% logic match

Original: "RF + ST + RSI Div Combined Strategy" by InschoShayne17122002

Components:
- Range Filter (RF) for trend detection
- SuperTrend (ST) for trend confirmation
- RSI Divergence for momentum signals
- Dual Flip entry logic
- Multiple TP/SL mechanisms
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any


class RFSTRSIStrategy:
    """
    Bar-by-bar strategy execution matching Pine Script logic exactly.
    No lookahead, no vectorization that changes logic.
    """

    def __init__(self, df: pd.DataFrame, params: dict):
        self.df = df.copy()
        self.params = params
        self.debug = params.get("debug", False)
        self.logs: List[str] = []

        # Extract params with defaults matching Pine Script
        # Entry params
        self.show_entry_long = params.get("showDualFlip", True)
        self.show_entry_rsi = params.get("showRSI", True)

        # SuperTrend Entry params
        self.st_atr_period = params.get("st_atrPeriod", 10)
        self.st_src = params.get("st_src", "hl2")  # hl2, close, hlc3, ohlc4
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

        # Initialize indicator columns
        self._init_columns()

        # Initialize position tracking state
        self._init_position_state()

    def _log(self, bar_idx: int, time, msg: str):
        """Debug logging"""
        if self.debug:
            log_entry = f"[Bar {bar_idx}] {time}: {msg}"
            self.logs.append(log_entry)
            if self.debug:
                print(log_entry)

    def _init_columns(self):
        """Initialize all indicator columns"""
        n = len(self.df)

        # Source calculations
        self.df["hl2"] = (self.df["high"] + self.df["low"]) / 2
        self.df["hlc3"] = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
        self.df["ohlc4"] = (self.df["open"] + self.df["high"] + self.df["low"] + self.df["close"]) / 4

        # True Range
        self.df["tr"] = self._calc_true_range()

        # Initialize indicator arrays (will be computed bar-by-bar)
        for col in [
            # SuperTrend Entry
            "st_atr", "st_up", "st_dn", "st_trend", "st_flipBuy", "st_flipSell",
            # Range Filter Entry
            "rf_rng", "rf_f", "rf_up", "rf_dn", "rf_long", "rf_short",
            "rf_flipBuy", "rf_flipSell", "rf_hb", "rf_lb", "rf_state",
            # SuperTrend SL
            "st_sl_atr", "st_sl_up", "st_sl_dn", "st_sl_trend", "st_sl_flipSell",
            # Range Filter SL
            "rf_sl_smrng", "rf_sl_filt", "rf_sl_hband", "rf_sl_lband",
            "rf_sl_upward", "rf_sl_downward", "rf_sl_state",
            "rf_sl_longCond", "rf_sl_shortCond", "rf_sl_CondIni",
            "rf_sl_flipBuy", "rf_sl_flipSell",
            # SuperTrend TP Dual
            "st_tp_dual_atr", "st_tp_dual_up", "st_tp_dual_dn",
            "st_tp_dual_trend", "st_tp_dual_flipSell",
            # SuperTrend TP RSI
            "st_tp_rsi_atr", "st_tp_rsi_up", "st_tp_rsi_dn",
            "st_tp_rsi_trend", "st_tp_rsi_flipSell",
            # RSI
            "rsi", "rsi_ma", "rsi_crossup", "rsi_crossdown",
            # Dual Flip
            "bars_rf_flip", "bars_st_flip", "dual_flip_long",
            # RSI Divergence
            "rsi_bull_div_signal",
            # Signals
            "entry_long", "entry_rsi_long", "exit_signal",
        ]:
            self.df[col] = np.nan

    def _calc_true_range(self) -> np.ndarray:
        """Calculate True Range"""
        high = self.df["high"].values
        low = self.df["low"].values
        close = self.df["close"].values

        tr = np.zeros(len(self.df))
        tr[0] = high[0] - low[0]

        for i in range(1, len(self.df)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )

        return tr

    def _get_source(self, src_name: str, idx: int) -> float:
        """Get source value at index"""
        if src_name == "hl2":
            return self.df["hl2"].iloc[idx]
        elif src_name == "hlc3":
            return self.df["hlc3"].iloc[idx]
        elif src_name == "ohlc4":
            return self.df["ohlc4"].iloc[idx]
        else:
            return self.df["close"].iloc[idx]

    def _ema(self, values: List[float], period: int) -> float:
        """Calculate EMA for the last value given history"""
        if len(values) < period:
            return np.mean(values) if values else 0

        multiplier = 2 / (period + 1)
        ema = np.mean(values[:period])

        for i in range(period, len(values)):
            ema = values[i] * multiplier + ema * (1 - multiplier)

        return ema

    def _ema_update(self, prev_ema: float, prev_sum: float, new_value: float, period: int, count: int):
        """
        Calculate EMA incrementally (O(1) per update).
        Returns: (new_ema, new_sum)

        For count < period: accumulate sum and return SMA
        For count >= period: use EMA formula
        """
        multiplier = 2 / (period + 1)

        if count < period:
            # Accumulate for initial SMA
            new_sum = prev_sum + new_value
            new_ema = new_sum / (count + 1)
            return new_ema, new_sum
        elif count == period:
            # First EMA value = SMA of first `period` values
            new_sum = prev_sum + new_value
            new_ema = new_sum / period
            return new_ema, new_sum
        else:
            # Standard EMA update
            new_ema = new_value * multiplier + prev_ema * (1 - multiplier)
            return new_ema, prev_sum  # sum no longer needed

    def _rma(self, values: List[float], period: int) -> float:
        """Calculate RMA (Wilder's Moving Average) - Pine's ta.rma"""
        if len(values) < 1:
            return 0

        alpha = 1 / period

        if len(values) < period:
            return np.mean(values)

        # Initialize with SMA
        rma = np.mean(values[:period])

        for i in range(period, len(values)):
            rma = alpha * values[i] + (1 - alpha) * rma

        return rma

    def _rma_update(self, prev_rma: float, prev_sum: float, new_value: float, period: int, count: int):
        """
        Calculate RMA incrementally (O(1) per update).
        Returns: (new_rma, new_sum)

        For count < period: accumulate sum and return SMA
        For count >= period: use RMA formula
        """
        alpha = 1 / period

        if count < period:
            # Accumulate for initial SMA
            new_sum = prev_sum + new_value
            new_rma = new_sum / (count + 1)
            return new_rma, new_sum
        elif count == period:
            # First RMA value = SMA of first `period` values
            new_sum = prev_sum + new_value
            new_rma = new_sum / period
            return new_rma, new_sum
        else:
            # Standard RMA update (Wilder's smoothing)
            new_rma = alpha * new_value + (1 - alpha) * prev_rma
            return new_rma, prev_sum  # sum no longer needed

    def _sma(self, values: List[float], period: int) -> float:
        """Calculate SMA for the last value"""
        if len(values) < period:
            return np.mean(values) if values else 0
        return np.mean(values[-period:])

    def _atr(self, tr_history: List[float], period: int, use_atr: bool = True) -> float:
        """Calculate ATR - uses EMA if use_atr=True, else SMA"""
        if use_atr:
            return self._ema(tr_history, period)
        else:
            return self._sma(tr_history, period)

    def _nz(self, value, default=0):
        """Pine Script nz() function"""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return value

    def prepare_indicators(self):
        """
        Calculate all indicators bar-by-bar to match Pine Script execution.
        This is the main computation that must match TradingView exactly.
        Optimized: collect values in numpy arrays, assign at end.
        """
        n = len(self.df)
        close = self.df["close"].values
        high = self.df["high"].values
        low = self.df["low"].values
        tr = self.df["tr"].values

        # Pre-allocate output arrays for performance
        out_st_atr = np.full(n, np.nan)
        out_st_up = np.full(n, np.nan)
        out_st_dn = np.full(n, np.nan)
        out_st_trend = np.full(n, np.nan)
        out_st_flipBuy = np.zeros(n)
        out_st_flipSell = np.zeros(n)
        out_rf_rng = np.full(n, np.nan)
        out_rf_f = np.full(n, np.nan)
        out_rf_up = np.zeros(n)
        out_rf_dn = np.zeros(n)
        out_rf_long = np.zeros(n)
        out_rf_short = np.zeros(n)
        out_rf_flipBuy = np.zeros(n)
        out_rf_flipSell = np.zeros(n)
        out_rf_hb = np.full(n, np.nan)
        out_rf_lb = np.full(n, np.nan)
        out_rf_state = np.zeros(n)
        out_st_sl_atr = np.full(n, np.nan)
        out_st_sl_up = np.full(n, np.nan)
        out_st_sl_dn = np.full(n, np.nan)
        out_st_sl_trend = np.full(n, np.nan)
        out_st_sl_flipSell = np.zeros(n)
        out_rf_sl_smrng = np.full(n, np.nan)
        out_rf_sl_filt = np.full(n, np.nan)
        out_rf_sl_hband = np.full(n, np.nan)
        out_rf_sl_lband = np.full(n, np.nan)
        out_rf_sl_upward = np.zeros(n)
        out_rf_sl_downward = np.zeros(n)
        out_rf_sl_longCond = np.zeros(n)
        out_rf_sl_shortCond = np.zeros(n)
        out_rf_sl_CondIni = np.zeros(n)
        out_rf_sl_flipBuy = np.zeros(n)
        out_rf_sl_flipSell = np.zeros(n)
        out_rf_sl_state = np.zeros(n)
        out_st_tp_dual_atr = np.full(n, np.nan)
        out_st_tp_dual_up = np.full(n, np.nan)
        out_st_tp_dual_dn = np.full(n, np.nan)
        out_st_tp_dual_trend = np.full(n, np.nan)
        out_st_tp_dual_flipSell = np.zeros(n)
        out_st_tp_rsi_atr = np.full(n, np.nan)
        out_st_tp_rsi_up = np.full(n, np.nan)
        out_st_tp_rsi_dn = np.full(n, np.nan)
        out_st_tp_rsi_trend = np.full(n, np.nan)
        out_st_tp_rsi_flipSell = np.zeros(n)
        out_rsi = np.full(n, np.nan)
        out_rsi_ma = np.full(n, np.nan)
        out_rsi_crossup = np.zeros(n)
        out_rsi_crossdown = np.zeros(n)
        out_rsi_bull_div_signal = np.zeros(n)
        out_bars_rf_flip = np.full(n, np.nan)
        out_bars_st_flip = np.full(n, np.nan)
        out_dual_flip_long = np.zeros(n)

        # History arrays for moving averages
        tr_history = []
        rf_src_history = []
        rf_abs_diff_history = []
        rf_sl_abs_diff_history = []
        rsi_change_history = []
        rsi_up_history = []
        rsi_down_history = []
        rsi_history = []

        # State variables (matching Pine's var declarations)
        # SuperTrend Entry
        st_trend = 1.0
        st_up_prev = np.nan
        st_dn_prev = np.nan

        # Range Filter Entry
        rf_f = np.nan
        rf_state = 0

        # SuperTrend SL
        st_sl_trend = 1.0
        st_sl_up_prev = np.nan
        st_sl_dn_prev = np.nan

        # Range Filter SL
        rf_sl_filt = np.nan
        rf_sl_upward = 0.0
        rf_sl_downward = 0.0
        rf_sl_CondIni = 0
        rf_sl_state = 0

        # SuperTrend TP Dual
        st_tp_dual_trend = 1.0
        st_tp_dual_up_prev = np.nan
        st_tp_dual_dn_prev = np.nan

        # SuperTrend TP RSI
        st_tp_rsi_trend = 1.0
        st_tp_rsi_up_prev = np.nan
        st_tp_rsi_dn_prev = np.nan

        # RSI Divergence state
        in_seg_bull = False
        seg_min_rsi = np.nan
        seg_min_low = np.nan
        arr_rsi_bull = []
        arr_low_bull = []

        # Bars since flip tracking
        bars_since_rf_flip = 9999
        bars_since_st_flip = 9999

        # EMA history for Range Filter
        rf_ema1_history = []
        rf_sl_avrng_history = []

        # ===== INCREMENTAL EMA/RMA STATE VARIABLES =====
        # For O(n) performance instead of O(n²)
        # ATR for SuperTrend Entry
        st_atr_rma = 0.0  # RMA for ATR (Pine ta.atr)
        st_atr_sum = 0.0  # keep for initialization
        # ATR for SuperTrend SL
        st_sl_atr_rma = 0.0
        st_sl_atr_sum = 0.0
        # ATR for SuperTrend TP Dual
        st_tp_dual_atr_rma = 0.0
        st_tp_dual_atr_sum = 0.0
        # ATR for SuperTrend TP RSI
        st_tp_rsi_atr_rma = 0.0
        st_tp_rsi_atr_sum = 0.0
        # Range Filter EMA 1
        rf_ema1 = 0.0
        rf_ema1_sum = 0.0
        # Range Filter EMA 2 (on ema1)
        rf_ema2 = 0.0
        rf_ema2_sum = 0.0
        # Range Filter SL avrng EMA
        rf_sl_avrng_ema = 0.0
        rf_sl_avrng_sum = 0.0
        # Range Filter SL smrng EMA (on avrng)
        rf_sl_smrng_ema = 0.0
        rf_sl_smrng_sum = 0.0
        # RSI RMA values
        rsi_up_rma = 0.0
        rsi_up_sum = 0.0
        rsi_down_rma = 0.0
        rsi_down_sum = 0.0
        # RSI MA (SMA) - need circular buffer
        rsi_ma_buffer = []
        rsi_ma_prev = 0.0  # Store previous MA for crossover detection

        for i in range(n):
            # Get current values
            c = close[i]
            h = high[i]
            lo = low[i]
            c_prev = close[i - 1] if i > 0 else c

            # Source values
            st_src_val = self._get_source(self.st_src, i)
            rf_src_val = self._get_source(self.rf_src, i)
            st_sl_src_val = self._get_source(self.st_sl_src, i)

            # Update history
            tr_history.append(tr[i])
            rf_src_history.append(rf_src_val)

            if i > 0:
                rf_abs_diff_history.append(abs(rf_src_val - rf_src_history[-2]))
                rf_sl_abs_diff_history.append(abs(rf_src_val - rf_src_history[-2]))
            else:
                rf_abs_diff_history.append(0)
                rf_sl_abs_diff_history.append(0)

            # ═══════════════════════════════════════════════════════
            # SUPERTREND ENTRY
            # ═══════════════════════════════════════════════════════
            # ATR using RMA (ta.atr) or SMA depending on flag (exact Pine logic)
            if self.st_use_atr:
                st_atr_rma, st_atr_sum = self._rma_update(st_atr_rma, st_atr_sum, tr[i], self.st_atr_period, i)
                st_atr_val = st_atr_rma
            else:
                # Pine uses sma(tr, period) when changeATR is false
                st_atr_val = self._sma(tr_history, self.st_atr_period)
            out_st_atr[i] = st_atr_val

            st_up = st_src_val - self.st_mult * st_atr_val
            st_dn = st_src_val + self.st_mult * st_atr_val

            # Smoothing logic: st_up := close[1] > nz(st_up[1]) ? math.max(st_up, nz(st_up[1])) : st_up
            if i > 0:
                up1 = st_up_prev if not np.isnan(st_up_prev) else st_up
                dn1 = st_dn_prev if not np.isnan(st_dn_prev) else st_dn
                if c_prev > up1:
                    st_up = max(st_up, up1)
                if c_prev < dn1:
                    st_dn = min(st_dn, dn1)
            else:
                up1 = st_up
                dn1 = st_dn

            out_st_up[i] = st_up
            out_st_dn[i] = st_dn

            # Trend logic
            st_trend_prev = st_trend
            if st_trend == -1 and c > dn1:
                st_trend = 1
            elif st_trend == 1 and c < up1:
                st_trend = -1

            out_st_trend[i] = st_trend

            # Flip signals
            st_flip_buy = (st_trend == 1 and st_trend_prev == -1)
            st_flip_sell = (st_trend == -1 and st_trend_prev == 1)
            out_st_flipBuy[i] = 1 if st_flip_buy else 0
            out_st_flipSell[i] = 1 if st_flip_sell else 0

            st_up_prev = st_up
            st_dn_prev = st_dn

            # ═══════════════════════════════════════════════════════
            # RANGE FILTER ENTRY
            # ═══════════════════════════════════════════════════════
            # smoothrng(x, t, m) => ta.ema(ta.ema(math.abs(x - x[1]), t), t * 2 - 1) * m
            # Incremental EMA 1
            abs_diff = rf_abs_diff_history[-1] if rf_abs_diff_history else 0
            rf_ema1, rf_ema1_sum = self._ema_update(rf_ema1, rf_ema1_sum, abs_diff, self.rf_period, i)
            # Incremental EMA 2 (on ema1 values)
            rf_ema2, rf_ema2_sum = self._ema_update(rf_ema2, rf_ema2_sum, rf_ema1, self.rf_period * 2 - 1, i)
            rf_rng = rf_ema2 * self.rf_mult

            out_rf_rng[i] = rf_rng

            # Range filter value
            # f := src > nz(f[1]) ? math.max(src - rng, nz(f[1])) : math.min(src + rng, nz(f[1], src))
            rf_f_prev = rf_f
            if rf_src_val > self._nz(rf_f_prev, rf_src_val):
                rf_f = max(rf_src_val - rf_rng, self._nz(rf_f_prev, rf_src_val))
            else:
                rf_f = min(rf_src_val + rf_rng, self._nz(rf_f_prev, rf_src_val))

            out_rf_f[i] = rf_f

            # Direction
            rf_up = rf_f > self._nz(rf_f_prev, rf_f)
            rf_dn = rf_f < self._nz(rf_f_prev, rf_f)
            rf_long = rf_src_val > rf_f and rf_up
            rf_short = rf_src_val < rf_f and rf_dn

            out_rf_up[i] = 1 if rf_up else 0
            out_rf_dn[i] = 1 if rf_dn else 0
            out_rf_long[i] = 1 if rf_long else 0
            out_rf_short[i] = 1 if rf_short else 0

            # Bands
            rf_hb = rf_f + rf_rng
            rf_lb = rf_f - rf_rng
            out_rf_hb[i] = rf_hb
            out_rf_lb[i] = rf_lb

            # State update BEFORE flip detection (matching Pine Script rf_CondIni)
            # Pine: rf_CondIni := rf_longCond ? 1 : rf_shortCond ? -1 : rf_CondIni[1]
            rf_state_prev = rf_state
            if rf_long:
                rf_state = 1
            elif rf_short:
                rf_state = -1
            # else: rf_state stays the same (matches rf_CondIni[1])
            out_rf_state[i] = rf_state

            # Flip signals - EXACT match to Pine Script
            # Pine: rf_flipBuy = rf_longCond and rf_CondIni[1] == -1
            # Pine: rf_flipSell = rf_shortCond and rf_CondIni[1] == 1
            # rf_state_prev is rf_CondIni[1], rf_long is rf_longCond
            rf_flip_buy = rf_long and rf_state_prev == -1
            rf_flip_sell = rf_short and rf_state_prev == 1

            out_rf_flipBuy[i] = 1 if rf_flip_buy else 0
            out_rf_flipSell[i] = 1 if rf_flip_sell else 0

            # ═══════════════════════════════════════════════════════
            # SUPERTREND SL
            # ═══════════════════════════════════════════════════════
            # ATR using incremental EMA (O(1) per bar)
            if self.st_sl_use_atr:
                st_sl_atr_rma, st_sl_atr_sum = self._rma_update(st_sl_atr_rma, st_sl_atr_sum, tr[i], self.st_sl_atr_period, i)
                st_sl_atr_val = st_sl_atr_rma
            else:
                st_sl_atr_val = self._sma(tr_history, self.st_sl_atr_period)
            out_st_sl_atr[i] = st_sl_atr_val

            st_sl_up = st_sl_src_val - self.st_sl_mult * st_sl_atr_val
            st_sl_dn = st_sl_src_val + self.st_sl_mult * st_sl_atr_val

            if i > 0:
                if c_prev > self._nz(st_sl_up_prev):
                    st_sl_up = max(st_sl_up, self._nz(st_sl_up_prev))
                if c_prev < self._nz(st_sl_dn_prev):
                    st_sl_dn = min(st_sl_dn, self._nz(st_sl_dn_prev))

            out_st_sl_up[i] = st_sl_up
            out_st_sl_dn[i] = st_sl_dn

            st_sl_trend_prev = st_sl_trend
            if st_sl_trend == -1 and c > self._nz(st_sl_dn_prev):
                st_sl_trend = 1
            elif st_sl_trend == 1 and c < self._nz(st_sl_up_prev):
                st_sl_trend = -1

            out_st_sl_trend[i] = st_sl_trend

            st_sl_flip_sell = (st_sl_trend == -1 and st_sl_trend_prev == 1)
            out_st_sl_flipSell[i] = 1 if st_sl_flip_sell else 0

            st_sl_up_prev = st_sl_up
            st_sl_dn_prev = st_sl_dn

            # ═══════════════════════════════════════════════════════
            # RANGE FILTER SL (matching indicator exactly)
            # ═══════════════════════════════════════════════════════
            rf_sl_wper = self.rf_sl_period * 2 - 1

            # rf_sl_avrng = ta.ema(math.abs(rf_sl_src - rf_sl_src[1]), rf_sl_per)
            # Incremental EMA 1 for SL
            rf_sl_abs_diff = rf_sl_abs_diff_history[-1] if rf_sl_abs_diff_history else 0
            rf_sl_avrng_ema, rf_sl_avrng_sum = self._ema_update(rf_sl_avrng_ema, rf_sl_avrng_sum, rf_sl_abs_diff, self.rf_sl_period, i)
            rf_sl_avrng = rf_sl_avrng_ema
            rf_sl_avrng_history.append(rf_sl_avrng)

            # rf_sl_smrng = ta.ema(rf_sl_avrng, rf_sl_wper) * rf_sl_multV
            # Incremental EMA 2 for SL (on avrng)
            rf_sl_smrng_ema, rf_sl_smrng_sum = self._ema_update(rf_sl_smrng_ema, rf_sl_smrng_sum, rf_sl_avrng, rf_sl_wper, i)
            rf_sl_smrng = rf_sl_smrng_ema * self.rf_sl_mult
            out_rf_sl_smrng[i] = rf_sl_smrng

            # Filter calculation (exact Pine logic)
            rf_sl_filt_prev = rf_sl_filt
            if np.isnan(rf_sl_filt_prev):
                rf_sl_filt = rf_src_val
            else:
                if rf_src_val > self._nz(rf_sl_filt_prev):
                    if rf_src_val - rf_sl_smrng < self._nz(rf_sl_filt_prev):
                        rf_sl_filt = self._nz(rf_sl_filt_prev)
                    else:
                        rf_sl_filt = rf_src_val - rf_sl_smrng
                else:
                    if rf_src_val + rf_sl_smrng > self._nz(rf_sl_filt_prev):
                        rf_sl_filt = self._nz(rf_sl_filt_prev)
                    else:
                        rf_sl_filt = rf_src_val + rf_sl_smrng

            out_rf_sl_filt[i] = rf_sl_filt

            rf_sl_hband = rf_sl_filt + rf_sl_smrng
            rf_sl_lband = rf_sl_filt - rf_sl_smrng
            out_rf_sl_hband[i] = rf_sl_hband
            out_rf_sl_lband[i] = rf_sl_lband

            # Upward/downward counting
            if rf_sl_filt > self._nz(rf_sl_filt_prev, rf_sl_filt):
                rf_sl_upward = self._nz(rf_sl_upward) + 1
            elif rf_sl_filt < self._nz(rf_sl_filt_prev, rf_sl_filt):
                rf_sl_upward = 0

            if rf_sl_filt < self._nz(rf_sl_filt_prev, rf_sl_filt):
                rf_sl_downward = self._nz(rf_sl_downward) + 1
            elif rf_sl_filt > self._nz(rf_sl_filt_prev, rf_sl_filt):
                rf_sl_downward = 0

            out_rf_sl_upward[i] = rf_sl_upward
            out_rf_sl_downward[i] = rf_sl_downward

            # Long/Short conditions
            rf_src_prev = rf_src_history[-2] if len(rf_src_history) > 1 else rf_src_val
            rf_sl_longCond = ((rf_src_val > rf_sl_filt and rf_src_val > rf_src_prev and rf_sl_upward > 0) or
                             (rf_src_val > rf_sl_filt and rf_src_val < rf_src_prev and rf_sl_upward > 0))
            rf_sl_shortCond = ((rf_src_val < rf_sl_filt and rf_src_val < rf_src_prev and rf_sl_downward > 0) or
                              (rf_src_val < rf_sl_filt and rf_src_val > rf_src_prev and rf_sl_downward > 0))

            out_rf_sl_longCond[i] = 1 if rf_sl_longCond else 0
            out_rf_sl_shortCond[i] = 1 if rf_sl_shortCond else 0

            # CondIni state
            rf_sl_CondIni_prev = rf_sl_CondIni
            if rf_sl_longCond:
                rf_sl_CondIni = 1
            elif rf_sl_shortCond:
                rf_sl_CondIni = -1

            out_rf_sl_CondIni[i] = rf_sl_CondIni

            # Flip signals for SL
            rf_sl_flip_buy = rf_sl_longCond and self._nz(rf_sl_CondIni_prev) == -1
            rf_sl_flip_sell = rf_sl_shortCond and self._nz(rf_sl_CondIni_prev) == 1

            out_rf_sl_flipBuy[i] = 1 if rf_sl_flip_buy else 0
            out_rf_sl_flipSell[i] = 1 if rf_sl_flip_sell else 0

            # State
            rf_sl_state_prev = rf_sl_state
            if rf_sl_longCond:
                rf_sl_state = 1
            elif rf_sl_shortCond:
                rf_sl_state = -1

            out_rf_sl_state[i] = rf_sl_state

            # ═══════════════════════════════════════════════════════
            # SUPERTREND TP DUAL FLIP
            # ═══════════════════════════════════════════════════════
            # ATR using incremental EMA (O(1) per bar)
            st_tp_dual_atr_rma, st_tp_dual_atr_sum = self._rma_update(st_tp_dual_atr_rma, st_tp_dual_atr_sum, tr[i], self.st_tp_dual_period, i)
            st_tp_dual_atr_val = st_tp_dual_atr_rma
            out_st_tp_dual_atr[i] = st_tp_dual_atr_val

            st_tp_dual_up = st_src_val - self.st_tp_dual_mult * st_tp_dual_atr_val
            st_tp_dual_dn = st_src_val + self.st_tp_dual_mult * st_tp_dual_atr_val

            if i > 0:
                if c_prev > self._nz(st_tp_dual_up_prev):
                    st_tp_dual_up = max(st_tp_dual_up, self._nz(st_tp_dual_up_prev))
                if c_prev < self._nz(st_tp_dual_dn_prev):
                    st_tp_dual_dn = min(st_tp_dual_dn, self._nz(st_tp_dual_dn_prev))

            out_st_tp_dual_up[i] = st_tp_dual_up
            out_st_tp_dual_dn[i] = st_tp_dual_dn

            st_tp_dual_trend_prev = st_tp_dual_trend
            if st_tp_dual_trend == -1 and c > self._nz(st_tp_dual_dn_prev):
                st_tp_dual_trend = 1
            elif st_tp_dual_trend == 1 and c < self._nz(st_tp_dual_up_prev):
                st_tp_dual_trend = -1

            out_st_tp_dual_trend[i] = st_tp_dual_trend

            st_tp_dual_flip_sell = (st_tp_dual_trend == -1 and st_tp_dual_trend_prev == 1)
            out_st_tp_dual_flipSell[i] = 1 if st_tp_dual_flip_sell else 0

            st_tp_dual_up_prev = st_tp_dual_up
            st_tp_dual_dn_prev = st_tp_dual_dn

            # ═══════════════════════════════════════════════════════
            # SUPERTREND TP RSI
            # ═══════════════════════════════════════════════════════
            # ATR using incremental EMA (O(1) per bar)
            st_tp_rsi_atr_rma, st_tp_rsi_atr_sum = self._rma_update(st_tp_rsi_atr_rma, st_tp_rsi_atr_sum, tr[i], self.st_tp_rsi_period, i)
            st_tp_rsi_atr_val = st_tp_rsi_atr_rma
            out_st_tp_rsi_atr[i] = st_tp_rsi_atr_val

            st_tp_rsi_up = st_src_val - self.st_tp_rsi_mult * st_tp_rsi_atr_val
            st_tp_rsi_dn = st_src_val + self.st_tp_rsi_mult * st_tp_rsi_atr_val

            if i > 0:
                if c_prev > self._nz(st_tp_rsi_up_prev):
                    st_tp_rsi_up = max(st_tp_rsi_up, self._nz(st_tp_rsi_up_prev))
                if c_prev < self._nz(st_tp_rsi_dn_prev):
                    st_tp_rsi_dn = min(st_tp_rsi_dn, self._nz(st_tp_rsi_dn_prev))

            out_st_tp_rsi_up[i] = st_tp_rsi_up
            out_st_tp_rsi_dn[i] = st_tp_rsi_dn

            st_tp_rsi_trend_prev = st_tp_rsi_trend
            if st_tp_rsi_trend == -1 and c > self._nz(st_tp_rsi_dn_prev):
                st_tp_rsi_trend = 1
            elif st_tp_rsi_trend == 1 and c < self._nz(st_tp_rsi_up_prev):
                st_tp_rsi_trend = -1

            out_st_tp_rsi_trend[i] = st_tp_rsi_trend

            st_tp_rsi_flip_sell = (st_tp_rsi_trend == -1 and st_tp_rsi_trend_prev == 1)
            out_st_tp_rsi_flipSell[i] = 1 if st_tp_rsi_flip_sell else 0

            st_tp_rsi_up_prev = st_tp_rsi_up
            st_tp_rsi_dn_prev = st_tp_rsi_dn

            # ═══════════════════════════════════════════════════════
            # RSI CALCULATION
            # ═══════════════════════════════════════════════════════
            rsi_crossup = False
            rsi_crossdown = False
            rsi_bull_div_signal = False

            if i > 0:
                change = c - c_prev
                rsi_change_history.append(change)

                # Up/down components
                up_val = max(change, 0)
                down_val = -min(change, 0)

                rsi_up_history.append(up_val)
                rsi_down_history.append(down_val)

                # Incremental RMA for RSI (O(1) per bar)
                # Note: RMA count is (i-1) since RSI starts at i=1
                rsi_count = i - 1  # 0-based count for RMA
                rsi_up_rma, rsi_up_sum = self._rma_update(rsi_up_rma, rsi_up_sum, up_val, self.rsi_length, rsi_count)
                rsi_down_rma, rsi_down_sum = self._rma_update(rsi_down_rma, rsi_down_sum, down_val, self.rsi_length, rsi_count)

                up_rma = rsi_up_rma
                down_rma = rsi_down_rma

                if down_rma == 0:
                    rsi_val = 100
                elif up_rma == 0:
                    rsi_val = 0
                else:
                    rsi_val = 100 - (100 / (1 + up_rma / down_rma))

                rsi_history.append(rsi_val)
                out_rsi[i] = rsi_val

                # RSI MA using circular buffer for O(1) SMA
                rsi_ma_buffer.append(rsi_val)
                if len(rsi_ma_buffer) > self.rsi_ma_length:
                    rsi_ma_buffer.pop(0)
                rsi_ma = sum(rsi_ma_buffer) / len(rsi_ma_buffer)
                out_rsi_ma[i] = rsi_ma

                # Crossover detection - store previous MA for next iteration
                if len(rsi_history) >= 2:
                    rsi_prev = rsi_history[-2]
                    # Use previous rsi_ma_prev stored from last iteration
                    rsi_crossup = rsi_prev <= rsi_ma_prev and rsi_val > rsi_ma
                    rsi_crossdown = rsi_prev >= rsi_ma_prev and rsi_val < rsi_ma

                rsi_ma_prev = rsi_ma  # Store for next iteration

                out_rsi_crossup[i] = 1 if rsi_crossup else 0
                out_rsi_crossdown[i] = 1 if rsi_crossdown else 0

                # ═══════════════════════════════════════════════════════
                # RSI BULLISH DIVERGENCE DETECTION
                # ═══════════════════════════════════════════════════════

                # Start bearish segment on crossdown
                if rsi_crossdown:
                    in_seg_bull = True
                    seg_min_rsi = rsi_val
                    seg_min_low = lo

                # Track minimum during segment
                if in_seg_bull and rsi_val < seg_min_rsi:
                    seg_min_rsi = rsi_val
                    seg_min_low = lo

                # End segment on crossup, check for divergence
                if rsi_crossup and in_seg_bull:
                    arr_rsi_bull.append(seg_min_rsi)
                    arr_low_bull.append(seg_min_low)

                    # Keep only last 2
                    if len(arr_rsi_bull) > 2:
                        arr_rsi_bull.pop(0)
                        arr_low_bull.pop(0)

                    # Check for divergence
                    if len(arr_rsi_bull) == 2:
                        r1, r2 = arr_rsi_bull[0], arr_rsi_bull[1]
                        l1, l2 = arr_low_bull[0], arr_low_bull[1]

                        # Regular bullish: lower low price, higher low RSI
                        regular = (l2 < l1) and (r2 > r1)
                        # Hidden bullish: higher low price, lower low RSI
                        hidden = (l2 > l1) and (r2 < r1)

                        rsi_bull_div_signal = regular or hidden

                    in_seg_bull = False
                    seg_min_rsi = np.nan
                    seg_min_low = np.nan

                out_rsi_bull_div_signal[i] = 1 if rsi_bull_div_signal else 0
            else:
                out_rsi[i] = 50
                out_rsi_ma[i] = 50

            # ═══════════════════════════════════════════════════════
            # DUAL FLIP TRACKING
            # ═══════════════════════════════════════════════════════
            # Pine's ta.barssince() returns bars since condition was true
            # If condition is true NOW, it returns 0
            # We need to track this correctly

            # Reset FIRST if flip happens on current bar
            if rf_flip_buy:
                bars_since_rf_flip = 0
            else:
                bars_since_rf_flip += 1

            if st_flip_buy:
                bars_since_st_flip = 0
            else:
                bars_since_st_flip += 1

            out_bars_rf_flip[i] = bars_since_rf_flip
            out_bars_st_flip[i] = bars_since_st_flip

            # Dual flip long condition
            # When rf_flip_buy is True, bars_since_rf_flip = 0
            # We check if the OTHER flip happened within 8 bars
            dual_flip_long = (rf_flip_buy and bars_since_st_flip <= 8) or (st_flip_buy and bars_since_rf_flip <= 8)
            out_dual_flip_long[i] = 1 if dual_flip_long else 0

        # ═══════════════════════════════════════════════════════
        # BULK ASSIGNMENT TO DATAFRAME (optimized)
        # ═══════════════════════════════════════════════════════
        self.df["st_atr"] = out_st_atr
        self.df["st_up"] = out_st_up
        self.df["st_dn"] = out_st_dn
        self.df["st_trend"] = out_st_trend
        self.df["st_flipBuy"] = out_st_flipBuy
        self.df["st_flipSell"] = out_st_flipSell
        self.df["rf_rng"] = out_rf_rng
        self.df["rf_f"] = out_rf_f
        self.df["rf_up"] = out_rf_up
        self.df["rf_dn"] = out_rf_dn
        self.df["rf_long"] = out_rf_long
        self.df["rf_short"] = out_rf_short
        self.df["rf_flipBuy"] = out_rf_flipBuy
        self.df["rf_flipSell"] = out_rf_flipSell
        self.df["rf_hb"] = out_rf_hb
        self.df["rf_lb"] = out_rf_lb
        self.df["rf_state"] = out_rf_state
        self.df["st_sl_atr"] = out_st_sl_atr
        self.df["st_sl_up"] = out_st_sl_up
        self.df["st_sl_dn"] = out_st_sl_dn
        self.df["st_sl_trend"] = out_st_sl_trend
        self.df["st_sl_flipSell"] = out_st_sl_flipSell
        self.df["rf_sl_smrng"] = out_rf_sl_smrng
        self.df["rf_sl_filt"] = out_rf_sl_filt
        self.df["rf_sl_hband"] = out_rf_sl_hband
        self.df["rf_sl_lband"] = out_rf_sl_lband
        self.df["rf_sl_upward"] = out_rf_sl_upward
        self.df["rf_sl_downward"] = out_rf_sl_downward
        self.df["rf_sl_longCond"] = out_rf_sl_longCond
        self.df["rf_sl_shortCond"] = out_rf_sl_shortCond
        self.df["rf_sl_CondIni"] = out_rf_sl_CondIni
        self.df["rf_sl_flipBuy"] = out_rf_sl_flipBuy
        self.df["rf_sl_flipSell"] = out_rf_sl_flipSell
        self.df["rf_sl_state"] = out_rf_sl_state
        self.df["st_tp_dual_atr"] = out_st_tp_dual_atr
        self.df["st_tp_dual_up"] = out_st_tp_dual_up
        self.df["st_tp_dual_dn"] = out_st_tp_dual_dn
        self.df["st_tp_dual_trend"] = out_st_tp_dual_trend
        self.df["st_tp_dual_flipSell"] = out_st_tp_dual_flipSell
        self.df["st_tp_rsi_atr"] = out_st_tp_rsi_atr
        self.df["st_tp_rsi_up"] = out_st_tp_rsi_up
        self.df["st_tp_rsi_dn"] = out_st_tp_rsi_dn
        self.df["st_tp_rsi_trend"] = out_st_tp_rsi_trend
        self.df["st_tp_rsi_flipSell"] = out_st_tp_rsi_flipSell
        self.df["rsi"] = out_rsi
        self.df["rsi_ma"] = out_rsi_ma
        self.df["rsi_crossup"] = out_rsi_crossup
        self.df["rsi_crossdown"] = out_rsi_crossdown
        self.df["rsi_bull_div_signal"] = out_rsi_bull_div_signal
        self.df["bars_rf_flip"] = out_bars_rf_flip
        self.df["bars_st_flip"] = out_bars_st_flip
        self.df["dual_flip_long"] = out_dual_flip_long

    def generate_signals(self) -> pd.DataFrame:
        """
        Generate entry/exit signals based on prepared indicators.
        Must be called after prepare_indicators().
        """
        n = len(self.df)

        # State for signal generation
        in_long = False
        in_dual_flip_long = False
        in_rsi_long = False

        entry_price_dual = np.nan
        entry_price_rsi = np.nan
        rf_sl_at_entry_dual = np.nan
        rf_sl_at_entry_rsi = np.nan
        risk_r_dual = np.nan
        risk_r_rsi = np.nan
        dual_flip_signal_close = np.nan
        dual_flip_signal_rf_lb = np.nan

        signals = []

        for i in range(n):
            row = self.df.iloc[i]
            time = row["time"]
            close = row["close"]

            signal = {
                "time": time,
                "bar_index": i,
                "entry_long": 0,
                "entry_rsi_long": 0,
                "exit_dual": 0,
                "exit_rsi": 0,
                "exit_reason": None,
            }

            # Get indicator values
            dual_flip_long = row.get("dual_flip_long", 0) == 1
            rf_state = row.get("rf_state", 0)
            rf_lb = row.get("rf_lb", np.nan)
            rf_sl_lband = row.get("rf_sl_lband", np.nan)
            rsi_bull_div = row.get("rsi_bull_div_signal", 0) == 1

            rf_sl_flip_sell = row.get("rf_sl_flipSell", 0) == 1
            st_sl_trend = row.get("st_sl_trend", 1)
            st_sl_flip_sell = row.get("st_sl_flipSell", 0) == 1
            rf_sl_state = row.get("rf_sl_state", 0)

            st_tp_dual_flip_sell = row.get("st_tp_dual_flipSell", 0) == 1
            st_tp_rsi_flip_sell = row.get("st_tp_rsi_flipSell", 0) == 1

            # ═══════════════════════════════════════════════════════
            # TRACK DUAL FLIP SIGNAL VALUES
            # ═══════════════════════════════════════════════════════
            if dual_flip_long:
                dual_flip_signal_close = close
                dual_flip_signal_rf_lb = rf_lb

            # ═══════════════════════════════════════════════════════
            # ENTRY LOGIC
            # ═══════════════════════════════════════════════════════
            # Dual Flip Long Entry
            strategy_long = dual_flip_long and rf_state == 1
            trigger_long = strategy_long and not in_long

            if trigger_long and self.show_entry_long:
                risk_long = dual_flip_signal_close - dual_flip_signal_rf_lb if not np.isnan(dual_flip_signal_rf_lb) else np.nan
                valid_long = not np.isnan(risk_long) and risk_long > 0

                if valid_long:
                    entry_price_dual = dual_flip_signal_close
                    rf_sl_at_entry_dual = rf_sl_lband
                    risk_r_dual = entry_price_dual - rf_sl_at_entry_dual

                    in_long = True
                    in_dual_flip_long = True
                    signal["entry_long"] = 1

                    self._log(i, time, f"ENTRY Dual Flip Long @ {entry_price_dual:.2f}, SL level: {rf_sl_at_entry_dual:.2f}, Risk: {risk_r_dual:.2f}")

            # RSI Divergence Long Entry
            rsi_long = rsi_bull_div and rf_state == 1

            if rsi_long and not in_long and self.show_entry_rsi:
                entry_price_rsi = close
                rf_sl_at_entry_rsi = rf_sl_lband
                risk_r_rsi = entry_price_rsi - rf_sl_at_entry_rsi

                if risk_r_rsi > 0:
                    in_long = True
                    in_rsi_long = True
                    signal["entry_rsi_long"] = 1

                    self._log(i, time, f"ENTRY RSI Long @ {entry_price_rsi:.2f}, SL level: {rf_sl_at_entry_rsi:.2f}, Risk: {risk_r_rsi:.2f}")

            # ═══════════════════════════════════════════════════════
            # EXIT LOGIC FOR DUAL FLIP
            # ═══════════════════════════════════════════════════════
            if in_long and in_dual_flip_long:
                # SL conditions
                sl_cond1 = rf_sl_flip_sell and st_sl_trend == -1
                sl_cond2 = st_sl_flip_sell and rf_sl_state == -1
                sl_cond3 = not np.isnan(rf_sl_at_entry_dual) and close <= rf_sl_at_entry_dual

                sl_condition = sl_cond1 or sl_cond2 or sl_cond3

                if sl_condition:
                    signal["exit_dual"] = 1
                    signal["exit_reason"] = "SL"
                    in_dual_flip_long = False
                    if not in_rsi_long:
                        in_long = False

                    self._log(i, time, f"EXIT Dual Flip SL @ {close:.2f}")

                    entry_price_dual = np.nan
                    rf_sl_at_entry_dual = np.nan

                # TP condition
                elif st_tp_dual_flip_sell and not np.isnan(entry_price_dual):
                    current_pnl = close - entry_price_dual
                    tp_target = risk_r_dual * self.rr_mult_dual

                    if not np.isnan(tp_target) and current_pnl > tp_target:
                        signal["exit_dual"] = 1
                        signal["exit_reason"] = "TP"
                        in_dual_flip_long = False
                        if not in_rsi_long:
                            in_long = False

                        self._log(i, time, f"EXIT Dual Flip TP @ {close:.2f}, PnL: {current_pnl:.2f}")

                        entry_price_dual = np.nan
                        rf_sl_at_entry_dual = np.nan

            # ═══════════════════════════════════════════════════════
            # EXIT LOGIC FOR RSI LONG
            # ═══════════════════════════════════════════════════════
            if in_long and in_rsi_long:
                valid_rsi_sl = (not np.isnan(rf_sl_at_entry_rsi) and
                               not np.isnan(entry_price_rsi) and
                               rf_sl_at_entry_rsi < entry_price_rsi)

                # SL condition
                rsi_sl_cond1 = close <= rf_sl_at_entry_rsi
                rsi_sl_cond2 = ((rf_sl_flip_sell and st_sl_trend == -1) or
                               (st_sl_flip_sell and rf_sl_state == -1)) and close <= entry_price_rsi

                rsi_sl_condition = valid_rsi_sl and (rsi_sl_cond1 or rsi_sl_cond2)

                if rsi_sl_condition:
                    signal["exit_rsi"] = 1
                    signal["exit_reason"] = "RSI SL"
                    in_rsi_long = False
                    if not in_dual_flip_long:
                        in_long = False

                    self._log(i, time, f"EXIT RSI SL @ {close:.2f}")

                    entry_price_rsi = np.nan
                    rf_sl_at_entry_rsi = np.nan

                # TP condition
                elif st_tp_rsi_flip_sell and not np.isnan(entry_price_rsi):
                    tp_rsi_target = risk_r_rsi * self.rr_mult_rsi
                    current_pnl_rsi = close - entry_price_rsi

                    if not np.isnan(tp_rsi_target) and current_pnl_rsi > tp_rsi_target:
                        signal["exit_rsi"] = 1
                        signal["exit_reason"] = "TP RSI"
                        in_rsi_long = False
                        if not in_dual_flip_long:
                            in_long = False

                        self._log(i, time, f"EXIT RSI TP @ {close:.2f}, PnL: {current_pnl_rsi:.2f}")

                        entry_price_rsi = np.nan
                        rf_sl_at_entry_rsi = np.nan

            signals.append(signal)

        return pd.DataFrame(signals)


    def check_entry(self, i: int) -> bool:
        """
        Check if there's an entry signal at bar index i.
        Called by backtest_engine.

        Signal at bar i means entry will execute at Open of bar i+1.
        We store rf_sl_lband from bar i (signal bar) to match TradingView behavior.
        """
        if i < 1:
            return False

        row = self.df.iloc[i]

        # Check if already in position (using state tracking)
        if self._in_long:
            return False

        # Dual Flip Long Entry
        dual_flip_long = row.get("dual_flip_long", 0) == 1
        rf_state = row.get("rf_state", 0)
        rf_lb = row.get("rf_lb", np.nan)
        rf_sl_lband = row.get("rf_sl_lband", np.nan)

        strategy_long = dual_flip_long and rf_state == 1

        if strategy_long and self.show_entry_long:
            # Track signal values
            self._dual_flip_signal_close = row["close"]
            self._dual_flip_signal_rf_lb = rf_lb

            risk_long = self._dual_flip_signal_close - self._dual_flip_signal_rf_lb if not np.isnan(self._dual_flip_signal_rf_lb) else np.nan
            valid_long = not np.isnan(risk_long) and risk_long > 0

            if valid_long:
                # Store entry price as signal bar close (will be updated by backtest_engine)
                # But store SL level from signal bar - this is what TradingView does
                self._entry_price_dual = self._dual_flip_signal_close
                self._rf_sl_at_entry_dual = rf_sl_lband  # SL from signal bar
                self._risk_r_dual = self._entry_price_dual - self._rf_sl_at_entry_dual
                self._in_long = True
                self._in_dual_flip_long = True
                self._entry_type = "dual_flip"
                self._log(i, row["time"], f"ENTRY Dual Flip Long @ {self._entry_price_dual:.2f}, SL: {rf_sl_lband:.2f}")
                return True

        # RSI Divergence Long Entry
        rsi_bull_div = row.get("rsi_bull_div_signal", 0) == 1
        rsi_long = rsi_bull_div and rf_state == 1

        if rsi_long and self.show_entry_rsi:
            entry_price = row["close"]
            risk_r = entry_price - rf_sl_lband if not np.isnan(rf_sl_lband) else np.nan

            if not np.isnan(risk_r) and risk_r > 0:
                self._entry_price_rsi = entry_price
                self._rf_sl_at_entry_rsi = rf_sl_lband  # SL from signal bar
                self._risk_r_rsi = risk_r
                self._in_long = True
                self._in_rsi_long = True
                self._entry_type = "rsi"
                self._log(i, row["time"], f"ENTRY RSI Long @ {entry_price:.2f}, SL: {rf_sl_lband:.2f}")
                return True

        return False

    def check_exit(self, i: int, position: dict) -> bool:
        """
        Check if there's an exit signal at bar index i.
        Called by backtest_engine.
        """
        if i < 1:
            return False

        if not self._in_long:
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
            # SL conditions
            sl_cond1 = rf_sl_flip_sell and st_sl_trend == -1
            sl_cond2 = st_sl_flip_sell and rf_sl_state == -1
            sl_cond3 = not np.isnan(self._rf_sl_at_entry_dual) and close <= self._rf_sl_at_entry_dual

            if sl_cond1 or sl_cond2 or sl_cond3:
                self._log(i, row["time"], f"EXIT Dual Flip SL @ {close:.2f}")
                self._reset_dual_flip_state()
                return True

            # TP condition
            if st_tp_dual_flip_sell and not np.isnan(self._entry_price_dual):
                current_pnl = close - self._entry_price_dual
                tp_target = self._risk_r_dual * self.rr_mult_dual

                if not np.isnan(tp_target) and current_pnl > tp_target:
                    self._log(i, row["time"], f"EXIT Dual Flip TP @ {close:.2f}")
                    self._reset_dual_flip_state()
                    return True

        # Check exit for RSI position
        if self._in_rsi_long:
            valid_rsi_sl = (not np.isnan(self._rf_sl_at_entry_rsi) and
                           not np.isnan(self._entry_price_rsi) and
                           self._rf_sl_at_entry_rsi < self._entry_price_rsi)

            # SL condition
            rsi_sl_cond1 = close <= self._rf_sl_at_entry_rsi
            rsi_sl_cond2 = ((rf_sl_flip_sell and st_sl_trend == -1) or
                           (st_sl_flip_sell and rf_sl_state == -1)) and close <= self._entry_price_rsi

            if valid_rsi_sl and (rsi_sl_cond1 or rsi_sl_cond2):
                self._log(i, row["time"], f"EXIT RSI SL @ {close:.2f}")
                self._reset_rsi_state()
                return True

            # TP condition
            if st_tp_rsi_flip_sell and not np.isnan(self._entry_price_rsi):
                tp_target = self._risk_r_rsi * self.rr_mult_rsi
                current_pnl = close - self._entry_price_rsi

                if not np.isnan(tp_target) and current_pnl > tp_target:
                    self._log(i, row["time"], f"EXIT RSI TP @ {close:.2f}")
                    self._reset_rsi_state()
                    return True

        return False

    def _reset_dual_flip_state(self):
        """Reset Dual Flip position state"""
        self._in_dual_flip_long = False
        if not self._in_rsi_long:
            self._in_long = False
        self._entry_price_dual = np.nan
        self._rf_sl_at_entry_dual = np.nan
        self._risk_r_dual = np.nan

    def _reset_rsi_state(self):
        """Reset RSI position state"""
        self._in_rsi_long = False
        if not self._in_dual_flip_long:
            self._in_long = False
        self._entry_price_rsi = np.nan
        self._rf_sl_at_entry_rsi = np.nan
        self._risk_r_rsi = np.nan

    def _init_position_state(self):
        """Initialize position tracking state"""
        self._in_long = False
        self._in_dual_flip_long = False
        self._in_rsi_long = False
        self._entry_type = None

        # Dual Flip tracking
        self._entry_price_dual = np.nan
        self._rf_sl_at_entry_dual = np.nan
        self._risk_r_dual = np.nan
        self._dual_flip_signal_close = np.nan
        self._dual_flip_signal_rf_lb = np.nan

        # RSI tracking
        self._entry_price_rsi = np.nan
        self._rf_sl_at_entry_rsi = np.nan
        self._risk_r_rsi = np.nan


def create_strategy(df: pd.DataFrame, params: dict) -> RFSTRSIStrategy:
    """Factory function for strategy creation"""
    return RFSTRSIStrategy(df, params)
