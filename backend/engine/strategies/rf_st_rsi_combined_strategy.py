# engine/strategies/rf_st_rsi_combined_strategy.py
"""
RF + ST + RSI Divergence Combined Strategy (Long & Short)
Converted from Pine Script v6 to Python - 100% logic match

Original: "RF + ST + RSI Div Combined Strategy (Long & Short)" by InschoShayne17122002

Components:
- Range Filter (RF) for trend detection
- SuperTrend (ST) for trend confirmation
- RSI Divergence for momentum signals (Bullish & Bearish)
- Dual Flip entry logic (8 bars Long, 12 bars Short)
- Side-specific SL/TP mechanisms (Long vs Short)
- Position reversal logic
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple


class RFSTRSICombinedStrategy:
    """
    Bar-by-bar strategy execution matching Pine Script logic exactly.
    Supports both Long and Short positions with separate SL/TP parameters.
    No lookahead, no vectorization that changes logic.
    """

    def __init__(self, df: pd.DataFrame, params: dict):
        self.df = df.copy()
        self.params = params
        self.debug = params.get("debug", False)
        self.logs: List[str] = []
        self.debug_events: List[dict] = []  # Store debug events for frontend

        # ═══════════════════════════════════════════════════════
        # ENABLE SETTINGS
        # ═══════════════════════════════════════════════════════
        self.enable_long = params.get("enableLong", True)
        self.enable_short = params.get("enableShort", True)

        # ═══════════════════════════════════════════════════════
        # ENTRY SETTINGS
        # ═══════════════════════════════════════════════════════
        self.show_entry_long = params.get("showEntryLong", True)
        self.show_entry_short = params.get("showEntryShort", True)
        self.show_entry_rsi_L = params.get("showEntryRSI_L", True)
        self.show_entry_rsi_S = params.get("showEntryRSI_S", True)

        # SuperTrend Entry params
        self.st_atr_period = params.get("st_atrPeriod", 10)
        self.st_src = params.get("st_src", "hl2")
        self.st_mult = params.get("st_mult", 2.0)
        self.st_use_atr = params.get("st_changeATR", True)

        # Range Filter Entry params
        self.rf_src = params.get("rf_src_in", "close")
        self.rf_period = params.get("rf_period", 100)
        self.rf_mult = params.get("rf_mult", 3.0)

        # RSI params
        self.rsi_length = params.get("lenRSI", 14)
        self.rsi_ma_length = params.get("lenMA", 6)

        # Dual Flip bars threshold
        # Pine Script: dualFlipLong = (rf_flipBuy and barsST_B <= 8) or (st_flipBuy and barsRF_B <= 8)
        # Pine Script: dualFlipShort = (rf_flipSell and barsST_S <= 12) or (st_flipSell and barsRF_S <= 12)
        self.dual_flip_bars_long = params.get("dualFlipBarsLong", 8)
        self.dual_flip_bars_short = params.get("dualFlipBarsShort", 12)

        # ═══════════════════════════════════════════════════════
        # STOP LOSS - LONG
        # Defaults match Pine Script source for exact behavior
        # Pine: st_sl_atrPeriod_L=10, st_sl_mult_L=3.0, rf_sl_period_L=20, rf_sl_mult_L=15.0
        # ═══════════════════════════════════════════════════════
        self.st_sl_atr_period_L = params.get("st_sl_atrPeriod_L", 10)
        self.st_sl_src_L = params.get("st_sl_src_L", "hl2")
        self.st_sl_mult_L = params.get("st_sl_mult_L", 3.0)  # Pine Script default
        self.st_sl_use_atr_L = params.get("st_sl_useATR_L", True)
        self.rf_sl_period_L = params.get("rf_sl_period_L", 20)  # Pine Script default
        self.rf_sl_mult_L = params.get("rf_sl_mult_L", 15.0)  # Pine Script default

        # ═══════════════════════════════════════════════════════
        # STOP LOSS - SHORT
        # Defaults match Pine Script source for exact behavior
        # Pine: st_sl_atrPeriod_S=10, st_sl_mult_S=2.0, rf_sl_period_S=20, rf_sl_mult_S=3.0
        # ═══════════════════════════════════════════════════════
        self.st_sl_atr_period_S = params.get("st_sl_atrPeriod_S", 10)
        self.st_sl_src_S = params.get("st_sl_src_S", "hl2")
        self.st_sl_mult_S = params.get("st_sl_mult_S", 2.0)  # Pine Script default
        self.st_sl_use_atr_S = params.get("st_sl_useATR_S", True)
        self.rf_sl_period_S = params.get("rf_sl_period_S", 20)  # Pine Script default
        self.rf_sl_mult_S = params.get("rf_sl_mult_S", 3.0)  # Pine Script default

        # ═══════════════════════════════════════════════════════
        # TAKE PROFIT - DUAL FLIP LONG
        # Pine: st_tp_dual_period_L=10, st_tp_dual_mult_L=2.0, rr_mult_dual_L=4.0
        # ═══════════════════════════════════════════════════════
        self.st_tp_dual_period_L = params.get("st_tp_dual_period_L", 10)
        self.st_tp_dual_mult_L = params.get("st_tp_dual_mult_L", 2.0)
        self.rr_mult_dual_L = params.get("rr_mult_dual_L", 4.0)  # Pine Script default

        # ═══════════════════════════════════════════════════════
        # TAKE PROFIT - DUAL FLIP SHORT
        # Pine: st_tp_dual_period_S=10, st_tp_dual_mult_S=2.0, rr_mult_dual_S=0.75
        # ═══════════════════════════════════════════════════════
        self.st_tp_dual_period_S = params.get("st_tp_dual_period_S", 10)
        self.st_tp_dual_mult_S = params.get("st_tp_dual_mult_S", 2.0)
        self.rr_mult_dual_S = params.get("rr_mult_dual_S", 0.75)  # Pine Script default

        # ═══════════════════════════════════════════════════════
        # TAKE PROFIT - RSI LONG
        # Pine: st_tp_rsi1_period_L=10, st_tp_rsi1_mult_L=2.0, rr_mult_rsi1_L=4.0
        # ═══════════════════════════════════════════════════════
        self.st_tp_rsi_period_L = params.get("st_tp_rsi1_period_L", 10)
        self.st_tp_rsi_mult_L = params.get("st_tp_rsi1_mult_L", 2.0)
        self.rr_mult_rsi_L = params.get("rr_mult_rsi1_L", 4.0)  # Pine Script default

        # ═══════════════════════════════════════════════════════
        # TAKE PROFIT - RSI SHORT
        # Pine: st_tp_rsi1_period_S=10, st_tp_rsi1_mult_S=2.0, rr_mult_rsi1_S=0.75
        # ═══════════════════════════════════════════════════════
        self.st_tp_rsi_period_S = params.get("st_tp_rsi1_period_S", 10)
        self.st_tp_rsi_mult_S = params.get("st_tp_rsi1_mult_S", 2.0)
        self.rr_mult_rsi_S = params.get("rr_mult_rsi1_S", 0.75)  # Pine Script default

        # Initialize indicator columns
        self._init_columns()

        # Initialize position tracking state
        self._init_position_state()

    def _log(self, bar_idx: int, time, msg: str):
        """Debug logging"""
        if self.debug:
            log_entry = f"[Bar {bar_idx}] {time}: {msg}"
            self.logs.append(log_entry)
            # Print to stderr to avoid mixing with JSON output in subprocess
            import sys
            print(log_entry, file=sys.stderr)

    def _add_debug_event(self, bar_idx: int, time, event_type: str, side: str, price: float, details: dict = None):
        """Add debug event for frontend visualization"""
        # Convert time to JSON-serializable format
        if hasattr(time, 'timestamp'):
            # pandas Timestamp or datetime
            time_val = int(time.timestamp() * 1000)  # milliseconds
        elif isinstance(time, (int, float)):
            time_val = int(time)
        else:
            time_val = str(time)

        # Convert any non-JSON-serializable values in details
        safe_details = {}
        if details:
            for k, v in details.items():
                if hasattr(v, 'timestamp'):
                    safe_details[k] = int(v.timestamp() * 1000)
                elif isinstance(v, (np.floating, np.integer)):
                    safe_details[k] = float(v) if isinstance(v, np.floating) else int(v)
                elif isinstance(v, np.bool_):
                    safe_details[k] = bool(v)
                elif isinstance(v, float) and np.isnan(v):
                    safe_details[k] = None
                elif isinstance(v, bool):
                    safe_details[k] = v
                else:
                    safe_details[k] = v

        event = {
            "bar": bar_idx,
            "time": time_val,
            "type": event_type,  # "entry", "exit_sl", "exit_tp"
            "side": side,  # "Long", "Short", "RSI Long", "RSI Short"
            "price": float(price) if isinstance(price, (np.floating, np.integer)) else price,
            "details": safe_details
        }
        self.debug_events.append(event)

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
        indicator_cols = [
            # SuperTrend MAIN (Entry)
            "st_atr", "st_up", "st_dn", "st_trend", "st_flipBuy", "st_flipSell",
            # Range Filter GLOBAL (Entry)
            "rf_rng", "rf_f", "rf_up", "rf_dn", "rf_long", "rf_short",
            "rf_flipBuy", "rf_flipSell", "rf_hb", "rf_lb", "rf_state",
            # SuperTrend SL LONG
            "st_sl_L_atr", "st_sl_L_up", "st_sl_L_dn", "st_sl_L_trend", "st_sl_L_flipSell",
            # SuperTrend SL SHORT
            "st_sl_S_atr", "st_sl_S_up", "st_sl_S_dn", "st_sl_S_trend", "st_sl_S_flipBuy",
            # Range Filter SL LONG
            "rf_sl_L_smrng", "rf_sl_L_filt", "rf_sl_L_hband", "rf_sl_L_lband",
            "rf_sl_L_upward", "rf_sl_L_downward", "rf_sl_L_state",
            "rf_sl_L_longCond", "rf_sl_L_shortCond", "rf_sl_L_CondIni",
            "rf_sl_L_flipBuy", "rf_sl_L_flipSell",
            # Range Filter SL SHORT
            "rf_sl_S_smrng", "rf_sl_S_filt", "rf_sl_S_hband", "rf_sl_S_lband",
            "rf_sl_S_upward", "rf_sl_S_downward", "rf_sl_S_state",
            "rf_sl_S_longCond", "rf_sl_S_shortCond", "rf_sl_S_CondIni",
            "rf_sl_S_flipBuy", "rf_sl_S_flipSell",
            # SuperTrend TP Dual LONG
            "st_tp_dual_L_atr", "st_tp_dual_L_up", "st_tp_dual_L_dn",
            "st_tp_dual_L_trend", "st_tp_dual_L_flipSell",
            # SuperTrend TP Dual SHORT
            "st_tp_dual_S_atr", "st_tp_dual_S_up", "st_tp_dual_S_dn",
            "st_tp_dual_S_trend", "st_tp_dual_S_flipBuy",
            # SuperTrend TP RSI LONG
            "st_tp_rsi_L_atr", "st_tp_rsi_L_up", "st_tp_rsi_L_dn",
            "st_tp_rsi_L_trend", "st_tp_rsi_L_flipSell",
            # SuperTrend TP RSI SHORT
            "st_tp_rsi_S_atr", "st_tp_rsi_S_up", "st_tp_rsi_S_dn",
            "st_tp_rsi_S_trend", "st_tp_rsi_S_flipBuy",
            # RSI
            "rsi", "rsi_ma", "rsi_crossup", "rsi_crossdown",
            # Dual Flip
            "bars_rf_flipBuy", "bars_rf_flipSell", "bars_st_flipBuy", "bars_st_flipSell",
            "dual_flip_long", "dual_flip_short",
            # RSI Divergence
            "rsi_bull_div_signal", "rsi_bear_div_signal",
            # Entry signals
            "entry_long", "entry_short", "entry_rsi_long", "entry_rsi_short",
        ]

        for col in indicator_cols:
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

    def _ema_update(self, prev_ema: float, prev_sum: float, new_value: float, period: int, count: int):
        """
        Calculate EMA incrementally (O(1) per update).
        Returns: (new_ema, new_sum)

        For count < period: accumulate sum and return SMA
        For count >= period: use EMA formula

        This matches the Long Only strategy implementation.
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

    def _rma_update(self, prev_rma: float, prev_sum: float, new_value: float, period: int, count: int):
        """Calculate RMA incrementally (O(1) per update)."""
        alpha = 1 / period

        if count < period:
            new_sum = prev_sum + new_value
            new_rma = new_sum / (count + 1)
            return new_rma, new_sum
        elif count == period:
            new_sum = prev_sum + new_value
            new_rma = new_sum / period
            return new_rma, new_sum
        else:
            new_rma = alpha * new_value + (1 - alpha) * prev_rma
            return new_rma, prev_sum

    def _sma(self, values: List[float], period: int) -> float:
        """Calculate SMA for the last value"""
        if len(values) < period:
            return np.mean(values) if values else 0
        return np.mean(values[-period:])

    def _nz(self, value, default=0):
        """Pine Script nz() function"""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return value

    def prepare_indicators(self):
        """
        Calculate all indicators bar-by-bar to match Pine Script execution.
        This is the main computation that must match TradingView exactly.
        """
        n = len(self.df)
        close = self.df["close"].values
        high = self.df["high"].values
        low = self.df["low"].values
        tr = self.df["tr"].values

        # Pre-allocate output arrays
        # SuperTrend MAIN
        out_st_atr = np.full(n, np.nan)
        out_st_up = np.full(n, np.nan)
        out_st_dn = np.full(n, np.nan)
        out_st_trend = np.full(n, np.nan)
        out_st_flipBuy = np.zeros(n)
        out_st_flipSell = np.zeros(n)

        # Range Filter GLOBAL
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

        # SuperTrend SL LONG
        out_st_sl_L_atr = np.full(n, np.nan)
        out_st_sl_L_up = np.full(n, np.nan)
        out_st_sl_L_dn = np.full(n, np.nan)
        out_st_sl_L_trend = np.full(n, np.nan)
        out_st_sl_L_flipSell = np.zeros(n)

        # SuperTrend SL SHORT
        out_st_sl_S_atr = np.full(n, np.nan)
        out_st_sl_S_up = np.full(n, np.nan)
        out_st_sl_S_dn = np.full(n, np.nan)
        out_st_sl_S_trend = np.full(n, np.nan)
        out_st_sl_S_flipBuy = np.zeros(n)

        # Range Filter SL LONG
        out_rf_sl_L_smrng = np.full(n, np.nan)
        out_rf_sl_L_filt = np.full(n, np.nan)
        out_rf_sl_L_hband = np.full(n, np.nan)
        out_rf_sl_L_lband = np.full(n, np.nan)
        out_rf_sl_L_upward = np.zeros(n)
        out_rf_sl_L_downward = np.zeros(n)
        out_rf_sl_L_state = np.zeros(n)
        out_rf_sl_L_longCond = np.zeros(n)
        out_rf_sl_L_shortCond = np.zeros(n)
        out_rf_sl_L_CondIni = np.zeros(n)
        out_rf_sl_L_flipBuy = np.zeros(n)
        out_rf_sl_L_flipSell = np.zeros(n)

        # Range Filter SL SHORT
        out_rf_sl_S_smrng = np.full(n, np.nan)
        out_rf_sl_S_filt = np.full(n, np.nan)
        out_rf_sl_S_hband = np.full(n, np.nan)
        out_rf_sl_S_lband = np.full(n, np.nan)
        out_rf_sl_S_upward = np.zeros(n)
        out_rf_sl_S_downward = np.zeros(n)
        out_rf_sl_S_state = np.zeros(n)
        out_rf_sl_S_longCond = np.zeros(n)
        out_rf_sl_S_shortCond = np.zeros(n)
        out_rf_sl_S_CondIni = np.zeros(n)
        out_rf_sl_S_flipBuy = np.zeros(n)
        out_rf_sl_S_flipSell = np.zeros(n)

        # SuperTrend TP Dual LONG
        out_st_tp_dual_L_atr = np.full(n, np.nan)
        out_st_tp_dual_L_up = np.full(n, np.nan)
        out_st_tp_dual_L_dn = np.full(n, np.nan)
        out_st_tp_dual_L_trend = np.full(n, np.nan)
        out_st_tp_dual_L_flipSell = np.zeros(n)

        # SuperTrend TP Dual SHORT
        out_st_tp_dual_S_atr = np.full(n, np.nan)
        out_st_tp_dual_S_up = np.full(n, np.nan)
        out_st_tp_dual_S_dn = np.full(n, np.nan)
        out_st_tp_dual_S_trend = np.full(n, np.nan)
        out_st_tp_dual_S_flipBuy = np.zeros(n)

        # SuperTrend TP RSI LONG
        out_st_tp_rsi_L_atr = np.full(n, np.nan)
        out_st_tp_rsi_L_up = np.full(n, np.nan)
        out_st_tp_rsi_L_dn = np.full(n, np.nan)
        out_st_tp_rsi_L_trend = np.full(n, np.nan)
        out_st_tp_rsi_L_flipSell = np.zeros(n)

        # SuperTrend TP RSI SHORT
        out_st_tp_rsi_S_atr = np.full(n, np.nan)
        out_st_tp_rsi_S_up = np.full(n, np.nan)
        out_st_tp_rsi_S_dn = np.full(n, np.nan)
        out_st_tp_rsi_S_trend = np.full(n, np.nan)
        out_st_tp_rsi_S_flipBuy = np.zeros(n)

        # RSI
        out_rsi = np.full(n, np.nan)
        out_rsi_ma = np.full(n, np.nan)
        out_rsi_crossup = np.zeros(n)
        out_rsi_crossdown = np.zeros(n)
        out_rsi_bull_div_signal = np.zeros(n)
        out_rsi_bear_div_signal = np.zeros(n)

        # Dual Flip
        out_bars_rf_flipBuy = np.full(n, np.nan)
        out_bars_rf_flipSell = np.full(n, np.nan)
        out_bars_st_flipBuy = np.full(n, np.nan)
        out_bars_st_flipSell = np.full(n, np.nan)
        out_dual_flip_long = np.zeros(n)
        out_dual_flip_short = np.zeros(n)

        # History arrays
        tr_history = []
        rf_src_history = []
        rf_abs_diff_history = []
        rsi_history = []

        # ═══════════════════════════════════════════════════════
        # STATE VARIABLES (matching Pine's var declarations)
        # ═══════════════════════════════════════════════════════
        # SuperTrend MAIN
        st_trend = 1.0
        st_up_prev = np.nan
        st_dn_prev = np.nan

        # Range Filter GLOBAL
        rf_f = np.nan
        rf_state = 0

        # SuperTrend SL LONG
        st_sl_L_trend = 1.0
        st_sl_L_up_prev = np.nan
        st_sl_L_dn_prev = np.nan

        # SuperTrend SL SHORT
        st_sl_S_trend = 1.0
        st_sl_S_up_prev = np.nan
        st_sl_S_dn_prev = np.nan

        # Range Filter SL LONG
        rf_sl_L_filt = np.nan
        rf_sl_L_upward = 0.0
        rf_sl_L_downward = 0.0
        rf_sl_L_CondIni = 0
        rf_sl_L_state = 0

        # Range Filter SL SHORT
        rf_sl_S_filt = np.nan
        rf_sl_S_upward = 0.0
        rf_sl_S_downward = 0.0
        rf_sl_S_CondIni = 0
        rf_sl_S_state = 0

        # SuperTrend TP Dual LONG
        st_tp_dual_L_trend = 1.0
        st_tp_dual_L_up_prev = np.nan
        st_tp_dual_L_dn_prev = np.nan

        # SuperTrend TP Dual SHORT
        st_tp_dual_S_trend = 1.0
        st_tp_dual_S_up_prev = np.nan
        st_tp_dual_S_dn_prev = np.nan

        # SuperTrend TP RSI LONG
        st_tp_rsi_L_trend = 1.0
        st_tp_rsi_L_up_prev = np.nan
        st_tp_rsi_L_dn_prev = np.nan

        # SuperTrend TP RSI SHORT
        st_tp_rsi_S_trend = 1.0
        st_tp_rsi_S_up_prev = np.nan
        st_tp_rsi_S_dn_prev = np.nan

        # RSI Divergence state - BULLISH
        in_seg_bull = False
        seg_min_rsi = np.nan
        seg_min_low = np.nan
        arr_rsi_bull = []
        arr_low_bull = []

        # RSI Divergence state - BEARISH
        in_seg_bear = False
        seg_max_rsi = np.nan
        seg_max_high = np.nan
        arr_rsi_bear = []
        arr_high_bear = []

        # Bars since flip tracking
        bars_since_rf_flipBuy = 9999
        bars_since_rf_flipSell = 9999
        bars_since_st_flipBuy = 9999
        bars_since_st_flipSell = 9999

        # ═══════════════════════════════════════════════════════
        # INCREMENTAL EMA/RMA STATE VARIABLES
        # ═══════════════════════════════════════════════════════
        # ATR for SuperTrend MAIN
        st_atr_rma = 0.0
        st_atr_sum = 0.0

        # ATR for SuperTrend SL LONG
        st_sl_L_atr_rma = 0.0
        st_sl_L_atr_sum = 0.0

        # ATR for SuperTrend SL SHORT
        st_sl_S_atr_rma = 0.0
        st_sl_S_atr_sum = 0.0

        # ATR for SuperTrend TP Dual LONG
        st_tp_dual_L_atr_rma = 0.0
        st_tp_dual_L_atr_sum = 0.0

        # ATR for SuperTrend TP Dual SHORT
        st_tp_dual_S_atr_rma = 0.0
        st_tp_dual_S_atr_sum = 0.0

        # ATR for SuperTrend TP RSI LONG
        st_tp_rsi_L_atr_rma = 0.0
        st_tp_rsi_L_atr_sum = 0.0

        # ATR for SuperTrend TP RSI SHORT
        st_tp_rsi_S_atr_rma = 0.0
        st_tp_rsi_S_atr_sum = 0.0

        # Range Filter GLOBAL EMA
        rf_ema1 = 0.0
        rf_ema1_sum = 0.0
        rf_ema2 = 0.0
        rf_ema2_sum = 0.0

        # Range Filter SL LONG EMA
        rf_sl_L_avrng_ema = 0.0
        rf_sl_L_avrng_sum = 0.0
        rf_sl_L_smrng_ema = 0.0
        rf_sl_L_smrng_sum = 0.0

        # Range Filter SL SHORT EMA
        rf_sl_S_avrng_ema = 0.0
        rf_sl_S_avrng_sum = 0.0
        rf_sl_S_smrng_ema = 0.0
        rf_sl_S_smrng_sum = 0.0

        # RSI RMA values
        rsi_up_rma = 0.0
        rsi_up_sum = 0.0
        rsi_down_rma = 0.0
        rsi_down_sum = 0.0
        rsi_ma_buffer = []
        rsi_ma_prev = 0.0

        for i in range(n):
            # Get current values
            c = close[i]
            h = high[i]
            lo = low[i]
            c_prev = close[i - 1] if i > 0 else c

            # Source values
            st_src_val = self._get_source(self.st_src, i)
            rf_src_val = self._get_source(self.rf_src, i)
            st_sl_L_src_val = self._get_source(self.st_sl_src_L, i)
            st_sl_S_src_val = self._get_source(self.st_sl_src_S, i)

            # Update history
            tr_history.append(tr[i])
            rf_src_history.append(rf_src_val)

            if i > 0:
                rf_abs_diff = abs(rf_src_val - rf_src_history[-2])
            else:
                rf_abs_diff = 0
            rf_abs_diff_history.append(rf_abs_diff)

            # ═══════════════════════════════════════════════════════
            # SUPERTREND MAIN (Entry)
            # ═══════════════════════════════════════════════════════
            if self.st_use_atr:
                st_atr_rma, st_atr_sum = self._rma_update(st_atr_rma, st_atr_sum, tr[i], self.st_atr_period, i)
                st_atr_val = st_atr_rma
            else:
                st_atr_val = self._sma(tr_history, self.st_atr_period)
            out_st_atr[i] = st_atr_val

            st_up = st_src_val - self.st_mult * st_atr_val
            st_dn = st_src_val + self.st_mult * st_atr_val

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

            st_trend_prev = st_trend
            if st_trend == -1 and c > dn1:
                st_trend = 1
            elif st_trend == 1 and c < up1:
                st_trend = -1

            out_st_trend[i] = st_trend

            st_flip_buy = (st_trend == 1 and st_trend_prev == -1)
            st_flip_sell = (st_trend == -1 and st_trend_prev == 1)
            out_st_flipBuy[i] = 1 if st_flip_buy else 0
            out_st_flipSell[i] = 1 if st_flip_sell else 0

            st_up_prev = st_up
            st_dn_prev = st_dn

            # ═══════════════════════════════════════════════════════
            # RANGE FILTER GLOBAL (Entry) - Matching Long Only Strategy
            # ═══════════════════════════════════════════════════════
            # smoothrng(x, t, m) => ta.ema(ta.ema(math.abs(x - x[1]), t), t * 2 - 1) * m
            rf_ema1, rf_ema1_sum = self._ema_update(rf_ema1, rf_ema1_sum, rf_abs_diff, self.rf_period, i)
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

            # Direction (matching Long Only exactly)
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
            # SUPERTREND SL LONG
            # ═══════════════════════════════════════════════════════
            if self.st_sl_use_atr_L:
                st_sl_L_atr_rma, st_sl_L_atr_sum = self._rma_update(
                    st_sl_L_atr_rma, st_sl_L_atr_sum, tr[i], self.st_sl_atr_period_L, i)
                st_sl_L_atr_val = st_sl_L_atr_rma
            else:
                st_sl_L_atr_val = self._sma(tr_history, self.st_sl_atr_period_L)
            out_st_sl_L_atr[i] = st_sl_L_atr_val

            st_sl_L_up = st_sl_L_src_val - self.st_sl_mult_L * st_sl_L_atr_val
            st_sl_L_dn = st_sl_L_src_val + self.st_sl_mult_L * st_sl_L_atr_val

            if i > 0:
                if c_prev > self._nz(st_sl_L_up_prev):
                    st_sl_L_up = max(st_sl_L_up, self._nz(st_sl_L_up_prev))
                if c_prev < self._nz(st_sl_L_dn_prev):
                    st_sl_L_dn = min(st_sl_L_dn, self._nz(st_sl_L_dn_prev))

            out_st_sl_L_up[i] = st_sl_L_up
            out_st_sl_L_dn[i] = st_sl_L_dn

            st_sl_L_trend_prev = st_sl_L_trend
            if st_sl_L_trend == -1 and c > self._nz(st_sl_L_dn_prev):
                st_sl_L_trend = 1
            elif st_sl_L_trend == 1 and c < self._nz(st_sl_L_up_prev):
                st_sl_L_trend = -1

            out_st_sl_L_trend[i] = st_sl_L_trend
            st_sl_L_flip_sell = (st_sl_L_trend == -1 and st_sl_L_trend_prev == 1)
            out_st_sl_L_flipSell[i] = 1 if st_sl_L_flip_sell else 0

            st_sl_L_up_prev = st_sl_L_up
            st_sl_L_dn_prev = st_sl_L_dn

            # ═══════════════════════════════════════════════════════
            # SUPERTREND SL SHORT
            # ═══════════════════════════════════════════════════════
            if self.st_sl_use_atr_S:
                st_sl_S_atr_rma, st_sl_S_atr_sum = self._rma_update(
                    st_sl_S_atr_rma, st_sl_S_atr_sum, tr[i], self.st_sl_atr_period_S, i)
                st_sl_S_atr_val = st_sl_S_atr_rma
            else:
                st_sl_S_atr_val = self._sma(tr_history, self.st_sl_atr_period_S)
            out_st_sl_S_atr[i] = st_sl_S_atr_val

            st_sl_S_up = st_sl_S_src_val - self.st_sl_mult_S * st_sl_S_atr_val
            st_sl_S_dn = st_sl_S_src_val + self.st_sl_mult_S * st_sl_S_atr_val

            if i > 0:
                if c_prev > self._nz(st_sl_S_up_prev):
                    st_sl_S_up = max(st_sl_S_up, self._nz(st_sl_S_up_prev))
                if c_prev < self._nz(st_sl_S_dn_prev):
                    st_sl_S_dn = min(st_sl_S_dn, self._nz(st_sl_S_dn_prev))

            out_st_sl_S_up[i] = st_sl_S_up
            out_st_sl_S_dn[i] = st_sl_S_dn

            st_sl_S_trend_prev = st_sl_S_trend
            if st_sl_S_trend == -1 and c > self._nz(st_sl_S_dn_prev):
                st_sl_S_trend = 1
            elif st_sl_S_trend == 1 and c < self._nz(st_sl_S_up_prev):
                st_sl_S_trend = -1

            out_st_sl_S_trend[i] = st_sl_S_trend
            st_sl_S_flip_buy = (st_sl_S_trend == 1 and st_sl_S_trend_prev == -1)
            out_st_sl_S_flipBuy[i] = 1 if st_sl_S_flip_buy else 0

            st_sl_S_up_prev = st_sl_S_up
            st_sl_S_dn_prev = st_sl_S_dn

            # ═══════════════════════════════════════════════════════
            # RANGE FILTER SL LONG
            # ═══════════════════════════════════════════════════════
            rf_sl_L_wper = self.rf_sl_period_L * 2 - 1
            rf_sl_L_avrng_ema, rf_sl_L_avrng_sum = self._ema_update(
                rf_sl_L_avrng_ema, rf_sl_L_avrng_sum, rf_abs_diff, self.rf_sl_period_L, i)
            rf_sl_L_avrng = rf_sl_L_avrng_ema

            rf_sl_L_smrng_ema, rf_sl_L_smrng_sum = self._ema_update(
                rf_sl_L_smrng_ema, rf_sl_L_smrng_sum, rf_sl_L_avrng, rf_sl_L_wper, i)
            rf_sl_L_smrng = rf_sl_L_smrng_ema * self.rf_sl_mult_L
            out_rf_sl_L_smrng[i] = rf_sl_L_smrng

            rf_sl_L_filt_prev = rf_sl_L_filt
            if np.isnan(rf_sl_L_filt_prev):
                rf_sl_L_filt = rf_src_val
            else:
                if rf_src_val > self._nz(rf_sl_L_filt_prev):
                    if rf_src_val - rf_sl_L_smrng < self._nz(rf_sl_L_filt_prev):
                        rf_sl_L_filt = self._nz(rf_sl_L_filt_prev)
                    else:
                        rf_sl_L_filt = rf_src_val - rf_sl_L_smrng
                else:
                    if rf_src_val + rf_sl_L_smrng > self._nz(rf_sl_L_filt_prev):
                        rf_sl_L_filt = self._nz(rf_sl_L_filt_prev)
                    else:
                        rf_sl_L_filt = rf_src_val + rf_sl_L_smrng

            out_rf_sl_L_filt[i] = rf_sl_L_filt
            rf_sl_L_hband = rf_sl_L_filt + rf_sl_L_smrng
            rf_sl_L_lband = rf_sl_L_filt - rf_sl_L_smrng
            out_rf_sl_L_hband[i] = rf_sl_L_hband
            out_rf_sl_L_lband[i] = rf_sl_L_lband

            if rf_sl_L_filt > self._nz(rf_sl_L_filt_prev, rf_sl_L_filt):
                rf_sl_L_upward = self._nz(rf_sl_L_upward) + 1
            elif rf_sl_L_filt < self._nz(rf_sl_L_filt_prev, rf_sl_L_filt):
                rf_sl_L_upward = 0

            if rf_sl_L_filt < self._nz(rf_sl_L_filt_prev, rf_sl_L_filt):
                rf_sl_L_downward = self._nz(rf_sl_L_downward) + 1
            elif rf_sl_L_filt > self._nz(rf_sl_L_filt_prev, rf_sl_L_filt):
                rf_sl_L_downward = 0

            out_rf_sl_L_upward[i] = rf_sl_L_upward
            out_rf_sl_L_downward[i] = rf_sl_L_downward

            # Simplified condition: src > filt and upward > 0 (or src < filt and downward > 0)
            # The rf_src_prev comparison is redundant as both conditions cover all cases
            rf_sl_L_longCond = rf_src_val > rf_sl_L_filt and rf_sl_L_upward > 0
            rf_sl_L_shortCond = rf_src_val < rf_sl_L_filt and rf_sl_L_downward > 0

            out_rf_sl_L_longCond[i] = 1 if rf_sl_L_longCond else 0
            out_rf_sl_L_shortCond[i] = 1 if rf_sl_L_shortCond else 0

            rf_sl_L_CondIni_prev = rf_sl_L_CondIni
            if rf_sl_L_longCond:
                rf_sl_L_CondIni = 1
            elif rf_sl_L_shortCond:
                rf_sl_L_CondIni = -1

            out_rf_sl_L_CondIni[i] = rf_sl_L_CondIni

            rf_sl_L_flip_buy = rf_sl_L_longCond and self._nz(rf_sl_L_CondIni_prev) == -1
            rf_sl_L_flip_sell = rf_sl_L_shortCond and self._nz(rf_sl_L_CondIni_prev) == 1

            out_rf_sl_L_flipBuy[i] = 1 if rf_sl_L_flip_buy else 0
            out_rf_sl_L_flipSell[i] = 1 if rf_sl_L_flip_sell else 0

            rf_sl_L_state_prev = rf_sl_L_state
            if rf_sl_L_longCond:
                rf_sl_L_state = 1
            elif rf_sl_L_shortCond:
                rf_sl_L_state = -1

            out_rf_sl_L_state[i] = rf_sl_L_state

            # ═══════════════════════════════════════════════════════
            # RANGE FILTER SL SHORT
            # ═══════════════════════════════════════════════════════
            rf_sl_S_wper = self.rf_sl_period_S * 2 - 1
            rf_sl_S_avrng_ema, rf_sl_S_avrng_sum = self._ema_update(
                rf_sl_S_avrng_ema, rf_sl_S_avrng_sum, rf_abs_diff, self.rf_sl_period_S, i)
            rf_sl_S_avrng = rf_sl_S_avrng_ema

            rf_sl_S_smrng_ema, rf_sl_S_smrng_sum = self._ema_update(
                rf_sl_S_smrng_ema, rf_sl_S_smrng_sum, rf_sl_S_avrng, rf_sl_S_wper, i)
            rf_sl_S_smrng = rf_sl_S_smrng_ema * self.rf_sl_mult_S
            out_rf_sl_S_smrng[i] = rf_sl_S_smrng

            rf_sl_S_filt_prev = rf_sl_S_filt
            if np.isnan(rf_sl_S_filt_prev):
                rf_sl_S_filt = rf_src_val
            else:
                if rf_src_val > self._nz(rf_sl_S_filt_prev):
                    if rf_src_val - rf_sl_S_smrng < self._nz(rf_sl_S_filt_prev):
                        rf_sl_S_filt = self._nz(rf_sl_S_filt_prev)
                    else:
                        rf_sl_S_filt = rf_src_val - rf_sl_S_smrng
                else:
                    if rf_src_val + rf_sl_S_smrng > self._nz(rf_sl_S_filt_prev):
                        rf_sl_S_filt = self._nz(rf_sl_S_filt_prev)
                    else:
                        rf_sl_S_filt = rf_src_val + rf_sl_S_smrng

            out_rf_sl_S_filt[i] = rf_sl_S_filt
            rf_sl_S_hband = rf_sl_S_filt + rf_sl_S_smrng
            rf_sl_S_lband = rf_sl_S_filt - rf_sl_S_smrng
            out_rf_sl_S_hband[i] = rf_sl_S_hband
            out_rf_sl_S_lband[i] = rf_sl_S_lband

            if rf_sl_S_filt > self._nz(rf_sl_S_filt_prev, rf_sl_S_filt):
                rf_sl_S_upward = self._nz(rf_sl_S_upward) + 1
            elif rf_sl_S_filt < self._nz(rf_sl_S_filt_prev, rf_sl_S_filt):
                rf_sl_S_upward = 0

            if rf_sl_S_filt < self._nz(rf_sl_S_filt_prev, rf_sl_S_filt):
                rf_sl_S_downward = self._nz(rf_sl_S_downward) + 1
            elif rf_sl_S_filt > self._nz(rf_sl_S_filt_prev, rf_sl_S_filt):
                rf_sl_S_downward = 0

            out_rf_sl_S_upward[i] = rf_sl_S_upward
            out_rf_sl_S_downward[i] = rf_sl_S_downward

            # Simplified condition: src > filt and upward > 0 (or src < filt and downward > 0)
            rf_sl_S_longCond = rf_src_val > rf_sl_S_filt and rf_sl_S_upward > 0
            rf_sl_S_shortCond = rf_src_val < rf_sl_S_filt and rf_sl_S_downward > 0

            out_rf_sl_S_longCond[i] = 1 if rf_sl_S_longCond else 0
            out_rf_sl_S_shortCond[i] = 1 if rf_sl_S_shortCond else 0

            rf_sl_S_CondIni_prev = rf_sl_S_CondIni
            if rf_sl_S_longCond:
                rf_sl_S_CondIni = 1
            elif rf_sl_S_shortCond:
                rf_sl_S_CondIni = -1

            out_rf_sl_S_CondIni[i] = rf_sl_S_CondIni

            rf_sl_S_flip_buy = rf_sl_S_longCond and self._nz(rf_sl_S_CondIni_prev) == -1
            rf_sl_S_flip_sell = rf_sl_S_shortCond and self._nz(rf_sl_S_CondIni_prev) == 1

            out_rf_sl_S_flipBuy[i] = 1 if rf_sl_S_flip_buy else 0
            out_rf_sl_S_flipSell[i] = 1 if rf_sl_S_flip_sell else 0

            rf_sl_S_state_prev = rf_sl_S_state
            if rf_sl_S_longCond:
                rf_sl_S_state = 1
            elif rf_sl_S_shortCond:
                rf_sl_S_state = -1

            out_rf_sl_S_state[i] = rf_sl_S_state

            # ═══════════════════════════════════════════════════════
            # SUPERTREND TP DUAL LONG
            # ═══════════════════════════════════════════════════════
            st_tp_dual_L_atr_rma, st_tp_dual_L_atr_sum = self._rma_update(
                st_tp_dual_L_atr_rma, st_tp_dual_L_atr_sum, tr[i], self.st_tp_dual_period_L, i)
            st_tp_dual_L_atr_val = st_tp_dual_L_atr_rma
            out_st_tp_dual_L_atr[i] = st_tp_dual_L_atr_val

            st_tp_dual_L_up = st_src_val - self.st_tp_dual_mult_L * st_tp_dual_L_atr_val
            st_tp_dual_L_dn = st_src_val + self.st_tp_dual_mult_L * st_tp_dual_L_atr_val

            if i > 0:
                if c_prev > self._nz(st_tp_dual_L_up_prev):
                    st_tp_dual_L_up = max(st_tp_dual_L_up, self._nz(st_tp_dual_L_up_prev))
                if c_prev < self._nz(st_tp_dual_L_dn_prev):
                    st_tp_dual_L_dn = min(st_tp_dual_L_dn, self._nz(st_tp_dual_L_dn_prev))

            out_st_tp_dual_L_up[i] = st_tp_dual_L_up
            out_st_tp_dual_L_dn[i] = st_tp_dual_L_dn

            st_tp_dual_L_trend_prev = st_tp_dual_L_trend
            if st_tp_dual_L_trend == -1 and c > self._nz(st_tp_dual_L_dn_prev):
                st_tp_dual_L_trend = 1
            elif st_tp_dual_L_trend == 1 and c < self._nz(st_tp_dual_L_up_prev):
                st_tp_dual_L_trend = -1

            out_st_tp_dual_L_trend[i] = st_tp_dual_L_trend
            st_tp_dual_L_flip_sell = (st_tp_dual_L_trend == -1 and st_tp_dual_L_trend_prev == 1)
            out_st_tp_dual_L_flipSell[i] = 1 if st_tp_dual_L_flip_sell else 0

            st_tp_dual_L_up_prev = st_tp_dual_L_up
            st_tp_dual_L_dn_prev = st_tp_dual_L_dn

            # ═══════════════════════════════════════════════════════
            # SUPERTREND TP DUAL SHORT
            # ═══════════════════════════════════════════════════════
            st_tp_dual_S_atr_rma, st_tp_dual_S_atr_sum = self._rma_update(
                st_tp_dual_S_atr_rma, st_tp_dual_S_atr_sum, tr[i], self.st_tp_dual_period_S, i)
            st_tp_dual_S_atr_val = st_tp_dual_S_atr_rma
            out_st_tp_dual_S_atr[i] = st_tp_dual_S_atr_val

            st_tp_dual_S_up = st_src_val - self.st_tp_dual_mult_S * st_tp_dual_S_atr_val
            st_tp_dual_S_dn = st_src_val + self.st_tp_dual_mult_S * st_tp_dual_S_atr_val

            if i > 0:
                if c_prev > self._nz(st_tp_dual_S_up_prev):
                    st_tp_dual_S_up = max(st_tp_dual_S_up, self._nz(st_tp_dual_S_up_prev))
                if c_prev < self._nz(st_tp_dual_S_dn_prev):
                    st_tp_dual_S_dn = min(st_tp_dual_S_dn, self._nz(st_tp_dual_S_dn_prev))

            out_st_tp_dual_S_up[i] = st_tp_dual_S_up
            out_st_tp_dual_S_dn[i] = st_tp_dual_S_dn

            st_tp_dual_S_trend_prev = st_tp_dual_S_trend
            if st_tp_dual_S_trend == -1 and c > self._nz(st_tp_dual_S_dn_prev):
                st_tp_dual_S_trend = 1
            elif st_tp_dual_S_trend == 1 and c < self._nz(st_tp_dual_S_up_prev):
                st_tp_dual_S_trend = -1

            out_st_tp_dual_S_trend[i] = st_tp_dual_S_trend
            st_tp_dual_S_flip_buy = (st_tp_dual_S_trend == 1 and st_tp_dual_S_trend_prev == -1)
            out_st_tp_dual_S_flipBuy[i] = 1 if st_tp_dual_S_flip_buy else 0

            st_tp_dual_S_up_prev = st_tp_dual_S_up
            st_tp_dual_S_dn_prev = st_tp_dual_S_dn

            # ═══════════════════════════════════════════════════════
            # SUPERTREND TP RSI LONG
            # ═══════════════════════════════════════════════════════
            st_tp_rsi_L_atr_rma, st_tp_rsi_L_atr_sum = self._rma_update(
                st_tp_rsi_L_atr_rma, st_tp_rsi_L_atr_sum, tr[i], self.st_tp_rsi_period_L, i)
            st_tp_rsi_L_atr_val = st_tp_rsi_L_atr_rma
            out_st_tp_rsi_L_atr[i] = st_tp_rsi_L_atr_val

            st_tp_rsi_L_up = st_src_val - self.st_tp_rsi_mult_L * st_tp_rsi_L_atr_val
            st_tp_rsi_L_dn = st_src_val + self.st_tp_rsi_mult_L * st_tp_rsi_L_atr_val

            if i > 0:
                if c_prev > self._nz(st_tp_rsi_L_up_prev):
                    st_tp_rsi_L_up = max(st_tp_rsi_L_up, self._nz(st_tp_rsi_L_up_prev))
                if c_prev < self._nz(st_tp_rsi_L_dn_prev):
                    st_tp_rsi_L_dn = min(st_tp_rsi_L_dn, self._nz(st_tp_rsi_L_dn_prev))

            out_st_tp_rsi_L_up[i] = st_tp_rsi_L_up
            out_st_tp_rsi_L_dn[i] = st_tp_rsi_L_dn

            st_tp_rsi_L_trend_prev = st_tp_rsi_L_trend
            if st_tp_rsi_L_trend == -1 and c > self._nz(st_tp_rsi_L_dn_prev):
                st_tp_rsi_L_trend = 1
            elif st_tp_rsi_L_trend == 1 and c < self._nz(st_tp_rsi_L_up_prev):
                st_tp_rsi_L_trend = -1

            out_st_tp_rsi_L_trend[i] = st_tp_rsi_L_trend
            st_tp_rsi_L_flip_sell = (st_tp_rsi_L_trend == -1 and st_tp_rsi_L_trend_prev == 1)
            out_st_tp_rsi_L_flipSell[i] = 1 if st_tp_rsi_L_flip_sell else 0

            st_tp_rsi_L_up_prev = st_tp_rsi_L_up
            st_tp_rsi_L_dn_prev = st_tp_rsi_L_dn

            # ═══════════════════════════════════════════════════════
            # SUPERTREND TP RSI SHORT
            # ═══════════════════════════════════════════════════════
            st_tp_rsi_S_atr_rma, st_tp_rsi_S_atr_sum = self._rma_update(
                st_tp_rsi_S_atr_rma, st_tp_rsi_S_atr_sum, tr[i], self.st_tp_rsi_period_S, i)
            st_tp_rsi_S_atr_val = st_tp_rsi_S_atr_rma
            out_st_tp_rsi_S_atr[i] = st_tp_rsi_S_atr_val

            st_tp_rsi_S_up = st_src_val - self.st_tp_rsi_mult_S * st_tp_rsi_S_atr_val
            st_tp_rsi_S_dn = st_src_val + self.st_tp_rsi_mult_S * st_tp_rsi_S_atr_val

            if i > 0:
                if c_prev > self._nz(st_tp_rsi_S_up_prev):
                    st_tp_rsi_S_up = max(st_tp_rsi_S_up, self._nz(st_tp_rsi_S_up_prev))
                if c_prev < self._nz(st_tp_rsi_S_dn_prev):
                    st_tp_rsi_S_dn = min(st_tp_rsi_S_dn, self._nz(st_tp_rsi_S_dn_prev))

            out_st_tp_rsi_S_up[i] = st_tp_rsi_S_up
            out_st_tp_rsi_S_dn[i] = st_tp_rsi_S_dn

            st_tp_rsi_S_trend_prev = st_tp_rsi_S_trend
            if st_tp_rsi_S_trend == -1 and c > self._nz(st_tp_rsi_S_dn_prev):
                st_tp_rsi_S_trend = 1
            elif st_tp_rsi_S_trend == 1 and c < self._nz(st_tp_rsi_S_up_prev):
                st_tp_rsi_S_trend = -1

            out_st_tp_rsi_S_trend[i] = st_tp_rsi_S_trend
            st_tp_rsi_S_flip_buy = (st_tp_rsi_S_trend == 1 and st_tp_rsi_S_trend_prev == -1)
            out_st_tp_rsi_S_flipBuy[i] = 1 if st_tp_rsi_S_flip_buy else 0

            st_tp_rsi_S_up_prev = st_tp_rsi_S_up
            st_tp_rsi_S_dn_prev = st_tp_rsi_S_dn

            # ═══════════════════════════════════════════════════════
            # RSI CALCULATION
            # ═══════════════════════════════════════════════════════
            rsi_crossup = False
            rsi_crossdown = False
            rsi_bull_div_signal = False
            rsi_bear_div_signal = False

            if i > 0:
                change = c - c_prev
                up_val = max(change, 0)
                down_val = -min(change, 0)

                rsi_count = i - 1
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

                # RSI MA
                rsi_ma_buffer.append(rsi_val)
                if len(rsi_ma_buffer) > self.rsi_ma_length:
                    rsi_ma_buffer.pop(0)
                rsi_ma = sum(rsi_ma_buffer) / len(rsi_ma_buffer)
                out_rsi_ma[i] = rsi_ma

                # Crossover detection
                if len(rsi_history) >= 2:
                    rsi_prev = rsi_history[-2]
                    rsi_crossup = rsi_prev <= rsi_ma_prev and rsi_val > rsi_ma
                    rsi_crossdown = rsi_prev >= rsi_ma_prev and rsi_val < rsi_ma

                rsi_ma_prev = rsi_ma

                out_rsi_crossup[i] = 1 if rsi_crossup else 0
                out_rsi_crossdown[i] = 1 if rsi_crossdown else 0

                # ═══════════════════════════════════════════════════════
                # RSI BULLISH DIVERGENCE DETECTION
                # ═══════════════════════════════════════════════════════
                if rsi_crossdown:
                    in_seg_bull = True
                    seg_min_rsi = rsi_val
                    seg_min_low = lo

                if in_seg_bull and rsi_val < seg_min_rsi:
                    seg_min_rsi = rsi_val
                    seg_min_low = lo

                if rsi_crossup and in_seg_bull:
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

                        rsi_bull_div_signal = regular or hidden

                    in_seg_bull = False
                    seg_min_rsi = np.nan
                    seg_min_low = np.nan

                out_rsi_bull_div_signal[i] = 1 if rsi_bull_div_signal else 0

                # ═══════════════════════════════════════════════════════
                # RSI BEARISH DIVERGENCE DETECTION
                # ═══════════════════════════════════════════════════════
                if rsi_crossup:
                    in_seg_bear = True
                    seg_max_rsi = rsi_val
                    seg_max_high = h

                if in_seg_bear and rsi_val > seg_max_rsi:
                    seg_max_rsi = rsi_val
                    seg_max_high = h

                if rsi_crossdown and in_seg_bear:
                    arr_rsi_bear.append(seg_max_rsi)
                    arr_high_bear.append(seg_max_high)

                    if len(arr_rsi_bear) > 2:
                        arr_rsi_bear.pop(0)
                        arr_high_bear.pop(0)

                    if len(arr_rsi_bear) == 2:
                        r1, r2 = arr_rsi_bear[0], arr_rsi_bear[1]
                        h1, h2 = arr_high_bear[0], arr_high_bear[1]

                        # Regular bearish: higher high price, lower high RSI
                        regular = (h2 > h1) and (r2 < r1)
                        # Hidden bearish: lower high price, higher high RSI
                        hidden = (h2 < h1) and (r2 > r1)

                        rsi_bear_div_signal = regular or hidden

                    in_seg_bear = False
                    seg_max_rsi = np.nan
                    seg_max_high = np.nan

                out_rsi_bear_div_signal[i] = 1 if rsi_bear_div_signal else 0
            else:
                out_rsi[i] = 50
                out_rsi_ma[i] = 50

            # ═══════════════════════════════════════════════════════
            # DUAL FLIP TRACKING - Count starts at 1
            # ═══════════════════════════════════════════════════════
            # User requirement:
            # - When Buy RF appears, count = 1
            # - Next bar = 2, next = 3...
            # - When Buy ST appears at count 4 => DualFlip passes (if threshold >= 4)
            #
            # RF flip buy/sell tracking (count from 1, not 0)
            if rf_flip_buy:
                bars_since_rf_flipBuy = 1  # Flip bar counts as 1
            else:
                bars_since_rf_flipBuy += 1

            if rf_flip_sell:
                bars_since_rf_flipSell = 1  # Flip bar counts as 1
            else:
                bars_since_rf_flipSell += 1

            # ST flip buy/sell tracking (count from 1, not 0)
            if st_flip_buy:
                bars_since_st_flipBuy = 1  # Flip bar counts as 1
            else:
                bars_since_st_flipBuy += 1

            if st_flip_sell:
                bars_since_st_flipSell = 1  # Flip bar counts as 1
            else:
                bars_since_st_flipSell += 1

            out_bars_rf_flipBuy[i] = bars_since_rf_flipBuy
            out_bars_rf_flipSell[i] = bars_since_rf_flipSell
            out_bars_st_flipBuy[i] = bars_since_st_flipBuy
            out_bars_st_flipSell[i] = bars_since_st_flipSell

            # ═══════════════════════════════════════════════════════
            # DUAL FLIP CONDITIONS
            # ═══════════════════════════════════════════════════════
            # Example: threshold = 8
            # - Buy RF at bar 0 (count = 1)
            # - Bar 1: count = 2
            # - Bar 2: count = 3
            # - ...
            # - Bar 7: count = 8, Buy ST appears => ST count = 1, RF count = 8
            #   Check: rf_flip_buy=False, st_flip_buy=True
            #   Condition: st_flip_buy and bars_since_rf_flipBuy <= 8 => True (8 <= 8) ✓
            #
            # Example: threshold = 8, but RF flip was 14 bars ago
            # - Buy RF at bar 0 (count = 1)
            # - ...
            # - Bar 13: count = 14, Buy ST appears
            #   Condition: st_flip_buy and bars_since_rf_flipBuy <= 8 => False (14 > 8) ✗

            # Dual flip long: RF flips Buy NOW + ST flipped within N bars, OR vice versa
            dual_flip_long = (rf_flip_buy and bars_since_st_flipBuy <= self.dual_flip_bars_long) or \
                             (st_flip_buy and bars_since_rf_flipBuy <= self.dual_flip_bars_long)
            out_dual_flip_long[i] = 1 if dual_flip_long else 0

            # Dual flip short: RF flips Sell NOW + ST flipped within N bars, OR vice versa
            dual_flip_short = (rf_flip_sell and bars_since_st_flipSell <= self.dual_flip_bars_short) or \
                              (st_flip_sell and bars_since_rf_flipSell <= self.dual_flip_bars_short)
            out_dual_flip_short[i] = 1 if dual_flip_short else 0

        # ═══════════════════════════════════════════════════════
        # BULK ASSIGNMENT TO DATAFRAME
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
        self.df["st_sl_L_atr"] = out_st_sl_L_atr
        self.df["st_sl_L_up"] = out_st_sl_L_up
        self.df["st_sl_L_dn"] = out_st_sl_L_dn
        self.df["st_sl_L_trend"] = out_st_sl_L_trend
        self.df["st_sl_L_flipSell"] = out_st_sl_L_flipSell
        self.df["st_sl_S_atr"] = out_st_sl_S_atr
        self.df["st_sl_S_up"] = out_st_sl_S_up
        self.df["st_sl_S_dn"] = out_st_sl_S_dn
        self.df["st_sl_S_trend"] = out_st_sl_S_trend
        self.df["st_sl_S_flipBuy"] = out_st_sl_S_flipBuy
        self.df["rf_sl_L_smrng"] = out_rf_sl_L_smrng
        self.df["rf_sl_L_filt"] = out_rf_sl_L_filt
        self.df["rf_sl_L_hband"] = out_rf_sl_L_hband
        self.df["rf_sl_L_lband"] = out_rf_sl_L_lband
        self.df["rf_sl_L_upward"] = out_rf_sl_L_upward
        self.df["rf_sl_L_downward"] = out_rf_sl_L_downward
        self.df["rf_sl_L_state"] = out_rf_sl_L_state
        self.df["rf_sl_L_longCond"] = out_rf_sl_L_longCond
        self.df["rf_sl_L_shortCond"] = out_rf_sl_L_shortCond
        self.df["rf_sl_L_CondIni"] = out_rf_sl_L_CondIni
        self.df["rf_sl_L_flipBuy"] = out_rf_sl_L_flipBuy
        self.df["rf_sl_L_flipSell"] = out_rf_sl_L_flipSell
        self.df["rf_sl_S_smrng"] = out_rf_sl_S_smrng
        self.df["rf_sl_S_filt"] = out_rf_sl_S_filt
        self.df["rf_sl_S_hband"] = out_rf_sl_S_hband
        self.df["rf_sl_S_lband"] = out_rf_sl_S_lband
        self.df["rf_sl_S_upward"] = out_rf_sl_S_upward
        self.df["rf_sl_S_downward"] = out_rf_sl_S_downward
        self.df["rf_sl_S_state"] = out_rf_sl_S_state
        self.df["rf_sl_S_longCond"] = out_rf_sl_S_longCond
        self.df["rf_sl_S_shortCond"] = out_rf_sl_S_shortCond
        self.df["rf_sl_S_CondIni"] = out_rf_sl_S_CondIni
        self.df["rf_sl_S_flipBuy"] = out_rf_sl_S_flipBuy
        self.df["rf_sl_S_flipSell"] = out_rf_sl_S_flipSell
        self.df["st_tp_dual_L_atr"] = out_st_tp_dual_L_atr
        self.df["st_tp_dual_L_up"] = out_st_tp_dual_L_up
        self.df["st_tp_dual_L_dn"] = out_st_tp_dual_L_dn
        self.df["st_tp_dual_L_trend"] = out_st_tp_dual_L_trend
        self.df["st_tp_dual_L_flipSell"] = out_st_tp_dual_L_flipSell
        self.df["st_tp_dual_S_atr"] = out_st_tp_dual_S_atr
        self.df["st_tp_dual_S_up"] = out_st_tp_dual_S_up
        self.df["st_tp_dual_S_dn"] = out_st_tp_dual_S_dn
        self.df["st_tp_dual_S_trend"] = out_st_tp_dual_S_trend
        self.df["st_tp_dual_S_flipBuy"] = out_st_tp_dual_S_flipBuy
        self.df["st_tp_rsi_L_atr"] = out_st_tp_rsi_L_atr
        self.df["st_tp_rsi_L_up"] = out_st_tp_rsi_L_up
        self.df["st_tp_rsi_L_dn"] = out_st_tp_rsi_L_dn
        self.df["st_tp_rsi_L_trend"] = out_st_tp_rsi_L_trend
        self.df["st_tp_rsi_L_flipSell"] = out_st_tp_rsi_L_flipSell
        self.df["st_tp_rsi_S_atr"] = out_st_tp_rsi_S_atr
        self.df["st_tp_rsi_S_up"] = out_st_tp_rsi_S_up
        self.df["st_tp_rsi_S_dn"] = out_st_tp_rsi_S_dn
        self.df["st_tp_rsi_S_trend"] = out_st_tp_rsi_S_trend
        self.df["st_tp_rsi_S_flipBuy"] = out_st_tp_rsi_S_flipBuy
        self.df["rsi"] = out_rsi
        self.df["rsi_ma"] = out_rsi_ma
        self.df["rsi_crossup"] = out_rsi_crossup
        self.df["rsi_crossdown"] = out_rsi_crossdown
        self.df["rsi_bull_div_signal"] = out_rsi_bull_div_signal
        self.df["rsi_bear_div_signal"] = out_rsi_bear_div_signal
        self.df["bars_rf_flipBuy"] = out_bars_rf_flipBuy
        self.df["bars_rf_flipSell"] = out_bars_rf_flipSell
        self.df["bars_st_flipBuy"] = out_bars_st_flipBuy
        self.df["bars_st_flipSell"] = out_bars_st_flipSell
        self.df["dual_flip_long"] = out_dual_flip_long
        self.df["dual_flip_short"] = out_dual_flip_short

    def check_entry(self, i: int) -> Optional[str]:
        """
        Check if there's an entry signal at bar index i.
        Returns: "Long", "Short", "RSI Long", "RSI Short", or None

        Pine Script semantics:
        - Entry ID "Long" = DualFlip Long
        - Entry ID "RSI Long" = RSI Divergence Long
        - Entry ID "Short" = DualFlip Short
        - Entry ID "RSI Short" = RSI Divergence Short

        Signal at bar i means entry will execute at Open of bar i+1.
        We store SL level from signal bar (bar i) - this matches TradingView behavior.
        """
        if i < 1:
            return None

        row = self.df.iloc[i]
        time = row.get("time", i)

        # Get indicator values
        dual_flip_long = row.get("dual_flip_long", 0) == 1
        dual_flip_short = row.get("dual_flip_short", 0) == 1
        rf_state = row.get("rf_state", 0)
        rf_lb = row.get("rf_lb", np.nan)
        rf_hb = row.get("rf_hb", np.nan)
        rf_sl_L_lband = row.get("rf_sl_L_lband", np.nan)
        rf_sl_S_hband = row.get("rf_sl_S_hband", np.nan)
        rsi_bull_div = row.get("rsi_bull_div_signal", 0) == 1
        rsi_bear_div = row.get("rsi_bear_div_signal", 0) == 1
        close = row["close"]

        # ═══════════════════════════════════════════════════════
        # ENTRY SIGNAL CONDITIONS
        # Pine Script:
        #   strategyLong  = dualFlipLong and rf_state == 1
        #   strategyShort = dualFlipShort  (NO rf_state check!)
        # Reversal is handled in check_exit() which is called BEFORE check_entry
        # ═══════════════════════════════════════════════════════
        strategy_long = dual_flip_long and rf_state == 1
        strategy_short = dual_flip_short  # Pine: strategyShort = dualFlipShort (no rf_state check)

        # ═══════════════════════════════════════════════════════
        # DUAL FLIP LONG ENTRY (ID: "Long")
        # Pine: strategy.entry("Long", strategy.long, stop=close, limit=close)
        # Condition: dualFlipLong and rf_state == 1
        # ═══════════════════════════════════════════════════════
        # Pine Script allows entry on same bar after exit (reversal behavior)
        # The pyramiding=1 check (not self._in_long) handles preventing duplicate entries
        if strategy_long and not self._in_long and self.enable_long and self.show_entry_long:
            # Track signal values from signal bar
            self._dual_flip_signal_close = close
            self._dual_flip_signal_rf_lb = rf_lb

            # Risk = close - rf_lb (for validation only)
            risk_long = close - rf_lb if not np.isnan(rf_lb) else np.nan
            valid_long = not np.isnan(risk_long) and risk_long > 0

            if valid_long:
                # Store entry price as signal bar close
                # Actual entry will be at Open of next bar (handled by backtest_engine)
                self._entry_price_dual_L = close
                # Use RF SL Lowband from current bar (bar i) - entry bar
                self._rf_sl_at_entry_dual_L = rf_sl_L_lband
                # Risk R for TP calculation
                self._risk_r_dual_L = self._entry_price_dual_L - self._rf_sl_at_entry_dual_L
                self._in_long = True
                self._in_dual_flip_long = True
                self._entry_type = "Long"

                # Get bars_since values for debug
                bars_rf_flipBuy = row.get("bars_rf_flipBuy", 9999)
                bars_st_flipBuy = row.get("bars_st_flipBuy", 9999)
                rf_flipBuy = row.get("rf_flipBuy", 0) == 1
                st_flipBuy = row.get("st_flipBuy", 0) == 1

                self._log(i, time, f"ENTRY Long (DualFlip) @ {close:.2f}, SL_lband: {rf_sl_L_lband:.2f}, Risk: {self._risk_r_dual_L:.2f}, barsRF_B={bars_rf_flipBuy}, barsST_B={bars_st_flipBuy}")
                self._add_debug_event(i, time, "entry", "Long", close, {
                    "entry_id": "Long",
                    "entry_type": "DualFlip",
                    "signal_bar": i,
                    "entry_price": close,
                    "rf_sl_L_lband": rf_sl_L_lband,
                    "risk_r": self._risk_r_dual_L,
                    "rf_state": rf_state,
                    "dual_flip_long": dual_flip_long,
                    "rf_lb": rf_lb,
                    "bars_rf_flipBuy": int(bars_rf_flipBuy),
                    "bars_st_flipBuy": int(bars_st_flipBuy),
                    "rf_flipBuy": rf_flipBuy,
                    "st_flipBuy": st_flipBuy,
                    "dual_flip_bars_long_threshold": self.dual_flip_bars_long
                })
                return "Long"

        # ═══════════════════════════════════════════════════════
        # DUAL FLIP SHORT ENTRY (ID: "Short")
        # Pine: strategy.entry("Short", strategy.short, stop=close, limit=close)
        # Condition: dualFlipShort and rf_state == -1
        # ═══════════════════════════════════════════════════════
        # Pine Script allows entry on same bar after exit (reversal behavior)
        if strategy_short and not self._in_short and self.enable_short and self.show_entry_short:
            # Track signal values
            self._dual_flip_signal_close_S = close
            self._dual_flip_signal_rf_hb = rf_hb

            risk_short = rf_hb - close if not np.isnan(rf_hb) else np.nan
            valid_short = not np.isnan(risk_short) and risk_short > 0

            if valid_short:
                self._entry_price_dual_S = close
                # Use RF SL Highband from current bar (bar i) - entry bar
                self._rf_sl_at_entry_dual_S = rf_sl_S_hband
                self._risk_r_dual_S = self._rf_sl_at_entry_dual_S - self._entry_price_dual_S
                self._in_short = True
                self._in_dual_flip_short = True
                self._entry_type = "Short"

                # Get bars_since values for debug
                bars_rf_flipSell = row.get("bars_rf_flipSell", 9999)
                bars_st_flipSell = row.get("bars_st_flipSell", 9999)
                rf_flipSell = row.get("rf_flipSell", 0) == 1
                st_flipSell = row.get("st_flipSell", 0) == 1

                self._log(i, time, f"ENTRY Short (DualFlip) @ {close:.2f}, SL_hband: {rf_sl_S_hband:.2f}, Risk: {self._risk_r_dual_S:.2f}, barsRF_S={bars_rf_flipSell}, barsST_S={bars_st_flipSell}")
                self._add_debug_event(i, time, "entry", "Short", close, {
                    "entry_id": "Short",
                    "entry_type": "DualFlip",
                    "signal_bar": i,
                    "entry_price": close,
                    "rf_sl_S_hband": rf_sl_S_hband,
                    "risk_r": self._risk_r_dual_S,
                    "rf_state": rf_state,
                    "dual_flip_short": dual_flip_short,
                    "rf_hb": rf_hb,
                    "bars_rf_flipSell": int(bars_rf_flipSell),
                    "bars_st_flipSell": int(bars_st_flipSell),
                    "rf_flipSell": rf_flipSell,
                    "st_flipSell": st_flipSell,
                    "dual_flip_bars_short_threshold": self.dual_flip_bars_short
                })
                return "Short"

        # ═══════════════════════════════════════════════════════
        # RSI DIVERGENCE LONG ENTRY (ID: "RSI Long")
        # Pine: strategy.entry("RSI Long", strategy.long, stop=close, limit=close)
        # Condition: rsiBullDivSignal and rf_state == 1
        # ═══════════════════════════════════════════════════════
        rsi_long = rsi_bull_div and rf_state == 1

        # Pine Script allows entry on same bar after exit (reversal behavior)
        if rsi_long and not self._in_long and self.enable_long and self.show_entry_rsi_L:
            entry_price = close
            # Use RF SL Lowband from current bar (bar i) - entry bar
            risk_r = entry_price - rf_sl_L_lband if not np.isnan(rf_sl_L_lband) else np.nan

            if not np.isnan(risk_r) and risk_r > 0:
                self._entry_price_rsi_L = entry_price
                self._rf_sl_at_entry_rsi_L = rf_sl_L_lband  # Use current bar's value
                self._risk_r_rsi_L = risk_r
                self._in_long = True
                self._in_rsi_long = True
                self._entry_type = "RSI Long"

                self._log(i, time, f"ENTRY RSI Long @ {entry_price:.2f}, SL_lband: {rf_sl_L_lband:.2f}, Risk: {risk_r:.2f}")
                self._add_debug_event(i, time, "entry", "RSI Long", entry_price, {
                    "entry_id": "RSI Long",
                    "entry_type": "RSI Divergence",
                    "signal_bar": i,
                    "entry_price": entry_price,
                    "rf_sl_L_lband": rf_sl_L_lband,
                    "risk_r": risk_r,
                    "rf_state": rf_state,
                    "rsi_bull_div": rsi_bull_div
                })
                return "RSI Long"

        # ═══════════════════════════════════════════════════════
        # RSI DIVERGENCE SHORT ENTRY (ID: "RSI Short")
        # Pine: strategy.entry("RSI Short", strategy.short, stop=close, limit=close)
        # Condition: rsiBearDivSignal and rf_state == -1
        # ═══════════════════════════════════════════════════════
        rsi_short = rsi_bear_div and rf_state == -1

        # Pine Script allows entry on same bar after exit (reversal behavior)
        if rsi_short and not self._in_short and self.enable_short and self.show_entry_rsi_S:
            entry_price = close
            # Use RF SL Highband from current bar (bar i) - entry bar
            risk_r = rf_sl_S_hband - entry_price if not np.isnan(rf_sl_S_hband) else np.nan

            if not np.isnan(risk_r) and risk_r > 0:
                self._entry_price_rsi_S = entry_price
                self._rf_sl_at_entry_rsi_S = rf_sl_S_hband  # Use current bar's value
                self._risk_r_rsi_S = risk_r
                self._in_short = True
                self._in_rsi_short = True
                self._entry_type = "RSI Short"

                self._log(i, time, f"ENTRY RSI Short @ {entry_price:.2f}, SL_hband: {rf_sl_S_hband:.2f}, Risk: {risk_r:.2f}")
                self._add_debug_event(i, time, "entry", "RSI Short", entry_price, {
                    "entry_id": "RSI Short",
                    "entry_type": "RSI Divergence",
                    "signal_bar": i,
                    "entry_price": entry_price,
                    "rf_sl_S_hband": rf_sl_S_hband,
                    "risk_r": risk_r,
                    "rf_state": rf_state,
                    "rsi_bear_div": rsi_bear_div
                })
                return "RSI Short"

        return None

    def check_exit(self, i: int, position: dict) -> Optional[Tuple[str, str]]:
        """
        Check if there's an exit signal at bar index i.
        Returns: (exit_type, reason) or None

        exit_type: "Long", "Short", "RSI Long", "RSI Short"
        reason: "SL", "TP"

        Pine Script SL conditions for Long:
        - slCondition_L = (rf_SL_L_flipSell and st_SL_L_trend == -1) or (st_SL_L_flipSell and rf_SL_L_state == -1)
        - strategy.exit("Long", from_entry="Long", stop=close, limit=close, when=slCondition_L)

        Pine Script TP conditions for Long:
        - strategy.exit("Long", from_entry="Long", stop=close, limit=close,
            when=st_TP_DualFlip_L_flipSell and (close - entryPriceDualFlip_L[1]) > riskRDual_L[1] * rr_mult_dual_L)
        """
        if i < 1:
            return None

        if not self._in_long and not self._in_short:
            return None

        row = self.df.iloc[i]
        time = row.get("time", i)
        close = row["close"]
        low = row["low"]    # For Long SL band check
        high = row["high"]  # For Short SL band check

        # ═══════════════════════════════════════════════════════
        # REVERSAL EXIT - Check if opposite entry signal exists on this bar
        # Must exit current position before entering opposite (same bar)
        # This handles Pine Script's automatic position reversal behavior
        # Note: check_exit is called BEFORE check_entry in backtest_engine
        # So we look at current bar's entry signals to detect reversal
        #
        # Pine Script logic:
        # wantEnterLong  = longDF_SignalNow or longRSI_SignalNow
        # wantEnterShort = shortDF_SignalNow or shortRSI_SignalNow
        # if inLong and wantEnterShort: exit Long positions
        # if inShort and wantEnterLong: exit Short positions
        # ═══════════════════════════════════════════════════════
        dual_flip_long = row.get("dual_flip_long", 0) == 1
        dual_flip_short = row.get("dual_flip_short", 0) == 1
        rf_state = row.get("rf_state", 0)
        rsi_bull_div = row.get("rsi_bull_div_signal", 0) == 1
        rsi_bear_div = row.get("rsi_bear_div_signal", 0) == 1
        rf_lb = row.get("rf_lb", np.nan)
        rf_hb = row.get("rf_hb", np.nan)
        rf_sl_L_lband = row.get("rf_sl_L_lband", np.nan)
        rf_sl_S_hband = row.get("rf_sl_S_hband", np.nan)

        # Check for reversal: ANY opposite signal would trigger entry
        # DualFlip Long signal - Pine: strategyLong = dualFlipLong and rf_state == 1
        strategy_long_signal = dual_flip_long and rf_state == 1
        # RSI Long signal (with valid risk)
        rsi_long_signal = rsi_bull_div and rf_state == 1
        rsi_long_valid = rsi_long_signal and not np.isnan(rf_sl_L_lband) and (close - rf_sl_L_lband) > 0

        # DualFlip Short signal - Pine: strategyShort = dualFlipShort (NO rf_state check!)
        strategy_short_signal = dual_flip_short  # Match Pine Script
        # RSI Short signal (with valid risk)
        rsi_short_signal = rsi_bear_div and rf_state == -1
        rsi_short_valid = rsi_short_signal and not np.isnan(rf_sl_S_hband) and (rf_sl_S_hband - close) > 0

        # wantEnterLong = DualFlip Long OR RSI Long (with valid risk)
        want_enter_long = (strategy_long_signal and self.show_entry_long) or (rsi_long_valid and self.show_entry_rsi_L)
        # wantEnterShort = DualFlip Short OR RSI Short (with valid risk)
        want_enter_short = (strategy_short_signal and self.show_entry_short) or (rsi_short_valid and self.show_entry_rsi_S)

        # ═══════════════════════════════════════════════════════
        # REVERSAL LOGIC - MATCH PINE SCRIPT
        # Pine: if inLong and wantEnterShort → exit ALL Long positions
        # Pine: if inShort and wantEnterLong → exit ALL Short positions
        # wantEnterLong = longDF_SignalNow or longRSI_SignalNow
        # wantEnterShort = shortDF_SignalNow or shortRSI_SignalNow
        # ═══════════════════════════════════════════════════════

        # Debug log for REVERSAL check when in position
        if (self._in_long or self._in_short) and self.debug:
            self._log(i, time, f"REVERSAL CHECK: in_long={self._in_long}, in_short={self._in_short}, want_enter_long={want_enter_long}, want_enter_short={want_enter_short}")

        # ANY Long position closed by ANY Short signal
        # DualFlip Long closed by wantEnterShort (DualFlip Short OR RSI Short)
        if self._in_dual_flip_long and want_enter_short and self.enable_short:
            self._log(i, time, f"EXIT DualFlip Long (REVERSAL) @ {close:.2f} - Short signal")
            self._add_debug_event(i, time, "exit_reversal", "Long", close, {
                "exit_id": "Long",
                "exit_type": "REVERSAL",
                "exit_bar": i,
                "exit_price": close,
                "entry_price": self._entry_price_dual_L,
                "pnl": close - self._entry_price_dual_L if not np.isnan(self._entry_price_dual_L) else 0,
                "reason": "Short signal reversal"
            })
            self._last_exit_bar_long = i
            self._reset_long_state()
            return ("Long", "REVERSAL")

        # RSI Long closed by wantEnterShort (DualFlip Short OR RSI Short)
        if self._in_rsi_long and want_enter_short and self.enable_short:
            self._log(i, time, f"EXIT RSI Long (REVERSAL) @ {close:.2f} - Short signal")
            self._add_debug_event(i, time, "exit_reversal", "RSI Long", close, {
                "exit_id": "RSI Long",
                "exit_type": "REVERSAL",
                "exit_bar": i,
                "exit_price": close,
                "entry_price": self._entry_price_rsi_L,
                "pnl": close - self._entry_price_rsi_L if not np.isnan(self._entry_price_rsi_L) else 0,
                "reason": "Short signal reversal"
            })
            self._last_exit_bar_long = i
            self._reset_long_state()
            return ("RSI Long", "REVERSAL")

        # ANY Short position closed by ANY Long signal
        # DualFlip Short closed by wantEnterLong (DualFlip Long OR RSI Long)
        if self._in_dual_flip_short and want_enter_long and self.enable_long:
            self._log(i, time, f"EXIT DualFlip Short (REVERSAL) @ {close:.2f} - Long signal")
            self._add_debug_event(i, time, "exit_reversal", "Short", close, {
                "exit_id": "Short",
                "exit_type": "REVERSAL",
                "exit_bar": i,
                "exit_price": close,
                "entry_price": self._entry_price_dual_S,
                "pnl": self._entry_price_dual_S - close if not np.isnan(self._entry_price_dual_S) else 0,
                "reason": "Long signal reversal"
            })
            self._last_exit_bar_short = i
            self._reset_short_state()
            return ("Short", "REVERSAL")

        # RSI Short closed by wantEnterLong (DualFlip Long OR RSI Long)
        if self._in_rsi_short and want_enter_long and self.enable_long:
            self._log(i, time, f"EXIT RSI Short (REVERSAL) @ {close:.2f} - Long signal")
            self._add_debug_event(i, time, "exit_reversal", "RSI Short", close, {
                "exit_id": "RSI Short",
                "exit_type": "REVERSAL",
                "exit_bar": i,
                "exit_price": close,
                "entry_price": self._entry_price_rsi_S,
                "pnl": self._entry_price_rsi_S - close if not np.isnan(self._entry_price_rsi_S) else 0,
                "reason": "Long signal reversal"
            })
            self._last_exit_bar_short = i
            self._reset_short_state()
            return ("RSI Short", "REVERSAL")

        # rf_state already retrieved above for reversal check (this is GLOBAL Entry RF state)
        # Get indicator values for Long SL/TP
        rf_sl_L_flip_sell = row.get("rf_sl_L_flipSell", 0) == 1
        st_sl_L_trend = row.get("st_sl_L_trend", 1)
        st_sl_L_flip_sell = row.get("st_sl_L_flipSell", 0) == 1
        rf_sl_L_lband = row.get("rf_sl_L_lband", np.nan)
        # Note: We use rf_state (GLOBAL) for cond2, not rf_sl_L_state (side-specific)
        # Per user requirement: (st_sl_flipSell_L and rf_state == -1)

        # Get indicator values for Short SL/TP
        rf_sl_S_flip_buy = row.get("rf_sl_S_flipBuy", 0) == 1
        st_sl_S_trend = row.get("st_sl_S_trend", 1)
        st_sl_S_flip_buy = row.get("st_sl_S_flipBuy", 0) == 1
        rf_sl_S_hband = row.get("rf_sl_S_hband", np.nan)
        # Note: We use rf_state (GLOBAL) for cond2, not rf_sl_S_state (side-specific)
        # Per user requirement: (st_sl_flipBuy_S and rf_state == 1)

        # TP flip signals
        st_tp_dual_L_flip_sell = row.get("st_tp_dual_L_flipSell", 0) == 1
        st_tp_dual_S_flip_buy = row.get("st_tp_dual_S_flipBuy", 0) == 1
        st_tp_rsi_L_flip_sell = row.get("st_tp_rsi_L_flipSell", 0) == 1
        st_tp_rsi_S_flip_buy = row.get("st_tp_rsi_S_flipBuy", 0) == 1

        # ═══════════════════════════════════════════════════════
        # EXIT LOGIC FOR DUAL FLIP LONG (ID: "Long")
        # EXIT NHÓM 1 - HARD SL: close <= RF SL LOW band
        # EXIT NHÓM 2 - SL ĐỘNG:
        #   (rf_sl_flipSell_L and st_sl_trend_L == -1) or
        #   (st_sl_flipSell_L and rf_state == -1)  <-- uses GLOBAL rf_state
        # ═══════════════════════════════════════════════════════
        if self._in_dual_flip_long:
            # NHÓM 1 - HARD SL: close <= RF SL LOW band (checked first as it's the hard stop)
            sl_cond_hard = not np.isnan(self._entry_price_dual_L) and not np.isnan(rf_sl_L_lband) and close <= rf_sl_L_lband

            # NHÓM 2 - SL ĐỘNG:
            # cond1: RF SL flip sell + ST SL trend confirms bearish
            sl_cond1 = rf_sl_L_flip_sell and st_sl_L_trend == -1
            # cond2: ST SL flip sell + GLOBAL rf_state confirms bearish
            sl_cond2 = st_sl_L_flip_sell and rf_state == -1  # USES GLOBAL rf_state

            sl_condition_L = sl_cond_hard or sl_cond1 or sl_cond2

            if sl_condition_L:
                self._log(i, time, f"EXIT Long SL @ {close:.2f} | rf_sl_L_flipSell={rf_sl_L_flip_sell}, st_sl_L_trend={st_sl_L_trend}, st_sl_L_flipSell={st_sl_L_flip_sell}, rf_state={rf_state}, rf_sl_L_lband={rf_sl_L_lband}, close={close}")
                self._add_debug_event(i, time, "exit_sl", "Long", close, {
                    "exit_id": "Long",
                    "exit_type": "SL",
                    "exit_bar": i,
                    "exit_price": close,
                    "entry_price": self._entry_price_dual_L,
                    "pnl": close - self._entry_price_dual_L if not np.isnan(self._entry_price_dual_L) else 0,
                    "rf_sl_L_flipSell": rf_sl_L_flip_sell,
                    "st_sl_L_trend": st_sl_L_trend,
                    "st_sl_L_flipSell": st_sl_L_flip_sell,
                    "rf_state": rf_state,
                    "rf_sl_L_lband": rf_sl_L_lband,
                    "sl_cond_hard": sl_cond_hard,
                    "sl_cond1": sl_cond1,
                    "sl_cond2": sl_cond2
                })
                self._last_exit_bar_long = i  # Track exit bar to prevent same-bar re-entry
                self._reset_long_state()
                return ("Long", "SL")

            # TP condition - EXACT match to Pine Script
            # st_TP_DualFlip_L_flipSell and (close - entryPriceDualFlip_L[1]) > riskRDual_L[1] * rr_mult_dual_L
            if st_tp_dual_L_flip_sell and not np.isnan(self._entry_price_dual_L) and not np.isnan(self._risk_r_dual_L):
                current_pnl = close - self._entry_price_dual_L
                tp_target = self._risk_r_dual_L * self.rr_mult_dual_L

                if current_pnl > tp_target:
                    self._log(i, time, f"EXIT Long TP @ {close:.2f} | PnL={current_pnl:.2f} > Target={tp_target:.2f}")
                    self._add_debug_event(i, time, "exit_tp", "Long", close, {
                        "exit_id": "Long",
                        "exit_type": "TP",
                        "exit_bar": i,
                        "exit_price": close,
                        "entry_price": self._entry_price_dual_L,
                        "pnl": current_pnl,
                        "tp_target": tp_target,
                        "risk_r": self._risk_r_dual_L,
                        "rr_mult": self.rr_mult_dual_L,
                        "st_tp_dual_L_flipSell": st_tp_dual_L_flip_sell
                    })
                    self._last_exit_bar_long = i  # Track exit bar to prevent same-bar re-entry
                    self._reset_long_state()
                    return ("Long", "TP")

        # ═══════════════════════════════════════════════════════
        # EXIT LOGIC FOR DUAL FLIP SHORT (ID: "Short")
        # Pine Script (EXACT):
        # slCondition_S = (rf_SL_S_flipBuy and st_SL_S_trend == 1) or (st_SL_S_flipBuy and rf_SL_S_state == 1) or (not na(entryPriceDualFlip_S) and close >= rf_SL_S_hband)
        # tpCondition_S = st_TP_DualFlip_S_flipBuy and (entryPriceDualFlip_S[1] - close) > riskRDual_S[1] * rr_mult_dual_S
        # NOTE: Pine Script does NOT have close > entryPrice check in cond1/cond2
        # NOTE: RF SL hard stop is triggered ONLY by candle CLOSE crossing the band.
        #       High/Low wick penetration must be ignored. Exit price is always current bar close.
        # ═══════════════════════════════════════════════════════
        # EXIT NHÓM 2 – SL ĐỘNG (Short): Uses GLOBAL rf_state
        #   (rf_sl_flipBuy_S and st_sl_trend_S == 1)  <-- RF SL flip + ST SL trend confirms bullish
        #   (st_sl_flipBuy_S and rf_state == 1)  <-- uses GLOBAL rf_state
        # ═══════════════════════════════════════════════════════
        if self._in_dual_flip_short:
            # NHÓM 1 - HARD SL: close >= RF SL HIGH band (checked first as it's the hard stop)
            sl_cond_hard = not np.isnan(self._entry_price_dual_S) and not np.isnan(rf_sl_S_hband) and close >= rf_sl_S_hband

            # NHÓM 2 - SL ĐỘNG:
            # cond1: RF SL flip buy + ST SL trend confirms bullish
            sl_cond1 = rf_sl_S_flip_buy and st_sl_S_trend == 1
            # cond2: ST SL flip buy + GLOBAL rf_state confirms bullish
            sl_cond2 = st_sl_S_flip_buy and rf_state == 1  # USES GLOBAL rf_state

            sl_condition_S = sl_cond_hard or sl_cond1 or sl_cond2

            if sl_condition_S:
                self._log(i, time, f"EXIT Short SL @ {close:.2f} | sl_cond_hard={sl_cond_hard}, sl_cond1={sl_cond1}, sl_cond2={sl_cond2}, rf_state={rf_state}, rf_sl_S_hband={rf_sl_S_hband}, close={close}")
                self._add_debug_event(i, time, "exit_sl", "Short", close, {
                    "exit_id": "Short",
                    "exit_type": "SL",
                    "exit_bar": i,
                    "exit_price": close,
                    "entry_price": self._entry_price_dual_S,
                    "pnl": self._entry_price_dual_S - close if not np.isnan(self._entry_price_dual_S) else 0,
                    "rf_sl_S_flipBuy": rf_sl_S_flip_buy,
                    "st_sl_S_trend": st_sl_S_trend,
                    "st_sl_S_flipBuy": st_sl_S_flip_buy,
                    "rf_state": rf_state,
                    "rf_sl_S_hband": rf_sl_S_hband,
                    "sl_cond_hard": sl_cond_hard,
                    "sl_cond1": sl_cond1,
                    "sl_cond2": sl_cond2
                })
                self._last_exit_bar_short = i  # Track exit bar to prevent same-bar re-entry
                self._reset_short_state()
                return ("Short", "SL")

            # TP condition
            if st_tp_dual_S_flip_buy and not np.isnan(self._entry_price_dual_S) and not np.isnan(self._risk_r_dual_S):
                current_pnl = self._entry_price_dual_S - close
                tp_target = self._risk_r_dual_S * self.rr_mult_dual_S

                if current_pnl > tp_target:
                    self._log(i, time, f"EXIT Short TP @ {close:.2f} | PnL={current_pnl:.2f} > Target={tp_target:.2f}")
                    self._add_debug_event(i, time, "exit_tp", "Short", close, {
                        "exit_id": "Short",
                        "exit_type": "TP",
                        "exit_bar": i,
                        "exit_price": close,
                        "entry_price": self._entry_price_dual_S,
                        "pnl": current_pnl,
                        "tp_target": tp_target,
                        "risk_r": self._risk_r_dual_S,
                        "rr_mult": self.rr_mult_dual_S,
                        "st_tp_dual_S_flipBuy": st_tp_dual_S_flip_buy
                    })
                    self._last_exit_bar_short = i  # Track exit bar to prevent same-bar re-entry
                    self._reset_short_state()
                    return ("Short", "TP")

        # ═══════════════════════════════════════════════════════
        # EXIT LOGIC FOR RSI LONG (ID: "RSI Long")
        # ═══════════════════════════════════════════════════════
        # EXIT LOGIC FOR RSI LONG (ID: "RSI Long")
        # NHÓM 1 – HARD SL: close <= RF SL LOW band (same as Dual Flip)
        # NHÓM 2 – SL ĐỘNG: Uses GLOBAL rf_state
        #   (rf_sl_flipSell_L and st_sl_trend_L == -1) or (st_sl_flipSell_L and rf_state == -1)
        #   + close <= entryPrice (must be losing) + validRSISL_L
        # ═══════════════════════════════════════════════════════
        if self._in_rsi_long:
            # Valid RSI SL check (Pine: validRSISL_L = not na(rfSlAtEntryRSI_L) and not na(entryPriceRSI_L) and rfSlAtEntryRSI_L < entryPriceRSI_L)
            valid_rsi_sl_L = (not np.isnan(self._entry_price_rsi_L) and
                             not np.isnan(self._rf_sl_at_entry_rsi_L) and
                             self._rf_sl_at_entry_rsi_L < self._entry_price_rsi_L)

            # NHÓM 1 - HARD SL: close <= RF SL band AT ENTRY (not current band!)
            # Pine: rsiSLCondition_L = validRSISL_L and (close <= rfSlAtEntryRSI_L or ...)
            sl_cond_hard = valid_rsi_sl_L and close <= self._rf_sl_at_entry_rsi_L

            if sl_cond_hard:
                self._log(i, time, f"EXIT RSI Long HARD SL @ {close:.2f} | close <= rf_sl_at_entry={self._rf_sl_at_entry_rsi_L:.2f}")
                self._add_debug_event(i, time, "exit_sl", "RSI Long", close, {
                    "exit_id": "RSI Long",
                    "exit_type": "HARD_SL",
                    "exit_bar": i,
                    "exit_price": close,
                    "entry_price": self._entry_price_rsi_L,
                    "pnl": close - self._entry_price_rsi_L if not np.isnan(self._entry_price_rsi_L) else 0,
                    "rf_sl_at_entry": self._rf_sl_at_entry_rsi_L,
                    "rf_sl_L_lband_current": rf_sl_L_lband,
                    "sl_cond_hard": sl_cond_hard
                })
                self._last_exit_bar_long = i
                self._reset_rsi_long_state()
                return ("RSI Long", "SL")

            # NHÓM 2 - SL ĐỘNG: valid_rsi_sl_L already defined above
            # SL condition for Long - Uses GLOBAL rf_state
            # (rf_sl_flipSell_L and st_sl_trend_L == -1) or (st_sl_flipSell_L and rf_state == -1)
            sl_cond1 = rf_sl_L_flip_sell and st_sl_L_trend == -1
            sl_cond2 = st_sl_L_flip_sell and rf_state == -1  # USES GLOBAL rf_state
            sl_condition_L = sl_cond1 or sl_cond2

            # RSI SL condition:
            # close <= entryPrice (must be losing) + slCondition_L + validRSISL_L
            close_below_entry = close <= self._entry_price_rsi_L
            rsi_sl_condition_L = close_below_entry and sl_condition_L and valid_rsi_sl_L

            if rsi_sl_condition_L:
                self._log(i, time, f"EXIT RSI Long SL @ {close:.2f} | close <= entry={close_below_entry}, slCondition_L={sl_condition_L}, valid={valid_rsi_sl_L}")
                self._add_debug_event(i, time, "exit_sl", "RSI Long", close, {
                    "exit_id": "RSI Long",
                    "exit_type": "SL",
                    "exit_bar": i,
                    "exit_price": close,
                    "entry_price": self._entry_price_rsi_L,
                    "pnl": close - self._entry_price_rsi_L if not np.isnan(self._entry_price_rsi_L) else 0,
                    "rf_sl_at_entry": self._rf_sl_at_entry_rsi_L,
                    "valid_rsi_sl": valid_rsi_sl_L,
                    "sl_condition_L": sl_condition_L,
                    "close_below_entry": close_below_entry,
                    "rf_state": rf_state
                })
                self._last_exit_bar_long = i  # Track exit bar to prevent same-bar re-entry
                self._reset_rsi_long_state()
                return ("RSI Long", "SL")

            # TP condition: st_TP_RSI1_L_flipSell and (close - entryPriceRSI1_L[1]) > riskRRSI1_L[1] * rr_mult_rsi1_L
            if st_tp_rsi_L_flip_sell and not np.isnan(self._entry_price_rsi_L) and not np.isnan(self._risk_r_rsi_L):
                current_pnl = close - self._entry_price_rsi_L
                tp_target = self._risk_r_rsi_L * self.rr_mult_rsi_L

                if current_pnl > tp_target:
                    self._log(i, time, f"EXIT RSI Long TP @ {close:.2f} | PnL={current_pnl:.2f} > Target={tp_target:.2f}")
                    self._add_debug_event(i, time, "exit_tp", "RSI Long", close, {
                        "exit_id": "RSI Long",
                        "exit_type": "TP",
                        "exit_bar": i,
                        "exit_price": close,
                        "entry_price": self._entry_price_rsi_L,
                        "pnl": current_pnl,
                        "tp_target": tp_target,
                        "risk_r": self._risk_r_rsi_L,
                        "rr_mult": self.rr_mult_rsi_L,
                        "st_tp_rsi_L_flipSell": st_tp_rsi_L_flip_sell
                    })
                    self._last_exit_bar_long = i  # Track exit bar to prevent same-bar re-entry
                    self._reset_rsi_long_state()
                    return ("RSI Long", "TP")

        # ═══════════════════════════════════════════════════════
        # EXIT LOGIC FOR RSI SHORT (ID: "RSI Short")
        # NHÓM 1 – HARD SL: close >= RF SL HIGH band (same as Dual Flip)
        # NHÓM 2 – SL ĐỘNG: Uses GLOBAL rf_state
        #   (rf_sl_flipBuy_S and st_sl_trend_S == 1) or (st_sl_flipBuy_S and rf_state == 1)
        #   + close >= entryPrice (must be losing) + validRSISL_S
        # ═══════════════════════════════════════════════════════
        if self._in_rsi_short:
            # Valid RSI SL check (Pine: validRSISL_S = not na(rfSlAtEntryRSI_S) and not na(entryPriceRSI_S) and rfSlAtEntryRSI_S > entryPriceRSI_S)
            valid_rsi_sl_S = (not np.isnan(self._entry_price_rsi_S) and
                             not np.isnan(self._rf_sl_at_entry_rsi_S) and
                             self._rf_sl_at_entry_rsi_S > self._entry_price_rsi_S)

            # NHÓM 1 - HARD SL: close >= RF SL band AT ENTRY (not current band!)
            # Pine: rsiSLCondition_S = validRSISL_S and (close >= rfSlAtEntryRSI_S or ...)
            sl_cond_hard = valid_rsi_sl_S and close >= self._rf_sl_at_entry_rsi_S

            if sl_cond_hard:
                self._log(i, time, f"EXIT RSI Short HARD SL @ {close:.2f} | close >= rf_sl_at_entry={self._rf_sl_at_entry_rsi_S:.2f}")
                self._add_debug_event(i, time, "exit_sl", "RSI Short", close, {
                    "exit_id": "RSI Short",
                    "exit_type": "HARD_SL",
                    "exit_bar": i,
                    "exit_price": close,
                    "entry_price": self._entry_price_rsi_S,
                    "pnl": self._entry_price_rsi_S - close if not np.isnan(self._entry_price_rsi_S) else 0,
                    "rf_sl_at_entry": self._rf_sl_at_entry_rsi_S,
                    "rf_sl_S_hband_current": rf_sl_S_hband,
                    "sl_cond_hard": sl_cond_hard
                })
                self._last_exit_bar_short = i
                self._reset_rsi_short_state()
                return ("RSI Short", "SL")

            # NHÓM 2 - SL ĐỘNG: valid_rsi_sl_S already defined above
            # SL condition for Short - Uses GLOBAL rf_state
            # (rf_sl_flipBuy_S and st_sl_trend_S == 1) or (st_sl_flipBuy_S and rf_state == 1)
            sl_cond1 = rf_sl_S_flip_buy and st_sl_S_trend == 1  # RF SL flip buy + ST SL trend confirms bullish
            sl_cond2 = st_sl_S_flip_buy and rf_state == 1  # USES GLOBAL rf_state
            sl_condition_S = sl_cond1 or sl_cond2

            # RSI SL condition:
            # close >= entryPrice (must be losing) + slCondition_S + validRSISL_S
            close_above_entry = close >= self._entry_price_rsi_S
            rsi_sl_condition_S = close_above_entry and sl_condition_S and valid_rsi_sl_S

            if rsi_sl_condition_S:
                self._log(i, time, f"EXIT RSI Short SL @ {close:.2f} | close >= entry={close_above_entry}, slCondition_S={sl_condition_S}, valid={valid_rsi_sl_S}")
                self._add_debug_event(i, time, "exit_sl", "RSI Short", close, {
                    "exit_id": "RSI Short",
                    "exit_type": "SL",
                    "exit_bar": i,
                    "exit_price": close,
                    "entry_price": self._entry_price_rsi_S,
                    "pnl": self._entry_price_rsi_S - close if not np.isnan(self._entry_price_rsi_S) else 0,
                    "rf_sl_at_entry": self._rf_sl_at_entry_rsi_S,
                    "valid_rsi_sl": valid_rsi_sl_S,
                    "sl_condition_S": sl_condition_S,
                    "close_above_entry": close_above_entry,
                    "rf_state": rf_state
                })
                self._last_exit_bar_short = i  # Track exit bar to prevent same-bar re-entry
                self._reset_rsi_short_state()
                return ("RSI Short", "SL")

            # TP condition
            if st_tp_rsi_S_flip_buy and not np.isnan(self._entry_price_rsi_S) and not np.isnan(self._risk_r_rsi_S):
                current_pnl = self._entry_price_rsi_S - close
                tp_target = self._risk_r_rsi_S * self.rr_mult_rsi_S

                if current_pnl > tp_target:
                    self._log(i, time, f"EXIT RSI Short TP @ {close:.2f} | PnL={current_pnl:.2f} > Target={tp_target:.2f}")
                    self._add_debug_event(i, time, "exit_tp", "RSI Short", close, {
                        "exit_id": "RSI Short",
                        "exit_type": "TP",
                        "exit_bar": i,
                        "exit_price": close,
                        "entry_price": self._entry_price_rsi_S,
                        "pnl": current_pnl,
                        "tp_target": tp_target,
                        "risk_r": self._risk_r_rsi_S,
                        "rr_mult": self.rr_mult_rsi_S,
                        "st_tp_rsi_S_flipBuy": st_tp_rsi_S_flip_buy
                    })
                    self._last_exit_bar_short = i  # Track exit bar to prevent same-bar re-entry
                    self._reset_rsi_short_state()
                    return ("RSI Short", "TP")

        return None

    def _reset_long_state(self):
        """Reset Long position state"""
        self._in_dual_flip_long = False
        self._in_rsi_long = False
        self._in_long = False
        self._entry_price_dual_L = np.nan
        self._rf_sl_at_entry_dual_L = np.nan
        self._risk_r_dual_L = np.nan
        self._entry_price_rsi_L = np.nan
        self._rf_sl_at_entry_rsi_L = np.nan
        self._risk_r_rsi_L = np.nan

    def _reset_short_state(self):
        """Reset Short position state"""
        self._in_dual_flip_short = False
        self._in_rsi_short = False
        self._in_short = False
        self._entry_price_dual_S = np.nan
        self._rf_sl_at_entry_dual_S = np.nan
        self._risk_r_dual_S = np.nan
        self._entry_price_rsi_S = np.nan
        self._rf_sl_at_entry_rsi_S = np.nan
        self._risk_r_rsi_S = np.nan

    def _reset_rsi_long_state(self):
        """Reset RSI Long position state"""
        self._in_rsi_long = False
        if not self._in_dual_flip_long:
            self._in_long = False
        self._entry_price_rsi_L = np.nan
        self._rf_sl_at_entry_rsi_L = np.nan
        self._risk_r_rsi_L = np.nan

    def _reset_rsi_short_state(self):
        """Reset RSI Short position state"""
        self._in_rsi_short = False
        if not self._in_dual_flip_short:
            self._in_short = False
        self._entry_price_rsi_S = np.nan
        self._rf_sl_at_entry_rsi_S = np.nan
        self._risk_r_rsi_S = np.nan

    def _init_position_state(self):
        """Initialize position tracking state"""
        # Long tracking
        self._in_long = False
        self._in_dual_flip_long = False
        self._in_rsi_long = False
        self._entry_type = None

        # Exit bar tracking - prevents re-entry on same bar after exit
        # This matches TradingView behavior where exit and entry don't happen same bar
        self._last_exit_bar_long = -1
        self._last_exit_bar_short = -1

        # Dual Flip Long tracking
        self._entry_price_dual_L = np.nan
        self._rf_sl_at_entry_dual_L = np.nan
        self._risk_r_dual_L = np.nan
        self._dual_flip_signal_close = np.nan
        self._dual_flip_signal_rf_lb = np.nan

        # RSI Long tracking
        self._entry_price_rsi_L = np.nan
        self._rf_sl_at_entry_rsi_L = np.nan
        self._risk_r_rsi_L = np.nan

        # Short tracking
        self._in_short = False
        self._in_dual_flip_short = False
        self._in_rsi_short = False

        # Dual Flip Short tracking
        self._entry_price_dual_S = np.nan
        self._rf_sl_at_entry_dual_S = np.nan
        self._risk_r_dual_S = np.nan
        self._dual_flip_signal_close_S = np.nan
        self._dual_flip_signal_rf_hb = np.nan

        # RSI Short tracking
        self._entry_price_rsi_S = np.nan
        self._rf_sl_at_entry_rsi_S = np.nan
        self._risk_r_rsi_S = np.nan

    def get_debug_events(self) -> List[dict]:
        """Return debug events for frontend visualization"""
        return self.debug_events

    def generate_flip_events(self):
        """
        Generate debug events for RF/ST flips and DualFlip signals.
        Called after prepare_indicators() to create events matching TradingView markers.

        Event Types:
        - RF_FLIP_BUY: Range Filter flip to bullish
        - RF_FLIP_SELL: Range Filter flip to bearish
        - ST_FLIP_BUY: SuperTrend flip to bullish
        - ST_FLIP_SELL: SuperTrend flip to bearish
        - DUALFLIP_LONG_SIGNAL: DualFlip Long signal (marker "Buy") - ONLY on first occurrence
        - DUALFLIP_SHORT_SIGNAL: DualFlip Short signal (marker "Sell") - ONLY on first occurrence
        - RSI_BULL_DIV: RSI Bullish Divergence signal
        - RSI_BEAR_DIV: RSI Bearish Divergence signal
        """
        n = len(self.df)

        # Track last marker type SEPARATELY for RF and ST
        # Each indicator has its own alternating Buy-Sell-Buy-Sell pattern
        # None = no marker yet, "BUY" = last was buy, "SELL" = last was sell
        rf_last_marker = None  # For Range Filter flip markers
        st_last_marker = None  # For SuperTrend flip markers

        for i in range(n):
            row = self.df.iloc[i]
            time_val = row.get("time", i)
            low = float(row["low"])
            high = float(row["high"])

            # Convert time to Unix timestamp (seconds) to match candles format
            # Candles use to_hcm_timestamp() which returns seconds, not milliseconds
            if hasattr(time_val, 'timestamp'):
                # Add UTC+7 offset for consistency with chart_data.py
                from datetime import timezone, timedelta
                TZ_HCM = timezone(timedelta(hours=7))
                if time_val.tzinfo is None:
                    time_val = time_val.replace(tzinfo=timezone.utc)
                dt_hcm = time_val.astimezone(TZ_HCM)
                time_sec = int(dt_hcm.timestamp())
            elif isinstance(time_val, (int, float)):
                # If already a timestamp, assume it's seconds (match candles)
                time_sec = int(time_val)
            else:
                time_sec = str(time_val)

            # Get indicator values
            rf_flip_buy = row.get("rf_flipBuy", 0) == 1
            rf_flip_sell = row.get("rf_flipSell", 0) == 1
            st_flip_buy = row.get("st_flipBuy", 0) == 1
            st_flip_sell = row.get("st_flipSell", 0) == 1
            dual_flip_long = row.get("dual_flip_long", 0) == 1
            dual_flip_short = row.get("dual_flip_short", 0) == 1
            rf_state = int(row.get("rf_state", 0))
            st_trend = int(row.get("st_trend", 0))
            rsi_bull_div = row.get("rsi_bull_div_signal", 0) == 1
            rsi_bear_div = row.get("rsi_bear_div_signal", 0) == 1

            # Bars since values
            bars_rf_B = int(row.get("bars_rf_flipBuy", 9999))
            bars_st_B = int(row.get("bars_st_flipBuy", 9999))
            bars_rf_S = int(row.get("bars_rf_flipSell", 9999))
            bars_st_S = int(row.get("bars_st_flipSell", 9999))

            # Common snapshot for all events at this bar
            # Ensure all values are native Python types (not numpy)
            snapshot = {
                "rf_state": int(rf_state),
                "rf_flipBuy": bool(rf_flip_buy),
                "rf_flipSell": bool(rf_flip_sell),
                "st_trend": int(st_trend),
                "st_flipBuy": bool(st_flip_buy),
                "st_flipSell": bool(st_flip_sell),
                "barsRF_B": int(bars_rf_B),
                "barsST_B": int(bars_st_B),
                "barsRF_S": int(bars_rf_S),
                "barsST_S": int(bars_st_S),
                "dualFlipLong": bool(dual_flip_long),
                "dualFlipShort": bool(dual_flip_short),
                "showEntryLong": bool(self.show_entry_long),
                "showEntryShort": bool(self.show_entry_short),
                "enableLong": bool(self.enable_long),
                "enableShort": bool(self.enable_short),
            }

            # ===== RF (Range Filter) MARKERS - alternating Buy-Sell =====
            # RF_FLIP_BUY: show "Buy" marker if last RF marker was not BUY
            if rf_flip_buy and rf_last_marker != "BUY":
                self.debug_events.append({
                    "bar_index": i,
                    "time": time_sec,
                    "price": low,  # Buy below candle
                    "event_type": "RF_FLIP_BUY",
                    "marker_type": "BUY",
                    "source": "RF",
                    "tag": "RF",
                    "snapshot": snapshot.copy()
                })
                rf_last_marker = "BUY"

            # RF_FLIP_SELL: show "Sell" marker if last RF marker was BUY
            if rf_flip_sell and rf_last_marker == "BUY":
                self.debug_events.append({
                    "bar_index": i,
                    "time": time_sec,
                    "price": high,  # Sell above candle
                    "event_type": "RF_FLIP_SELL",
                    "marker_type": "SELL",
                    "source": "RF",
                    "tag": "RF",
                    "snapshot": snapshot.copy()
                })
                rf_last_marker = "SELL"

            # ===== ST (SuperTrend) MARKERS - alternating Buy-Sell =====
            # ST_FLIP_BUY: show "Buy" marker if last ST marker was not BUY
            if st_flip_buy and st_last_marker != "BUY":
                self.debug_events.append({
                    "bar_index": i,
                    "time": time_sec,
                    "price": low,  # Buy below candle
                    "event_type": "ST_FLIP_BUY",
                    "marker_type": "BUY",
                    "source": "ST",
                    "tag": "ST",
                    "snapshot": snapshot.copy()
                })
                st_last_marker = "BUY"

            # ST_FLIP_SELL: show "Sell" marker if last ST marker was BUY
            if st_flip_sell and st_last_marker == "BUY":
                self.debug_events.append({
                    "bar_index": i,
                    "time": time_sec,
                    "price": high,  # Sell above candle
                    "event_type": "ST_FLIP_SELL",
                    "marker_type": "SELL",
                    "source": "ST",
                    "tag": "ST",
                    "snapshot": snapshot.copy()
                })
                st_last_marker = "SELL"

            # ===== RSI DIVERGENCE EVENTS (logged without markers) =====
            if rsi_bull_div:
                self.debug_events.append({
                    "bar_index": i,
                    "time": time_sec,
                    "price": low,
                    "event_type": "RSI_BULL_DIV",
                    "marker_type": None,  # No marker, just event log
                    "source": "RSI",
                    "tag": "RSI Long" if rf_state == 1 else None,
                    "snapshot": snapshot.copy()
                })

            if rsi_bear_div:
                self.debug_events.append({
                    "bar_index": i,
                    "time": time_sec,
                    "price": high,
                    "event_type": "RSI_BEAR_DIV",
                    "marker_type": None,  # No marker, just event log
                    "source": "RSI",
                    "tag": "RSI Short" if rf_state == -1 else None,
                    "snapshot": snapshot.copy()
                })


def create_strategy(df: pd.DataFrame, params: dict) -> RFSTRSICombinedStrategy:
    """Factory function for strategy creation"""
    return RFSTRSICombinedStrategy(df, params)
