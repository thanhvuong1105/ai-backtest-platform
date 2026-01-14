# engine/backtest_engine.py
import json
import sys
import os
from datetime import timezone, timedelta

import numpy as np
import pandas as pd

from .data_loader import load_csv
from .metrics import calculate_metrics
from .strategies.factory import create_strategy, USE_FAST_STRATEGY, FAST_AVAILABLE

# Try to import Numba fast backtest loop
try:
    from .numba_indicators import run_backtest_loop
    NUMBA_BACKTEST_AVAILABLE = True
except ImportError:
    NUMBA_BACKTEST_AVAILABLE = False

# UTC+7 Ho Chi Minh timezone
TZ_HCM = timezone(timedelta(hours=7))


def to_hcm_time(dt_str):
    """Convert datetime string to UTC+7 Ho Chi Minh timezone string"""
    try:
        dt = pd.to_datetime(dt_str)
        if dt.tzinfo is None:
            # Assume UTC if no timezone
            dt = dt.replace(tzinfo=timezone.utc)
        dt_hcm = dt.astimezone(TZ_HCM)
        return dt_hcm.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(dt_str)




def run_backtest(strategy_config):
    # ===== META =====
    symbol = strategy_config["meta"]["symbols"][0]
    timeframe = strategy_config["meta"]["timeframe"]

    strategy_type = strategy_config["strategy"]["type"]
    strategy_params = strategy_config["strategy"]["params"]

    # ===== COST CONFIG =====
    # Use `or {}` to handle explicit None values
    costs = strategy_config.get("costs") or {}
    fee_rate = costs.get("fee", 0.0)
    slippage_ticks = costs.get("slippage", 0.0)

    # ===== ACCOUNT CONFIG =====
    initial_equity = float(strategy_config.get("initial_equity", 10000.0))
    # order size config: percent of equity hoặc fixed USDT
    # Use `or {}` to handle explicit None values
    props = strategy_config.get("properties") or {}
    order_cfg = props.get("orderSize") or {}
    order_type = (order_cfg.get("type") or "percent").lower()
    order_value = float(order_cfg.get("value", strategy_config.get("positionSize", 1.0)))
    pyramiding = max(int(props.get("pyramiding", 1) or 1), 1)
    # compound=True: sizing % dựa trên equity hiện tại (giống TradingView)
    # compound=False: sizing % dựa trên initial_equity cố định
    compound = bool(props.get("compound", True))
    # slippage ticks (TradingView style)
    if props.get("slippage") is not None:
        slippage_ticks = float(props.get("slippage"))

    # ===== LOAD DATA (must be before tick_size calculation) =====
    df = strategy_config.get("df")
    df_is_preprocessed = strategy_config.get("_preprocessed", False)
    if df is None:
        df = load_csv(symbol, timeframe)

    # tick size: lấy từ props hoặc đoán theo độ chính xác giá
    tick_size = props.get("tickSize") or props.get("tick_size")
    if tick_size is None:
        try:
            sample_price = float(df["close"].iloc[0])
            s = f"{sample_price:.10f}".rstrip("0").rstrip(".")
            decimals = len(s.split(".")[1]) if "." in s else 0
            tick_size = 10 ** (-decimals) if decimals > 0 else 1.0
        except Exception:
            tick_size = 1.0
    tick_size = float(tick_size)

    # ===== PARSE DATE RANGE (for execution filtering, NOT indicator calculation) =====
    date_range = strategy_config.get("range") or {}
    from_date = date_range.get("from")
    to_date = date_range.get("to")

    # Convert to pandas Timestamp for comparison
    from_date_ts = pd.to_datetime(from_date) if from_date else None
    to_date_ts = pd.to_datetime(to_date) if to_date else None

    # ===== HANDLE DATAFRAME =====
    # IMPORTANT: Always copy because cached DataFrame is shared across threads
    # But skip heavy processing if already pre-processed by multi_runner
    if df is not None and not df.empty:
        df = df.copy()

        if not df_is_preprocessed:
            # Only do date processing if not already done
            if not pd.api.types.is_datetime64_any_dtype(df["time"]):
                df["time"] = pd.to_datetime(df["time"])

            # Filter data up to to_date for indicator calculation
            if to_date_ts is not None:
                df = df[df["time"] <= to_date_ts].reset_index(drop=True)

            # ===== PERFORMANCE OPTIMIZATION =====
            # Instead of using ALL historical data (which can be 40K+ candles),
            # we use a lookback buffer before from_date for indicator warmup.
            # This gives identical results but is MUCH faster.
            #
            # Lookback buffer: 300 bars is enough for most indicators:
            # - EMA 200: needs ~200 bars for convergence
            # - RSI 14: needs ~14 bars
            # - ATR 14: needs ~14 bars
            # - SuperTrend: needs ATR warmup (~50 bars)
            # - 300 bars provides 1.5x safety margin (optimized for speed)
            #
            INDICATOR_WARMUP_BARS = 300

            if from_date_ts is not None:
                # Find index of from_date
                mask = df["time"] >= from_date_ts
                if mask.any():
                    from_idx = mask.idxmax()
                    # Keep warmup buffer before from_date for indicator calculation
                    start_idx = max(0, from_idx - INDICATOR_WARMUP_BARS)
                    df = df.iloc[start_idx:].reset_index(drop=True)

    # Cắt window nếu có (vd: lấy N bar cuối) - for debugging only
    window = strategy_config.get("window")
    if window:
        if isinstance(window, int) and window > 0:
            df = df.tail(window).reset_index(drop=True)
        elif isinstance(window, (list, tuple)) and len(window) == 2:
            start, end = window
            df = df.iloc[start:end].reset_index(drop=True)

    if df.empty:
        return {
            "meta": {
                "strategyId": f"{strategy_type}_v1",
                "strategyName": strategy_type,
                "symbol": symbol,
                "timeframe": timeframe
            },
            "summary": calculate_metrics([], [], initial_equity),
            "equityCurve": [],
            "trades": []
        }

    # ===== INIT STRATEGY & CALCULATE INDICATORS ON FULL DATA =====
    # CRITICAL: Indicators are calculated on FULL history (from symbol start to to_date)
    # This ensures proper warm-up for EMA, RSI, ATR, SuperTrend, Range Filter, etc.
    strategy = create_strategy(strategy_type, df, strategy_params)
    strategy.prepare_indicators()

    # ===== DETERMINE EXECUTION START INDEX =====
    # Backtest execution only happens within the user's selected Range
    # But indicators are already calculated on full history above
    execution_start_idx = 0
    if from_date_ts is not None:
        # Find the first bar >= from_date
        mask = df["time"] >= from_date_ts
        if mask.any():
            execution_start_idx = mask.idxmax()
        else:
            # No bars in range
            return {
                "meta": {
                    "strategyId": f"{strategy_type}_v1",
                    "strategyName": strategy_type,
                    "symbol": symbol,
                    "timeframe": timeframe
                },
                "summary": calculate_metrics([], [], initial_equity),
                "equityCurve": [],
                "trades": []
            }

    # ===== FAST NUMBA BACKTEST PATH =====
    # Use Numba-optimized backtest loop if available and strategy supports it
    use_fast_backtest = (
        NUMBA_BACKTEST_AVAILABLE and
        USE_FAST_STRATEGY and
        FAST_AVAILABLE and
        strategy_type == "rf_st_rsi" and
        hasattr(strategy, 'get_indicator_arrays')
    )

    if use_fast_backtest:
        try:
            # Get arrays from strategy
            indicators = strategy.get_indicator_arrays()
            price_arrays = strategy.get_price_arrays()

            # Determine order type for Numba (0=percent, 1=fixed)
            order_type_num = 0 if order_type == "percent" else 1

            # Calculate slippage
            slip = slippage_ticks * tick_size

            # Run Numba-optimized backtest loop
            (entry_bars, exit_bars, entry_prices, exit_prices,
             sizes, pnls, entry_types, equity_arr) = run_backtest_loop(
                n=len(df),
                execution_start_idx=execution_start_idx,
                price_open=price_arrays["open"],
                price_close=price_arrays["close"],
                dual_flip_long=indicators["dual_flip_long"],
                rsi_bull_div_signal=indicators["rsi_bull_div_signal"],
                rf_state=indicators["rf_state"],
                rf_lb=indicators["rf_lb"],
                rf_sl_lband=indicators["rf_sl_lband"],
                rf_sl_flip_sell=indicators["rf_sl_flipSell"],
                st_sl_trend=indicators["st_sl_trend"],
                st_sl_flip_sell=indicators["st_sl_flipSell"],
                rf_sl_state=indicators["rf_sl_state"],
                st_tp_dual_flip_sell=indicators["st_tp_dual_flipSell"],
                st_tp_rsi_flip_sell=indicators["st_tp_rsi_flipSell"],
                show_entry_long=strategy_params.get("showDualFlip", True),
                show_entry_rsi=strategy_params.get("showRSI", True),
                rr_mult_dual=float(strategy_params.get("tp_dual_rr_mult", 1.3)),
                rr_mult_rsi=float(strategy_params.get("tp_rsi_rr_mult", 1.3)),
                fee_rate=fee_rate,
                slippage=slip,
                initial_equity=initial_equity,
                order_type=order_type_num,
                order_value=order_value,
                pyramiding=pyramiding,
                compound=compound,
            )

            # Convert results to trade list format
            trades = []
            time_arr = price_arrays["time"]
            for j in range(len(entry_bars)):
                entry_bar = entry_bars[j]
                exit_bar = exit_bars[j]
                entry_time = to_hcm_time(str(time_arr[entry_bar]))
                exit_time = to_hcm_time(str(time_arr[exit_bar]))
                entry_price = entry_prices[j]
                exit_price = exit_prices[j]
                size = sizes[j]
                pnl = pnls[j]
                notional = entry_price * size
                pnl_pct = (pnl / notional) * 100 if notional else 0.0
                entry_type_str = "dual_flip" if entry_types[j] == 0 else "rsi"

                entry_fee = entry_price * fee_rate * size
                exit_fee = exit_price * fee_rate * size

                trades.append({
                    "side": "Long",
                    "entry_type": entry_type_str,
                    "entry_time": entry_time,
                    "entry_price": round(entry_price, 6),
                    "entry_fee": round(entry_fee, 6),
                    "exit_time": exit_time,
                    "exit_price": round(exit_price, 6),
                    "exit_fee": round(exit_fee, 6),
                    "size": size,
                    "notional": round(notional, 6),
                    "pnl": round(pnl, 6),
                    "pnl_pct": round(pnl_pct, 6),
                })

            # Build equity curve (only for execution range)
            equity_curve = []
            for i in range(execution_start_idx, len(df)):
                time_hcm = to_hcm_time(str(time_arr[i]))
                equity_curve.append({
                    "time": time_hcm,
                    "equity": round(equity_arr[i], 6)
                })

            # Calculate metrics
            summary = calculate_metrics(
                trades=trades,
                equity_curve=equity_curve,
                initial_equity=initial_equity
            )

            return {
                "meta": {
                    "strategyId": f"{strategy_type}_v1",
                    "strategyName": strategy_type,
                    "symbol": symbol,
                    "timeframe": timeframe
                },
                "summary": summary,
                "equityCurve": equity_curve,
                "trades": trades
            }
        except Exception as e:
            # Fall back to regular backtest on error
            pass

    # ===== REGULAR BACKTEST PATH (fallback) =====
    # ===== ACCOUNT =====
    equity = initial_equity

    open_trades = []
    trades = []
    equity_curve = []
    pending_entry = False  # Flag: có tín hiệu entry từ nến trước

    # Loop through ALL bars - indicators need full history context
    # But only EXECUTE trades and RECORD equity within execution range
    for i in range(len(df)):
        row = df.iloc[i]
        time = str(row["time"])
        time_hcm = to_hcm_time(time)
        price_close = float(row["close"])
        price_open = float(row["open"])

        # Check if we're in the execution range
        in_execution_range = (i >= execution_start_idx)

        # Reset pending_entry when entering execution range for the first time
        # This ensures signals from before the range are NOT executed
        if i == execution_start_idx:
            pending_entry = False

        # ===== EXECUTE PENDING ENTRY (at Open of current bar) =====
        # Entry tại Open của nến hiện tại nếu có tín hiệu từ nến trước
        # Only execute if we're in the execution range
        if pending_entry and len(open_trades) < pyramiding and in_execution_range:
            slip = slippage_ticks * tick_size
            entry_price = price_open + slip
            # Tính size theo Default order size, tránh over-exposure khi còn lệnh mở
            current_exposure = sum(t["entry_price"] * t["size"] for t in open_trades)
            base_equity = (equity - current_exposure) if compound else max(initial_equity - current_exposure, 0)
            position_size = order_value
            if order_type == "percent":
                position_size = max((order_value / 100.0) * base_equity / entry_price, 0)
            elif order_type in ["usdt", "fixed"]:
                position_size = max(order_value / entry_price, 0)
            entry_fee = entry_price * fee_rate * position_size
            equity -= entry_fee

            # Get entry type from strategy if available
            entry_type = getattr(strategy, '_entry_type', 'Long')

            new_trade = {
                "side": "Long",
                "entry_type": entry_type,  # Track entry type for exit logic
                "entry_time": time_hcm,
                "entry_price": round(entry_price, 6),
                "entry_fee": round(entry_fee, 6),
                "size": position_size,
                "notional": round(entry_price * position_size, 6),
            }
            open_trades.append(new_trade)

        pending_entry = False  # Reset flag regardless of execution

        # ===== EXIT (at Close of current bar) =====
        # Exits can happen anytime we have open trades (even if entry was before range)
        if open_trades and strategy.check_exit(i, open_trades[-1]):
            # đóng lần lượt theo FIFO để tách trade riêng
            trade_to_close = open_trades.pop(0)
            slip = slippage_ticks * tick_size
            exit_price = price_close - slip
            pos_size = trade_to_close.get("size", position_size)
            gross_pnl = (exit_price - trade_to_close["entry_price"]) * pos_size
            exit_fee = exit_price * fee_rate * pos_size
            pnl = gross_pnl - exit_fee
            notional = trade_to_close["entry_price"] * pos_size
            pnl_pct = (pnl / notional) * 100 if notional else 0.0

            equity += pnl

            trade_to_close.update({
                "exit_time": time_hcm,
                "exit_price": round(exit_price, 6),
                "exit_fee": round(exit_fee, 6),
                "pnl": round(pnl, 6),
                "pnl_pct": round(pnl_pct, 6),
                "notional": round(notional, 6),
                "size": pos_size,
            })

            trades.append(trade_to_close)

        # ===== CHECK ENTRY SIGNAL (will execute at Open of next bar) =====
        # IMPORTANT: Only check entry signals if we're in the execution range
        # This matches TradingView behavior - signals from before range are NOT considered
        # Indicators are still calculated on full history for proper warm-up
        if in_execution_range and strategy.check_entry(i) and len(open_trades) < pyramiding and not pending_entry:
            pending_entry = True

        # Only record equity curve for bars in execution range
        if in_execution_range:
            equity_curve.append({
                "time": time_hcm,
                "equity": round(equity, 6)
            })

    # Đóng vị thế còn mở tại giá cuối cùng (nếu có)
    if open_trades:
        slip = slippage_ticks * tick_size
        exit_price = price_close - slip
        while open_trades:
            trade_to_close = open_trades.pop(0)
            pos_size = trade_to_close.get("size", position_size)
            gross_pnl = (exit_price - trade_to_close["entry_price"]) * pos_size
            exit_fee = exit_price * fee_rate * pos_size
            pnl = gross_pnl - exit_fee
            notional = trade_to_close["entry_price"] * pos_size
            pnl_pct = (pnl / notional) * 100 if notional else 0.0

            equity += pnl

            trade_to_close.update({
                "exit_time": time_hcm,
                "exit_price": round(exit_price, 6),
                "exit_fee": round(exit_fee, 6),
                "pnl": round(pnl, 6),
                "pnl_pct": round(pnl_pct, 6),
                "notional": round(notional, 6),
                "size": pos_size,
            })

            trades.append(trade_to_close)

    # ===== METRICS =====
    summary = calculate_metrics(
        trades=trades,
        equity_curve=equity_curve,
        initial_equity=initial_equity
    )

    return {
        "meta": {
            "strategyId": f"{strategy_type}_v1",
            "strategyName": strategy_type,
            "symbol": symbol,
            "timeframe": timeframe
        },
        "summary": summary,
        "equityCurve": equity_curve,
        "trades": trades
    }


if __name__ == "__main__":
    raw = sys.stdin.read()
    strategy_config = json.loads(raw)
    result = run_backtest(strategy_config)
    print(json.dumps(result))
