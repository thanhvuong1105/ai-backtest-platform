# engine/backtest_engine.py
import json
import sys
from datetime import timezone, timedelta

from .data_loader import load_csv
from .metrics import calculate_metrics
from .strategies.factory import create_strategy
import pandas as pd

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
    fee_rate = strategy_config.get("costs", {}).get("fee", 0.0)
    slippage_ticks = strategy_config.get("costs", {}).get("slippage", 0.0)

    # ===== ACCOUNT CONFIG =====
    initial_equity = float(strategy_config.get("initial_equity", 10000.0))
    # order size config: percent of equity hoặc fixed USDT
    props = strategy_config.get("properties", {})
    order_cfg = props.get("orderSize", {})
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

    # Ensure time column is datetime
    if df is not None and not df.empty:
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"])

        # Filter data up to to_date for indicator calculation
        if to_date_ts is not None:
            df = df[df["time"] <= to_date_ts].reset_index(drop=True)

        # ===== PERFORMANCE OPTIMIZATION =====
        # Instead of using ALL historical data (which can be 40K+ candles),
        # we use a lookback buffer before from_date for indicator warmup.
        # This gives identical results but is MUCH faster.
        #
        # Lookback buffer: 500 bars is enough for most indicators:
        # - EMA 200: needs ~200 bars for convergence
        # - RSI 14: needs ~14 bars
        # - ATR 14: needs ~14 bars
        # - SuperTrend: needs ATR warmup (~50 bars)
        # - 500 bars provides 2.5x safety margin
        #
        INDICATOR_WARMUP_BARS = 500

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
