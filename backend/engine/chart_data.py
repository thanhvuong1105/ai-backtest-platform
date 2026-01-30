# engine/chart_data.py
"""
Generates chart data for debug/verify view:
- OHLCV candles
- Indicator values (EMA, RSI, etc.)
- Entry/Exit signals from backtest

ARCHITECTURE PRINCIPLE:
- Chart Signal Timeline: ALWAYS shows FULL history (from symbol start to now)
- Trade List: FILTERED by user's range selection
- Range selector only affects Trade List metrics, NOT chart display
"""
import sys
import json
import numpy as np
import pandas as pd
from datetime import timezone, timedelta

# Handle both module import and script execution
if __name__ == "__main__":
    # Running as script via subprocess - use absolute imports
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from engine.data_loader import load_csv
    from engine.indicators import ema, rsi
    from engine.strategies.factory import create_strategy
    from engine.backtest_engine import run_backtest
else:
    # Imported as module - use relative imports
    from .data_loader import load_csv
    from .indicators import ema, rsi
    from .strategies.factory import create_strategy
    from .backtest_engine import run_backtest

# UTC+7 Ho Chi Minh timezone
TZ_HCM = timezone(timedelta(hours=7))


def to_hcm_timestamp(dt):
    """Convert datetime to UTC+7 timestamp"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_hcm = dt.astimezone(TZ_HCM)
    return int(dt_hcm.timestamp())


def filter_trades_by_range(trades, start_date, end_date):
    """
    Filter trades by date range.
    A trade is included if its entry_time falls within the range.
    """
    if not trades:
        return []

    filtered = []
    for trade in trades:
        entry_time_str = trade.get("entry_time")
        if not entry_time_str:
            continue

        entry_time = pd.to_datetime(entry_time_str)

        # Check if within range
        in_range = True
        if start_date is not None and entry_time < start_date:
            in_range = False
        if end_date is not None and entry_time > end_date:
            in_range = False

        if in_range:
            filtered.append(trade)

    return filtered


def calculate_summary_from_trades(trades, initial_equity):
    """
    Calculate summary statistics from a list of trades.
    """
    if not trades:
        return {
            "totalTrades": 0,
            "winrate": 0,
            "profitFactor": 0,
            "netProfitPct": 0,
            "maxDrawdownPct": 0,
            "finalEquity": initial_equity,
        }

    wins = 0
    losses = 0
    gross_profit = 0
    gross_loss = 0
    total_pnl = 0

    equity = initial_equity
    peak_equity = initial_equity
    max_drawdown = 0

    for trade in trades:
        pnl = trade.get("pnl", 0)
        total_pnl += pnl
        equity += pnl

        if pnl > 0:
            wins += 1
            gross_profit += pnl
        else:
            losses += 1
            gross_loss += abs(pnl)

        # Track drawdown
        if equity > peak_equity:
            peak_equity = equity
        drawdown = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    total_trades = wins + losses
    winrate = (wins / total_trades * 100) if total_trades > 0 else 0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999 if gross_profit > 0 else 0)
    net_profit_pct = (total_pnl / initial_equity * 100) if initial_equity > 0 else 0

    return {
        "totalTrades": total_trades,
        "winrate": winrate,
        "profitFactor": profit_factor,
        "netProfitPct": net_profit_pct,
        "maxDrawdownPct": max_drawdown,
        "finalEquity": equity,
    }


def get_chart_data(config: dict) -> dict:
    """
    Returns chart data for visualization.

    IMPORTANT: Chart always shows FULL signal history.
    Range only affects Trade List filtering.

    Config:
    {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "strategy": {
            "type": "ema_cross",
            "params": {"emaFast": 12, "emaSlow": 26}
        },
        "capital": {"initial": 10000, "orderPct": 100},
        "risk": {"pyramiding": 1, "commission": 0.04, "slippage": 0.01},
        "dateRange": {"start": null, "end": null}
    }
    """
    symbol = config["symbol"]
    timeframe = config["timeframe"]
    strategy_cfg = config["strategy"]
    capital = config.get("capital", {"initial": 10000, "orderPct": 100})
    risk = config.get("risk", {"pyramiding": 1, "commission": 0.0, "slippage": 0.0})
    initial_equity = capital.get("initial", 10000)

    # Support both 'range' (from frontend) and 'dateRange' (legacy)
    date_range = config.get("range") or config.get("dateRange") or {}

    # ===== LOAD FULL DATA =====
    # Always load ALL data from symbol start (e.g., 2017-08-17 for BTCUSDT)
    # Range only affects trade filtering, NOT chart display or indicator calculation
    df_all = load_csv(symbol, timeframe)
    df_all["time"] = pd.to_datetime(df_all["time"])

    # Parse date range for TRADE FILTERING ONLY
    start_str = date_range.get("from") or date_range.get("start")
    end_str = date_range.get("to") or date_range.get("end")
    filter_start_date = pd.to_datetime(start_str) if start_str else None
    filter_end_date = pd.to_datetime(end_str) if end_str else None

    # Use ALL data for chart display - no end_date filtering
    df_full = df_all.reset_index(drop=True)

    # ===== CALCULATE INDICATORS ON FULL DATA =====
    # Indicators are calculated on COMPLETE history for accurate values
    strategy = create_strategy(strategy_cfg["type"], df_full.copy(), strategy_cfg["params"])
    strategy.prepare_indicators()

    # Generate flip events for debug visualization (if supported by strategy)
    if hasattr(strategy, 'generate_flip_events'):
        strategy.generate_flip_events()

    df_with_indicators = strategy.df

    # ===== RUN BACKTEST ON FULL DATA (NO RANGE FILTER) =====
    # Get ALL trades from symbol start to generate FULL markers
    backtest_config = {
        "meta": {
            "symbols": [symbol],
            "timeframe": timeframe
        },
        "strategy": strategy_cfg,
        "initial_equity": initial_equity,
        "properties": {
            "orderSize": {
                "type": "percent",
                "value": capital.get("orderPct", 100)
            },
            "pyramiding": risk.get("pyramiding", 1),
            "slippage": risk.get("slippage", 0.0)
        },
        "costs": {
            "fee": risk.get("commission", 0.0) / 100,
            "slippage": risk.get("slippage", 0.0)
        },
        "range": {
            # Chạy backtest FULL history để chart/markers luôn đủ dữ liệu
            "from": None,
            "to": None
        },
        "df": df_full
    }

    # Run backtest to get ALL trades (for chart markers)
    backtest_result = run_backtest(backtest_config)
    all_trades = backtest_result.get("trades", [])

    # ===== FILTER TRADES FOR TRADE LIST (entry within range) =====
    filtered_trades = filter_trades_by_range(all_trades, filter_start_date, filter_end_date)

    # ===== RECOMPUTE EQUITY FROM RANGE START WITH INITIAL CAPITAL =====
    fee_rate = risk.get("commission", 0.04) / 100.0
    equity = initial_equity
    eq_curve = []
    recomputed_trades = []
    for t in filtered_trades:
        entry_price = t.get("entry_price")
        exit_price = t.get("exit_price")
        if not entry_price or not exit_price:
            continue
        size = equity * (capital.get("orderPct", 100) / 100.0) / entry_price
        notional = entry_price * size
        # Recompute fee trên notional hiện tại
        entry_fee = entry_price * fee_rate * size
        exit_fee = exit_price * fee_rate * size
        fee = entry_fee + exit_fee

        # Calculate PnL based on side - Long vs Short
        trade_side = t.get("side", "Long").lower()
        if trade_side == "short":
            # Short: profit when price goes DOWN (entry > exit)
            pnl_usdt = (entry_price - exit_price) * size - fee
        else:
            # Long: profit when price goes UP (exit > entry)
            pnl_usdt = (exit_price - entry_price) * size - fee

        pnl_pct = (pnl_usdt / notional) * 100 if notional else 0.0
        equity += pnl_usdt
        eq_curve.append(equity)
        recomputed_trades.append({
            **t,
            "size": size,
            "notional": notional,
            "pnl": pnl_usdt,
            "pnl_pct": pnl_pct,
            "fee": fee,
        })

    # Calculate summary from recomputed trades only
    summary = calculate_summary_from_trades(recomputed_trades, initial_equity)

    # ===== OPTIMIZED: Build candles and indicators using vectorized operations =====
    # Pre-compute timestamps once (vectorized)
    timestamps = df_with_indicators["time"].apply(to_hcm_timestamp).values
    n = len(timestamps)

    # Build candles array
    candles = []
    opens = df_with_indicators["open"].values
    highs = df_with_indicators["high"].values
    lows = df_with_indicators["low"].values
    closes = df_with_indicators["close"].values
    has_volume = "volume" in df_with_indicators.columns
    volumes = df_with_indicators["volume"].values if has_volume else None

    for i in range(n):
        candle = {
            "time": int(timestamps[i]),
            "open": float(opens[i]),
            "high": float(highs[i]),
            "low": float(lows[i]),
            "close": float(closes[i]),
        }
        if has_volume:
            candle["volume"] = float(volumes[i])
        candles.append(candle)

    # Build indicators array using vectorized operations
    indicators = {}

    def build_simple_indicator(col_name):
        """Build indicator from single column using vectorized ops"""
        if col_name not in df_with_indicators.columns:
            return None
        vals = df_with_indicators[col_name].values
        result = []
        for i in range(n):
            v = vals[i]
            result.append({"time": int(timestamps[i]), "value": float(v) if pd.notna(v) else None})
        return result

    def build_supertrend_indicator(trend_col, up_col, dn_col):
        """Build SuperTrend up/dn lines based on trend direction"""
        if trend_col not in df_with_indicators.columns:
            return None, None
        trends = df_with_indicators[trend_col].values
        ups = df_with_indicators[up_col].values
        dns = df_with_indicators[dn_col].values
        up_line = []
        dn_line = []
        for i in range(n):
            t = int(timestamps[i])
            trend = trends[i]
            if pd.notna(trend) and trend == 1:
                up_line.append({"time": t, "value": float(ups[i]) if pd.notna(ups[i]) else None})
                dn_line.append({"time": t, "value": None})
            elif pd.notna(trend) and trend == -1:
                up_line.append({"time": t, "value": None})
                dn_line.append({"time": t, "value": float(dns[i]) if pd.notna(dns[i]) else None})
            else:
                up_line.append({"time": t, "value": None})
                dn_line.append({"time": t, "value": None})
        return up_line, dn_line

    # EMA indicators (for ema_cross strategy)
    for col in ["emaFast", "emaSlow"]:
        data = build_simple_indicator(col)
        if data:
            indicators[col] = data

    # Range Filter Entry indicators
    for col in ["rf_f", "rf_hb", "rf_lb"]:
        data = build_simple_indicator(col)
        if data:
            indicators[col] = data

    # SuperTrend Entry indicator
    if "st_trend" in df_with_indicators.columns:
        up_line, dn_line = build_supertrend_indicator("st_trend", "st_up", "st_dn")
        if up_line:
            indicators["st_up"] = up_line
            indicators["st_dn"] = dn_line

    # ===== SL/TP Indicators (for rf_st_rsi_combined strategy) =====
    # Only export Short-side indicators to reduce response size

    # Range Filter SL Short only
    for col in ["rf_sl_S_filt", "rf_sl_S_hband", "rf_sl_S_lband"]:
        data = build_simple_indicator(col)
        if data:
            indicators[col] = data

    # SuperTrend SL Short only
    if "st_sl_S_trend" in df_with_indicators.columns:
        up_line, dn_line = build_supertrend_indicator("st_sl_S_trend", "st_sl_S_up", "st_sl_S_dn")
        if up_line:
            indicators["st_sl_S_up"] = up_line
            indicators["st_sl_S_dn"] = dn_line

    # SuperTrend TP RSI Short only
    if "st_tp_rsi_S_trend" in df_with_indicators.columns:
        up_line, dn_line = build_supertrend_indicator("st_tp_rsi_S_trend", "st_tp_rsi_S_up", "st_tp_rsi_S_dn")
        if up_line:
            indicators["st_tp_rsi_S_up"] = up_line
            indicators["st_tp_rsi_S_dn"] = dn_line

    # ===== BUILD MARKERS FROM ALL TRADES (FULL HISTORY) =====
    # Chart always shows ALL entry/exit markers regardless of range
    # Labels match TradingView: Long, Short, RSI Long, RSI Short, exit
    markers = []
    for i, trade in enumerate(all_trades):
        # Get trade side and entry type
        trade_side = trade.get("side", "Long")
        entry_type = trade.get("entry_type", "dual_flip_long")
        is_short = trade_side.lower() == "short"
        is_rsi = "rsi" in str(entry_type).lower()

        # Determine entry label based on entry_type (matching TradingView)
        if is_rsi:
            entry_label = "RSI Short" if is_short else "RSI Long"
            entry_color = "#E040FB" if is_short else "#00BCD4"  # magenta for RSI Short, cyan for RSI Long
        else:
            entry_label = "Short" if is_short else "Long"
            entry_color = "#FF6D00" if is_short else "#2962FF"  # orange for Short, blue for Long

        # Entry marker
        entry_time = trade.get("entry_time")
        if entry_time:
            dt = pd.to_datetime(entry_time)
            dt_hcm = dt.replace(tzinfo=TZ_HCM)
            entry_ts = int(dt_hcm.timestamp())

            if is_short:
                # Short entry: arrow down from above (sell)
                markers.append({
                    "time": entry_ts,
                    "position": "aboveBar",
                    "color": entry_color,
                    "shape": "arrowDown",
                    "text": entry_label
                })
            else:
                # Long entry: arrow up from below (buy)
                markers.append({
                    "time": entry_ts,
                    "position": "belowBar",
                    "color": entry_color,
                    "shape": "arrowUp",
                    "text": entry_label
                })

        # Exit marker - always show "exit" like TradingView
        exit_time = trade.get("exit_time")
        if exit_time:
            dt = pd.to_datetime(exit_time)
            dt_hcm = dt.replace(tzinfo=TZ_HCM)
            exit_ts = int(dt_hcm.timestamp())

            # Exit color: magenta/pink like TradingView
            exit_color = "#E040FB"
            exit_label = "exit"

            if is_short:
                # Short exit: arrow up from below (buy to cover)
                markers.append({
                    "time": exit_ts,
                    "position": "belowBar",
                    "color": exit_color,
                    "shape": "arrowUp",
                    "text": exit_label
                })
            else:
                # Long exit: arrow down from above (sell)
                markers.append({
                    "time": exit_ts,
                    "position": "aboveBar",
                    "color": exit_color,
                    "shape": "arrowDown",
                    "text": exit_label
                })

    # Sort markers by time
    markers.sort(key=lambda m: m["time"])

    # ===== FORMAT FILTERED TRADES FOR TRADE LIST =====
    # Trade List only shows trades within user's selected range
    formatted_trades = []
    for i, trade in enumerate(recomputed_trades):
        pnl_usdt = trade.get("pnl", 0)
        entry_price = trade.get("entry_price", 0)
        exit_price = trade.get("exit_price", 0)
        size = trade.get("size")
        notional = trade.get("notional") or (entry_price * size if entry_price and size else 0)
        # Prefer backend-computed pnl_pct if available
        pnl_pct = trade.get("pnl_pct")
        if pnl_pct is None:
            if entry_price and size:
                pnl_pct = (pnl_usdt / (entry_price * size)) * 100
            elif entry_price:
                # Fallback: calculate based on side
                is_short = trade.get("side", "Long").lower() == "short"
                if is_short:
                    pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                else:
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = 0

        # Get entry type for display
        entry_type = trade.get("entry_type", "dual_flip_long")
        trade_side = trade.get("side", "LONG").upper()
        is_rsi = "rsi" in str(entry_type).lower()

        # Format entry_type label like TradingView
        if is_rsi:
            entry_type_label = f"RSI {trade_side.title()}"
        else:
            entry_type_label = trade_side.title()

        formatted_trades.append({
            "id": i + 1,
            "entry_time": trade.get("entry_time", ""),
            "exit_time": trade.get("exit_time", "Open"),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": size,
            "notional": notional,
            "side": trade_side,
            "entry_type": entry_type_label,
            "pnl": round(pnl_pct, 2) if pnl_pct is not None else None,
            "pnl_usdt": round(pnl_usdt, 2),
            "fee": round(trade.get("entry_fee", 0) + trade.get("exit_fee", 0), 2),
        })

    # ===== GET DEBUG EVENTS (for RF/ST flip visualization) =====
    debug_events = []
    if hasattr(strategy, 'get_debug_events'):
        raw_events = strategy.get_debug_events()
        # Filter to only include flip/signal events (not entry/exit from check_entry)
        # These have the new format with bar_index, event_type, marker_type, etc.
        for evt in raw_events:
            if isinstance(evt, dict) and "event_type" in evt:
                debug_events.append(evt)

    # ===== BUILD DEBUG MARKERS (Buy/Sell from DualFlip signals) =====
    debug_markers = []
    for evt in debug_events:
        if evt.get("marker_type") in ["BUY", "SELL"]:
            debug_markers.append({
                "bar_index": evt.get("bar_index"),
                "time": evt.get("time"),
                "price": evt.get("price"),
                "type": evt.get("marker_type"),  # "BUY" or "SELL"
                "tag": evt.get("tag"),  # "Long", "Short", "RSI Long", "RSI Short"
                "source": evt.get("source"),  # "DUALFLIP" or "RSI"
                "event_type": evt.get("event_type"),
            })

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy": strategy_cfg,
        "candles": candles,
        "indicators": indicators,
        "markers": markers,
        "trades": formatted_trades,
        "summary": summary,
        "debug": {
            "enabled": True,
            "events": debug_events,
            "markers": debug_markers,
        },
    }


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types"""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj) if not np.isnan(obj) else None
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


if __name__ == "__main__":
    # Read config from stdin
    raw = sys.stdin.read()
    config = json.loads(raw)

    result = get_chart_data(config)
    print(json.dumps(result, cls=NumpyEncoder))
