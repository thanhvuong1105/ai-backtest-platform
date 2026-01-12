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
import pandas as pd
from datetime import timezone, timedelta
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

    # Build candles array with UTC+7 timezone
    candles = []
    for _, row in df_with_indicators.iterrows():
        candle = {
            "time": to_hcm_timestamp(row["time"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }
        if "volume" in row:
            candle["volume"] = float(row["volume"])
        candles.append(candle)

    # Build indicators array
    indicators = {}

    # EMA indicators (for ema_cross strategy)
    if "emaFast" in df_with_indicators.columns:
        indicators["emaFast"] = [
            {"time": to_hcm_timestamp(row["time"]), "value": float(row["emaFast"]) if pd.notna(row["emaFast"]) else None}
            for _, row in df_with_indicators.iterrows()
        ]
    if "emaSlow" in df_with_indicators.columns:
        indicators["emaSlow"] = [
            {"time": to_hcm_timestamp(row["time"]), "value": float(row["emaSlow"]) if pd.notna(row["emaSlow"]) else None}
            for _, row in df_with_indicators.iterrows()
        ]

    # Range Filter indicator (for rf_st_rsi strategy) - full history, không reset
    if "rf_f" in df_with_indicators.columns:
        indicators["rf_f"] = [
            {"time": to_hcm_timestamp(row["time"]), "value": float(row["rf_f"]) if pd.notna(row["rf_f"]) else None}
            for _, row in df_with_indicators.iterrows()
        ]
    if "rf_hb" in df_with_indicators.columns:
        indicators["rf_hb"] = [
            {"time": to_hcm_timestamp(row["time"]), "value": float(row["rf_hb"]) if pd.notna(row["rf_hb"]) else None}
            for _, row in df_with_indicators.iterrows()
        ]
    if "rf_lb" in df_with_indicators.columns:
        indicators["rf_lb"] = [
            {"time": to_hcm_timestamp(row["time"]), "value": float(row["rf_lb"]) if pd.notna(row["rf_lb"]) else None}
            for _, row in df_with_indicators.iterrows()
        ]

    # SuperTrend indicator (for rf_st_rsi strategy) - display based on trend
    if "st_trend" in df_with_indicators.columns and "st_up" in df_with_indicators.columns and "st_dn" in df_with_indicators.columns:
        st_up_line = []
        st_dn_line = []
        for _, row in df_with_indicators.iterrows():
            t = to_hcm_timestamp(row["time"])
            if pd.notna(row["st_trend"]) and row["st_trend"] == 1:
                st_up_line.append({"time": t, "value": float(row["st_up"]) if pd.notna(row["st_up"]) else None})
                st_dn_line.append({"time": t, "value": None})
            elif pd.notna(row["st_trend"]) and row["st_trend"] == -1:
                st_up_line.append({"time": t, "value": None})
                st_dn_line.append({"time": t, "value": float(row["st_dn"]) if pd.notna(row["st_dn"]) else None})
            else:
                st_up_line.append({"time": t, "value": None})
                st_dn_line.append({"time": t, "value": None})
        indicators["st_up"] = st_up_line
        indicators["st_dn"] = st_dn_line

    # ===== BUILD MARKERS FROM ALL TRADES (FULL HISTORY) =====
    # Chart always shows ALL entry/exit markers regardless of range
    markers = []
    for i, trade in enumerate(all_trades):
        # Entry marker
        entry_time = trade.get("entry_time")
        if entry_time:
            # Parse UTC+7 string and convert to timestamp
            dt = pd.to_datetime(entry_time)
            # Since the string is already in UTC+7, we treat it as local time
            dt_hcm = dt.replace(tzinfo=TZ_HCM)
            entry_ts = int(dt_hcm.timestamp())
            markers.append({
                "time": entry_ts,
                "position": "belowBar",
                "color": "#2962FF",  # blue like TradingView
                "shape": "arrowUp",
                "text": "Long"
            })

        # Exit marker
        exit_time = trade.get("exit_time")
        if exit_time:
            # Parse UTC+7 string and convert to timestamp
            dt = pd.to_datetime(exit_time)
            dt_hcm = dt.replace(tzinfo=TZ_HCM)
            exit_ts = int(dt_hcm.timestamp())
            entry_price = trade.get("entry_price", 0)
            exit_price = trade.get("exit_price", 0)

            # Calculate PnL percentage for display
            if entry_price > 0:
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = 0

            # Determine if TP (profit) or SL (loss)
            if pnl_pct >= 0:
                color = "#26A69A"  # green/teal like TradingView
                text = f"TP {i}"
            else:
                color = "#EF5350"  # red like TradingView
                text = f"Exit {i}"

            markers.append({
                "time": exit_ts,
                "position": "aboveBar",
                "color": color,
                "shape": "arrowDown",
                "text": text
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
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = 0

        formatted_trades.append({
            "id": i + 1,
            "entry_time": trade.get("entry_time", ""),
            "exit_time": trade.get("exit_time", "Open"),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": size,
            "notional": notional,
            "side": "LONG",
            "pnl": round(pnl_pct, 2) if pnl_pct is not None else None,
            "pnl_usdt": round(pnl_usdt, 2),
            "fee": round(trade.get("entry_fee", 0) + trade.get("exit_fee", 0), 2),
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
    }


if __name__ == "__main__":
    # Read config from stdin
    raw = sys.stdin.read()
    config = json.loads(raw)

    result = get_chart_data(config)
    print(json.dumps(result))
