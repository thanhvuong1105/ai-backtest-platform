#!/usr/bin/env python3
"""
Download OHLCV data from Binance API.

Usage:
    python download_data.py BTCUSDT 30m
    python download_data.py BTCUSDT 1h --days 365
    python download_data.py --all  # Download all common pairs/timeframes
"""

import os
import sys
import argparse
import time
from datetime import datetime, timedelta
import requests
import pandas as pd

# Binance API endpoint
BINANCE_API = "https://api.binance.com/api/v3/klines"

# Common symbols and timeframes
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
DEFAULT_TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h", "1d"]

# Data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def download_klines(symbol: str, timeframe: str, days: int = 365) -> pd.DataFrame:
    """
    Download OHLCV data from Binance.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        timeframe: Kline interval (e.g., '30m', '1h')
        days: Number of days to download

    Returns:
        DataFrame with OHLCV data
    """
    print(f"Downloading {symbol} {timeframe} for {days} days...")

    # Calculate start time
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    all_data = []
    current_start = start_time

    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": timeframe,
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000
        }

        try:
            response = requests.get(BINANCE_API, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data:
                break

            all_data.extend(data)

            # Move start time to after last candle
            current_start = data[-1][0] + 1

            # Rate limiting
            time.sleep(0.1)

            print(f"  Downloaded {len(all_data)} candles...", end="\r")

        except Exception as e:
            print(f"\nError downloading {symbol} {timeframe}: {e}")
            break

    if not all_data:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        "time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])

    # Keep only OHLCV columns
    df = df[["time", "open", "high", "low", "close", "volume"]]

    # Convert types
    df["time"] = df["time"].astype(int)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # Remove duplicates
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

    print(f"\n  Downloaded {len(df)} candles for {symbol} {timeframe}")
    return df


def save_data(df: pd.DataFrame, symbol: str, timeframe: str):
    """Save DataFrame to CSV."""
    if df.empty:
        print(f"  No data to save for {symbol} {timeframe}")
        return

    # Create directory structure
    symbol_dir = os.path.join(DATA_DIR, symbol)
    os.makedirs(symbol_dir, exist_ok=True)

    # Save file
    filepath = os.path.join(symbol_dir, f"{timeframe}.csv")
    df.to_csv(filepath, index=False)
    print(f"  Saved to {filepath}")


def download_all(days: int = 365):
    """Download all common pairs/timeframes."""
    for symbol in DEFAULT_SYMBOLS:
        for timeframe in DEFAULT_TIMEFRAMES:
            df = download_klines(symbol, timeframe, days)
            save_data(df, symbol, timeframe)
            time.sleep(0.5)  # Rate limiting between pairs


def main():
    parser = argparse.ArgumentParser(description="Download OHLCV data from Binance")
    parser.add_argument("symbol", nargs="?", help="Trading pair (e.g., BTCUSDT)")
    parser.add_argument("timeframe", nargs="?", help="Timeframe (e.g., 30m, 1h)")
    parser.add_argument("--days", type=int, default=365, help="Days of history (default: 365)")
    parser.add_argument("--all", action="store_true", help="Download all common pairs/timeframes")

    args = parser.parse_args()

    if args.all:
        download_all(args.days)
    elif args.symbol and args.timeframe:
        df = download_klines(args.symbol, args.timeframe, args.days)
        save_data(df, args.symbol, args.timeframe)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python download_data.py BTCUSDT 30m")
        print("  python download_data.py BTCUSDT 1h --days 365")
        print("  python download_data.py --all")


if __name__ == "__main__":
    main()
