#!/usr/bin/env python3
"""
Download Binance Vision spot klines (daily zip) and merge into one CSV per symbol/timeframe.
No dependencies beyond the standard library.

Usage:
  python download_binance_data.py --symbol BTCUSDT --tf 30m --from 2017-08-01 --to 2025-01-01

Supports multiple symbols/timeframes (comma separated):
  python download_binance_data.py --symbol BTCUSDT,ETHUSDT --tf 30m,1h --from 2020-01-01 --to 2020-02-01

Output:
  data/{symbol}/{tf}.csv   # normalized columns: time,open,high,low,close,volume
"""

import argparse
import csv
import datetime as dt
import io
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://data.binance.vision/data/spot/daily/klines"


def daterange(start: dt.date, end: dt.date):
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)


def fetch_zip(url: str, retries: int = 3, backoff: float = 1.5) -> bytes | None:
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == retries:
                raise
        except Exception:
            if attempt == retries:
                raise
        time.sleep(backoff * attempt)
    return None


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_existing_times(out_file: Path) -> set[int]:
    if not out_file.exists():
        return set()
    seen = set()
    with out_file.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                seen.add(int(row["time"]))
            except Exception:
                continue
    return seen


def append_rows(out_file: Path, rows: list[dict], header: list[str]):
    file_exists = out_file.exists()
    ensure_dir(out_file.parent)
    with out_file.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def process_zip_content(content: bytes, symbol: str, tf: str) -> list[dict]:
    rows = []
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        name = next((n for n in zf.namelist() if n.endswith(".csv")), None)
        if not name:
            return rows
        with zf.open(name) as csvfile:
            reader = csv.reader(io.TextIOWrapper(csvfile, encoding="utf-8"))
            for rec in reader:
                # Kline format: [
                # 0 open time (ms), 1 open, 2 high, 3 low, 4 close, 5 volume,
                # 6 close time, 7 quote asset volume, 8 number of trades,
                # 9 taker buy base, 10 taker buy quote, 11 ignore
                # ]
                if len(rec) < 6:
                    continue
                try:
                    rows.append(
                        {
                            "time": int(rec[0]),
                            "open": float(rec[1]),
                            "high": float(rec[2]),
                            "low": float(rec[3]),
                            "close": float(rec[4]),
                            "volume": float(rec[5]),
                        }
                    )
                except Exception:
                    continue
    return rows


def download_range(symbol: str, tf: str, start: dt.date, end: dt.date, out_dir: Path):
    out_file = out_dir / symbol / f"{tf}.csv"
    seen = load_existing_times(out_file)
    header = ["time", "open", "high", "low", "close", "volume"]

    for day in daterange(start, end):
        fname = f"{symbol}-{tf}-{day.strftime('%Y-%m-%d')}.zip"
        url = f"{BASE_URL}/{symbol}/{tf}/{fname}"
        tag = f"[{symbol} {tf}] {day}"

        # skip if already present by time
        # (rough check: if start-of-day exists, assume day processed)
        # Use UTC timezone to match Binance timestamps
        day_start_utc = dt.datetime.combine(day, dt.time.min, tzinfo=dt.timezone.utc)
        day_start_ms = int(day_start_utc.timestamp() * 1000)
        if day_start_ms in seen:
            print(f"{tag} skip (already in output)")
            continue

        print(f"{tag} downloading {url}")
        content = fetch_zip(url)
        if content is None:
            print(f"{tag} not found (404), skipping")
            continue

        day_rows = process_zip_content(content, symbol, tf)
        if not day_rows:
            print(f"{tag} empty/invalid zip, skipping")
            continue

        # dedup against existing times
        new_rows = [r for r in day_rows if r["time"] not in seen]
        if not new_rows:
            print(f"{tag} no new rows after dedup, skipping append")
            continue

        append_rows(out_file, new_rows, header)
        seen.update(r["time"] for r in new_rows)
        print(f"{tag} ✓ appended {len(new_rows)} rows")


def parse_args():
    p = argparse.ArgumentParser(description="Download Binance Vision klines and merge CSV.")
    p.add_argument("--symbol", required=True, help="Symbol or comma-separated list, e.g., BTCUSDT,ETHUSDT")
    p.add_argument("--tf", required=True, help="Timeframe or comma-separated list, e.g., 30m,1h")
    p.add_argument("--from", dest="start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--to", dest="end", required=True, help="End date YYYY-MM-DD (inclusive)")
    p.add_argument("--out", dest="out_dir", default="data", help="Output base dir (default: data)")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        start = dt.datetime.strptime(args.start, "%Y-%m-%d").date()
        end = dt.datetime.strptime(args.end, "%Y-%m-%d").date()
    except ValueError:
        print("Invalid date format, use YYYY-MM-DD")
        sys.exit(1)
    if end < start:
        print("End date must be >= start date")
        sys.exit(1)

    symbols = [s.strip().upper() for s in args.symbol.split(",") if s.strip()]
    tfs = [t.strip() for t in args.tf.split(",") if t.strip()]
    out_dir = Path(args.out_dir)

    for sym in symbols:
        for tf in tfs:
            download_range(sym, tf, start, end, out_dir)


if __name__ == "__main__":
    main()
