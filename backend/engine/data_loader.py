# engine/data_loader.py
import os
import logging
import pandas as pd
from functools import lru_cache
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# Increase cache size for better performance
DATA_CACHE_SIZE = int(os.getenv("DATA_CACHE_SIZE", 64))

# Preloaded data storage
_preloaded_data = {}
_preload_enabled = os.getenv("DATA_PRELOAD", "true").lower() == "true"


def _safe_join(base_dir: str, *paths: str) -> str:
    """
    Safe join helper to avoid path traversal outside base_dir.
    """
    candidate = os.path.abspath(os.path.normpath(os.path.join(base_dir, *paths)))
    if not candidate.startswith(os.path.abspath(base_dir)):
        raise ValueError("Invalid symbol/timeframe path")
    return candidate


def get_available_data() -> List[Tuple[str, str]]:
    """
    Scan data directory and return list of available (symbol, timeframe) pairs.

    Returns:
        List of (symbol, timeframe) tuples
    """
    data_dir = os.path.abspath(os.path.join("engine", "data"))
    available = []

    if not os.path.exists(data_dir):
        return available

    # Check for new structure: data/{symbol}/{timeframe}.csv
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            symbol = item
            for file in os.listdir(item_path):
                if file.endswith(".csv"):
                    timeframe = file[:-4]  # Remove .csv
                    available.append((symbol, timeframe))

    # Check for old structure: data/{symbol}_{timeframe}.csv
    for file in os.listdir(data_dir):
        if file.endswith(".csv") and "_" in file:
            parts = file[:-4].rsplit("_", 1)
            if len(parts) == 2:
                symbol, timeframe = parts
                if (symbol, timeframe) not in available:
                    available.append((symbol, timeframe))

    return available


def preload_all_data() -> int:
    """
    Preload all available data into cache.

    Call this during worker startup to warm up the cache.

    Returns:
        Number of datasets preloaded
    """
    global _preloaded_data

    if not _preload_enabled:
        logger.info("Data preloading disabled")
        return 0

    available = get_available_data()
    loaded = 0

    for symbol, timeframe in available:
        try:
            key = (symbol, timeframe)
            if key not in _preloaded_data:
                df = load_csv(symbol, timeframe)
                _preloaded_data[key] = df
                loaded += 1
                logger.debug(f"Preloaded {symbol} {timeframe}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"Failed to preload {symbol} {timeframe}: {e}")

    logger.info(f"Preloaded {loaded} datasets into cache")
    return loaded


def get_preloaded(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    """
    Get preloaded data if available.

    Args:
        symbol: Trading pair symbol
        timeframe: Timeframe string

    Returns:
        DataFrame if preloaded, None otherwise
    """
    return _preloaded_data.get((symbol, timeframe))


def clear_preloaded_data():
    """Clear all preloaded data."""
    global _preloaded_data
    _preloaded_data.clear()
    logger.info("Cleared preloaded data cache")


def load_csv(symbol: str, timeframe: str):
    """
    Load CSV trong thư mục engine/data, ưu tiên cấu trúc mới:
    engine/data/{symbol}/{timeframe}.csv
    Sau đó fallback về dạng cũ: engine/data/{symbol}_{timeframe}.csv

    Returns a SHARED reference to cached DataFrame - DO NOT modify directly!
    """
    data_dir = os.path.abspath(os.path.join("engine", "data"))
    safe_symbol = symbol.replace("/", "_")
    safe_timeframe = timeframe.replace("/", "_")

    # Candidate 1: thư mục con theo symbol
    candidate_new = _safe_join(data_dir, safe_symbol, f"{safe_timeframe}.csv")
    # Candidate 2: file phẳng kiểu cũ
    candidate_old = _safe_join(data_dir, f"{safe_symbol}_{safe_timeframe}.csv")

    filename = None
    for path in (candidate_new, candidate_old):
        if os.path.exists(path):
            filename = path
            break

    if filename is None:
        raise FileNotFoundError(
            f"Data file not found for {symbol} {timeframe}. "
            f"Tried: {candidate_new} and {candidate_old}"
        )

    # Use processed cache to avoid repeated time conversion
    return _load_and_process_cached(filename)


@lru_cache(maxsize=DATA_CACHE_SIZE)
def _load_csv_cached(path: str) -> pd.DataFrame:
    """Load CSV file with LRU caching (raw, no processing)."""
    return pd.read_csv(path)


@lru_cache(maxsize=DATA_CACHE_SIZE)
def _load_and_process_cached(path: str) -> pd.DataFrame:
    """
    Load and process CSV with time conversion - fully cached.
    Returns the same DataFrame reference for repeated calls.
    """
    df = pd.read_csv(path)

    # Chuẩn hóa cột time: Binance dùng ms, nhưng nếu lẫn microseconds ( >1e13 ) thì chia 1000.
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    micro_mask = df["time"] > 1e13
    if micro_mask.any():
        df.loc[micro_mask, "time"] = (df.loc[micro_mask, "time"] // 1000)

    # Giới hạn khoảng thời gian hợp lý (1970 -> 2100)
    min_ms = 0  # 1970-01-01
    max_ms = 4102444800000  # ~2100-01-01 in ms
    df = df[(df["time"] >= min_ms) & (df["time"] <= max_ms)]

    # Convert to datetime - this is now cached!
    df["time"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
    df = df.dropna(subset=["time"])
    return df.sort_values("time").reset_index(drop=True)


def align_htf(df_ltf, df_htf):
    """
    Align HTF to LTF giống TradingView (no repaint)
    """
    df_htf = df_htf.copy()
    df_htf = df_htf.set_index("time")

    htf_aligned = []

    for t in df_ltf["time"]:
        past_htf = df_htf[df_htf.index <= t]
        if len(past_htf) == 0:
            htf_aligned.append(None)
        else:
            htf_aligned.append(past_htf.iloc[-1])

    return pd.DataFrame(htf_aligned)
