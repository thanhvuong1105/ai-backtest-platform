# engine/data_loader.py
import os
import pandas as pd
from functools import lru_cache


def _safe_join(base_dir: str, *paths: str) -> str:
    """
    Safe join helper to avoid path traversal outside base_dir.
    """
    candidate = os.path.abspath(os.path.normpath(os.path.join(base_dir, *paths)))
    if not candidate.startswith(os.path.abspath(base_dir)):
        raise ValueError("Invalid symbol/timeframe path")
    return candidate


def load_csv(symbol: str, timeframe: str):
    """
    Load CSV trong thư mục engine/data, ưu tiên cấu trúc mới:
    engine/data/{symbol}/{timeframe}.csv
    Sau đó fallback về dạng cũ: engine/data/{symbol}_{timeframe}.csv
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

    df = _load_csv_cached(filename)

    # Chuẩn hóa cột time: Binance dùng ms, nhưng nếu lẫn microseconds ( >1e13 ) thì chia 1000.
    # Đồng thời loại bỏ timestamp vượt quá bound để tránh OutOfBoundsDatetime.
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    micro_mask = df["time"] > 1e13
    if micro_mask.any():
        df.loc[micro_mask, "time"] = (df.loc[micro_mask, "time"] // 1000)

    # Giới hạn khoảng thời gian hợp lý (1970 -> 2100)
    min_ms = 0  # 1970-01-01
    max_ms = 4102444800000  # ~2100-01-01 in ms
    df = df[(df["time"] >= min_ms) & (df["time"] <= max_ms)]

    df["time"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
    df = df.dropna(subset=["time"])
    return df.sort_values("time").reset_index(drop=True)


@lru_cache(maxsize=32)
def _load_csv_cached(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


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
