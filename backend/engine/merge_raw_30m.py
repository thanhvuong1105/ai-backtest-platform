import pandas as pd
from pathlib import Path

RAW_DIR = Path("engine/data/raw")
OUT_FILE = Path("engine/data/BTCUSDT_30m.csv")

def read_and_normalize(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Chuẩn hoá header Binance CM
    # Thường là: open_time,open,high,low,close,volume,...
    # Đổi về schema engine dùng: time,open,high,low,close,volume
    col_map = {
        "open_time": "time",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }

    # Nếu file đã có 'time' thì giữ, nếu có 'open_time' thì đổi
    if "open_time" in df.columns and "time" not in df.columns:
        df = df.rename(columns=col_map)

    # Chỉ giữ các cột cần thiết
    keep = ["time", "open", "high", "low", "close", "volume"]
    df = df[keep]

    # Chuẩn hoá time (ms → datetime nếu cần)
    df["time"] = pd.to_datetime(df["time"], errors="coerce", unit="ms")

    return df.dropna(subset=["time"])

def main():
    files = sorted(RAW_DIR.glob("BTCUSD_PERP-30m-*.csv"))
    if not files:
        raise RuntimeError("Không tìm thấy file raw CSV nào!")

    frames = [read_and_normalize(f) for f in files]
    df_all = pd.concat(frames, ignore_index=True)

    # Sort + drop duplicate time
    df_all = df_all.sort_values("time")
    df_all = df_all.drop_duplicates(subset=["time"], keep="last")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(OUT_FILE, index=False)
    print(f"OK ✅ Gộp xong: {OUT_FILE} | rows={len(df_all)}")

if __name__ == "__main__":
    main()
