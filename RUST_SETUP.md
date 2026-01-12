# Rust Backtest Engine Setup Guide

## Tổng quan

Rust engine cung cấp **50-100x speedup** so với Python engine gốc.

| Benchmark | Python | Rust | Speedup |
|-----------|--------|------|---------|
| 1 backtest (1h, 1 năm) | ~1-4s | ~10-50ms | 20-80x |
| 80 backtests (batch) | ~343s | ~3-10s | 30-100x |

## Cài đặt Rust

### Bước 1: Cài Rust toolchain

```bash
# macOS / Linux
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Sau khi cài, restart terminal hoặc chạy:
source $HOME/.cargo/env

# Verify
rustc --version
cargo --version
```

### Bước 2: Cài maturin (Python-Rust bridge)

```bash
pip install maturin
```

### Bước 3: Build Rust engine

```bash
# Từ thư mục gốc của project
./build_rust.sh

# Hoặc build thủ công:
cd rust_engine
maturin develop --release
```

### Bước 4: Test

```bash
# Quick test
python -c "import backtest_engine; print('Rust engine loaded!')"

# Full test suite
python test_rust_engine.py
```

## Cách sử dụng

### Automatic (Recommended)

Rust engine sẽ tự động được sử dụng nếu có:

```python
from engine.rust_bridge import run_backtest_hybrid

result = run_backtest_hybrid(config)  # Tự động dùng Rust nếu có
```

### Manual

```python
from engine.rust_bridge import run_backtest_rust, is_rust_available

if is_rust_available():
    result = run_backtest_rust(config, df)
else:
    # Fallback to Python
    from engine.backtest_engine import run_backtest
    result = run_backtest(config)
```

### Batch Processing (Parallel)

```python
from engine.rust_bridge import run_batch_backtests_rust

# Chạy song song nhiều backtests
results = run_batch_backtests_rust(configs_list, df)
```

## Cấu trúc files

```
rust_engine/
├── Cargo.toml          # Rust dependencies
├── pyproject.toml      # Python package config
└── src/
    ├── lib.rs          # Python bindings (PyO3)
    ├── indicators.rs   # EMA, RSI, ATR, SuperTrend, RangeFilter
    ├── strategies.rs   # EMA Cross, RF+ST+RSI (future)
    └── engine.rs       # Core backtest loop
```

## Strategies được hỗ trợ

| Strategy | Python | Rust |
|----------|--------|------|
| EMA Cross | ✅ | ✅ |
| RF + ST + RSI | ✅ | 🔄 (planned) |

## Troubleshooting

### Build error: "linker not found"

```bash
# macOS
xcode-select --install
```

### Import error sau khi build

```bash
# Rebuild
cd rust_engine
maturin develop --release
```

### Performance không như mong đợi

- Đảm bảo build với `--release` flag
- Check CPU usage - Rust sử dụng tất cả cores với rayon

## Benchmark tự chạy

```bash
python test_rust_engine.py
```

## So sánh kết quả

Test script sẽ so sánh:
- Total trades
- Win rate
- Profit factor
- Max drawdown
- Net profit

Kết quả phải match chính xác (sai số < 0.01) giữa Python và Rust.
