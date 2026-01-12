#!/bin/bash
# Build script for Rust backtest engine

set -e

echo "Building Rust backtest engine..."

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "Error: Rust is not installed. Install from https://rustup.rs/"
    exit 1
fi

# Check if maturin is installed
if ! command -v maturin &> /dev/null; then
    echo "Installing maturin..."
    pip install maturin
fi

# Build the Rust module
cd rust_engine

echo "Building release version..."
maturin develop --release

echo ""
echo "Build complete! Rust backtest engine is now available."
echo ""
echo "Test with:"
echo "  python -c \"import backtest_engine; print('Rust engine loaded!')\""
