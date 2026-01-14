"""
Pytest configuration and fixtures for backend tests.
"""

import pytest
from fastapi.testclient import TestClient

# Add backend to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_pine_code():
    """Sample Pine Script code for testing."""
    return """
    //@version=5
    strategy("EMA Cross", overlay=true)
    emaFast = ta.ema(close, 12)
    emaSlow = ta.ema(close, 26)
    longCondition = ta.crossover(emaFast, emaSlow)
    shortCondition = ta.crossunder(emaFast, emaSlow)
    if (longCondition)
        strategy.entry("Long", strategy.long)
    if (shortCondition)
        strategy.close("Long")
    """


@pytest.fixture
def sample_optimize_config():
    """Sample optimization configuration for testing."""
    return {
        "symbols": ["BTCUSDT"],
        "timeframes": ["1h"],
        "strategy": {
            "type": "ema_cross",
            "params": {
                "emaFast": [8, 12, 16],
                "emaSlow": [21, 26, 30]
            }
        },
        "filters": {
            "minPF": 1.0,
            "minTrades": 10,
            "maxDD": 50
        },
        "topN": 10
    }


@pytest.fixture
def sample_backtest_config():
    """Sample backtest configuration for testing."""
    return {
        "meta": {
            "symbols": ["BTCUSDT"],
            "timeframe": "1h"
        },
        "strategy": {
            "type": "ema_cross",
            "params": {
                "emaFast": 12,
                "emaSlow": 26
            }
        },
        "initial_equity": 10000,
        "properties": {
            "orderSize": {
                "type": "percent",
                "value": 100
            },
            "pyramiding": 1
        },
        "costs": {
            "fee": 0.0004,
            "slippage": 0
        }
    }


@pytest.fixture
def sample_chart_config():
    """Sample chart data configuration for testing."""
    return {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "strategy": {
            "type": "ema_cross",
            "params": {
                "emaFast": 12,
                "emaSlow": 26
            }
        },
        "capital": {
            "initial": 10000,
            "orderPct": 100
        },
        "risk": {
            "pyramiding": 1,
            "commission": 0.04,
            "slippage": 0
        }
    }
