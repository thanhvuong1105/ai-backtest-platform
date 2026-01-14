"""
Tests for engine package imports.

Verifies that the engine package can be imported and used directly
without subprocess spawning.
"""

import pytest


def test_engine_import():
    """Test that engine package can be imported."""
    from engine import (
        run_backtest,
        run_optimizer,
        run_ai_agent,
        generate_chart_data,
        load_data,
    )

    assert callable(run_backtest)
    assert callable(run_optimizer)
    assert callable(run_ai_agent)
    assert callable(generate_chart_data)
    assert callable(load_data)


def test_engine_submodule_imports():
    """Test that engine submodules can be imported."""
    from engine.backtest_engine import run_backtest
    from engine.optimizer import optimize
    from engine.ai_agent import ai_recommend
    from engine.chart_data import get_chart_data
    from engine.data_loader import load_csv

    assert callable(run_backtest)
    assert callable(optimize)
    assert callable(ai_recommend)
    assert callable(get_chart_data)
    assert callable(load_csv)


def test_strategy_imports():
    """Test that strategy modules can be imported."""
    from engine.strategies.factory import create_strategy
    from engine.strategies.ema_cross import EMACrossStrategy
    from engine.strategies.rf_st_rsi_strategy import RFSTRSIStrategy

    assert callable(create_strategy)
    assert EMACrossStrategy is not None
    assert RFSTRSIStrategy is not None


def test_helper_module_imports():
    """Test that helper modules can be imported."""
    from engine.indicators import ema, rsi
    from engine.metrics import calculate_metrics
    from engine.scoring import score_strategy
    from engine.guards import equity_smoothness
    from engine.filtering import filter_results
    from engine.stability import stability_metrics, pass_stability

    assert callable(ema)
    assert callable(rsi)
    assert callable(calculate_metrics)
    assert callable(score_strategy)
    assert callable(equity_smoothness)
    assert callable(filter_results)
    assert callable(stability_metrics)
    assert callable(pass_stability)


def test_run_backtest_basic():
    """Test basic backtest execution with minimal config."""
    from engine import run_backtest
    import pandas as pd
    import numpy as np

    # Create minimal test data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)

    df = pd.DataFrame({
        'time': dates,
        'open': prices,
        'high': prices + np.random.rand(100) * 2,
        'low': prices - np.random.rand(100) * 2,
        'close': prices + np.random.randn(100) * 0.3,
        'volume': np.random.randint(1000, 10000, 100)
    })

    config = {
        "meta": {
            "symbols": ["TEST"],
            "timeframe": "1h"
        },
        "strategy": {
            "type": "ema_cross",
            "params": {
                "emaFast": 5,
                "emaSlow": 20
            }
        },
        "initial_equity": 10000,
        "df": df  # Pass DataFrame directly
    }

    result = run_backtest(config)

    # Verify result structure
    assert "meta" in result
    assert "summary" in result
    assert "equityCurve" in result
    assert "trades" in result

    # Verify summary has expected keys
    summary = result["summary"]
    assert "totalTrades" in summary
    assert "netProfit" in summary


def test_tasks_import_engine():
    """Test that Celery tasks can import engine functions."""
    # This test verifies the import works without errors
    from app.services.tasks import (
        run_backtest_task,
        optimize_task,
        ai_agent_task,
        chart_data_task,
    )

    assert callable(run_backtest_task)
    assert callable(optimize_task)
    assert callable(ai_agent_task)
    assert callable(chart_data_task)
