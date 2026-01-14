# engine/strategies/factory.py
import os

# Try to import fast version first, fallback to original
USE_FAST_STRATEGY = os.getenv("USE_FAST_STRATEGY", "true").lower() == "true"

try:
    from .rf_st_rsi_fast import RFSTRSIStrategyFast
    FAST_AVAILABLE = True
except ImportError:
    FAST_AVAILABLE = False

from .rf_st_rsi_strategy import RFSTRSIStrategy


def create_strategy(strategy_type, df, params):
    if strategy_type == "rf_st_rsi":
        # Use fast version if available and enabled
        if USE_FAST_STRATEGY and FAST_AVAILABLE:
            return RFSTRSIStrategyFast(df, params)
        return RFSTRSIStrategy(df, params)

    raise ValueError(f"Unknown strategy type: {strategy_type}")
