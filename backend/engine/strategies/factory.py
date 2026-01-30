# engine/strategies/factory.py
import os

from .rf_st_rsi_strategy import RFSTRSIStrategy
from .rf_st_rsi_combined_strategy import RFSTRSICombinedStrategy


def create_strategy(strategy_type, df, params):
    if strategy_type == "rf_st_rsi":
        return RFSTRSIStrategy(df, params)
    elif strategy_type == "rf_st_rsi_combined":
        return RFSTRSICombinedStrategy(df, params)

    raise ValueError(f"Unknown strategy type: {strategy_type}")
