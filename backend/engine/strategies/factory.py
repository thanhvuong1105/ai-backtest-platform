# engine/strategies/factory.py
from .ema_cross import EMACrossStrategy
from .rf_st_rsi_strategy import RFSTRSIStrategy


def create_strategy(strategy_type, df, params):
    if strategy_type == "ema_cross":
        return EMACrossStrategy(df, params)
    elif strategy_type == "rf_st_rsi":
        return RFSTRSIStrategy(df, params)

    raise ValueError(f"Unknown strategy type: {strategy_type}")
