# engine/strategy_hash.py
"""
Strategy Hash Module for Quant AI Brain

Generates deterministic hash for strategy + engine version
to ensure genome memory is strategy-specific.
"""

import hashlib
import json
from typing import Dict, Any, Optional

# Engine version - update when backtest logic changes
ENGINE_VERSION = "1.0.0"


def generate_strategy_hash(
    strategy_type: str,
    engine_version: str = ENGINE_VERSION
) -> str:
    """
    Generate SHA256 hash for strategy + engine version.

    This ensures:
    - Genome memory is strategy-specific
    - Old memory invalidated when engine logic changes
    - Different strategies have separate memory pools

    Args:
        strategy_type: Strategy identifier (e.g., "rf_st_rsi")
        engine_version: Backtest engine version

    Returns:
        12-character hex hash
    """
    content = f"{strategy_type}:{engine_version}"
    full_hash = hashlib.sha256(content.encode()).hexdigest()
    return full_hash[:12]


def generate_genome_hash(genome: Dict[str, Any]) -> str:
    """
    Generate hash for a specific genome configuration.

    Used for deduplication and lookup in ParamMemory.

    Args:
        genome: Genome dictionary with entry, sl, tp params

    Returns:
        12-character hex hash
    """
    # Sort keys for deterministic serialization
    sorted_genome = json.dumps(genome, sort_keys=True)
    full_hash = hashlib.sha256(sorted_genome.encode()).hexdigest()
    return full_hash[:12]


def flatten_params_to_genome(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert flat params dict to structured genome format.

    Args:
        params: Flat parameter dictionary from UI/config

    Returns:
        Structured genome with entry, sl, tp_dual, tp_rsi, mode blocks
    """
    return {
        "entry": {
            "st_atrPeriod": params.get("st_atrPeriod", 10),
            "st_src": params.get("st_src", "hl2"),
            "st_mult": params.get("st_mult", 2.0),
            "st_useATR": params.get("st_useATR", True),
            "rf_src": params.get("rf_src", "close"),
            "rf_period": params.get("rf_period", 100),
            "rf_mult": params.get("rf_mult", 3.0),
            "rsi_length": params.get("rsi_length", 14),
            "rsi_ma_length": params.get("rsi_ma_length", 6),
        },
        "sl": {
            "st_atrPeriod": params.get("sl_st_atrPeriod", 10),
            "st_src": params.get("sl_st_src", "hl2"),
            "st_mult": params.get("sl_st_mult", 4.0),
            "st_useATR": params.get("sl_st_useATR", True),
            "rf_period": params.get("sl_rf_period", 100),
            "rf_mult": params.get("sl_rf_mult", 7.0),
        },
        "tp_dual": {
            "st_atrPeriod": params.get("tp_dual_st_atrPeriod", 10),
            "st_mult": params.get("tp_dual_st_mult", 2.0),
            "rr_mult": params.get("tp_dual_rr_mult", 1.3),
        },
        "tp_rsi": {
            "st_atrPeriod": params.get("tp_rsi_st_atrPeriod", 10),
            "st_mult": params.get("tp_rsi_st_mult", 2.0),
            "rr_mult": params.get("tp_rsi_rr_mult", 1.3),
        },
        "mode": {
            "showDualFlip": params.get("showDualFlip", True),
            "showRSI": params.get("showRSI", True),
        }
    }


def genome_to_flat_params(genome: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert structured genome back to flat params for backtest engine.

    Args:
        genome: Structured genome dictionary

    Returns:
        Flat parameter dictionary for strategy
    """
    params = {}

    # Entry params
    entry = genome.get("entry", {})
    params["st_atrPeriod"] = entry.get("st_atrPeriod", 10)
    params["st_src"] = entry.get("st_src", "hl2")
    params["st_mult"] = entry.get("st_mult", 2.0)
    params["st_useATR"] = entry.get("st_useATR", True)
    params["rf_src"] = entry.get("rf_src", "close")
    params["rf_period"] = entry.get("rf_period", 100)
    params["rf_mult"] = entry.get("rf_mult", 3.0)
    params["rsi_length"] = entry.get("rsi_length", 14)
    params["rsi_ma_length"] = entry.get("rsi_ma_length", 6)

    # SL params
    sl = genome.get("sl", {})
    params["sl_st_atrPeriod"] = sl.get("st_atrPeriod", 10)
    params["sl_st_src"] = sl.get("st_src", "hl2")
    params["sl_st_mult"] = sl.get("st_mult", 4.0)
    params["sl_st_useATR"] = sl.get("st_useATR", True)
    params["sl_rf_period"] = sl.get("rf_period", 100)
    params["sl_rf_mult"] = sl.get("rf_mult", 7.0)

    # TP Dual params
    tp_dual = genome.get("tp_dual", {})
    params["tp_dual_st_atrPeriod"] = tp_dual.get("st_atrPeriod", 10)
    params["tp_dual_st_mult"] = tp_dual.get("st_mult", 2.0)
    params["tp_dual_rr_mult"] = tp_dual.get("rr_mult", 1.3)

    # TP RSI params
    tp_rsi = genome.get("tp_rsi", {})
    params["tp_rsi_st_atrPeriod"] = tp_rsi.get("st_atrPeriod", 10)
    params["tp_rsi_st_mult"] = tp_rsi.get("st_mult", 2.0)
    params["tp_rsi_rr_mult"] = tp_rsi.get("rr_mult", 1.3)

    # Mode params
    mode = genome.get("mode", {})
    params["showDualFlip"] = mode.get("showDualFlip", True)
    params["showRSI"] = mode.get("showRSI", True)

    return params


def verify_determinism(
    run_backtest_fn,
    config: Dict[str, Any],
    runs: int = 2
) -> tuple:
    """
    Verify that running the same config produces identical results.

    Args:
        run_backtest_fn: Backtest function to call
        config: Full backtest configuration
        runs: Number of runs to compare

    Returns:
        (is_deterministic, results_list)
    """
    results = []

    for _ in range(runs):
        result = run_backtest_fn(config)
        summary = result.get("summary", {})
        # Extract key metrics for comparison
        metrics = {
            "pf": round(summary.get("profitFactor", 0), 6),
            "wr": round(summary.get("winrate", 0), 6),
            "dd": round(summary.get("maxDrawdownPct", 0), 6),
            "trades": summary.get("totalTrades", 0),
            "profit": round(summary.get("netProfit", 0), 2),
        }
        results.append(metrics)

    # Check all results are identical
    is_deterministic = all(r == results[0] for r in results)

    return is_deterministic, results


def get_strategy_registry_key(strategy_hash: str) -> str:
    """Get Redis key for strategy registry."""
    return f"strategy_registry:{strategy_hash}"


def create_registry_entry(
    strategy_type: str,
    engine_version: str = ENGINE_VERSION
) -> Dict[str, Any]:
    """
    Create a registry entry for a strategy hash.

    Args:
        strategy_type: Strategy identifier
        engine_version: Engine version

    Returns:
        Registry entry dict
    """
    import time

    return {
        "type": strategy_type,
        "version": engine_version,
        "hash": generate_strategy_hash(strategy_type, engine_version),
        "created": int(time.time()),
    }
