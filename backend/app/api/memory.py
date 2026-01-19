# app/api/memory.py
"""
API endpoints for Quant Brain Memory management.

View and manage stored genomes from ParamMemory.
"""

from fastapi import APIRouter, Query
from typing import List, Optional
from datetime import datetime

from engine.param_memory import (
    get_memory_stats,
    get_top_genomes_by_score,
    get_genomes_by_scan,
    get_all_top_genomes,
    clear_strategy_memory,
    get_strategy_info,
)
from engine.strategy_hash import generate_strategy_hash, ENGINE_VERSION

router = APIRouter(prefix="/api/memory", tags=["memory"])


@router.get("/stats")
async def get_stats(
    strategy_type: str = Query(default="rf_st_rsi", description="Strategy type")
):
    """
    Get memory statistics for a strategy.

    Returns:
        - total_genomes: Total number of stored genomes
        - symbols: Count per symbol
        - timeframes: Count per timeframe
    """
    strategy_hash = generate_strategy_hash(strategy_type, ENGINE_VERSION)
    stats = get_memory_stats(strategy_hash)

    return {
        "success": True,
        "strategy_hash": strategy_hash,
        "strategy_type": strategy_type,
        "engine_version": ENGINE_VERSION,
        **stats
    }


@router.get("/genomes")
async def get_genomes(
    symbol: str = Query(default="BTCUSDT", description="Trading symbol"),
    timeframe: str = Query(default="30m", description="Timeframe"),
    strategy_type: str = Query(default="rf_st_rsi", description="Strategy type"),
    limit: int = Query(default=20, ge=1, le=100, description="Max genomes to return")
):
    """
    Get top-performing genomes for a symbol/timeframe.

    Returns list of genomes sorted by score (descending).
    """
    strategy_hash = generate_strategy_hash(strategy_type, ENGINE_VERSION)

    # Use scan to get ALL genomes (sorted set may not have all records)
    # This ensures we show all stored genomes, not just recent ones
    genomes = get_genomes_by_scan(strategy_hash, symbol, timeframe, limit)

    # If scan returns fewer than expected, also check sorted set
    if len(genomes) < limit:
        sorted_genomes = get_top_genomes_by_score(strategy_hash, symbol, timeframe, limit)
        # Merge and deduplicate by genome_hash
        existing_hashes = {g.get("genome_hash") for g in genomes}
        for g in sorted_genomes:
            if g.get("genome_hash") not in existing_hashes:
                genomes.append(g)
                existing_hashes.add(g.get("genome_hash"))
        # Re-sort by score
        genomes.sort(key=lambda x: x.get("results", {}).get("score", 0), reverse=True)
        genomes = genomes[:limit]

    # Format for frontend display
    formatted_genomes = []
    for g in genomes:
        results = g.get("results", {})
        genome_data = g.get("genome", {})
        equity_curve = g.get("equity_curve", [])

        # Map keys: quant_brain saves with lowercase/underscore, UI expects camelCase
        # quant_brain saves: pf, winrate, max_dd, net_profit, net_profit_pct, total_trades, score
        formatted_genomes.append({
            "id": g.get("genome_hash", "")[:8],
            "genome_hash": g.get("genome_hash", ""),
            "timeframe": timeframe,  # Add timeframe to each genome
            "score": round(results.get("score", 0), 4),
            "brainScore": round(results.get("score", 0), 4),  # score = brainScore
            "pf": round(results.get("pf", 0), 2),
            "winrate": round(results.get("winrate", 0), 1),
            "maxDD": round(results.get("max_dd", results.get("maxDrawdownPct", 0)), 1),
            "netProfit": round(results.get("net_profit", results.get("netProfit", 0)), 2),
            "netProfitPct": round(results.get("net_profit_pct", results.get("netProfitPct", 0)), 2),
            "totalTrades": results.get("total_trades", results.get("totalTrades", 0)),
            "robustness": round(results.get("robustness_score", 0), 2),
            "timestamp": g.get("timestamp", 0),
            "timestampStr": datetime.fromtimestamp(g.get("timestamp", 0)).strftime("%Y-%m-%d %H:%M") if g.get("timestamp") else "",
            "test_count": g.get("test_count", 1),
            "genome": genome_data,
            "market_profile": g.get("market_profile", {}),
            # Real equity curve data from stored record
            "equityCurve": equity_curve,
            # Backtest period
            "backtest_start": g.get("backtest_start", ""),
            "backtest_end": g.get("backtest_end", ""),
        })

    return {
        "success": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy_hash": strategy_hash,
        "count": len(formatted_genomes),
        "genomes": formatted_genomes
    }


@router.get("/genomes/all")
async def get_all_genomes(
    symbols: str = Query(default="BTCUSDT", description="Comma-separated symbols"),
    timeframes: str = Query(default="30m", description="Comma-separated timeframes"),
    strategy_type: str = Query(default="rf_st_rsi", description="Strategy type"),
    limit_per_combo: int = Query(default=10, ge=1, le=50, description="Max genomes per combo")
):
    """
    Get top genomes across multiple symbols/timeframes.
    """
    strategy_hash = generate_strategy_hash(strategy_type, ENGINE_VERSION)

    symbol_list = [s.strip() for s in symbols.split(",")]
    tf_list = [t.strip() for t in timeframes.split(",")]

    genomes = get_all_top_genomes(strategy_hash, symbol_list, tf_list, limit_per_combo)

    # Format for frontend (support both old and new key formats)
    formatted_genomes = []
    for g in genomes:
        results = g.get("results", {})

        formatted_genomes.append({
            "id": g.get("genome_hash", "")[:8],
            "symbol": g.get("symbol", ""),
            "timeframe": g.get("timeframe", ""),
            "score": round(results.get("score", 0), 4),
            "pf": round(results.get("pf", 0), 2),
            "winrate": round(results.get("winrate", 0), 1),
            "maxDD": round(results.get("max_dd", results.get("maxDrawdownPct", 0)), 1),
            "netProfitPct": round(results.get("net_profit_pct", results.get("netProfitPct", 0)), 2),
            "totalTrades": results.get("total_trades", results.get("totalTrades", 0)),
            "timestamp": g.get("timestamp", 0),
        })

    return {
        "success": True,
        "symbols": symbol_list,
        "timeframes": tf_list,
        "count": len(formatted_genomes),
        "genomes": formatted_genomes
    }


@router.delete("/clear")
async def clear_memory(
    strategy_type: str = Query(default="rf_st_rsi", description="Strategy type"),
    confirm: bool = Query(default=False, description="Confirm deletion")
):
    """
    Clear all stored genomes for a strategy.

    WARNING: This is irreversible!
    """
    if not confirm:
        return {
            "success": False,
            "error": "Set confirm=true to delete all genomes"
        }

    strategy_hash = generate_strategy_hash(strategy_type, ENGINE_VERSION)
    deleted = clear_strategy_memory(strategy_hash)

    return {
        "success": True,
        "deleted": deleted,
        "message": f"Deleted {deleted} genome records"
    }
