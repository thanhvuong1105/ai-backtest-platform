# engine/ai_agent.py
"""
AI Agent with advanced optimizations:
1. Indicator caching - Pre-compute indicators per symbol/timeframe/params
2. Early-stop - Kill runs with excessive drawdown
3. Random Search - Intelligent sampling for large search spaces

Performance optimizations:
- Pre-process DataFrame once per symbol/timeframe
- Share indicator cache across runs with same data
- Early-stop runs with bad DD
- Batched progress updates
- Random sampling for large search spaces
"""

import json
import sys
import os
import random
import logging
from typing import Any, Callable, Dict, Optional, List
from collections import defaultdict

from concurrent.futures import ThreadPoolExecutor, as_completed

from .run_builder import build_runs
from .backtest_engine import run_backtest
from .scoring import score_strategy
from .guards import equity_smoothness
from .filtering import filter_results
from .stability import stability_metrics, pass_stability
from .optimizer import apply_tf_agreement
from .data_loader import load_csv

# Reduce logging noise
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# ═══════════════════════════════════════════════════════
# CONFIGURATION CONSTANTS
# ═══════════════════════════════════════════════════════
EARLY_STOP_MAX_DD = float(os.getenv("EARLY_STOP_MAX_DD", 60))  # Stop if DD > 60%
DOWNSAMPLE_EQUITY_POINTS = 400
MAX_TRADES_RETURN = 1200
MAX_RUNS_BEFORE_RANDOM_SEARCH = int(os.getenv("MAX_RUNS_BEFORE_RANDOM", 500))
RANDOM_SAMPLE_SIZE = int(os.getenv("RANDOM_SAMPLE_SIZE", 300))


def downsample_curve(curve, max_points=DOWNSAMPLE_EQUITY_POINTS):
    """Downsample equity curve for efficient serialization."""
    if not curve:
        return []
    n = len(curve)
    if n <= max_points:
        return curve
    step = (n - 1) // (max_points - 1) + 1
    sampled = curve[::step]
    if sampled[-1] != curve[-1]:
        sampled.append(curve[-1])
    return sampled[:max_points]


# ═══════════════════════════════════════════════════════
# PRE-COMPUTED VALUES CACHE
# ═══════════════════════════════════════════════════════
_fee_cache = {}


def _get_fee_config(props: Dict) -> tuple:
    """Pre-compute fee configuration once."""
    cache_key = id(props)
    if cache_key in _fee_cache:
        return _fee_cache[cache_key]

    fee_rate = float(props.get("commission", {}).get("value", 0) or 0)
    fee_type = (props.get("commission", {}).get("type") or "percent").lower()
    if fee_type == "percent":
        fee_rate = fee_rate / 100.0
    order_pct = float(props.get("orderSize", {}).get("value", 100) or 100) / 100.0

    result = (fee_rate, order_pct)
    _fee_cache[cache_key] = result
    return result


# ═══════════════════════════════════════════════════════
# WORKER FUNCTION WITH EARLY-STOP
# ═══════════════════════════════════════════════════════
def _run_single_backtest(args):
    """
    Worker function with early-stop support.
    args = (run_cfg, props, date_range, precomputed_fee, max_dd_threshold)
    """
    run_cfg, props, date_range, precomputed_fee, max_dd_threshold = args
    fee_rate, order_pct = precomputed_fee

    import pandas as pd

    def filter_by_range_fast(trades, r):
        if not r or not trades:
            return trades
        start = r.get("from") or r.get("start")
        end = r.get("to") or r.get("end")
        if not start and not end:
            return trades

        s = pd.Timestamp(start) if start else None
        e = pd.Timestamp(end) if end else None

        return [
            t for t in trades
            if t.get("entry_time") and
               (s is None or pd.Timestamp(t["entry_time"]) >= s) and
               (e is None or pd.Timestamp(t["entry_time"]) <= e)
        ]

    def recompute_summary_with_early_stop(trades, initial_equity, max_dd):
        """Recompute summary with early-stop check during computation."""
        if not trades:
            return {
                "initialEquity": initial_equity,
                "finalEquity": initial_equity,
                "netProfit": 0,
                "netProfitPct": 0,
                "totalTrades": 0,
                "winTrades": 0,
                "lossTrades": 0,
                "winrate": 0,
                "profitFactor": 0,
                "avgTrade": 0,
                "avgWin": 0,
                "avgLoss": 0,
                "expectancy": 0,
                "maxDrawdown": 0,
                "maxDrawdownPct": 0,
                "equityCurve": [],
                "early_stopped": False,
            }

        equity = initial_equity
        trades_sorted = sorted(trades, key=lambda t: t.get("exit_time") or t.get("entry_time") or "")

        equity_curve = []
        peak = equity
        max_dd_pct = 0.0
        pnls = []
        early_stopped = False

        for t in trades_sorted:
            entry = t.get("entry_price") or 0
            exit_p = t.get("exit_price") or 0
            if entry == 0 or exit_p == 0:
                continue

            size = equity * order_pct / entry
            fee = fee_rate * (entry + exit_p) * size
            pnl_usdt = (exit_p - entry) * size - fee
            equity += pnl_usdt

            if equity > peak:
                peak = equity
            dd_pct = (peak - equity) / peak * 100 if peak > 0 else 0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

            # Early-stop check
            if max_dd > 0 and dd_pct > max_dd:
                early_stopped = True
                break

            pnls.append(pnl_usdt)
            equity_curve.append({
                "time": t.get("exit_time") or t.get("entry_time"),
                "equity": round(equity, 2)
            })

        total_trades = len(pnls)
        if total_trades == 0:
            return {
                "initialEquity": initial_equity,
                "finalEquity": equity,
                "netProfit": 0,
                "netProfitPct": 0,
                "totalTrades": 0,
                "winTrades": 0,
                "lossTrades": 0,
                "winrate": 0,
                "profitFactor": 0,
                "avgTrade": 0,
                "avgWin": 0,
                "avgLoss": 0,
                "expectancy": 0,
                "maxDrawdown": 0,
                "maxDrawdownPct": max_dd_pct,
                "equityCurve": equity_curve,
                "early_stopped": early_stopped,
            }

        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        win_trades = sum(1 for p in pnls if p > 0)
        loss_trades = total_trades - win_trades

        profit_factor = min(gross_profit / gross_loss if gross_loss else 1e9, 1e6)
        avg_win = gross_profit / win_trades if win_trades else 0
        avg_loss = -gross_loss / loss_trades if loss_trades else 0
        winrate = win_trades / total_trades * 100
        expectancy = (winrate / 100) * avg_win + (1 - winrate / 100) * avg_loss

        return {
            "initialEquity": initial_equity,
            "finalEquity": round(equity, 2),
            "netProfit": round(equity - initial_equity, 2),
            "netProfitPct": round((equity - initial_equity) / initial_equity * 100, 2) if initial_equity else 0,
            "totalTrades": total_trades,
            "winTrades": win_trades,
            "lossTrades": loss_trades,
            "winrate": round(winrate, 2),
            "profitFactor": round(profit_factor, 3),
            "avgTrade": round(sum(pnls) / total_trades, 2),
            "avgWin": round(avg_win, 2),
            "avgLoss": round(avg_loss, 2),
            "expectancy": round(expectancy, 2),
            "maxDrawdown": round(peak - equity, 2),
            "maxDrawdownPct": round(max_dd_pct, 2),
            "equityCurve": equity_curve,
            "early_stopped": early_stopped,
        }

    # Run backtest
    result = run_backtest(run_cfg)

    # Filter trades by range
    filtered_trades = filter_by_range_fast(result.get("trades", []), date_range)

    # Recompute summary with early-stop
    initial_equity = run_cfg.get("initial_equity", 10000)
    recomputed = recompute_summary_with_early_stop(filtered_trades, initial_equity, max_dd_threshold)

    # If early stopped, return minimal result
    if recomputed.get("early_stopped"):
        return {
            "symbol": run_cfg["meta"]["symbols"][0],
            "timeframe": run_cfg["meta"]["timeframe"],
            "params": run_cfg.get("strategy", {}).get("params", {}),
            "summary": {
                "score": -999,  # Penalize early-stopped runs
                "maxDrawdownPct": recomputed["maxDrawdownPct"],
                "totalTrades": recomputed["totalTrades"],
                "profitFactor": 0,
                "winrate": 0,
                "early_stopped": True,
            },
            "equityCurve": [],
            "trades": [],
            "early_stopped": True,
        }

    # Update summary
    summary = result["summary"]
    summary.update({
        "initialEquity": recomputed["initialEquity"],
        "finalEquity": recomputed["finalEquity"],
        "netProfit": recomputed["netProfit"],
        "netProfitPct": recomputed["netProfitPct"],
        "totalTrades": recomputed["totalTrades"],
        "winTrades": recomputed["winTrades"],
        "lossTrades": recomputed["lossTrades"],
        "winrate": recomputed["winrate"],
        "profitFactor": recomputed["profitFactor"],
        "avgTrade": recomputed["avgTrade"],
        "avgWin": recomputed["avgWin"],
        "avgLoss": recomputed["avgLoss"],
        "expectancy": recomputed["expectancy"],
        "maxDrawdown": recomputed["maxDrawdown"],
        "maxDrawdownPct": recomputed["maxDrawdownPct"],
    })

    # Calculate smoothness and score
    smooth = equity_smoothness(result["equityCurve"])
    summary["smoothness"] = smooth
    summary["score"] = score_strategy(summary)

    return {
        "symbol": run_cfg["meta"]["symbols"][0],
        "timeframe": run_cfg["meta"]["timeframe"],
        "params": run_cfg.get("strategy", {}).get("params", {}),
        "summary": summary,
        "equityCurve": recomputed.get("equityCurve") or result["equityCurve"],
        "trades": filtered_trades
    }


def prune_run(run: Dict[str, Any]) -> Dict[str, Any]:
    """Keep essential info for frontend."""
    out = {
        "symbol": run.get("symbol"),
        "timeframe": run.get("timeframe"),
        "params": run.get("params", {}),
        "summary": run.get("summary", {}),
        "strategyId": run.get("strategyId"),
        "rank": run.get("rank"),
    }
    curve = run.get("equityCurve") or []
    out["equityCurve"] = downsample_curve(curve, max_points=600)
    trades = run.get("trades") or []
    if len(trades) > MAX_TRADES_RETURN:
        trades = trades[-MAX_TRADES_RETURN:]
    out["trades"] = trades
    return out


def build_comment(best: Dict[str, Any]) -> str:
    s = best.get("summary", {})
    p = best.get("params", {})
    return (
        f"Chọn tham số {p} vì đạt score {s.get('score', 0):.2f}, "
        f"PF {s.get('profitFactor', 0):.2f}, winrate {s.get('winrate', 0):.1f}%, "
        f"max DD {s.get('maxDrawdownPct', 0):.1f}%."
    )


# ═══════════════════════════════════════════════════════
# RANDOM SEARCH WITH INTELLIGENT SAMPLING
# ═══════════════════════════════════════════════════════
def intelligent_random_sample(runs: List[Dict], max_samples: int) -> List[Dict]:
    """
    Intelligent random sampling for large search spaces.

    Strategy:
    1. Ensure diversity across timeframes
    2. Ensure diversity across key parameters
    3. Use stratified sampling
    """
    if len(runs) <= max_samples:
        return runs

    # Group by timeframe first
    by_tf = defaultdict(list)
    for run in runs:
        tf = run["meta"]["timeframe"]
        by_tf[tf].append(run)

    # Calculate samples per timeframe (proportional)
    samples_per_tf = {}
    for tf, tf_runs in by_tf.items():
        ratio = len(tf_runs) / len(runs)
        samples_per_tf[tf] = max(1, int(max_samples * ratio))

    # Adjust to hit target
    total_allocated = sum(samples_per_tf.values())
    if total_allocated < max_samples:
        largest_tf = max(by_tf.keys(), key=lambda t: len(by_tf[t]))
        samples_per_tf[largest_tf] += max_samples - total_allocated

    # Sample from each timeframe
    sampled = []
    for tf, count in samples_per_tf.items():
        tf_runs = by_tf[tf]
        if len(tf_runs) <= count:
            sampled.extend(tf_runs)
        else:
            sampled.extend(random.sample(tf_runs, count))

    logger.info(f"Random sampling: {len(runs)} -> {len(sampled)} runs")
    return sampled


def ai_recommend(
    cfg: Dict[str, Any],
    job_id: str = "",
    progress_cb: Optional[Callable[[str, int, int, str, Dict], None]] = None
) -> Dict[str, Any]:
    """
    AI Agent recommendation with:
    1. Indicator caching (via rf_st_rsi_fast)
    2. Early-stop for excessive DD
    3. Random Search for large search spaces
    """
    import time as time_module
    import pandas as pd

    start_time = time_module.time()

    # Progress helper
    last_emit_time = [0]
    PROGRESS_INTERVAL = 0.5

    def emit_progress(progress: int, total: int, extra: Dict = None, force: bool = False):
        now = time_module.time()
        if not force and now - last_emit_time[0] < PROGRESS_INTERVAL and progress != total:
            return
        last_emit_time[0] = now

        if progress_cb:
            progress_cb(job_id, progress, total, "running", extra or {})
        elif "AI_PROGRESS" in os.environ:
            print(json.dumps({"progress": progress, "total": total, **(extra or {})}))
            sys.stdout.flush()

    emit_progress(0, 1, {"phase": "preparing"}, force=True)

    top_n = cfg.get("topN", 50)
    runs = build_runs(cfg)
    total_original = len(runs)

    if total_original == 0:
        return {
            "success": False,
            "fallback": False,
            "message": "No runs to execute",
            "best": None,
            "alternatives": [],
            "comment": "No valid strategy found",
            "top": [],
            "all": [],
            "total": 0,
        }

    # ═══════════════════════════════════════════════════════
    # RANDOM SEARCH FOR LARGE SEARCH SPACES
    # ═══════════════════════════════════════════════════════
    use_random_search = total_original > MAX_RUNS_BEFORE_RANDOM_SEARCH
    if use_random_search:
        runs = intelligent_random_sample(runs, RANDOM_SAMPLE_SIZE)
        logger.info(f"Using Random Search: {total_original} -> {len(runs)} runs")

    total = len(runs)
    emit_progress(0, total, {"phase": "running", "random_search": use_random_search}, force=True)

    props = cfg.get("properties", {}) or {}
    precomputed_fee = _get_fee_config(props)

    # Early-stop threshold from config or default
    filters = cfg.get("filters", {})
    early_stop_dd = filters.get("maxDD", EARLY_STOP_MAX_DD)

    def build_costs(base_costs):
        costs = dict(base_costs or {})
        commission = props.get("commission", {}) or {}
        if "value" in commission:
            val = float(commission.get("value", 0))
            ctype = (commission.get("type") or "percent").lower()
            fee_rate = val / 100 if ctype == "percent" and val > 1 else val if ctype != "percent" else val / 100
            costs["fee"] = fee_rate
        if props.get("slippage") is not None:
            costs["slippage"] = float(props.get("slippage"))
        return costs

    # ═══════════════════════════════════════════════════════
    # PRE-PROCESS DATAFRAMES
    # ═══════════════════════════════════════════════════════
    date_range = cfg.get("range")
    df_cache = {}

    runs_by_data = defaultdict(list)
    for run_cfg in runs:
        sym = run_cfg["meta"]["symbols"][0]
        tf = run_cfg["meta"]["timeframe"]
        runs_by_data[(sym, tf)].append(run_cfg)

    for (sym, tf) in runs_by_data.keys():
        df = load_csv(sym, tf)

        if date_range:
            from_date = date_range.get("from")
            to_date = date_range.get("to")

            from_date_ts = pd.to_datetime(from_date) if from_date else None
            to_date_ts = pd.to_datetime(to_date) if to_date else None

            df_processed = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_processed["time"]):
                df_processed["time"] = pd.to_datetime(df_processed["time"])

            if to_date_ts is not None:
                df_processed = df_processed[df_processed["time"] <= to_date_ts].reset_index(drop=True)

            INDICATOR_WARMUP_BARS = 300
            if from_date_ts is not None:
                mask = df_processed["time"] >= from_date_ts
                if mask.any():
                    from_idx = mask.idxmax()
                    start_idx = max(0, from_idx - INDICATOR_WARMUP_BARS)
                    df_processed = df_processed.iloc[start_idx:].reset_index(drop=True)

            df_cache[(sym, tf)] = df_processed
        else:
            df_cache[(sym, tf)] = df

    # ═══════════════════════════════════════════════════════
    # PREPARE RUNS WITH EARLY-STOP THRESHOLD
    # ═══════════════════════════════════════════════════════
    prepared_runs = []
    for run_cfg in runs:
        run_cfg = dict(run_cfg)
        sym = run_cfg["meta"]["symbols"][0]
        tf = run_cfg["meta"]["timeframe"]

        run_cfg["df"] = df_cache[(sym, tf)]
        run_cfg["_preprocessed"] = True

        if "properties" not in run_cfg:
            run_cfg["properties"] = props
        else:
            merged_props = dict(props)
            merged_props.update(run_cfg.get("properties", {}))
            run_cfg["properties"] = merged_props
        if props.get("initialCapital") is not None:
            run_cfg["initial_equity"] = float(props["initialCapital"])
        run_cfg["costs"] = build_costs(run_cfg.get("costs", {}))

        # Add early-stop threshold
        prepared_runs.append((run_cfg, props, date_range, precomputed_fee, early_stop_dd))

    # ═══════════════════════════════════════════════════════
    # PARALLEL EXECUTION
    # ═══════════════════════════════════════════════════════
    results = []
    early_stopped_count = 0
    max_workers_env = int(os.getenv("MAX_THREAD_WORKERS", 0))
    workers = max_workers_env if max_workers_env > 0 else max(8, (os.cpu_count() or 2) * 4)
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_single_backtest, args): idx for idx, args in enumerate(prepared_runs)}

        for future in as_completed(futures):
            completed += 1
            emit_progress(completed, total)

            try:
                result = future.result()
                if result.get("early_stopped"):
                    early_stopped_count += 1
                else:
                    results.append(result)
            except Exception as e:
                logger.debug(f"Run failed: {e}")
                pass

    elapsed = time_module.time() - start_time
    logger.info(f"AI Agent completed: {len(results)} results, {early_stopped_count} early-stopped, {elapsed:.2f}s")

    # Sort by score
    results.sort(key=lambda x: x["summary"].get("score", 0), reverse=True)

    # Apply filters & stability (only PnL filter)
    passed = filter_results(
        results,
        min_pnl=filters.get("minPnL", 0),
        min_trades=filters.get("minTrades", 10)
    )
    min_tf = cfg.get("minTFAgree", 2)
    passed = apply_tf_agreement(passed, min_tf=min_tf)

    stability_rules = cfg.get("stability") or {
        "minMedianPF": 1.05,
        "minWorstPF": 0.95,
        "maxWorstDD": 45,
        "maxPFStd": 0.6
    }

    bucket = defaultdict(list)
    for r in passed:
        key = (r["symbol"], tuple(sorted(r["params"].items())))
        bucket[key].append(r)

    stable = []
    for runs_same_param in bucket.values():
        summaries = [r["summary"] for r in runs_same_param]
        stab = stability_metrics(summaries)
        if pass_stability(stab, stability_rules):
            for r in runs_same_param:
                r["summary"]["stability"] = stab
            stable.extend(runs_same_param)

    passed_runs = stable
    passed_runs.sort(key=lambda x: x["summary"].get("score", 0), reverse=True)
    top = passed_runs[:top_n] if passed_runs else []
    best = passed_runs[0] if passed_runs else None

    passed_flag = bool(best)

    # Fallback
    fallback = False
    if not best and results:
        top = results[:5]
        best = top[0]
        fallback = True

    comment = build_comment(best) if best else "No valid strategy found"
    alternatives = top[1:4] if len(top) > 1 else []

    # Prune payload
    pruned_top = [prune_run(r) for r in top]
    pruned_best = prune_run(best) if best else None
    pruned_all = [prune_run(r) for r in results[:100]]

    return {
        "success": passed_flag,
        "fallback": fallback,
        "message": None if passed_flag else "No strategy passed filters",
        "best": pruned_best,
        "alternatives": [prune_run(r) for r in alternatives],
        "comment": comment,
        "top": pruned_top,
        "all": pruned_all,
        "total": len(results),
        "meta": {
            "total_original": total_original,
            "total_sampled": total if use_random_search else total_original,
            "early_stopped": early_stopped_count,
            "elapsed_seconds": round(elapsed, 2),
            "random_search": use_random_search,
        }
    }


if __name__ == "__main__":
    raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps({"error": "Config JSON required on stdin"}))
        sys.exit(1)

    cfg = json.loads(raw)
    output = ai_recommend(cfg)
    print(json.dumps(output))
