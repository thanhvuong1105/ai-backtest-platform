# engine/ai_agent_fast.py
"""
AI Agent sử dụng Rust engine cho hiệu năng tối đa.
Fallback về Python nếu Rust không có.

Performance:
- Rust available: 50-100x faster than pure Python
- Python fallback: Same as original ai_agent.py
"""

import json
import sys
import os
from typing import Any, Dict, List
from collections import defaultdict

from engine.run_builder import build_runs
from engine.scoring import score_strategy
from engine.guards import equity_smoothness
from engine.filtering import filter_results
from engine.stability import stability_metrics, pass_stability
from engine.optimizer import apply_tf_agreement
from engine.data_loader import load_csv

# Check for Rust engine
try:
    from engine.rust_bridge import (
        run_backtest_rust,
        run_batch_backtests_rust,
        is_rust_available
    )
    RUST_AVAILABLE = is_rust_available()
except ImportError:
    RUST_AVAILABLE = False

# Fallback imports
if not RUST_AVAILABLE:
    from engine.ai_agent import ai_recommend as ai_recommend_python


def downsample_curve(curve, max_points=400):
    if not curve:
        return []
    n = len(curve)
    if n <= max_points:
        return curve
    import math
    step = max(1, math.ceil((n - 1) / (max_points - 1)))
    sampled = curve[::step]
    if sampled[-1] != curve[-1]:
        sampled.append(curve[-1])
    return sampled[:max_points]


def prune_run(run: Dict[str, Any]) -> Dict[str, Any]:
    """Giữ thông tin cần thiết cho frontend."""
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
    max_trades = 1200
    if len(trades) > max_trades:
        trades = trades[-max_trades:]
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


def ai_recommend_fast(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    AI recommend sử dụng Rust engine cho hiệu năng tối đa.

    Args:
        cfg: Configuration dict

    Returns:
        Result dict with best, top, all strategies
    """
    # Fallback to Python if Rust not available
    if not RUST_AVAILABLE:
        print("Rust engine not available, using Python fallback...")
        return ai_recommend_python(cfg)

    import time
    start_time = time.time()

    def tf_to_minutes(tf: str) -> int:
        if not tf:
            return 60
        tf = tf.strip().lower()
        try:
            if tf.endswith("m"):
                return int(tf[:-1])
            if tf.endswith("h"):
                return int(tf[:-1]) * 60
            if tf.endswith("d"):
                return int(tf[:-1]) * 60 * 24
        except ValueError:
            pass
        return 60

    top_n = cfg.get("topN", 50)
    runs = build_runs(cfg)
    total = len(runs)
    props = cfg.get("properties", {}) or {}
    date_range = cfg.get("range", {})

    print(f"[Rust Engine] Running {total} backtests...")

    # Group runs by symbol/timeframe to batch by data
    grouped_runs = defaultdict(list)
    for run_cfg in runs:
        symbol = run_cfg["meta"]["symbols"][0]
        timeframe = run_cfg["meta"]["timeframe"]
        key = (symbol, timeframe)
        grouped_runs[key].append(run_cfg)

    results = []
    completed = 0

    # Process each group with batch Rust execution
    for (symbol, timeframe), group_runs in grouped_runs.items():
        # Load data once for this group
        df = load_csv(symbol, timeframe)

        # Prepare configs for batch execution
        batch_configs = []
        for run_cfg in group_runs:
            # Add properties and costs
            run_cfg = dict(run_cfg)
            if "properties" not in run_cfg:
                run_cfg["properties"] = props
            else:
                merged = dict(props)
                merged.update(run_cfg.get("properties", {}))
                run_cfg["properties"] = merged

            if props.get("initialCapital") is not None:
                run_cfg["initial_equity"] = float(props["initialCapital"])

            # Build costs
            costs = dict(run_cfg.get("costs", {}) or {})
            commission = props.get("commission", {}) or {}
            if "value" in commission:
                val = float(commission.get("value", 0))
                ctype = (commission.get("type") or "percent").lower()
                fee_rate = val / 100 if ctype == "percent" and val > 1 else val if ctype != "percent" else val / 100
                costs["fee"] = fee_rate
            if props.get("slippage") is not None:
                costs["slippage"] = float(props.get("slippage"))
            run_cfg["costs"] = costs

            batch_configs.append(run_cfg)

        # Run batch backtests with Rust
        try:
            batch_results = run_batch_backtests_rust(batch_configs, df)

            for i, result in enumerate(batch_results):
                run_cfg = batch_configs[i]

                # Add metadata
                result["symbol"] = symbol
                result["timeframe"] = timeframe
                result["params"] = run_cfg.get("strategy", {}).get("params", {})

                # Calculate smoothness and score
                summary = result.get("summary", {})
                eq_curve = result.get("equityCurve", [])
                smooth = equity_smoothness(eq_curve)
                summary["smoothness"] = smooth
                summary["score"] = score_strategy(summary)
                result["summary"] = summary

                results.append(result)
                completed += 1

            # Progress
            if "AI_PROGRESS" in os.environ:
                print(json.dumps({"progress": completed, "total": total}))
                sys.stdout.flush()

        except Exception as e:
            print(f"Error in Rust batch: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to individual runs
            for run_cfg in batch_configs:
                try:
                    result = run_backtest_rust(run_cfg, df)
                    result["symbol"] = symbol
                    result["timeframe"] = timeframe
                    result["params"] = run_cfg.get("strategy", {}).get("params", {})
                    summary = result.get("summary", {})
                    eq_curve = result.get("equityCurve", [])
                    smooth = equity_smoothness(eq_curve)
                    summary["smoothness"] = smooth
                    summary["score"] = score_strategy(summary)
                    result["summary"] = summary
                    results.append(result)
                except Exception as e2:
                    print(f"Error in individual run: {e2}")
                completed += 1

    elapsed = time.time() - start_time
    print(f"[Rust Engine] Completed {len(results)} backtests in {elapsed:.2f}s ({elapsed/len(results)*1000:.1f}ms avg)")

    # Sort by score
    results.sort(key=lambda x: x["summary"].get("score", 0), reverse=True)

    # Apply filters & stability
    filters = cfg.get("filters", {})
    passed = filter_results(
        results,
        min_pf=filters.get("minPF", 1.0),
        min_trades=filters.get("minTrades", 30),
        max_dd=filters.get("maxDD", 40)
    )
    min_tf = cfg.get("minTFAgree", 2)
    passed = apply_tf_agreement(passed, min_tf=min_tf)

    stability_rules = cfg.get("stability", {
        "minMedianPF": 1.05,
        "minWorstPF": 0.95,
        "maxWorstDD": 45,
        "maxPFStd": 0.6
    })

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
        "message": None if passed_flag else "No strategy passed filters, trả về best từ all",
        "best": pruned_best,
        "alternatives": [prune_run(r) for r in alternatives],
        "comment": comment,
        "top": pruned_top,
        "all": pruned_all,
        "total": len(results),
        "engine": "rust" if RUST_AVAILABLE else "python",
        "elapsed_seconds": elapsed,
    }


# Alias for drop-in replacement
ai_recommend = ai_recommend_fast


if __name__ == "__main__":
    raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps({"error": "Config JSON required on stdin"}))
        sys.exit(1)

    cfg = json.loads(raw)
    output = ai_recommend_fast(cfg)
    print(json.dumps(output))
