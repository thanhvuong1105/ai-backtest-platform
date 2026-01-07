# engine/ai_agent.py
"""
AI Agent đơn giản:
- Nhận config JSON (stdin hoặc qua hàm ai_recommend(cfg))
- Chạy optimize() để lấy best/top
- Trả lời kèm nhận xét logic-based (chưa dùng LLM)
"""

import json
import sys
import os
from typing import Any, Dict
from collections import defaultdict

# Use ThreadPoolExecutor for better performance on macOS
# ProcessPoolExecutor has too much overhead for this workload
from concurrent.futures import ThreadPoolExecutor, as_completed

from engine.run_builder import build_runs
from engine.backtest_engine import run_backtest
from engine.scoring import score_strategy
from engine.guards import equity_smoothness
from engine.filtering import filter_results
from engine.stability import stability_metrics, pass_stability
from engine.optimizer import apply_tf_agreement
from engine.data_loader import load_csv


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


# Worker function for parallel execution (must be at module level for pickling)
def _run_single_backtest(args):
    """
    Worker function chạy trong process riêng.
    args = (run_cfg, props, date_range, min_tf_minutes)
    """
    run_cfg, props, date_range, min_tf_minutes = args

    import pandas as pd

    def filter_by_range(trades, r):
        if not r:
            return trades
        start = r.get("from") or r.get("start")
        end = r.get("to") or r.get("end")
        if not start and not end:
            return trades
        s = pd.to_datetime(start) if start else None
        e = pd.to_datetime(end) if end else None
        out = []
        for t in trades:
            et = pd.to_datetime(t.get("entry_time")) if t.get("entry_time") else None
            if et is None:
                continue
            if s is not None and et < s:
                continue
            if e is not None and et > e:
                continue
            out.append(t)
        return out

    def recompute_summary_fast(trades, props, initial_equity):
        """Simplified recompute for parallel execution"""
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
            }

        equity = initial_equity
        fee_rate = float(props.get("commission", {}).get("value", 0) or 0)
        fee_type = (props.get("commission", {}).get("type") or "percent").lower()
        if fee_type == "percent":
            fee_rate = fee_rate / 100.0
        order_pct = float(props.get("orderSize", {}).get("value", 100) or 100)

        trades_sorted = sorted(trades, key=lambda t: pd.to_datetime(t.get("exit_time") or t.get("entry_time") or 0))

        equity_curve = []
        peak = equity
        max_dd_abs = 0
        max_dd_pct = 0
        pnls = []

        for t in trades_sorted:
            entry = t.get("entry_price") or 0
            exit_p = t.get("exit_price") or 0
            if entry == 0 or exit_p == 0:
                continue
            size = equity * (order_pct / 100.0) / entry
            notional = entry * size
            fee = fee_rate * (entry + exit_p) * size
            pnl_usdt = (exit_p - entry) * size - fee
            equity += pnl_usdt
            peak = max(peak, equity)
            dd_abs = equity - peak
            dd_pct = (peak - equity) / peak * 100 if peak > 0 else 0
            max_dd_abs = min(max_dd_abs, dd_abs)
            max_dd_pct = max(max_dd_pct, dd_pct)
            pnls.append(pnl_usdt)
            equity_curve.append({
                "time": t.get("exit_time") or t.get("entry_time"),
                "equity": equity
            })

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        total_trades = len(pnls)
        win_trades = len(wins)
        loss_trades = len(losses)
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        raw_pf = gross_profit / gross_loss if gross_loss else 1e9
        profit_factor = min(raw_pf, 1e6)
        avg_win = gross_profit / win_trades if win_trades else 0
        avg_loss = sum(losses) / loss_trades if loss_trades else 0
        expectancy = (win_trades / total_trades) * avg_win + (loss_trades / total_trades) * avg_loss if total_trades else 0

        return {
            "initialEquity": initial_equity,
            "finalEquity": equity,
            "netProfit": equity - initial_equity,
            "netProfitPct": (equity - initial_equity) / initial_equity * 100 if initial_equity else 0,
            "totalTrades": total_trades,
            "winTrades": win_trades,
            "lossTrades": loss_trades,
            "winrate": (win_trades / total_trades * 100) if total_trades else 0,
            "profitFactor": profit_factor,
            "avgTrade": sum(pnls) / total_trades if total_trades else 0,
            "avgWin": avg_win,
            "avgLoss": avg_loss,
            "expectancy": expectancy,
            "maxDrawdown": max_dd_abs,
            "maxDrawdownPct": max_dd_pct,
            "equityCurve": equity_curve,
        }

    # Run backtest
    result = run_backtest(run_cfg)

    # Filter trades by range
    filtered_trades = filter_by_range(result.get("trades", []), date_range)

    # Recompute summary
    recomputed = recompute_summary_fast(filtered_trades, props, run_cfg.get("initial_equity", 0))

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
    """
    Giữ thông tin cần thiết cho frontend, tránh trả về payload quá lớn.
    """
    out = {
        "symbol": run.get("symbol"),
        "timeframe": run.get("timeframe"),
        "params": run.get("params", {}),
        "summary": run.get("summary", {}),
        "strategyId": run.get("strategyId"),  # nếu có
        "rank": run.get("rank"),
    }
    curve = run.get("equityCurve") or []
    out["equityCurve"] = downsample_curve(curve, max_points=600)
    trades = run.get("trades") or []
    # giới hạn số trade để tránh Output too large
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


def ai_recommend(cfg: Dict[str, Any]) -> Dict[str, Any]:
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

    # Emit initial progress (preparing phase)
    if "AI_PROGRESS" in os.environ:
        print(json.dumps({"progress": 0, "total": 1, "phase": "preparing"}))
        sys.stdout.flush()

    top_n = cfg.get("topN", 50)
    # Build runs trước để biết total
    runs = build_runs(cfg)
    total = len(runs)

    # Emit progress after build_runs
    if "AI_PROGRESS" in os.environ:
        print(json.dumps({"progress": 0, "total": total, "phase": "running"}))
        sys.stdout.flush()
    props = cfg.get("properties", {}) or {}

    def build_costs(base_costs):
        costs = dict(base_costs or {})
        commission = props.get("commission", {}) or {}

        # commission: percent -> rate
        if "value" in commission:
            val = float(commission.get("value", 0))
            ctype = (commission.get("type") or "percent").lower()
            fee_rate = val / 100 if ctype == "percent" and val > 1 else val if ctype != "percent" else val / 100
            costs["fee"] = fee_rate
        # slippage: giữ nguyên đơn vị tick (giống TradingView)
        if props.get("slippage") is not None:
            costs["slippage"] = float(props.get("slippage"))
        return costs

    # Tính timeline chuẩn (tf nhỏ nhất) để overlay mượt giữa nhiều khung
    min_tf_minutes = min([tf_to_minutes(r["meta"]["timeframe"]) for r in runs]) if runs else 60

    # ===== PARALLEL EXECUTION =====
    # DON'T preload DataFrames here - let each worker load with LRU cache
    # This avoids expensive DataFrame serialization across processes
    # Each worker will load CSV once (cached via @lru_cache in data_loader.py)

    # Prepare run configs with injected properties (NO DataFrame)
    prepared_runs = []
    date_range = cfg.get("range")
    for run_cfg in runs:
        run_cfg = dict(run_cfg)
        # DON'T inject DataFrame - let worker load it (with cache)
        # Inject properties
        if "properties" not in run_cfg:
            run_cfg["properties"] = props
        else:
            merged_props = dict(props)
            merged_props.update(run_cfg.get("properties", {}))
            run_cfg["properties"] = merged_props
        if props.get("initialCapital") is not None:
            run_cfg["initial_equity"] = float(props["initialCapital"])
        run_cfg["costs"] = build_costs(run_cfg.get("costs", {}))
        prepared_runs.append((run_cfg, props, date_range, min_tf_minutes))

    # Run backtests using ThreadPoolExecutor with more workers
    # Increased workers for I/O-bound data loading
    results = []
    workers = max(4, (os.cpu_count() or 2) * 2)  # 2x CPU cores for I/O-bound work
    completed = 0
    last_progress_time = 0  # Throttle progress updates

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all jobs
        futures = {executor.submit(_run_single_backtest, args): idx for idx, args in enumerate(prepared_runs)}

        # Collect results as they complete
        for future in as_completed(futures):
            completed += 1
            # Progress emit (throttled to every 200ms to avoid spam)
            current_time = __import__('time').time()
            if "AI_PROGRESS" in os.environ and (current_time - last_progress_time >= 0.2 or completed == total):
                print(json.dumps({"progress": completed, "total": total}))
                sys.stdout.flush()
                last_progress_time = current_time

            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # Log error but continue with other runs
                if "AI_PROGRESS" in os.environ:
                    print(json.dumps({"progress": completed, "total": total}), file=sys.stderr)
                    sys.stderr.flush()

    # sort desc score
    results.sort(key=lambda x: x["summary"]["score"], reverse=True)

    # Apply filters & stability giống optimizer.optimize
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
    passed_runs.sort(key=lambda x: x["summary"]["score"], reverse=True)
    top = passed_runs[:top_n] if passed_runs else []
    best = passed_runs[0] if passed_runs else None

    passed_flag = bool(best)

    # Fallback: nếu không có best (do filter quá chặt), lấy best từ all theo score
    fallback = False
    if not best and results:
        top = results[:5]
        best = top[0]
        fallback = True

    comment = build_comment(best) if best else "No valid strategy found"
    alternatives = top[1:4] if len(top) > 1 else []

    # Prune payload để tránh quá lớn
    pruned_top = [prune_run(r) for r in top]
    pruned_best = prune_run(best) if best else None
    # giới hạn all để tránh phình payload nhưng vẫn đủ cho overlay/bảng (vd 200)
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
    }


if __name__ == "__main__":
    raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps({"error": "Config JSON required on stdin"}))
        sys.exit(1)

    cfg = json.loads(raw)
    output = ai_recommend(cfg)
    print(json.dumps(output))
