from .run_builder import build_runs
from .multi_runner import run_multi
from .filtering import filter_results
from .stability import stability_metrics, pass_stability
from collections import defaultdict
from typing import Callable, Dict, Optional


def apply_tf_agreement(passed, min_tf=2):
    """
    Giữ strategy params xuất hiện ở >= min_tf timeframe
    """
    bucket = defaultdict(list)

    for r in passed:
        key = (
            r["symbol"],
            tuple(sorted(r["params"].items()))
        )
        bucket[key].append(r)

    agreed = []
    for runs_same_param in bucket.values():
        if len(runs_same_param) >= min_tf:
            agreed.extend(runs_same_param)

    return agreed


def optimize(
    cfg,
    top_n=5,
    job_id: str = "",
    progress_cb: Optional[Callable[[str, int, int, str, Dict], None]] = None
):
    """
    Level 6 Optimizer (FULL)
    - Build grid
    - Multi-run backtest
    - Quality filter
    - Multi-timeframe agreement (6.3B)
    - Stability guard (6.3C)
    - Pick best / top

    Args:
        cfg: Optimization config
        top_n: Number of top results to return
        job_id: Job ID for progress tracking
        progress_cb: Optional callback(job_id, progress, total, status, extra)
    """

    # =========================
    # 1) Build all runs
    # =========================
    runs = build_runs(cfg)

    # =========================
    # 2) Run all backtests
    # =========================
    all_results = run_multi({"runs": runs}, job_id=job_id, progress_cb=progress_cb)

    # =========================
    # 3) Quality filters
    # =========================
    filters = cfg.get("filters", {})
    passed = filter_results(
        all_results,
        min_pnl=filters.get("minPnL", 0),
        min_trades=filters.get("minTrades", 10)
    )

    # =========================
    # 4) Multi-Timeframe Agreement
    # =========================
    min_tf = cfg.get("minTFAgree", 2)
    passed = apply_tf_agreement(passed, min_tf=min_tf)

    # =========================
    # 5) Stability Guard (6.3C)
    # =========================
    stability_rules = cfg.get("stability") or {
        "minMedianPF": 1.05,
        "minWorstPF": 0.95,
        "maxWorstDD": 45,
        "maxPFStd": 0.6
    }

    bucket = defaultdict(list)
    for r in passed:
        key = (
            r["symbol"],
            tuple(sorted(r["params"].items()))
        )
        bucket[key].append(r)

    stable = []
    for runs_same_param in bucket.values():
        summaries = [r["summary"] for r in runs_same_param]
        stab = stability_metrics(summaries)

        if pass_stability(stab, stability_rules):
            for r in runs_same_param:
                r["summary"]["stability"] = stab
            stable.extend(runs_same_param)

    passed = stable

    # =========================
    # 6) Sort & pick top
    # =========================
    passed.sort(key=lambda x: x["summary"]["score"], reverse=True)
    top = passed[:top_n] if passed else []

    # =========================
    # 7) Return
    # =========================
    return {
        "stats": {
            "totalRuns": len(all_results),
            "passedRuns": len(passed),
            "rejectedRuns": len(all_results) - len(passed)
        },
        "best": passed[0] if passed else None,
        "top": top,
        "all": all_results,
        "passed": passed,  # giữ danh sách đã qua filter/stability
    }
