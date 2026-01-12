from .backtest_engine import run_backtest
from .scoring import score_strategy
from .guards import equity_smoothness
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from typing import Callable, Dict, Optional


def _run_single(run_cfg):
    result = run_backtest(run_cfg)
    summary = result["summary"]
    smooth = equity_smoothness(result["equityCurve"])
    summary["smoothness"] = smooth
    summary["score"] = score_strategy(summary)
    return {
        "symbol": run_cfg["meta"]["symbols"][0],
        "timeframe": run_cfg["meta"]["timeframe"],
        "params": run_cfg.get("strategy", {}).get("params", {}),
        "summary": summary,
        "equityCurve": result["equityCurve"],
        "trades": result["trades"],
    }


def run_multi(
    cfg,
    job_id: str = "",
    progress_cb: Optional[Callable[[str, int, int, str, Dict], None]] = None
):
    """
    Chạy nhiều backtest (multi-run) với ThreadPoolExecutor để tăng tốc.

    Args:
        cfg: Config với key "runs" chứa list các run configs
        job_id: Job ID để track progress
        progress_cb: Callback function(job_id, progress, total, status, extra)
    """
    runs = cfg.get("runs")
    if runs is None:
        raise ValueError("cfg must contain 'runs'")

    total = len(runs)

    # preload dataframes theo symbol/tf nếu có range
    from .data_loader import load_csv
    df_cache = {}
    for r in runs:
        sym = r["meta"]["symbols"][0]
        tf = r["meta"]["timeframe"]
        key = (sym, tf)
        if key not in df_cache:
            df_cache[key] = load_csv(sym, tf)
        r["df"] = df_cache[key]

    results = []
    completed = 0
    # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid
    # serialization overhead and allow cancel flag checking
    workers = max(2, (os.cpu_count() or 2))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {executor.submit(_run_single, rc): idx for idx, rc in enumerate(runs)}
        for fut in as_completed(future_to_idx):
            try:
                res = fut.result()
                results.append(res)
            except Exception:
                # Log error but continue
                pass

            completed += 1

            # Call progress callback
            if progress_cb:
                # This will check cancel flag and raise InterruptedError if canceled
                progress_cb(job_id, completed, total, "running", {})

    results.sort(key=lambda x: x["summary"]["score"], reverse=True)
    return results
