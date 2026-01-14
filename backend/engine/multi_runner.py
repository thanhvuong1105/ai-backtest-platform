from .backtest_engine import run_backtest
from .scoring import score_strategy
from .guards import equity_smoothness
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import logging
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


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

    # ═══════════════════════════════════════════════════════
    # PRE-PROCESS DATA FOR EACH UNIQUE SYMBOL/TIMEFRAME
    # This avoids repeated date filtering in each backtest
    # ═══════════════════════════════════════════════════════
    import pandas as pd

    indicator_cache = {}
    for (sym, tf), df in df_cache.items():
        # Get date range from first run with this symbol/tf
        sample_run = next((r for r in runs if r["meta"]["symbols"][0] == sym and r["meta"]["timeframe"] == tf), None)
        if sample_run:
            date_range = sample_run.get("range") or {}
            from_date = date_range.get("from")
            to_date = date_range.get("to")

            from_date_ts = pd.to_datetime(from_date) if from_date else None
            to_date_ts = pd.to_datetime(to_date) if to_date else None

            # Prepare df with date range filtering
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

            # Store processed df for all runs with this symbol/tf
            indicator_cache[(sym, tf)] = df_processed
            logger.debug(f"Pre-processed data for {sym} {tf}: {len(df_processed)} bars")

    # Update runs with pre-processed df and mark as preprocessed
    for r in runs:
        sym = r["meta"]["symbols"][0]
        tf = r["meta"]["timeframe"]
        key = (sym, tf)
        if key in indicator_cache:
            r["df"] = indicator_cache[key]
            r["_preprocessed"] = True

    results = []
    completed = 0
    # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid
    # serialization overhead and allow cancel flag checking
    # 4x CPU cores for I/O-bound work (configurable via env)
    max_workers_env = int(os.getenv("MAX_THREAD_WORKERS", 0))
    workers = max_workers_env if max_workers_env > 0 else max(8, (os.cpu_count() or 2) * 4)
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
