from .backtest_engine import run_backtest
from .scoring import score_strategy
from .guards import equity_smoothness
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


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


def run_multi(cfg):
    """
    Chạy nhiều backtest (multi-run) với multiprocessing để tăng tốc.
    """
    runs = cfg.get("runs")
    if runs is None:
        raise ValueError("cfg must contain 'runs'")

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
    workers = max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {executor.submit(_run_single, rc): idx for idx, rc in enumerate(runs)}
        for fut in as_completed(future_to_idx):
            res = fut.result()
            results.append(res)

    results.sort(key=lambda x: x["summary"]["score"], reverse=True)
    return results
