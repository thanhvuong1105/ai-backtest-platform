import statistics as stats

def stability_metrics(window_summaries):
    """
    window_summaries: list[summary] của cùng 1 (symbol, params) trên các window
    """
    if not window_summaries:
        return None

    pfs = [s.get("profitFactor", 0) for s in window_summaries]
    dds = [s.get("maxDrawdownPct", 100) for s in window_summaries]
    wrs = [s.get("winrate", 0) for s in window_summaries]

    return {
        "medianPF": round(stats.median(pfs), 4),
        "worstPF": round(min(pfs), 4),
        "pfStd": round(stats.pstdev(pfs) if len(pfs) > 1 else 0.0, 4),
        "worstDD": round(max(dds), 4),
        "medianWR": round(stats.median(wrs), 2),
        "windows": len(window_summaries)
    }


def pass_stability(stab, rules):
    """
    rules: dict các ngưỡng stability
    """
    if stab is None:
        return False

    if stab["medianPF"] < rules.get("minMedianPF", 1.0):
        return False
    if stab["worstPF"] < rules.get("minWorstPF", 0.9):
        return False
    if stab["worstDD"] > rules.get("maxWorstDD", 45):
        return False
    if stab["pfStd"] > rules.get("maxPFStd", 0.6):
        return False

    return True
