def score_strategy(summary):
    pf = summary.get("profitFactor", 0)
    wr = summary.get("winrate", 0)
    dd = summary.get("maxDrawdownPct", 0)
    smooth = summary.get("smoothness", 0)

    # clamp PF contribution để tránh phóng đại
    pf_capped = min(pf, 100.0)

    score = (
        pf_capped * 2
        + wr / 100
        + smooth
        - dd / 100
    )

    return round(score, 4)
