def filter_results(
    results,
    min_pf=1.0,
    min_trades=30,
    max_dd=40.0
):
    """
    Lọc các strategy run kém chất lượng
    """

    passed = []

    for r in results:
        s = r["summary"]

        if s.get("profitFactor", 0) < min_pf:
            continue

        if s.get("totalTrades", 0) < min_trades:
            continue

        if s.get("maxDrawdownPct", 100) > max_dd:
            continue

        passed.append(r)

    return passed
