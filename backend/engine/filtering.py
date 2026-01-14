def filter_results(
    results,
    min_pnl=0.0,
    min_trades=10,
    **kwargs  # Accept but ignore old params (min_pf, max_dd) for backwards compatibility
):
    """
    Lọc các strategy run kém chất lượng.

    Chỉ lọc theo:
    - PnL (Net Profit) > min_pnl (mặc định > 0)
    - Min trades (mặc định >= 10)

    Args:
        results: List of backtest results
        min_pnl: Minimum net profit required (default 0 = profitable)
        min_trades: Minimum number of trades (default 10)
        **kwargs: Ignored (backwards compatibility for min_pf, max_dd)

    Returns:
        List of results that passed filters
    """

    passed = []

    for r in results:
        s = r.get("summary", {})

        # Filter by Net Profit (PnL)
        if s.get("netProfit", 0) <= min_pnl:
            continue

        # Filter by minimum trades
        if s.get("totalTrades", 0) < min_trades:
            continue

        passed.append(r)

    return passed
