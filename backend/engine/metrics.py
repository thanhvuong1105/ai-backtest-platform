# engine/metrics.py
def calculate_metrics(trades, equity_curve, initial_equity):
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
            "maxDrawdownPct": 0
        }

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    total_trades = len(pnls)
    win_trades = len(wins)
    loss_trades = len(losses)

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    final_equity = equity_curve[-1]["equity"]

    # Drawdown calculation
    # Track peak equity and calculate drawdown from peak
    peak = initial_equity
    max_dd_absolute = 0  # Absolute dollar amount
    max_dd_pct = 0  # Percentage from peak

    for e in equity_curve:
        current_equity = e["equity"]
        peak = max(peak, current_equity)

        # Drawdown is (current - peak) / peak
        dd_absolute = current_equity - peak
        if peak > 0:
            dd_pct = abs(dd_absolute) / peak * 100
        else:
            dd_pct = 0

        # Track maximum drawdown
        if dd_absolute < max_dd_absolute:
            max_dd_absolute = dd_absolute
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

    avg_win = gross_profit / win_trades if win_trades else 0
    avg_loss = sum(losses) / loss_trades if loss_trades else 0
    expectancy = (win_trades / total_trades) * avg_win + (loss_trades / total_trades) * avg_loss

    # Tránh giá trị Infinity làm hỏng JSON và clamp để không phóng đại score
    raw_pf = gross_profit / gross_loss if gross_loss else 1e9
    profit_factor = min(raw_pf, 1e6)  # clamp

    return {
        "initialEquity": initial_equity,
        "finalEquity": final_equity,
        "netProfit": final_equity - initial_equity,
        "netProfitPct": (final_equity - initial_equity) / initial_equity * 100,

        "totalTrades": total_trades,
        "winTrades": win_trades,
        "lossTrades": loss_trades,
        "winrate": win_trades / total_trades * 100,

        "profitFactor": profit_factor,
        "avgTrade": sum(pnls) / total_trades,
        "avgWin": avg_win,
        "avgLoss": avg_loss,
        "expectancy": expectancy,

        "maxDrawdown": max_dd_absolute,
        "maxDrawdownPct": max_dd_pct
    }
