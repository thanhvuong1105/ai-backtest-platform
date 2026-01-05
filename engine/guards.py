import math


def equity_smoothness(equity_curve):
    """
    Đo độ mượt của equity curve
    Trả về số càng cao càng mượt
    """

    if len(equity_curve) < 3:
        return 0.0

    returns = []
    for i in range(1, len(equity_curve)):
        prev = equity_curve[i - 1]["equity"]
        cur = equity_curve[i]["equity"]
        if prev > 0:
            returns.append((cur - prev) / prev)

    if not returns:
        return 0.0

    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    std = math.sqrt(variance)

    # Smoothness score: thấp std = mượt
    return round(1 / (1 + std), 6)
