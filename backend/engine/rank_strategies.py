# engine/rank_strategies.py
from optimize_ema import optimize_ema

def score_strategy(summary):
    return (
        summary["netProfit"] * 0.4
        + summary["profitFactor"] * 100 * 0.3
        + summary["winrate"] * 0.2
        - summary["maxDrawdownPct"] * 0.3
    )


def rank_strategies(results):
    ranked = []

    for r in results:
        s = r["summary"]

        # ===== FILTER =====
        if s["maxDrawdownPct"] > 200:
            continue
        if s["profitFactor"] < 0.9:
            continue
        if s["totalTrades"] < 80:
            continue

        score = score_strategy(s)

        ranked.append({
            "params": r["params"],
            "summary": s,
            "score": round(score, 2)
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


if __name__ == "__main__":
    print("Running EMA optimization...")
    results = optimize_ema()

    print("\nRanking strategies...")
    ranked = rank_strategies(results)

    print("\n===== TOP 10 STRATEGIES =====")
    for i, r in enumerate(ranked[:10], start=1):
        s = r["summary"]
        print(
            f"{i}. EMA {r['params']['emaFast']}/{r['params']['emaSlow']} | "
            f"Score {r['score']} | "
            f"PNL {s['netProfit']:.2f} | "
            f"DD {s['maxDrawdownPct']:.1f}% | "
            f"PF {s['profitFactor']:.2f} | "
            f"WR {s['winrate']:.1f}%"
        )
