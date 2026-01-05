# engine/strategies/ema_cross.py
from engine.strategies.base import StrategyBase
from engine.indicators import ema

class EMACrossStrategy(StrategyBase):
    def prepare_indicators(self):
        fast = self.params["emaFast"]
        slow = self.params["emaSlow"]

        self.df["emaFast"] = ema(self.df["close"], fast)
        self.df["emaSlow"] = ema(self.df["close"], slow)

    def check_entry(self, i):
        if i == 0:
            return False

        prev = self.df.iloc[i - 1]
        curr = self.df.iloc[i]

        return (
            prev["emaFast"] <= prev["emaSlow"]
            and curr["emaFast"] > curr["emaSlow"]
        )

    def check_exit(self, i, position):
        if i == 0:
            return False

        prev = self.df.iloc[i - 1]
        curr = self.df.iloc[i]

        return (
            prev["emaFast"] >= prev["emaSlow"]
            and curr["emaFast"] < curr["emaSlow"]
        )
