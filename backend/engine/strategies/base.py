# engine/strategies/base.py

class StrategyBase:
    def __init__(self, df, params):
        self.df = df
        self.params = params

    def prepare_indicators(self):
        raise NotImplementedError

    def check_entry(self, i):
        raise NotImplementedError

    def check_exit(self, i, position):
        raise NotImplementedError
