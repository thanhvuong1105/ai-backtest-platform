# engine/indicators.py
import pandas as pd
import numpy as np

def ema(series: pd.Series, length: int):
    return series.ewm(span=length, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    gain = pd.Series(gain).rolling(length).mean()
    loss = pd.Series(loss).rolling(length).mean()

    rs = gain / loss
    return 100 - (100 / (1 + rs))
