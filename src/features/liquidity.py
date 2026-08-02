import numpy as np


def compute_amihud_illiquidity(prices, simple_returns, window=21):
    """Calculate the rolling Amihud illiquidity ratio for each ticker.

    The daily ratio is the absolute simple return divided by dollar volume.
    Dollar volume is calculated as close price times trading volume.
    """
    close = prices["Adj Close"]
    volume = prices["Volume"]

    dollar_volume = close * volume
    dollar_volume = dollar_volume.replace(0, np.nan)

    daily_amihud = simple_returns.abs() / dollar_volume

    return daily_amihud.rolling(
        window=window,
        min_periods=15,
    ).mean()