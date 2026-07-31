# =============================================================================
# CONSTRUCTION OF RETURNS
# =============================================================================


import numpy as np


def compute_simple_returns(prices):
    """Calculate daily simple returns for each ticker."""
    adj_close = prices["Adj Close"]
    return adj_close.pct_change()


def compute_log_returns(prices):
    """Calculate daily logarithmic returns for each ticker."""
    adj_close = prices["Adj Close"]
    return np.log(adj_close / adj_close.shift(1))


def compute_cumulative_returns(prices):
    """Calculate cumulative simple returns for each ticker."""
    returns = compute_simple_returns(prices)
    return (1 + returns).cumprod() - 1