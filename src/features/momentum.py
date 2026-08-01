# =============================================================================
# CONSTRUCTION OF MOMENTUM AND SHORT-TERM REVERSAL FACTORS
# =============================================================================


def compute_12_1_momentum(prices):
    """Calculate 12-1 momentum for each ticker.

    It measures the cumulative return from 12 months ago to one month ago,
    excluding the most recent month.
    """
    adj_close = prices["Adj Close"]

    return adj_close.shift(21) / adj_close.shift(252) - 1


def compute_short_term_reversal(prices):
    """Calculate the most recent one-month return for each ticker."""
    adj_close = prices["Adj Close"]

    return adj_close / adj_close.shift(21) - 1