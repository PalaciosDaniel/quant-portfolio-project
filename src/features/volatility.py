# =============================================================================
# CONSTRUCTION OF VOLATILITY FACTORS
# =============================================================================

def compute_rolling_volatility(log_returns, window=252):
    """Calculate rolling volatility using logarithmic returns."""
    return log_returns.rolling(window=window).std()


def compute_downside_volatility(log_returns, window=252):
    """Calculate rolling volatility using only negative logarithmic returns."""
    negative_returns = log_returns.where(log_returns < 0)

    return negative_returns.rolling(
        window=window,
        min_periods=21,
    ).std()


def compute_upside_volatility(log_returns, window=252):
    """Calculate rolling volatility using only positive logarithmic returns."""
    positive_returns = log_returns.where(log_returns > 0)

    return positive_returns.rolling(
        window=window,
        min_periods=21,
    ).std()


def compute_low_volatility_factor(log_returns, window=252):
    """Calculate the low-volatility factor as negative rolling volatility."""
    rolling_volatility = compute_rolling_volatility(
        log_returns,
        window=window,
    )

    return -rolling_volatility