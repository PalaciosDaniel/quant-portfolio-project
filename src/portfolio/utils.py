import numpy as np 
import pandas as pd

BASE_TRANSACTION_COST = 0.0015       
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.0

# =============================================================================
# Drawdown Metrics
# =============================================================================

def calculate_drawdown_metrics(returns):
    """Compute Maximum Drawdown, Average Peak-to-Recovery Drawdown,

    and Maximum Underwater Duration.
    """

    wealth = (1.0 + returns).cumprod()
    running_max = wealth.cummax()
    drawdown = (wealth / running_max) - 1.0

    # -------------------------------------------------------------------------
    # Maximum Drawdown
    # -------------------------------------------------------------------------
    max_dd = drawdown.min()

    # -------------------------------------------------------------------------
    # Average Peak-to-Recovery Drawdown (Episode Troughs)
    # -------------------------------------------------------------------------
    # Identify days at all-time highs
    is_at_high = drawdown == 0.0

    # Assign a unique identifier to each drawdown period
    episode_id = is_at_high.cumsum()

    # Filter only underwater days (where drawdown is strictly negative)
    underwater = drawdown[drawdown < 0.0]

    if len(underwater) > 0:
        # Group by episode and extract the lowest point (trough) of each one
        episode_grouping = episode_id.loc[underwater.index]
        episode_troughs = underwater.groupby(episode_grouping).min()

        # Average of the trough drawdowns across all episodes
        avg_dd = episode_troughs.mean()
    else:
        avg_dd = 0.0

    # -------------------------------------------------------------------------
    # Maximum Underwater Duration
    # -------------------------------------------------------------------------
    max_duration = 0
    current_duration = 0

    for dd in drawdown:
        if dd < 0:
            current_duration += 1
        else:
            max_duration = max(max_duration, current_duration)
            current_duration = 0

    max_duration = max(max_duration, current_duration)

    return (max_dd, avg_dd, max_duration)


def calculate_cagr(returns):

    years = (
        len(returns)
        / TRADING_DAYS_PER_YEAR
    )

    if years <= 0:
        return np.nan

    cumulative_return = (
        1.0 + returns
    ).prod()

    return (
        cumulative_return ** (1.0 / years)
    ) - 1.0


def calculate_sharpe(returns):

    annualized_return = calculate_cagr(
        returns
    )

    annualized_volatility = (
        returns.std(ddof=1)
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    if (
        pd.isna(annualized_volatility)
        or annualized_volatility == 0
    ):
        return np.nan

    return (
        annualized_return - RISK_FREE_RATE
    ) / annualized_volatility


def calculate_sortino(returns):

    cagr = calculate_cagr(
        returns
    )

    target_return = (
        RISK_FREE_RATE
        / TRADING_DAYS_PER_YEAR
    )

    downside_diff = (
        returns - target_return
    )

    downside_returns = downside_diff[
        downside_diff < 0
    ]

    if len(downside_returns) == 0:
        return np.nan

    downside_deviation = (
        np.sqrt(
            np.sum(
                downside_returns ** 2
            )
            / len(returns)
        )
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    if (
        pd.isna(downside_deviation)
        or downside_deviation <= 0
    ):
        return np.nan

    return (
        cagr - RISK_FREE_RATE
    ) / downside_deviation



# =============================================================================
# Helper Functions
# =============================================================================

def calculate_benchmark_metrics(
    returns: pd.Series,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """
    Computes annualized performance and risk metrics for a benchmark return series.

    Parameters
    ----------
    returns : pd.Series
        Daily return series of the benchmark index (e.g., S&P 500).
    trading_days : int, default 252
        Number of trading days in a year for annualization.

    Returns
    -------
    dict
        Dictionary containing standardized performance and risk metrics.
    """
    # Clean input series
    clean_returns = returns.dropna().astype(float)

    if clean_returns.empty:
        raise ValueError("Provided return series is empty after removing NaNs.")

    # Calculate annualized return and volatility
    cagr = calculate_cagr(clean_returns)
    daily_vol = clean_returns.std(ddof=1)
    ann_vol = daily_vol * np.sqrt(trading_days)

    # Compute risk-adjusted ratios passing only the required `returns` argument
    sharpe = calculate_sharpe(clean_returns)
    sortino = calculate_sortino(clean_returns)
    max_dd, _, _ = calculate_drawdown_metrics(clean_returns)

    # Compute Calmar ratio (CAGR over absolute maximum drawdown)
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "role": "Benchmark",
        "model": "S&P 500",
        "portfolio": "buy_and_hold",
        "frequency_days": np.nan,
        "frequency_label": "N/A",
        "CAGR": cagr,
        "annualized_volatility": ann_vol,
        "annualized_turnover": 0.0,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "maximum_drawdown": max_dd,
    }
