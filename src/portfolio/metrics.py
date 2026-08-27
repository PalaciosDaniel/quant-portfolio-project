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
# PORTFOLIO PERFORMANCE EVALUATION
# =============================================================================

def calculate_performance_metrics(
    net_returns: pd.DataFrame,
    return_col: str = "net_return_base",
    verbose: bool = True,
) -> pd.DataFrame:
    """Calculates and validates consolidated performance metrics for each strategy

    (model x portfolio) using dedicated metric helper functions.
    """
    df = net_returns.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["model", "portfolio", "date"]).reset_index(drop=True)

    performance_results = []

    for (model, portfolio), group in df.groupby(["model", "portfolio"]):
        returns = group[return_col].dropna().astype(float)

        if returns.empty:
            continue

        # Basic counts & cumulative return
        observations = len(returns)
        cumulative_return = (1.0 + returns).prod() - 1.0

        # Performance & Risk metrics via helper functions
        cagr = calculate_cagr(returns)
        annualized_volatility = returns.std(ddof=1) * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )
        sharpe_ratio = calculate_sharpe(returns)
        sortino_ratio = calculate_sortino(returns)

        # Drawdown Metrics & Calmar Ratio
        (
            maximum_drawdown,
            average_drawdown,
            maximum_underwater_duration,
        ) = calculate_drawdown_metrics(returns)

        calmar_ratio = (
            cagr / abs(maximum_drawdown) if maximum_drawdown < 0 else np.nan
        )

        performance_results.append(
            {
                "model": model,
                "portfolio": portfolio,
                "observations": observations,
                "cumulative_return": cumulative_return,
                "CAGR": cagr,
                "Ann_vol": annualized_volatility,
                "Sharpe": sharpe_ratio,
                "Sortino": sortino_ratio,
                "Calmar": calmar_ratio,
                "max_DD": maximum_drawdown,
                "average_drawdown": average_drawdown,
                "max_underwater_in_days": maximum_underwater_duration,
            }
        )

    # Consolidation
    performance_metrics = pd.DataFrame(performance_results)
    if not performance_metrics.empty:
        performance_metrics = performance_metrics.sort_values(
            "Sharpe", ascending=False
        ).reset_index(drop=True)

    # Integrated Assertions / Validations
    assert (
        performance_metrics["model"].nunique() == df["model"].nunique()
    ), "Mismatch in the number of evaluated models."
    assert (
        performance_metrics["portfolio"].nunique()
        == df["portfolio"].nunique()
    ), "Mismatch in the number of evaluated portfolios."
    assert (
        performance_metrics[
            [
                "CAGR",
                "Ann_vol",
                "Sharpe",
                "Sortino",
                "max_DD",
            ]
        ]
        .notna()
        .all()
        .all()
    ), "Critical performance metrics contain null values (NaN)."

    # Output / Audit Printouts
    if verbose:
        print("=" * 80)
        print("3.1 — NET PORTFOLIO PERFORMANCE")
        print("=" * 80)
        print(f"✓ Strategies evaluated = {len(performance_metrics):,}")
        print(f"✓ Models evaluated = {performance_metrics['model'].nunique():,}")
        print(
            f"✓ Date range = {df['date'].min().date()} → {df['date'].max().date()}"
        )
        print(
            f"✓ Missing performance metrics = {performance_metrics.isna().sum().sum():,}\n"
        )

        print(
            performance_metrics[
                [
                    "model",
                    "portfolio",
                    "CAGR",
                    "Ann_vol",
                    "Sharpe",
                    "Sortino",
                    "Calmar",
                    "max_DD",
                    "max_underwater_in_days",
                ]
            ].to_string(
                index=False,
                formatters={
                    "CAGR": lambda x: f"{x:.2%}",
                    "Ann_vol": lambda x: f"{x:.2%}",
                    "Sharpe": lambda x: f"{x:.3f}",
                    "Sortino": lambda x: f"{x:.3f}",
                    "Calmar": lambda x: f"{x:.3f}",
                    "max_DD": lambda x: f"{x:.2%}",
                },
            )
        )

    return performance_metrics


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


def calculate_benchmark_b_metrics(
    returns: pd.Series,
    ann_turnover_val: float,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """Computes performance and risk metrics for Benchmark B (Momentum Top 10 Constrained)."""
    clean_returns = returns.dropna().astype(float)

    if clean_returns.empty:
        raise ValueError("Return series is empty after dropping NaNs.")

    # Core performance metrics
    cagr = calculate_cagr(clean_returns)
    daily_vol = clean_returns.std(ddof=1)
    ann_vol = daily_vol * np.sqrt(trading_days)

    # Risk-adjusted metrics
    sharpe = calculate_sharpe(clean_returns)
    sortino = calculate_sortino(clean_returns)
    max_dd, _, _ = calculate_drawdown_metrics(clean_returns)
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "role": "Benchmark B",
        "model": "Momentum Top 10",
        "portfolio": "long_only_max_cap_5pct",
        "frequency_days": 21,
        "CAGR": cagr,
        "annualized_volatility": ann_vol,
        "annualized_turnover": ann_turnover_val,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "maximum_drawdown": max_dd,
    }

# =============================================================================
# HELPER FUNCTIONS (Performance by Model, Portfolio type and universe)
# =============================================================================


def extract_weighting_scheme(portfolio):
    """
    Extract the portfolio weighting philosophy from the portfolio name.
    """
    portfolio = portfolio.lower()

    if "equal_weight" in portfolio:
        return "Equal Weight"

    if "inverse_volatility" in portfolio:
        return "Inverse Volatility"

    if "risk_parity" in portfolio:
        return "Risk Parity"

    if "signal_weighting" in portfolio:
        return "Signal Weighting"

    if "maximum_sharpe" in portfolio:
        return "Maximum Sharpe"

    return "Other"


def extract_universe(portfolio):
    """
    Extract the investment universe from the portfolio name.
    """
    portfolio = portfolio.lower()

    if "top_10" in portfolio:
        return "Top 10%"

    if "top_20" in portfolio:
        return "Top 20%"

    if "top_30" in portfolio:
        return "Top 30%"

    # Baseline portfolios without quantile selection
    return "Full / Baseline"
