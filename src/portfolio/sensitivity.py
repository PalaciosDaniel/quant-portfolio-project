import numpy as np
import pandas as pd

from src.portfolio.metrics import (
    calculate_cagr,
    calculate_drawdown_metrics,
    calculate_sharpe,
    calculate_sortino,
)


def run_frequency_sensitivity_execution(
    weights_raw: pd.DataFrame,
    asset_returns: pd.DataFrame,
    all_trading_dates: pd.Index,
    rebalancing_frequencies: list[int],
    verbose: bool = True,
) -> pd.DataFrame:
    """Executes frequency sensitivity analysis across various rebalancing grids.

    Parameters
    ----------
    weights_raw : pd.DataFrame
        Raw target weights containing ['date', 'model', 'portfolio', 'ticker',
        'weight'].
    asset_returns : pd.DataFrame
        Asset daily returns matrix (dates as index, tickers as columns).
    all_trading_dates : pd.Index
        Complete trading calendar index.
    rebalancing_frequencies : list[int]
        List of rebalancing step intervals (e.g., [1, 5, 10, 21]).
    verbose : bool, default True
        If True, prints execution audit details.

    Returns
    -------
    pd.DataFrame
        Consolidated gross returns across all frequencies and strategies.
    """
    frequency_results = []

    for freq in rebalancing_frequencies:
        # Construct & Filter Rebalancing Calendar
        all_weight_dates = (
            weights_raw["date"]
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )
        rebalancing_dates = all_weight_dates.iloc[::freq]

        weights_df = weights_raw[
            weights_raw["date"].isin(rebalancing_dates)
        ].copy()
        rebal_dates = sorted(weights_df["date"].unique())

        # Validate Rebalancing Frequency Grid
        rebal_positions = [
            all_trading_dates.get_loc(date) for date in rebal_dates
        ]
        rebal_intervals = np.diff(rebal_positions)

        assert np.all(rebal_intervals == freq), (
            f"Rebalancing dates do not follow the expected {freq}-day"
            " frequency."
        )

        # Execution Engine per Model / Portfolio combination
        gross_returns_list = []
        grouped_weights = weights_df.groupby(["model", "portfolio"])

        for (model, portfolio), group in grouped_weights:
            # Pivot target weight matrix
            weight_matrix = group.pivot(
                index="date", columns="ticker", values="weight"
            ).fillna(0.0)

            # Prevent look-ahead bias (shift 1 day and forward-fill)
            executed_weights = (
                weight_matrix.reindex(all_trading_dates).shift(1).ffill()
            )

            # Trim dates prior to first execution date
            first_rebalance_pos = all_trading_dates.get_loc(rebal_dates[0])
            first_execution_pos = first_rebalance_pos + 1
            executed_weights = executed_weights.iloc[first_execution_pos:]

            # Align asset returns
            returns_subset = asset_returns.reindex(
                index=executed_weights.index, columns=executed_weights.columns
            )

            # Daily gross portfolio returns
            portfolio_daily_returns = (executed_weights * returns_subset).sum(
                axis=1
            )

            # Build result DataFrame
            result = pd.DataFrame(
                {
                    "date": executed_weights.index,
                    "model": model,
                    "portfolio": portfolio,
                    "rebalancing_frequency": freq,
                    "gross_return": portfolio_daily_returns.values,
                }
            )

            result["cumulative_return"] = (
                1.0 + result["gross_return"]
            ).cumprod() - 1.0
            gross_returns_list.append(result)

        # Consolidate Current Frequency
        frequency_results.append(
            pd.concat(gross_returns_list, ignore_index=True)
        )

    # Consolidation — All Frequencies
    frequency_gross_returns = (
        pd.concat(frequency_results, ignore_index=True)[
            [
                "date",
                "model",
                "portfolio",
                "rebalancing_frequency",
                "gross_return",
                "cumulative_return",
            ]
        ]
        .sort_values(
            ["rebalancing_frequency", "date", "model", "portfolio"]
        )
        .reset_index(drop=True)
    )

    # Execution Audit
    duplicates = frequency_gross_returns.duplicated(
        subset=["date", "model", "portfolio", "rebalancing_frequency"]
    ).sum()

    assert (
        duplicates == 0
    ), f"Found {duplicates:,} duplicated observations in frequency sensitivity."

    if verbose:
        print("=" * 80)
        print("REBALANCING FREQUENCY SENSITIVITY — EXECUTION AUDIT")
        print("=" * 80)
        print(f"✓ Frequencies tested = {rebalancing_frequencies}")
        print(f"✓ Total observations = {len(frequency_gross_returns):,}")
        print(
            f"✓ Unique frequencies ="
            f" {frequency_gross_returns['rebalancing_frequency'].nunique()}"
        )
        print(
            f"✓ Unique portfolios ="
            f" {frequency_gross_returns['portfolio'].nunique()}"
        )
        print(
            f"✓ Unique models = {frequency_gross_returns['model'].nunique()}"
        )
        print(
            f"✓ Date range = {frequency_gross_returns['date'].min().date()} →"
            f" {frequency_gross_returns['date'].max().date()}"
        )
        print(
            f"✓ Missing gross returns ="
            f" {frequency_gross_returns['gross_return'].isna().sum():,}"
        )
        print(f"✓ Duplicate observations = {duplicates:,}")
        print("=" * 80)
        print("\nObservations by frequency:")
        print(
            frequency_gross_returns["rebalancing_frequency"]
            .value_counts()
            .sort_index()
        )

    return frequency_gross_returns

def run_frequency_sensitivity_engine(
    weights_raw: pd.DataFrame,
    asset_returns: pd.DataFrame,
    all_trading_dates: pd.Index,
    rebalancing_frequencies: dict[int, str],
    base_transaction_cost: float = 0.0015,
    trading_days_per_year: int = 252,
    risk_free_rate: float = 0.0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluates strategy performance and turnover sensitivity across various rebalancing frequencies.

    Parameters
    ----------
    weights_raw : pd.DataFrame
        Raw target weights containing ['date', 'model', 'portfolio', 'ticker',
        'weight'].
    asset_returns : pd.DataFrame
        Asset daily returns matrix (dates as index, tickers as columns).
    all_trading_dates : pd.Index
        Complete trading calendar index.
    rebalancing_frequencies : dict[int, str]
        Dictionary mapping frequency in trading days to its label (e.g., {5:
        "Weekly", 21: "Monthly"}).
    base_transaction_cost : float, default 0.0015
        Transaction cost factor per unit of turnover (15 bps).
    trading_days_per_year : int, default 252
        Number of trading days per year used for annualization.
    risk_free_rate : float, default 0.0
        Annualized risk-free rate for ratio calculations if required by external
        functions.
    verbose : bool, default True
        If True, prints execution audit details.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - frequency_sensitivity: Summary performance metrics per
        frequency/strategy.
        - frequency_turnover: Detailed turnover time series across dates and
        frequencies.
    """
    frequency_results = []
    frequency_turnover = []

    all_weight_dates = (
        weights_raw["date"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    for frequency, frequency_label in rebalancing_frequencies.items():
        # 1. Construct Rebalancing Calendar
        rebalancing_dates = all_weight_dates.iloc[::frequency]

        weights_df = weights_raw[
            weights_raw["date"].isin(rebalancing_dates)
        ].copy()
        rebal_dates = sorted(weights_df["date"].unique())

        # 2. Validate Rebalancing Calendar
        rebal_positions = [
            all_trading_dates.get_loc(date) for date in rebal_dates
        ]
        rebal_intervals = np.diff(rebal_positions)

        assert np.all(rebal_intervals == frequency), (
            f"Invalid {frequency}-day rebalancing calendar."
        )

        # 3. Process Each Portfolio
        grouped_weights = weights_df.groupby(["model", "portfolio"])

        for (model, portfolio), group in grouped_weights:
            # Target Weight Matrix
            weight_matrix = (
                group.pivot(index="date", columns="ticker", values="weight")
                .fillna(0.0)
                .reindex(rebalancing_dates)
                .fillna(0.0)
            )

            # Turnover
            previous_weights = weight_matrix.shift(1)
            turnover = (
                0.5 * (weight_matrix - previous_weights).abs().sum(axis=1)
            )

            # First formation has no previous portfolio
            turnover = turnover.iloc[1:]

            # Map Rebalance Dates to Execution Dates
            execution_dates = [
                all_trading_dates[all_trading_dates.get_loc(date) + 1]
                for date in turnover.index
            ]

            turnover_series = pd.Series(
                turnover.values,
                index=execution_dates,
                name="turnover",
            )

            # Executed Buy-and-Hold Weights
            executed_weights = (
                weight_matrix.reindex(all_trading_dates).shift(1).ffill()
            )

            first_rebalance = rebal_dates[0]
            first_rebalance_pos = all_trading_dates.get_loc(first_rebalance)
            first_execution_pos = first_rebalance_pos + 1

            executed_weights = executed_weights.iloc[first_execution_pos:]

            # Align Asset Returns
            returns_subset = asset_returns.reindex(
                index=executed_weights.index,
                columns=executed_weights.columns,
            ).fillna(0.0)

            # Gross & Net Portfolio Returns
            gross_returns = (executed_weights * returns_subset).sum(axis=1)

            transaction_cost = (
                turnover_series.reindex(gross_returns.index).fillna(0.0)
                * base_transaction_cost
            )

            net_returns = gross_returns - transaction_cost

            # Performance Metrics using imported functions
            cagr = calculate_cagr(net_returns)
            annualized_volatility = net_returns.std(ddof=1) * np.sqrt(
                trading_days_per_year
            )
            sharpe = calculate_sharpe(net_returns)
            sortino = calculate_sortino(net_returns)

            (
                maximum_drawdown,
                average_drawdown,
                maximum_underwater_duration_days,
            ) = calculate_drawdown_metrics(net_returns)

            # Annualized Turnover
            mean_turnover = turnover_series.mean()
            rebalancings_per_year = trading_days_per_year / frequency
            annualized_turnover = mean_turnover * rebalancings_per_year

            # Cumulative Returns
            cumulative_gross_return = (1.0 + gross_returns).prod() - 1.0
            cumulative_net_return = (1.0 + net_returns).prod() - 1.0
            cumulative_transaction_cost = transaction_cost.sum()

            # Store Performance Result
            frequency_results.append(
                {
                    "frequency_days": frequency,
                    "frequency_label": frequency_label,
                    "model": model,
                    "portfolio": portfolio,
                    "CAGR": cagr,
                    "annualized_turnover": annualized_turnover,
                    "mean_turnover": mean_turnover,
                    "cumulative_gross_return": cumulative_gross_return,
                    "cumulative_net_return": cumulative_net_return,
                    "cumulative_transaction_cost": cumulative_transaction_cost,
                    "annualized_volatility": annualized_volatility,
                    "Sharpe": sharpe,
                    "Sortino": sortino,
                    "maximum_drawdown": maximum_drawdown,
                    "average_drawdown": average_drawdown,
                    "maximum_underwater_duration_days": (
                        maximum_underwater_duration_days
                    ),
                }
            )

            # Store Turnover Series
            turnover_frequency_df = pd.DataFrame(
                {
                    "date": turnover_series.index,
                    "frequency_days": frequency,
                    "frequency_label": frequency_label,
                    "model": model,
                    "portfolio": portfolio,
                    "turnover": turnover_series.values,
                }
            )

            frequency_turnover.append(turnover_frequency_df)

    # Consolidate Results
    frequency_sensitivity = (
        pd.DataFrame(frequency_results)
        .sort_values(["frequency_days", "model", "portfolio"])
        .reset_index(drop=True)
    )

    frequency_turnover = (
        pd.concat(frequency_turnover, ignore_index=True)
        .sort_values(["frequency_days", "model", "portfolio", "date"])
        .reset_index(drop=True)
    )

    # Audit
    if verbose:
        print("\n" + "=" * 80)
        print("REBALANCING FREQUENCY SENSITIVITY — AUDIT")
        print("=" * 80)
        print(f"✓ Frequencies tested = {list(rebalancing_frequencies.keys())}")
        print(
            "✓ Total strategy-frequency observations ="
            f" {len(frequency_sensitivity):,}"
        )
        print(
            f"✓ Unique models = {frequency_sensitivity['model'].nunique()}"
        )
        print(
            f"✓ Unique portfolios ="
            f" {frequency_sensitivity['portfolio'].nunique()}"
        )
        print(
            f"✓ Missing CAGR = {frequency_sensitivity['CAGR'].isna().sum():,}"
        )
        print(
            f"✓ Missing Sharpe ="
            f" {frequency_sensitivity['Sharpe'].isna().sum():,}"
        )
        print(
            "✓ Missing turnover ="
            f" {frequency_sensitivity['annualized_turnover'].isna().sum():,}\n"
        )

        for frequency in rebalancing_frequencies:
            rebalancing_dates = all_weight_dates.iloc[::frequency]
            n_rebalances = len(rebalancing_dates)
            print(
                f" {frequency:>2} trading days → {n_rebalances:>3} rebalancings"
            )

        print("=" * 80)

    return frequency_sensitivity, frequency_turnover