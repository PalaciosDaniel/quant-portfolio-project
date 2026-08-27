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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executes frequency sensitivity analysis across various rebalancing grids using

    exact buy-and-hold daily weight drift between rebalances.

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
        List of rebalancing step intervals (e.g., [5, 10, 21, 42, 63]).
    verbose : bool, default True
        If True, prints execution audit details.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - Consolidated daily gross returns across all frequencies and strategies.
        - Consolidated turnover records per rebalance event across all
        frequencies.
    """
    all_returns_list = []
    all_turnover_list = []

    for freq in rebalancing_frequencies:
        # 1. Calendar Construction
        all_weight_dates = (
            weights_raw["date"]
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )
        rebalancing_dates = all_weight_dates.iloc[::freq].reset_index(
            drop=True
        )

        weights_df = weights_raw[
            weights_raw["date"].isin(rebalancing_dates)
        ].copy()
        rebal_dates = sorted(weights_df["date"].unique())

        # Validate frequency consistency
        rebal_positions = [
            all_trading_dates.get_loc(date) for date in rebal_dates
        ]
        rebal_intervals = np.diff(rebal_positions)

        assert np.all(rebal_intervals == freq), (
            f"Rebalancing dates do not follow the expected {freq}-day"
            " frequency."
        )

        grouped_weights = weights_df.groupby(["model", "portfolio"])

        for (model, portfolio), group in grouped_weights:
            # Pivot target weight matrix (rebalance dates x tickers)
            weight_matrix = (
                group.pivot(index="date", columns="ticker", values="weight")
                .fillna(0.0)
                .reindex(rebal_dates)
                .fillna(0.0)
            )

            # Align return matrix to the portfolio's tickers
            returns_subset = asset_returns.reindex(
                columns=weight_matrix.columns
            ).fillna(0.0)

            daily_returns = []
            daily_dates = []
            turnover_records = []

            # Loop through rebalance intervals
            for k in range(len(rebal_dates) - 1):
                t_rebal = rebal_dates[k]
                t_next_rebal = rebal_dates[k + 1]

                # Execution starts T+1 to avoid look-ahead bias
                pos_rebal = all_trading_dates.get_loc(t_rebal)
                pos_next_rebal = all_trading_dates.get_loc(t_next_rebal)

                exec_dates_window = all_trading_dates[
                    pos_rebal + 1 : pos_next_rebal + 1
                ]

                # Target weight assigned at t_rebal executed at pos_rebal + 1
                w_current = weight_matrix.loc[t_rebal].copy()

                for d in exec_dates_window:
                    r_d = returns_subset.loc[d]

                    # 1. Daily Gross Return with current drifted weights
                    p_ret = (w_current * r_d).sum()
                    daily_returns.append(p_ret)
                    daily_dates.append(d)

                    # 2. Daily Weight Drift (buy-and-hold update)
                    w_current = w_current * (1.0 + r_d) / (1.0 + p_ret)

                # 3. Compute Turnover at t_next_rebal against drifted weights
                w_target_next = weight_matrix.loc[t_next_rebal]

                # Long-Short / Gross exposure aware drift handling
                is_long_short = (weight_matrix < 0).any().any()
                if is_long_short:
                    w_prev_target = weight_matrix.loc[t_rebal]
                    long_mask = w_prev_target > 0
                    short_mask = w_prev_target < 0

                    long_target_sum = w_prev_target[long_mask].sum()
                    short_target_sum = -w_prev_target[short_mask].sum()

                    drifted_long_sum = w_current[long_mask].sum()
                    drifted_short_sum = -w_current[short_mask].sum()

                    if drifted_long_sum > 0:
                        w_current[long_mask] *= (
                            long_target_sum / drifted_long_sum
                        )
                    if drifted_short_sum > 0:
                        w_current[short_mask] *= (
                            short_target_sum / drifted_short_sum
                        )

                turnover_k = 0.5 * (w_target_next - w_current).abs().sum()

                # Execution date of next rebalance is pos_next_rebal + 1
                next_exec_pos = (
                    pos_next_rebal + 1
                    if (pos_next_rebal + 1) < len(all_trading_dates)
                    else pos_next_rebal
                )
                exec_date_next = all_trading_dates[next_exec_pos]

                turnover_records.append(
                    {
                        "rebalance_date": t_next_rebal,
                        "execution_date": exec_date_next,
                        "model": model,
                        "portfolio": portfolio,
                        "rebalancing_frequency": freq,
                        "turnover": turnover_k,
                    }
                )

            # Build Daily Gross Returns DataFrame for this strategy
            res_df = pd.DataFrame(
                {
                    "date": daily_dates,
                    "model": model,
                    "portfolio": portfolio,
                    "rebalancing_frequency": freq,
                    "gross_return": daily_returns,
                }
            )
            res_df["cumulative_return"] = (
                1.0 + res_df["gross_return"]
            ).cumprod() - 1.0

            all_returns_list.append(res_df)
            all_turnover_list.append(pd.DataFrame(turnover_records))

    # Consolidate outputs
    frequency_gross_returns = (
        pd.concat(all_returns_list, ignore_index=True)
        .sort_values(
            ["rebalancing_frequency", "date", "model", "portfolio"]
        )
        .reset_index(drop=True)
    )

    frequency_turnover = (
        pd.concat(all_turnover_list, ignore_index=True)
        .sort_values(
            ["rebalancing_frequency", "execution_date", "model", "portfolio"]
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
        print("REBALANCING FREQUENCY SENSITIVITY WITH DRIFT — EXECUTION AUDIT")
        print("=" * 80)
        print(f"✓ Frequencies tested = {rebalancing_frequencies}")
        print(f"✓ Total daily observations = {len(frequency_gross_returns):,}")
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
            f"✓ Total portfolios evaluated = {frequency_gross_returns['model'].nunique() * frequency_gross_returns['portfolio'].nunique() * len(rebalancing_frequencies)}"
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

    return frequency_gross_returns, frequency_turnover


def evaluate_frequency_tradeoff(
    frequency_gross_returns: pd.DataFrame,
    frequency_turnover: pd.DataFrame,
    rebalancing_frequencies: dict[int, str],
    base_transaction_cost: float = 0.0015,  # 15 bps
    trading_days_per_year: int = 252,
    verbose: bool = True,
) -> pd.DataFrame:
    """Evaluates net performance metrics by applying transaction costs over the

    exact drifted turnover time series generated in execution.
    """
    summary_results = []
    freq_label_map = rebalancing_frequencies

    # Copy 
    freq_gross = frequency_gross_returns.copy()
    freq_to_df = frequency_turnover.copy()

    freq_gross["date"] = pd.to_datetime(freq_gross["date"])
    freq_to_df["execution_date"] = pd.to_datetime(freq_to_df["execution_date"])

    grouped_returns = freq_gross.groupby(
        ["rebalancing_frequency", "model", "portfolio"]
    )

    for (freq, model, portfolio), group_ret in grouped_returns:
        freq_label = freq_label_map.get(freq, f"{freq}D")

        # 1. Filter by turnover and freq
        group_to = freq_to_df[
            (freq_to_df["rebalancing_frequency"] == freq)
            & (freq_to_df["model"] == model)
            & (freq_to_df["portfolio"] == portfolio)
        ]

        # Map turnover
        to_map = group_to.set_index("execution_date")["turnover"].to_dict()
        daily_to = group_ret["date"].map(to_map).fillna(0.0)

        # 2. Net returns (without NaNs)
        daily_tc = daily_to * base_transaction_cost
        net_returns_array = group_ret["gross_return"].values - daily_tc.values
        net_returns = pd.Series(net_returns_array, index=group_ret["date"])

        # Aditional security check
        if net_returns.isna().any():
            net_returns = net_returns.fillna(0.0)

        # 3. Metrics
        cagr = calculate_cagr(net_returns)
        annualized_vol = net_returns.std(ddof=1) * np.sqrt(
            trading_days_per_year
        )
        sharpe = calculate_sharpe(net_returns)
        sortino = calculate_sortino(net_returns)

        (
            max_dd,
            avg_dd,
            max_underwater,
        ) = calculate_drawdown_metrics(net_returns)

        # Ann Rotation
        to_series = group_to["turnover"]
        mean_turnover = to_series.mean() if len(to_series) > 0 else 0.0
        rebalancings_per_year = trading_days_per_year / freq
        annualized_turnover = mean_turnover * rebalancings_per_year

        cum_gross = (1.0 + group_ret["gross_return"]).prod() - 1.0
        cum_net = (1.0 + net_returns).prod() - 1.0
        cum_tc = daily_tc.sum()

        summary_results.append(
            {
                "frequency_days": freq,
                "frequency_label": freq_label,
                "model": model,
                "portfolio": portfolio,
                "CAGR": cagr,
                "annualized_turnover": annualized_turnover,
                "mean_turnover": mean_turnover,
                "cumulative_gross_return": cum_gross,
                "cumulative_net_return": cum_net,
                "cumulative_transaction_cost": cum_tc,
                "annualized_volatility": annualized_vol,
                "Sharpe": sharpe,
                "Sortino": sortino,
                "maximum_drawdown": max_dd,
                "average_drawdown": avg_dd,
                "maximum_underwater_duration_days": max_underwater,
            }
        )

    frequency_sensitivity = (
        pd.DataFrame(summary_results)
        .sort_values(["frequency_days", "model", "portfolio"])
        .reset_index(drop=True)
    )

    if verbose:
        print("\n" + "=" * 80)
        print("REBALANCING FREQUENCY TRADEOFF ANALYSIS — AUDIT")
        print("=" * 80)
        print(f"✓ Frequencies evaluated = {list(rebalancing_frequencies.keys())}")
        print(
            f"✓ Applied transaction cost = {base_transaction_cost * 10000:.1f} bps"
        )
        print(
            "✓ Total strategy-frequency observations ="
            f" {len(frequency_sensitivity):,}"
        )
        print(
            f"✓ Missing net CAGR ="
            f" {frequency_sensitivity['CAGR'].isna().sum():,}"
        )
        print(
            f"✓ Missing net Sharpe ="
            f" {frequency_sensitivity['Sharpe'].isna().sum():,}"
        )
        print("=" * 80)

    return frequency_sensitivity