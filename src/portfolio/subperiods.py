import numpy as np
import pandas as pd
from src.portfolio.metrics import (
    calculate_cagr,
    calculate_drawdown_metrics,
    calculate_sharpe,
    calculate_sortino,
)


def compute_subperiod_performance_with_drift(
    net_returns: pd.DataFrame,
    weights_df: pd.DataFrame,
    simple_returns: pd.DataFrame,
    rebalancing_days: int = 21,
    trading_days_per_year: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Computes performance metrics and exact turnover considering asset price drift

    across three equal-length out-of-sample (OOS) subperiods (Early, Middle, Late).
    """
    # -------------------------------------------------------------------------
    # 1. Assign OOS Subperiods (Early, Middle, Late)
    # -------------------------------------------------------------------------
    returns_df = net_returns.copy()
    returns_df["date"] = pd.to_datetime(returns_df["date"])
    returns_df = returns_df.sort_values(["model", "portfolio", "date"]).reset_index(
        drop=True
    )

    oos_dates = (
        returns_df["date"].drop_duplicates().sort_values().reset_index(drop=True)
    )
    n_dates = len(oos_dates)

    period_labels = (
        ["Early OOS"] * (n_dates // 3)
        + ["Middle OOS"] * (n_dates // 3)
        + ["Late OOS"] * (n_dates - 2 * (n_dates // 3))
    )

    date_period_map = pd.DataFrame({"date": oos_dates, "period": period_labels})

    returns_df = returns_df.merge(date_period_map, on="date", how="left")

    # -------------------------------------------------------------------------
    # 2. Exact Turnover Calculation Accounting for Price Drift
    # -------------------------------------------------------------------------
    weights = weights_df.copy()
    weights["date"] = pd.to_datetime(weights["date"])

    turnover_list = []

    for (model, portfolio), data in weights.groupby(["model", "portfolio"]):
        wide = (
            data.pivot(index="date", columns="ticker", values="weight")
            .fillna(0.0)
            .sort_index()
        )

        all_dates = wide.index
        rebalance_dates = all_dates[::rebalancing_days]

        if len(rebalance_dates) < 2:
            continue

        is_long_short = (wide < 0).any().any()

        for i in range(1, len(rebalance_dates)):
            prev_rebal_date = rebalance_dates[i - 1]
            curr_rebal_date = rebalance_dates[i]

            old_target = wide.loc[prev_rebal_date]
            new_target = wide.loc[curr_rebal_date]

            window_returns = simple_returns.loc[
                (simple_returns.index > prev_rebal_date)
                & (simple_returns.index <= curr_rebal_date)
            ]
            compounded_return = (1.0 + window_returns).prod() - 1.0
            compounded_return = compounded_return.reindex(
                old_target.index
            ).fillna(0.0)

            drifted = old_target * (1.0 + compounded_return)

            if is_long_short:
                # Renormalize each leg independently to its original target exposure
                long_mask = old_target > 0
                short_mask = old_target < 0

                long_gross_target = old_target[long_mask].sum()
                short_gross_target = -old_target[short_mask].sum()

                drifted_long_sum = drifted[long_mask].sum()
                drifted_short_sum = -drifted[short_mask].sum()

                if drifted_long_sum > 0:
                    drifted[long_mask] *= (
                        long_gross_target / drifted_long_sum
                    )
                if drifted_short_sum > 0:
                    drifted[short_mask] *= (
                        short_gross_target / drifted_short_sum
                    )
            else:
                drifted_sum = drifted.sum()
                if drifted_sum > 0:
                    drifted = drifted / drifted_sum
                else:
                    drifted = old_target

            diff = (new_target - drifted).abs()
            turnover = 0.5 * diff.sum()

            turnover_list.append(
                {
                    "date": curr_rebal_date,
                    "model": model,
                    "portfolio": portfolio,
                    "turnover": turnover,
                }
            )

    turnover_df = pd.DataFrame(turnover_list)
    turnover_df = turnover_df.merge(date_period_map, on="date", how="left")

    # -------------------------------------------------------------------------
    # 3. Subperiod Performance & Risk Metrics
    # -------------------------------------------------------------------------
    subperiod_results = []

    for (period, model, portfolio), group in returns_df.groupby(
        ["period", "model", "portfolio"]
    ):
        group = group.sort_values("date").copy()
        returns = group["net_return_base"].dropna()

        if len(returns) == 0:
            continue

        observations = len(returns)

        cagr = calculate_cagr(returns)
        daily_volatility = returns.std(ddof=1)
        annualized_volatility = daily_volatility * np.sqrt(
            trading_days_per_year
        )
        sharpe_ratio = calculate_sharpe(returns)
        sortino_ratio = calculate_sortino(returns)

        (
            maximum_drawdown,
            average_drawdown,
            maximum_underwater_duration,
        ) = calculate_drawdown_metrics(returns)

        # Retrieve adjusted turnover
        t_group = turnover_df[
            (turnover_df["model"] == model)
            & (turnover_df["portfolio"] == portfolio)
            & (turnover_df["period"] == period)
        ]

        t_series = t_group["turnover"].dropna()
        if len(t_series) > 0:
            mean_turnover = t_series.mean()
            annualized_turnover = mean_turnover * (
                trading_days_per_year / rebalancing_days
            )
        else:
            mean_turnover = np.nan
            annualized_turnover = np.nan

        subperiod_results.append(
            {
                "period": period,
                "model": model,
                "portfolio": portfolio,
                "observations": observations,
                "CAGR": cagr,
                "annualized_volatility": annualized_volatility,
                "Sharpe": sharpe_ratio,
                "Sortino": sortino_ratio,
                "maximum_drawdown": maximum_drawdown,
                "average_drawdown": average_drawdown,
                "maximum_underwater_duration_days": maximum_underwater_duration,
                "mean_turnover": mean_turnover,
                "annualized_turnover": annualized_turnover,
            }
        )

    subperiod_performance = (
        pd.DataFrame(subperiod_results)
        .sort_values(
            ["period", "model", "CAGR"], ascending=[True, True, False]
        )
        .reset_index(drop=True)
    )

    return subperiod_performance, date_period_map