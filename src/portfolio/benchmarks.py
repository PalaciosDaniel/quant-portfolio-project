import numpy as np
import pandas as pd

from src.portfolio.constraints import apply_max_position_weight
from src.portfolio.constraints import apply_min_effective_weight

def apply_benchmark_constraints_with_drift(
    target_weights_wide: pd.DataFrame,
    simple_returns: pd.DataFrame,
    max_turnover_design: float = 0.25,
    min_weight: float = 0.005,
    max_weight: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applies gamma-interpolation turnover constraints (accounting for asset

    price drift between rebalances) and box constraints (min/max position
    weights) onto the raw target weights matrix of Benchmark B.

    Parameters
    ----------
    target_weights_wide : pd.DataFrame
        Unconstrained target weights matrix at rebalance dates (date x ticker).
    simple_returns : pd.DataFrame
        Full daily simple returns matrix (date x ticker).
    max_turnover_design : float
        Internal turnover limit applied per rebalance event (e.g., 0.25).
    min_weight, max_weight : float
        Lower and upper position bounds per asset.

    Returns
    -------
    df_bmk_target_constrained : pd.DataFrame
        Constrained target weights evaluated strictly at rebalance dates (date x
        ticker).
    df_bmk_executed_daily : pd.DataFrame
        Daily executed exposures incorporating T+1 execution lag and asset
        price drift (date x ticker).
    """
    target_weights = target_weights_wide.copy()
    target_weights.index = pd.to_datetime(target_weights.index)
    simple_returns = simple_returns.copy()
    simple_returns.index = pd.to_datetime(simple_returns.index)

    rebalance_dates = target_weights.index.sort_values()
    all_trading_dates = simple_returns.index.sort_values()

    constrained_targets = {}
    previous_constrained = None
    previous_rebal_date = None

    # =========================================================================
    # 1. Rebalance Loop (Constrained Target Calculation at T)
    # =========================================================================
    for curr_rebal_date in rebalance_dates:
        raw_target = target_weights.loc[curr_rebal_date]

        if previous_constrained is None:
            # Initial rebalance: no prior position to drift from
            constrained = raw_target.copy()
        else:
            # Extract simple returns window between previous and current rebalance
            window_returns = simple_returns.loc[
                (simple_returns.index > previous_rebal_date)
                & (simple_returns.index <= curr_rebal_date)
            ]

            # Compounded return per asset over the holding period
            compounded_return = (1.0 + window_returns).prod() - 1.0
            compounded_return = compounded_return.reindex(
                raw_target.index
            ).fillna(0.0)

            # Project previous position forward via market price drift
            drifted = previous_constrained * (1.0 + compounded_return)

            # Renormalize total portfolio exposure (Long-Only)
            drifted_sum = drifted.sum()
            drifted = (
                drifted / drifted_sum
                if drifted_sum > 0
                else previous_constrained
            )

            # Compute raw turnover against the drifted portfolio
            raw_turnover = 0.5 * (raw_target - drifted).abs().sum()

            # Apply gamma interpolation if turnover exceeds design cap
            if raw_turnover > max_turnover_design:
                gamma = max_turnover_design / raw_turnover
                constrained = drifted + gamma * (raw_target - drifted)
            else:
                constrained = raw_target.copy()

        # ---------------------------------------------------------------------
        # Apply Box Constraints (Min / Max position weights)
        # ---------------------------------------------------------------------
        arr = constrained.to_numpy()

        try:
            arr = apply_min_effective_weight(
                arr, min_weight=min_weight, long_short_side_exposure=None
            )
        except (ValueError, NameError):
            pass  # Retain array if min constraint renders portfolio infeasible

        try:
            arr = apply_max_position_weight(arr, max_weight=max_weight)
        except (ValueError, NameError):
            pass  # Retain array if max constraint renders portfolio infeasible

        constrained = pd.Series(arr, index=constrained.index)

        constrained_targets[curr_rebal_date] = constrained
        previous_constrained = constrained
        previous_rebal_date = curr_rebal_date

    df_bmk_target_constrained = pd.DataFrame(constrained_targets).T
    df_bmk_target_constrained.index.name = "date"

    # =========================================================================
    # 2. Build Daily Executed Weights Matrix with Asset Drift (T+1 Lag)
    # =========================================================================
    executed_daily_list = []

    for i in range(len(rebalance_dates)):
        reb_date = rebalance_dates[i]

        # Determine next rebalance date or end of sample
        next_reb_date = (
            rebalance_dates[i + 1]
            if i + 1 < len(rebalance_dates)
            else all_trading_dates[-1]
        )

        # Index position for T+1 execution lag
        reb_idx = all_trading_dates.get_loc(reb_date)
        if reb_idx + 1 >= len(all_trading_dates):
            break

        exec_start_date = all_trading_dates[reb_idx + 1]

        # Active holding window [T+1, T_next]
        window_returns = simple_returns.loc[exec_start_date:next_reb_date]

        # Initial executed weights at T+1 close
        w_current = df_bmk_target_constrained.loc[reb_date].values

        # Track daily portfolio drift
        for date, ret in window_returns.iterrows():
            executed_daily_list.append(
                pd.Series(w_current, index=simple_returns.columns, name=date)
            )

            w_gross = w_current * (1.0 + ret.values)
            total_gross = w_gross.sum()
            w_current = w_gross / total_gross if total_gross != 0 else w_gross

    df_bmk_executed_daily = pd.DataFrame(executed_daily_list)
    df_bmk_executed_daily.index.name = "date"

    return df_bmk_target_constrained, df_bmk_executed_daily