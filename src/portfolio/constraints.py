# =============================================================================
# Portfolio Constraints — Helper Functions
# =============================================================================

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
REBALANCING_DAYS = 21
REBALANCINGS_PER_YEAR = (
    TRADING_DAYS_PER_YEAR
    / REBALANCING_DAYS
)


def apply_max_position_weight(
    weights,
    max_weight=0.05,
):
    """
    Apply a maximum absolute position weight constraint.

    Excess weight from positions above the limit is
    iteratively redistributed among positions that
    remain below the maximum allowed weight.

    Parameters
    ----------
    weights : array-like
        Portfolio weights.

    max_weight : float
        Maximum allowed absolute position weight.

    Returns
    -------
    constrained_weights : np.ndarray
        Portfolio weights satisfying the constraint.
    """

    weights = np.asarray(
        weights,
        dtype=float,
    ).copy()

    if len(weights) == 0:
        return weights

    if max_weight <= 0:
        raise ValueError(
            "max_weight must be positive."
        )

    if np.sum(np.abs(weights)) == 0:
        return weights

    # -------------------------------------------------------------------------
    # Long-only portfolio
    # -------------------------------------------------------------------------

    if np.all(weights >= 0):

        total_weight = weights.sum()

        if total_weight <= 0:
            return weights

        weights /= total_weight

        # Check feasibility
        if len(weights) * max_weight < 1:
            raise ValueError(
                "Maximum weight constraint is infeasible "
                "for the number of positions."
            )

        # ---------------------------------------------------------------------
        # Iteratively redistribute excess weight
        # ---------------------------------------------------------------------

        while np.any(weights > max_weight + 1e-12):

            excess = (
                weights[weights > max_weight]
                - max_weight
            ).sum()

            weights[
                weights > max_weight
            ] = max_weight

            eligible = (
                weights < max_weight - 1e-12
            )

            if not np.any(eligible):
                break

            eligible_weight = weights[eligible]

            if eligible_weight.sum() == 0:
                redistribution = (
                    excess / eligible.sum()
                )
            else:
                redistribution = (
                    excess
                    * eligible_weight
                    / eligible_weight.sum()
                )

            weights[eligible] += redistribution

        # Numerical normalization
        weights /= weights.sum()

        return weights

    # -------------------------------------------------------------------------
    # Long-short portfolio
    # -------------------------------------------------------------------------

    long_mask = weights > 0
    short_mask = weights < 0

    long_weights = weights[long_mask]
    short_weights = weights[short_mask]

    # -------------------------------------------------------------------------
    # Apply constraint independently to long and short legs
    # Each leg is scaled to 0.50 (50% gross exposure per side)
    # -------------------------------------------------------------------------

    if len(long_weights) > 0:
        # Pass double max_weight since sub-portfolio is normalized to 1.0 internally,
        # then scale down to target 0.50 exposure
        long_weights = apply_max_position_weight(
            long_weights,
            max_weight=max_weight * 2.0,
        )
        long_weights *= 0.50

    if len(short_weights) > 0:
        short_weights = apply_max_position_weight(
            -short_weights,
            max_weight=max_weight * 2.0,
        )
        short_weights = -short_weights * 0.50

    constrained_weights = np.zeros_like(weights)
    constrained_weights[long_mask] = long_weights
    constrained_weights[short_mask] = short_weights

    return constrained_weights


def apply_min_effective_weight(
    weights,
    min_weight=0.005,
    long_short_side_exposure=None,
):
    """
    Remove positions below the minimum effective weight
    and renormalize the remaining positions.

    Parameters
    ----------
    weights : array-like
        Portfolio weights.

    min_weight : float
        Minimum effective absolute position weight.

    long_short_side_exposure : float or None
        Target gross exposure of each side for long-short
        portfolios. If None, portfolio is treated as long-only.

    Returns
    -------
    constrained_weights : np.ndarray
        Portfolio weights after removing small positions
        and renormalizing the remaining positions.
    """

    weights = np.asarray(
        weights,
        dtype=float,
    ).copy()

    if len(weights) == 0:
        return weights

    if min_weight <= 0:
        raise ValueError(
            "min_weight must be positive."
        )

    # =========================================================================
    # Long-only
    # =========================================================================

    if long_short_side_exposure is None:

        effective = (
            weights >= min_weight
        )

        weights[
            ~effective
        ] = 0.0

        total_weight = weights.sum()

        if total_weight <= 0:
            raise ValueError(
                "Minimum effective weight removes "
                "all portfolio positions."
            )

        weights /= total_weight

        return weights

    # =========================================================================
    # Long-short
    # =========================================================================

    if long_short_side_exposure <= 0:
        raise ValueError(
            "long_short_side_exposure must be positive."
        )

    long_mask = weights > 0
    short_mask = weights < 0

    long_weights = weights[long_mask]
    short_weights = weights[short_mask]

    # -------------------------------------------------------------------------
    # Long side
    # -------------------------------------------------------------------------

    if len(long_weights) > 0:

        long_weights[
            long_weights
            < min_weight
        ] = 0.0

        long_total = (
            long_weights.sum()
        )

        if long_total <= 0:
            raise ValueError(
                "Minimum effective weight removes "
                "all long positions."
            )

        long_weights *= (
            long_short_side_exposure
            / long_total
        )

    # -------------------------------------------------------------------------
    # Short side
    # -------------------------------------------------------------------------

    if len(short_weights) > 0:

        short_abs = np.abs(
            short_weights
        )

        short_abs[
            short_abs
            < min_weight
        ] = 0.0

        short_total = (
            short_abs.sum()
        )

        if short_total <= 0:
            raise ValueError(
                "Minimum effective weight removes "
                "all short positions."
            )

        short_weights = (
            -short_abs
            * (
                long_short_side_exposure
                / short_total
            )
        )

    # -------------------------------------------------------------------------
    # Reconstruct portfolio
    # -------------------------------------------------------------------------

    constrained_weights = (
        np.zeros_like(weights)
    )

    constrained_weights[
        long_mask
    ] = long_weights

    constrained_weights[
        short_mask
    ] = short_weights

    return constrained_weights

# =============================================================================
# TURNOVER AT REBALANCE FREQUENCY, ACCOUNTING FOR MARKET DRIFT
# =============================================================================

def compute_turnover_with_drift(
    weights_df,
    simple_returns,
    rebalancing_days=21,
):
    weights_df = weights_df.copy()
    weights_df["date"] = pd.to_datetime(weights_df["date"])

    turnover_list = []

    for (model, portfolio), data in weights_df.groupby(["model", "portfolio"]):

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
            compounded_return = compounded_return.reindex(old_target.index).fillna(0.0)

            drifted = old_target * (1.0 + compounded_return)

            if is_long_short:
                # Renormalize each leg independently to its own original
                # gross exposure — never mix signs into a single total
                long_mask = old_target > 0
                short_mask = old_target < 0

                long_gross_target = old_target[long_mask].sum()
                short_gross_target = -old_target[short_mask].sum()

                drifted_long_sum = drifted[long_mask].sum()
                drifted_short_sum = -drifted[short_mask].sum()

                if drifted_long_sum > 0:
                    drifted[long_mask] *= long_gross_target / drifted_long_sum
                if drifted_short_sum > 0:
                    drifted[short_mask] *= short_gross_target / drifted_short_sum
            else:
                drifted_sum = drifted.sum()
                if drifted_sum > 0:
                    drifted = drifted / drifted_sum
                else:
                    drifted = old_target

            diff = (new_target - drifted).abs()
            turnover = 0.5 * diff.sum()

            turnover_list.append({
                "date": curr_rebal_date,
                "model": model,
                "portfolio": portfolio,
                "turnover": turnover,
            })

    return pd.DataFrame(turnover_list)

# =============================================================================
# TURNOVER CONSTRAINT WITH MARKET DRIFT — REBALANCE-ONLY + DAILY HOLD OUTPUT
# =============================================================================

def apply_turnover_constraint_with_drift(
    portfolio_weights,
    simple_returns,
    rebalancing_days=21,
    max_turnover_design=0.25,
    min_weight=0.005,
    max_weight=0.05,
):
    """Applies the gamma-interpolation turnover constraint at rebalance
    frequency, comparing the raw target against the PREVIOUS constrained
    position drifted forward by realized market returns (not the stale,
    un-drifted previous target).

    For long-short portfolios, drift renormalization and box constraints
    treat the long and short legs independently, consistent with
    apply_max_position_weight / apply_min_effective_weight.

    Parameters
    ----------
    portfolio_weights : pd.DataFrame
        Daily weights (date, ticker, model, portfolio, weight), as produced
        by Block 6.
    simple_returns : pd.DataFrame
        Daily simple returns, date x ticker (from adj_close.pct_change()).
    rebalancing_days : int
        Spacing, in trading days, between rebalance events.
    max_turnover_design : float
        Internal (conservative) turnover cap used by the gamma-interpolation
        step. Deliberately set below the nominal 0.30 mandate to absorb the
        expansion introduced by the subsequent box-constraint renormalization
        (not yet validated empirically — deferred).
    min_weight, max_weight : float
        Box constraints applied after the turnover interpolation, via
        apply_min_effective_weight and apply_max_position_weight.

    Returns
    -------
    rebalance_weights : pd.DataFrame
        (date, ticker, model, portfolio, weight) — one row per rebalance
        date only.
    daily_weights : pd.DataFrame
        Same schema, but with the constrained rebalance weights held
        (propagated forward, unchanged) across all daily dates until the
        next rebalance.
    """
    portfolio_weights = portfolio_weights.copy()
    portfolio_weights["date"] = pd.to_datetime(portfolio_weights["date"])

    rebalance_records = []
    daily_frames = []

    for (model, portfolio), data in portfolio_weights.groupby(["model", "portfolio"]):

        wide = (
            data.pivot(index="date", columns="ticker", values="weight")
            .fillna(0.0)
            .sort_index()
        )

        all_dates = wide.index
        rebalance_dates = all_dates[::rebalancing_days]

        if len(rebalance_dates) == 0:
            continue

        is_long_short = (wide < 0).any().any()
        long_short_exposure = 0.5 if is_long_short else None

        constrained_rows = {}
        previous_constrained = None
        previous_rebal_date = None

        for curr_rebal_date in rebalance_dates:

            raw_target = wide.loc[curr_rebal_date]

            # ---------------------------------------------------------------
            # 1. Turnover interpolation vs. the DRIFTED previous position
            # ---------------------------------------------------------------
            if previous_constrained is None:
                # First rebalance: nothing to drift from, nothing to constrain
                constrained = raw_target.copy()
            else:
                window_returns = simple_returns.loc[
                    (simple_returns.index > previous_rebal_date)
                    & (simple_returns.index <= curr_rebal_date)
                ]
                compounded_return = (1.0 + window_returns).prod() - 1.0
                compounded_return = compounded_return.reindex(raw_target.index).fillna(0.0)

                drifted = previous_constrained * (1.0 + compounded_return)

                if is_long_short:
                    long_mask = previous_constrained > 0
                    short_mask = previous_constrained < 0

                    long_gross_target = previous_constrained[long_mask].sum()
                    short_gross_target = -previous_constrained[short_mask].sum()

                    drifted_long_sum = drifted[long_mask].sum()
                    drifted_short_sum = -drifted[short_mask].sum()

                    if drifted_long_sum > 0:
                        drifted[long_mask] *= long_gross_target / drifted_long_sum
                    if drifted_short_sum > 0:
                        drifted[short_mask] *= short_gross_target / drifted_short_sum
                else:
                    drifted_sum = drifted.sum()
                    drifted = drifted / drifted_sum if drifted_sum > 0 else previous_constrained

                raw_turnover = 0.5 * (raw_target - drifted).abs().sum()

                if raw_turnover > max_turnover_design:
                    gamma = max_turnover_design / raw_turnover
                    constrained = drifted + gamma * (raw_target - drifted)
                else:
                    constrained = raw_target.copy()

            # ---------------------------------------------------------------
            # 2. Box constraints (min then max), long-short aware
            # ---------------------------------------------------------------
            arr = constrained.to_numpy()

            try:
                arr = apply_min_effective_weight(
                    arr, min_weight=min_weight, long_short_side_exposure=long_short_exposure
                )
            except ValueError:
                pass  # min_weight would remove every position on this leg; keep pre-min array

            try:
                arr = apply_max_position_weight(arr, max_weight=max_weight)
            except ValueError:
                pass  # infeasible cap for this number of positions; keep pre-cap array

            constrained = pd.Series(arr, index=constrained.index)

            constrained_rows[curr_rebal_date] = constrained
            previous_constrained = constrained
            previous_rebal_date = curr_rebal_date

        # ---------------------------------------------------------------
        # 3. Rebalance-only output
        # ---------------------------------------------------------------
        rebalance_wide = pd.DataFrame(constrained_rows).T
        rebalance_wide.index.name = "date"

        rebal_long = rebalance_wide.reset_index().melt(
            id_vars="date", var_name="ticker", value_name="weight"
        )
        rebal_long = rebal_long[rebal_long["weight"].abs() > 1e-10]
        rebal_long["model"] = model
        rebal_long["portfolio"] = portfolio
        rebalance_records.append(rebal_long[["date", "ticker", "model", "portfolio", "weight"]])

        # ---------------------------------------------------------------
        # 4. Daily output — hold constrained weights until next rebalance
        # ---------------------------------------------------------------
        daily_wide = rebalance_wide.reindex(all_dates).ffill()

        daily_long = daily_wide.reset_index().melt(
            id_vars="date", var_name="ticker", value_name="weight"
        )
        daily_long = daily_long[daily_long["weight"].abs() > 1e-10]
        daily_long["model"] = model
        daily_long["portfolio"] = portfolio
        daily_frames.append(daily_long[["date", "ticker", "model", "portfolio", "weight"]])

    rebalance_weights = pd.concat(rebalance_records, ignore_index=True)
    daily_weights = pd.concat(daily_frames, ignore_index=True)

    return rebalance_weights, daily_weights


def validate_turnover_and_box_constraints(
    rebalance_weights: pd.DataFrame,
    daily_weights: pd.DataFrame,
    max_turnover_threshold: float = 0.30,
    max_weight_threshold: float = 0.05,
    min_weight_threshold: float = 0.005,
) -> pd.DataFrame:
    """
    Quantitatively validates turnover limits, box constraints (min/max weights),
    weight summation integrity, and daily weight drift consistency across all portfolios.

    Parameters:
    -----------
    rebalance_weights : pd.DataFrame
        DataFrame containing discrete rebalancing weights (only rebalancing dates).
    daily_weights : pd.DataFrame
        DataFrame containing expanded daily weights including inter-rebalance drift.
    max_turnover_threshold : float
        Maximum allowed single-period turnover (e.g., 0.30 for 30%).
    max_weight_threshold : float
        Maximum allowed individual asset allocation (e.g., 0.05 for 5.0%).
    min_weight_threshold : float
        Minimum allowed active allocation (e.g., 0.005 for 0.5%).

    Returns:
    --------
    pd.DataFrame
        Summary DataFrame with validation metrics and violation counts per (model, portfolio).
    """
    # 1. Ensure temporal order and deep copy
    reb = rebalance_weights.copy()
    reb = reb.sort_values(["model", "portfolio", "date", "ticker"])

    validation_results = []

    # Iterate over each unique model and portfolio strategy
    for (model, portfolio), group in reb.groupby(["model", "portfolio"]):
        # Pivot into wide format: rows=dates, columns=tickers, values=weights
        pivot_weights = group.pivot(
            index="date", columns="ticker", values="weight"
        ).fillna(0.0)

        # ---------------------------------------------------------------------
        # A. TURNOVER VALIDATION (AT REBALANCE DATES)
        # ---------------------------------------------------------------------
        # Turnover_t = 0.5 * sum(|w_{i, t} - w_{i, t-1}|)
        weight_diffs = pivot_weights.diff().abs().sum(axis=1) / 2.0

        # Drop the first rebalance date (no prior portfolio to compare turnover against)
        turnovers = weight_diffs.iloc[1:]

        max_obs_turnover = turnovers.max() if not turnovers.empty else 0.0
        mean_obs_turnover = turnovers.mean() if not turnovers.empty else 0.0
        # Allow tiny numerical precision margin (+1e-6)
        turnover_violations = (
            turnovers > (max_turnover_threshold + 1e-6)
        ).sum()

        # ---------------------------------------------------------------------
        # B. BOX CONSTRAINTS VALIDATION (MIN & MAX WEIGHTS)
        # ---------------------------------------------------------------------
        # Isolate active non-zero positions (ignore exact zero weights resulting from divestment)
        active_weights = group[group["weight"] > 1e-6]["weight"]

        max_obs_weight = group["weight"].max()
        min_obs_effective_weight = (
            active_weights.min() if not active_weights.empty else 0.0
        )

        # Count positions exceeding maximum threshold (> 5%)
        positions_above_max = (
            group["weight"] > (max_weight_threshold + 1e-6)
        ).sum()

        # Count active positions violating minimum threshold (0 < weight < 0.5%)
        positions_below_min = (
            (group["weight"] > 1e-6)
            & (group["weight"] < (min_weight_threshold - 1e-6))
        ).sum()

        # ---------------------------------------------------------------------
        # C. WEIGHT SUM INTEGRITY CHECK (SUM(W_i) == 1.0)
        # ---------------------------------------------------------------------
        # Ensure normalization step preserved 100% total investment allocation
        # Long-Only portfolios must sum to 1.0
        # Long-Short portfolios must sum to 0.0 (Market Neutral)
        weight_sums = pivot_weights.sum(axis=1)
        if portfolio.startswith("long_only"):
            sum_violations = (~np.isclose(weight_sums, 1.0, atol=1e-4)).sum()
        else:
            # For Long-Short, net weight must equal 0.0
            sum_violations = (~np.isclose(weight_sums, 0.0, atol=1e-4)).sum()

        # ---------------------------------------------------------------------
        # D. DRIFT DATASET INTEGRITY CHECK (DAILY WEIGHTS)
        # ---------------------------------------------------------------------
        daily_group = daily_weights[
            (daily_weights["model"] == model)
            & (daily_weights["portfolio"] == portfolio)
        ]

        # Check for missing values in the daily series
        daily_nans = daily_group["weight"].isnull().sum()

        # Check for invalid negative weights in Long-Only portfolios
        is_long_only = portfolio.startswith("long_only")
        negative_weight_violations = 0
        if is_long_only:
            negative_weight_violations = (daily_group["weight"] < -1e-6).sum()

        # Append validation metrics for current strategy
        validation_results.append(
            {
                "model": model,
                "portfolio": portfolio,
                "mean_turnover": mean_obs_turnover,
                "max_turnover": max_obs_turnover,
                "turnover_violations": turnover_violations,
                "max_weight": max_obs_weight,
                "min_effective_weight": min_obs_effective_weight,
                "viol_above_max_weight": positions_above_max,
                "viol_below_min_weight": positions_below_min,
                "sum_weight_violations": sum_violations,
                "daily_nan_count": daily_nans,
                "daily_negative_weights": negative_weight_violations,
            }
        )

    return pd.DataFrame(validation_results)

# =============================================================================
# Helper function to extract weighting strategy from portfolio name
# =============================================================================
def extract_weighting_scheme(portfolio_name):
    if "maximum_sharpe" in portfolio_name:
        return "Maximum Sharpe"
    elif "risk_parity" in portfolio_name:
        return "Risk Parity"
    elif "inverse_volatility" in portfolio_name:
        return "Inverse Volatility"
    elif "signal_weight" in portfolio_name:
        return "Signal Weighting"
    elif "equal_weight" in portfolio_name:
        return "Equal Weight"
    return "Other"


# =============================================================================
# Helper function to extract exposure type (long-only vs long-short)
# =============================================================================
def extract_exposure_type(portfolio_name):
    if portfolio_name.startswith("long_short_"):
        return "Long-Short"
    elif portfolio_name.startswith("long_only_"):
        return "Long-Only"
    return "Other"


# =============================================================================
# BUFFER ZONE SELECTION — SEQUENTIAL HYSTERESIS ON THE SELECTION UNIVERSE
# =============================================================================

def apply_buffer_zone_selection(
    pct_wide_model,
    entry_cutoff,
    buffer_width=0.03,
):
    """Applies asymmetric hysteresis (buffer zone) to a cross-sectional
    percentile-based selection universe.

    An asset already held in the previous rebalance date remains eligible
    as long as its percentile stays above (entry_cutoff - buffer_width).
    An asset NOT currently held must clear the full entry_cutoff to enter.
    This reduces boundary-driven turnover without relaxing the entry bar
    for genuinely new positions.

    The strict '>' comparison on entry mirrors the original decile filter's
    behavior (np.ceil(pct * 10) == 10  <=>  pct > 0.9, not pct >= 0.9).

    Parameters
    ----------
    pct_wide_model : pd.DataFrame
        Cross-sectional percentile, indexed by date, columns = tickers,
        sourced from signal_ranks (same source as quantile_data).
    entry_cutoff : float
        Original selection threshold (e.g. 0.90 for Top 10%).
    buffer_width : float, optional
        Width of the hysteresis band subtracted from entry_cutoff to form
        the exit threshold, by default 0.03.

    Returns
    -------
    dict[pd.Timestamp, list[str]]
        Buffered ticker selection per date.
    """
    exit_cutoff = entry_cutoff - buffer_width

    dates = pct_wide_model.index.sort_values()
    buffered_tickers_by_date = {}
    previously_held = set()

    for date in dates:
        pct_row = pct_wide_model.loc[date]

        eligible_new = set(pct_row.index[pct_row > entry_cutoff])
        eligible_retained = set(
            pct_row.index[pct_row >= exit_cutoff]
        ) & previously_held

        selected = eligible_new | eligible_retained
        buffered_tickers_by_date[date] = sorted(selected)
        previously_held = selected

    return buffered_tickers_by_date


def analyze_buffer_eclipse_by_min_weight(
    comparison_pairs,
    portfolio_weights,
    buffered_weights_map,
    pct_wide_by_model,
    buffered_tickers_by_model,
    rank_columns,
    max_weight=0.05,
    min_weight=0.005,
    entry_cutoff=0.90,
):
    """
    Evaluates how many buffer-retained positions are eclipsed (dropped) 
    by the minimum position weight cleanup threshold.
    """
    records = []

    for nature, original_name, buffered_name in comparison_pairs:
        buffered_source = buffered_weights_map[nature]

        for model in rank_columns:
            # 1. Process original and buffered weights under max/min constraints
            dfs_stats = {}
            for key, src_df, p_name in [
                ("orig", portfolio_weights, original_name),
                ("buf", buffered_source, buffered_name),
            ]:
                subset = src_df[
                    (src_df["model"] == model) & (src_df["portfolio"] == p_name)
                ]

                daily_records = []
                for date, group in subset.groupby("date"):
                    raw_w = group["weight"].to_numpy()
                    capped_w = apply_max_position_weight(
                        raw_w.copy(), max_weight=max_weight
                    )

                    try:
                        final_w = apply_min_effective_weight(
                            capped_w.copy(), min_weight=min_weight
                        )
                    except ValueError:
                        final_w = capped_w

                    daily_records.append(
                        {
                            "n_dropped_by_min": (raw_w > 1e-8).sum()
                            - (final_w > 1e-8).sum()
                        }
                    )

                dfs_stats[key] = pd.DataFrame(daily_records)

            # 2. Track "Buffer-Only" positions (retained purely by hysteresis)
            pct_wide_model = pct_wide_by_model[model]
            buf_tickers_date = buffered_tickers_by_model[model]
            buf_subset = buffered_source[
                (buffered_source["model"] == model)
                & (buffered_source["portfolio"] == buffered_name)
            ]

            n_buffer_only_total = 0
            n_buffer_only_eclipsed = 0

            for date, group in buf_subset.groupby("date"):
                if date not in pct_wide_model.index:
                    continue

                pct_row = pct_wide_model.loc[date]
                held_tickers = buf_tickers_date.get(date, [])
                strict_pass = set(pct_row.index[pct_row > entry_cutoff])
                buffer_only_tickers = set(held_tickers) - strict_pass

                if not buffer_only_tickers:
                    continue

                date_weights = group.set_index("ticker")["weight"]
                bo_weights = date_weights.reindex(
                    list(buffer_only_tickers)
                ).fillna(0.0)

                n_buffer_only_total += len(buffer_only_tickers)
                n_buffer_only_eclipsed += (bo_weights < min_weight).sum()

            records.append(
                {
                    "nature": nature,
                    "model": model,
                    "orig_avg_dropped_by_min": dfs_stats["orig"]["n_dropped_by_min"].mean(),
                    "buf_avg_dropped_by_min": dfs_stats["buf"]["n_dropped_by_min"].mean(),
                    "n_buffer_only_positions": n_buffer_only_total,
                    "n_buffer_only_eclipsed": n_buffer_only_eclipsed,
                }
            )

    return pd.DataFrame(records)


# =============================================================================
# APPLY MAX/MIN WEIGHT CONSTRAINTS TO BUFFERED PORTFOLIOS
# =============================================================================

MAX_POSITION_WEIGHT = 0.05
MIN_POSITION_WEIGHT = 0.005

def apply_block7_constraints(weights_df, model, portfolio_name):
    """Applies apply_max_position_weight then apply_min_effective_weight,
    per rebalance date, to a single (model, portfolio) slice.

    Returns a long DataFrame with the constrained weights, and a per-date
    diagnostic log of any infeasibility encountered.
    """
    subset = weights_df[
        (weights_df["model"] == model) & (weights_df["portfolio"] == portfolio_name)
    ]

    constrained_records = []
    diagnostic_records = []

    for date, group in subset.groupby("date"):
        tickers = group["ticker"].to_numpy()
        raw_weights = group["weight"].to_numpy()

        n_active_raw = (raw_weights > 1e-8).sum()

        try:
            capped = apply_max_position_weight(
                raw_weights.copy(), max_weight=MAX_POSITION_WEIGHT
            )
        except ValueError as e:
            diagnostic_records.append({
                "date": date, "step": "max_weight", "error": str(e),
                "n_active_raw": n_active_raw,
            })
            capped = raw_weights.copy()  # fallback: keep uncapped for this date

        try:
            final_weights = apply_min_effective_weight(
                capped.copy(), min_weight=MIN_POSITION_WEIGHT
            )
        except ValueError as e:
            diagnostic_records.append({
                "date": date, "step": "min_weight", "error": str(e),
                "n_active_raw": n_active_raw,
            })
            final_weights = capped  # fallback: keep capped-only for this date

        for ticker, w in zip(tickers, final_weights):
            constrained_records.append({
                "date": date, "ticker": ticker, "model": model,
                "portfolio": portfolio_name, "weight": w,
            })

    constrained_df = pd.DataFrame(constrained_records)
    diagnostics_df = pd.DataFrame(diagnostic_records)
    return constrained_df, diagnostics_df