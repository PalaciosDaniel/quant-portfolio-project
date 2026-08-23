# =============================================================================
# Portfolio Constraints — Helper Functions
# =============================================================================

import numpy as np
import pandas as pd


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
# Turnover Diagnosis
# =============================================================================

TRADING_DAYS_PER_YEAR = 252
REBALANCING_DAYS = 21
REBALANCINGS_PER_YEAR = (
    TRADING_DAYS_PER_YEAR
    / REBALANCING_DAYS
)


def compute_turnover(
    weights,
):
    """
    Compute portfolio turnover between
    consecutive rebalancing dates.

    Turnover = 0.5 * sum(|w_t - w_{t-1}|)
    """

    weights = (
        weights
        .copy()
    )

    weights["date"] = pd.to_datetime(
        weights["date"]
    )

    turnover_list = []

    # -------------------------------------------------------------------------
    # Process each model and portfolio independently
    # -------------------------------------------------------------------------

    for (
        model,
        portfolio,
    ), data in weights.groupby(
        [
            "model",
            "portfolio",
        ]
    ):

        data = (
            data
            .pivot(
                index="date",
                columns="ticker",
                values="weight",
            )
            .fillna(0.0)
            .sort_index()
        )

        # ---------------------------------------------------------------------
        # Consecutive portfolio weights
        # ---------------------------------------------------------------------

        previous = data.shift(1)

        # ---------------------------------------------------------------------
        # L1 turnover
        # ---------------------------------------------------------------------

        turnover = (
            0.5
            * (
                data
                - previous
            )
            .abs()
            .sum(axis=1)
        )

        turnover = (
            turnover
            .dropna()
        )

        if len(turnover) == 0:
            continue

        result = pd.DataFrame(
            {
                "date": turnover.index,
                "model": model,
                "portfolio": portfolio,
                "turnover": turnover.values,
            }
        )

        turnover_list.append(
            result
        )

    if not turnover_list:
        return pd.DataFrame(
            columns=[
                "date",
                "model",
                "portfolio",
                "turnover",
            ]
        )

    return pd.concat(
        turnover_list,
        ignore_index=True,
    )


# =============================================================================
# Turnover Constraint — Maximum Sharpe
# =============================================================================

MAX_TURNOVER_TARGET = 0.30
MAX_TURNOVER_DESIGN = 0.25


def apply_turnover_constraint(
    weights,
    max_turnover=MAX_TURNOVER_DESIGN,
    min_weight=0.005,
    max_weight=0.05,
):
    """Apply turnover constraint via interpolation, followed by box constraints

    (min_weight, max_weight) post-processing.
    """

    weights = weights.copy()
    weights["date"] = pd.to_datetime(weights["date"])

    constrained_list = []

    for (model, portfolio), data in weights.groupby(["model", "portfolio"]):
        data = data.sort_values("date")
        dates = data["date"].unique()

        previous_weights = None

        for date in dates:
            current = data[data["date"] == date].set_index("ticker")["weight"]

            if previous_weights is None:
                constrained = current.copy()
            else:
                all_tickers = previous_weights.index.union(current.index)
                previous = previous_weights.reindex(
                    all_tickers, fill_value=0.0
                )
                target = current.reindex(all_tickers, fill_value=0.0)

                raw_turnover = 0.5 * np.abs(target - previous).sum()

                # 1. Turnover constraint via gamma interpolation
                if raw_turnover > max_turnover:
                    gamma = max_turnover / raw_turnover
                    constrained = previous + gamma * (target - previous)
                else:
                    constrained = target.copy()

            # -----------------------------------------------------------------
            # 2. Post-processing: Box Constraints (Min & Max Weight)
            # -----------------------------------------------------------------
            # Step A: Filter out tiny positions below 0.5%
            constrained[constrained < min_weight] = 0.0

            # Step B: Iterative Clipping & Normalization to enforce max 5%
            # Cap at max_weight and renormalize iteratively until convergence
            for _ in range(10):
                if constrained.sum() <= 0:
                    break
                constrained = constrained / constrained.sum()

                # Cap weights strictly above max_weight
                over_max = constrained > max_weight
                if not over_max.any():
                    break
                constrained[over_max] = max_weight

            # Final check on min weights after capping
            constrained[constrained < min_weight] = 0.0
            if constrained.sum() > 0:
                constrained = constrained / constrained.sum()

            # -----------------------------------------------------------------
            # Store results
            # -----------------------------------------------------------------
            date_weights = pd.DataFrame(
                {
                    "date": date,
                    "ticker": constrained.index,
                    "model": model,
                    "portfolio": portfolio,
                    "weight": constrained.values,
                }
            )

            date_weights = date_weights[date_weights["weight"] > 1e-10]
            constrained_list.append(date_weights)

            previous_weights = constrained.copy()

    return pd.concat(constrained_list, ignore_index=True)


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