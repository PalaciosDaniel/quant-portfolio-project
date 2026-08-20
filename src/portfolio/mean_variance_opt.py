# =============================================================================
# Mean-Variance / Maximum Sharpe — Helper Function
# =============================================================================

from scipy.optimize import minimize
import numpy as np


def compute_maximum_sharpe_weights(
    expected_returns,
    covariance_matrix,
):
    """
    Compute long-only Maximum Sharpe Ratio weights.

    The Maximum Sharpe problem is reformulated as a convex
    minimum-variance problem:

        minimize    x' Σ x

        subject to  μ' x = 1
                    x >= 0

    The resulting solution is normalized to obtain portfolio weights.
    """

    expected_returns = np.asarray(
        expected_returns,
        dtype=float,
    )

    covariance_matrix = np.asarray(
        covariance_matrix,
        dtype=float,
    )

    n_assets = len(expected_returns)

    # -------------------------------------------------------------------------
    # Initial solution
    # -------------------------------------------------------------------------

    initial_x = np.full(
        n_assets,
        1.0 / expected_returns.sum(),
    )

    # -------------------------------------------------------------------------
    # Objective: portfolio variance
    # -------------------------------------------------------------------------

    def objective(x):

        return (
            x
            @ covariance_matrix
            @ x
        )

    # -------------------------------------------------------------------------
    # Analytical gradient
    # -------------------------------------------------------------------------

    def gradient(x):

        return (
            2.0
            * covariance_matrix
            @ x
        )

    # -------------------------------------------------------------------------
    # Expected return constraint
    # -------------------------------------------------------------------------

    constraint = {
        "type": "eq",
        "fun": lambda x:
            expected_returns @ x - 1.0,
        "jac": lambda x:
            expected_returns,
    }

    # -------------------------------------------------------------------------
    # Long-only constraint
    # -------------------------------------------------------------------------

    bounds = [
        (0.0, None)
        for _ in range(n_assets)
    ]

    # -------------------------------------------------------------------------
    # Optimization
    # -------------------------------------------------------------------------

    result = minimize(
        objective,
        initial_x,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraint,
        options={
            "maxiter": 500,
            "ftol": 1e-9,
        },
    )

    if not result.success:
        raise RuntimeError(
            f"Maximum Sharpe optimization failed: "
            f"{result.message}"
        )

    # -------------------------------------------------------------------------
    # Convert auxiliary solution into portfolio weights
    # -------------------------------------------------------------------------

    weights = (
        result.x
        / result.x.sum()
    )

    return weights


# =============================================================================
# Isotonic calibration
# =============================================================================

from sklearn.isotonic import IsotonicRegression

def fit_isotonic_calibration(
    calibration_data,
):
    """
    Fit isotonic regression from cross-sectional
    prediction percentile to realized forward return.
    """

    # -------------------------------------------------------------------------
    # Aggregate realized returns by decile
    # -------------------------------------------------------------------------

    decile_data = (
        calibration_data
        .groupby("decile")
        .agg(
            percentile=("percentile", "mean"),
            target=("target", "mean"),
            observations=("target", "count"),
        )
        .reset_index()
    )

    # -------------------------------------------------------------------------
    # Fit isotonic regression
    # -------------------------------------------------------------------------

    isotonic_model = IsotonicRegression(
        increasing=True,
        out_of_bounds="clip",
    )

    isotonic_model.fit(
        decile_data["percentile"],
        decile_data["target"],
        sample_weight=decile_data["observations"],
    )

    return (
        isotonic_model,
        decile_data,
    )