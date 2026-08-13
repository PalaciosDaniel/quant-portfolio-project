# =============================================================================
# Risk Parity — Helper Function
# =============================================================================

from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
import numpy as np

VOLATILITY_WINDOW = 21

def compute_risk_contributions(weights, covariance_matrix):
    """
    Compute total risk contributions for a portfolio.
    """

    portfolio_variance = (
        weights
        @ covariance_matrix
        @ weights
    )

    portfolio_volatility = np.sqrt(
        portfolio_variance
    )

    marginal_contribution = (
        covariance_matrix
        @ weights
    ) / portfolio_volatility

    risk_contribution = (
        weights
        * marginal_contribution
    )

    return risk_contribution


# =============================================================================
# Risk Parity — Convex ERC Solver
# =============================================================================

def compute_risk_parity_weights(
    returns_window,
):
    """
    Estimate Ledoit-Wolf covariance and
    compute long-only Equal Risk Contribution weights.
    """

    # -------------------------------------------------------------------------
    # Ledoit-Wolf covariance
    # -------------------------------------------------------------------------

    lw = LedoitWolf()

    covariance_matrix = (
        lw.fit(returns_window)
        .covariance_
    )

    n_assets = (
        covariance_matrix.shape[0]
    )

    # -------------------------------------------------------------------------
    # Risk budgets
    # -------------------------------------------------------------------------

    risk_budget = np.full(
        n_assets,
        1.0 / n_assets,
    )

    # -------------------------------------------------------------------------
    # Convex Risk Parity objective
    # -------------------------------------------------------------------------

    def objective(y):

        return (
            0.5
            * y
            @ covariance_matrix
            @ y
            - risk_budget
            @ np.log(y)
        )

    # -------------------------------------------------------------------------
    # Analytical gradient
    # -------------------------------------------------------------------------

    def gradient(y):

        return (
            covariance_matrix @ y
            - risk_budget / y
        )

    # -------------------------------------------------------------------------
    # Initial solution
    # -------------------------------------------------------------------------

    initial_y = np.ones(
        n_assets
    )

    # -------------------------------------------------------------------------
    # Optimization
    # -------------------------------------------------------------------------

    result = minimize(
        objective,
        initial_y,
        jac=gradient,
        method="L-BFGS-B",
        bounds=[
            (1e-8, None)
            for _ in range(n_assets)
        ],
        options={
            "maxiter": 1000,
            "ftol": 1e-8,
        },
    )

    if not result.success:
        raise RuntimeError(
            f"Risk Parity optimization failed: "
            f"{result.message}"
        )

    # -------------------------------------------------------------------------
    # Convert auxiliary solution into portfolio weights
    # -------------------------------------------------------------------------

    weights = (
        result.x
        / result.x.sum()
    )

    return (
        weights,
        covariance_matrix,
    )