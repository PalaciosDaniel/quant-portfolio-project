import numpy as np
import pandas as pd

# =============================================================================
# 1. BASE CHECK: Core Data Integrity
# =============================================================================

def validate_core_integrity(weights_df: pd.DataFrame) -> None:
    """Validates structural portfolio integrity:

    1. No negative weights in long-only portfolios.
    2. No duplicated observations (date-ticker-model-portfolio).
    3. Total portfolio exposure equals exactly 1.00 (100%).
    """
    # 1. Non-negativity check
    assert (
        weights_df["weight"] >= 0
    ).all(), "Integrity Error: Negative weights detected in long-only strategy."

    # 2. Uniqueness check
    keys = ["date", "ticker", "model", "portfolio"]
    duplicates = weights_df.duplicated(subset=keys).sum()
    assert (
        duplicates == 0
    ), f"Integrity Error: Found {duplicates} duplicate date-ticker-model-portfolio rows."

    # 3. Total exposure check
    exposure = weights_df.groupby(["date", "model", "portfolio"])[
        "weight"
    ].sum()
    assert np.allclose(
        exposure.values, 1.00
    ), "Exposure Error: Total exposure does not equal 1.00 for all portfolios."

    print(
        "✓ Core Integrity Passed: Non-negative weights, zero duplicates, exact"
        " 1.00 exposure."
    )


# =============================================================================
# 2. STRATEGY CHECK: Concentration Monotonicity
# =============================================================================

def validate_concentration_monotonicity(
    weights_df: pd.DataFrame, portfolio_names: list
) -> None:
    """Evaluates maximum weight monotonicity across selection levels (Top 10% >= Top 20% >= Top 30%).

    Instead of raising a strict assertion error, it logs the number of dates
    violating the monotonicity condition to account for risk-based dynamics.
    """
    # 1. Pivot to obtain maximum weight per date, model, and portfolio
    max_w = (
        weights_df.groupby(["date", "model", "portfolio"])["weight"]
        .max()
        .unstack("portfolio")
    )

    # 2. Safety check: Ensure requested portfolios exist in the DataFrame
    missing_cols = [p for p in portfolio_names if p not in max_w.columns]
    assert (
        not missing_cols
    ), f"Concentration Error: Portfolio names {missing_cols} not found in DataFrame."

    p_10, p_20, p_30 = portfolio_names[0], portfolio_names[1], portfolio_names[2]

    # 3. Identify dates where monotonicity is violated
    # Condition: Top 10% >= Top 20% and Top 20% >= Top 30%
    violations_10_vs_20 = (max_w[p_10] < max_w[p_20]).sum()
    violations_20_vs_30 = (max_w[p_20] < max_w[p_30]).sum()
    total_dates = len(max_w)

    # 4. Report findings instead of crashing
    if violations_10_vs_20 == 0 and violations_20_vs_30 == 0:
        print(
            "✓ Concentration Check Passed: Monotonicity holds across 100% of"
            f" dates ({total_dates}/{total_dates})."
        )
    else:
        print(
            f"⚠️ Concentration Monotonicity Warnings ({total_dates} dates total):"
        )
        if violations_10_vs_20 > 0:
            pct = (violations_10_vs_20 / total_dates) * 100
            print(
                f"   - {p_10} < {p_20} in {violations_10_vs_20} dates"
                f" ({pct:.2f}%)"
            )
        if violations_20_vs_30 > 0:
            pct = (violations_20_vs_30 / total_dates) * 100
            print(
                f"   - {p_20} < {p_30} in {violations_20_vs_30} dates"
                f" ({pct:.2f}%)"
            )


# =============================================================================
# 3. REPORTING CHECK: Position Counts
# =============================================================================

def print_position_counts(
    weights_df: pd.DataFrame, portfolio_names: list
) -> None:
    """Calculates and prints the minimum and maximum active position counts

    per model and portfolio level across time.
    """
    print("\n--- POSITION COUNT SUMMARY ---\n")
    models = weights_df["model"].unique()

    for model in models:
        counts = []
        for p_name in portfolio_names:
            sub = weights_df[
                (weights_df["model"] == model)
                & (weights_df["portfolio"] == p_name)
            ]

            if sub.empty:
                counts.append(f"{p_name} = No Data")
                continue

            actual = sub.groupby("date")["ticker"].nunique()

            # Clean label formatting (e.g., 'TOP 10%')
            label = (
                p_name.replace("long_only_", "")
                .replace("_inverse_volatility", "")
                .replace("_signal_weight", "")
                .replace("_risk_parity", "")
                .replace("_", " ")
                .upper()
            )

            counts.append(f"{label} = {actual.min()}–{actual.max()}")

        print(f"✓ {model}: " + " | ".join(counts))


# =============================================================================
# MAIN ORCHESTRATOR: Unified Interface
# =============================================================================

def validate_portfolio(
    weights_df: pd.DataFrame,
    scheme_name: str,
    selection_levels: list = ["top_10", "top_20", "top_30"],
) -> None:
    """Main orchestrator function. Calls all sub-validations sequentially

    for any weighting scheme.

    Parameters
    ----------
    weights_df : pd.DataFrame
        DataFrame containing portfolio weights. Must include columns:
        ['date', 'ticker', 'model', 'portfolio', 'weight'].
    scheme_name : str
        The weighting methodology identifier suffix used in the 'portfolio' column.
        Supported options include:
            - 'signal_weight'
            - 'inverse_volatility'
            - 'risk_parity'
    selection_levels : list, default=['top_10', 'top_20', 'top_30']
        List of selection level prefixes ordered from most concentrated to least.
    """
    print("=" * 80)
    print(f"PORTFOLIO VALIDATION REPORT: {scheme_name.upper()}")
    print("=" * 80)

    # Step 1: Run core integrity assertions
    validate_core_integrity(weights_df)

    # Reconstruct expected portfolio column names dynamically
    portfolio_names = [
        f"long_only_{level}_{scheme_name}" for level in selection_levels
    ]

    # Step 2: Print position counts report
    print_position_counts(weights_df, portfolio_names)

    # Step 3: Print weight ranges and run concentration assertions
    print("\n--- WEIGHT RANGE & CONCENTRATION ---\n")
    for p_name in portfolio_names:
        p_weights = weights_df.loc[weights_df["portfolio"] == p_name, "weight"]
        if not p_weights.empty:
            print(
                f"✓ {p_name}: Range = [{p_weights.min():.6f},"
                f" {p_weights.max():.6f}]"
            )

    validate_concentration_monotonicity(weights_df, portfolio_names)
    print("\n" + "=" * 80)


# =============================================================================
# MODULAR FUNTIONS (Especifically for one portfolio)
# =============================================================================


def validate_inverse_volatility_weights(
    weights_df: pd.DataFrame, rolling_volatility: pd.DataFrame
) -> None:
    """Validates that assigned portfolio weights strictly match the theoretical

    inverse volatility allocation: w_i = (1 / vol_i) / sum(1 / vol_k).
    """
    max_error = 0.0

    # Group by portfolio rebalance date, model, and portfolio type
    for (date, model, portfolio), group in weights_df.groupby(
        ["date", "model", "portfolio"]
    ):
        tickers = group["ticker"].tolist()
        actual_weights = group["weight"].to_numpy()

        # Extract underlying rolling volatilities for the exact date and tickers
        volatilities = rolling_volatility.loc[date, tickers].to_numpy()

        # Re-calculate expected theoretical weights independently
        expected_weights = 1.0 / volatilities
        expected_weights /= expected_weights.sum()

        # Compute maximum point-in-time absolute discrepancy
        error = np.max(np.abs(actual_weights - expected_weights))
        max_error = max(max_error, error)

    # Assert mathematical exactness within numerical precision tolerance
    assert np.isclose(max_error, 0.0, atol=1e-10), (
        "Inverse Volatility Error: Assigned weights do not match the expected "
        f"inverse-volatility allocation. Max error = {max_error:.2e}"
    )

    print(
        "\n--- SPECIFIC MODEL CHECK: INVERSE VOLATILITY ---"
    )
    print()
    print(
        "✓ Mathematical Exactness Passed: Weights match inverse volatility"
        " allocation."
    )
    print(f"✓ Maximum absolute numerical discrepancy = {max_error:.2e}")


def validate_equal_risk_contribution_weights(
    weights_df: pd.DataFrame,
    covariances: dict,
    compute_risk_contributions_fn,
    scheme_name: str = "risk_parity",
    tolerance_threshold: float = 0.03,
) -> pd.DataFrame:
    """Validates that assigned portfolio weights achieve equal risk contribution across assets:

    RC_i = Total_Risk / N.

    Parameters
    ----------
    weights_df : pd.DataFrame
        DataFrame containing portfolio weights with columns:
        ['date', 'ticker', 'model', 'portfolio', 'weight'].
    covariances : dict
        Dictionary mapping (date, model, portfolio) tuples to their corresponding
        covariance matrix (np.ndarray or pd.DataFrame).
    compute_risk_contributions_fn : callable
        Function that accepts (weights, covariance_matrix) and returns
        the array of risk contributions per asset.
    scheme_name : str, default='risk_parity'
        Filter string to isolate target portfolios within the 'portfolio'
        column.
    tolerance_threshold : float, default=0.01
        Maximum acceptable relative deviation threshold (e.g., 0.01 = 1%) for
        assertions.

    Returns
    -------
    pd.DataFrame
        DataFrame containing detailed relative error metrics per date, model,
        and portfolio.
    """
    # 1. Filter target portfolios
    target_df = weights_df[weights_df["portfolio"].str.contains(scheme_name)]
    assert (
        not target_df.empty
    ), f"Equal Risk Contribution Error: No portfolio found containing '{scheme_name}'."

    risk_contribution_errors = []

    # 2. Iterate over rebalance observations
    for (date, model, portfolio), group in target_df.groupby(
        ["date", "model", "portfolio"]
    ):
        weights = group["weight"].to_numpy()

        cov_key = (date, model, portfolio)
        assert (
            cov_key in covariances
        ), f"Covariance Error: Key {cov_key} not found in covariances dictionary."

        covariance_matrix = np.asarray(covariances[cov_key])

        # Dimension validation
        assert covariance_matrix.shape == (
            len(weights),
            len(weights),
        ), f"Dimension Mismatch Error: Covariance shape {covariance_matrix.shape} != portfolio size ({len(weights)}) on {date} for {portfolio}."

        # Compute risk contributions using supplied helper
        risk_contributions = compute_risk_contributions_fn(
            weights, covariance_matrix
        )

        # Target risk contribution per asset (Total Risk / N)
        target_risk = risk_contributions.sum() / len(risk_contributions)

        # Maximum relative deviation from target risk contribution
        max_relative_error = (
            np.max(np.abs(risk_contributions - target_risk)) / target_risk
        )

        risk_contribution_errors.append(
            {
                "date": date,
                "model": model,
                "portfolio": portfolio,
                "max_relative_error": max_relative_error,
            }
        )

    errors_df = pd.DataFrame(risk_contribution_errors)

    # 3. Overall Summary Report
    print("\n--- SPECIFIC MODEL CHECK: EQUAL RISK CONTRIBUTION ---")
    print(f"✓ Portfolios validated = {len(errors_df):,}")
    print(
        f"✓ Mean relative deviation = {errors_df['max_relative_error'].mean():.6%}"
    )
    print(
        f"✓ Median relative deviation = {errors_df['max_relative_error'].median():.6%}"
    )
    print(
        "✓ 95th percentile relative deviation ="
        f" {errors_df['max_relative_error'].quantile(0.95):.6%}"
    )
    print(
        "✓ Maximum relative deviation ="
        f" {errors_df['max_relative_error'].max():.6%}"
    )

    # 4. Summary by Selection Level
    summary_by_selection = (
        errors_df.assign(
            selection=errors_df["portfolio"].str.extract(r"(top_\d+)")[0]
        )
        .groupby("selection")["max_relative_error"]
        .agg(
            mean="mean",
            median="median",
            p95=lambda x: x.quantile(0.95),
            max="max",
        )
    )

    print("\n--- RISK CONTRIBUTION DEVIATION BY SELECTION LEVEL ---")
    print(
        summary_by_selection.to_string(
            float_format=lambda x: f"{x:.3%}"
        )
    )

    # 5. Safety assertion against extreme solver convergence failures
    max_error = errors_df["max_relative_error"].max()
    assert max_error < tolerance_threshold, (
        "Equal Risk Contribution Error: Optimization solver failed to converge "
        f"within tolerance. Observed max error = {max_error:.2%} (Threshold = {tolerance_threshold:.2%})."
    )

    return errors_df

# =============================================================================
# GLOBAL PORTFOLIO VALIDATION
# =============================================================================

from typing import Dict, Any


def validate_long_only_portfolios(
    df: pd.DataFrame, 
    max_position_weight: float = 0.05, 
    tolerance: float = 1e-10
) -> Dict[str, Any]:
    """Validates Long-Only portfolio constraints and outputs summary logs."""
    if df.empty:
        print("No Long-Only portfolios found in DataFrame.")
        return {}

    grouped = df.groupby(["date", "model", "portfolio"])["weight"]

    max_weights = grouped.max()
    weight_sums = grouped.sum()

    max_violations = (max_weights > max_position_weight + tolerance).sum()
    negative_violations = (df["weight"] < -tolerance).sum()
    sum_error_max = np.abs(weight_sums - 1.0).max()

    metrics = {
        "max_weight": max_weights.max(),
        "max_violations": max_violations,
        "negative_violations": negative_violations,
        "max_weight_sum_error": sum_error_max,
    }

    print("=" * 80)
    print("MAXIMUM POSITION WEIGHT — LONG-ONLY VALIDATION")
    print("=" * 80)
    print(f"✓ Maximum position weight     = {metrics['max_weight']:.6f}")
    print(f"✓ Maximum position violations = {metrics['max_violations']}")
    print(f"✓ Negative weight violations  = {metrics['negative_violations']}")
    print(f"✓ Maximum weight-sum error    = {metrics['max_weight_sum_error']:.2e}")
    print()

    return metrics


def validate_long_short_portfolios(
    df: pd.DataFrame, 
    max_position_weight: float = 0.05, 
    tolerance: float = 1e-10
) -> Dict[str, Any]:
    """Validates Long-Short portfolio constraints and exposures."""
    if df.empty:
        print("No Long-Short portfolios found in DataFrame.")
        return {}

    grouped = df.groupby(["date", "model", "portfolio"])["weight"]

    long_exposure = grouped.apply(lambda x: x[x > 0].sum())
    short_exposure = grouped.apply(lambda x: x[x < 0].sum())
    gross_exposure = grouped.apply(lambda x: np.abs(x).sum())
    net_exposure = grouped.sum()

    max_abs_weights = grouped.apply(lambda x: np.abs(x).max())

    long_positions = grouped.apply(lambda x: (x > 0).sum())
    short_positions = grouped.apply(lambda x: (x < 0).sum())

    metrics = {
        "max_abs_weight": max_abs_weights.max(),
        "max_violations": (max_abs_weights > max_position_weight + tolerance).sum(),
        "long_exposure_violations": (np.abs(long_exposure - 0.50) > tolerance).sum(),
        "short_exposure_violations": (np.abs(short_exposure + 0.50) > tolerance).sum(),
        "gross_exposure_violations": (np.abs(gross_exposure - 1.00) > tolerance).sum(),
        "net_exposure_violations": (np.abs(net_exposure) > tolerance).sum(),
        "min_long_positions": long_positions.min(),
        "min_short_positions": short_positions.min(),
    }

    print("=" * 80)
    print("MAXIMUM POSITION WEIGHT — LONG-SHORT VALIDATION")
    print("=" * 80)
    print(f"✓ Maximum absolute position weight = {metrics['max_abs_weight']:.6f}")
    print(f"✓ Maximum position violations     = {metrics['max_violations']}")
    print(f"✓ Long exposure violations        = {metrics['long_exposure_violations']}")
    print(f"✓ Short exposure violations       = {metrics['short_exposure_violations']}")
    print(f"✓ Gross exposure violations       = {metrics['gross_exposure_violations']}")
    print(f"✓ Net exposure violations         = {metrics['net_exposure_violations']}")
    print(f"✓ Minimum long positions          = {metrics['min_long_positions']}")
    print(f"✓ Minimum short positions         = {metrics['min_short_positions']}")
    print()

    return metrics


def validate_all_portfolios(
    df: pd.DataFrame, 
    max_position_weight: float = 0.05, 
    tolerance: float = 1e-10
) -> Dict[str, Dict[str, Any]]:
    """Master function to validate the full portfolio DataFrame containing both

    Long-Only and Long-Short portfolio structures.
    """
    df_long_only = df[df["portfolio"].str.startswith("long_only_")].copy()
    df_long_short = df[df["portfolio"].str.startswith("long_short_")].copy()

    lo_results = validate_long_only_portfolios(
        df_long_only, max_position_weight, tolerance
    )
    ls_results = validate_long_short_portfolios(
        df_long_short, max_position_weight, tolerance
    )

    return {
        "long_only": lo_results,
        "long_short": ls_results,
    }


def validate_minimum_effective_weight(
    df: pd.DataFrame, 
    min_effective_weight: float = 0.005, 
    tolerance: float = 1e-10
) -> Dict[str, Any]:
    """Validates minimum effective position weight constraints across all portfolios.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns: ['date', 'model', 'portfolio', 'weight']
    min_effective_weight : float, default=0.005
        The minimum weight threshold below which non-zero positions are invalid.
    tolerance : float, default=1e-10
        Numerical tolerance for floating point comparisons.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing all summary metrics and violation counts.
    """
    if df.empty:
        print("No portfolios found in DataFrame for minimum weight validation.")
        return {}

    abs_weights = df["weight"].abs()

    # Position checks
    effective_positions_mask = abs_weights >= (min_effective_weight - tolerance)
    small_positions_mask = (abs_weights > 1e-12) & (abs_weights < (min_effective_weight - tolerance))

    n_effective_positions = int(effective_positions_mask.sum())
    n_small_positions = int(small_positions_mask.sum())

    # --- Long-Only subset ---
    df_long_only = df[df["portfolio"].str.startswith("long_only_")]
    lo_negative_violations = 0
    max_lo_exposure_error = 0.0

    if not df_long_only.empty:
        lo_grouped = df_long_only.groupby(["date", "model", "portfolio"])["weight"]
        lo_weight_sum = lo_grouped.sum()
        
        lo_negative_violations = int((df_long_only["weight"] < -tolerance).sum())
        max_lo_exposure_error = float(np.abs(lo_weight_sum - 1.0).max())

    # --- Long-Short subset ---
    df_long_short = df[df["portfolio"].str.startswith("long_short_")]
    max_ls_long_error = 0.0
    max_ls_short_error = 0.0
    max_ls_gross_error = 0.0
    max_ls_net_error = 0.0

    if not df_long_short.empty:
        ls_grouped = df_long_short.groupby(["date", "model", "portfolio"])["weight"]
        
        ls_long_exp = ls_grouped.apply(lambda x: x[x > 0].sum())
        ls_short_exp = ls_grouped.apply(lambda x: x[x < 0].sum())
        ls_gross_exp = ls_grouped.apply(lambda x: np.abs(x).sum())
        ls_net_exp = ls_grouped.sum()

        max_ls_long_error = float(np.abs(ls_long_exp - 0.50).max())
        max_ls_short_error = float(np.abs(ls_short_exp + 0.50).max())
        max_ls_gross_error = float(np.abs(ls_gross_exp - 1.00).max())
        max_ls_net_error = float(np.abs(ls_net_exp).max())

    # Consolidation
    metrics = {
        "min_effective_weight": min_effective_weight,
        "positions_below_threshold": n_small_positions,
        "effective_positions_retained": n_effective_positions,
        "long_only_negative_violations": lo_negative_violations,
        "max_long_only_exposure_error": max_lo_exposure_error,
        "max_long_short_long_error": max_ls_long_error,
        "max_long_short_short_error": max_ls_short_error,
        "max_long_short_gross_error": max_ls_gross_error,
        "max_long_short_net_error": max_ls_net_error,
    }

    # Logging
    print("=" * 80)
    print("MINIMUM EFFECTIVE POSITION WEIGHT — VALIDATION")
    print("=" * 80)
    print(f"✓ Minimum effective weight              = {min_effective_weight:.2%}")
    print(f"✓ Positions below threshold             = {metrics['positions_below_threshold']:,}")
    print(f"✓ Long-only negative weights            = {metrics['long_only_negative_violations']}")
    print(f"✓ Maximum long-only exposure error      = {metrics['max_long_only_exposure_error']:.2e}")
    print(f"✓ Maximum long-short long exposure error= {metrics['max_long_short_long_error']:.2e}")
    print(f"✓ Maximum long-short short exposure err = {metrics['max_long_short_short_error']:.2e}")
    print(f"✓ Maximum long-short gross exposure err = {metrics['max_long_short_gross_error']:.2e}")
    print(f"✓ Maximum long-short net exposure error   = {metrics['max_long_short_net_error']:.2e}")
    print()
    print(f"✓ Effective positions retained          = {metrics['effective_positions_retained']:,}")
    print()

    return metrics