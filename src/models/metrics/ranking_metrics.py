import numpy as np
import pandas as pd

# =============================================================================
# Cross-Sectional Information Coefficient
# =============================================================================

def compute_daily_ic(
    df_eval: pd.DataFrame,
    pred_col: str = "pred",
    target_col: str = "target",
    method: str = "spearman",
) -> pd.Series:
    """
    Compute the daily cross-sectional Information Coefficient (IC).

    Parameters
    ----------
    df_eval : pd.DataFrame
        DataFrame indexed by ('date', 'ticker') containing predictions
        and target values.

    pred_col : str, default="pred"
        Prediction column.

    target_col : str, default="target"
        Target column.

    method : {"pearson", "spearman"}, default="spearman"
        Correlation method.
        - "pearson"   : Linear Information Coefficient (IC)
        - "spearman"  : Rank Information Coefficient (Rank IC)

    Returns
    -------
    pd.Series
        Daily IC values indexed by date.
    """

    # Validate correlation method
    if method not in {"pearson", "spearman"}:
        raise ValueError(
            "method must be either 'pearson' or 'spearman'."
        )

    # Check required columns
    required = {pred_col, target_col}
    missing = required - set(df_eval.columns)

    if missing:
        raise KeyError(
            f"Missing required columns: {missing}"
        )

    # Compute daily cross-sectional correlation
    ic_series = (
        df_eval
        .groupby(level="date")
        .apply(
            lambda g: g[pred_col].corr(
                g[target_col],
                method=method
            )
        )
        .dropna()
    )

    return ic_series


# =============================================================================
# IC Summary Statistics
# =============================================================================

def compute_ic_summary(
    ic_series: pd.Series,
) -> dict[str, float]:
    """
    Compute summary statistics for a daily IC series.

    Parameters
    ----------
    ic_series : pd.Series
        Daily Information Coefficient values.

    Returns
    -------
    dict[str, float]
        Summary statistics.
    """

    ic_mean = float(ic_series.mean())
    ic_median = float(ic_series.median())
    ic_std = float(ic_series.std())

    ic_ir = (
        ic_mean / ic_std
        if ic_std != 0
        else 0.0
    )

    hit_rate = float((ic_series > 0).mean() * 100)

    ic_tstat = (
        ic_mean
        / (ic_std / np.sqrt(len(ic_series)))
        if ic_std != 0
        else 0.0
    )

    return {
        "IC Mean": ic_mean,
        "IC Median": ic_median,
        "IC Std": ic_std,
        "IC Information Ratio": ic_ir,
        "Hit Rate (%)": hit_rate,
        "IC t-statistic": ic_tstat,
    }


# =============================================================================
# Complete IC Evaluation
# =============================================================================

def compute_ic_metrics(
    df_eval: pd.DataFrame,
    pred_col: str = "pred",
    target_col: str = "target",
    method: str = "spearman",
) -> tuple[pd.Series, dict[str, float]]:
    """
    Compute daily IC values together with summary statistics.

    Parameters
    ----------
    df_eval : pd.DataFrame
        Evaluation DataFrame.

    pred_col : str, default="pred"
        Prediction column.

    target_col : str, default="target"
        Target column.

    method : {"pearson", "spearman"}, default="spearman"
        Correlation method.

    Returns
    -------
    tuple
        (
            daily_ic : pd.Series,
            summary : dict[str, float]
        )
    """

    daily_ic = compute_daily_ic(
        df_eval=df_eval,
        pred_col=pred_col,
        target_col=target_col,
        method=method,
    )

    summary = compute_ic_summary(daily_ic)

    return daily_ic, summary