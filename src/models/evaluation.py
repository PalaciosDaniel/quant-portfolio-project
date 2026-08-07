"""
High-level evaluation utilities for supervised learning models.

This module provides functions that combine multiple evaluation metrics
into a unified scoring interface.
"""

import pandas as pd

from .metrics.error_metrics import compute_error_metrics
from .metrics.ranking_metrics import compute_ic_metrics


# =============================================================================
# Prediction Evaluator
# =============================================================================

def evaluate_predictions(
    df_eval: pd.DataFrame,
    pred_col: str = "pred",
    target_col: str = "target",
    rank_method: str = "spearman",
) -> dict[str, float]:
    """
    Evaluate model predictions using both regression and ranking metrics.

    Parameters
    ----------
    df_eval : pd.DataFrame
        DataFrame indexed by ('date', 'ticker') containing predictions
        and target values.

    pred_col : str, default="pred"
        Prediction column.

    target_col : str, default="target"
        Target column.

    rank_method : {"pearson", "spearman"}, default="spearman"
        Correlation method used to compute the Information Coefficient.
        - "pearson"  : Linear Information Coefficient (IC)
        - "spearman" : Rank Information Coefficient (Rank IC)

    Returns
    -------
    dict[str, float]
        Dictionary containing all evaluation metrics.
    """

    # Validate required columns
    required = {pred_col, target_col}
    missing = required - set(df_eval.columns)

    if missing:
        raise KeyError(
            f"Missing required columns: {missing}"
        )

    # -------------------------------------------------------------------------
    # Compute regression error metrics
    # -------------------------------------------------------------------------

    error_metrics = compute_error_metrics(
        y_true=df_eval[target_col],
        y_pred=df_eval[pred_col],
    )

    # -------------------------------------------------------------------------
    # Compute Information Coefficient metrics
    # -------------------------------------------------------------------------

    _, ic_summary = compute_ic_metrics(
        df_eval=df_eval,
        pred_col=pred_col,
        target_col=target_col,
        method=rank_method,
    )

    # Use an appropriate prefix depending on the correlation method
    prefix = "Rank IC" if rank_method == "spearman" else "IC"

    ranking_metrics = {
        f"{prefix} {metric.replace('IC ', '')}": value
        for metric, value in ic_summary.items()
    }

    # -------------------------------------------------------------------------
    # Merge all evaluation metrics
    # -------------------------------------------------------------------------

    results = {}
    results.update(error_metrics)
    results.update(ranking_metrics)

    return results