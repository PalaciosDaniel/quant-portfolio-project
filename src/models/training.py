from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .evaluation import evaluate_predictions


# =============================================================================
# Single-Fold Training
# =============================================================================

def _train_single_fold(
    fold_idx: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    model_cls: Any,
    model_params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    rank_method: str,
) -> tuple[int, pd.DataFrame, dict[str, float]]:
    """
    Train and evaluate a model on a single CPCV fold.

    Parameters
    ----------
    fold_idx : int
        Fold identifier.
    train_idx : np.ndarray
        Integer row positions for training.
    val_idx : np.ndarray
        Integer row positions for validation.
    model_cls : Any
        Estimator class exposing fit and predict methods.
    model_params : dict[str, Any]
        Model initialization parameters.
    X : pd.DataFrame
        Feature matrix indexed by ('date', 'ticker').
    y : pd.Series
        Target vector indexed by ('date', 'ticker').
    rank_method : str
        Correlation method used for IC evaluation.

    Returns
    -------
    tuple
        Fold identifier, validation predictions, and fold metrics.
    """

    # -------------------------------------------------------------------------
    # Subset data for the current fold
    # -------------------------------------------------------------------------

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]

    # -------------------------------------------------------------------------
    # Instantiate and fit model
    # -------------------------------------------------------------------------

    model = model_cls(**model_params)

    model.fit(X_train, y_train)

    # -------------------------------------------------------------------------
    # Generate validation predictions
    # -------------------------------------------------------------------------

    preds = model.predict(X_val)

    # -------------------------------------------------------------------------
    # Build validation evaluation DataFrame
    # -------------------------------------------------------------------------

    df_val_eval = pd.DataFrame(
        {
            "target": y_val.to_numpy(),
            "pred": preds,
        },
        index=X_val.index,
    )

    # -------------------------------------------------------------------------
    # Evaluate current fold
    # -------------------------------------------------------------------------

    metrics = evaluate_predictions(
        df_eval=df_val_eval,
        pred_col="pred",
        target_col="target",
        rank_method=rank_method,
    )

    metrics["fold"] = fold_idx

    # Store fold identifier with predictions
    df_val_eval["fold"] = fold_idx

    return fold_idx, df_val_eval, metrics


# =============================================================================
# CPCV Training Pipeline
# =============================================================================

def run_cpcv_training(
    model_cls: Any,
    model_params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    rank_method: str = "spearman",
    n_jobs: int = 4,
) -> tuple[pd.DataFrame, list[dict[str, float]], dict[str, float]]:
    """
    Train a model across all CPCV folds using parallel fold execution.

    Each CPCV fold is executed independently in parallel. Internal model
    parallelization should therefore be disabled when possible to avoid
    nested parallelism.

    Parameters
    ----------
    model_cls : Any
        Estimator class exposing standard fit and predict methods.
    model_params : dict[str, Any]
        Dictionary of hyperparameters for model initialization.
    X : pd.DataFrame
        Feature matrix indexed by ('date', 'ticker').
    y : pd.Series
        Target vector indexed by ('date', 'ticker').
    splits : list[tuple[np.ndarray, np.ndarray]]
        CPCV train/validation index pairs.
    rank_method : str, default="spearman"
        Method used for rank correlation in IC evaluation.
    n_jobs : int, default=4
        Number of CPCV folds executed in parallel.

    Returns
    -------
    tuple
        - oof_predictions : Combined validation predictions.
        - fold_metrics : Metrics for each CPCV fold.
        - aggregated_metrics : Mean metrics across all folds.
    """

    # =========================================================================
    # Execute CPCV folds in parallel
    # =========================================================================

    results = Parallel(n_jobs=n_jobs)(
        delayed(_train_single_fold)(
            fold_idx=fold_idx,
            train_idx=train_idx,
            val_idx=val_idx,
            model_cls=model_cls,
            model_params=model_params,
            X=X,
            y=y,
            rank_method=rank_method,
        )
        for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1)
    )

    # =========================================================================
    # Collect results
    # =========================================================================

    oof_preds_list = []
    fold_metrics = []

    for fold_idx, df_val_eval, metrics in results:

        oof_preds_list.append(df_val_eval)

        fold_metrics.append(metrics)

    # =========================================================================
    # Concatenate Out-of-Fold Predictions
    # =========================================================================

    oof_predictions = pd.concat(oof_preds_list)

    # =========================================================================
    # Aggregate Fold Metrics
    # =========================================================================

    df_fold_metrics = (
        pd.DataFrame(fold_metrics)
        .sort_values("fold")
        .reset_index(drop=True)
    )

    aggregated_metrics = (
        df_fold_metrics
        .drop(columns=["fold"])
        .mean()
        .to_dict()
    )

    return (
        oof_predictions,
        fold_metrics,
        aggregated_metrics,
    )