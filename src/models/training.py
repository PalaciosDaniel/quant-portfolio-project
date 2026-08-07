
import numpy as np
import pandas as pd

from sklearn.base import RegressorMixin
from typing import Any, Type
from .evaluation import evaluate_predictions


# =============================================================================
# CPCV Training Pipeline
# =============================================================================

def run_cpcv_training(
    model_cls: Type[RegressorMixin],
    model_params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    rank_method: str = "spearman",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train and evaluate a regression model across all CPCV folds.

    Parameters
    ----------
    model_cls : Type[RegressorMixin]
        Regressor class following the scikit-learn API.

    model_params : dict[str, Any]
        Hyperparameters passed to the estimator constructor.

    X : pd.DataFrame
        Feature matrix indexed by ('date', 'ticker').

    y : pd.Series
        Target vector indexed by ('date', 'ticker').

    splits : list[tuple[np.ndarray, np.ndarray]]
        List of (train_idx, val_idx) tuples produced by
        CombinatorialPurgedCV.

    rank_method : {"pearson", "spearman"}, default="spearman"
        Correlation method used to compute the Information Coefficient.

    Returns
    -------
    tuple
        oof_predictions : pd.DataFrame
            Combined out-of-fold predictions.

        fold_metrics : pd.DataFrame
            Evaluation metrics for each CPCV fold.

        aggregated_metrics : pd.DataFrame
            Mean and standard deviation of every performance metric
            across all folds.
    """

    fold_metrics_list = []
    oof_predictions_list = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):

        # ---------------------------------------------------------------------
        # Build training and validation datasets
        # ---------------------------------------------------------------------

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        # ---------------------------------------------------------------------
        # Train model
        # ---------------------------------------------------------------------

        model = model_cls(**model_params)
        model.fit(X_train, y_train)

        # ---------------------------------------------------------------------
        # Generate validation predictions
        # ---------------------------------------------------------------------

        predictions = model.predict(X_val)

        df_val_eval = pd.DataFrame(
            {
                "target": y_val,
                "pred": predictions,
            },
            index=X_val.index,
        )

        # ---------------------------------------------------------------------
        # Evaluate current fold
        # ---------------------------------------------------------------------

        metrics = evaluate_predictions(
            df_eval=df_val_eval,
            pred_col="pred",
            target_col="target",
            rank_method=rank_method,
        )

        metrics["Fold"] = fold_idx
        metrics["Train Size"] = len(train_idx)
        metrics["Validation Size"] = len(val_idx)

        fold_metrics_list.append(metrics)

        # Store out-of-fold predictions
        df_val_eval["Fold"] = fold_idx
        oof_predictions_list.append(df_val_eval)

    # =========================================================================
    # Consolidate outputs
    # =========================================================================

    # Combine and sort all out-of-fold predictions
    oof_predictions = (
        pd.concat(oof_predictions_list)
        .sort_index()
    )

    # Build fold-level metrics table
    fold_metrics = pd.DataFrame(fold_metrics_list)

    # Aggregate only performance metrics across folds
    aggregated_metrics = (
        fold_metrics
        .drop(
            columns=[
                "Fold",
                "Train Size",
                "Validation Size",
            ]
        )
        .agg(["mean", "std"])
    )

    return (
        oof_predictions,
        fold_metrics,
        aggregated_metrics,
    )