"""
High-level evaluation utilities for supervised learning models.

This module provides functions that combine multiple evaluation metrics
into a unified scoring interface.
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

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


# =============================================================================
# Feature Importance Extraction
# =============================================================================

def extract_cpcv_feature_importance(
    model_cls,
    model_params,
    X,
    y,
    splits,
):
    """
    Extract model-specific feature importance across Combinatorial Purged
    Cross-Validation (CPCV) folds.

    Supported model types:
    - Linear Models (e.g., Ridge): Absolute magnitude of model coefficients.
    - XGBoost: Gain-based feature importance.
    - Tree Ensembles (e.g., Random Forest): Impurity-based feature importance.

    Parameters
    ----------
    model_cls : class
        Model class to be instantiated for each fold (e.g., Ridge, XGBRegressor).
    model_params : dict
        Dictionary containing optimal hyperparameters for the model class.
    X : pd.DataFrame
        Predictor features matrix.
    y : pd.Series or pd.DataFrame
        Target variable vector.
    splits : iterable of tuples
        CPCV splits yielding (train_indices, validation_indices) for each fold.

    Returns
    -------
    pd.DataFrame
        DataFrame containing normalized feature importance percentages (0-100%)
        for every feature across all CPCV folds.
    """

    fold_importances = []

    # =========================================================================
    # Process each CPCV fold
    # =========================================================================

    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):

        # ---------------------------------------------------------------------
        # Slice training data for the current fold
        # ---------------------------------------------------------------------

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        # ---------------------------------------------------------------------
        # Instantiate and fit model instance
        # ---------------------------------------------------------------------

        model = model_cls(**model_params)
        model.fit(X_train, y_train)

        # ---------------------------------------------------------------------
        # Extract raw feature importance based on model architecture
        # ---------------------------------------------------------------------

        if hasattr(model, "coef_"):

            # Linear models (e.g., Ridge): Use absolute coefficient values
            coefficients = np.asarray(model.coef_).ravel()
            importance = np.abs(coefficients)

        elif hasattr(model, "get_booster"):

            # XGBoost: Use gain-based feature importance
            booster = model.get_booster()
            gain_importance = booster.get_score(importance_type="gain")

            importance = np.array([
                gain_importance.get(feature, 0.0)
                for feature in X_train.columns
            ])

        elif hasattr(model, "feature_importances_"):

            # Tree Ensembles (e.g., Random Forest): Use impurity-based importance
            importance = model.feature_importances_

        else:

            raise ValueError(
                f"Feature importance extraction is not supported for "
                f"the model type: {type(model).__name__}."
            )

        # ---------------------------------------------------------------------
        # Normalize importance values to sum to 100%
        # ---------------------------------------------------------------------

        total_importance = importance.sum()

        if total_importance > 0:
            importance_pct = (importance / total_importance) * 100
        else:
            importance_pct = importance

        # ---------------------------------------------------------------------
        # Format and store fold results
        # ---------------------------------------------------------------------

        fold_result = {"fold": fold_idx}
        fold_result.update(dict(zip(X_train.columns, importance_pct)))

        fold_importances.append(fold_result)

    # =========================================================================
    # Build final DataFrame
    # =========================================================================

    df_importance = pd.DataFrame(fold_importances)

    return df_importance


def compute_cpcv_shap(
    model_cls,
    model_params,
    X,
    y,
    splits,
    sample_size=3000,
    random_state=42,
):
    """
    Compute SHAP values on sampled validation observations
    across CPCV folds.

    SHAP is used as a diagnostic interpretability tool rather
    than as a performance evaluation. A reduced CPCV split set
    and validation subsampling are used to control computational
    cost.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the original feature values,
        corresponding SHAP values, and CPCV fold identifier.
    """

    rng = np.random.default_rng(random_state)

    shap_results = []

    # =========================================================================
    # CPCV folds
    # =========================================================================

    for fold_idx, (train_idx, val_idx) in enumerate(
        splits,
        start=1,
    ):

        # ---------------------------------------------------------------------
        # Training and validation data
        # ---------------------------------------------------------------------

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        X_val = X.iloc[val_idx]

        # ---------------------------------------------------------------------
        # Sample validation observations
        # ---------------------------------------------------------------------

        n_sample = min(
            sample_size,
            len(X_val),
        )

        selected_idx = rng.choice(
            len(X_val),
            size=n_sample,
            replace=False,
        )

        X_val_sample = X_val.iloc[selected_idx]

        # ---------------------------------------------------------------------
        # Train model
        # ---------------------------------------------------------------------

        model = model_cls(
            **model_params
        )

        model.fit(
            X_train,
            y_train,
        )

        # ---------------------------------------------------------------------
        # Tree SHAP
        # ---------------------------------------------------------------------

        explainer = shap.TreeExplainer(
            model
        )

        shap_values = explainer.shap_values(
            X_val_sample
        )

        # ---------------------------------------------------------------------
        # Build result DataFrame
        # ---------------------------------------------------------------------

        df_shap_fold = X_val_sample.copy()

        shap_columns = [
            f"SHAP_{column}"
            for column in X.columns
        ]

        df_shap_fold[shap_columns] = shap_values

        df_shap_fold["fold"] = fold_idx

        shap_results.append(
            df_shap_fold
        )

    # =========================================================================
    # Combine folds
    # =========================================================================

    df_shap = pd.concat(
        shap_results,
        axis=0,
    )

    return df_shap


# =============================================================================
# SHAP Dependence Plots — Multiple Features
# =============================================================================

def plot_shap_dependence_grid(
    df_shap,
    feature_columns,
    display_names=None,
    model_name="Model",
    figsize=(8, 4),
):
    fig, axes = plt.subplots(
        len(feature_columns),
        1,
        figsize=(figsize[0], figsize[1] * len(feature_columns)),
    )

    if len(feature_columns) == 1:
        axes = [axes]

    for ax, feature in zip(axes, feature_columns):

        display_name = (
            display_names.get(feature, feature)
            if display_names is not None
            else feature
        )

        shap_cols = [f"SHAP_{f}" for f in feature_columns]

        shap.dependence_plot(
            feature,
            df_shap[shap_cols].to_numpy(),      # SHAP values
            df_shap[feature_columns],            # Features original values
            feature_names=feature_columns,
            interaction_index=None,
            ax=ax,
            show=False,
        )

        ax.set_xlabel(display_name, fontsize=12)
        ax.set_ylabel("SHAP value", fontsize=11)
        ax.tick_params(axis="both", labelsize=10)

    fig.suptitle(
        f"SHAP Dependence Plots — {model_name}",
        fontsize=15,
        y=1.02,
    )

    fig.tight_layout()
    plt.show()
