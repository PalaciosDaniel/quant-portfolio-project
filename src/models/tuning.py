from typing import Any, Callable, Type

import numpy as np
import optuna
import pandas as pd
from sklearn.base import RegressorMixin
from typing import Literal

from .training import run_cpcv_training

# =============================================================================
# Hyperparameter Optimization Pipeline
# =============================================================================

def optimize_hyperparameters(
    model_cls: Type[RegressorMixin],
    parameter_space: Callable[[optuna.trial.Trial], dict[str, Any]],
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_trials: int = 50,
    rank_method: Literal["pearson", "spearman"] = "spearman",
    scoring_metric: str = "Rank IC Mean",
    direction: str = "maximize",
    random_state: int = 42,
) -> tuple[optuna.study.Study, dict[str, Any], float]:
    """
    Optimize model hyperparameters using Optuna and CPCV evaluation.

    Parameters
    ----------
    model_cls : Type[RegressorMixin]
        Regressor class following the scikit-learn API.

    parameter_space : Callable
        Function receiving an Optuna Trial and returning a dictionary
        of hyperparameters to evaluate.

    X : pd.DataFrame
        Feature matrix indexed by ('date', 'ticker').

    y : pd.Series
        Target vector indexed by ('date', 'ticker').

    splits : list[tuple[np.ndarray, np.ndarray]]
        CPCV train/validation splits.

    n_trials : int, default=50
        Number of Optuna optimization trials.

    rank_method : {"pearson", "spearman"}, default="spearman"
        Correlation method used to compute the Information Coefficient.

    scoring_metric : str, default="Rank IC Mean"
        Performance metric optimized during the search.

    direction : {"maximize", "minimize"}, default="maximize"
        Optimization direction.

    random_state : int, default=42
        Random seed for the Optuna sampler.

    Returns
    -------
    study : optuna.study.Study
        Complete Optuna optimization study.

    best_params : dict[str, Any]
        Best hyperparameter configuration found.

    best_score : float
        Best value obtained for the selected scoring metric.
    """

    # -------------------------------------------------------------------------
    # Create Optuna study
    # -------------------------------------------------------------------------

    sampler = optuna.samplers.TPESampler(
        seed=random_state
    )

    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
    )

    # -------------------------------------------------------------------------
    # Define optimization objective
    # -------------------------------------------------------------------------

    def objective(trial: optuna.trial.Trial) -> float:

        params = parameter_space(trial)

        _, _, aggregated_metrics = run_cpcv_training(
            model_cls=model_cls,
            model_params=params,
            X=X,
            y=y,
            splits=splits,
            rank_method=rank_method,
        )

        score = aggregated_metrics[scoring_metric]

        return score

    # -------------------------------------------------------------------------
    # Run optimization
    # -------------------------------------------------------------------------

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    return (
        study,
        study.best_params,
        study.best_value,
    )