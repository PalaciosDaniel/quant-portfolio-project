import numpy as np
import pandas as pd

def compute_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute the Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth target values.
    y_pred : np.ndarray
        Predicted target values.

    Returns
    -------
    float
        Root Mean Squared Error.
    """

    # Convert inputs to NumPy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Validate input dimensions
    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape."
        )

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute the Mean Absolute Error (MAE).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth target values.
    y_pred : np.ndarray
        Predicted target values.

    Returns
    -------
    float
        Mean Absolute Error.
    """

    # Convert inputs to NumPy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Validate input dimensions
    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape."
        )

    return float(np.mean(np.abs(y_true - y_pred)))


def compute_error_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Compute the main regression error metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth target values.
    y_pred : np.ndarray
        Predicted target values.

    Returns
    -------
    dict[str, float]
        Dictionary containing RMSE and MAE.
    """

    return {
        "RMSE": compute_rmse(y_true, y_pred),
        "MAE": compute_mae(y_true, y_pred),
    }