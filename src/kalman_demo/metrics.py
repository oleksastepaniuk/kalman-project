import numpy as np
from numpy.typing import NDArray


def rmse(y_true: NDArray[np.floating], y_pred: NDArray[np.floating]) -> float:
    """
    Root Mean Squared Error between two 1D arrays.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError(f"rmse: shape mismatch {y_true.shape} vs {y_pred.shape}")

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
