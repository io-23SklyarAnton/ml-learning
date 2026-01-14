from typing import Optional

import numpy as np


def standard_scale(
        X: np.ndarray
) -> np.ndarray:
    X_float = X.astype(float)

    mean = np.mean(X_float, axis=0)
    std = np.std(X_float, axis=0)

    return (X_float - mean) / std
