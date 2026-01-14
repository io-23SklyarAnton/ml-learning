from typing import Optional

import numpy as np


def standard_scale(
        X: np.ndarray
) -> np.ndarray:
    scaled_X: Optional[np.ndarray] = None

    for i in range(X.shape[1]):
        feature = X[:, i:i + 1]
        mean = np.mean(feature)
        std = np.std(feature)

        feature = feature - mean
        feature = feature / std

        if scaled_X is None:
            scaled_X = feature
        else:
            scaled_X = np.hstack((scaled_X, feature))

    return scaled_X
