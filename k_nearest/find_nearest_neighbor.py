import numpy as np


class KNearest:
    def __init__(
            self,
            k: int,
            X: np.ndarray,
            y: np.ndarray
    ):
        self._k = k
        self._X = X.astype(float)

        self._y = y

    def find(
            self,
            x: np.ndarray
    ):
        deltas = self._X - x

        distances_sq = np.sum(deltas ** 2, axis=1)

        distances = np.sqrt(distances_sq)

        sorted_indices = np.argsort(distances)

        k_indices = sorted_indices[:self._k]

        k_nearest_labels = self._y[k_indices]

        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        max_index = np.argmax(counts)

        return unique_labels[max_index]
