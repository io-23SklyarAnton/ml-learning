import numpy as np


class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(
            self,
            X: np.ndarray,
    ) -> 'MinMaxScaler':
        self.min = np.min(X)
        self.max = np.max(X)

        return self

    def transform(
            self,
            X: np.ndarray
    ) -> np.ndarray:
        if self.min is None or self.max is None:
            raise Exception("Scaler wasn't fitted yet")

        return (X - self.min) / (self.max - self.min)

    def fit_transform(
            self,
            X: np.ndarray,
    ) -> np.ndarray:
        return self.fit(X).transform(X)
