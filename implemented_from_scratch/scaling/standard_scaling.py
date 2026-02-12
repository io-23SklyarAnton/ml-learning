import numpy as np


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(
            self,
            X: np.ndarray,
    ) -> 'StandardScaler':
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

        self.std_[self.std_ == 0] = 1.0

        return self

    def transform(
            self,
            X: np.ndarray
    ) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise Exception("Scaler wasn't fitted yet")

        return (X - self.mean_) / self.std_

    def fit_transform(
            self,
            X: np.ndarray,
    ) -> np.ndarray:
        return self.fit(X).transform(X)
