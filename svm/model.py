from typing import Optional

import numpy as np

from svm.smo import SMOSolver


class SVMModel:
    def __init__(
            self,
            C: float,
            tol: float,
            max_passes: int,
            gamma: float,
    ):
        self.__smo = SMOSolver(
            C=C,
            tol=tol,
            max_passes=max_passes,
            gamma=gamma
        )
        self._C: float = C
        self._alphas: Optional[np.ndarray] = None
        self._b: Optional[float] = None
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._m: Optional[int] = None

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
    ):
        self._X = X
        self._y = y
        self._m = len(y)
        self._alphas: np.ndarray = self.__smo.fit(
            X=X,
            y=y
        )
        self._recalculate_bias()

    def predict(
            self,
            u: np.ndarray
    ) -> int:
        if not self.__is_model_fitted():
            raise ValueError("You can't predict the unfitted model")

        prediction: float = self._b
        for i in range(self._m):
            prediction += self._alphas[i] * self._y[i] * self.__smo.kernel(self._X[i], u)

        return 1 if prediction > 0 else -1

    def _recalculate_bias(self) -> None:
        sv_indices = np.where((self._alphas > 1e-5) & (self._alphas < self._C - 1e-5))[0]

        if len(sv_indices) == 0:
            self._b = 0.0
            return

        b_sum = 0.0
        for k in sv_indices:
            prediction_part = 0
            for i in range(self._m):
                prediction_part += self._alphas[i] * self._y[i] * self.__smo.kernel(self._X[i], self._X[k])

            b_sum += (self._y[k] - prediction_part)

        self._b = b_sum / len(sv_indices)

    def __is_model_fitted(self):
        return all((
            self._X is not None,
            self._y is not None,
            self._alphas is not None,
            self._b is not None
        ))
