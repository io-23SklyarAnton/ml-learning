import numpy as np
import random
from typing import Optional


class SMOSolver:
    def __init__(
            self,
            C: float,
            tol: float,
            max_passes: int,
            gamma: float,
    ):
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.gamma = gamma
        self.alphas: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.m: int = 0
        self.n: int = 0

    def kernel(
            self,
            x1: np.ndarray,
            x2: np.ndarray
    ) -> float:
        diff = x1 - x2
        return np.exp(-self.gamma * np.dot(diff, diff))

    def _predict_score(
            self,
            x_input: np.ndarray
    ) -> float:
        prediction = 0.0
        for i in range(self.m):
            prediction += self.alphas[i] * self.y[i] * self.kernel(self.X[i], x_input)
        return prediction + self.b

    def _calc_error(
            self,
            index: int
    ) -> float:
        return self._predict_score(self.X[index]) - self.y[index]  # noqa

    def _get_random_j(
            self,
            i: int
    ) -> int:
        j = i
        while j == i:
            j = random.randint(0, self.m - 1)
        return j

    def _calc_bounds(
            self,
            i: int,
            j: int
    ) -> tuple[float, float]:
        if self.y[i] != self.y[j]:
            L = max(0.0, self.alphas[j] - self.alphas[i])
            H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
        else:
            L = max(0.0, self.alphas[i] + self.alphas[j] - self.C)
            H = min(self.C, self.alphas[i] + self.alphas[j])
        return L, H

    def _optimize_pair(
            self,
            i: int,
            j: int
    ) -> bool:
        E_i = self._calc_error(i)
        E_j = self._calc_error(j)

        alpha_i_old = self.alphas[i]
        alpha_j_old = self.alphas[j]

        L, H = self._calc_bounds(i, j)
        if L == H:
            return False

        eta = (2 * self.kernel(self.X[i], self.X[j]) -
               self.kernel(self.X[i], self.X[i]) -
               self.kernel(self.X[j], self.X[j]))

        if eta >= 0:
            return False

        self.alphas[j] -= self.y[j] * (E_i - E_j) / eta

        if self.alphas[j] > H:
            self.alphas[j] = H
        elif self.alphas[j] < L:
            self.alphas[j] = L

        if abs(self.alphas[j] - alpha_j_old) < 1e-5:
            return False

        self.alphas[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alphas[j])

        b1 = (self.b - E_i -
              self.y[i] * (self.alphas[i] - alpha_i_old) * self.kernel(self.X[i], self.X[i]) -
              self.y[j] * (self.alphas[j] - alpha_j_old) * self.kernel(self.X[i], self.X[j]))

        b2 = (self.b - E_j -
              self.y[i] * (self.alphas[i] - alpha_i_old) * self.kernel(self.X[i], self.X[j]) -
              self.y[j] * (self.alphas[j] - alpha_j_old) * self.kernel(self.X[j], self.X[j]))

        if 0 < self.alphas[i] < self.C:
            self.b = b1
        elif 0 < self.alphas[j] < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2.0

        return True

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray
    ) -> np.ndarray:
        self.X = X
        self.y = y
        self.m, self.n = X.shape
        self.alphas = np.zeros(self.m)
        self.b = 0.0

        passes = 0
        while passes < self.max_passes:
            num_changed_alphas = 0
            for i in range(self.m):
                E_i = self._calc_error(i)

                if ((self.y[i] * E_i < -self.tol and self.alphas[i] < self.C) or
                        (self.y[i] * E_i > self.tol and self.alphas[i] > 0)):

                    j = self._get_random_j(i)
                    if self._optimize_pair(i, j):
                        num_changed_alphas += 1

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

        return self.alphas
