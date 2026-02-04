import abc
from typing import Optional

import numpy as np


class Optimizer(abc.ABC):
    @abc.abstractmethod
    def compute_step(self, d_w: np.ndarray) -> np.ndarray: ...


class OptimizerFactory(abc.ABC):
    @abc.abstractmethod
    def create_optimizer(self) -> 'Optimizer': ...


class Adam(Optimizer):
    def __init__(
            self,
            p1: float,
            p2: float,
            epsilon: float
    ):
        self._p1 = p1
        self._p2 = p2
        self._epsilon = epsilon

        self._p: Optional[np.ndarray] = None
        self._r: Optional[np.ndarray] = None
        self._t = 1

    def compute_step(
            self,
            d_w: np.ndarray
    ) -> np.ndarray:
        if self._p is None:
            self._p = np.zeros_like(d_w)
            self._r = np.zeros_like(d_w)

        self._p = self._p1 * self._p + (1 - self._p1) * d_w
        p_hat = self._p / (1 - self._p1 ** self._t)

        self._r = self._p2 * self._r + (1 - self._p2) * (d_w ** 2)
        r_hat = self._r / (1 - self._p2 ** self._t)

        self._t += 1

        return p_hat / (self._epsilon + np.sqrt(r_hat))


class AdamFactory(OptimizerFactory):
    def __init__(
            self,
            p1: float,
            p2: float,
            epsilon: float
    ):
        self._p1 = p1
        self._p2 = p2
        self._epsilon = epsilon

    def create_optimizer(self) -> Optimizer:
        return Adam(
            p1=self._p1,
            p2=self._p2,
            epsilon=self._epsilon
        )
