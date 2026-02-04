import abc

import numpy as np


class Layer(abc.ABC):
    def forward_pass(self, x: np.ndarray[np.float64]) -> np.ndarray[np.float64]: ...

    def backward_pass(self, delta: np.ndarray[np.float64]) -> np.ndarray[np.float64]: ...
