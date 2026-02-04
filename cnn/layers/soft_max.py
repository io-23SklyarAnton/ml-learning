import numpy as np

from cnn.layers.base import Base


class Softmax(Base):
    def __init__(self):
        self.cache = None

    def forward_pass(self, x: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        shift_x = x - np.max(x)
        exps = np.exp(shift_x)
        out = exps / np.sum(exps)

        self.cache = out
        return out

    def backward_pass(self, delta: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        jacobian_m = np.diag(self.cache)
        jacobian_m -= np.outer(self.cache, self.cache)
        d_x = np.dot(jacobian_m, delta)

        return d_x
