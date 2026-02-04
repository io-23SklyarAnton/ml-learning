import numpy as np

from cnn.layers.base import Base


class ReLU(Base):
    def __init__(self):
        self.cache = None

    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        self.cache = x
        return np.maximum(0, x)

    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        z = self.cache
        d_z = delta * (z > 0).astype(float)
        return d_z
