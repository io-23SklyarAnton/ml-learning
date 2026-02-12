import numpy as np

from implemented_from_scratch.cnn.layers.base import Base


class Flatten(Base):
    def __init__(self):
        self.input_shape = None

    def forward_pass(
            self,
            x: np.ndarray[np.float64],
    ) -> np.ndarray[np.float64]:
        self.input_shape = x.shape

        return x.ravel()

    def backward_pass(
            self,
            delta: np.ndarray[np.float64],
    ) -> np.ndarray[np.float64]:
        return delta.reshape(self.input_shape)
