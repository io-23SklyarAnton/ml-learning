import numpy as np

from cnn.layers.base import Base


class MaxPooling(Base):
    def __init__(
            self,
            pool_size: int,
    ):
        self.pool_size = pool_size
        self.stride = pool_size
        self.cache = None

    def forward_pass(
            self,
            x: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        c, h, w = x.shape

        h_out = ((h - self.pool_size) // self.stride + 1)
        w_out = ((w - self.pool_size) // self.stride + 1)

        h_cropped = h_out * self.pool_size
        w_cropped = w_out * self.pool_size

        x_reshaped = x[:, :h_cropped, :w_cropped].reshape(
            c, h_out, self.pool_size, w_out, self.pool_size
        )

        out = x_reshaped.max(axis=(2, 4))

        self.cache = (x.shape, x_reshaped, out)

        return out

    def backward_pass(self, delta: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        x_shape, x_reshaped, out = self.cache

        out_expanded = out[:, :, np.newaxis, :, np.newaxis]

        mask = (x_reshaped == out_expanded)

        delta_expanded = delta[:, :, np.newaxis, :, np.newaxis]

        d_x_reshaped = delta_expanded * mask

        d_x = np.zeros(x_shape)

        c, h_out, pool_h, w_out, pool_w = x_reshaped.shape
        d_x_flat = d_x_reshaped.reshape(c, h_out * pool_h, w_out * pool_w)
        d_x[:, :h_out * pool_h, :w_out * pool_w] = d_x_flat  # TODO: change after padding implementation

        return d_x
