import numpy as np
from scipy.signal import correlate, convolve

from cnn.layers.base import Base
from optimizers import OptimizerFactory


class Convolution(Base):
    def __init__(
            self,
            optimizer_factory: OptimizerFactory,
            learning_rate: float,
            filter_size: int,
            in_channels: int,
            out_channels: int
    ) -> None:
        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self._learning_rate = learning_rate

        self._initialize_filters()
        self._initialize_optimizers(optimizer_factory)
        self._x = None

    def _initialize_filters(self) -> None:
        fan_in = self.filter_size * self.filter_size * self.in_channels
        scale = np.sqrt(2.0 / fan_in)

        self._filters = np.random.normal(
            loc=0.0,
            scale=scale,
            size=(self.out_channels, self.in_channels, self.filter_size, self.filter_size)
        )

        self._bias = np.zeros((self.out_channels, 1))

    def _initialize_optimizers(
            self,
            optimizer_factory: OptimizerFactory,
    ) -> None:
        self._filters_optimizer = optimizer_factory.create_optimizer()
        self._bias_optimizer = optimizer_factory.create_optimizer()

    def forward_pass(
            self,
            x: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        self._x = x

        h_out = x.shape[1] - self.filter_size + 1
        w_out = x.shape[2] - self.filter_size + 1
        output = np.zeros((self.out_channels, h_out, w_out))

        for i in range(self.out_channels):
            result = correlate(x, self._filters[i], mode='valid')
            output[i] = result[0] + self._bias[i]

        return output

    def backward_pass(
            self,
            delta: np.ndarray[np.float64],
    ) -> np.ndarray[np.float64]:
        d_x = np.zeros_like(self._x)
        d_w = np.zeros_like(self._filters)
        d_b = np.zeros_like(self._bias)

        for i in range(self.out_channels):
            delta_i = delta[i]
            d_b[i] = np.sum(delta_i)

            for c in range(self.in_channels):
                d_x[c] += convolve(delta_i, self._filters[i, c], mode='full')
                d_w[i, c] = correlate(self._x[c], delta_i, mode='valid')

        step_w = self._filters_optimizer.compute_step(d_w)
        step_b = self._bias_optimizer.compute_step(d_b)

        self._filters -= self._learning_rate * step_w
        self._bias -= self._learning_rate * step_b

        return d_x
