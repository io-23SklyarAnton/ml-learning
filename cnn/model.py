import abc

import numpy as np
from scipy.signal import correlate, convolve

from optimizers import OptimizerFactory


class Layer(abc.ABC):
    def forward_pass(self, x: np.ndarray[np.float64]) -> np.ndarray[np.float64]: ...

    def backward_pass(self, delta: np.ndarray[np.float64]) -> np.ndarray[np.float64]: ...


class ReLU(Layer):
    def __init__(self):
        self.cache = None

    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        self.cache = x
        return np.maximum(0, x)

    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        z = self.cache
        d_z = delta * (z > 0).astype(float)
        return d_z


class Convolution(Layer):
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
        self._optimizers = []
        self._bias_optimizers = []

        for _ in range(self.out_channels):
            self._optimizers.append(
                [optimizer_factory.create_optimizer() for _ in range(self.in_channels)]
            )
            self._bias_optimizers.append(optimizer_factory.create_optimizer())

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
            raw_db = np.sum(delta_i)

            step_b = self._bias_optimizers[i].compute_step(raw_db)
            d_b[i] = step_b

            for c in range(self.in_channels):
                d_x[c] += convolve(delta_i, self._filters[i, c], mode='full')

            for c in range(self.in_channels):
                optimizer = self._optimizers[i][c]
                d_w[i, c] = optimizer.compute_step(correlate(self._x[c], delta_i, mode='valid'))

        self._filters -= self._learning_rate * d_w
        self._bias -= self._learning_rate * d_b

        return d_x


class MaxPooling(Layer):
    def __init__(
            self,
            pool_size: int,
            stride: int,
    ):
        self.pool_size = pool_size
        self.stride = stride
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

        if self.stride != self.pool_size:
            raise NotImplementedError()

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
        d_x[:, :h_out * pool_h, :w_out * pool_w] = d_x_flat

        return d_x


class Softmax(Layer):
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


class Flatten(Layer):
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


class Dense(Layer):
    def __init__(
            self,
            optimizer_factory: OptimizerFactory,
            learning_rate: float,
            in_features: int,
            out_features: int
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self._learning_rate = learning_rate

        self._initialize_weights()
        self._initialize_optimizers(optimizer_factory)

        self._x = None

    def _initialize_weights(self) -> None:
        scale = np.sqrt(2.0 / self.in_features)

        self._weights = np.random.normal(
            loc=0.0,
            scale=scale,
            size=(self.out_features, self.in_features)
        )

        self._bias = np.zeros((self.out_features, 1))

    def _initialize_optimizers(
            self,
            optimizer_factory: OptimizerFactory
    ) -> None:
        self._weight_optimizer = optimizer_factory.create_optimizer()
        self._bias_optimizer = optimizer_factory.create_optimizer()

    def forward_pass(
            self,
            x: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        self._x = x.reshape(-1, 1)
        z = (self._weights @ self._x) + self._bias

        return z.ravel()

    def backward_pass(
            self,
            delta: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        delta = delta.reshape(-1, 1)

        d_w = delta @ self._x.T
        d_b = delta

        d_x = self._weights.T @ delta

        step_w = self._weight_optimizer.compute_step(d_w)
        step_b = self._bias_optimizer.compute_step(d_b)

        self._weights -= self._learning_rate * step_w
        self._bias -= self._learning_rate * step_b

        return d_x.ravel()
