import numpy as np

from cnn.layers.base import Layer
from optimizers import OptimizerFactory


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
