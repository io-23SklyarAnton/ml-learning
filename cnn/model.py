from cnn import layers

import numpy as np


class CNN:
    def __init__(
            self,
            layers_architecture: list[layers.Base],
    ):
        self._layers: list[layers.Base] = layers_architecture

    def fit(
            self,
            X: np.ndarray[np.float64],
            Y: np.ndarray[np.float64],
            n_epochs: int,
    ) -> None:
        for _ in range(n_epochs):
            self._process_epoch(
                X=X,
                Y=Y,
            )

    def predict(
            self,
            x: np.ndarray[np.float64],
    ) -> np.ndarray[np.float64]:
        return self._forward_pass(x)

    def _process_epoch(
            self,
            X: np.ndarray[np.float64],
            Y: np.ndarray[np.float64],
    ) -> None:
        for i_sample in range(len(Y)):
            self._process_sample(
                X=X,
                Y=Y,
                i_sample=i_sample,
            )

    def _process_sample(
            self,
            X: np.ndarray[np.float64],
            Y: np.ndarray[np.float64],
            i_sample: int,
    ) -> None:
        x = X[i_sample]
        y = Y[i_sample]

        y_hat: np.ndarray[np.float64] = self._forward_pass(x)
        delta: np.ndarray[np.float64] = -y / (y_hat + 1e-15)

        self._backward_pass(delta)

    def _forward_pass(
            self,
            x: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        for layer in self._layers:
            x = layer.forward_pass(x)

        return x

    def _backward_pass(
            self,
            delta: np.ndarray[np.float64]
    ) -> None:
        for layer in reversed(self._layers):
            delta = layer.backward_pass(delta)
