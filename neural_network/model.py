import dataclasses
import numpy as np


@dataclasses.dataclass
class LayerSize:
    n_neurons: int


class NeuralNetwork:
    def __init__(
            self,
            layer_sizes: list[LayerSize],
            learning_rate: float,
    ):
        self._layer_sizes = layer_sizes
        self._learning_rate = learning_rate
        self._init_weights(layer_sizes)
        self._init_biases(layer_sizes)

    def _sigmoid(
            self,
            z: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        return 1 / (1 + np.exp(-1 * z))

    def _init_weights(
            self,
            layer_sizes: list[LayerSize],
    ) -> None:
        self._weights = []
        for i in range(len(layer_sizes) - 1):
            current_layer_size = layer_sizes[i]
            next_layer = layer_sizes[i + 1]
            size = (next_layer.n_neurons, current_layer_size.n_neurons)

            self._weights.append(np.random.normal(loc=0.0, scale=1.0, size=size))

    def _init_biases(
            self,
            layer_sizes: list[LayerSize],
    ) -> None:
        self._biases = []
        for i in range(len(layer_sizes) - 1):
            next_layer = layer_sizes[i + 1]
            size = (next_layer.n_neurons, 1)

            self._biases.append(np.random.normal(loc=0.0, scale=1.0, size=size))

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
        return self._get_activations(x)[-1]

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
    ):
        x = X[i_sample]
        y = Y[i_sample]
        activations = self._get_activations(x)

        deltas = self._get_deltas(activations=activations, y=y)

        self._update_weights(
            activations=activations,
            deltas=deltas,
        )

    def _get_activations(
            self,
            x: np.ndarray[np.float64],
    ) -> list[np.ndarray[np.float64]]:
        activations = [x]

        for layer_idx in range(len(self._weights)):
            a = self._forward_pass(
                W=self._weights[layer_idx],
                x=activations[-1],
                b=self._biases[layer_idx],
            )
            activations.append(a)

        return activations

    def _get_deltas(
            self,
            activations: list[np.ndarray[np.float64]],
            y: np.ndarray[np.float64]
    ) -> list[np.ndarray[np.float64]]:
        deltas = [
            (activations[-1] - y) * activations[-1] * (1 - activations[-1])
        ]
        for layer_idx in range(len(self._layer_sizes) - 2, -1, -1):
            sigmoid_prime = activations[layer_idx] * (1 - activations[layer_idx])
            delta = (self._weights[layer_idx + 1].T @ deltas[-1]) * sigmoid_prime
            deltas.append(delta)

        return deltas

    def _update_weights(
            self,
            activations: list[np.ndarray[np.float64]],
            deltas: list[np.ndarray[np.float64]],
    ):
        for i in range(len(deltas)):
            delta = deltas[i]
            layer_idx = -1 * (i + 1)

            a = activations[layer_idx - 1]
            W = self._weights[layer_idx]
            b = self._biases[layer_idx]

            d_w = delta @ a.T

            self._weights[layer_idx] = W - self._learning_rate * d_w
            self._biases[layer_idx] = b - self._learning_rate * delta

    def _forward_pass(
            self,
            W: np.ndarray[np.float64],
            x: np.ndarray[np.float64],
            b: np.ndarray[np.float64],
    ) -> np.ndarray[np.float64]:
        return self._sigmoid(
            (W @ x) + b
        )
