import dataclasses
import numpy as np


@dataclasses.dataclass
class Layer:
    n_neurons: int


class NeuroNetwork:
    def __init__(
            self,
            layers: list[Layer],
            nu: float,
    ):
        self._layers = layers
        self._nu = nu
        self.__init_weights(layers)
        self.__init_biases(layers)

    def _sigmoid(
            self,
            input_: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        return 1 / (1 + np.exp(-1 * input_))

    def __init_weights(
            self,
            layers: list[Layer],
    ) -> None:
        self._weights = []
        for i in range(len(layers) - 1):
            layer = layers[i]
            next_layer = layers[i + 1]
            size = (next_layer.n_neurons, layer.n_neurons)

            self._weights.append(np.random.normal(loc=0.0, scale=1.0, size=size))

    def __init_biases(
            self,
            layers: list[Layer],
    ) -> None:
        self._biases = []
        for i in range(len(layers) - 1):
            next_layer = layers[i + 1]
            size = (next_layer.n_neurons, 1)

            self._weights.append(np.random.normal(loc=0.0, scale=1.0, size=size))

    def fit(
            self,
            X: np.ndarray[np.float64],
            Y: np.ndarray[np.float64],
            epoch: int,
    ) -> None:
        for _ in range(epoch):
            self._process_epoch(
                X=X,
                Y=Y,
            )

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
        a_history = self._get_a_history(x)

        deltas = self._get_deltas(a_history=a_history, y=y)

        self._update_weights(
            a_history=a_history,
            deltas=deltas,
        )

    def _get_a_history(
            self,
            x: np.ndarray[np.float64],
    ) -> list[np.ndarray[np.float64]]:
        a_history = [x]

        for l in range(len(self._layers)):
            a = self._forward_pass(
                W=self._weights[l],
                x=x,
                b=self._biases[l],
            )
            a_history.append(a)

        return a_history

    def _get_deltas(
            self,
            a_history: list[np.ndarray[np.float64]],
            y: np.ndarray[np.float64]
    ) -> list[np.ndarray[np.float64]]:
        deltas = [
            (a_history[-1] - y) * a_history[-1] * (1 - a_history[-1])
        ]
        for l in range(len(self._layers) - 2, -1, -1):
            d_sigma = a_history[l] * (1 - a_history[l])
            delta = (self._weights[l + 1].T @ deltas[-1]) * d_sigma
            deltas.append(delta)

        return deltas

    def _update_weights(
            self,
            a_history: list[np.ndarray[np.float64]],
            deltas: list[np.ndarray[np.float64]],
    ):
        for i in range(len(deltas)):
            delta = deltas[i]
            reverse_index = -1 * (i + 1)

            a = a_history[reverse_index - 1]
            W = self._weights[reverse_index]
            b = self._biases[reverse_index]

            d_w = delta @ a.T

            self._weights[reverse_index] = W - self._nu * d_w
            self._biases[reverse_index] = b - self._nu * delta

    def predict(
            self,
            x: np.ndarray[np.float64],
    ):
        pass

    def _forward_pass(
            self,
            W: np.ndarray[np.float64],
            x: np.ndarray[np.float64],
            b: np.ndarray[np.float64],
    ) -> np.ndarray[np.float64]:
        return self._sigmoid(
            (W @ x) + b
        )
