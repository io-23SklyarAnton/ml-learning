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

    def _sigmoid(
            self,
            input_: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        return 1 / (1 + np.exp(-1 * input_))

    def __init_weights(
            self,
            layers: list[Layer],
    ) -> None:
        pass

    def __init_biases(
            self,
            layers: list[Layer],
    ) -> None:
        pass

    def fit(
            self,
            X: np.ndarray[np.float64],
            Y: np.ndarray[np.float64],
    ) -> np.ndarray[np.float64]:
        for i_sample in range(len(Y)):
            x = X[i_sample]
            y = Y[i_sample]
            a_s = [x]

            for l in range(len(self._layers)):
                a = self._forward_pass(
                    W=self._weights[l],
                    x=x,
                    b=self._biases[l],
                )
                a_s.append(a)

            deltas = [
                (a_s[-1] - y) * a_s[-1] * (1 - a_s[-1])
            ]
            for l in range(len(self._layers) - 2, -1, -1):
                d_sigma = a_s[l] * (1 - a_s[l])
                delta = (self._weights[l + 1].T @ deltas[-1]) * d_sigma
                deltas.append(delta)

            for i in range(len(deltas)):
                delta = deltas[i]
                reverse_index = -1 * (i + 1)

                a = a_s[reverse_index - 1]
                W = self._weights[reverse_index]
                b = self._biases[reverse_index]

                d_w = delta @ a.T

                self._weights[reverse_index] = W - self._nu * d_w
                self._biases[reverse_index] = b - self._nu * delta

    def _forward_pass(
            self,
            W: np.ndarray[np.float64],
            x: np.ndarray[np.float64],
            b: np.ndarray[np.float64],
    ) -> np.ndarray[np.float64]:
        return self._sigmoid(
            (W @ x) + b
        )
