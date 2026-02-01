import dataclasses

import numpy as np

from neural_network.activation_functions import ActivationFunction
from optimizers import Optimizer, OptimizerFactory


@dataclasses.dataclass
class LayerSize:
    n_neurons: int


class NeuralNetwork:
    def __init__(
            self,
            activation_function: ActivationFunction,
            layer_sizes: list[LayerSize],
            learning_rate: float,
            optimizer_factory: OptimizerFactory,
    ):
        self._layer_sizes = layer_sizes
        self._learning_rate = learning_rate
        self._activation_function = activation_function

        self._init_weights(layer_sizes)
        self._init_biases(layer_sizes)
        self._weight_optimizers: list[Optimizer] = [
            optimizer_factory.create_optimizer()
            for _ in range(len(self._weights))
        ]
        self._bias_optimizers: list[Optimizer] = [
            optimizer_factory.create_optimizer()
            for _ in range(len(self._weights))
        ]

    def _sigmoid(
            self,
            z: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        return 1 / (1 + np.exp(-1 * z))

    def _sigmoid_derivative(
            self,
            z: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        return z * (1 - z)

    def _relu(
            self,
            z: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        return np.maximum(0, z)

    def _relu_derivative(
            self,
            z: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        return (z > 0).astype(float)

    def _get_derivative(
            self,
            z: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        match self._activation_function:
            case ActivationFunction.SIGMOID:
                return self._sigmoid_derivative(z)
            case ActivationFunction.RELU:
                return self._relu_derivative(z)
            case _:
                raise NotImplemented()

    def _init_weights(
            self,
            layer_sizes: list[LayerSize],
    ) -> None:
        self._weights = []
        for i in range(len(layer_sizes) - 1):
            current_layer_size = layer_sizes[i]
            next_layer = layer_sizes[i + 1]
            size = (next_layer.n_neurons, current_layer_size.n_neurons)

            scale: float = self._calculate_scale(current_layer_size.n_neurons)

            self._weights.append(np.random.normal(loc=0.0, scale=scale, size=size))

    def _calculate_scale(
            self,
            layer_size: int
    ) -> float:
        match self._activation_function:
            case ActivationFunction.SIGMOID:
                return 0.1
            case ActivationFunction.RELU:
                return np.sqrt(2 / layer_size)
            case _:
                raise NotImplemented()

    def _init_biases(
            self,
            layer_sizes: list[LayerSize],
    ) -> None:
        self._biases = []
        for i in range(len(layer_sizes) - 1):
            next_layer = layer_sizes[i + 1]
            size = (next_layer.n_neurons, 1)

            scale: float = self._calculate_scale(next_layer.n_neurons)

            self._biases.append(np.random.normal(loc=0.0, scale=scale, size=size))

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
        x_reshaped = x.reshape(-1, 1)
        return self._get_activations(x_reshaped)[-1]

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
        x = X[i_sample].reshape(-1, 1)
        y = Y[i_sample].reshape(-1, 1)
        activations = self._get_activations(x)

        deltas = self._get_deltas(
            activations=activations,
            y=y
        )

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
            is_output_layer = layer_idx == len(self._weights) - 1

            a = self._forward_pass(
                W=self._weights[layer_idx],
                x=activations[-1],
                b=self._biases[layer_idx],
                is_output_layer=is_output_layer
            )
            activations.append(a)

        return activations

    def _get_deltas(
            self,
            activations: list[np.ndarray[np.float64]],
            y: np.ndarray[np.float64]
    ) -> list[np.ndarray[np.float64]]:
        deltas = [
            (activations[-1] - y) * self._sigmoid_derivative(activations[-1])
        ]
        for layer_idx in range(len(self._weights) - 1, 0, -1):
            derivative = self._get_derivative(activations[layer_idx])

            delta = (self._weights[layer_idx].T @ deltas[-1]) * derivative
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
            w_optimizer = self._weight_optimizers[layer_idx]
            b_optimizer = self._bias_optimizers[layer_idx]

            d_w = delta @ a.T

            self._weights[layer_idx] = W - self._learning_rate * w_optimizer.compute_step(d_w)
            self._biases[layer_idx] = b - self._learning_rate * b_optimizer.compute_step(delta)

    def _forward_pass(
            self,
            W: np.ndarray[np.float64],
            x: np.ndarray[np.float64],
            b: np.ndarray[np.float64],
            is_output_layer: bool,
    ) -> np.ndarray[np.float64]:
        z = (W @ x) + b

        if is_output_layer:
            return self._sigmoid(z)

        match self._activation_function:
            case ActivationFunction.RELU:
                return self._relu(z)
            case ActivationFunction.SIGMOID:
                return self._sigmoid(z)
            case _:
                raise NotImplemented()
