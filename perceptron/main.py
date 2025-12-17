from dataclasses import dataclass
from typing import Optional

from pandas import read_csv
import numpy as np

data = read_csv("iris.csv")


class Perceptron:
    def __init__(
            self,
            layer_sizes: list[int],
    ):
        self._set_neurons(layer_sizes)
        self._set_weights()

    def _set_neurons(
            self,
            layer_sizes: list[int]
    ) -> None:
        layers = []
        for i, size in enumerate(layer_sizes):
            layer = Layer(n=i, neurons=[])
            for y in range(size):
                layer.neurons.append(Neuron(
                    value=None,
                    y=y,
                    b=np.random.randint(0, size)
                ))

            layers.append(layer)

        self._layers = layers

    def _set_weights(self):
        self._weights = []
        for layer_1, layer_2 in np.column_stack((self._layers[:-1], self._layers[1:])):
            self._weights.append([])
            for neuron_1, neuron_2 in np.meshgrid(layer_1.neurons, layer_2.neurons, indexing='ij'):
                self._weights[layer_1.n].append(Weight(
                    value=np.random.normal(loc=0.0, scale=1.0),
                    l1=layer_1.n,
                    l2=layer_2.n,
                    y1=neuron_1.y,
                    y2=neuron_2.y,
                ))

    def predict(
            self,
            input_values: list[float],
    ) -> list["Neuron"]:
        if len(self._layers[0].neurons) != len(input_values):
            raise ValueError("input must be the same size as 1st layer")

        for input_value, neuron in zip(input_values, self._layers[0].neurons):
            neuron.value = input_value

        for layer in self._layers[1:]:
            for neurons, weights in zip(layer.neurons, self._weights):
                pass

    @staticmethod
    def _calculate_result(
            weight_values: np.ndarray,
            neuron_values: np.ndarray,
            bias: int,
    ) -> int:
        result_matrix = neuron_values @ weight_values.T
        result = np.sum(result_matrix)

        return 1 if result >= bias else -1


@dataclass
class Layer:
    n: int
    neurons: list["Neuron"]


@dataclass
class Neuron:
    value: Optional[float]
    y: int
    b: int


@dataclass
class Weight:
    l1: Layer
    y1: int
    l2: Layer
    y2: int
    value: float


for row in data.values:
    print(row)
