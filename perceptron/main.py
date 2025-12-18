from dataclasses import dataclass
from typing import Optional

from pandas import read_csv
import numpy as np

data = read_csv("iris.csv")


class Perceptron:
    def __init__(
            self,
            layer_sizes: list[int],
            learning_rate: float = 0.1
    ):
        self._learning_rate = learning_rate
        self._set_neurons(layer_sizes)
        self._set_weights(layer_sizes[0], layer_sizes[1])

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
                    b=np.random.normal(loc=0.0, scale=1.0)
                ))

            layers.append(layer)

        self._layers = layers

    def _set_weights(
            self,
            n_inputs: int,
            n_outputs: int,
    ):
        self._weights = np.random.normal(loc=0.0, scale=1.0, size=(n_outputs, n_inputs))

    def predict(
            self,
            input_values: np.ndarray,
    ) -> np.ndarray:
        if len(self._layers[0].neurons) != len(input_values):
            raise ValueError("input must be the same size as 1st layer")

        for neuron, val in zip(self._layers[0].neurons, input_values):
            neuron.value = val

        result_values = self._weights @ input_values

        for neuron, result_value in zip(self._layers[-1].neurons, result_values):
            neuron.value = 1 if result_value >= neuron.b else -1

        return self._layers[-1].get_neuron_values()

    def fit_model(
            self,
            epochs: int,
            features: np.ndarray,
            correct_answers: np.ndarray,
    ) -> None:
        for _ in range(epochs):
            for feature, correct_answer in zip(features, correct_answers):
                prediction = self.predict(feature)
                difference = prediction - correct_answer
                delta_w = np.outer(
                    self._learning_rate * difference,
                    self._layers[0].get_neuron_values()
                )
                self._weights = self._weights - delta_w

                for neuron, y_diff in zip(self._layers[-1].neurons, difference):
                    neuron.b = neuron.b - self._learning_rate * y_diff

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

    def get_neuron_values(self) -> np.ndarray:
        return np.array([n.value for n in self.neurons])


@dataclass
class Neuron:
    value: Optional[float]
    y: int
    b: float


@dataclass
class Weight:
    l1: Layer
    y1: int
    l2: Layer
    y2: int
    value: float


perceptron = Perceptron(layer_sizes=[4, 3])
data = np.array(data.values)
np.random.shuffle(data)

training_data = data[:-5]
unseen_data = data[-5:]

_, indices = np.unique(training_data[:, 4], return_inverse=True)
correct_answers = []

for i in indices:
    answer = [-1, -1, -1]
    answer[i] = 1
    correct_answers.append(answer)

perceptron.fit_model(
    epochs=100,
    features=training_data[:, :-1],
    correct_answers=np.array(correct_answers),
)

for case in unseen_data:
    print(
        perceptron.predict(case[:-1]),
        case[-1]
    )