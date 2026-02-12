from dataclasses import dataclass

import numpy as np

from implemented_from_scratch.perceptron.neuron import Neuron


@dataclass
class Layer:
    n: int
    neurons: list[Neuron]

    def get_neuron_values(self) -> np.ndarray:
        return np.array([n.value for n in self.neurons])
