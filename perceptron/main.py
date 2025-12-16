from dataclasses import dataclass

from pandas import read_csv
import numpy as np

data = read_csv("iris.csv")


@dataclass
class Neuron:
    value: float
    x: int
    y: int
    b: int


@dataclass
class Weight:
    x1: int
    y1: int
    x2: int
    y2: int


def calculate_result(
        weight_values: np.ndarray,
        neuron_values: np.ndarray,
        bias: int,
) -> int:
    result_matrix = neuron_values @ weight_values.T
    result = np.sum(result_matrix)

    return 1 if result >= bias else -1


for row in data.values:
    print(row)
