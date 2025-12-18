from dataclasses import dataclass
from typing import Optional


@dataclass
class Neuron:
    value: Optional[float]
    y: int
    b: float
