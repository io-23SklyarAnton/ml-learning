import torch
from torch import nn, Tensor


class Model(nn.Module):
    def __init__(
            self,
            n_input: int,
            n_hidden: int,
            n_output: int,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_input, n_hidden, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output, dtype=torch.float64),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
