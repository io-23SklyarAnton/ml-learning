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
        self.layer_1 = nn.Linear(n_input, n_hidden, dtype=torch.float64)
        self.layer_2 = nn.Linear(n_hidden, n_hidden, dtype=torch.float64)
        self.layer_3 = nn.Linear(n_hidden, n_hidden, dtype=torch.float64)
        self.layer_4 = nn.Linear(n_hidden, n_output, dtype=torch.float64)

    def forward(
            self,
            x: Tensor
    ) -> Tensor:
        x = self.layer_1(x)
        x = nn.ReLU()(x)
        x = self.layer_2(x)
        x = nn.ReLU()(x)
        x = self.layer_4(x)

        x = nn.Sigmoid()(x)
        return x
