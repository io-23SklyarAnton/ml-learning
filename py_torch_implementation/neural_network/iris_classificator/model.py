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
        layer_1 = nn.Linear(n_input, n_hidden, dtype=torch.float64)
        activation_1 = nn.ReLU()
        layer_2 = nn.Linear(n_hidden, n_hidden, dtype=torch.float64)
        activation_2 = nn.ReLU()
        layer_3 = nn.Linear(n_hidden, n_output, dtype=torch.float64)
        activation_3 = nn.Softmax()
        layers = [layer_1, activation_1, layer_2, activation_2, layer_3, activation_3]

        self.layers = nn.ModuleList(layers)

    def forward(
            self,
            x: Tensor
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x)

        return x
