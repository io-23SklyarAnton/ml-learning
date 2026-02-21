import torch
from torch import nn


class Model(nn.Module):
    def __init__(
            self,
            n_features: int,
            n_hidden: int,
            embedding_dim: int,
    ):
        super().__init__()
        self._embedding_layer = nn.Embedding(3, embedding_dim)
        layer_1 = nn.Linear(n_features + embedding_dim, n_hidden)
        relu_1 = nn.ReLU()
        layer_2 = nn.Linear(n_hidden, n_hidden)
        relu_2 = nn.ReLU()
        layer_3 = nn.Linear(n_hidden, n_hidden)
        relu_3 = nn.ReLU()
        layer_4 = nn.Linear(n_hidden, 1)

        layers = [layer_1, relu_1, layer_2, relu_2, layer_3, relu_3, layer_4]
        self._layers = nn.ModuleList(layers)

    def forward(
            self,
            x: torch.FloatTensor,
            origin: torch.LongTensor,
    ):
        embedding_vectors = self._embedding_layer(origin)
        x = torch.cat((x, embedding_vectors), dim=1)

        for layer in self._layers:
            x = layer(x)

        return x
