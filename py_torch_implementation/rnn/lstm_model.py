import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class Model(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            hidden_size: int,
    ):
        super().__init__()
        self._embedding_layer = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )
        self._lstm_layer = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.3,
            batch_first=True
        )
        self._linear_layer_1 = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size
        )
        self._relu_layer = nn.ReLU()
        self._linear_layer_2 = nn.Linear(
            in_features=hidden_size,
            out_features=1
        )

    def forward(
            self,
            x: torch.LongTensor,
            lengths: torch.Tensor
    ) -> torch.FloatTensor:
        input_ = self._embedding_layer(x)

        packed_embedded = pack_padded_sequence(
            input_,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden, cell) = self._lstm_layer(packed_embedded)
        x = hidden[-1, :, :]

        x = self._linear_layer_1(x)
        x = self._relu_layer(x)

        return self._linear_layer_2(x)
