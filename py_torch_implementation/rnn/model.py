import torch
from torch import nn


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
        self._rnn_layer = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True
        )
        self._linear_layer = nn.Linear(
            in_features=hidden_size,
            out_features=1
        )

    def forward(
            self,
            x: torch.LongTensor
    ) -> torch.FloatTensor:
        input_ = self._embedding_layer(x)

        rnn_out, hidden = self._rnn_layer(input_)
        last_time_step_out = rnn_out[:, -1, :]

        logits = self._linear_layer(last_time_step_out)

        return logits
