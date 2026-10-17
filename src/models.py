"""Four recurrent architectures sharing a next-token prediction interface."""

from __future__ import annotations

from typing import Type

import torch
from torch import nn

from src.utils import count_parameters


class BaseLanguageModel(nn.Module):
    architecture_name: str = "base"

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        pad_index: int,
        dropout: float = 0.2,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.pad_index = pad_index
        self.dropout_p = dropout
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.embed_dropout = nn.Dropout(dropout)

    def encode_inputs(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_dropout(self.embedding(token_ids))

    def sequence_lengths(self, token_ids: torch.Tensor) -> torch.Tensor:
        not_pad = token_ids != self.pad_index
        lengths = not_pad.sum(dim=1)
        return lengths.clamp(min=1)

    def gather_last(self, sequence_output: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        lengths = self.sequence_lengths(token_ids)
        last_index = lengths - 1
        batch_index = torch.arange(sequence_output.size(0), device=sequence_output.device)
        return sequence_output[batch_index, last_index]

    def parameter_count(self) -> int:
        return count_parameters(self)

    def hyperparameters(self) -> dict:
        return {
            "architecture": self.architecture_name,
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout_p,
            "num_layers": self.num_layers,
            "pad_index": self.pad_index,
            "parameter_count": self.parameter_count(),
        }


class SimpleRNNModel(BaseLanguageModel):
    architecture_name = "simple_rnn"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rnn = nn.RNN(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            nonlinearity="tanh",
            dropout=self.dropout_p if self.num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.output_dropout = nn.Dropout(self.dropout_p)
        self.projection = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.encode_inputs(token_ids)
        output, _hidden = self.rnn(embedded)
        last = self.gather_last(output, token_ids)
        return self.projection(self.output_dropout(last))


class BidirectionalRNNModel(BaseLanguageModel):
    """Bidirectional RNN over the observed input window only.

    Both directions may read the already observed tokens, but the target token
    is never provided as input. This is not a strictly causal language model.
    """

    architecture_name = "bidirectional_rnn"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rnn = nn.RNN(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            nonlinearity="tanh",
            dropout=self.dropout_p if self.num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.output_dropout = nn.Dropout(self.dropout_p)
        self.projection = nn.Linear(self.hidden_dim * 2, self.vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.encode_inputs(token_ids)
        _output, hidden = self.rnn(embedded)
        # hidden: (num_layers * 2, batch, hidden_dim)
        # Use final forward state (end of window) and final backward state (start of window).
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        combined = torch.cat([forward_hidden, backward_hidden], dim=-1)
        return self.projection(self.output_dropout(combined))


class LSTMModel(BaseLanguageModel):
    architecture_name = "lstm"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rnn = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_p if self.num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.output_dropout = nn.Dropout(self.dropout_p)
        self.projection = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.encode_inputs(token_ids)
        _output, (hidden, _cell) = self.rnn(embedded)
        last = hidden[-1]
        return self.projection(self.output_dropout(last))


class GRUModel(BaseLanguageModel):
    architecture_name = "gru"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rnn = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_p if self.num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.output_dropout = nn.Dropout(self.dropout_p)
        self.projection = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.encode_inputs(token_ids)
        _output, hidden = self.rnn(embedded)
        last = hidden[-1]
        return self.projection(self.output_dropout(last))


MODEL_REGISTRY: dict[str, Type[BaseLanguageModel]] = {
    "simple_rnn": SimpleRNNModel,
    "bidirectional_rnn": BidirectionalRNNModel,
    "lstm": LSTMModel,
    "gru": GRUModel,
}

MODEL_DISPLAY_NAMES = {
    "simple_rnn": "Simple RNN",
    "bidirectional_rnn": "Bidirectional RNN",
    "lstm": "LSTM",
    "gru": "GRU",
}


def build_model(
    architecture: str,
    vocab_size: int,
    embedding_dim: int,
    hidden_dim: int,
    pad_index: int,
    dropout: float = 0.2,
    num_layers: int = 1,
) -> BaseLanguageModel:
    if architecture not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {architecture}")
    model_cls = MODEL_REGISTRY[architecture]
    return model_cls(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        pad_index=pad_index,
        dropout=dropout,
        num_layers=num_layers,
    )
