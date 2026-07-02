"""Small PyTorch acoustic models for CTC experiments."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TinyBiLstmCtc(nn.Module):
    """A compact BiLSTM CTC head suitable for CPU smoke tests and baselines."""

    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if vocab_size <= 1:
            raise ValueError("vocab_size must be greater than one")

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.output = nn.Linear(hidden_dim * 2, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, time, feature]")
        packed = pack_padded_sequence(
            features,
            feature_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        encoded, _ = self.encoder(packed)
        padded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=features.shape[1])
        logits = self.output(padded)
        log_probs = self.log_softmax(logits).transpose(0, 1).contiguous()
        return log_probs, feature_lengths

