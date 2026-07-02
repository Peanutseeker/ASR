"""Training utilities for PyTorch CTC baselines."""

from __future__ import annotations

import torch
from torch import nn


def ctc_loss_for_batch(
    model: nn.Module,
    features: torch.Tensor,
    feature_lengths: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int = 0,
) -> torch.Tensor:
    log_probs, output_lengths = model(features, feature_lengths)
    criterion = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    return criterion(log_probs, targets, output_lengths, target_lengths)


def train_one_batch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    feature_lengths: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int = 0,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = ctc_loss_for_batch(
        model=model,
        features=features,
        feature_lengths=feature_lengths,
        targets=targets,
        target_lengths=target_lengths,
        blank_id=blank_id,
    )
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())

