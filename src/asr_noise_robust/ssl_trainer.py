"""Train a lightweight CTC head on cached frozen SSL features."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from asr_noise_robust.ssl_features import CachedFeatureRow, read_cached_feature_manifest
from asr_noise_robust.text import Vocabulary
from asr_noise_robust.torch_models import TinyBiLstmCtc
from asr_noise_robust.torch_train import train_one_batch


@dataclass(frozen=True)
class CachedSslTrainingOutputs:
    checkpoint_path: Path
    history_path: Path
    summary_path: Path
    final_loss: float


def _batched(rows: list[CachedFeatureRow], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def _validate_feature_identity(rows: list[CachedFeatureRow]) -> tuple[int, str, int]:
    feature_dim = rows[0].feature_dim
    model_id = rows[0].model_id
    layer = rows[0].layer
    for row in rows:
        if row.feature_dim != feature_dim:
            raise ValueError("Cached feature manifest mixes feature dimensions")
        if row.model_id != model_id or row.layer != layer:
            raise ValueError("Cached feature manifest mixes SSL model identities")
    return feature_dim, model_id, layer


def _make_cached_training_batch(
    rows: list[CachedFeatureRow],
    vocab: Vocabulary,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features = []
    targets = []
    target_lengths = []
    for row in rows:
        item_features = torch.load(row.feature_path, map_location="cpu", weights_only=False).float()
        token_ids = torch.tensor(vocab.encode(row.normalized_text), dtype=torch.long)
        if item_features.shape[0] <= token_ids.numel():
            continue
        features.append(item_features)
        targets.append(token_ids)
        target_lengths.append(token_ids.numel())
    if not features:
        raise ValueError("Batch has no CTC-valid cached feature examples")
    padded_features = pad_sequence(features, batch_first=True)
    feature_lengths = torch.tensor([item.shape[0] for item in features], dtype=torch.long)
    return (
        padded_features,
        feature_lengths,
        torch.cat(targets),
        torch.tensor(target_lengths, dtype=torch.long),
    )


def run_cached_ssl_ctc_training(
    train_feature_manifest: str | Path,
    checkpoint_path: str | Path,
    history_path: str | Path,
    summary_path: str | Path,
    hidden_dim: int = 128,
    max_epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    log_every: int = 0,
) -> CachedSslTrainingOutputs:
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    torch.manual_seed(1337)
    rows = read_cached_feature_manifest(train_feature_manifest)
    if not rows:
        raise ValueError("training feature manifest has no rows")
    input_dim, feature_model_id, feature_layer = _validate_feature_identity(rows)

    vocab = Vocabulary.ctc_english()
    model = TinyBiLstmCtc(
        input_dim=input_dim,
        vocab_size=len(vocab.symbols),
        hidden_dim=hidden_dim,
        num_layers=1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history_target = Path(history_path)
    history_target.parent.mkdir(parents=True, exist_ok=True)
    final_loss = 0.0
    history_rows: list[dict[str, str]] = []
    global_step = 0
    for epoch in range(1, max_epochs + 1):
        for batch_index, batch_rows in enumerate(_batched(rows, batch_size), start=1):
            features, feature_lengths, targets, target_lengths = _make_cached_training_batch(
                batch_rows,
                vocab=vocab,
            )
            final_loss = train_one_batch(
                model=model,
                optimizer=optimizer,
                features=features,
                feature_lengths=feature_lengths,
                targets=targets,
                target_lengths=target_lengths,
                blank_id=vocab.blank_id,
            )
            global_step += 1
            history_rows.append(
                {
                    "epoch": str(epoch),
                    "batch": str(batch_index),
                    "global_step": str(global_step),
                    "num_examples": str(features.shape[0]),
                    "loss": f"{final_loss:.6f}",
                }
            )
            if log_every > 0 and (global_step == 1 or global_step % log_every == 0):
                print(
                    f"epoch={epoch} batch={batch_index} "
                    f"step={global_step} loss={final_loss:.6f}",
                    flush=True,
                )

    with history_target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "batch", "global_step", "num_examples", "loss"],
        )
        writer.writeheader()
        writer.writerows(history_rows)

    checkpoint_target = Path(checkpoint_path)
    checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "vocab_size": len(vocab.symbols),
            "hidden_dim": hidden_dim,
            "feature_model_id": feature_model_id,
            "feature_layer": feature_layer,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "final_loss": final_loss,
        },
        checkpoint_target,
    )

    summary_target = Path(summary_path)
    summary_target.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "cached_ssl_ctc_training",
        "num_train_examples": len(rows),
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "feature_dim": input_dim,
        "feature_model_id": feature_model_id,
        "feature_layer": feature_layer,
        "hidden_dim": hidden_dim,
        "final_loss": final_loss,
        "log_every": log_every,
        "checkpoint_path": str(checkpoint_target),
        "history_path": str(history_target),
    }
    summary_target.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return CachedSslTrainingOutputs(
        checkpoint_path=checkpoint_target,
        history_path=history_target,
        summary_path=summary_target,
        final_loss=final_loss,
    )
