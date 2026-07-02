"""Evaluate a CTC head trained on cached frozen SSL features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from asr_noise_robust.ctc import greedy_ctc_decode
from asr_noise_robust.ssl_features import CachedFeatureRow, read_cached_feature_manifest
from asr_noise_robust.text import Vocabulary
from asr_noise_robust.torch_models import TinyBiLstmCtc
from asr_noise_robust.train_eval import PredictionRow, write_metrics, write_predictions


@dataclass(frozen=True)
class CachedSslEvalOutputs:
    predictions_path: Path
    metrics_path: Path


def _decode_cached_rows(
    model: TinyBiLstmCtc,
    rows: list[CachedFeatureRow],
    vocab: Vocabulary,
    model_name: str,
) -> list[PredictionRow]:
    model.eval()
    predictions: list[PredictionRow] = []
    with torch.no_grad():
        for row in rows:
            features = torch.load(row.feature_path, map_location="cpu", weights_only=False).float()
            batched = features.unsqueeze(0)
            lengths = torch.tensor([features.shape[0]], dtype=torch.long)
            log_probs, _ = model(batched, lengths)
            best_path = log_probs.argmax(dim=-1).squeeze(1).tolist()
            token_ids = greedy_ctc_decode(best_path, blank_id=vocab.blank_id)
            predictions.append(
                PredictionRow(
                    utt_id=row.utt_id,
                    condition=row.condition,
                    reference=row.normalized_text,
                    prediction=vocab.decode_tokens(token_ids),
                    feature="frozen_ssl",
                    model=model_name,
                )
            )
    return predictions


def evaluate_cached_ssl_checkpoint(
    checkpoint_path: str | Path,
    eval_feature_manifest: str | Path,
    predictions_path: str | Path,
    metrics_path: str | Path,
) -> CachedSslEvalOutputs:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    input_dim = int(checkpoint["input_dim"])
    vocab_size = int(checkpoint["vocab_size"])
    hidden_dim = int(checkpoint["hidden_dim"])
    feature_model_id = str(checkpoint["feature_model_id"])
    feature_layer = int(checkpoint["feature_layer"])

    model = TinyBiLstmCtc(input_dim=input_dim, vocab_size=vocab_size, hidden_dim=hidden_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model_name = f"{feature_model_id}:layer{feature_layer}:tiny_bilstm_ctc"

    vocab = Vocabulary.ctc_english()
    eval_rows = read_cached_feature_manifest(eval_feature_manifest)
    predictions = _decode_cached_rows(
        model=model,
        rows=eval_rows,
        vocab=vocab,
        model_name=model_name,
    )

    predictions_target = Path(predictions_path)
    metrics_target = Path(metrics_path)
    write_predictions(predictions_target, predictions)
    write_metrics(metrics_target, predictions, experiment="cached_ssl_ctc_eval")
    return CachedSslEvalOutputs(predictions_path=predictions_target, metrics_path=metrics_target)
