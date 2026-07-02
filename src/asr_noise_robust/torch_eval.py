"""Evaluate a saved log-mel CTC checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from asr_noise_robust.features import LogMelFeatureExtractor
from asr_noise_robust.text import Vocabulary
from asr_noise_robust.torch_experiment import decode_eval_rows, read_eval_audio_manifest
from asr_noise_robust.torch_models import TinyBiLstmCtc
from asr_noise_robust.train_eval import write_metrics, write_predictions


@dataclass(frozen=True)
class EvalOutputs:
    predictions_path: Path
    metrics_path: Path


def evaluate_logmel_checkpoint(
    checkpoint_path: str | Path,
    eval_manifest: str | Path,
    predictions_path: str | Path,
    metrics_path: str | Path,
) -> EvalOutputs:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    input_dim = int(checkpoint["input_dim"])
    vocab_size = int(checkpoint["vocab_size"])
    hidden_dim = int(checkpoint["hidden_dim"])
    sample_rate = int(checkpoint.get("sample_rate", 16000))
    feature_normalize = bool(checkpoint.get("feature_normalize", False))

    model = TinyBiLstmCtc(input_dim=input_dim, vocab_size=vocab_size, hidden_dim=hidden_dim)
    model.load_state_dict(checkpoint["model_state_dict"])

    vocab = Vocabulary.ctc_english()
    extractor = LogMelFeatureExtractor(
        sample_rate=sample_rate,
        n_mels=input_dim,
        normalize=feature_normalize,
    )
    eval_rows = read_eval_audio_manifest(eval_manifest)
    predictions = decode_eval_rows(
        model=model,
        eval_rows=eval_rows,
        extractor=extractor,
        vocab=vocab,
        sample_rate=sample_rate,
    )

    predictions_target = Path(predictions_path)
    metrics_target = Path(metrics_path)
    write_predictions(predictions_target, predictions)
    write_metrics(metrics_target, predictions, experiment="logmel_ctc_checkpoint_eval")
    return EvalOutputs(predictions_path=predictions_target, metrics_path=metrics_target)
