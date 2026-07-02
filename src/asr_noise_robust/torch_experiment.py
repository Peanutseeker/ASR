"""Train/decode/evaluate a small log-mel CTC experiment."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch

from asr_noise_robust.ctc import greedy_ctc_decode
from asr_noise_robust.features import LogMelFeatureExtractor
from asr_noise_robust.text import Vocabulary
from asr_noise_robust.torch_models import TinyBiLstmCtc
from asr_noise_robust.torch_smoke import _load_mono_audio, _make_batch
from asr_noise_robust.torch_train import train_one_batch
from asr_noise_robust.train_eval import PredictionRow, write_metrics, write_predictions


@dataclass(frozen=True)
class TorchExperimentOutputs:
    checkpoint_path: Path
    predictions_path: Path
    metrics_path: Path
    final_loss: float


@dataclass(frozen=True)
class EvalAudioRow:
    utt_id: str
    audio_path: str
    condition: str
    reference: str


def read_eval_audio_manifest(path: str | Path) -> list[EvalAudioRow]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"utt_id", "audio_path", "condition", "normalized_text"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Eval manifest is missing required fields: {missing}")
        return [
            EvalAudioRow(
                utt_id=row["utt_id"],
                audio_path=row["audio_path"],
                condition=row["condition"],
                reference=row["normalized_text"],
            )
            for row in reader
        ]


def decode_eval_rows(
    model: TinyBiLstmCtc,
    eval_rows: list[EvalAudioRow],
    extractor: LogMelFeatureExtractor,
    vocab: Vocabulary,
    sample_rate: int,
) -> list[PredictionRow]:
    model.eval()
    predictions: list[PredictionRow] = []
    with torch.no_grad():
        for row in eval_rows:
            waveform = _load_mono_audio(row.audio_path, target_sample_rate=sample_rate)
            features = extractor(waveform).unsqueeze(0)
            lengths = torch.tensor([features.shape[1]], dtype=torch.long)
            log_probs, _ = model(features, lengths)
            best_path = log_probs.argmax(dim=-1).squeeze(1).tolist()
            token_ids = greedy_ctc_decode(best_path, blank_id=vocab.blank_id)
            prediction = vocab.decode_tokens(token_ids)
            predictions.append(
                PredictionRow(
                    utt_id=row.utt_id,
                    condition=row.condition,
                    reference=row.reference,
                    prediction=prediction,
                    feature="logmel",
                    model="tiny_bilstm_ctc",
                )
            )
    return predictions


def run_torch_logmel_experiment(
    train_manifest: str | Path,
    eval_manifest: str | Path,
    checkpoint_path: str | Path,
    predictions_path: str | Path,
    metrics_path: str | Path,
    n_mels: int = 16,
    hidden_dim: int = 8,
    max_epochs: int = 5,
    learning_rate: float = 0.01,
    sample_rate: int = 16000,
) -> TorchExperimentOutputs:
    torch.manual_seed(1337)
    features, feature_lengths, targets, target_lengths, vocab_size = _make_batch(
        train_manifest=train_manifest,
        n_mels=n_mels,
        sample_rate=sample_rate,
    )
    model = TinyBiLstmCtc(
        input_dim=n_mels,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    final_loss = 0.0
    for _ in range(max_epochs):
        final_loss = train_one_batch(
            model=model,
            optimizer=optimizer,
            features=features,
            feature_lengths=feature_lengths,
            targets=targets,
            target_lengths=target_lengths,
            blank_id=0,
        )

    checkpoint_target = Path(checkpoint_path)
    checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": n_mels,
            "vocab_size": vocab_size,
            "hidden_dim": hidden_dim,
            "sample_rate": sample_rate,
            "feature_normalize": True,
            "max_epochs": max_epochs,
            "final_loss": final_loss,
        },
        checkpoint_target,
    )

    vocab = Vocabulary.ctc_english()
    extractor = LogMelFeatureExtractor(sample_rate=sample_rate, n_mels=n_mels)
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
    write_metrics(metrics_target, predictions, experiment="torch_logmel_experiment")

    return TorchExperimentOutputs(
        checkpoint_path=checkpoint_target,
        predictions_path=predictions_target,
        metrics_path=metrics_target,
        final_loss=final_loss,
    )
