"""Evaluate pretrained HuggingFace CTC ASR models on condition manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from asr_noise_robust.text import normalize_transcript
from asr_noise_robust.torch_experiment import EvalAudioRow, read_eval_audio_manifest
from asr_noise_robust.torch_smoke import _load_mono_audio
from asr_noise_robust.train_eval import PredictionRow, write_metrics, write_predictions


class SpeechTranscriber(Protocol):
    def __call__(self, waveform: torch.Tensor) -> str:
        """Return a raw transcript for a mono waveform tensor."""


@dataclass(frozen=True)
class PretrainedAsrOutputs:
    predictions_path: Path
    metrics_path: Path


class TransformersCtcTranscriber:
    """Thin runtime wrapper around AutoProcessor + AutoModelForCTC."""

    def __init__(
        self,
        model_id: str,
        sample_rate: int = 16000,
        device: str = "cpu",
    ) -> None:
        from transformers import AutoModelForCTC, AutoProcessor

        self.model_id = model_id
        self.sample_rate = sample_rate
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCTC.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def __call__(self, waveform: torch.Tensor) -> str:
        audio = waveform.detach().cpu().numpy()
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(predicted_ids)[0]


def transcribe_eval_rows(
    eval_rows: list[EvalAudioRow],
    transcriber: SpeechTranscriber,
    sample_rate: int,
    model_id: str,
) -> list[PredictionRow]:
    predictions: list[PredictionRow] = []
    for row in eval_rows:
        waveform = _load_mono_audio(row.audio_path, target_sample_rate=sample_rate)
        raw_prediction = transcriber(waveform)
        predictions.append(
            PredictionRow(
                utt_id=row.utt_id,
                condition=row.condition,
                reference=row.reference,
                prediction=normalize_transcript(raw_prediction),
                feature="pretrained_ctc",
                model=model_id,
            )
        )
    return predictions


def evaluate_pretrained_asr(
    model_id: str,
    eval_manifest: str | Path,
    predictions_path: str | Path,
    metrics_path: str | Path,
    sample_rate: int = 16000,
    device: str = "cpu",
    max_eval_examples: int | None = None,
    transcriber: SpeechTranscriber | None = None,
) -> PretrainedAsrOutputs:
    eval_rows = read_eval_audio_manifest(eval_manifest)
    if max_eval_examples is not None:
        eval_rows = eval_rows[:max_eval_examples]
    if transcriber is None:
        transcriber = TransformersCtcTranscriber(
            model_id=model_id,
            sample_rate=sample_rate,
            device=device,
        )

    predictions = transcribe_eval_rows(
        eval_rows=eval_rows,
        transcriber=transcriber,
        sample_rate=sample_rate,
        model_id=model_id,
    )

    predictions_target = Path(predictions_path)
    metrics_target = Path(metrics_path)
    write_predictions(predictions_target, predictions)
    write_metrics(metrics_target, predictions, experiment="pretrained_asr_eval")
    return PretrainedAsrOutputs(predictions_path=predictions_target, metrics_path=metrics_target)
