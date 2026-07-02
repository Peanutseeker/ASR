"""Dependency-light end-to-end pipeline used to prove the project wiring."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from asr_noise_robust.audio import deterministic_noise, sine_wave, write_pcm16_wav
from asr_noise_robust.manifest import Utterance, write_manifest
from asr_noise_robust.metrics import edit_counts
from asr_noise_robust.noise import mix_at_snr_db
from asr_noise_robust.text import normalize_transcript


SAMPLE_RATE = 16000
TOY_UTTERANCES = (
    ("toy_0001", "HELLO WORLD", 220.0),
    ("toy_0002", "NOISE ROBUST ASR", 330.0),
    ("toy_0003", "SELF SUPERVISED SPEECH", 440.0),
    ("toy_0004", "CLEAN TRAINING", 550.0),
)
SNR_CONDITIONS = {
    "clean": None,
    "snr_20": 20.0,
    "snr_10": 10.0,
    "snr_5": 5.0,
}


@dataclass(frozen=True)
class PipelineOutputs:
    train_manifest: Path
    eval_manifest: Path
    predictions_csv: Path
    metrics_csv: Path


@dataclass(frozen=True)
class PredictionRow:
    utt_id: str
    condition: str
    reference: str
    prediction: str
    feature: str = "toy_oracle"
    model: str = "oracle"


def _write_eval_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "utt_id",
        "audio_path",
        "duration",
        "text",
        "normalized_text",
        "split",
        "condition",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_predictions(path: Path, rows: list[PredictionRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["utt_id", "test_condition", "reference", "prediction", "feature", "model"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "utt_id": row.utt_id,
                    "test_condition": row.condition,
                    "reference": row.reference,
                    "prediction": row.prediction,
                    "feature": row.feature,
                    "model": row.model,
                }
            )


def _word_counts_for_condition(rows: list[PredictionRow], condition: str):
    substitutions = deletions = insertions = reference_length = 0
    for row in rows:
        if row.condition != condition:
            continue
        counts = edit_counts(row.reference.split(), row.prediction.split())
        substitutions += counts.substitutions
        deletions += counts.deletions
        insertions += counts.insertions
        reference_length += counts.reference_length
    return substitutions, deletions, insertions, reference_length


def _char_counts_for_condition(rows: list[PredictionRow], condition: str):
    edits = reference_length = 0
    for row in rows:
        if row.condition != condition:
            continue
        counts = edit_counts(list(row.reference), list(row.prediction))
        edits += counts.distance
        reference_length += counts.reference_length
    return edits, reference_length


def _write_metrics(path: Path, rows: list[PredictionRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "feature",
        "model",
        "train_condition",
        "test_condition",
        "wer",
        "cer",
        "substitutions",
        "deletions",
        "insertions",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for condition in SNR_CONDITIONS:
            substitutions, deletions, insertions, word_ref_len = _word_counts_for_condition(
                rows, condition
            )
            char_edits, char_ref_len = _char_counts_for_condition(rows, condition)
            word_edits = substitutions + deletions + insertions
            writer.writerow(
                {
                    "experiment": "toy_pipeline",
                    "feature": "toy_oracle",
                    "model": "oracle",
                    "train_condition": "clean",
                    "test_condition": condition,
                    "wer": f"{word_edits / word_ref_len:.6f}",
                    "cer": f"{char_edits / char_ref_len:.6f}",
                    "substitutions": substitutions,
                    "deletions": deletions,
                    "insertions": insertions,
                    "notes": "oracle sanity check for pipeline wiring",
                }
            )


def run_toy_pipeline(root: str | Path) -> PipelineOutputs:
    """Generate toy audio, manifests, predictions, and aggregate ASR metrics."""
    base = Path(root)
    clean_dir = base / "data" / "raw" / "toy"
    noisy_dir = base / "data" / "noisy" / "toy"
    manifest_dir = base / "data" / "manifests"
    prediction_dir = base / "results" / "predictions"
    metrics_path = base / "results" / "metrics.csv"

    train_rows: list[Utterance] = []
    eval_rows: list[dict[str, str]] = []
    predictions: list[PredictionRow] = []

    for index, (utt_id, text, frequency) in enumerate(TOY_UTTERANCES, start=1):
        samples = sine_wave(frequency_hz=frequency, seconds=0.25, sample_rate=SAMPLE_RATE)
        clean_path = clean_dir / f"{utt_id}.wav"
        write_pcm16_wav(clean_path, samples, sample_rate=SAMPLE_RATE)
        normalized = normalize_transcript(text)
        duration = len(samples) / SAMPLE_RATE

        train_rows.append(
            Utterance(
                utt_id=utt_id,
                audio_path=str(clean_path),
                duration=duration,
                text=text,
                normalized_text=normalized,
                split="train",
            )
        )

        for condition, snr_db in SNR_CONDITIONS.items():
            if snr_db is None:
                audio_path = clean_path
            else:
                noise = deterministic_noise(len(samples), seed=1000 + index)
                noisy = mix_at_snr_db(samples, noise, snr_db=snr_db)
                audio_path = noisy_dir / condition / f"{utt_id}.wav"
                write_pcm16_wav(audio_path, noisy, sample_rate=SAMPLE_RATE)

            eval_rows.append(
                {
                    "utt_id": utt_id,
                    "audio_path": str(audio_path),
                    "duration": f"{duration:.6f}",
                    "text": text,
                    "normalized_text": normalized,
                    "split": "test",
                    "condition": condition,
                }
            )
            predictions.append(
                PredictionRow(
                    utt_id=utt_id,
                    condition=condition,
                    reference=normalized,
                    prediction=normalized,
                )
            )

    train_manifest = manifest_dir / "toy_train.csv"
    eval_manifest = manifest_dir / "toy_eval_conditions.csv"
    predictions_csv = prediction_dir / "toy_oracle_predictions.csv"

    write_manifest(train_manifest, train_rows)
    _write_eval_manifest(eval_manifest, eval_rows)
    _write_predictions(predictions_csv, predictions)
    _write_metrics(metrics_path, predictions)

    return PipelineOutputs(
        train_manifest=train_manifest,
        eval_manifest=eval_manifest,
        predictions_csv=predictions_csv,
        metrics_csv=metrics_path,
    )

