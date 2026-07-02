"""Dependency-light train/evaluate loop for validating experiment wiring."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path

from asr_noise_robust.manifest import read_manifest
from asr_noise_robust.metrics import edit_counts


@dataclass(frozen=True)
class TrainEvalOutputs:
    model_path: Path
    predictions_path: Path
    metrics_path: Path


@dataclass(frozen=True)
class EvalRow:
    utt_id: str
    condition: str
    reference: str


@dataclass(frozen=True)
class PredictionRow:
    utt_id: str
    condition: str
    reference: str
    prediction: str
    feature: str
    model: str


def train_manifest_memorizer(train_manifest: str | Path) -> dict[str, object]:
    """Train a tiny recognizer that memorizes transcripts by utterance id."""
    rows = read_manifest(train_manifest)
    return {
        "model_type": "manifest_memorizer",
        "utterances": {row.utt_id: row.normalized_text for row in rows},
    }


def save_model(model: dict[str, object], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(model, indent=2, sort_keys=True), encoding="utf-8")


def load_model(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_eval_manifest(path: str | Path) -> list[EvalRow]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"utt_id", "normalized_text", "condition"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Eval manifest is missing required fields: {missing}")
        return [
            EvalRow(
                utt_id=row["utt_id"],
                condition=row["condition"],
                reference=row["normalized_text"],
            )
            for row in reader
        ]


def predict_with_manifest_memorizer(
    model: dict[str, object],
    eval_rows: list[EvalRow],
) -> list[PredictionRow]:
    utterances = model.get("utterances", {})
    if not isinstance(utterances, dict):
        raise ValueError("Model is missing the utterances mapping")
    model_type = str(model.get("model_type", "manifest_memorizer"))
    predictions: list[PredictionRow] = []
    for row in eval_rows:
        prediction = str(utterances.get(row.utt_id, ""))
        predictions.append(
            PredictionRow(
                utt_id=row.utt_id,
                condition=row.condition,
                reference=row.reference,
                prediction=prediction,
                feature="manifest_text",
                model=model_type,
            )
        )
    return predictions


def write_predictions(path: str | Path, rows: list[PredictionRow]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["utt_id", "test_condition", "reference", "prediction", "feature", "model"]
    with target.open("w", newline="", encoding="utf-8") as handle:
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


def _condition_order(rows: list[PredictionRow]) -> list[str]:
    ordered: list[str] = []
    for row in rows:
        if row.condition not in ordered:
            ordered.append(row.condition)
    return ordered


def _word_counts(rows: list[PredictionRow], condition: str) -> tuple[int, int, int, int]:
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


def _char_counts(rows: list[PredictionRow], condition: str) -> tuple[int, int]:
    edits = reference_length = 0
    for row in rows:
        if row.condition != condition:
            continue
        counts = edit_counts(list(row.reference), list(row.prediction))
        edits += counts.distance
        reference_length += counts.reference_length
    return edits, reference_length


def write_metrics(path: str | Path, rows: list[PredictionRow], experiment: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
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
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for condition in _condition_order(rows):
            condition_rows = [row for row in rows if row.condition == condition]
            substitutions, deletions, insertions, word_ref_len = _word_counts(rows, condition)
            char_edits, char_ref_len = _char_counts(rows, condition)
            word_edits = substitutions + deletions + insertions
            writer.writerow(
                {
                    "experiment": experiment,
                    "feature": condition_rows[0].feature,
                    "model": condition_rows[0].model,
                    "train_condition": "clean",
                    "test_condition": condition,
                    "wer": f"{word_edits / word_ref_len:.6f}",
                    "cer": f"{char_edits / char_ref_len:.6f}",
                    "substitutions": substitutions,
                    "deletions": deletions,
                    "insertions": insertions,
                    "notes": "dependency-light train/eval wiring check",
                }
            )


def run_manifest_train_eval(
    train_manifest: str | Path,
    eval_manifest: str | Path,
    model_path: str | Path,
    predictions_path: str | Path,
    metrics_path: str | Path,
) -> TrainEvalOutputs:
    model_target = Path(model_path)
    predictions_target = Path(predictions_path)
    metrics_target = Path(metrics_path)

    model = train_manifest_memorizer(train_manifest)
    save_model(model, model_target)
    eval_rows = read_eval_manifest(eval_manifest)
    predictions = predict_with_manifest_memorizer(model, eval_rows)
    write_predictions(predictions_target, predictions)
    write_metrics(metrics_target, predictions, experiment="manifest_train_eval")

    return TrainEvalOutputs(
        model_path=model_target,
        predictions_path=predictions_target,
        metrics_path=metrics_target,
    )

