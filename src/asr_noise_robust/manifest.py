"""CSV manifest utilities for ASR data splits and experiment metadata."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


MANIFEST_FIELDS = ("utt_id", "audio_path", "duration", "text", "normalized_text", "split")


@dataclass(frozen=True)
class Utterance:
    utt_id: str
    audio_path: str
    duration: float
    text: str
    normalized_text: str
    split: str


@dataclass(frozen=True)
class DurationSummary:
    num_utterances: int
    total_seconds: float

    @property
    def total_hours(self) -> float:
        return self.total_seconds / 3600.0


def write_manifest(path: str | Path, rows: list[Utterance] | tuple[Utterance, ...]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "utt_id": row.utt_id,
                    "audio_path": row.audio_path,
                    "duration": f"{row.duration:.6f}",
                    "text": row.text,
                    "normalized_text": row.normalized_text,
                    "split": row.split,
                }
            )


def read_manifest(path: str | Path) -> list[Utterance]:
    source = Path(path)
    with source.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in MANIFEST_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Manifest is missing required fields: {missing}")
        return [
            Utterance(
                utt_id=row["utt_id"],
                audio_path=row["audio_path"],
                duration=float(row["duration"]),
                text=row["text"],
                normalized_text=row["normalized_text"],
                split=row["split"],
            )
            for row in reader
        ]


def summarize_durations(rows: list[Utterance] | tuple[Utterance, ...]) -> DurationSummary:
    return DurationSummary(
        num_utterances=len(rows),
        total_seconds=sum(row.duration for row in rows),
    )

