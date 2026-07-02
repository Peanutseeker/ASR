#!/usr/bin/env python3
"""Create a CSV manifest from a LibriSpeech-style directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from asr_noise_robust.manifest import Utterance, write_manifest
from asr_noise_robust.text import normalize_transcript


def _duration_seconds(audio_path: Path) -> float:
    try:
        import torchaudio
    except ImportError:
        return 0.0
    if hasattr(torchaudio, "info"):
        info = torchaudio.info(str(audio_path))
        if info.sample_rate == 0:
            return 0.0
        return info.num_frames / info.sample_rate
    waveform, sample_rate = torchaudio.load(str(audio_path))
    if sample_rate == 0:
        return 0.0
    return waveform.shape[-1] / sample_rate


def build_manifest(root: Path, split: str) -> list[Utterance]:
    rows: list[Utterance] = []
    for transcript_path in sorted(root.rglob("*.trans.txt")):
        audio_dir = transcript_path.parent
        for line in transcript_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            utt_id, text = line.split(" ", 1)
            audio_path = audio_dir / f"{utt_id}.flac"
            if not audio_path.exists():
                raise FileNotFoundError(f"Missing audio for transcript row: {audio_path}")
            rows.append(
                Utterance(
                    utt_id=utt_id,
                    audio_path=str(audio_path),
                    duration=_duration_seconds(audio_path),
                    text=text,
                    normalized_text=normalize_transcript(text),
                    split=split,
                )
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = build_manifest(args.root, args.split)
    write_manifest(args.output, rows)
    total_hours = sum(row.duration for row in rows) / 3600.0
    print(f"wrote {len(rows)} rows to {args.output} ({total_hours:.3f} h)")


if __name__ == "__main__":
    main()
