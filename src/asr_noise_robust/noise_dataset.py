"""Materialize clean/noisy evaluation manifests and audio files."""

from __future__ import annotations

import csv
from pathlib import Path

from asr_noise_robust.audio import deterministic_noise, write_pcm16_wav
from asr_noise_robust.manifest import read_manifest
from asr_noise_robust.noise import mix_at_snr_db
from asr_noise_robust.torch_smoke import _load_mono_audio


def materialize_noisy_eval_set(
    source_manifest: str | Path,
    output_manifest: str | Path,
    output_audio_dir: str | Path,
    snr_values: tuple[int, ...] = (20, 10, 5),
    sample_rate: int = 16000,
) -> int:
    rows = read_manifest(source_manifest)
    manifest_target = Path(output_manifest)
    audio_dir = Path(output_audio_dir)
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "utt_id",
        "audio_path",
        "duration",
        "text",
        "normalized_text",
        "split",
        "condition",
    ]
    count = 0
    with manifest_target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "utt_id": row.utt_id,
                    "audio_path": row.audio_path,
                    "duration": f"{row.duration:.6f}",
                    "text": row.text,
                    "normalized_text": row.normalized_text,
                    "split": row.split,
                    "condition": "clean",
                }
            )
            count += 1

            waveform = _load_mono_audio(row.audio_path, target_sample_rate=sample_rate)
            samples = [float(value) for value in waveform.tolist()]
            for snr_db in snr_values:
                condition = f"snr_{snr_db}"
                noise = deterministic_noise(len(samples), seed=10_000 + row_index + snr_db)
                noisy = mix_at_snr_db(samples, noise, snr_db=float(snr_db))
                noisy_path = audio_dir / condition / f"{row.utt_id}.wav"
                write_pcm16_wav(noisy_path, noisy, sample_rate=sample_rate)
                writer.writerow(
                    {
                        "utt_id": row.utt_id,
                        "audio_path": str(noisy_path),
                        "duration": f"{row.duration:.6f}",
                        "text": row.text,
                        "normalized_text": row.normalized_text,
                        "split": row.split,
                        "condition": condition,
                    }
                )
                count += 1
    return count

