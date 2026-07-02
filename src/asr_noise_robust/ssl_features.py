"""Cache frozen SSL hidden states for lightweight downstream CTC training."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Protocol

import torch

from asr_noise_robust.torch_smoke import _load_mono_audio


class SslFeatureExtractor(Protocol):
    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return time-major SSL features with shape [frames, dim]."""


@dataclass(frozen=True)
class AudioManifestRow:
    utt_id: str
    audio_path: str
    condition: str
    normalized_text: str


@dataclass(frozen=True)
class CachedFeatureRow:
    utt_id: str
    condition: str
    normalized_text: str
    feature_path: Path
    num_frames: int
    feature_dim: int
    model_id: str
    layer: int


class TransformersSslFeatureExtractor:
    """Frozen HuggingFace SSL encoder wrapper."""

    def __init__(
        self,
        model_id: str,
        layer: int = -1,
        sample_rate: int = 16000,
        device: str = "cpu",
    ) -> None:
        from transformers import AutoFeatureExtractor, AutoModel

        self.model_id = model_id
        self.layer = layer
        self.sample_rate = sample_rate
        self.device = torch.device(device)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        audio = waveform.detach().cpu().numpy()
        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        features = outputs.hidden_states[self.layer].squeeze(0).detach().cpu()
        if features.ndim != 2:
            raise ValueError(f"Expected SSL features [frames, dim], got {tuple(features.shape)}")
        return features


def read_audio_manifest(path: str | Path) -> list[AudioManifestRow]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"utt_id", "audio_path", "normalized_text"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Audio manifest is missing required fields: {missing}")
        return [
            AudioManifestRow(
                utt_id=row["utt_id"],
                audio_path=row["audio_path"],
                condition=row.get("condition") or "clean",
                normalized_text=row["normalized_text"],
            )
            for row in reader
        ]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def cache_ssl_features(
    source_manifest: str | Path,
    output_manifest: str | Path,
    output_dir: str | Path,
    model_id: str,
    layer: int = -1,
    sample_rate: int = 16000,
    device: str = "cpu",
    max_examples: int | None = None,
    extractor: SslFeatureExtractor | None = None,
) -> list[CachedFeatureRow]:
    rows = read_audio_manifest(source_manifest)
    if max_examples is not None:
        rows = rows[:max_examples]
    if extractor is None:
        extractor = TransformersSslFeatureExtractor(
            model_id=model_id,
            layer=layer,
            sample_rate=sample_rate,
            device=device,
        )

    feature_root = Path(output_dir)
    feature_root.mkdir(parents=True, exist_ok=True)
    cached_rows: list[CachedFeatureRow] = []
    for row in rows:
        waveform = _load_mono_audio(row.audio_path, target_sample_rate=sample_rate)
        features = extractor.extract(waveform).contiguous()
        feature_path = feature_root / f"{_safe_name(row.utt_id)}_{_safe_name(row.condition)}.pt"
        torch.save(features, feature_path)
        cached_rows.append(
            CachedFeatureRow(
                utt_id=row.utt_id,
                condition=row.condition,
                normalized_text=row.normalized_text,
                feature_path=feature_path,
                num_frames=int(features.shape[0]),
                feature_dim=int(features.shape[1]),
                model_id=model_id,
                layer=layer,
            )
        )

    write_cached_feature_manifest(output_manifest, cached_rows)
    return cached_rows


def write_cached_feature_manifest(path: str | Path, rows: list[CachedFeatureRow]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "utt_id",
        "condition",
        "normalized_text",
        "feature_path",
        "num_frames",
        "feature_dim",
        "model_id",
        "layer",
    ]
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "utt_id": row.utt_id,
                    "condition": row.condition,
                    "normalized_text": row.normalized_text,
                    "feature_path": str(row.feature_path),
                    "num_frames": row.num_frames,
                    "feature_dim": row.feature_dim,
                    "model_id": row.model_id,
                    "layer": row.layer,
                }
            )


def read_cached_feature_manifest(path: str | Path) -> list[CachedFeatureRow]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "utt_id",
            "condition",
            "normalized_text",
            "feature_path",
            "num_frames",
            "feature_dim",
            "model_id",
            "layer",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Cached feature manifest is missing required fields: {missing}")
        return [
            CachedFeatureRow(
                utt_id=row["utt_id"],
                condition=row["condition"],
                normalized_text=row["normalized_text"],
                feature_path=Path(row["feature_path"]),
                num_frames=int(row["num_frames"]),
                feature_dim=int(row["feature_dim"]),
                model_id=row["model_id"],
                layer=int(row["layer"]),
            )
            for row in reader
        ]
