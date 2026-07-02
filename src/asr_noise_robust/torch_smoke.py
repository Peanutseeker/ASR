"""End-to-end PyTorch smoke pipeline over a tiny manifest."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct
import wave

import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio

from asr_noise_robust.features import LogMelFeatureExtractor
from asr_noise_robust.manifest import read_manifest
from asr_noise_robust.text import Vocabulary
from asr_noise_robust.torch_models import TinyBiLstmCtc
from asr_noise_robust.torch_train import train_one_batch


@dataclass(frozen=True)
class TorchSmokeOutputs:
    checkpoint_path: Path
    summary_path: Path
    loss: float


def _load_mono_audio(path: str | Path, target_sample_rate: int) -> torch.Tensor:
    source = Path(path)
    try:
        waveform, sample_rate = torchaudio.load(str(source))
    except ImportError:
        if source.suffix.lower() != ".wav":
            raise
        waveform, sample_rate = _load_pcm16_wav(source)
    if waveform.ndim != 2:
        raise ValueError(f"Expected torchaudio waveform [channels, time], got {tuple(waveform.shape)}")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    return waveform.squeeze(0)


def _load_pcm16_wav(path: Path) -> tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM wav fallback is supported, got {sample_width} bytes")
    values = struct.unpack("<" + "h" * (len(frames) // 2), frames)
    tensor = torch.tensor(values, dtype=torch.float32).view(-1, channels).transpose(0, 1)
    tensor = tensor / 32768.0
    return tensor, sample_rate


def _make_batch(
    train_manifest: str | Path,
    n_mels: int,
    sample_rate: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    rows = read_manifest(train_manifest)
    vocab = Vocabulary.ctc_english()
    extractor = LogMelFeatureExtractor(sample_rate=sample_rate, n_mels=n_mels)
    features = []
    targets = []
    target_lengths = []

    for row in rows:
        waveform = _load_mono_audio(row.audio_path, target_sample_rate=sample_rate)
        item_features = extractor(waveform)
        features.append(item_features)
        token_ids = torch.tensor(vocab.encode(row.normalized_text), dtype=torch.long)
        targets.append(token_ids)
        target_lengths.append(token_ids.numel())

    padded_features = pad_sequence(features, batch_first=True)
    feature_lengths = torch.tensor([item.shape[0] for item in features], dtype=torch.long)
    flat_targets = torch.cat(targets)
    return (
        padded_features,
        feature_lengths,
        flat_targets,
        torch.tensor(target_lengths, dtype=torch.long),
        len(vocab.symbols),
    )


def run_torch_logmel_smoke(
    train_manifest: str | Path,
    checkpoint_path: str | Path,
    summary_path: str | Path,
    n_mels: int = 16,
    hidden_dim: int = 8,
    sample_rate: int = 16000,
) -> TorchSmokeOutputs:
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = train_one_batch(
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
        },
        checkpoint_target,
    )

    summary_target = Path(summary_path)
    summary_target.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "torch_logmel_smoke",
        "num_examples": int(features.shape[0]),
        "max_frames": int(features.shape[1]),
        "feature_dim": int(features.shape[2]),
        "feature_normalize": True,
        "vocab_size": int(vocab_size),
        "loss": float(loss),
        "checkpoint_path": str(checkpoint_target),
    }
    summary_target.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return TorchSmokeOutputs(
        checkpoint_path=checkpoint_target,
        summary_path=summary_target,
        loss=loss,
    )
