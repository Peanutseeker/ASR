"""Feature extraction modules for ASR baselines."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torchaudio


@dataclass(frozen=True)
class LogMelFeatureExtractor:
    """Extract time-major log-mel features from a mono waveform tensor."""

    sample_rate: int = 16000
    n_mels: int = 80
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    f_min: float = 20.0
    f_max: float | None = None
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.n_mels <= 0:
            raise ValueError("n_mels must be positive")

    @property
    def win_length(self) -> int:
        return int(round(self.sample_rate * self.frame_ms / 1000.0))

    @property
    def hop_length(self) -> int:
        return int(round(self.sample_rate * self.hop_ms / 1000.0))

    @property
    def n_fft(self) -> int:
        value = 1
        while value < self.win_length:
            value *= 2
        return value

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2 or waveform.shape[0] != 1:
            raise ValueError("waveform must have shape [time] or [1, time]")

        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            f_min=self.f_min,
            f_max=self.f_max,
            n_mels=self.n_mels,
            power=2.0,
        )(waveform)
        log_mel = torch.log(torch.clamp(mel, min=1e-10))
        features = log_mel.squeeze(0).transpose(0, 1).contiguous()
        if not self.normalize:
            return features

        mean = features.mean()
        std = features.std(unbiased=False).clamp_min(1e-5)
        return (features - mean) / std
