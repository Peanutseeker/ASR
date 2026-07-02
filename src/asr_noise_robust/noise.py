"""Deterministic helpers for SNR-controlled noise mixing."""

from __future__ import annotations

import math


def _as_float_list(samples: list[float] | tuple[float, ...]) -> list[float]:
    values = [float(sample) for sample in samples]
    if not values:
        raise ValueError("Audio sample sequence must not be empty")
    return values


def rms(samples: list[float] | tuple[float, ...]) -> float:
    values = _as_float_list(samples)
    return math.sqrt(sum(sample * sample for sample in values) / len(values))


def repeat_or_crop(samples: list[float] | tuple[float, ...], target_length: int) -> list[float]:
    values = _as_float_list(samples)
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    repeated: list[float] = []
    while len(repeated) < target_length:
        repeated.extend(values)
    return repeated[:target_length]


def mix_at_snr_db(
    clean: list[float] | tuple[float, ...],
    noise: list[float] | tuple[float, ...],
    snr_db: float,
) -> list[float]:
    clean_values = _as_float_list(clean)
    noise_values = repeat_or_crop(noise, len(clean_values))
    clean_rms = rms(clean_values)
    noise_rms = rms(noise_values)
    if noise_rms == 0:
        raise ValueError("Noise RMS must be greater than zero")

    target_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
    scale = target_noise_rms / noise_rms
    return [clean_sample + scale * noise_sample for clean_sample, noise_sample in zip(clean_values, noise_values)]


def measured_snr_db(
    clean: list[float] | tuple[float, ...],
    mixed: list[float] | tuple[float, ...],
) -> float:
    clean_values = _as_float_list(clean)
    mixed_values = _as_float_list(mixed)
    if len(clean_values) != len(mixed_values):
        raise ValueError("clean and mixed sequences must have the same length")
    noise_values = [mixed_sample - clean_sample for clean_sample, mixed_sample in zip(clean_values, mixed_values)]
    noise_rms = rms(noise_values)
    if noise_rms == 0:
        return math.inf
    return 20.0 * math.log10(rms(clean_values) / noise_rms)

