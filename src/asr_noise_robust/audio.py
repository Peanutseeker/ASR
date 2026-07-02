"""Small stdlib audio helpers for dependency-light pipeline smoke tests."""

from __future__ import annotations

import math
from pathlib import Path
import random
import struct
import wave


def sine_wave(
    frequency_hz: float,
    seconds: float,
    sample_rate: int = 16000,
    amplitude: float = 0.25,
) -> list[float]:
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    total = int(round(seconds * sample_rate))
    return [
        amplitude * math.sin(2.0 * math.pi * frequency_hz * frame / sample_rate)
        for frame in range(total)
    ]


def deterministic_noise(length: int, seed: int, amplitude: float = 0.25) -> list[float]:
    if length <= 0:
        raise ValueError("length must be positive")
    rng = random.Random(seed)
    return [rng.uniform(-amplitude, amplitude) for _ in range(length)]


def write_pcm16_wav(path: str | Path, samples: list[float], sample_rate: int = 16000) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    for sample in samples:
        clipped = max(-1.0, min(1.0, float(sample)))
        frames.extend(struct.pack("<h", int(round(clipped * 32767.0))))

    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(frames))

