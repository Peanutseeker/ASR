"""Small operations over CSV manifests."""

from __future__ import annotations

from pathlib import Path

from asr_noise_robust.manifest import read_manifest, write_manifest


def write_manifest_subset(source: str | Path, target: str | Path, limit: int) -> int:
    if limit <= 0:
        raise ValueError("limit must be positive")
    rows = read_manifest(source)[:limit]
    write_manifest(target, rows)
    return len(rows)

