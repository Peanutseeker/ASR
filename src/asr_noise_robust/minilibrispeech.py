"""Mini LibriSpeech download helpers."""

from __future__ import annotations

from pathlib import Path
import shutil
import tarfile
from urllib.request import urlopen


MINILIBRISPEECH_PARTS = ("dev-clean-2", "train-clean-5")
OPENSLR_31_BASE_URL = "https://www.openslr.org/resources/31"


def minilibrispeech_url(part: str) -> str:
    if part not in MINILIBRISPEECH_PARTS:
        raise ValueError(f"Unknown Mini LibriSpeech part: {part}")
    return f"{OPENSLR_31_BASE_URL}/{part}.tar.gz"


def download_file(url: str, path: str | Path, chunk_size: int = 1024 * 1024) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return target
    with urlopen(url) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=chunk_size)
    return target


def extract_tarball(tarball: str | Path, destination: str | Path) -> Path:
    source = Path(tarball)
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source, "r:gz") as archive:
        archive.extractall(target)
    return target


def ensure_minilibrispeech_part(root: str | Path, part: str, extract: bool = True) -> Path:
    base = Path(root)
    url = minilibrispeech_url(part)
    tarball = download_file(url, base / "archives" / f"{part}.tar.gz")
    if extract:
        extract_tarball(tarball, base)
    return tarball

