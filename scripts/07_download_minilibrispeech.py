#!/usr/bin/env python3
"""Download Mini LibriSpeech splits from OpenSLR SLR31."""

from __future__ import annotations

import argparse
from pathlib import Path

from asr_noise_robust.minilibrispeech import MINILIBRISPEECH_PARTS, ensure_minilibrispeech_part


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/mini_librispeech"))
    parser.add_argument("--parts", nargs="+", choices=MINILIBRISPEECH_PARTS, default=["dev-clean-2"])
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()

    for part in args.parts:
        tarball = ensure_minilibrispeech_part(args.root, part, extract=not args.no_extract)
        print(f"{part}={tarball}")


if __name__ == "__main__":
    main()

