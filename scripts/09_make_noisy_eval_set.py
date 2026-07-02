#!/usr/bin/env python3
"""Create clean/noisy eval manifest and noisy wav files from a clean manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from asr_noise_robust.noise_dataset import materialize_noisy_eval_set


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--output-audio-dir", required=True, type=Path)
    parser.add_argument("--snrs", nargs="+", type=int, default=[20, 10, 5])
    args = parser.parse_args()

    count = materialize_noisy_eval_set(
        source_manifest=args.source_manifest,
        output_manifest=args.output_manifest,
        output_audio_dir=args.output_audio_dir,
        snr_values=tuple(args.snrs),
    )
    print(f"wrote {count} eval rows to {args.output_manifest}")


if __name__ == "__main__":
    main()

