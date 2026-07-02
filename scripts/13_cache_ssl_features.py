#!/usr/bin/env python3
"""Cache frozen SSL hidden states for a manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from asr_noise_robust.ssl_features import cache_ssl_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-id", default="facebook/wav2vec2-base")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-examples", type=int)
    args = parser.parse_args()

    rows = cache_ssl_features(
        source_manifest=args.source_manifest,
        output_manifest=args.output_manifest,
        output_dir=args.output_dir,
        model_id=args.model_id,
        layer=args.layer,
        sample_rate=args.sample_rate,
        device=args.device,
        max_examples=args.max_examples,
    )
    print(f"output_manifest={args.output_manifest}")
    print(f"num_cached={len(rows)}")


if __name__ == "__main__":
    main()
