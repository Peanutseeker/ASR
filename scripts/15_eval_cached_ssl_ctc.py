#!/usr/bin/env python3
"""Evaluate a cached-SSL CTC checkpoint on cached eval features."""

from __future__ import annotations

import argparse
from pathlib import Path

from asr_noise_robust.ssl_eval import evaluate_cached_ssl_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--eval-feature-manifest", required=True, type=Path)
    parser.add_argument("--predictions-path", required=True, type=Path)
    parser.add_argument("--metrics-path", required=True, type=Path)
    args = parser.parse_args()

    outputs = evaluate_cached_ssl_checkpoint(
        checkpoint_path=args.checkpoint_path,
        eval_feature_manifest=args.eval_feature_manifest,
        predictions_path=args.predictions_path,
        metrics_path=args.metrics_path,
    )
    print(f"predictions_path={outputs.predictions_path}")
    print(f"metrics_path={outputs.metrics_path}")


if __name__ == "__main__":
    main()
