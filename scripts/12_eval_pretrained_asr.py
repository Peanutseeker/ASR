#!/usr/bin/env python3
"""Evaluate a pretrained HuggingFace CTC ASR model on a condition manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from asr_noise_robust.hf_asr import evaluate_pretrained_asr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--eval-manifest", required=True, type=Path)
    parser.add_argument("--predictions-path", required=True, type=Path)
    parser.add_argument("--metrics-path", required=True, type=Path)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-eval-examples", type=int)
    args = parser.parse_args()

    outputs = evaluate_pretrained_asr(
        model_id=args.model_id,
        eval_manifest=args.eval_manifest,
        predictions_path=args.predictions_path,
        metrics_path=args.metrics_path,
        sample_rate=args.sample_rate,
        device=args.device,
        max_eval_examples=args.max_eval_examples,
    )
    print(f"predictions_path={outputs.predictions_path}")
    print(f"metrics_path={outputs.metrics_path}")


if __name__ == "__main__":
    main()
