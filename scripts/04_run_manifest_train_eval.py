#!/usr/bin/env python3
"""Run dependency-light manifest train/eval over a prepared condition manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from asr_noise_robust.train_eval import run_manifest_train_eval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--eval-manifest", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--predictions-path", required=True, type=Path)
    parser.add_argument("--metrics-path", required=True, type=Path)
    args = parser.parse_args()

    outputs = run_manifest_train_eval(
        train_manifest=args.train_manifest,
        eval_manifest=args.eval_manifest,
        model_path=args.model_path,
        predictions_path=args.predictions_path,
        metrics_path=args.metrics_path,
    )
    print(f"model_path={outputs.model_path}")
    print(f"predictions_path={outputs.predictions_path}")
    print(f"metrics_path={outputs.metrics_path}")


if __name__ == "__main__":
    main()

