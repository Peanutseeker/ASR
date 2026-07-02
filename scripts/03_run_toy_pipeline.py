#!/usr/bin/env python3
"""Run the dependency-light toy ASR pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from asr_noise_robust.pipeline import run_toy_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()

    outputs = run_toy_pipeline(args.root)
    print(f"train_manifest={outputs.train_manifest}")
    print(f"eval_manifest={outputs.eval_manifest}")
    print(f"predictions_csv={outputs.predictions_csv}")
    print(f"metrics_csv={outputs.metrics_csv}")


if __name__ == "__main__":
    main()

