#!/usr/bin/env python3
"""Train and evaluate a tiny log-mel CTC model over condition manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from asr_noise_robust.torch_experiment import run_torch_logmel_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--eval-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--predictions-path", required=True, type=Path)
    parser.add_argument("--metrics-path", required=True, type=Path)
    parser.add_argument("--n-mels", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=5)
    args = parser.parse_args()

    outputs = run_torch_logmel_experiment(
        train_manifest=args.train_manifest,
        eval_manifest=args.eval_manifest,
        checkpoint_path=args.checkpoint_path,
        predictions_path=args.predictions_path,
        metrics_path=args.metrics_path,
        n_mels=args.n_mels,
        hidden_dim=args.hidden_dim,
        max_epochs=args.max_epochs,
    )
    print(f"checkpoint_path={outputs.checkpoint_path}")
    print(f"predictions_path={outputs.predictions_path}")
    print(f"metrics_path={outputs.metrics_path}")
    print(f"final_loss={outputs.final_loss:.6f}")


if __name__ == "__main__":
    main()

