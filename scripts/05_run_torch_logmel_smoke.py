#!/usr/bin/env python3
"""Run a one-batch Torch log-mel CTC smoke test over a prepared train manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from asr_noise_robust.torch_smoke import run_torch_logmel_smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--summary-path", required=True, type=Path)
    parser.add_argument("--n-mels", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=8)
    args = parser.parse_args()

    outputs = run_torch_logmel_smoke(
        train_manifest=args.train_manifest,
        checkpoint_path=args.checkpoint_path,
        summary_path=args.summary_path,
        n_mels=args.n_mels,
        hidden_dim=args.hidden_dim,
    )
    print(f"checkpoint_path={outputs.checkpoint_path}")
    print(f"summary_path={outputs.summary_path}")
    print(f"loss={outputs.loss:.6f}")


if __name__ == "__main__":
    main()

