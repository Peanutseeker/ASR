#!/usr/bin/env python3
"""Train a lightweight CTC head on cached frozen SSL features."""

from __future__ import annotations

import argparse
from pathlib import Path

from asr_noise_robust.ssl_trainer import run_cached_ssl_ctc_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-feature-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--history-path", required=True, type=Path)
    parser.add_argument("--summary-path", required=True, type=Path)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    outputs = run_cached_ssl_ctc_training(
        train_feature_manifest=args.train_feature_manifest,
        checkpoint_path=args.checkpoint_path,
        history_path=args.history_path,
        summary_path=args.summary_path,
        hidden_dim=args.hidden_dim,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        log_every=args.log_every,
    )
    print(f"checkpoint_path={outputs.checkpoint_path}")
    print(f"history_path={outputs.history_path}")
    print(f"summary_path={outputs.summary_path}")
    print(f"final_loss={outputs.final_loss:.6f}")


if __name__ == "__main__":
    main()
