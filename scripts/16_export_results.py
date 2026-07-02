#!/usr/bin/env python3
"""Export final experiment tables and figures."""

from __future__ import annotations

import argparse
from pathlib import Path

from asr_noise_robust.results_export import ResultSpec, export_result_artifacts


DEFAULT_SPECS = [
    ResultSpec(
        system_id="logmel_train128",
        system_label="Log-mel CTC",
        system_group="from_scratch",
        metrics_path=Path(
            "runs/baseline_train128_norm_eval32_e10/results/"
            "logmel_ctc_train128_norm_e10_metrics.csv"
        ),
    ),
    ResultSpec(
        system_id="wav2vec2_base_ssl_head",
        system_label="Wav2Vec2 base + CTC head",
        system_group="pure_ssl",
        metrics_path=Path(
            "runs/frozen_wav2vec2_base_train128_eval32/results/"
            "ssl_ctc_train128_e80_metrics.csv"
        ),
    ),
    ResultSpec(
        system_id="wav2vec2_base960h_hidden_head",
        system_label="Wav2Vec2 960h hidden + CTC head",
        system_group="asr_finetuned_encoder",
        metrics_path=Path(
            "runs/frozen_wav2vec2_base960h_train128_eval32/results/"
            "ssl_ctc_train128_e30_metrics.csv"
        ),
    ),
    ResultSpec(
        system_id="wav2vec2_base960h_ctc",
        system_label="Wav2Vec2 960h pretrained CTC",
        system_group="reference",
        metrics_path=Path("runs/pretrained_wav2vec2_base960h_eval32/results/full_metrics.csv"),
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    outputs = export_result_artifacts(
        specs=DEFAULT_SPECS,
        tables_dir=args.tables_dir,
        figures_dir=args.figures_dir,
    )
    print(f"long_metrics_path={outputs.long_metrics_path}")
    print(f"summary_table_path={outputs.summary_table_path}")
    for figure_path in outputs.figure_paths:
        print(f"figure_path={figure_path}")


if __name__ == "__main__":
    main()
