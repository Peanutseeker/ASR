# Noise-Robust Low-Resource ASR

This repository contains a small research project on **low-resource ASR with frozen self-supervised speech representations**.

We train lightweight CTC heads on 128 clean LibriSpeech utterances, then evaluate the same utterances under clean, 20 dB, 10 dB, and 5 dB additive-noise conditions.

## Research Question

Can frozen pretrained speech representations improve low-resource ASR robustness compared with a from-scratch log-mel CTC baseline?

## Main Result

| System | Clean WER | 20 dB | 10 dB | 5 dB |
|---|---:|---:|---:|---:|
| Log-mel CTC | 1.000 | 1.000 | 1.000 | 1.000 |
| wav2vec2-base frozen + CTC head | 1.000 | 1.142 | 1.187 | 1.084 |
| wav2vec2-base-960h frozen + CTC head | 0.055 | 0.073 | 0.441 | 0.869 |
| wav2vec2-base-960h pretrained CTC | 0.051 | 0.082 | 0.350 | 0.825 |

Frozen ASR-pretrained wav2vec 2.0 representations work well on clean and mild-noise speech, but severe 5 dB noise still causes large WER increases. The paper also reports CER and word-level substitution/deletion/insertion counts.

## Repository Layout

```text
configs/      Experiment configs
docs/         Project notes and final result summary
paper/        ICASSP-style report, figures, and requirement audit
results/      Final metric tables
scripts/      Data, training, evaluation, and figure-generation CLIs
src/          Reusable ASR/noise/metric package
tests/        Fast unittest suite
```

Ignored local artifacts include raw audio, cached features, noisy audio, model runs, predictions, Canvas cache, and LaTeX build files.

## Setup

```bash
cd ASR
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The project also supports `uv run ...` commands, which were used for the final experiments.

## Quick Check

```bash
uv run python -B -m unittest discover -s tests -v
```

This verifies the manifest utilities, CTC decoding, WER/CER metrics, noise mixing, toy pipeline, log-mel model code, frozen-feature pipeline, and result export code.

## Reproduce Final Tables And Figures

The final metric CSVs are already stored in `results/tables/`. Regenerate paper figures with:

```bash
uv run python -B scripts/17_make_paper_figures.py \
  --metrics-long results/tables/metrics_long.csv \
  --figures-dir paper/figures
```

Regenerate the report PDF:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Primary deliverables:

- `paper/main.pdf`
- `paper/main.tex`
- `paper/figures/paper_robustness_main.pdf`
- `results/tables/metrics_long.csv`
- `results/tables/metrics_summary.csv`
- `paper/requirement_audit.md`

## Full Experiment Entry Points

The main scripts are:

```text
scripts/07_download_minilibrispeech.py
scripts/09_make_noisy_eval_set.py
scripts/10_train_logmel_ctc.py
scripts/12_eval_pretrained_asr.py
scripts/13_cache_ssl_features.py
scripts/14_train_cached_ssl_ctc.py
scripts/15_eval_cached_ssl_ctc.py
scripts/16_export_results.py
```

The final configuration is in `configs/final_train128_eval32.yaml`; the compact result summary is in `docs/final_results_package.md`.
