# Noise-Robust SSL ASR

Course project codebase for **Noise-Robust Low-Resource ASR with Self-Supervised Speech Representations**.

The project asks a narrow research question:

> Can frozen self-supervised speech representations improve low-resource ASR robustness under additive noise, compared with a conventional log-mel CTC baseline?

## Codebase Choice

This repository intentionally starts as a small research codebase instead of an ESPnet or SpeechBrain recipe.

- **Why not ESPnet/SpeechBrain as the main codebase:** they are excellent full systems, but recipe-level changes can hide the experiment logic. For this course project, we need transparent control over noise conditions, SNR curves, frozen SSL features, and CTC baselines.
- **What we reuse conceptually:** CTC training, WER/CER evaluation, frozen SSL feature extraction, and recipe-style experiment configs.
- **What this repo optimizes for:** fast CPU smoke tests, clear experiment tables, and code that maps directly to the ICASSP-style report.

Use ESPnet/SpeechBrain later only as reference baselines or optional sanity checks, not as the primary project skeleton.

## Planned Experiment Story

1. Train lightweight ASR heads on clean low-resource speech.
2. Evaluate the same systems on clean and noisy test sets.
3. Compare how quickly WER/CER degrades as SNR drops.
4. Analyze whether SSL features reduce substitutions, deletions, or insertions.

Main comparison:

| System | Feature | Encoder/head | Backbone training |
|---|---|---|---|
| Baseline | log-mel | BiLSTM-CTC | none |
| Main | HuBERT hidden states | linear/BiLSTM-CTC | frozen |
| Optional | WavLM hidden states | linear/BiLSTM-CTC | frozen |

Noise conditions:

```text
clean, 20 dB, 10 dB, 5 dB
```

Primary metrics:

```text
WER, CER, relative WER increase, edit-count breakdown
```

## Repository Layout

```text
ASR/
  configs/                  # Experiment configs
  data/
    manifests/              # CSV manifests generated from LibriSpeech-style data
    raw/                    # Ignored raw data
    features/               # Ignored cached log-mel / SSL features
    noisy/                  # Ignored generated noisy audio
  docs/                     # Project story and implementation notes
  paper/                    # ICASSP paper draft area
  results/                  # Metrics, predictions, figures
  scripts/                  # CLI entry points
  src/asr_noise_robust/     # Reusable project package
  tests/                    # Fast stdlib unittest suite
```

## Quick Smoke Test

```bash
cd /Users/howardx/Documents/projects/ASR
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/00_smoke_test.py
PYTHONPATH=src python3 scripts/03_run_toy_pipeline.py --root runs/toy_pipeline
PYTHONPATH=src python3 scripts/04_run_manifest_train_eval.py \
  --train-manifest runs/toy_pipeline/data/manifests/toy_train.csv \
  --eval-manifest runs/toy_pipeline/data/manifests/toy_eval_conditions.csv \
  --model-path runs/toy_pipeline/checkpoints/toy_memorizer.json \
  --predictions-path runs/toy_pipeline/results/predictions/toy_memorizer_predictions.csv \
  --metrics-path runs/toy_pipeline/results/toy_memorizer_metrics.csv
uv run python scripts/05_run_torch_logmel_smoke.py \
  --train-manifest runs/toy_pipeline/data/manifests/toy_train.csv \
  --checkpoint-path runs/toy_pipeline/checkpoints/torch_logmel_smoke.pt \
  --summary-path runs/toy_pipeline/results/torch_logmel_smoke.json
uv run python scripts/06_run_torch_logmel_experiment.py \
  --train-manifest runs/toy_pipeline/data/manifests/toy_train.csv \
  --eval-manifest runs/toy_pipeline/data/manifests/toy_eval_conditions.csv \
  --checkpoint-path runs/toy_pipeline/checkpoints/torch_logmel_experiment.pt \
  --predictions-path runs/toy_pipeline/results/predictions/torch_logmel_predictions.csv \
  --metrics-path runs/toy_pipeline/results/torch_logmel_metrics.csv \
  --max-epochs 5
uv run python scripts/09_make_noisy_eval_set.py \
  --source-manifest data/manifests/dev-clean-2-smoke8.csv \
  --output-manifest runs/dev-clean-2-smoke8/data/manifests/noisy_eval.csv \
  --output-audio-dir runs/dev-clean-2-smoke8/data/noisy_eval_audio
```

The smoke test does not download models or data. It verifies text normalization, CTC decoding, metrics, and SNR mixing.
The toy pipeline additionally writes a tiny synthetic corpus, clean/noisy manifests, oracle predictions, and `results/metrics.csv` under the selected run root.
The manifest train/eval command proves that a training artifact, prediction CSV, and condition-level metrics can be produced from manifests before introducing heavy ASR dependencies.
The Torch log-mel smoke command proves that torchaudio feature extraction, a BiLSTM-CTC model, CTC loss, and checkpoint writing work in the local environment.
The Torch log-mel experiment command closes the first real ASR loop: train CTC, greedy-decode clean/noisy audio, and write WER/CER metrics.
The pretrained ASR command evaluates a HuggingFace CTC model on the same clean/noisy condition manifest:

```bash
uv run python scripts/12_eval_pretrained_asr.py \
  --model-id facebook/wav2vec2-base-960h \
  --eval-manifest runs/baseline_train128_eval32/data/manifests/noisy_eval.csv \
  --predictions-path runs/pretrained_wav2vec2_base960h_eval32/results/predictions/full_predictions.csv \
  --metrics-path runs/pretrained_wav2vec2_base960h_eval32/results/full_metrics.csv \
  --device cpu
```

## Environment

Minimal utilities only need Python stdlib. Full experiments need:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Heavy dependencies are intentionally kept out of the smoke tests.

## Export Final Tables And Figures

```bash
uv run python -B scripts/16_export_results.py \
  --tables-dir results/tables \
  --figures-dir results/figures
```

This writes:

- `results/tables/metrics_long.csv`
- `results/tables/metrics_summary.csv`
- `results/figures/wer_vs_snr.png`
- `results/figures/cer_vs_snr.png`
- `results/figures/relative_wer_increase.png`
- `results/figures/edit_breakdown_5db.png`

## First Real Run

Recommended order:

1. Download Mini LibriSpeech:
   ```bash
   uv run python scripts/07_download_minilibrispeech.py --root data/raw/mini_librispeech --parts dev-clean-2
   ```
2. Generate manifests in `data/manifests/`.
3. Run the toy pipeline once to verify local wiring.
4. Run manifest train/eval once to verify model-output wiring.
5. Run Torch log-mel smoke once to verify feature/model training wiring.
6. Train the log-mel CTC baseline.
7. Cache HuBERT features.
8. Train the frozen-HuBERT CTC model.
9. Evaluate clean/20dB/10dB/5dB conditions.
10. Export tables and figures to `results/`.

## Current CPU Results

| Run | Clean WER | 20 dB WER | 10 dB WER | 5 dB WER |
|---|---:|---:|---:|---:|
| log-mel TinyBiLSTM-CTC, train128 | 1.000 | 1.000 | 1.000 | 1.000 |
| cached `facebook/wav2vec2-base` hidden states + TinyBiLSTM-CTC, train128 | 1.000 | 1.142 | 1.187 | 1.084 |
| cached `facebook/wav2vec2-base-960h` hidden states + TinyBiLSTM-CTC, train128 | 0.055 | 0.073 | 0.441 | 0.869 |
| `facebook/wav2vec2-base-960h` pretrained CTC | 0.051 | 0.082 | 0.350 | 0.825 |

The pretrained CTC model is a reference baseline, not the final frozen-SSL-head experiment. It establishes that pretrained speech representations give a strong clean/mild-noise baseline while severe 5 dB noise remains challenging.
For pure `facebook/wav2vec2-base`, set `HF_HUB_DISABLE_XET=1` if the HuggingFace download stalls through the xet path.

## Frozen SSL Feature Pipeline

Cache SSL features once:

```bash
uv run python scripts/13_cache_ssl_features.py \
  --source-manifest data/manifests/train-clean-5-short8.csv \
  --output-manifest runs/frozen_wav2vec2_base960h_short8/data/manifests/train_features.csv \
  --output-dir runs/frozen_wav2vec2_base960h_short8/data/features/train \
  --model-id facebook/wav2vec2-base-960h \
  --layer -1 \
  --device cpu
```

Train a lightweight CTC head on cached features:

```bash
uv run python scripts/14_train_cached_ssl_ctc.py \
  --train-feature-manifest runs/frozen_wav2vec2_base960h_short8/data/manifests/train_features.csv \
  --checkpoint-path runs/frozen_wav2vec2_base960h_short8/checkpoints/ssl_ctc_short8.pt \
  --history-path runs/frozen_wav2vec2_base960h_short8/results/history.csv \
  --summary-path runs/frozen_wav2vec2_base960h_short8/results/summary.json \
  --hidden-dim 128 \
  --max-epochs 150 \
  --batch-size 2 \
  --learning-rate 0.001
```

Cache eval-condition SSL features:

```bash
uv run python scripts/13_cache_ssl_features.py \
  --source-manifest runs/overfit_short8_norm/data/manifests/noisy_eval.csv \
  --output-manifest runs/frozen_wav2vec2_base960h_short8/data/manifests/eval_features.csv \
  --output-dir runs/frozen_wav2vec2_base960h_short8/data/features/eval \
  --model-id facebook/wav2vec2-base-960h \
  --layer -1 \
  --device cpu
```

Evaluate the cached-SSL CTC head:

```bash
uv run python scripts/15_eval_cached_ssl_ctc.py \
  --checkpoint-path runs/frozen_wav2vec2_base960h_short8/checkpoints/ssl_ctc_short8.pt \
  --eval-feature-manifest runs/frozen_wav2vec2_base960h_short8/data/manifests/eval_features.csv \
  --predictions-path runs/frozen_wav2vec2_base960h_short8/results/predictions/ssl_ctc_short8_predictions.csv \
  --metrics-path runs/frozen_wav2vec2_base960h_short8/results/ssl_ctc_short8_metrics.csv
```

## Deliverables

- `paper/main.pdf`: ICASSP-style report.
- `results/tables/metrics_long.csv`: all WER/CER results in long format.
- `results/tables/metrics_summary.csv`: paper-ready summary table.
- `results/figures/`: final PNG figures.
- `runs/*/results/predictions/`: reference/prediction CSVs.
- `README.md`: reproduction commands.
- `configs/`: exact experiment settings.
