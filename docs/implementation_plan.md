# Implementation Plan

## Phase 1: Foundation

- Keep the current stdlib unit tests green.
- Keep `scripts/03_run_toy_pipeline.py --root runs/toy_pipeline` green before touching real data.
- Keep `scripts/04_run_manifest_train_eval.py` green on the toy manifests before replacing the recognizer.
- Keep `scripts/05_run_torch_logmel_smoke.py` green before starting full baseline training.
- Keep `scripts/06_run_torch_logmel_experiment.py` green on toy manifests before using real LibriSpeech data.
- Generate LibriSpeech-style manifests.
- Verify transcript normalization and vocabulary coverage.

## Phase 2: Baseline

- Implement log-mel feature extraction with `torchaudio`.
- Train `log-mel + BiLSTM-CTC` on clean low-resource training data.
- Evaluate clean test WER/CER.

## Phase 3: Noise Protocol

- Generate noisy test manifests for 20 dB, 10 dB, and 5 dB.
- Keep training data clean first.
- Evaluate the log-mel baseline on all SNR conditions.

## Phase 4: SSL Features

- Cache HuBERT hidden states for layers 6, 9, and 12.
- Train lightweight CTC heads on frozen HuBERT features.
- Evaluate the same clean/noisy test conditions.

## Phase 5: Analysis

- Build `results/metrics.csv`.
- Plot WER/CER versus SNR.
- Compute relative WER increase from clean to each noisy condition.
- Produce edit-count breakdown and representative error cases.

## Phase 6: Paper

- Write the ICASSP-style report around the SNR robustness story.
- Keep optional WavLM or fine-tuning experiments as stretch goals.
