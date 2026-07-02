# Experiment Matrix

## Required Experiments

| ID | Train condition | Test condition | Feature | Model | Purpose |
|---|---|---|---|---|---|
| E1 | clean | clean | log-mel | BiLSTM-CTC | clean baseline |
| E2 | clean | 20 dB | log-mel | BiLSTM-CTC | mild-noise baseline |
| E3 | clean | 10 dB | log-mel | BiLSTM-CTC | medium-noise baseline |
| E4 | clean | 5 dB | log-mel | BiLSTM-CTC | hard-noise baseline |
| E5 | clean | clean | HuBERT layer 9 | BiLSTM-CTC | clean SSL model |
| E6 | clean | 20 dB | HuBERT layer 9 | BiLSTM-CTC | mild-noise SSL model |
| E7 | clean | 10 dB | HuBERT layer 9 | BiLSTM-CTC | medium-noise SSL model |
| E8 | clean | 5 dB | HuBERT layer 9 | BiLSTM-CTC | hard-noise SSL model |

## Completed CPU Baselines

| Run | Train/eval data | Feature/model | Clean WER | 20 dB WER | 10 dB WER | 5 dB WER | Notes |
|---|---|---|---:|---:|---:|---:|---|
| `baseline_train128_norm_eval32_e10` | train-clean-5 128 train, dev-clean-2 32 eval | log-mel + TinyBiLSTM-CTC | 1.000 | 1.000 | 1.000 | 1.000 | from-scratch CPU baseline collapses to repeated characters |
| `frozen_wav2vec2_base_train128_eval32` | train-clean-5 128 train, dev-clean-2 32 eval | cached `facebook/wav2vec2-base` hidden states + TinyBiLSTM-CTC | 1.000 | 1.142 | 1.187 | 1.084 | pure SSL checkpoint, layer -1, 80 epochs; CER improves but word-level decoding remains poor |
| `frozen_wav2vec2_base960h_train128_eval32` | train-clean-5 128 train, dev-clean-2 32 eval | cached `facebook/wav2vec2-base-960h` hidden states + TinyBiLSTM-CTC | 0.055 | 0.073 | 0.441 | 0.869 | lightweight head trained on cached frozen ASR-fine-tuned encoder features |
| `pretrained_wav2vec2_base960h_eval32` | dev-clean-2 32 eval | `facebook/wav2vec2-base-960h` pretrained CTC | 0.051 | 0.082 | 0.350 | 0.825 | reference pretrained ASR baseline; strong clean/mild-noise, degrades sharply at 5 dB |

## Pipeline Sanity Checks

| Run | Data | Result | Purpose |
|---|---|---|---|
| `overfit_short8_norm` | shortest 8 train-clean-5 utterances | clean WER 0.171, clean CER 0.033 | proves the log-mel CTC pipeline can learn when the sanity set is appropriately short |
| `frozen_wav2vec2_base960h_short8` | same shortest 8 train/eval utterances | clean WER 0.029, 20 dB WER 0.171, 10 dB WER 0.543, 5 dB WER 0.743 | proves cached frozen SSL features plus a lightweight CTC head learn faster than log-mel on the same sanity set |
| `overfit_train8_norm` | first 8 train-clean-5 utterances | clean WER 0.978, clean CER 0.848 | shows long-utterance 8-example overfit is a poor sanity test |

## Optional Layer Ablation

| ID | Feature | Layer | Test conditions |
|---|---|---:|---|
| A1 | HuBERT | 6 | clean, 10 dB |
| A2 | HuBERT | 9 | clean, 10 dB |
| A3 | HuBERT | 12 | clean, 10 dB |

## Optional Model Ablation

| ID | Feature | Model | Test conditions |
|---|---|---|---|
| M1 | HuBERT layer 9 | BiLSTM-CTC | clean, 10 dB |
| M2 | WavLM layer 9 | BiLSTM-CTC | clean, 10 dB |
