# Final Results Package

This document tracks the non-paper deliverables for the ASR project.

## Research Question

Does a clean-trained low-resource ASR system degrade under additive noise, and do pretrained speech representations reduce that degradation compared with log-mel features?

## Completed Systems

| System | Training data | Feature/model | Role |
|---|---|---|---|
| Log-mel CTC | train-clean-5 128 utterances | log-mel + TinyBiLSTM-CTC | from-scratch baseline |
| Wav2Vec2 base + CTC head | train-clean-5 128 utterances | cached `facebook/wav2vec2-base` hidden states + TinyBiLSTM-CTC | pure SSL ablation |
| Wav2Vec2 960h hidden + CTC head | train-clean-5 128 utterances | cached `facebook/wav2vec2-base-960h` hidden states + TinyBiLSTM-CTC | frozen ASR-fine-tuned encoder baseline |
| Wav2Vec2 960h pretrained CTC | no local training | `facebook/wav2vec2-base-960h` CTC head | pretrained reference |

All systems are evaluated on the same dev-clean-2 32-utterance set under `clean`, `20 dB`, `10 dB`, and `5 dB` conditions.

## Main Metrics

| System | Clean WER | 20 dB WER | 10 dB WER | 5 dB WER |
|---|---:|---:|---:|---:|
| Log-mel CTC | 1.000 | 1.000 | 1.000 | 1.000 |
| Wav2Vec2 base + CTC head | 1.000 | 1.142 | 1.187 | 1.084 |
| Wav2Vec2 960h hidden + CTC head | 0.055 | 0.073 | 0.441 | 0.869 |
| Wav2Vec2 960h pretrained CTC | 0.051 | 0.082 | 0.350 | 0.825 |

## Interpretation

1. The from-scratch log-mel baseline collapses in the 128-utterance setting.
2. Pure `facebook/wav2vec2-base` final-layer features are not enough with this small CTC head and 128 utterances; CER improves, but word-level decoding remains poor.
3. Frozen hidden states from the ASR-fine-tuned `facebook/wav2vec2-base-960h` checkpoint nearly match the pretrained CTC reference on clean and 20 dB conditions.
4. All strong pretrained systems degrade sharply at 10 dB and 5 dB, which supports the noise-robustness framing.

## Generated Artifacts

Tables:

- `results/tables/metrics_long.csv`
- `results/tables/metrics_summary.csv`

Figures:

- `results/figures/wer_vs_snr.png`
- `results/figures/cer_vs_snr.png`
- `results/figures/relative_wer_increase.png`
- `results/figures/edit_breakdown_5db.png`

Core scripts:

- `scripts/13_cache_ssl_features.py`
- `scripts/14_train_cached_ssl_ctc.py`
- `scripts/15_eval_cached_ssl_ctc.py`
- `scripts/16_export_results.py`

Config:

- `configs/final_train128_eval32.yaml`

## Reproduction Notes

Use `HF_HUB_DISABLE_XET=1` when downloading `facebook/wav2vec2-base` if HuggingFace's xet download path stalls.

Run final artifact export with:

```bash
uv run python -B scripts/16_export_results.py --tables-dir results/tables --figures-dir results/figures
```

Run the full test suite with:

```bash
uv run python -B -m unittest discover -s tests -v
```
