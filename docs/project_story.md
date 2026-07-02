# Project Story

## Working Title

**Noise-Robust Low-Resource ASR with Self-Supervised Speech Representations**

## Core Hypothesis

In low-resource ASR, a model trained only on clean labeled speech degrades quickly when test audio contains background noise. Traditional log-mel features expose local spectral energy directly, so additive noise can strongly perturb the downstream CTC model. Pre-trained self-supervised speech models such as HuBERT and WavLM may provide more stable phonetic representations, even when the backbone is frozen.

This project tests whether frozen SSL representations improve noise robustness without requiring GPU-heavy full fine-tuning.

## Main Question

When the downstream ASR model is trained on clean low-resource speech, do frozen SSL features produce a smaller WER/CER increase than log-mel features as SNR decreases?

## Experimental Claim Shape

The report should be able to make one of these defensible claims:

1. SSL features improve both clean and noisy ASR performance.
2. SSL features have similar clean performance but degrade more slowly under noise.
3. SSL features help at moderate noise but fail at very low SNR, showing the limit of frozen representations.
4. SSL features do not help under this setup, suggesting that noise-aware training or fine-tuning is needed.

Any of these outcomes is publishable as a course-project story because the experiment is controlled and the conclusion is interpretable.

## Current Evidence

The initial CPU runs show a useful contrast:

1. A from-scratch log-mel TinyBiLSTM-CTC baseline is too weak in the 128-utterance setting. On the dev-clean-2 32-utterance clean/noisy evaluation set, WER remains 1.000 across clean, 20 dB, 10 dB, and 5 dB conditions.
2. The same log-mel pipeline can learn in a controlled sanity check. On the shortest 8 training utterances, the model reaches clean WER 0.171 and clean CER 0.033, so the implementation is not fundamentally broken.
3. Cached frozen Wav2Vec2 hidden states plus a lightweight CTC head improve the same short-utterance sanity setup: clean WER is 0.029, and WER at 5 dB is 0.743 instead of the log-mel model's 1.000.
4. On the train128/dev32 setting, a lightweight CTC head trained on pure `facebook/wav2vec2-base` hidden states remains weak after 80 epochs: clean WER is 1.000, although CER improves to 0.496. This suggests that pure SSL final-layer features plus a small CTC head need more data, a better layer choice, or a stronger decoder.
5. A lightweight CTC head trained on cached frozen Wav2Vec2 `base-960h` encoder states reaches WER 0.055 on clean, 0.073 at 20 dB, 0.441 at 10 dB, and 0.869 at 5 dB. This nearly matches the reference CTC model in clean and mild noise, but still degrades sharply in stronger noise.
6. A pretrained Wav2Vec2 CTC reference model performs strongly on clean and mild noise but still degrades under harder noise: WER is 0.051 on clean, 0.082 at 20 dB, 0.350 at 10 dB, and 0.825 at 5 dB.

This supports a sharper project story: low-resource from-scratch ASR is unstable, pretrained speech representations are dramatically better, and severe noise remains a meaningful failure mode worth targeting with noise-aware evaluation or adaptation.

Implementation note: the first `facebook/wav2vec2-base` download stalled through HuggingFace's xet path, but setting `HF_HUB_DISABLE_XET=1` allowed the pure SSL ablation to run. The completed results should distinguish pure SSL hidden states from the stronger ASR-fine-tuned `base-960h` hidden states.

## Figures To Aim For

1. WER vs SNR curve.
2. CER vs SNR curve.
3. Bar chart of relative WER increase from clean to 5 dB.
4. Edit-count breakdown for baseline and SSL model.

## Minimal Paper Contributions

1. A reproducible low-resource ASR pipeline for comparing log-mel and frozen SSL representations under controlled noise.
2. A clean/noisy evaluation protocol with fixed SNR conditions.
3. An analysis of whether SSL representations reduce WER/CER degradation and which error types change.
