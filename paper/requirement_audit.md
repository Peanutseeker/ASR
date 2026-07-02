# Course Project Requirement Audit

Source checked: Canvas assignment "课程大作业：基于语音自监督表征的 ASR/TTS 系统设计" in
`/Users/howardx/Documents/private-notes/Coding-Notes/realwonder_project/canvas_sync/courses/87522-III/assignments.md`,
lines 715-1033. Due date: 2026-07-05 23:59:59 Asia/Shanghai.

## Chosen Direction

- Requirement: choose ASR or TTS.
- Status: satisfied.
- Evidence: the project chooses low-resource ASR and reports transcription WER/CER.

## Self-Supervised Speech Representation

- Requirement: use at least one speech SSL model or representation.
- Status: satisfied.
- Evidence: wav2vec 2.0 base hidden states and wav2vec 2.0 base-960h hidden states are used as frozen continuous speech representations.

## Downstream System

- Requirement: build a system for a clear ASR or TTS downstream task.
- Status: satisfied.
- Evidence: the system performs speech-to-text recognition with CTC heads and greedy decoding.

## Quantitative Evaluation

- Requirement: include quantitative metrics.
- Status: satisfied.
- Evidence: WER and CER are reported across clean, 20 dB, 10 dB, and 5 dB conditions.

## Comparison or Ablation

- Requirement: include at least one comparison or ablation experiment.
- Status: satisfied.
- Evidence: comparisons include log-mel CTC, wav2vec 2.0 base frozen features, wav2vec 2.0 base-960h frozen features, and a pretrained wav2vec 2.0 CTC reference.

## Independent Thinking and Innovation

- Requirement: show independent thinking; innovation may appear in representation choice, discretization, model structure, training objective, or system combination.
- Status: satisfied after the latest paper revision.
- Evidence: the paper now frames the contribution as a CPU-friendly stress-test protocol combining low-resource clean training, frozen continuous SSL representations, matched CTC heads, clean-to-noisy SNR evaluation, and edit-type failure diagnosis. This is an innovation in system combination and evaluation design, not a new ASR architecture.

## Recommended ASR Analyses

- Different SSL models or settings: partially satisfied through wav2vec 2.0 base versus ASR-fine-tuned wav2vec 2.0 base-960h.
- Low-resource data scale: satisfied by the fixed 128-utterance training regime.
- Error-case analysis: satisfied through substitution/deletion/insertion counts at 5 dB.
- Hidden-layer comparison: not included.
- Discrete unit/codebook/bitrate analysis: not applicable because the chosen system uses continuous hidden states rather than discrete units.
- Inference speed/RTF: not included; optional in the assignment.

## Paper and Submission Format

- Requirement: ICASSP 2026 LaTeX format.
- Status: satisfied.
- Evidence: `paper/main.tex` uses the local ICASSP-style template and compiles to `paper/main.pdf`.

- Requirement: submit PDF plus reproducible code/supporting materials.
- Status: satisfied.
- Evidence: `paper/main.pdf`, scripts, configs, result CSVs, and figure-generation scripts are present.

- Requirement: note group member names and student IDs in the report.
- Status: partially satisfied.
- Evidence: names are now in `paper/main.tex`: Haowen Xu, Yuchen Jiang, and Jiajun Wang.
- Remaining gap: student IDs are not available in the project context and still need to be added before submission.
