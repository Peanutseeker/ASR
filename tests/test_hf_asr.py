import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from asr_noise_robust.audio import sine_wave, write_pcm16_wav
from asr_noise_robust.hf_asr import evaluate_pretrained_asr, transcribe_eval_rows
from asr_noise_robust.torch_experiment import EvalAudioRow


class FakeTranscriber:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.seen_shapes = []

    def __call__(self, waveform: torch.Tensor) -> str:
        self.seen_shapes.append(tuple(waveform.shape))
        return self.outputs.pop(0)


class HfAsrTests(unittest.TestCase):
    def test_transcribe_eval_rows_normalizes_transcriber_output(self):
        with TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / "audio.wav"
            write_pcm16_wav(wav_path, sine_wave(440, seconds=0.1))
            transcriber = FakeTranscriber(["HELLO, WORLD!"])
            rows = [
                EvalAudioRow(
                    utt_id="utt-1",
                    audio_path=str(wav_path),
                    condition="clean",
                    reference="hello world",
                )
            ]

            predictions = transcribe_eval_rows(
                eval_rows=rows,
                transcriber=transcriber,
                sample_rate=16000,
                model_id="fake/model",
            )

            self.assertEqual(len(predictions), 1)
            self.assertEqual(predictions[0].prediction, "hello world")
            self.assertEqual(predictions[0].feature, "pretrained_ctc")
            self.assertEqual(predictions[0].model, "fake/model")
            self.assertEqual(transcriber.seen_shapes, [(1600,)])

    def test_evaluate_pretrained_asr_writes_predictions_and_metrics(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            wav_path = root / "audio.wav"
            write_pcm16_wav(wav_path, sine_wave(440, seconds=0.1))
            eval_manifest = root / "eval.csv"
            with eval_manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["utt_id", "audio_path", "condition", "normalized_text"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "utt_id": "utt-1",
                        "audio_path": str(wav_path),
                        "condition": "clean",
                        "normalized_text": "hello world",
                    }
                )
                writer.writerow(
                    {
                        "utt_id": "utt-1",
                        "audio_path": str(wav_path),
                        "condition": "snr_20",
                        "normalized_text": "hello world",
                    }
                )

            outputs = evaluate_pretrained_asr(
                model_id="fake/model",
                eval_manifest=eval_manifest,
                predictions_path=root / "predictions.csv",
                metrics_path=root / "metrics.csv",
                transcriber=FakeTranscriber(["HELLO WORLD", "HELLO"]),
            )

            self.assertTrue(outputs.predictions_path.exists())
            self.assertTrue(outputs.metrics_path.exists())
            with outputs.metrics_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["test_condition"] for row in rows], ["clean", "snr_20"])
            self.assertEqual(rows[0]["wer"], "0.000000")


if __name__ == "__main__":
    unittest.main()
