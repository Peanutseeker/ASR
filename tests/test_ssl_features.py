import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from asr_noise_robust.audio import sine_wave, write_pcm16_wav
from asr_noise_robust.ssl_features import cache_ssl_features, read_cached_feature_manifest


class FakeSslExtractor:
    def __init__(self, feature_dim: int = 3):
        self.feature_dim = feature_dim
        self.calls = 0

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        frames = max(2, waveform.numel() // 800)
        return torch.arange(frames * self.feature_dim, dtype=torch.float32).view(
            frames, self.feature_dim
        )


class SslFeatureTests(unittest.TestCase):
    def test_cache_ssl_features_writes_feature_tensors_and_manifest(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            wav_path = root / "utt.wav"
            write_pcm16_wav(wav_path, sine_wave(440, seconds=0.2))
            source_manifest = root / "train.csv"
            with source_manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["utt_id", "audio_path", "duration", "text", "normalized_text", "split"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "utt_id": "utt-1",
                        "audio_path": str(wav_path),
                        "duration": "0.2",
                        "text": "Hello World",
                        "normalized_text": "hello world",
                        "split": "train",
                    }
                )

            output_manifest = root / "features.csv"
            rows = cache_ssl_features(
                source_manifest=source_manifest,
                output_manifest=output_manifest,
                output_dir=root / "features",
                model_id="fake/ssl",
                layer=9,
                extractor=FakeSslExtractor(feature_dim=4),
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].utt_id, "utt-1")
            self.assertEqual(rows[0].condition, "clean")
            self.assertEqual(rows[0].feature_dim, 4)
            self.assertEqual(rows[0].model_id, "fake/ssl")
            self.assertEqual(rows[0].layer, 9)
            self.assertTrue(rows[0].feature_path.exists())
            self.assertEqual(tuple(torch.load(rows[0].feature_path, weights_only=False).shape), (4, 4))

            loaded = read_cached_feature_manifest(output_manifest)
            self.assertEqual(loaded, rows)

    def test_cache_ssl_features_preserves_eval_condition(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            wav_path = root / "utt.wav"
            write_pcm16_wav(wav_path, sine_wave(440, seconds=0.1))
            source_manifest = root / "eval.csv"
            with source_manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["utt_id", "audio_path", "condition", "normalized_text"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "utt_id": "utt-1",
                        "audio_path": str(wav_path),
                        "condition": "snr_10",
                        "normalized_text": "hello world",
                    }
                )

            rows = cache_ssl_features(
                source_manifest=source_manifest,
                output_manifest=root / "features.csv",
                output_dir=root / "features",
                model_id="fake/ssl",
                layer=-1,
                extractor=FakeSslExtractor(),
            )

            self.assertEqual(rows[0].condition, "snr_10")


if __name__ == "__main__":
    unittest.main()
