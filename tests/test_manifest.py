from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from asr_noise_robust.manifest import Utterance, read_manifest, summarize_durations, write_manifest


class ManifestTests(unittest.TestCase):
    def test_manifest_round_trips_rows(self):
        rows = [
            Utterance(
                utt_id="utt1",
                audio_path="audio/utt1.flac",
                duration=1.25,
                text="HELLO",
                normalized_text="hello",
                split="train",
            )
        ]
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "manifest.csv"

            write_manifest(path, rows)

            self.assertEqual(read_manifest(path), rows)

    def test_summarize_durations_counts_hours_and_examples(self):
        summary = summarize_durations(
            [
                Utterance("a", "a.flac", 1.0, "A", "a", "train"),
                Utterance("b", "b.flac", 2.0, "B", "b", "train"),
            ]
        )

        self.assertEqual(summary.num_utterances, 2)
        self.assertEqual(summary.total_seconds, 3.0)
        self.assertEqual(round(summary.total_hours, 6), round(3.0 / 3600.0, 6))


if __name__ == "__main__":
    unittest.main()
