from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import unittest

from asr_noise_robust.noise_dataset import materialize_noisy_eval_set
from asr_noise_robust.pipeline import run_toy_pipeline


class NoiseDatasetTests(unittest.TestCase):
    def test_materialize_noisy_eval_set_writes_condition_manifest_and_audio(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            toy_outputs = run_toy_pipeline(root / "toy")
            output_manifest = root / "eval" / "conditions.csv"
            output_audio_dir = root / "eval" / "audio"

            count = materialize_noisy_eval_set(
                source_manifest=toy_outputs.train_manifest,
                output_manifest=output_manifest,
                output_audio_dir=output_audio_dir,
                snr_values=(20, 10, 5),
            )

            self.assertEqual(count, 16)
            with output_manifest.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["condition"] for row in rows}, {"clean", "snr_20", "snr_10", "snr_5"})
            noisy_paths = [Path(row["audio_path"]) for row in rows if row["condition"] != "clean"]
            self.assertTrue(noisy_paths)
            self.assertTrue(all(path.exists() for path in noisy_paths))


if __name__ == "__main__":
    unittest.main()

