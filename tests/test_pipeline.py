from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import unittest

from asr_noise_robust.pipeline import run_toy_pipeline


class PipelineTests(unittest.TestCase):
    def test_run_toy_pipeline_writes_metrics_and_predictions(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)

            outputs = run_toy_pipeline(root)

            self.assertTrue(outputs.train_manifest.exists())
            self.assertTrue(outputs.eval_manifest.exists())
            self.assertTrue(outputs.predictions_csv.exists())
            self.assertTrue(outputs.metrics_csv.exists())

            with outputs.metrics_csv.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            conditions = {row["test_condition"] for row in rows}
            self.assertEqual(conditions, {"clean", "snr_20", "snr_10", "snr_5"})
            self.assertTrue(all(row["feature"] == "toy_oracle" for row in rows))
            self.assertTrue(all(float(row["wer"]) == 0.0 for row in rows))
            self.assertTrue(all(float(row["cer"]) == 0.0 for row in rows))


if __name__ == "__main__":
    unittest.main()

