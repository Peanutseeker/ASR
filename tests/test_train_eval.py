from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import json
import unittest

from asr_noise_robust.pipeline import run_toy_pipeline
from asr_noise_robust.train_eval import run_manifest_train_eval


class TrainEvalTests(unittest.TestCase):
    def test_manifest_train_eval_writes_model_predictions_and_metrics(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            toy_outputs = run_toy_pipeline(root)
            model_path = root / "checkpoints" / "toy_memorizer.json"
            predictions_path = root / "results" / "predictions" / "toy_memorizer_predictions.csv"
            metrics_path = root / "results" / "toy_memorizer_metrics.csv"

            outputs = run_manifest_train_eval(
                train_manifest=toy_outputs.train_manifest,
                eval_manifest=toy_outputs.eval_manifest,
                model_path=model_path,
                predictions_path=predictions_path,
                metrics_path=metrics_path,
            )

            self.assertEqual(outputs.model_path, model_path)
            self.assertTrue(outputs.model_path.exists())
            self.assertTrue(outputs.predictions_path.exists())
            self.assertTrue(outputs.metrics_path.exists())

            model = json.loads(outputs.model_path.read_text(encoding="utf-8"))
            self.assertEqual(model["model_type"], "manifest_memorizer")
            self.assertEqual(len(model["utterances"]), 4)

            with outputs.metrics_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual({row["test_condition"] for row in rows}, {"clean", "snr_20", "snr_10", "snr_5"})
            self.assertTrue(all(row["model"] == "manifest_memorizer" for row in rows))
            self.assertTrue(all(float(row["wer"]) == 0.0 for row in rows))
            self.assertTrue(all(float(row["cer"]) == 0.0 for row in rows))


if __name__ == "__main__":
    unittest.main()

