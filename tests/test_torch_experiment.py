from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import math
import unittest

import torch

from asr_noise_robust.pipeline import run_toy_pipeline
from asr_noise_robust.torch_experiment import run_torch_logmel_experiment


class TorchExperimentTests(unittest.TestCase):
    def test_torch_logmel_experiment_writes_predictions_and_metrics(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            toy_outputs = run_toy_pipeline(root)

            outputs = run_torch_logmel_experiment(
                train_manifest=toy_outputs.train_manifest,
                eval_manifest=toy_outputs.eval_manifest,
                checkpoint_path=root / "checkpoints" / "torch_logmel_experiment.pt",
                predictions_path=root / "results" / "predictions" / "torch_logmel_predictions.csv",
                metrics_path=root / "results" / "torch_logmel_metrics.csv",
                n_mels=12,
                hidden_dim=4,
                max_epochs=2,
            )

            self.assertTrue(outputs.checkpoint_path.exists())
            self.assertTrue(outputs.predictions_path.exists())
            self.assertTrue(outputs.metrics_path.exists())
            self.assertTrue(math.isfinite(outputs.final_loss))
            checkpoint = torch.load(outputs.checkpoint_path, map_location="cpu", weights_only=False)
            self.assertTrue(checkpoint["feature_normalize"])

            with outputs.predictions_path.open("r", newline="", encoding="utf-8") as handle:
                prediction_rows = list(csv.DictReader(handle))
            self.assertEqual(len(prediction_rows), 16)

            with outputs.metrics_path.open("r", newline="", encoding="utf-8") as handle:
                metric_rows = list(csv.DictReader(handle))
            self.assertEqual({row["test_condition"] for row in metric_rows}, {"clean", "snr_20", "snr_10", "snr_5"})
            self.assertTrue(all(row["feature"] == "logmel" for row in metric_rows))
            self.assertTrue(all(row["model"] == "tiny_bilstm_ctc" for row in metric_rows))


if __name__ == "__main__":
    unittest.main()
