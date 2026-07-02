from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import json
import math
import unittest

import torch

from asr_noise_robust.pipeline import run_toy_pipeline
from asr_noise_robust.torch_trainer import run_batched_logmel_training


class TorchTrainerTests(unittest.TestCase):
    def test_batched_logmel_training_writes_checkpoint_history_and_summary(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            toy_outputs = run_toy_pipeline(root)

            outputs = run_batched_logmel_training(
                train_manifest=toy_outputs.train_manifest,
                checkpoint_path=root / "checkpoints" / "model.pt",
                history_path=root / "results" / "history.csv",
                summary_path=root / "results" / "summary.json",
                n_mels=12,
                hidden_dim=4,
                max_epochs=2,
                batch_size=2,
                max_train_examples=4,
            )

            self.assertTrue(outputs.checkpoint_path.exists())
            self.assertTrue(outputs.history_path.exists())
            self.assertTrue(outputs.summary_path.exists())
            self.assertTrue(math.isfinite(outputs.final_loss))
            with outputs.history_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["epoch"] for row in rows}, {"1", "2"})

            summary = json.loads(outputs.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["num_train_examples"], 4)
            self.assertEqual(summary["batch_size"], 2)
            self.assertEqual(summary["max_epochs"], 2)
            self.assertEqual(summary["feature_dim"], 12)
            self.assertTrue(summary["feature_normalize"])

            checkpoint = torch.load(outputs.checkpoint_path, map_location="cpu", weights_only=False)
            self.assertTrue(checkpoint["feature_normalize"])


if __name__ == "__main__":
    unittest.main()
