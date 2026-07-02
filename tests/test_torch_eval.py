from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import unittest

from asr_noise_robust.noise_dataset import materialize_noisy_eval_set
from asr_noise_robust.pipeline import run_toy_pipeline
from asr_noise_robust.torch_eval import evaluate_logmel_checkpoint
from asr_noise_robust.torch_trainer import run_batched_logmel_training


class TorchEvalTests(unittest.TestCase):
    def test_evaluate_logmel_checkpoint_writes_predictions_and_metrics(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            toy_outputs = run_toy_pipeline(root / "toy")
            checkpoint = root / "checkpoints" / "model.pt"
            run_batched_logmel_training(
                train_manifest=toy_outputs.train_manifest,
                checkpoint_path=checkpoint,
                history_path=root / "results" / "history.csv",
                summary_path=root / "results" / "summary.json",
                n_mels=12,
                hidden_dim=4,
                max_epochs=1,
                batch_size=2,
            )
            eval_manifest = root / "eval" / "noisy.csv"
            materialize_noisy_eval_set(
                source_manifest=toy_outputs.train_manifest,
                output_manifest=eval_manifest,
                output_audio_dir=root / "eval" / "audio",
            )

            outputs = evaluate_logmel_checkpoint(
                checkpoint_path=checkpoint,
                eval_manifest=eval_manifest,
                predictions_path=root / "results" / "predictions.csv",
                metrics_path=root / "results" / "metrics.csv",
            )

            self.assertTrue(outputs.predictions_path.exists())
            self.assertTrue(outputs.metrics_path.exists())
            with outputs.metrics_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["test_condition"] for row in rows}, {"clean", "snr_20", "snr_10", "snr_5"})


if __name__ == "__main__":
    unittest.main()

