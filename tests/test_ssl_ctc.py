import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from asr_noise_robust.ssl_eval import evaluate_cached_ssl_checkpoint
from asr_noise_robust.ssl_features import CachedFeatureRow, write_cached_feature_manifest
from asr_noise_robust.ssl_trainer import run_cached_ssl_ctc_training


def _write_feature(path: Path, frames: int, dim: int, offset: float) -> None:
    values = torch.arange(frames * dim, dtype=torch.float32).view(frames, dim)
    torch.save((values + offset) / 100.0, path)


class SslCtcTests(unittest.TestCase):
    def test_cached_ssl_ctc_training_and_eval_write_outputs(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            feature_dir = root / "features"
            feature_dir.mkdir()
            train_rows = []
            eval_rows = []
            for index, text in enumerate(["alpha", "bravo", "charlie", "delta"]):
                feature_path = feature_dir / f"train_{index}.pt"
                _write_feature(feature_path, frames=18, dim=6, offset=float(index))
                train_rows.append(
                    CachedFeatureRow(
                        utt_id=f"utt-{index}",
                        condition="clean",
                        normalized_text=text,
                        feature_path=feature_path,
                        num_frames=18,
                        feature_dim=6,
                        model_id="fake/ssl",
                        layer=9,
                    )
                )
            for index, condition in enumerate(["clean", "snr_20"]):
                feature_path = feature_dir / f"eval_{index}.pt"
                _write_feature(feature_path, frames=18, dim=6, offset=float(index + 10))
                eval_rows.append(
                    CachedFeatureRow(
                        utt_id=f"eval-{index}",
                        condition=condition,
                        normalized_text="alpha",
                        feature_path=feature_path,
                        num_frames=18,
                        feature_dim=6,
                        model_id="fake/ssl",
                        layer=9,
                    )
                )
            train_manifest = root / "train_features.csv"
            eval_manifest = root / "eval_features.csv"
            write_cached_feature_manifest(train_manifest, train_rows)
            write_cached_feature_manifest(eval_manifest, eval_rows)

            train_outputs = run_cached_ssl_ctc_training(
                train_feature_manifest=train_manifest,
                checkpoint_path=root / "checkpoints" / "ssl_ctc.pt",
                history_path=root / "results" / "history.csv",
                summary_path=root / "results" / "summary.json",
                hidden_dim=4,
                max_epochs=2,
                batch_size=2,
                learning_rate=0.01,
            )

            self.assertTrue(train_outputs.checkpoint_path.exists())
            self.assertTrue(train_outputs.history_path.exists())
            self.assertTrue(train_outputs.summary_path.exists())
            self.assertTrue(torch.isfinite(torch.tensor(train_outputs.final_loss)))
            checkpoint = torch.load(
                train_outputs.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(checkpoint["input_dim"], 6)
            self.assertEqual(checkpoint["feature_model_id"], "fake/ssl")
            self.assertEqual(checkpoint["feature_layer"], 9)

            eval_outputs = evaluate_cached_ssl_checkpoint(
                checkpoint_path=train_outputs.checkpoint_path,
                eval_feature_manifest=eval_manifest,
                predictions_path=root / "results" / "predictions.csv",
                metrics_path=root / "results" / "metrics.csv",
            )

            self.assertTrue(eval_outputs.predictions_path.exists())
            self.assertTrue(eval_outputs.metrics_path.exists())
            with eval_outputs.metrics_path.open("r", newline="", encoding="utf-8") as handle:
                metric_rows = list(csv.DictReader(handle))
            self.assertEqual([row["test_condition"] for row in metric_rows], ["clean", "snr_20"])
            self.assertTrue(all(row["feature"] == "frozen_ssl" for row in metric_rows))


if __name__ == "__main__":
    unittest.main()
