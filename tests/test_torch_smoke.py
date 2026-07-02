from pathlib import Path
from tempfile import TemporaryDirectory
import json
import math
import unittest

import torch

from asr_noise_robust.pipeline import run_toy_pipeline
from asr_noise_robust.torch_smoke import run_torch_logmel_smoke


class TorchSmokeTests(unittest.TestCase):
    def test_torch_logmel_smoke_trains_one_batch_and_writes_outputs(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            toy_outputs = run_toy_pipeline(root)

            outputs = run_torch_logmel_smoke(
                train_manifest=toy_outputs.train_manifest,
                checkpoint_path=root / "checkpoints" / "torch_logmel_smoke.pt",
                summary_path=root / "results" / "torch_logmel_smoke.json",
                n_mels=12,
                hidden_dim=4,
            )

            self.assertTrue(outputs.checkpoint_path.exists())
            self.assertTrue(outputs.summary_path.exists())
            self.assertGreater(outputs.loss, 0.0)
            self.assertTrue(math.isfinite(outputs.loss))

            summary = json.loads(outputs.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["num_examples"], 4)
            self.assertEqual(summary["feature_dim"], 12)
            self.assertGreater(summary["max_frames"], 1)
            self.assertGreater(summary["loss"], 0.0)
            self.assertTrue(summary["feature_normalize"])

            checkpoint = torch.load(outputs.checkpoint_path, map_location="cpu", weights_only=False)
            self.assertTrue(checkpoint["feature_normalize"])


if __name__ == "__main__":
    unittest.main()
