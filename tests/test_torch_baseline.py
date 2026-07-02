import unittest

import torch

from asr_noise_robust.features import LogMelFeatureExtractor
from asr_noise_robust.torch_models import TinyBiLstmCtc
from asr_noise_robust.torch_train import ctc_loss_for_batch, train_one_batch


class TorchBaselineTests(unittest.TestCase):
    def test_logmel_feature_extractor_returns_time_major_features(self):
        sample_rate = 16000
        waveform = torch.sin(torch.linspace(0, 8 * torch.pi, sample_rate // 2))
        extractor = LogMelFeatureExtractor(sample_rate=sample_rate, n_mels=16)

        features = extractor(waveform)

        self.assertEqual(features.ndim, 2)
        self.assertEqual(features.shape[1], 16)
        self.assertGreater(features.shape[0], 1)
        self.assertTrue(torch.isfinite(features).all())

    def test_logmel_feature_extractor_normalizes_features_by_default(self):
        sample_rate = 16000
        waveform = 0.25 * torch.sin(torch.linspace(0, 12 * torch.pi, sample_rate))
        extractor = LogMelFeatureExtractor(sample_rate=sample_rate, n_mels=16)

        features = extractor(waveform)

        self.assertLess(abs(float(features.mean())), 1e-5)
        self.assertAlmostEqual(float(features.std(unbiased=False)), 1.0, places=5)

    def test_tiny_bilstm_ctc_forward_matches_ctc_shape(self):
        model = TinyBiLstmCtc(input_dim=8, vocab_size=5, hidden_dim=4, num_layers=1)
        features = torch.randn(2, 12, 8)
        lengths = torch.tensor([12, 9], dtype=torch.long)

        log_probs, output_lengths = model(features, lengths)

        self.assertEqual(tuple(log_probs.shape), (12, 2, 5))
        self.assertEqual(output_lengths.tolist(), [12, 9])
        self.assertTrue(torch.isfinite(log_probs).all())

    def test_train_one_batch_returns_finite_loss_and_updates_parameters(self):
        torch.manual_seed(0)
        model = TinyBiLstmCtc(input_dim=8, vocab_size=6, hidden_dim=4, num_layers=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        features = torch.randn(2, 10, 8)
        feature_lengths = torch.tensor([10, 10], dtype=torch.long)
        targets = torch.tensor([1, 2, 3, 1, 3], dtype=torch.long)
        target_lengths = torch.tensor([3, 2], dtype=torch.long)
        before = [parameter.detach().clone() for parameter in model.parameters()]

        loss = train_one_batch(
            model=model,
            optimizer=optimizer,
            features=features,
            feature_lengths=feature_lengths,
            targets=targets,
            target_lengths=target_lengths,
            blank_id=0,
        )
        after = list(model.parameters())

        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertGreater(loss, 0.0)
        self.assertTrue(any(not torch.equal(left, right) for left, right in zip(before, after)))

    def test_ctc_loss_for_batch_is_finite(self):
        model = TinyBiLstmCtc(input_dim=8, vocab_size=6, hidden_dim=4, num_layers=1)
        features = torch.randn(2, 10, 8)
        feature_lengths = torch.tensor([10, 8], dtype=torch.long)
        targets = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        target_lengths = torch.tensor([2, 2], dtype=torch.long)

        loss = ctc_loss_for_batch(
            model=model,
            features=features,
            feature_lengths=feature_lengths,
            targets=targets,
            target_lengths=target_lengths,
            blank_id=0,
        )

        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
