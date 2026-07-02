import unittest

from asr_noise_robust.minilibrispeech import MINILIBRISPEECH_PARTS, minilibrispeech_url


class MiniLibriSpeechTests(unittest.TestCase):
    def test_minilibrispeech_url_points_to_openslr_resource_31(self):
        self.assertEqual(
            minilibrispeech_url("dev-clean-2"),
            "https://www.openslr.org/resources/31/dev-clean-2.tar.gz",
        )
        self.assertEqual(
            minilibrispeech_url("train-clean-5"),
            "https://www.openslr.org/resources/31/train-clean-5.tar.gz",
        )

    def test_minilibrispeech_url_rejects_unknown_part(self):
        with self.assertRaises(ValueError):
            minilibrispeech_url("train-clean-100")

    def test_known_parts_are_ordered_small_to_large_for_smoke_runs(self):
        self.assertEqual(MINILIBRISPEECH_PARTS, ("dev-clean-2", "train-clean-5"))


if __name__ == "__main__":
    unittest.main()

