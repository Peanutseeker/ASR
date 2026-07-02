import unittest

from asr_noise_robust.noise import measured_snr_db, mix_at_snr_db


class NoiseTests(unittest.TestCase):
    def test_mix_at_snr_db_matches_requested_snr_for_repeated_noise(self):
        clean = [1.0, -1.0, 1.0, -1.0]
        noise = [0.5, -0.5]
        mixed = mix_at_snr_db(clean, noise, snr_db=0.0)

        self.assertEqual(round(measured_snr_db(clean, mixed), 6), 0.0)


if __name__ == "__main__":
    unittest.main()
