import unittest

from asr_noise_robust.ctc import greedy_ctc_decode


class CtcTests(unittest.TestCase):
    def test_greedy_ctc_decode_collapses_repeats_and_removes_blanks(self):
        self.assertEqual(
            greedy_ctc_decode([0, 1, 1, 0, 2, 2, 3, 0, 3], blank_id=0),
            [1, 2, 3, 3],
        )


if __name__ == "__main__":
    unittest.main()
