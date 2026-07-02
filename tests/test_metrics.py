import unittest

from asr_noise_robust.metrics import cer, edit_counts, wer


class MetricsTests(unittest.TestCase):
    def test_wer_computes_word_error_rate(self):
        self.assertEqual(wer("the quick fox", "the slow fox"), 1 / 3)

    def test_cer_computes_character_error_rate(self):
        self.assertEqual(cer("abc", "axc"), 1 / 3)

    def test_edit_counts_reports_substitution_deletion_and_insertion(self):
        counts = edit_counts("a b c".split(), "a x c d".split())

        self.assertEqual(counts.substitutions, 1)
        self.assertEqual(counts.deletions, 0)
        self.assertEqual(counts.insertions, 1)


if __name__ == "__main__":
    unittest.main()
