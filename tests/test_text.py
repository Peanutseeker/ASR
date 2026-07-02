import unittest

from asr_noise_robust.text import Vocabulary, normalize_transcript


class TextTests(unittest.TestCase):
    def test_normalize_transcript_keeps_letters_spaces_and_apostrophe(self):
        text = "HELLO, World! It's 2026."

        self.assertEqual(normalize_transcript(text), "hello world it's")

    def test_vocabulary_round_trips_normalized_text(self):
        vocab = Vocabulary.ctc_english()
        token_ids = vocab.encode("hello world")

        self.assertEqual(vocab.decode_tokens(token_ids), "hello world")
        self.assertEqual(vocab.blank_id, 0)


if __name__ == "__main__":
    unittest.main()
