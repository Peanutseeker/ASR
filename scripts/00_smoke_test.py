#!/usr/bin/env python3
"""Run a dependency-light smoke test for the project utilities."""

from asr_noise_robust.ctc import greedy_ctc_decode
from asr_noise_robust.metrics import cer, wer
from asr_noise_robust.noise import measured_snr_db, mix_at_snr_db
from asr_noise_robust.text import Vocabulary, normalize_transcript


def main() -> None:
    vocab = Vocabulary.ctc_english()
    text = normalize_transcript("HELLO, World!")
    ctc_path: list[int] = [vocab.blank_id]
    for token_id in vocab.encode(text):
        ctc_path.extend([token_id, vocab.blank_id])
    decoded = vocab.decode_tokens(greedy_ctc_decode(ctc_path, blank_id=vocab.blank_id))
    mixed = mix_at_snr_db([1.0, -1.0, 1.0, -1.0], [0.5, -0.5], snr_db=0.0)

    print(f"normalized={text}")
    print(f"decoded={decoded}")
    print(f"wer={wer('hello world', decoded):.6f}")
    print(f"cer={cer('hello world', decoded):.6f}")
    print(f"snr={measured_snr_db([1.0, -1.0, 1.0, -1.0], mixed):.6f}")


if __name__ == "__main__":
    main()
