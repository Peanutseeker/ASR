"""Text normalization and character vocabulary for CTC ASR."""

from __future__ import annotations

from dataclasses import dataclass
import re
import string


def normalize_transcript(text: str) -> str:
    """Normalize English ASR transcripts for character-level CTC training."""
    normalized = text.lower().replace("’", "'")
    normalized = re.sub(r"[^a-z' ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


@dataclass(frozen=True)
class Vocabulary:
    """A small immutable vocabulary for character-level CTC models."""

    symbols: tuple[str, ...]

    @classmethod
    def ctc_english(cls) -> "Vocabulary":
        return cls(("<blank>", *tuple(string.ascii_lowercase), " ", "'"))

    @property
    def blank_id(self) -> int:
        return 0

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: idx for idx, token in enumerate(self.symbols)}

    @property
    def id_to_token(self) -> dict[int, str]:
        return {idx: token for idx, token in enumerate(self.symbols)}

    def encode(self, text: str) -> list[int]:
        normalized = normalize_transcript(text)
        token_to_id = self.token_to_id
        unknown = sorted({char for char in normalized if char not in token_to_id})
        if unknown:
            raise ValueError(f"Text contains characters outside the vocabulary: {unknown}")
        return [token_to_id[char] for char in normalized]

    def decode_tokens(self, token_ids: list[int] | tuple[int, ...], skip_blank: bool = True) -> str:
        id_to_token = self.id_to_token
        chars: list[str] = []
        for token_id in token_ids:
            if skip_blank and token_id == self.blank_id:
                continue
            if token_id not in id_to_token:
                raise ValueError(f"Token id {token_id} is outside the vocabulary")
            token = id_to_token[token_id]
            if token != "<blank>":
                chars.append(token)
        return "".join(chars)

