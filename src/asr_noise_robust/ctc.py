"""Small CTC decoding helpers used by training and evaluation scripts."""

from __future__ import annotations


def greedy_ctc_decode(token_ids: list[int] | tuple[int, ...], blank_id: int = 0) -> list[int]:
    """Collapse repeated token ids and remove CTC blanks."""
    decoded: list[int] = []
    previous: int | None = None
    for token_id in token_ids:
        if token_id != previous and token_id != blank_id:
            decoded.append(token_id)
        previous = token_id
    return decoded


def greedy_ctc_decode_batch(
    batch_token_ids: list[list[int]] | tuple[tuple[int, ...], ...], blank_id: int = 0
) -> list[list[int]]:
    return [greedy_ctc_decode(token_ids, blank_id=blank_id) for token_ids in batch_token_ids]

