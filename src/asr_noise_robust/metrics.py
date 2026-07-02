"""WER, CER, and edit-count utilities for ASR experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EditCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_length: int

    @property
    def distance(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        if self.reference_length == 0:
            raise ValueError("Reference length must be greater than zero")
        return self.distance / self.reference_length


def _better(left: EditCounts, right: EditCounts) -> EditCounts:
    left_key = (left.distance, left.substitutions, left.deletions, left.insertions)
    right_key = (right.distance, right.substitutions, right.deletions, right.insertions)
    return left if left_key <= right_key else right


def edit_counts(reference: list[str] | tuple[str, ...], hypothesis: list[str] | tuple[str, ...]) -> EditCounts:
    """Return Levenshtein edit counts between token sequences."""
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    dp: list[list[EditCounts]] = [
        [EditCounts(0, 0, 0, len(reference)) for _ in range(cols)] for _ in range(rows)
    ]

    for i in range(1, rows):
        previous = dp[i - 1][0]
        dp[i][0] = EditCounts(
            previous.substitutions,
            previous.deletions + 1,
            previous.insertions,
            len(reference),
        )
    for j in range(1, cols):
        previous = dp[0][j - 1]
        dp[0][j] = EditCounts(
            previous.substitutions,
            previous.deletions,
            previous.insertions + 1,
            len(reference),
        )

    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                match = dp[i - 1][j - 1]
            else:
                previous = dp[i - 1][j - 1]
                match = EditCounts(
                    previous.substitutions + 1,
                    previous.deletions,
                    previous.insertions,
                    len(reference),
                )

            deletion_prev = dp[i - 1][j]
            deletion = EditCounts(
                deletion_prev.substitutions,
                deletion_prev.deletions + 1,
                deletion_prev.insertions,
                len(reference),
            )

            insertion_prev = dp[i][j - 1]
            insertion = EditCounts(
                insertion_prev.substitutions,
                insertion_prev.deletions,
                insertion_prev.insertions + 1,
                len(reference),
            )

            dp[i][j] = _better(_better(match, deletion), insertion)

    return dp[-1][-1]


def wer(reference: str, hypothesis: str) -> float:
    ref_words = [word for word in reference.split(" ") if word]
    hyp_words = [word for word in hypothesis.split(" ") if word]
    return edit_counts(ref_words, hyp_words).rate


def cer(reference: str, hypothesis: str) -> float:
    return edit_counts(list(reference), list(hypothesis)).rate

