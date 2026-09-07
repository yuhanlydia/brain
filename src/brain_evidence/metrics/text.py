"""Text-generation metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class WordErrorRate:
    """Word-level edit counts and their reference-normalized error rate."""

    substitutions: int
    deletions: int
    insertions: int
    reference_words: int
    wer: float

    @property
    def errors(self) -> int:
        """Return the total number of word edits."""

        return self.substitutions + self.deletions + self.insertions


def _as_tokens(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(value.split())
    return tuple(value)


def word_error_rate(
    reference: str | Sequence[str], hypothesis: str | Sequence[str]
) -> WordErrorRate:
    """Compute word-level Levenshtein counts and WER.

    Empty references use a denominator of one: an empty/empty pair has zero
    error, while every hypothesized word counts as one insertion.
    """

    reference_tokens = _as_tokens(reference)
    hypothesis_tokens = _as_tokens(hypothesis)
    n_reference = len(reference_tokens)

    prefix_length = 0
    while (
        prefix_length < len(reference_tokens)
        and prefix_length < len(hypothesis_tokens)
        and reference_tokens[prefix_length] == hypothesis_tokens[prefix_length]
    ):
        prefix_length += 1
    reference_middle = reference_tokens[prefix_length:]
    hypothesis_middle = hypothesis_tokens[prefix_length:]

    suffix_length = 0
    while (
        suffix_length < len(reference_middle)
        and suffix_length < len(hypothesis_middle)
        and reference_middle[-1 - suffix_length]
        == hypothesis_middle[-1 - suffix_length]
    ):
        suffix_length += 1
    if suffix_length:
        reference_middle = reference_middle[:-suffix_length]
        hypothesis_middle = hypothesis_middle[:-suffix_length]

    n_middle_reference = len(reference_middle)
    n_middle_hypothesis = len(hypothesis_middle)
    distances = [
        [0 for _ in range(n_middle_hypothesis + 1)]
        for _ in range(n_middle_reference + 1)
    ]
    for reference_index in range(n_middle_reference + 1):
        distances[reference_index][0] = reference_index
    for hypothesis_index in range(n_middle_hypothesis + 1):
        distances[0][hypothesis_index] = hypothesis_index

    for reference_index in range(1, n_middle_reference + 1):
        for hypothesis_index in range(1, n_middle_hypothesis + 1):
            substitution_cost = (
                reference_middle[reference_index - 1]
                != hypothesis_middle[hypothesis_index - 1]
            )
            distances[reference_index][hypothesis_index] = min(
                distances[reference_index - 1][hypothesis_index] + 1,
                distances[reference_index][hypothesis_index - 1] + 1,
                distances[reference_index - 1][hypothesis_index - 1]
                + substitution_cost,
            )

    # Recover the same optimal path selected by RapidFuzz/JiWER. Its bit-parallel
    # VP/VN recovery prefers a deletion only when moving left lowers the score;
    # the second comparison below is the equivalent insertion condition.
    reference_index = n_middle_reference
    hypothesis_index = n_middle_hypothesis
    substitutions = 0
    deletions = 0
    insertions = 0
    while reference_index and hypothesis_index:
        if (
            distances[reference_index][hypothesis_index]
            == distances[reference_index - 1][hypothesis_index] + 1
        ):
            deletions += 1
            reference_index -= 1
            continue

        hypothesis_index -= 1
        if (
            hypothesis_index
            and distances[reference_index][hypothesis_index]
            == distances[reference_index - 1][hypothesis_index] - 1
        ):
            insertions += 1
            continue

        reference_index -= 1
        if reference_middle[reference_index] != hypothesis_middle[hypothesis_index]:
            substitutions += 1

    deletions += reference_index
    insertions += hypothesis_index
    errors = substitutions + deletions + insertions
    return WordErrorRate(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_words=n_reference,
        wer=errors / max(n_reference, 1),
    )


__all__ = ["WordErrorRate", "word_error_rate"]
