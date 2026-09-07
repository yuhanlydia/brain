import pytest

from brain_evidence.metrics.text import word_error_rate


def test_word_error_rate_exposes_substitutions_and_deletions() -> None:
    result = word_error_rate("the quick brown fox", "the slow brown")

    assert result.substitutions == 1
    assert result.deletions == 1
    assert result.insertions == 0
    assert result.reference_words == 4
    assert result.wer == pytest.approx(0.5)


def test_word_error_rate_exposes_insertions() -> None:
    result = word_error_rate("read this", "please read this now")

    assert result.substitutions == 0
    assert result.deletions == 0
    assert result.insertions == 2
    assert result.reference_words == 2
    assert result.wer == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("reference", "hypothesis", "expected_counts", "expected_wer"),
    [
        ("a b", "b a", (0, 1, 1), 1.0),
        ("a a b", "a b a", (0, 1, 1), 2.0 / 3.0),
        ("one two one", "one one two", (0, 1, 1), 2.0 / 3.0),
        ("a b c", "b c c", (2, 0, 0), 2.0 / 3.0),
    ],
)
def test_word_error_rate_matches_rapidfuzz_on_ambiguous_alignments(
    reference: str,
    hypothesis: str,
    expected_counts: tuple[int, int, int],
    expected_wer: float,
) -> None:
    result = word_error_rate(reference, hypothesis)

    assert (result.substitutions, result.deletions, result.insertions) == (
        expected_counts
    )
    assert result.wer == pytest.approx(expected_wer)


def test_word_error_rate_handles_empty_reference_and_hypothesis() -> None:
    inserted = word_error_rate("", "two words")
    empty = word_error_rate("", "")
    deleted = word_error_rate("two words", "")

    assert (inserted.substitutions, inserted.deletions, inserted.insertions) == (
        0,
        0,
        2,
    )
    assert inserted.reference_words == 0
    assert inserted.wer == pytest.approx(2.0)
    assert empty.wer == pytest.approx(0.0)
    assert deleted.deletions == 2
    assert deleted.wer == pytest.approx(1.0)
