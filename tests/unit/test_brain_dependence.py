import pytest

from brain_evidence.metrics.brain_dependence import brain_dependence


def test_brain_dependence_reports_gap_and_difference_against_ce() -> None:
    result = brain_dependence(
        real_score=0.72,
        wrong_brain_score=0.43,
        ce_real_score=0.63,
        ce_wrong_brain_score=0.51,
    )

    assert result.brain_gap == pytest.approx(0.29)
    assert result.ce_brain_gap == pytest.approx(0.12)
    assert result.difference_in_gap == pytest.approx(0.17)


def test_brain_dependence_keeps_signed_real_minus_wrong_direction() -> None:
    result = brain_dependence(
        real_score=0.4,
        wrong_brain_score=0.5,
        ce_real_score=0.6,
        ce_wrong_brain_score=0.5,
    )

    assert result.brain_gap == pytest.approx(-0.1)
    assert result.difference_in_gap == pytest.approx(-0.2)
