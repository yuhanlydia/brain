from __future__ import annotations

import numpy as np
import pytest
import torch

from brain_npp.metrics import (
    brier_score,
    cluster_bootstrap_difference,
    expected_calibration_error,
    neural_dependence_gap,
    topk_coverage,
)


def test_neural_dependence_gap_has_expected_sign() -> None:
    assert neural_dependence_gap([0.9, 0.7], [0.4, 0.5]) == pytest.approx(0.35)
    assert neural_dependence_gap([0.1], [0.4]) == pytest.approx(-0.3)


def test_multiclass_brier_score_uses_all_class_probabilities() -> None:
    probabilities = torch.tensor([[0.8, 0.2], [0.1, 0.9]])

    assert brier_score(probabilities, torch.tensor([0, 0])) == pytest.approx(0.85)


def test_topk_coverage_includes_cutoff_ties() -> None:
    scores = torch.tensor([[0.5, 0.5, 0.0], [0.4, 0.3, 0.3]])

    assert topk_coverage(scores, torch.tensor([1, 2]), k=1) == pytest.approx(0.5)


def test_expected_calibration_error_uses_equal_width_confidence_bins() -> None:
    probabilities = torch.tensor([[0.9, 0.1], [0.6, 0.4]])

    assert expected_calibration_error(probabilities, torch.tensor([0, 1]), bins=2) == pytest.approx(0.25)


@pytest.mark.parametrize("metric", [brier_score, expected_calibration_error])
def test_probability_metrics_reject_values_outside_the_simplex(metric) -> None:
    with pytest.raises(ValueError, match="probabilities"):
        metric(torch.tensor([[2.0, -1.0]]), torch.tensor([0]))


@pytest.mark.parametrize("metric", [brier_score, expected_calibration_error])
def test_probability_metrics_reject_rows_that_do_not_sum_to_one(metric) -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        metric(torch.tensor([[0.8, 0.8]]), torch.tensor([0]))


def test_topk_coverage_accepts_non_probability_scores() -> None:
    assert topk_coverage(torch.tensor([[2.0, -1.0]]), torch.tensor([0]), k=1) == pytest.approx(1.0)


def test_cluster_bootstrap_is_seeded_and_resamples_whole_image_clusters() -> None:
    correct = np.array([0.0, 0.0, 2.0])
    control = np.zeros_like(correct)
    image_ids = ["image-a", "image-a", "image-b"]

    point, draws = cluster_bootstrap_difference(
        correct, control, image_ids, seed=5, samples=30
    )
    repeated_point, repeated_draws = cluster_bootstrap_difference(
        correct, control, image_ids, seed=5, samples=30
    )

    assert point == pytest.approx(np.mean(correct - control))
    assert repeated_point == point
    assert np.array_equal(draws, repeated_draws)
    assert set(np.unique(draws)).issubset({0.0, 2.0 / 3.0, 2.0})
