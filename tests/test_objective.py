import math

import pytest
import torch

from brain_npp.objective import (
    build_npp_target,
    build_teacher_correction,
    forward_kl_loss,
    opsd_pointwise_clipped_surrogate,
)
from brain_npp.posterior import build_neural_posterior


def _fixture():
    teachers = torch.log(torch.tensor([[[[0.9, 0.1], [0.2, 0.8]]]], dtype=torch.float64))
    posterior = build_neural_posterior(
        torch.log(torch.tensor([[0.8, 0.2]], dtype=torch.float64)),
        torch.log(torch.tensor([[0.5, 0.5]], dtype=torch.float64)),
    )
    return teachers, posterior


def test_default_is_arithmetic_density_ratio_with_constant_unit_strength():
    teachers, posterior = _fixture()
    anchor = torch.log(torch.tensor([[[0.25, 0.75]]], dtype=torch.float64))
    target = build_npp_target(teachers, anchor, posterior)

    expected_utility = torch.log(torch.tensor([[[0.76 / 0.55, 0.24 / 0.45]]], dtype=torch.float64))
    torch.testing.assert_close(target.utility, expected_utility)
    torch.testing.assert_close(
        target.log_probabilities.exp(),
        torch.tensor([[[19 / 41, 22 / 41]]], dtype=torch.float64),
    )
    torch.testing.assert_close(target.strength, torch.ones(1, dtype=torch.float64))


def test_anchor_equal_reference_mixture_recovers_posterior_predictive():
    teachers, posterior = _fixture()
    correction = build_teacher_correction(teachers, posterior)
    target = build_npp_target(
        teachers,
        correction.reference_predictive_log_probabilities,
        posterior,
    )
    torch.testing.assert_close(
        target.log_probabilities,
        correction.posterior_predictive_log_probabilities,
    )


def test_equal_posterior_and_prior_is_exact_anchor_fixed_point():
    posterior = build_neural_posterior(
        torch.zeros((1, 2), dtype=torch.float64),
        torch.log(torch.tensor([[0.3, 0.7]], dtype=torch.float64)),
    )
    teachers = torch.tensor([[[[2.0, -1.0], [-3.0, 4.0]]]], dtype=torch.float64)
    anchor = torch.tensor([[[0.4, -0.7]]], dtype=torch.float64)
    target = build_npp_target(teachers, anchor, posterior)
    torch.testing.assert_close(target.utility, torch.zeros_like(target.utility))
    assert torch.equal(target.log_probabilities, torch.log_softmax(anchor, dim=-1))


def test_streaming_candidates_are_consumed_once_and_match_dense():
    teachers, posterior = _fixture()
    yielded = []

    def stream():
        for index in range(2):
            yielded.append(index)
            yield teachers[:, :, index]

    actual = build_teacher_correction(stream(), posterior)
    expected = build_teacher_correction(teachers, posterior)
    assert yielded == [0, 1]
    torch.testing.assert_close(actual.utility, expected.utility)


def test_geometric_expected_log_is_named_ablation_not_default():
    teachers, posterior = _fixture()
    arithmetic = build_teacher_correction(teachers, posterior)
    geometric = build_teacher_correction(teachers, posterior, pooling="geometric")
    assert not torch.equal(arithmetic.utility, geometric.utility)


def test_true_forward_kl_matches_oracle_and_gradient():
    target = torch.log(torch.tensor([[[0.25, 0.75]]], dtype=torch.float64))
    student = torch.tensor([[[0.3, -0.4]]], dtype=torch.float64, requires_grad=True)
    stats = forward_kl_loss(target, student)
    stats.loss.backward()
    q = target.exp()
    p = torch.softmax(student.detach(), dim=-1)
    oracle = (q * (target - torch.log_softmax(student.detach(), dim=-1))).sum()
    torch.testing.assert_close(stats.loss, oracle)
    torch.testing.assert_close(student.grad, p - q)
    assert stats.loss.item() >= 0


def test_exact_serialized_anchor_has_exact_zero_gradient():
    student = torch.tensor([[[0.8, -0.0]]], requires_grad=True)
    target = torch.log_softmax(student.detach(), dim=-1)
    stats = forward_kl_loss(target, student)
    stats.loss.backward()
    assert stats.loss.item() == 0.0
    assert torch.equal(student.grad, torch.zeros_like(student))


def test_mask_is_applied_before_nan_arithmetic():
    target = torch.tensor([[[0.0, -torch.inf], [torch.nan, torch.nan]]])
    student = torch.tensor([[[0.0, 0.0], [torch.nan, torch.nan]]], requires_grad=True)
    stats = forward_kl_loss(target, student, torch.tensor([[True, False]]))
    stats.loss.backward()
    assert stats.loss.item() == pytest.approx(math.log(2.0))
    torch.testing.assert_close(student.grad[:, 1], torch.zeros_like(student.grad[:, 1]))


def test_official_pointwise_clip_is_separate_from_forward_kl():
    target = torch.log(torch.tensor([[[0.9, 0.1]]]))
    student = torch.tensor([[[-2.0, 2.0]]])
    full = forward_kl_loss(target, student).loss
    clipped = opsd_pointwise_clipped_surrogate(target, student, token_clip=0.01).loss
    assert full.item() >= 0
    assert clipped.item() != pytest.approx(full.item())


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_ratio_strength_rejects_invalid_values(bad):
    teachers, posterior = _fixture()
    with pytest.raises(ValueError, match="ratio_strength"):
        build_npp_target(teachers, torch.zeros((1, 1, 2), dtype=torch.float64), posterior, ratio_strength=bad)
