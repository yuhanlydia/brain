import pytest
import torch

from brain_evidence.recoverability.geometry import (
    fisher_center,
    fisher_inner,
    fisher_norm_sq,
)


def test_fisher_center_has_zero_probability_weighted_mean_per_row() -> None:
    values = torch.tensor([[1.0, 3.0], [2.0, 6.0]])
    probabilities = torch.tensor([[0.25, 0.75], [0.5, 0.5]])

    centered = fisher_center(values, probabilities)

    expected = torch.tensor([[-1.5, 0.5], [-2.0, 2.0]])
    torch.testing.assert_close(centered, expected)
    torch.testing.assert_close(
        (probabilities * centered).sum(dim=-1),
        torch.zeros(2),
        atol=1e-7,
        rtol=0.0,
    )


def test_fisher_center_rejects_different_shapes() -> None:
    values = torch.ones(2, 3)
    probabilities = torch.full((3,), 1.0 / 3.0)

    with pytest.raises(ValueError):
        fisher_center(values, probabilities)


@pytest.mark.parametrize(
    "probabilities",
    [
        torch.tensor([[0.4, 0.5]]),
        torch.tensor([[-0.1, 1.1]]),
        torch.tensor([[float("nan"), float("nan")]]),
    ],
)
def test_fisher_center_rejects_invalid_probability_mass(
    probabilities: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        fisher_center(torch.ones_like(probabilities), probabilities)


def test_fisher_inner_and_norm_reduce_only_the_vocabulary_axis() -> None:
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    right = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    probabilities = torch.tensor([[0.25, 0.75], [0.5, 0.5]])

    inner = fisher_inner(left, right, probabilities)
    norm_sq = fisher_norm_sq(left, probabilities)

    torch.testing.assert_close(inner, torch.tensor([10.25, 26.5]))
    torch.testing.assert_close(norm_sq, torch.tensor([3.25, 12.5]))
    assert inner.shape == (2,)
    assert norm_sq.shape == (2,)


def test_fisher_geometry_outputs_are_detached() -> None:
    values = torch.tensor([[1.0, 2.0]], requires_grad=True)
    probabilities = torch.tensor([[0.25, 0.75]], requires_grad=True)

    centered = fisher_center(values, probabilities)
    inner = fisher_inner(values, values, probabilities)
    norm_sq = fisher_norm_sq(values, probabilities)

    assert not centered.requires_grad
    assert not inner.requires_grad
    assert not norm_sq.requires_grad


def test_fisher_geometry_promotes_float16_arithmetic_to_float32() -> None:
    values = torch.tensor([[1.0, 3.0]], dtype=torch.float16)
    probabilities = torch.tensor([[0.25, 0.75]], dtype=torch.float16)

    centered = fisher_center(values, probabilities)
    norm_sq = fisher_norm_sq(centered, probabilities)

    assert centered.dtype == torch.float32
    assert norm_sq.dtype == torch.float32
    torch.testing.assert_close(centered, torch.tensor([[-1.5, 0.5]]))
    torch.testing.assert_close(norm_sq, torch.tensor([0.75]))


def test_fisher_center_normalizes_accepted_float16_probability_mass() -> None:
    values = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float16)
    probabilities = torch.tensor([[0.3333, 0.3333, 0.3333]], dtype=torch.float16)

    centered = fisher_center(values, probabilities)

    normalized_probabilities = probabilities.float()
    normalized_probabilities /= normalized_probabilities.sum(
        dim=-1,
        keepdim=True,
    )
    torch.testing.assert_close(
        (normalized_probabilities * centered).sum(dim=-1),
        torch.zeros(1),
        atol=5e-7,
        rtol=0.0,
    )


def test_fisher_center_rejects_non_floating_values() -> None:
    with pytest.raises(ValueError, match="floating-point"):
        fisher_center(
            torch.tensor([[1, 2]]),
            torch.tensor([[0.5, 0.5]]),
        )


def test_fisher_center_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        fisher_center(
            torch.tensor([[1.0, float("nan")]]),
            torch.tensor([[0.5, 0.5]]),
        )
