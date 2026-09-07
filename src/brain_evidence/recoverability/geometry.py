"""Fisher geometry over the final, vocabulary dimension."""

from __future__ import annotations

import torch
from torch import Tensor


def _fisher_work_dtype(*tensors: Tensor) -> torch.dtype:
    """Return a common floating dtype with a float32 precision floor."""
    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        dtype = torch.promote_types(dtype, tensor.dtype)
    if dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return dtype


def _validate_probability_geometry(
    probabilities: Tensor,
    *values: Tensor,
) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Validate and detach tensors participating in Fisher geometry."""
    if probabilities.ndim == 0 or probabilities.shape[-1] == 0:
        raise ValueError("probabilities must have a non-empty vocabulary axis")
    if not probabilities.is_floating_point():
        raise ValueError("probabilities must be a floating-point tensor")

    expected_shape = probabilities.shape
    expected_device = probabilities.device
    for value in values:
        if value.shape != expected_shape:
            raise ValueError(
                "values and probabilities must have exactly the same shape"
            )
        if value.device != expected_device:
            raise ValueError("values and probabilities must be on the same device")
        if not value.is_floating_point():
            raise ValueError("values must be floating-point tensors")
        if not bool(torch.isfinite(value.detach()).all()):
            raise ValueError("values must be finite")

    work_dtype = _fisher_work_dtype(probabilities, *values)
    detached_probabilities = probabilities.detach().to(dtype=work_dtype)
    if not bool(torch.isfinite(detached_probabilities).all()):
        raise ValueError("probabilities must be finite")
    if bool((detached_probabilities < 0).any()):
        raise ValueError("probabilities must be non-negative")

    mass = detached_probabilities.sum(dim=-1)
    probability_tolerance = max(
        1e-6,
        2.0 * torch.finfo(probabilities.dtype).eps,
    )
    if not torch.allclose(
        mass,
        torch.ones_like(mass),
        atol=probability_tolerance,
        rtol=1e-5,
    ):
        raise ValueError("probabilities must sum to one on the vocabulary axis")
    detached_probabilities = detached_probabilities / mass.unsqueeze(-1)

    return detached_probabilities, tuple(
        value.detach().to(dtype=work_dtype) for value in values
    )


def fisher_center(values: Tensor, probabilities: Tensor) -> Tensor:
    """Center ``values`` under ``probabilities`` along the vocabulary axis."""
    detached_probabilities, (detached_values,) = _validate_probability_geometry(
        probabilities,
        values,
    )
    weighted_mean = (detached_probabilities * detached_values).sum(
        dim=-1,
        keepdim=True,
    )
    return (detached_values - weighted_mean).detach()


def fisher_inner(left: Tensor, right: Tensor, probabilities: Tensor) -> Tensor:
    """Return the Fisher-weighted inner product on the final axis."""
    detached_probabilities, (detached_left, detached_right) = (
        _validate_probability_geometry(probabilities, left, right)
    )
    return (
        (detached_probabilities * detached_left * detached_right).sum(dim=-1).detach()
    )


def fisher_norm_sq(values: Tensor, probabilities: Tensor) -> Tensor:
    """Return the squared Fisher norm on the final axis."""
    return fisher_inner(values, values, probabilities)
