"""Legacy scalar evidence gate retained as a matched-compute ablation."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from brain_evidence.recoverability.geometry import (
    fisher_center,
    fisher_inner,
    fisher_norm_sq,
)


def legacy_dual_cosine_gate(
    teacher_contrast: Tensor,
    brain_contrast: Tensor,
    probabilities: Tensor,
    *,
    reliability: Tensor | float | None = None,
    eps: float = 1e-12,
) -> Tensor:
    """Return the historical reliability-weighted positive Fisher cosine.

    Unlike interaction projection, this legacy gate scales the complete
    privileged correction. It is therefore exposed only as an ablation.
    """

    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and greater than zero")
    _validate_finite(teacher_contrast, brain_contrast)

    validated_teacher = fisher_center(teacher_contrast, probabilities)
    validated_brain = fisher_center(brain_contrast, probabilities)
    work_dtype = _gate_work_dtype(
        validated_teacher,
        validated_brain,
        probabilities,
    )
    work_probabilities = _to_finite_work_tensor(
        probabilities,
        dtype=work_dtype,
        name="probabilities",
    )
    work_probabilities = work_probabilities / work_probabilities.sum(
        dim=-1,
        keepdim=True,
    )
    centered_teacher = fisher_center(
        _to_finite_work_tensor(
            validated_teacher,
            dtype=work_dtype,
            name="teacher_contrast",
        ),
        work_probabilities,
    )
    centered_brain = fisher_center(
        _to_finite_work_tensor(
            validated_brain,
            dtype=work_dtype,
            name="brain_contrast",
        ),
        work_probabilities,
    )
    positive_cosine = _positive_fisher_cosine(
        centered_teacher,
        centered_brain,
        work_probabilities,
        eps=eps,
    )
    calibrated_reliability = _broadcast_reliability(
        reliability,
        positive_cosine,
    )
    return (calibrated_reliability * positive_cosine).detach()


def dual_cosine_gate(
    teacher_contrast: Tensor,
    brain_contrast: Tensor,
    probabilities: Tensor,
    *,
    reliability: Tensor | float | None = None,
    eps: float = 1e-12,
) -> Tensor:
    """Short alias for :func:`legacy_dual_cosine_gate`."""

    return legacy_dual_cosine_gate(
        teacher_contrast,
        brain_contrast,
        probabilities,
        reliability=reliability,
        eps=eps,
    )


def _positive_fisher_cosine(
    left: Tensor,
    right: Tensor,
    probabilities: Tensor,
    *,
    eps: float,
) -> Tensor:
    numerator = fisher_inner(left, right, probabilities)
    left_norm_sq = fisher_norm_sq(left, probabilities)
    right_norm_sq = fisher_norm_sq(right, probabilities)
    left_norm = torch.sqrt(left_norm_sq.clamp_min(0.0))
    right_norm = torch.sqrt(right_norm_sq.clamp_min(0.0))
    effective_eps = max(eps, torch.finfo(numerator.dtype).eps)
    denominator = left_norm.clamp_min(effective_eps) * right_norm.clamp_min(
        effective_eps
    )
    valid = (left_norm > effective_eps) & (right_norm > effective_eps)
    cosine = numerator / denominator
    return torch.where(
        valid,
        cosine.clamp(min=0.0, max=1.0),
        torch.zeros_like(cosine),
    ).detach()


def _validate_finite(*values: Tensor) -> None:
    for value in values:
        if not isinstance(value, Tensor):
            raise TypeError("gate inputs must be torch.Tensor values")
        if not value.is_floating_point():
            raise TypeError("gate inputs must have floating dtypes")
        if not bool(torch.isfinite(value.detach()).all()):
            raise ValueError("gate inputs must be finite")


def _broadcast_reliability(
    reliability: Tensor | float | None,
    target: Tensor,
) -> Tensor:
    if reliability is None:
        return torch.ones_like(target)
    if isinstance(reliability, bool):
        raise TypeError("reliability must be a tensor or numeric scalar")
    if isinstance(reliability, Tensor):
        weight = reliability.detach().to(device=target.device, dtype=target.dtype)
    else:
        try:
            weight = torch.as_tensor(
                reliability,
                device=target.device,
                dtype=target.dtype,
            )
        except (TypeError, ValueError) as error:
            raise TypeError("reliability must be a tensor or numeric scalar") from error
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("reliability must be finite")
    if not bool(((weight >= 0) & (weight <= 1)).all()):
        raise ValueError("reliability must lie in [0, 1]")
    try:
        _, broadcast_weight = torch.broadcast_tensors(target, weight)
    except RuntimeError as error:
        raise ValueError("reliability must broadcast to the gate shape") from error
    return broadcast_weight.detach()


def _gate_work_dtype(*values: Tensor) -> torch.dtype:
    """Promote low-precision geometry while preserving wider input dtypes."""

    dtype = values[0].dtype
    for value in values[1:]:
        dtype = torch.promote_types(dtype, value.dtype)
    if dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return dtype


def _to_finite_work_tensor(
    value: Tensor,
    *,
    dtype: torch.dtype,
    name: str,
) -> Tensor:
    converted = value.detach().to(dtype=dtype)
    if not bool(torch.isfinite(converted).all()):
        raise ValueError(f"{name} must be finite after dtype conversion")
    return converted
