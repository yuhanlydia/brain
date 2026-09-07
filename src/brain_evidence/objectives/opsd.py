"""On-policy self-distillation objectives."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def forward_kl(
    student_log_probs: Tensor,
    target_probs: Tensor,
    *,
    token_mask: Tensor | None = None,
    token_weights: Tensor | None = None,
    pointwise_clip: float | None = None,
) -> Tensor:
    """Return mean ``KL(target || student)`` over selected token positions.

    The final axis is vocabulary. Masks and weights address all preceding
    token axes. Target probabilities and weights are detached so gradients
    flow only through the student distribution. Only rows with positive
    effective weight must contain normalized distributions; inactive padding
    may be non-finite. Negative infinity is allowed only off target support.
    Half-precision inputs are evaluated in float32, preserving float64 inputs.
    Rows within source-precision normalization tolerance are renormalized in
    that working dtype, so the loss compares probability distributions despite
    input rounding. Student gradients include this normalization.
    """

    for name, tensor in (
        ("student_log_probs", student_log_probs),
        ("target_probs", target_probs),
    ):
        if not isinstance(tensor, Tensor) or not tensor.is_floating_point():
            raise TypeError(f"{name} must be a real floating-point tensor")
    if student_log_probs.shape != target_probs.shape:
        raise ValueError("student_log_probs and target_probs must have equal shape")
    if student_log_probs.ndim < 1 or student_log_probs.shape[-1] == 0:
        raise ValueError("student_log_probs must include a non-empty vocabulary axis")
    if student_log_probs.device != target_probs.device:
        raise ValueError(
            "student_log_probs and target_probs must be on the same device"
        )
    if pointwise_clip is not None:
        pointwise_clip = float(pointwise_clip)
        if not math.isfinite(pointwise_clip) or pointwise_clip < 0:
            raise ValueError("pointwise_clip must be finite and non-negative")

    token_shape = student_log_probs.shape[:-1]
    work_dtype = torch.promote_types(student_log_probs.dtype, target_probs.dtype)
    for name, selector in (
        ("token_mask", token_mask),
        ("token_weights", token_weights),
    ):
        if selector is None:
            continue
        if not isinstance(selector, Tensor) or selector.is_complex():
            raise TypeError(f"{name} must be a real-valued tensor")
        if selector.shape != token_shape:
            raise ValueError(f"{name} must match the non-vocabulary axes")
    if token_weights is not None:
        work_dtype = torch.promote_types(work_dtype, token_weights.dtype)
    if work_dtype in (torch.float16, torch.bfloat16):
        work_dtype = torch.float32
    student = student_log_probs.to(dtype=work_dtype)
    target = target_probs.detach().to(dtype=work_dtype)
    reduction_weights = student.new_ones(token_shape)
    if token_mask is not None:
        mask = token_mask.detach()
        if mask.dtype != torch.bool and (
            not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all()
        ):
            raise ValueError("token_mask must be boolean or finite and binary")
        reduction_weights = reduction_weights * mask.to(
            device=student.device, dtype=work_dtype
        )
    if token_weights is not None:
        weights = token_weights.detach().to(device=student.device, dtype=work_dtype)
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("token_weights must be finite and non-negative")
        reduction_weights = reduction_weights * weights

    active = reduction_weights > 0
    active_target = target[active]
    if not bool(torch.isfinite(active_target).all()) or bool((active_target < 0).any()):
        raise ValueError("active target_probs must be finite and non-negative")
    target_mass = active_target.sum(dim=-1)
    if not torch.allclose(
        target_mass,
        torch.ones_like(target_mass),
        atol=max(1e-6, 2.0 * torch.finfo(target_probs.dtype).eps),
        rtol=0.0,
    ):
        raise ValueError("active target_probs must sum to one on the vocabulary axis")
    active_student = student.detach()[active]
    if bool(torch.isnan(active_student).any()) or bool(
        torch.isposinf(active_student).any()
    ):
        raise ValueError("active student_log_probs must not contain NaN or +inf")
    if not bool(torch.isfinite(active_student[active_target > 0]).all()):
        raise ValueError("active student_log_probs must be finite on target support")
    # Quantized log probabilities near -log(V) have an absolute rounding error
    # proportional to log(V). Validate in log space, using THIS input's precision
    # so a low-precision peer never relaxes a float64 distribution's contract.
    student_epsilon = torch.finfo(student_log_probs.dtype).eps
    log_mass_tolerance = max(
        1e-6,
        2.0 * student_epsilon,
        0.5 * student_epsilon * math.log(student_log_probs.shape[-1]),
    )
    student_log_mass = torch.logsumexp(active_student, dim=-1)
    if not torch.allclose(
        student_log_mass,
        torch.zeros_like(student_log_mass),
        atol=log_mass_tolerance,
        rtol=0.0,
    ):
        raise ValueError(
            "active student_log_probs must encode normalized probabilities"
        )

    # Sanitize BEFORE arithmetic so inactive NaN/Inf cannot contaminate backward.
    target = torch.where(active.unsqueeze(-1), target, torch.zeros_like(target))
    target = target / torch.where(
        active, target.sum(dim=-1), torch.ones_like(reduction_weights)
    ).unsqueeze(-1)
    student = torch.where(active.unsqueeze(-1), student, torch.zeros_like(student))
    student = student - torch.logsumexp(student, dim=-1, keepdim=True)
    student_on_target_support = torch.where(
        target > 0, student, torch.zeros_like(student)
    )
    token_kl = torch.special.xlogy(target, target).sum(dim=-1) - (
        target * student_on_target_support
    ).sum(dim=-1)
    if pointwise_clip is not None:
        token_kl = token_kl.clamp(max=pointwise_clip)
    numerator = (token_kl * reduction_weights).sum()
    denominator = reduction_weights.sum()
    return numerator / denominator.clamp_min(torch.finfo(denominator.dtype).tiny)


__all__ = ["forward_kl"]
