"""Arithmetic neural posterior-predictive OPSD targets and losses."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from numbers import Real
from typing import Literal

import torch

from .posterior import NeuralPosterior


Pooling = Literal["arithmetic", "geometric"]
StrengthMode = Literal["constant", "sqrt_information_gain", "information_gain"]
TeacherCandidateSource = torch.Tensor | Iterable[torch.Tensor]


@dataclass(frozen=True)
class NPPTeacherCorrection:
    utility: torch.Tensor
    pooling: str
    posterior_predictive_log_probabilities: torch.Tensor | None = None
    reference_predictive_log_probabilities: torch.Tensor | None = None

    @property
    def log_density_ratio(self) -> torch.Tensor:
        return self.utility


@dataclass(frozen=True)
class NPPTarget:
    log_probabilities: torch.Tensor
    utility: torch.Tensor
    strength: torch.Tensor

    @property
    def correction(self) -> torch.Tensor:
        return self.utility


@dataclass(frozen=True)
class NPPLossStats:
    loss: torch.Tensor
    token_count: torch.Tensor


def build_teacher_correction(
    teacher_candidate_logits: TeacherCandidateSource,
    neural_posterior: NeuralPosterior,
    *,
    pooling: Pooling = "arithmetic",
) -> NPPTeacherCorrection:
    """Compute log m_w-log m_r using only two [B,T,V] accumulators.

    Arithmetic posterior-predictive mixing is the primary method. Geometric
    expected-log mixing is retained only as an explicitly named ablation.
    """
    if pooling not in {"arithmetic", "geometric"}:
        raise ValueError("pooling must be one of: arithmetic, geometric")
    _validate_posterior(neural_posterior)
    batch, candidate_count = neural_posterior.posterior.shape

    dense = isinstance(teacher_candidate_logits, torch.Tensor)
    if dense:
        values = teacher_candidate_logits
        if values.ndim != 4:
            raise ValueError("dense teacher logits must have shape [batch,time,candidates,vocab]")
        if values.shape[0] != batch or values.shape[2] != candidate_count:
            raise ValueError("teacher batch/candidate axes must match the posterior")
        if values.shape[1] < 1 or values.shape[3] < 1:
            raise ValueError("teacher time/vocabulary axes must be non-empty")
        if not values.is_floating_point():
            raise ValueError("teacher logits must be floating-point")
        iterator = None
    else:
        try:
            iterator = iter(teacher_candidate_logits)
        except TypeError as error:
            raise ValueError("teacher logits must be a rank-4 tensor or iterable") from error
        values = None

    posterior_acc = reference_acc = None
    expected_shape = None
    expected_dtype = expected_device = None
    with torch.no_grad():
        for index in range(candidate_count):
            if values is not None:
                logits = values[:, :, index, :]
            else:
                assert iterator is not None
                try:
                    logits = next(iterator)
                except StopIteration as error:
                    raise ValueError(f"teacher stream must yield exactly {candidate_count} candidates") from error
            if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
                raise ValueError("each teacher candidate must be [batch,time,vocab]")
            if not logits.is_floating_point() or logits.shape[0] != batch:
                raise ValueError("teacher candidates must be floating-point with matching batch")
            if logits.shape[1] < 1 or logits.shape[2] < 1:
                raise ValueError("teacher time/vocabulary axes must be non-empty")
            if expected_shape is None:
                expected_shape = logits.shape
                expected_dtype, expected_device = logits.dtype, logits.device
                dtype = _promoted_dtype(logits, neural_posterior.posterior, neural_posterior.prior)
                shape = tuple(logits.shape)
                fill = -torch.inf if pooling == "arithmetic" else 0.0
                posterior_acc = torch.full(shape, fill, dtype=dtype, device=logits.device)
                reference_acc = torch.full_like(posterior_acc, fill)
                log_w = _renormalize_log_weights(neural_posterior.posterior, dtype)
                log_r = _renormalize_log_weights(neural_posterior.prior, dtype)
            elif logits.shape != expected_shape or logits.dtype != expected_dtype or logits.device != expected_device:
                raise ValueError("teacher candidate tensors must have a consistent shape, dtype, and device")
            if logits.device != neural_posterior.posterior.device:
                raise ValueError("teacher candidates and posterior must share a device")

            assert posterior_acc is not None and reference_acc is not None
            active = neural_posterior.candidate_mask[:, index]
            promoted = logits.to(posterior_acc.dtype)
            active_logits = promoted[active]
            if active_logits.numel() and (
                torch.isnan(active_logits).any()
                or torch.isposinf(active_logits).any()
                or not torch.isfinite(active_logits).any(dim=-1).all()
            ):
                raise ValueError("each active teacher row must have finite support and no NaN/+inf")
            if pooling == "geometric" and active_logits.numel() and not torch.isfinite(active_logits).all():
                raise ValueError("geometric pooling requires strictly positive teacher probabilities")
            safe = torch.where(active[:, None, None], promoted, torch.zeros_like(promoted))
            log_t = torch.log_softmax(safe, dim=-1)
            wk = log_w[:, index][:, None, None]
            rk = log_r[:, index][:, None, None]
            if pooling == "arithmetic":
                posterior_acc = torch.logaddexp(posterior_acc, wk + log_t)
                reference_acc = torch.logaddexp(reference_acc, rk + log_t)
            else:
                posterior_acc = posterior_acc + torch.where(
                    torch.isfinite(wk), wk.exp() * log_t, torch.zeros_like(log_t)
                )
                reference_acc = reference_acc + torch.where(
                    torch.isfinite(rk), rk.exp() * log_t, torch.zeros_like(log_t)
                )

        if values is None:
            assert iterator is not None
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise ValueError(f"teacher stream must yield exactly {candidate_count} candidates")

        assert posterior_acc is not None and reference_acc is not None
        if pooling == "geometric":
            return NPPTeacherCorrection(posterior_acc - reference_acc, pooling)
        invalid = torch.isfinite(posterior_acc) & torch.isneginf(reference_acc)
        if invalid.any():
            raise ValueError("posterior predictive support must be contained in reference support")
        shared_zero = torch.isneginf(posterior_acc) & torch.isneginf(reference_acc)
        raw = posterior_acc - reference_acc
        utility = torch.where(shared_zero, torch.zeros_like(raw), raw)
        if torch.isnan(utility).any() or torch.isposinf(utility).any():
            raise ValueError("arithmetic density ratio is not well defined")
        return NPPTeacherCorrection(utility, pooling, posterior_acc, reference_acc)


def compose_npp_target(
    correction: NPPTeacherCorrection,
    student_anchor_logits: torch.Tensor,
    neural_posterior: NeuralPosterior,
    *,
    ratio_strength: float = 1.0,
    strength_mode: StrengthMode = "constant",
    strength_max: float | None = None,
) -> NPPTarget:
    """Compute log Q = log S_ref + lambda(log m_w-log m_r)-log Z."""
    _validate_nonnegative("ratio_strength", ratio_strength)
    if strength_max is not None:
        _validate_nonnegative("strength_max", strength_max)
    if strength_mode not in {"constant", "sqrt_information_gain", "information_gain"}:
        raise ValueError("invalid strength_mode")
    if not isinstance(correction, NPPTeacherCorrection) or correction.utility.ndim != 3:
        raise ValueError("correction must contain [batch,time,vocab] utility")
    if not isinstance(student_anchor_logits, torch.Tensor) or student_anchor_logits.ndim != 3:
        raise ValueError("student_anchor_logits must be [batch,time,vocab]")
    if correction.utility.shape != student_anchor_logits.shape:
        raise ValueError("student reference and correction shapes must match")
    if correction.utility.device != student_anchor_logits.device:
        raise ValueError("student reference and correction must share a device")
    _validate_posterior(neural_posterior)

    with torch.no_grad():
        dtype = _promoted_dtype(correction.utility, student_anchor_logits, neural_posterior.information_gain)
        utility = correction.utility.detach().to(dtype)
        reference = student_anchor_logits.detach().to(dtype)
        if not reference.is_floating_point() or not torch.isfinite(reference).all():
            raise ValueError("student reference logits must be finite floating-point")
        ig = neural_posterior.information_gain.detach().to(torch.float64)
        if strength_mode == "constant":
            multiplier = torch.ones_like(ig)
        elif strength_mode == "sqrt_information_gain":
            multiplier = ig.sqrt()
        else:
            multiplier = ig
        strength64 = multiplier * float(ratio_strength)
        if strength_max is not None:
            strength64 = strength64.clamp(max=float(strength_max))
        if not torch.isfinite(strength64).all():
            raise ValueError("effective correction strength must be finite")
        if dtype != torch.float64:
            cast = strength64.to(dtype)
            positive = strength64 > 0
            if (positive & (cast == 0)).any() or not torch.isfinite(cast).all():
                dtype = torch.float64
                utility, reference = utility.to(dtype), reference.to(dtype)
        strength = strength64.to(dtype)
        reference_log_probs = torch.log_softmax(reference, dim=-1)
        support = torch.isfinite(reference_log_probs)
        finite_ratio = torch.isfinite(utility)
        center_support = support & finite_ratio
        if (strength[:, None] > 0).logical_and(~center_support.any(dim=-1)).any():
            raise ValueError("positive-strength target must retain finite support")
        center = torch.where(center_support, utility, -torch.inf).amax(dim=-1, keepdim=True)
        center = torch.where(torch.isfinite(center), center, torch.zeros_like(center))
        centered = torch.where(support, utility - center, torch.zeros_like(utility))
        scaled = torch.zeros_like(centered)
        positive = strength > 0
        scaled[positive] = strength[positive, None, None] * centered[positive]
        unnormalized = reference_log_probs + scaled
        normalized = torch.log_softmax(unnormalized, dim=-1)
        # Preserve Q=S_ref as an exact finite-precision fixed point.
        anchor_rows = (scaled == 0).all(dim=-1, keepdim=True)
        log_probabilities = torch.where(anchor_rows, reference_log_probs, normalized)
        if torch.isnan(log_probabilities).any() or torch.isposinf(log_probabilities).any():
            raise ValueError("target normalization produced invalid values")
        return NPPTarget(log_probabilities, utility, strength)


def build_npp_target(
    teacher_candidate_logits: TeacherCandidateSource,
    student_anchor_logits: torch.Tensor,
    neural_posterior: NeuralPosterior,
    *,
    pooling: Pooling = "arithmetic",
    ratio_strength: float = 1.0,
    strength_mode: StrengthMode = "constant",
    strength_max: float | None = None,
) -> NPPTarget:
    correction = build_teacher_correction(teacher_candidate_logits, neural_posterior, pooling=pooling)
    return compose_npp_target(
        correction,
        student_anchor_logits,
        neural_posterior,
        ratio_strength=ratio_strength,
        strength_mode=strength_mode,
        strength_max=strength_max,
    )


def forward_kl_loss(
    target_log_probs: torch.Tensor,
    student_logits: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> NPPLossStats:
    """Genuine masked full-vocabulary KL(Q || S), never pointwise clipped."""
    target, student, count, exact_anchor = _prepare_loss(target_log_probs, student_logits, token_mask)
    q = target.exp()
    log_p = torch.log_softmax(student, dim=-1)
    p = log_p.exp()
    positive = q > 0
    d = torch.where(positive, target - log_p, torch.zeros_like(log_p))
    limit = math.log(torch.finfo(d.dtype).max) - 2.0
    near = positive & (d >= -limit)
    safe_minus_d = torch.where(near, -d, torch.zeros_like(d))
    pointwise = torch.where(near, q * (d + torch.expm1(safe_minus_d)), torch.zeros_like(d))
    far = positive & ~near
    pointwise = torch.where(far, q * d + p - q, pointwise)
    pointwise = torch.where(positive, pointwise, p)
    per_token = pointwise.sum(dim=-1)
    per_token = torch.where(exact_anchor, student[..., 0] * 0.0, per_token)
    raw = per_token.mean()
    loss = raw + (raw.clamp_min(0.0) - raw).detach()
    if not torch.isfinite(loss):
        raise ValueError("forward KL must be finite")
    return NPPLossStats(loss, count)


def opsd_pointwise_clipped_surrogate(
    target_log_probs: torch.Tensor,
    student_logits: torch.Tensor,
    token_mask: torch.Tensor | None = None,
    *,
    token_clip: float,
) -> NPPLossStats:
    """Official OPSD pointwise upper-clipped surrogate (not a true KL)."""
    _validate_nonnegative("token_clip", token_clip)
    target, student, count, _ = _prepare_loss(target_log_probs, student_logits, token_mask)
    q = target.exp()
    log_p = torch.log_softmax(student, dim=-1)
    terms = torch.where(q > 0, q * (target - log_p), torch.zeros_like(log_p))
    loss = terms.clamp(max=float(token_clip)).sum(dim=-1).mean()
    if not torch.isfinite(loss):
        raise ValueError("clipped OPSD surrogate must be finite")
    return NPPLossStats(loss, count)


def npp_opsd_loss(
    target_log_probs: torch.Tensor,
    student_logits: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> NPPLossStats:
    return forward_kl_loss(target_log_probs, student_logits, token_mask)


def _prepare_loss(target_log_probs, student_logits, token_mask):
    if not isinstance(target_log_probs, torch.Tensor) or target_log_probs.ndim != 3:
        raise ValueError("target_log_probs must be [batch,time,vocab]")
    if not isinstance(student_logits, torch.Tensor) or student_logits.ndim != 3:
        raise ValueError("student_logits must be [batch,time,vocab]")
    if target_log_probs.shape != student_logits.shape or target_log_probs.shape[-1] < 1:
        raise ValueError("target and student shapes must match with non-empty vocab")
    if target_log_probs.device != student_logits.device:
        raise ValueError("target and student must share a device")
    if not target_log_probs.is_floating_point() or not student_logits.is_floating_point():
        raise ValueError("target and student must be floating-point")
    if token_mask is None:
        mask = torch.ones_like(target_log_probs[..., 0], dtype=torch.bool)
    else:
        if not isinstance(token_mask, torch.Tensor) or token_mask.shape != target_log_probs.shape[:2] or token_mask.dtype != torch.bool:
            raise ValueError("token_mask must be boolean [batch,time]")
        if token_mask.device != target_log_probs.device:
            raise ValueError("token_mask must share the logits device")
        mask = token_mask
    count = mask.sum()
    if count.item() == 0:
        raise ValueError("token_mask must contain an active token")
    stored_target = target_log_probs.detach()[mask]
    stored_student = student_logits[mask]
    dtype = _promoted_dtype(stored_target, stored_student)
    target, student = stored_target.to(dtype), stored_student.to(dtype)
    if torch.isnan(target).any() or torch.isposinf(target).any() or not torch.isfinite(target).any(dim=-1).all():
        raise ValueError("each active target row must have valid finite support")
    tolerance = 5e-3 if target_log_probs.dtype in {torch.float16, torch.bfloat16} else (1e-10 if target_log_probs.dtype == torch.float64 else 1e-5)
    log_z = torch.logsumexp(target, dim=-1)
    if (log_z.abs() > tolerance).any():
        raise ValueError("active target_log_probs must be normalized")
    if not torch.isfinite(student).all():
        raise ValueError("active student logits must be finite")
    stored_anchor = torch.log_softmax(stored_student.detach(), dim=-1)
    exact_anchor = (stored_target == stored_anchor.to(stored_target.dtype)).all(dim=-1)
    return target - log_z[:, None], student, count, exact_anchor


def _renormalize_log_weights(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    promoted = values.detach().to(dtype)
    return promoted - torch.logsumexp(promoted, dim=-1, keepdim=True)


def _validate_posterior(value: NeuralPosterior) -> None:
    if not isinstance(value, NeuralPosterior):
        raise ValueError("neural_posterior must be a NeuralPosterior")
    p, r, ig, mask = value.posterior, value.prior, value.information_gain, value.candidate_mask
    if p.ndim != 2 or r.shape != p.shape or mask.shape != p.shape or ig.shape != (p.shape[0],):
        raise ValueError("invalid NeuralPosterior shapes")
    if mask.dtype != torch.bool or not mask.any(dim=-1).all():
        raise ValueError("invalid NeuralPosterior candidate mask")
    if not (p.device == r.device == ig.device == mask.device):
        raise ValueError("NeuralPosterior tensors must share a device")
    if (mask & torch.isfinite(p) & torch.isneginf(r)).any():
        raise ValueError("posterior support must be contained in prior support")
    if not torch.isfinite(ig).all() or (ig < 0).any():
        raise ValueError("information_gain must be finite and non-negative")


def _promoted_dtype(*values: torch.Tensor) -> torch.dtype:
    dtype = torch.float32
    for value in values:
        dtype = torch.promote_types(dtype, value.dtype)
    return dtype


def _validate_nonnegative(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} must be a finite non-negative real number")


__all__ = [
    "NPPTeacherCorrection", "NPPTarget", "NPPLossStats", "Pooling", "StrengthMode",
    "TeacherCandidateSource", "build_teacher_correction", "compose_npp_target",
    "build_npp_target", "forward_kl_loss", "opsd_pointwise_clipped_surrogate",
    "npp_opsd_loss",
]
