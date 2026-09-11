"""Tensor-only objectives for the NSD baseline adaptations.

CREDIT here is the disclosed frozen-teacher, full-vocabulary image-contrastive
adaptation.  It is not an implementation of unreleased official code.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Literal, Mapping

import torch


TeacherTargetMode = Literal["exact", "map", "uniform", "posterior"]

DAPD_PAIRS = (
    "entangled_rollout",
    "inference_reference",
    "privileged_rollout",
    "entangled_reference",
    "inference_rollout",
    "privileged_reference",
)

_DAPD = {
    "entangled_rollout": ("rollout_none", "rollout", 2.0 / 15.0),
    "inference_reference": ("reference_reference", "reference", 2.0 / 15.0),
    "privileged_rollout": ("rollout_rollout", "rollout", 2.0 / 15.0),
    "entangled_reference": ("reference_none", "reference", 2.0 / 5.0),
    "inference_rollout": ("rollout_rollout", "rollout", 2.0 / 5.0),
    "privileged_reference": ("reference_reference", "reference", 4.0 / 5.0),
}


@dataclass(frozen=True)
class VADLoss:
    loss: torch.Tensor
    target: torch.Tensor
    student_support: torch.Tensor
    rho: torch.Tensor
    importance: torch.Tensor


@dataclass(frozen=True)
class DAPDLoss:
    loss: torch.Tensor
    terms: Mapping[str, torch.Tensor]


def full_vocabulary_teacher_target(
    teacher_probabilities: torch.Tensor,
    *,
    mode: TeacherTargetMode,
    weights: torch.Tensor | None = None,
    exact_index: int = 0,
) -> torch.Tensor:
    """Return a detached arithmetic teacher distribution over the full vocabulary."""
    _rank4_probabilities(teacher_probabilities, "teacher_probabilities")
    candidate_count = teacher_probabilities.shape[2]
    if mode not in {"exact", "map", "uniform", "posterior"}:
        raise ValueError("mode must be exact, map, uniform, or posterior")
    if mode == "exact":
        _finite_scalar("exact_index", exact_index)
        if not isinstance(exact_index, int) or isinstance(exact_index, bool):
            raise ValueError("exact_index must be an integer")
        if exact_index < 0 or exact_index >= candidate_count:
            raise ValueError("exact_index is outside the candidate axis")
        return teacher_probabilities[:, :, exact_index].detach()
    if mode == "uniform":
        return teacher_probabilities.detach().mean(dim=2)
    normalized = _candidate_weights(weights, teacher_probabilities)
    if mode == "map":
        index = normalized.argmax(dim=-1)
        batch = torch.arange(teacher_probabilities.shape[0], device=teacher_probabilities.device)
        return teacher_probabilities.detach()[batch[:, None], torch.arange(teacher_probabilities.shape[1], device=teacher_probabilities.device)[None, :], index[:, None]]
    return (teacher_probabilities.detach() * normalized[:, None, :, None]).sum(dim=2)


def masked_cross_entropy(logits: torch.Tensor, gold_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    _logits_and_mask(logits, mask, "logits")
    if gold_tokens.shape != mask.shape or gold_tokens.dtype != torch.long:
        raise ValueError("gold_tokens must be int64 with the same shape as mask")
    active_gold = gold_tokens[mask]
    if ((active_gold < 0) | (active_gold >= logits.shape[-1])).any():
        raise ValueError("active gold token is outside the vocabulary")
    losses = -torch.log_softmax(logits[mask], dim=-1).gather(-1, active_gold[:, None]).squeeze(-1)
    return losses.mean()


def credit_image_contrastive_loss(
    student_logits: torch.Tensor,
    positive_teacher_probabilities: torch.Tensor,
    negative_teacher_probabilities: torch.Tensor,
    mask: torch.Tensor,
    *,
    negative_power: float = 0.1,
) -> torch.Tensor:
    """Frozen-teacher full-vocabulary RKL to Q proportional to T+ / T-**lambda."""
    _logits_and_mask(student_logits, mask, "student_logits")
    _matching_probabilities(positive_teacher_probabilities, student_logits, mask, "positive teacher")
    _matching_probabilities(negative_teacher_probabilities, student_logits, mask, "negative teacher")
    _finite_scalar("negative_power", negative_power)
    if negative_power < 0:
        raise ValueError("negative_power must be nonnegative")
    student_log = torch.log_softmax(student_logits[mask], dim=-1)
    with torch.no_grad():
        target_logits = positive_teacher_probabilities[mask].log() - negative_power * negative_teacher_probabilities[mask].log()
        target_log = torch.log_softmax(target_logits, dim=-1)
    student = student_log.exp()
    return (student * (student_log - target_log)).sum(dim=-1).mean()


def vad_budgeted_asvc_loss(
    student_logits: torch.Tensor,
    clear_teacher_probabilities: torch.Tensor,
    degraded_teacher_probabilities: torch.Tensor,
    mask: torch.Tensor,
    *,
    top_k: int = 100,
    positive_cap: float = 0.8,
    element_clip: float = 20.0,
    alpha: float = 0.5,
    anchor_eta: float = 0.1,
    sampled_log_probs: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    importance_cap: float = 2.0,
) -> VADLoss:
    """Released budgeted-ASVC reconstruction with student support and tail."""
    _logits_and_mask(student_logits, mask, "student_logits")
    _matching_probabilities(clear_teacher_probabilities, student_logits, mask, "clear teacher")
    _matching_probabilities(degraded_teacher_probabilities, student_logits, mask, "degraded teacher")
    for name, value in (
        ("positive_cap", positive_cap),
        ("element_clip", element_clip),
        ("alpha", alpha),
        ("anchor_eta", anchor_eta),
        ("importance_cap", importance_cap),
    ):
        _finite_scalar(name, value)
    _finite_scalar("top_k", top_k)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not 0 <= positive_cap <= 1 or element_clip <= 0:
        raise ValueError("top_k, positive_cap, or element_clip is invalid")
    if not 0 < alpha < 1 or anchor_eta < 0 or importance_cap <= 0:
        raise ValueError("alpha, anchor_eta, or importance_cap is invalid")

    student_full_log = torch.log_softmax(student_logits, dim=-1)
    student_full = student_full_log.exp()
    support_size = min(top_k, student_logits.shape[-1])
    indices = student_full.detach().topk(support_size, dim=-1).indices
    student, student_log = _support_with_tail(student_full, indices, student_full_log)
    clear, clear_log = _support_with_tail(
        clear_teacher_probabilities.detach(), indices, clear_teacher_probabilities.detach().log()
    )
    degraded, degraded_log = _support_with_tail(
        degraded_teacher_probabilities.detach(), indices, degraded_teacher_probabilities.detach().log()
    )

    with torch.no_grad():
        center = lambda log_value: log_value - log_value.mean(dim=-1, keepdim=True)
        phi_s, phi_plus, phi_minus = center(student_log.detach()), center(clear_log), center(degraded_log)
        residual, intervention = phi_plus - phi_s, phi_plus - phi_minus
        dot = (residual * intervention).sum(dim=-1, keepdim=True)
        beta = dot.clamp(min=0) / (intervention.square().sum(dim=-1, keepdim=True) + 1e-3)
        budget = torch.linalg.vector_norm(beta * intervention, dim=-1, keepdim=True)
        positive, negative = intervention.clamp(min=0), intervention.clamp(max=0)
        score_positive = (residual * positive).sum(dim=-1, keepdim=True).clamp(min=0)
        score_negative = (residual * negative).sum(dim=-1, keepdim=True).clamp(min=0)
        denominator = score_positive + score_negative + 1e-8
        allocation_positive = torch.minimum(score_positive / denominator, torch.full_like(score_positive, positive_cap))
        allocation_negative = score_negative / denominator
        shift = budget * (
            allocation_positive * positive / (torch.linalg.vector_norm(positive, dim=-1, keepdim=True) + 1e-8)
            + allocation_negative * negative / (torch.linalg.vector_norm(negative, dim=-1, keepdim=True) + 1e-8)
        )
        rho = torch.linalg.vector_norm(shift, dim=-1) / (torch.linalg.vector_norm(residual, dim=-1) + 1e-8)
        target_log = torch.log_softmax(phi_s + shift.clamp(-element_clip, element_clip), dim=-1)
        target = target_log.exp()

    main = _jsd_per_token(student_log, target_log, alpha)
    anchor = _jsd_per_token(student_log, clear_log, alpha)
    raw = main + anchor_eta * (1 - rho).clamp(0, 1) * anchor
    importance = _importance(sampled_log_probs, old_log_probs, mask, raw, importance_cap)
    loss = (raw[mask] * importance[mask]).mean()
    return VADLoss(loss, target.detach(), student, rho.detach(), importance.detach())


def dapd_loss(
    live_logits: Mapping[str, torch.Tensor],
    anchor_logits: Mapping[str, torch.Tensor],
    completion_masks: Mapping[str, torch.Tensor],
    *,
    temperature: float = 1.1,
    component_clip: float = 0.05,
) -> DAPDLoss:
    """Six directed detached-anchor DAPD losses with componentwise clipping."""
    _finite_scalar("temperature", temperature)
    _finite_scalar("component_clip", component_clip)
    if temperature <= 0 or component_clip <= 0:
        raise ValueError("temperature and component_clip must be positive")
    for completion in ("rollout", "reference"):
        if completion not in completion_masks:
            raise ValueError(f"missing {completion} completion mask")
    terms = {}
    total = None
    for name in DAPD_PAIRS:
        live_key, completion, weight = _DAPD[name]
        if live_key not in live_logits or name not in anchor_logits:
            raise ValueError(f"missing DAPD view for {name}")
        live, anchor, mask = live_logits[live_key], anchor_logits[name], completion_masks[completion]
        _logits_and_mask(live, mask, f"live_logits[{live_key}]")
        if (
            anchor.shape != live.shape
            or anchor.device != live.device
            or anchor.dtype != live.dtype
            or not anchor.is_floating_point()
        ):
            raise ValueError(f"anchor_logits[{name}] shape/device/dtype must match its live view")
        _finite_active(anchor, mask, f"anchor_logits[{name}]")
        with torch.no_grad():
            log_anchor = torch.log_softmax(anchor.detach()[mask] / temperature, dim=-1)
            anchor_probability = log_anchor.exp()
        log_live = torch.log_softmax(live[mask] / temperature, dim=-1)
        divergence = (anchor_probability * (log_anchor - log_live)).clamp(max=component_clip).sum(dim=-1).mean()
        terms[name] = divergence
        weighted = weight * divergence
        total = weighted if total is None else total + weighted
    assert total is not None
    return DAPDLoss(total, terms)


def _candidate_weights(weights: torch.Tensor | None, teachers: torch.Tensor) -> torch.Tensor:
    expected = (teachers.shape[0], teachers.shape[2])
    if (
        weights is None
        or weights.shape != expected
        or weights.device != teachers.device
        or weights.dtype != teachers.dtype
    ):
        raise ValueError(f"weights must have shape {expected} and the teacher dtype/device")
    if not weights.is_floating_point() or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weights must be finite nonnegative floating-point values")
    sums = weights.sum(dim=-1, keepdim=True)
    if (sums <= 0).any():
        raise ValueError("each weights row must have positive mass")
    detached = weights.detach()
    return detached / detached.sum(dim=-1, keepdim=True)


def _rank4_probabilities(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 4 or min(value.shape) < 1:
        raise ValueError(f"{name} must have non-empty shape [batch,time,candidates,vocab]")
    if not value.is_floating_point() or not torch.isfinite(value).all() or (value < 0).any():
        raise ValueError(f"{name} must contain finite probabilities")
    if not torch.allclose(value.sum(-1), torch.ones_like(value.sum(-1)), atol=1e-6, rtol=1e-6):
        raise ValueError(f"{name} probabilities must sum to one")


def _logits_and_mask(logits: torch.Tensor, mask: torch.Tensor, name: str) -> None:
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or min(logits.shape) < 1:
        raise ValueError(f"{name} must have non-empty shape [batch,time,vocab]")
    if (
        not logits.is_floating_point()
        or mask.shape != logits.shape[:2]
        or mask.dtype != torch.bool
        or mask.device != logits.device
    ):
        raise ValueError(f"{name} and boolean mask shapes are inconsistent")
    if not mask.any():
        raise ValueError("token mask must not be empty")
    _finite_active(logits, mask, name)


def _finite_active(value: torch.Tensor, mask: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value[mask]).all():
        raise ValueError(f"active {name} values must be finite")


def _matching_probabilities(value: torch.Tensor, logits: torch.Tensor, mask: torch.Tensor, name: str) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != logits.shape
        or value.device != logits.device
        or value.dtype != logits.dtype
    ):
        raise ValueError(f"{name} probabilities shape/dtype/device must match student logits")
    active = value[mask]
    if not value.is_floating_point() or not torch.isfinite(active).all() or (active <= 0).any():
        raise ValueError(f"active {name} probabilities must be finite and positive")
    if not torch.allclose(active.sum(-1), torch.ones_like(active.sum(-1)), atol=1e-6, rtol=1e-6):
        raise ValueError(f"active {name} probabilities must sum to one")


def _support_with_tail(
    probabilities: torch.Tensor,
    indices: torch.Tensor,
    full_log_probabilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather support and aggregate tail without subtractive cancellation.

    Log probabilities stay in the log domain, so finite logits remain finite
    even when their represented probabilities underflow to exact zero. No
    probability floor is applied; divergence terms use the zero-mass limit.
    """
    selected = probabilities.gather(-1, indices)
    selected_log = full_log_probabilities.gather(-1, indices)
    if indices.shape[-1] == probabilities.shape[-1]:
        return selected, selected_log
    in_support = torch.zeros_like(probabilities, dtype=torch.bool).scatter(-1, indices, True)
    tail_log = full_log_probabilities.masked_fill(in_support, -torch.inf).logsumexp(dim=-1, keepdim=True)
    return torch.cat((selected, tail_log.exp()), dim=-1), torch.cat((selected_log, tail_log), dim=-1)


def _jsd_per_token(student_log: torch.Tensor, target_log: torch.Tensor, alpha: float) -> torch.Tensor:
    mixture_log = torch.logaddexp(student_log + math.log1p(-alpha), target_log + math.log(alpha))
    student_term = student_log.exp() * (student_log - mixture_log)
    target_term = target_log.exp() * (target_log - mixture_log)
    return (1 - alpha) * student_term.sum(-1) + alpha * target_term.sum(-1)


def _importance(sampled: torch.Tensor | None, old: torch.Tensor | None, mask: torch.Tensor, raw: torch.Tensor, cap: float) -> torch.Tensor:
    if sampled is None and old is None:
        return torch.ones_like(raw)
    if (
        sampled is None
        or old is None
        or sampled.shape != mask.shape
        or old.shape != mask.shape
        or not sampled.is_floating_point()
        or not old.is_floating_point()
        or sampled.dtype != raw.dtype
        or old.dtype != raw.dtype
        or sampled.device != raw.device
        or old.device != raw.device
    ):
        raise ValueError(
            "sampled_log_probs and old_log_probs must be floating-point and match the loss shape/dtype/device"
        )
    if not torch.isfinite(sampled[mask]).all() or not torch.isfinite(old[mask]).all():
        raise ValueError("active importance log probabilities must be finite")
    return (sampled.detach() - old.detach()).clamp(-20, 20).exp().clamp(max=cap)


def _finite_scalar(name: str, value: float) -> None:
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite real scalar")
