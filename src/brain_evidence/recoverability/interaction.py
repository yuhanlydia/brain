"""Paired brain--pronunciation interactions and negative aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..pronunciation import Pronunciation, canonicalize_pronunciation
from .geometry import _fisher_work_dtype, fisher_center

PronunciationIdentity = Pronunciation


@dataclass(frozen=True)
class AnchorLocalValidity:
    """Pronunciation identities required by an anchor-local control contrast."""

    anchor_pronunciation: PronunciationIdentity
    negative_pronunciation: PronunciationIdentity
    control_pronunciation: PronunciationIdentity

    def __post_init__(self) -> None:
        anchor = canonicalize_pronunciation(self.anchor_pronunciation)
        negative = canonicalize_pronunciation(self.negative_pronunciation)
        control = canonicalize_pronunciation(self.control_pronunciation)
        if negative == anchor:
            raise ValueError("negative pronunciation must differ from the anchor")
        if control in {anchor, negative}:
            raise ValueError(
                "control pronunciation must differ from anchor and negative"
            )


def anchor_local_interaction(
    xi_pi: Tensor,
    xi_pj: Tensor,
    xc_pi: Tensor,
    xc_pj: Tensor,
    student_probs: Tensor,
    *,
    validity: AnchorLocalValidity,
    null_interaction: Tensor | None = None,
) -> Tensor:
    """Return an anchor-local, Fisher-centered difference-in-differences.

    ``x_c`` has its own pronunciation ``p_c``, distinct from both ``p_i`` and
    ``p_j``. Consequently only ``(x_i, p_i)`` is a matched cell; the other
    three inputs are mismatches. A registered null can remove a generic
    brain-modulated copying interaction before Fisher centering.
    """
    if not isinstance(validity, AnchorLocalValidity):
        raise TypeError("validity must be an AnchorLocalValidity instance")
    return _raw_anchor_local_interaction(
        xi_pi,
        xi_pj,
        xc_pi,
        xc_pj,
        student_probs,
        null_interaction=null_interaction,
    )


def _raw_anchor_local_interaction(
    xi_pi: Tensor,
    xi_pj: Tensor,
    xc_pi: Tensor,
    xc_pj: Tensor,
    student_probs: Tensor,
    *,
    null_interaction: Tensor | None = None,
) -> Tensor:
    """Compute unchecked four-cell algebra for low-level compatibility."""
    teacher_states = (xi_pi, xi_pj, xc_pi, xc_pj)
    if any(state.shape != student_probs.shape for state in teacher_states):
        raise ValueError("all teacher states and student_probs must share a shape")
    if any(state.device != student_probs.device for state in teacher_states):
        raise ValueError("all tensors must be on the same device")
    if any(not state.is_floating_point() for state in teacher_states):
        raise ValueError("teacher states must be floating-point tensors")
    if any(not bool(torch.isfinite(state.detach()).all()) for state in teacher_states):
        raise ValueError("teacher states must be finite")
    if null_interaction is not None:
        if null_interaction.shape != student_probs.shape:
            raise ValueError("null_interaction must share the teacher-state shape")
        if null_interaction.device != student_probs.device:
            raise ValueError("null_interaction must be on the same device")
        if not null_interaction.is_floating_point():
            raise ValueError("null_interaction must be floating-point")
        if not bool(torch.isfinite(null_interaction.detach()).all()):
            raise ValueError("null_interaction must be finite")

    work_dtype = _fisher_work_dtype(student_probs, *teacher_states)
    detached_states = tuple(
        state.detach().to(dtype=work_dtype) for state in teacher_states
    )
    detached_student_probs = student_probs.detach().to(dtype=work_dtype)
    relative_states = tuple(state - state[..., :1] for state in detached_states)
    raw_interaction = (relative_states[0] - relative_states[1]) - (
        relative_states[2] - relative_states[3]
    )
    if null_interaction is not None:
        detached_null = null_interaction.detach().to(dtype=work_dtype)
        raw_interaction = raw_interaction - (detached_null - detached_null[..., :1])
    return fisher_center(raw_interaction, detached_student_probs).detach()


def factorial_interaction(
    xi_pi: Tensor,
    xi_pj: Tensor,
    xj_pi: Tensor,
    xj_pj: Tensor,
    student_probs: Tensor,
) -> Tensor:
    """Compatibility alias for the four-cell algebra.

    The primary method is :func:`anchor_local_interaction`, whose control
    trial has an unrelated true pronunciation and therefore does not include
    a second matched diagonal cell.
    """
    return _raw_anchor_local_interaction(
        xi_pi,
        xi_pj,
        xj_pi,
        xj_pj,
        student_probs,
    )


def _normalized_weights(interactions: Tensor, weights: Tensor) -> Tensor:
    """Expand documented weight layouts and normalize the negative axis."""
    detached_weights = weights.detach()
    if detached_weights.device != interactions.device:
        raise ValueError("weights and interactions must be on the same device")
    if detached_weights.dtype == torch.bool or detached_weights.is_complex():
        raise ValueError("weights must have a real numeric dtype")
    detached_weights = detached_weights.to(dtype=interactions.dtype)

    if detached_weights.ndim == interactions.ndim:
        if detached_weights.shape[-1] != 1:
            raise ValueError(
                "weights may only retain the vocabulary axis as a singleton"
            )
        detached_weights = detached_weights.squeeze(-1)

    target_shape = interactions.shape[:-1]
    negative_count = target_shape[-1]
    if detached_weights.shape == (negative_count,):
        reshape = (1,) * (len(target_shape) - 1) + (negative_count,)
        detached_weights = detached_weights.reshape(reshape)
    elif (
        len(target_shape) >= 3
        and detached_weights.ndim == 2
        and detached_weights.shape[0] in {1, target_shape[0]}
        and detached_weights.shape[1] == negative_count
    ):
        reshape = (
            detached_weights.shape[0],
            *((1,) * (len(target_shape) - 2)),
            negative_count,
        )
        detached_weights = detached_weights.reshape(reshape)
    elif detached_weights.ndim != len(target_shape):
        raise ValueError("weights must be [K], [B,K], or explicitly shaped [...,K]")

    if detached_weights.shape[-1] != negative_count or any(
        size not in {1, target}
        for size, target in zip(detached_weights.shape, target_shape, strict=True)
    ):
        raise ValueError("weights must be [K], [B,K], or explicitly shaped [...,K]")
    broadcast_weights = detached_weights.expand(target_shape)

    if not bool(torch.isfinite(broadcast_weights).all()):
        raise ValueError("weights must be finite")
    if bool((broadcast_weights < 0).any()):
        raise ValueError("weights must be non-negative")

    weight_mass = broadcast_weights.sum(dim=-1, keepdim=True)
    if bool((weight_mass <= 0).any()):
        raise ValueError("weights must have positive mass for every aggregate")
    return (broadcast_weights / weight_mass).detach()


def aggregate_interactions(
    interactions: Tensor,
    probabilities: Tensor,
    *,
    method: str = "mean",
    weights: Tensor | None = None,
) -> Tensor:
    """Aggregate and Fisher-center ``[..., negatives, vocabulary]`` tensors.

    ``probabilities`` has shape ``[..., vocabulary]``. Optional weights use one
    of the explicit layouts ``[K]``, ``[B, K]``, or an axis-explicit tensor
    broadcastable to ``[..., K]``. With ``method="median"`` they define a
    coordinatewise weighted median; without weights, PyTorch's deterministic
    lower median is used.
    """
    if interactions.ndim < 2:
        raise ValueError("interactions must have negative and vocabulary axes")
    if interactions.shape[-2] == 0 or interactions.shape[-1] == 0:
        raise ValueError("negative and vocabulary axes must be non-empty")
    if not interactions.is_floating_point():
        raise ValueError("interactions must be a floating-point tensor")
    expected_probability_shape = interactions.shape[:-2] + interactions.shape[-1:]
    if probabilities.shape != expected_probability_shape:
        raise ValueError(
            "probabilities must match interaction leading and vocabulary axes"
        )
    if probabilities.device != interactions.device:
        raise ValueError("probabilities and interactions must share a device")
    if method not in {"mean", "median"}:
        raise ValueError("method must be 'mean' or 'median'")

    work_dtype = _fisher_work_dtype(interactions, probabilities)
    detached_interactions = interactions.detach().to(dtype=work_dtype)
    detached_probabilities = probabilities.detach().to(dtype=work_dtype)
    normalized_weights = (
        None if weights is None else _normalized_weights(detached_interactions, weights)
    )

    if method == "mean":
        if normalized_weights is None:
            aggregate = detached_interactions.mean(dim=-2)
        else:
            aggregate = (detached_interactions * normalized_weights.unsqueeze(-1)).sum(
                dim=-2
            )
    elif normalized_weights is None:
        aggregate = detached_interactions.median(dim=-2).values
    else:
        sorted_values, sorted_indices = detached_interactions.sort(dim=-2)
        expanded_weights = normalized_weights.unsqueeze(-1).expand_as(
            detached_interactions
        )
        sorted_weights = expanded_weights.gather(dim=-2, index=sorted_indices)
        cumulative_weights = sorted_weights.cumsum(dim=-2)
        median_indices = (cumulative_weights >= 0.5).to(torch.int64).argmax(dim=-2)
        aggregate = sorted_values.gather(
            dim=-2,
            index=median_indices.unsqueeze(-2),
        ).squeeze(-2)

    return fisher_center(aggregate, detached_probabilities).detach()
