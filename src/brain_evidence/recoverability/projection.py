"""One-sided neural-recoverability projection target construction."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .geometry import (
    _fisher_work_dtype,
    fisher_center,
    fisher_inner,
    fisher_norm_sq,
)
from .interaction import _normalized_weights, aggregate_interactions

ProjectionMode = Literal["nra", "interaction_only_ablation"]


def _require_nonempty_string(value: object, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{name} must not have leading or trailing whitespace")


def _require_id_tuple(value: object, name: str) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple of strings")
    if not value:
        raise ValueError(f"{name} must be non-empty")
    for item in value:
        _require_nonempty_string(item, name)
    if len(value) != len(set(value)):
        raise ValueError(f"{name} must contain distinct IDs")


@dataclass(frozen=True, kw_only=True)
class NRAInteractionBatch:
    """Current interaction values and the identity of their calibrated statistic.

    Each batch shares one provenance identity. ``values`` uses the same shape
    convention as the low-level tensor API; the negative count is always
    derived from that tensor, never supplied as an unchecked metadata field.
    """

    values: Tensor
    subject_id: str
    pronunciation_id: str
    reliability_stratum: str
    current_fold_id: str
    current_session_id: str
    current_trial_id: str
    checkpoint_id: str
    prefix_policy: str
    alignment_policy: str
    support_policy: str
    sampler: str
    aggregation_policy: str
    weight_policy: str
    seed: int
    weights: Tensor | None = None
    control_policy: str = "anchor_local_three_mismatch"

    def __post_init__(self) -> None:
        if not isinstance(self.values, Tensor):
            raise TypeError("values must be a torch.Tensor")
        if self.weights is not None and not isinstance(self.weights, Tensor):
            raise TypeError("weights must be a torch.Tensor or None")
        for name in (
            "subject_id",
            "pronunciation_id",
            "reliability_stratum",
            "current_fold_id",
            "current_session_id",
            "current_trial_id",
            "checkpoint_id",
            "prefix_policy",
            "alignment_policy",
            "support_policy",
            "sampler",
            "aggregation_policy",
            "weight_policy",
            "control_policy",
        ):
            _require_nonempty_string(getattr(self, name), name)
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")


@dataclass(frozen=True)
class NRAEvidence:
    """Frozen cross-fitted neural evidence and its complete use provenance."""

    brain_only_direction: Tensor
    null_threshold: Tensor | float
    source: str
    split: str
    subject_id: str
    pronunciation_id: str
    reliability_stratum: str
    producer_fold_ids: tuple[str, ...]
    current_fold_id: str
    producer_session_ids: tuple[str, ...]
    current_session_id: str
    producer_trial_ids: tuple[str, ...]
    current_trial_id: str
    checkpoint_id: str
    statistic_id: str
    alpha: float
    phase: Literal["pilot", "confirmatory"]
    permutation_count: int
    k: int
    sampler: str
    aggregation_policy: str
    weight_policy: str
    seed: int
    producer_prefix_policy: str
    current_prefix_policy: str
    producer_alignment_policy: str
    current_alignment_policy: str
    producer_support_policy: str
    current_support_policy: str
    statistic_epsilon: float = 1e-8
    control_policy: Literal["anchor_local_three_mismatch"] = (
        "anchor_local_three_mismatch"
    )

    def __post_init__(self) -> None:
        if not isinstance(self.brain_only_direction, Tensor):
            raise TypeError("brain_only_direction must be a torch.Tensor")
        if not self.brain_only_direction.is_floating_point():
            raise ValueError("brain_only_direction must be floating-point")
        if not bool(torch.isfinite(self.brain_only_direction.detach()).all()):
            raise ValueError("brain_only_direction must be finite")

        if isinstance(self.null_threshold, Tensor):
            if (
                not self.null_threshold.is_floating_point()
                or self.null_threshold.is_complex()
            ):
                raise ValueError("null_threshold must have a real floating dtype")
            threshold = self.null_threshold.detach()
            if not bool(torch.isfinite(threshold).all()):
                raise ValueError("null_threshold must be finite")
            if bool(((threshold < -1) | (threshold > 1)).any()):
                raise ValueError("null_threshold must lie in [-1, 1]")
        else:
            if isinstance(self.null_threshold, bool) or not isinstance(
                self.null_threshold,
                (int, float),
            ):
                raise TypeError("null_threshold must be a real scalar or tensor")
            if not math.isfinite(self.null_threshold):
                raise ValueError("null_threshold must be finite")
            if not -1 <= self.null_threshold <= 1:
                raise ValueError("null_threshold must lie in [-1, 1]")

        if self.source != "frozen_crossfit_ce":
            raise ValueError("source must be exactly 'frozen_crossfit_ce'")
        if self.split != "train":
            raise ValueError("NRA evidence must come from the training split")
        for name in (
            "subject_id",
            "pronunciation_id",
            "reliability_stratum",
            "current_fold_id",
            "current_session_id",
            "current_trial_id",
            "checkpoint_id",
            "sampler",
            "aggregation_policy",
            "weight_policy",
            "producer_prefix_policy",
            "current_prefix_policy",
            "producer_alignment_policy",
            "current_alignment_policy",
            "producer_support_policy",
            "current_support_policy",
        ):
            _require_nonempty_string(getattr(self, name), name)
        _require_id_tuple(self.producer_fold_ids, "producer_fold_ids")
        _require_id_tuple(self.producer_session_ids, "producer_session_ids")
        _require_id_tuple(self.producer_trial_ids, "producer_trial_ids")
        if self.current_fold_id in self.producer_fold_ids:
            raise ValueError("NRA evidence must leave the current fold out")
        if self.current_session_id in self.producer_session_ids:
            raise ValueError("NRA evidence must leave the current session out")
        if self.current_trial_id in self.producer_trial_ids:
            raise ValueError("NRA evidence must leave the current trial out")
        if self.statistic_id != "signed_fisher_alignment":
            raise ValueError("statistic_id must be 'signed_fisher_alignment'")
        if isinstance(self.statistic_epsilon, bool) or not isinstance(
            self.statistic_epsilon,
            (int, float),
        ):
            raise TypeError("statistic_epsilon must be a real scalar")
        if not math.isfinite(self.statistic_epsilon) or self.statistic_epsilon < 0:
            raise ValueError("statistic_epsilon must be finite and non-negative")
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, (int, float)):
            raise TypeError("NRA evidence alpha must be a real scalar")
        if not math.isfinite(self.alpha) or self.alpha != 0.05:
            raise ValueError("NRA evidence alpha must be exactly 0.05")
        if self.phase not in {"pilot", "confirmatory"}:
            raise ValueError("NRA evidence phase must be pilot or confirmatory")
        if not isinstance(self.permutation_count, int) or isinstance(
            self.permutation_count,
            bool,
        ):
            raise TypeError("permutation_count must be an integer")
        minimum = 99 if self.phase == "pilot" else 999
        if self.permutation_count < minimum:
            raise ValueError(
                f"{self.phase} NRA evidence requires at least {minimum} permutations"
            )
        if not isinstance(self.k, int) or isinstance(self.k, bool):
            raise TypeError("k must be an integer")
        minimum_k = 1 if self.phase == "pilot" else 4
        if self.k < minimum_k:
            raise ValueError(
                f"{self.phase} NRA evidence requires k of at least {minimum_k}"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        if self.control_policy != "anchor_local_three_mismatch":
            raise ValueError("control_policy must be 'anchor_local_three_mismatch'")
        if self.producer_prefix_policy != self.current_prefix_policy:
            raise ValueError("producer and current prefix policy must be identical")
        if self.producer_alignment_policy != self.current_alignment_policy:
            raise ValueError("producer and current alignment policy must be identical")
        if self.producer_support_policy != self.current_support_policy:
            raise ValueError("producer and current support policy must be identical")


def _as_leading_tensor(
    name: str,
    value: Tensor | float,
    *,
    leading_shape: torch.Size,
    reference: Tensor,
) -> Tensor:
    if isinstance(value, Tensor) and (value.dtype == torch.bool or value.is_complex()):
        raise ValueError(f"{name} must have a real numeric dtype")
    tensor = (
        value.detach().to(device=reference.device, dtype=reference.dtype)
        if isinstance(value, Tensor)
        else torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    )
    if isinstance(value, Tensor) and value.device != reference.device:
        raise ValueError(f"{name} must be on the same device as student_log_probs")
    if tensor.shape == leading_shape + (1,):
        tensor = tensor.squeeze(-1)
    try:
        tensor = torch.broadcast_to(tensor, leading_shape)
    except RuntimeError as error:
        raise ValueError(f"{name} must broadcast to {tuple(leading_shape)}") from error
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")
    return tensor.detach()


def _bounded_gate(
    name: str,
    value: Tensor | float,
    *,
    leading_shape: torch.Size,
    reference: Tensor,
) -> Tensor:
    gate = _as_leading_tensor(
        name,
        value,
        leading_shape=leading_shape,
        reference=reference,
    )
    if bool(((gate < 0) | (gate > 1)).any()):
        raise ValueError(f"{name} must lie in [0, 1]")
    return gate


def _interaction_stack(
    interactions: Tensor,
    *,
    target_shape: torch.Size,
    reference: Tensor,
) -> Tensor:
    if interactions.device != reference.device:
        raise ValueError("interactions must be on the same device as student_log_probs")
    if not interactions.is_floating_point():
        raise ValueError("interactions must be floating-point")
    if interactions.shape == target_shape:
        stack = interactions.detach().to(dtype=reference.dtype).unsqueeze(-2)
    elif (
        interactions.ndim == len(target_shape) + 1
        and interactions.shape[:-2] == target_shape[:-1]
        and interactions.shape[-1] == target_shape[-1]
    ):
        stack = interactions.detach().to(dtype=reference.dtype)
    else:
        raise ValueError(
            "interactions must have shape [..., vocabulary] or "
            "[..., negatives, vocabulary] matching student_log_probs"
        )
    if not bool(torch.isfinite(stack).all()):
        raise ValueError("interactions must be finite")
    return stack


def _broadcast_null(null_interaction: Tensor, stack: Tensor) -> Tensor:
    if null_interaction.device != stack.device:
        raise ValueError("null_interaction must be on the same device as interactions")
    detached_null = null_interaction.detach().to(dtype=stack.dtype)
    if detached_null.shape == stack.shape[:-2] + stack.shape[-1:]:
        detached_null = detached_null.unsqueeze(-2)
    try:
        detached_null = torch.broadcast_to(detached_null, stack.shape)
    except RuntimeError as error:
        raise ValueError("null_interaction must broadcast to interactions") from error
    if not bool(torch.isfinite(detached_null).all()):
        raise ValueError("null_interaction must be finite")
    return detached_null.detach()


def _positive_projection_coefficient(
    numerator: Tensor,
    norm_sq: Tensor,
    epsilon: Tensor,
) -> Tensor:
    denominator = norm_sq + epsilon
    safe_denominator = torch.where(
        denominator > 0,
        denominator,
        torch.ones_like(denominator),
    )
    coefficient = numerator.clamp_min(0) / safe_denominator
    return torch.where(norm_sq > 0, coefficient, torch.zeros_like(coefficient))


def _validate_uniform_interaction_weights(stack: Tensor, weights: Tensor) -> None:
    """Verify the original weights without rounding or rescaling their values."""
    detached_weights = weights.detach()
    # Reuse layout/device/dtype validation with unit weights to avoid overflow
    # or underflow when an otherwise valid common weight has an extreme scale.
    _normalized_weights(stack, torch.ones_like(detached_weights))
    negative_axis = -2 if detached_weights.ndim == stack.ndim else -1
    first_weight = detached_weights.select(negative_axis, 0).unsqueeze(negative_axis)
    if (
        not bool(torch.isfinite(detached_weights).all())
        or bool((detached_weights <= 0).any())
        or not bool((detached_weights == first_weight).all())
    ):
        raise ValueError("uniform weights must be finite, positive, and equal along K")


def recoverable_target(
    student_log_probs: Tensor,
    teacher_log_probs: Tensor,
    anchor_interactions: NRAInteractionBatch | Tensor | None = None,
    neural_reliability: Tensor | float | None = None,
    session_stability: Tensor | float | None = None,
    *,
    mode: ProjectionMode,
    factorial_interactions: NRAInteractionBatch | Tensor | None = None,
    nra_evidence: NRAEvidence | None = None,
    null_interaction: Tensor | None = None,
    aggregation: str = "mean",
    interaction_weights: Tensor | None = None,
    alpha: float = 1.0,
    clip_value: float = 5.0,
    epsilon: float = 1e-8,
) -> tuple[Tensor, Mapping[str, Tensor]]:
    """Construct a detached target supported by recoverable neural evidence.

    ``anchor_interactions`` is the primary API. ``factorial_interactions`` is
    retained as a mutually exclusive compatibility name. ``mode="nra"`` fails
    closed unless a complete :class:`NRAEvidence` artifact supplies the frozen,
    cross-fitted CE direction, registered null calibration, leave-out identity,
    and matching geometry policies. Its interaction input must be an
    :class:`NRAInteractionBatch` with matching current context, statistic
    identity, and verifiable uniform weights. Separate ``interaction_weights``
    and raw ``null_interaction`` subtraction are forbidden in NRA mode. Raw
    tensors, arbitrary weights, and null-vector subtraction remain available
    only through ``mode="interaction_only_ablation"``.
    """
    if mode not in {"nra", "interaction_only_ablation"}:
        raise ValueError("mode must be 'nra' or 'interaction_only_ablation'")
    if mode == "nra":
        if neural_reliability is None:
            raise ValueError("nra mode requires explicit neural_reliability")
        if session_stability is None:
            raise ValueError("nra mode requires explicit session_stability")
        if nra_evidence is None:
            raise ValueError("nra mode requires a complete NRAEvidence artifact")
        if not isinstance(nra_evidence, NRAEvidence):
            raise TypeError("nra_evidence must be an NRAEvidence artifact")
        if null_interaction is not None:
            raise ValueError(
                "nra mode does not accept null_interaction; "
                "use interaction_only_ablation for null-vector subtraction"
            )
        brain_only_direction = nra_evidence.brain_only_direction
        null_threshold = nra_evidence.null_threshold
    else:
        if nra_evidence is not None:
            raise ValueError("interaction_only_ablation does not accept NRAEvidence")
        if neural_reliability is None:
            neural_reliability = 1.0
        if session_stability is None:
            session_stability = 1.0
        brain_only_direction = None
        null_threshold = None

    if anchor_interactions is None:
        if factorial_interactions is None:
            raise ValueError("anchor_interactions is required")
        interactions = factorial_interactions
    else:
        if factorial_interactions is not None:
            raise ValueError(
                "provide anchor_interactions or factorial_interactions, not both"
            )
        interactions = anchor_interactions

    if mode == "nra":
        if not isinstance(interactions, NRAInteractionBatch):
            raise TypeError("nra mode requires an NRAInteractionBatch artifact")
        if interaction_weights is not None:
            raise ValueError("nra mode does not accept separate interaction_weights")
        for name in (
            "subject_id",
            "pronunciation_id",
            "reliability_stratum",
            "current_fold_id",
            "current_session_id",
            "current_trial_id",
            "checkpoint_id",
            "sampler",
            "aggregation_policy",
            "weight_policy",
            "seed",
            "control_policy",
        ):
            if getattr(interactions, name) != getattr(nra_evidence, name):
                raise ValueError(f"NRA interaction {name} does not match NRA evidence")
        for name in ("prefix_policy", "alignment_policy", "support_policy"):
            if getattr(interactions, name) != getattr(nra_evidence, f"current_{name}"):
                raise ValueError(f"NRA interaction {name} does not match NRA evidence")
        if interactions.weight_policy != "uniform":
            raise ValueError("nra mode requires weight_policy='uniform'")
        interaction_weights = interactions.weights
        interactions = interactions.values

    if student_log_probs.ndim == 0 or student_log_probs.shape[-1] == 0:
        raise ValueError("student_log_probs must have a vocabulary axis")
    if student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError("student_log_probs and teacher_log_probs must share a shape")
    if student_log_probs.device != teacher_log_probs.device:
        raise ValueError("student and teacher tensors must be on the same device")
    if (
        not student_log_probs.is_floating_point()
        or not teacher_log_probs.is_floating_point()
    ):
        raise ValueError("student and teacher log-probabilities must be floating-point")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    if not math.isfinite(clip_value) or clip_value <= 0:
        raise ValueError("clip_value must be finite and positive")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be a real scalar")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and non-negative")
    if mode == "nra":
        assert nra_evidence is not None
        if epsilon != nra_evidence.statistic_epsilon:
            raise ValueError(
                "epsilon must exactly match NRA evidence statistic_epsilon"
            )

    with torch.no_grad():
        work_tensors = [student_log_probs, teacher_log_probs, interactions]
        if brain_only_direction is not None:
            work_tensors.append(brain_only_direction)
        work_dtype = _fisher_work_dtype(*work_tensors)
        detached_student_log_probs = student_log_probs.detach().to(dtype=work_dtype)
        detached_teacher_log_probs = teacher_log_probs.detach().to(dtype=work_dtype)
        if not bool(torch.isfinite(detached_student_log_probs).all()):
            raise ValueError("student_log_probs must be finite")
        if not bool(torch.isfinite(detached_teacher_log_probs).all()):
            raise ValueError("teacher_log_probs must be finite")

        epsilon_tensor = detached_student_log_probs.new_tensor(epsilon)
        if epsilon > 0 and epsilon_tensor.item() == 0:
            raise ValueError("epsilon must be representable in the work dtype")

        student_log_mass = torch.logsumexp(
            detached_student_log_probs,
            dim=-1,
            keepdim=True,
        )
        source_epsilon = torch.finfo(student_log_probs.dtype).eps
        log_mass_tolerance = max(
            1e-6,
            2.0 * source_epsilon,
            0.5 * source_epsilon * math.log(student_log_probs.shape[-1]),
        )
        if not torch.allclose(
            student_log_mass,
            torch.zeros_like(student_log_mass),
            atol=log_mass_tolerance,
            rtol=0.0,
        ):
            raise ValueError("student_log_probs must encode normalized probabilities")
        detached_student_log_probs = detached_student_log_probs - student_log_mass
        student_probs = detached_student_log_probs.exp()
        stack = _interaction_stack(
            interactions,
            target_shape=student_log_probs.shape,
            reference=detached_student_log_probs,
        )
        if nra_evidence is not None:
            if nra_evidence.k != stack.shape[-2]:
                raise ValueError("NRA evidence k does not match current interactions")
            if nra_evidence.aggregation_policy != aggregation:
                raise ValueError(
                    "NRA evidence aggregation policy does not match current use"
                )
            if interaction_weights is not None:
                _validate_uniform_interaction_weights(stack, interaction_weights)
            # Equal weights define the same registered statistic at every scale.
            interaction_weights = None
        if null_interaction is not None:
            stack = stack - _broadcast_null(null_interaction, stack)

        expanded_student_probs = student_probs.unsqueeze(-2).expand_as(stack)
        centered_stack = fisher_center(stack, expanded_student_probs)
        aggregate_interaction = aggregate_interactions(
            centered_stack,
            student_probs,
            method=aggregation,
            weights=interaction_weights,
        )

        leading_shape = student_log_probs.shape[:-1]
        reliability = _bounded_gate(
            "neural_reliability",
            neural_reliability,
            leading_shape=leading_shape,
            reference=detached_student_log_probs,
        )
        stability = _bounded_gate(
            "session_stability",
            session_stability,
            leading_shape=leading_shape,
            reference=detached_student_log_probs,
        )
        if null_threshold is None:
            threshold = detached_student_log_probs.new_zeros(leading_shape)
        else:
            threshold = _as_leading_tensor(
                "null_threshold",
                null_threshold,
                leading_shape=leading_shape,
                reference=detached_student_log_probs,
            )
            if bool(((threshold < -1) | (threshold > 1)).any()):
                raise ValueError("null_threshold must lie in [-1, 1]")

        aggregate_norm_sq = fisher_norm_sq(
            aggregate_interaction,
            student_probs,
        )
        if mode == "interaction_only_ablation":
            centered_brain_direction = torch.zeros_like(aggregate_interaction)
            brain_projection_coefficient = torch.ones_like(aggregate_norm_sq)
            brain_supported_interaction = aggregate_interaction
            alignment_score = torch.zeros_like(aggregate_norm_sq)
            kappa = torch.ones_like(aggregate_norm_sq)
            brain_only_filter_applied = detached_student_log_probs.new_tensor(0.0)
        else:
            assert brain_only_direction is not None
            if brain_only_direction.shape != student_log_probs.shape:
                raise ValueError(
                    "brain_only_direction must share the student_log_probs shape"
                )
            if brain_only_direction.device != student_log_probs.device:
                raise ValueError(
                    "brain_only_direction must be on the student_log_probs device"
                )
            if not brain_only_direction.is_floating_point():
                raise ValueError("brain_only_direction must be floating-point")
            if not bool(torch.isfinite(brain_only_direction.detach()).all()):
                raise ValueError("brain_only_direction must be finite")

            centered_brain_direction = fisher_center(
                brain_only_direction.detach(),
                student_probs,
            )
            interaction_brain_inner = fisher_inner(
                aggregate_interaction,
                centered_brain_direction,
                student_probs,
            )
            brain_norm_sq = fisher_norm_sq(
                centered_brain_direction,
                student_probs,
            )
            brain_projection_coefficient = _positive_projection_coefficient(
                interaction_brain_inner,
                brain_norm_sq,
                epsilon_tensor,
            )
            brain_supported_interaction = (
                brain_projection_coefficient.unsqueeze(-1) * centered_brain_direction
            )

            alignment_denominator = (
                aggregate_norm_sq.clamp_min(0).sqrt()
                * brain_norm_sq.clamp_min(0).sqrt()
                + epsilon_tensor
            )
            safe_alignment_denominator = torch.where(
                alignment_denominator > 0,
                alignment_denominator,
                torch.ones_like(alignment_denominator),
            )
            alignment_score = interaction_brain_inner / safe_alignment_denominator
            supported_alignment = (aggregate_norm_sq > 0) & (brain_norm_sq > 0)
            alignment_score = torch.where(
                supported_alignment,
                alignment_score,
                torch.zeros_like(alignment_score),
            ).clamp(min=-1.0, max=1.0)
            kappa_denominator = 1.0 - threshold + epsilon_tensor
            safe_kappa_denominator = torch.where(
                kappa_denominator > 0,
                kappa_denominator,
                torch.ones_like(kappa_denominator),
            )
            kappa = ((alignment_score - threshold) / safe_kappa_denominator).clamp(
                min=0.0,
                max=1.0,
            )
            kappa = torch.where(
                kappa_denominator > 0,
                kappa,
                torch.zeros_like(kappa),
            )
            brain_only_filter_applied = detached_student_log_probs.new_tensor(1.0)

        full_correction = fisher_center(
            detached_teacher_log_probs - detached_student_log_probs,
            student_probs,
        )
        projection_numerator = fisher_inner(
            full_correction,
            brain_supported_interaction,
            student_probs,
        )
        supported_norm_sq = fisher_norm_sq(
            brain_supported_interaction,
            student_probs,
        )
        projection_coefficient = _positive_projection_coefficient(
            projection_numerator,
            supported_norm_sq,
            epsilon_tensor,
        )
        projected_correction = (
            projection_coefficient.unsqueeze(-1) * brain_supported_interaction
        )
        recoverable_correction = (reliability * stability * kappa).unsqueeze(
            -1
        ) * projected_correction
        correction_max_abs = recoverable_correction.abs().amax(dim=-1)
        nonzero_correction = correction_max_abs > 0
        safe_max_abs = torch.where(
            nonzero_correction,
            correction_max_abs,
            torch.ones_like(correction_max_abs),
        )
        clip_scale = torch.where(
            nonzero_correction,
            (clip_value / safe_max_abs).clamp(max=1.0),
            torch.ones_like(correction_max_abs),
        )
        clipped_correction = clip_scale.unsqueeze(-1) * recoverable_correction
        target_work = torch.softmax(
            detached_student_log_probs + alpha * clipped_correction,
            dim=-1,
        ).detach()
        target = target_work.to(dtype=student_log_probs.dtype).detach()

        projection_magnitude = (
            fisher_norm_sq(
                recoverable_correction,
                student_probs,
            )
            .clamp_min(0)
            .sqrt()
        )
        geometric_coverage = (
            (supported_norm_sq > epsilon_tensor)
            & (projection_numerator > 0)
            & (kappa > 0)
        )
        transfer_coverage = geometric_coverage & (reliability > 0) & (stability > 0)
        geometric_coverage = geometric_coverage.to(dtype=student_probs.dtype)
        transfer_coverage = transfer_coverage.to(dtype=student_probs.dtype)
        target_kl = (
            (
                target_work
                * (
                    target_work.clamp_min(torch.finfo(target_work.dtype).tiny).log()
                    - detached_student_log_probs
                )
            )
            .sum(dim=-1)
            .clamp_min(0)
        )

        diagnostics = {
            "aggregate_interaction": aggregate_interaction,
            "aggregate_interaction_norm_sq": aggregate_norm_sq,
            "alignment_score": alignment_score,
            "brain_only_direction": centered_brain_direction,
            "brain_only_filter_applied": brain_only_filter_applied,
            "brain_projection_coefficient": brain_projection_coefficient,
            "brain_supported_interaction": brain_supported_interaction,
            "clip_scale": clip_scale,
            "clipped_correction": clipped_correction,
            "coverage": transfer_coverage,
            "full_correction": full_correction,
            "geometric_coverage": geometric_coverage,
            "kappa": kappa,
            "null_threshold": threshold,
            "projected_correction": projected_correction,
            "projection_coefficient": projection_coefficient,
            "projection_magnitude": projection_magnitude,
            "projection_numerator": projection_numerator,
            "recoverable_correction": recoverable_correction,
            "reliability": reliability,
            "stability": stability,
            "supported_interaction_norm_sq": supported_norm_sq,
            "target_kl": target_kl,
            "transfer_coverage": transfer_coverage,
        }
        return target, {name: value.detach() for name, value in diagnostics.items()}
