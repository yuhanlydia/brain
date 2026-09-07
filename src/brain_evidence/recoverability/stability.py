"""Leakage-safe cross-session stability for raw vocabulary contrasts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ..pronunciation import Pronunciation, canonicalize_pronunciation
from .geometry import fisher_center, fisher_inner, fisher_norm_sq

_MISSING_SUPPORT_STABILITY = 0.0


def _require_nonempty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{name} must not have leading or trailing whitespace")


@dataclass(frozen=True)
class StabilityBankEntry:
    """A raw contrast and the provenance needed to decide if it is reusable."""

    raw_contrast: Tensor
    split: str
    fold_id: str
    subject_id: str
    session_id: str
    trial_id: str
    checkpoint_id: str
    pronunciation: Pronunciation
    vocabulary_support: str
    prefix_policy: str
    alignment_policy: str

    def __post_init__(self) -> None:
        for name in (
            "split",
            "fold_id",
            "subject_id",
            "session_id",
            "trial_id",
            "checkpoint_id",
            "vocabulary_support",
            "prefix_policy",
            "alignment_policy",
        ):
            _require_nonempty_string(getattr(self, name), name)
        canonicalize_pronunciation(self.pronunciation)


@dataclass(frozen=True)
class StabilityResult:
    """Detached stability score and support diagnostics."""

    score: Tensor
    missing_support: Tensor
    support_count: Tensor
    matched_trial_ids: tuple[str, ...]

    @property
    def stability(self) -> Tensor:
        """Name used by the recoverable-target constructor."""

        return self.score


def cross_session_stability(
    current_interaction: Tensor,
    bank: Sequence[StabilityBankEntry],
    probabilities: Tensor,
    *,
    pronunciation: Pronunciation,
    current_fold_id: str,
    current_subject_id: str,
    current_session_id: str,
    current_trial_id: str,
    checkpoint_id: str,
    vocabulary_support: str,
    prefix_policy: str,
    alignment_policy: str,
    eps: float = 1e-12,
) -> StabilityResult:
    """Score against compatible leave-session-out training-bank entries.

    Stored contrasts are deliberately raw. Both current and stored directions
    are recentered under the *current* student probabilities before computing
    Fisher cosines, so stale bank centering cannot leak into the geometry.
    Tokens without valid support fail closed with score zero and an explicit
    missing-support diagnostic, rather than asserting perfect stability.
    """

    _validate_current(current_interaction, probabilities, eps=eps)
    matched = _lookup_compatible_entries(
        bank,
        pronunciation=pronunciation,
        current_fold_id=current_fold_id,
        current_subject_id=current_subject_id,
        current_session_id=current_session_id,
        current_trial_id=current_trial_id,
        checkpoint_id=checkpoint_id,
        vocabulary_support=vocabulary_support,
        prefix_policy=prefix_policy,
        alignment_policy=alignment_policy,
    )
    _validate_matched_entries(matched, current_interaction)

    work_dtype = _stable_work_dtype(
        current_interaction,
        probabilities,
        *(entry.raw_contrast for entry in matched),
    )
    current = _to_finite_work_tensor(
        current_interaction,
        dtype=work_dtype,
        name="current_interaction",
    )
    student_probs = _to_finite_work_tensor(
        probabilities,
        dtype=work_dtype,
        name="probabilities",
    )
    student_probs = student_probs / student_probs.sum(dim=-1, keepdim=True)
    if matched:
        raw_bank = torch.stack(
            [
                _to_finite_work_tensor(
                    entry.raw_contrast,
                    dtype=work_dtype,
                    name=f"raw contrast for trial {entry.trial_id!r}",
                )
                for entry in matched
            ],
            dim=-2,
        )
    else:
        raw_bank = current.new_empty((*current.shape[:-1], 0, current.shape[-1]))

    centered_current = fisher_center(current, student_probs)
    expanded_current = centered_current.unsqueeze(-2).expand_as(raw_bank)
    expanded_probs = student_probs.unsqueeze(-2).expand_as(raw_bank)
    centered_bank = fisher_center(raw_bank, expanded_probs)

    numerator = fisher_inner(expanded_current, centered_bank, expanded_probs)
    current_norm_sq = fisher_norm_sq(expanded_current, expanded_probs)
    bank_norm_sq = fisher_norm_sq(centered_bank, expanded_probs)
    current_norm = torch.sqrt(current_norm_sq.clamp_min(0.0))
    bank_norm = torch.sqrt(bank_norm_sq.clamp_min(0.0))
    effective_eps = max(eps, torch.finfo(current.dtype).eps)
    denominator = current_norm.clamp_min(effective_eps) * bank_norm.clamp_min(
        effective_eps
    )
    valid = (current_norm > effective_eps) & (bank_norm > effective_eps)
    positive_cosine = (numerator / denominator).clamp(min=0.0, max=1.0)
    positive_cosine = torch.where(
        valid,
        positive_cosine,
        torch.zeros_like(numerator),
    )

    support_count = valid.sum(dim=-1)
    missing_support = support_count == 0
    score = positive_cosine.sum(dim=-1) / support_count.clamp_min(1)
    score = torch.where(
        missing_support,
        torch.full_like(score, _MISSING_SUPPORT_STABILITY),
        score,
    )

    return StabilityResult(
        score=score.detach(),
        missing_support=missing_support.detach(),
        support_count=support_count.detach(),
        matched_trial_ids=tuple(entry.trial_id for entry in matched),
    )


def session_stability(
    current_interaction: Tensor,
    bank: Sequence[StabilityBankEntry],
    probabilities: Tensor,
    **lookup: object,
) -> StabilityResult:
    """Alias matching the shorter diagnostic name used in experiment configs."""

    return cross_session_stability(
        current_interaction,
        bank,
        probabilities,
        **lookup,
    )


def _lookup_compatible_entries(
    bank: Sequence[StabilityBankEntry],
    *,
    pronunciation: Pronunciation,
    current_fold_id: str,
    current_subject_id: str,
    current_session_id: str,
    current_trial_id: str,
    checkpoint_id: str,
    vocabulary_support: str,
    prefix_policy: str,
    alignment_policy: str,
) -> tuple[StabilityBankEntry, ...]:
    requested_pronunciation = canonicalize_pronunciation(pronunciation)
    for name, value in (
        ("current_fold_id", current_fold_id),
        ("current_subject_id", current_subject_id),
        ("current_session_id", current_session_id),
        ("current_trial_id", current_trial_id),
        ("checkpoint_id", checkpoint_id),
        ("vocabulary_support", vocabulary_support),
        ("prefix_policy", prefix_policy),
        ("alignment_policy", alignment_policy),
    ):
        _require_nonempty_string(value, name)
    matched: list[StabilityBankEntry] = []
    for entry in bank:
        if not isinstance(entry, StabilityBankEntry):
            raise TypeError("stability bank items must be StabilityBankEntry values")
        entry_pronunciation = canonicalize_pronunciation(entry.pronunciation)
        if entry.split != "train":
            continue
        if entry_pronunciation != requested_pronunciation:
            continue
        if entry.fold_id == current_fold_id:
            continue
        if entry.subject_id != current_subject_id:
            continue
        if entry.session_id == current_session_id:
            continue
        if entry.trial_id == current_trial_id:
            continue
        if entry.checkpoint_id != checkpoint_id:
            continue
        if entry.vocabulary_support != vocabulary_support:
            raise ValueError(
                f"stability vocabulary support mismatch for trial {entry.trial_id!r}"
            )
        if entry.prefix_policy != prefix_policy:
            raise ValueError(
                f"stability prefix policy mismatch for trial {entry.trial_id!r}"
            )
        if entry.alignment_policy != alignment_policy:
            raise ValueError(
                f"stability alignment policy mismatch for trial {entry.trial_id!r}"
            )
        matched.append(entry)

    trial_ids = [entry.trial_id for entry in matched]
    if len(trial_ids) != len(set(trial_ids)):
        raise ValueError("stability bank contains duplicate matched trial provenance")
    return tuple(matched)


def _validate_current(
    current_interaction: Tensor,
    probabilities: Tensor,
    *,
    eps: float,
) -> None:
    if not isinstance(current_interaction, Tensor) or not isinstance(
        probabilities,
        Tensor,
    ):
        raise TypeError("interaction and probabilities must be torch.Tensor values")
    if current_interaction.ndim < 1 or current_interaction.shape[-1] == 0:
        raise ValueError("current_interaction needs a non-empty vocabulary axis")
    if probabilities.shape != current_interaction.shape:
        raise ValueError("probabilities must have the current interaction shape")
    if not current_interaction.is_floating_point():
        raise TypeError("current_interaction must have a floating dtype")
    if not bool(torch.isfinite(current_interaction.detach()).all()):
        raise ValueError("current_interaction must be finite")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and greater than zero")
    fisher_center(current_interaction, probabilities)


def _validate_matched_entries(
    entries: Sequence[StabilityBankEntry],
    current_interaction: Tensor,
) -> None:
    for entry in entries:
        contrast = entry.raw_contrast
        if not isinstance(contrast, Tensor):
            raise TypeError(
                f"raw contrast for trial {entry.trial_id!r} must be a torch.Tensor"
            )
        if contrast.shape != current_interaction.shape:
            raise ValueError(
                f"raw contrast shape for trial {entry.trial_id!r} is incompatible"
            )
        if contrast.device != current_interaction.device:
            raise ValueError(
                f"raw contrast device for trial {entry.trial_id!r} is incompatible"
            )
        if not contrast.is_floating_point():
            raise TypeError(
                f"raw contrast for trial {entry.trial_id!r} must be floating point"
            )
        if not bool(torch.isfinite(contrast.detach()).all()):
            raise ValueError(
                f"raw contrast for trial {entry.trial_id!r} must be finite"
            )


def _stable_work_dtype(*values: Tensor) -> torch.dtype:
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
