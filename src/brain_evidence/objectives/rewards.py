"""Verifiable sequence and grounding rewards."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from brain_evidence.metrics.grounding import box_iou, parse_box
from brain_evidence.metrics.text import word_error_rate


@dataclass(frozen=True)
class GroupRelativeRewards:
    """Standardized group advantages and equal-reward skip diagnostics."""

    advantages: Tensor
    skipped: Tensor
    skip_rate: float


def wer_reward(
    reference: str | Sequence[str], hypothesis: str | Sequence[str]
) -> float:
    """Return negative WER for sequence-level reward optimization."""

    return -word_error_rate(reference, hypothesis).wer


def dense_box_reward(
    prediction: str | Sequence[float],
    target: str | Sequence[float],
    *,
    iou_threshold: float = 0.5,
    threshold_bonus: float = 1.0,
    invalid_penalty: float = 1.0,
) -> float:
    """Combine dense IoU, a threshold bonus, and an invalid-output penalty."""

    if not all(
        math.isfinite(value)
        for value in (iou_threshold, threshold_bonus, invalid_penalty)
    ):
        raise ValueError("reward scalars must be finite")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between zero and one")
    if threshold_bonus < 0.0 or invalid_penalty < 0.0:
        raise ValueError("reward bonus and penalty must be non-negative")

    target_box = parse_box(target)
    try:
        prediction_box = parse_box(prediction)
    except (TypeError, ValueError):
        return -float(invalid_penalty)

    iou = box_iou(prediction_box, target_box)
    bonus = threshold_bonus if iou >= iou_threshold else 0.0
    return iou + bonus


def group_relative_standardize(
    rewards: Tensor, *, equal_tolerance: float = 1e-8
) -> GroupRelativeRewards:
    """Standardize rewards per group and zero groups without reward variance."""

    if rewards.ndim != 2 or rewards.shape[0] == 0 or rewards.shape[1] == 0:
        raise ValueError("rewards must have shape [groups, samples] and be non-empty")
    if not math.isfinite(equal_tolerance) or equal_tolerance < 0:
        raise ValueError("equal_tolerance must be finite and non-negative")

    values = rewards.detach()
    if not torch.isfinite(values).all():
        raise ValueError("rewards must be finite")
    centered = values - values.mean(dim=-1, keepdim=True)
    standard_deviation = centered.square().mean(dim=-1).sqrt()
    skipped = standard_deviation <= equal_tolerance
    safe_deviation = torch.where(
        skipped, torch.ones_like(standard_deviation), standard_deviation
    )
    advantages = centered / safe_deviation.unsqueeze(-1)
    advantages = torch.where(
        skipped.unsqueeze(-1), torch.zeros_like(advantages), advantages
    )
    return GroupRelativeRewards(
        advantages=advantages,
        skipped=skipped,
        skip_rate=skipped.to(dtype=torch.float32).mean().item(),
    )


__all__ = [
    "GroupRelativeRewards",
    "dense_box_reward",
    "group_relative_standardize",
    "wer_reward",
]
