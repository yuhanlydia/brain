"""Evaluation metrics for NPP-OPSD neural evidence experiments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


def neural_dependence_gap(correct: Sequence[float] | torch.Tensor, control: Sequence[float] | torch.Tensor) -> float:
    """Return mean correct-condition score minus its paired control score."""
    correct_values = torch.as_tensor(correct, dtype=torch.float64)
    control_values = torch.as_tensor(control, dtype=torch.float64)
    if correct_values.shape != control_values.shape or correct_values.numel() == 0:
        raise ValueError("correct and control must have the same non-empty shape")
    if not torch.isfinite(correct_values).all() or not torch.isfinite(control_values).all():
        raise ValueError("correct and control scores must be finite")
    return (correct_values - control_values).mean().item()


def brier_score(probabilities: torch.Tensor | Sequence[Sequence[float]], targets: torch.Tensor | Sequence[int]) -> float:
    """Return the multiclass Brier score against integer class targets."""
    probabilities, targets = _probability_inputs(probabilities, targets)
    one_hot = torch.nn.functional.one_hot(targets, num_classes=probabilities.shape[1])
    squared_error = (probabilities - one_hot.to(probabilities.dtype)).square().sum(dim=1)
    return squared_error.mean().item()


def expected_calibration_error(
    probabilities: torch.Tensor | Sequence[Sequence[float]],
    targets: torch.Tensor | Sequence[int],
    *,
    bins: int = 10,
) -> float:
    """Return equal-width confidence-bin expected calibration error."""
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    probabilities, targets = _probability_inputs(probabilities, targets)
    confidence, predictions = probabilities.max(dim=1)
    correct = predictions.eq(targets).to(probabilities.dtype)
    error = torch.zeros((), dtype=probabilities.dtype)
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        if index == 0:
            in_bin = (confidence >= lower) & (confidence <= upper)
        else:
            in_bin = (confidence > lower) & (confidence <= upper)
        if in_bin.any():
            bin_weight = in_bin.to(probabilities.dtype).mean()
            error = error + bin_weight * (correct[in_bin].mean() - confidence[in_bin].mean()).abs()
    return error.item()


def topk_coverage(
    scores: torch.Tensor | Sequence[Sequence[float]],
    targets: torch.Tensor | Sequence[int],
    *,
    k: int,
) -> float:
    """Return target coverage at k, treating cutoff ties as covered."""
    scores, targets = _score_inputs(scores, targets)
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= scores.shape[1]:
        raise ValueError("k must be an integer between 1 and the class count")
    cutoff = scores.topk(k, dim=1).values[:, -1]
    target_scores = scores.gather(1, targets[:, None]).squeeze(1)
    return target_scores.ge(cutoff).to(scores.dtype).mean().item()


def cluster_bootstrap_difference(
    correct: Sequence[float] | np.ndarray,
    control: Sequence[float] | np.ndarray,
    image_ids: Sequence[str],
    *,
    seed: int,
    samples: int = 1_000,
) -> tuple[float, np.ndarray]:
    """Bootstrap a paired score difference by resampling whole image clusters."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    correct_values = np.asarray(correct, dtype=float)
    control_values = np.asarray(control, dtype=float)
    if (
        correct_values.ndim != 1
        or control_values.ndim != 1
        or correct_values.shape != control_values.shape
        or correct_values.size == 0
    ):
        raise ValueError("correct and control must be same-length, non-empty one-dimensional arrays")
    if len(image_ids) != correct_values.size:
        raise ValueError("image_ids must have one ID for each paired score")
    if not np.isfinite(correct_values).all() or not np.isfinite(control_values).all():
        raise ValueError("correct and control scores must be finite")
    if any(not isinstance(image_id, str) or not image_id for image_id in image_ids):
        raise ValueError("image_ids must be non-empty strings")

    differences = correct_values - control_values
    clusters: dict[str, np.ndarray] = {}
    for image_id in sorted(set(image_ids)):
        clusters[image_id] = np.flatnonzero(np.asarray(image_ids) == image_id)
    cluster_ids = tuple(clusters)
    generator = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for draw_index in range(samples):
        sampled_ids = generator.choice(cluster_ids, size=len(cluster_ids), replace=True)
        sampled_indices = np.concatenate([clusters[image_id] for image_id in sampled_ids])
        draws[draw_index] = differences[sampled_indices].mean()
    return float(differences.mean()), draws


def _probability_inputs(
    probabilities: torch.Tensor | Sequence[Sequence[float]],
    targets: torch.Tensor | Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    score_values, target_values = _score_inputs(probabilities, targets)
    if (score_values < 0).any() or (score_values > 1).any():
        raise ValueError("probabilities must have entries in [0, 1]")
    if not torch.allclose(
        score_values.sum(dim=1),
        torch.ones(score_values.shape[0], dtype=score_values.dtype),
        rtol=0,
        atol=1e-6,
    ):
        raise ValueError("probability rows must sum to 1 within tolerance 1e-6")
    return score_values, target_values


def _score_inputs(
    probabilities: torch.Tensor | Sequence[Sequence[float]],
    targets: torch.Tensor | Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    score_values = torch.as_tensor(probabilities, dtype=torch.float64)
    target_values = torch.as_tensor(targets)
    if score_values.ndim != 2 or score_values.shape[0] == 0 or score_values.shape[1] < 2:
        raise ValueError("probabilities must have shape [examples, classes] with at least two classes")
    if target_values.ndim != 1 or target_values.shape[0] != score_values.shape[0]:
        raise ValueError("targets must have shape [examples]")
    if target_values.dtype == torch.bool or target_values.is_floating_point():
        raise ValueError("targets must contain integer class indices")
    target_values = target_values.to(torch.long)
    if (target_values < 0).any() or (target_values >= score_values.shape[1]).any():
        raise ValueError("targets contain a class index outside probabilities")
    if not torch.isfinite(score_values).all():
        raise ValueError("probabilities must be finite")
    return score_values, target_values
