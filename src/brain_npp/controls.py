"""Leakage-safe matched-wrong mappings and brain control transforms."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence

import torch

from .data import TrialRecord, validate_manifest


def build_block_derangement(
    records: Sequence[TrialRecord],
    seed: int,
    *,
    preserve_run: bool = True,
    adjacency_radius: int = 0,
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Match trials only within subject/session (and optionally run) strata.

    Every mapping is a one-to-one matched-wrong assignment.  A trial cannot be
    paired with itself, another presentation of the same image, or a trial at
    most ``adjacency_radius`` positions away.  Trials that cannot be assigned
    in their own stratum are returned explicitly in ``unmatched_ids``.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not isinstance(preserve_run, bool):
        raise ValueError("preserve_run must be a boolean")
    if (
        isinstance(adjacency_radius, bool)
        or not isinstance(adjacency_radius, int)
        or adjacency_radius < 0
    ):
        raise ValueError("adjacency_radius must be a non-negative integer")

    validated = validate_manifest(records)
    blocks: dict[tuple[str, ...], list[TrialRecord]] = defaultdict(list)
    for record in validated:
        key = (record.subject_id, record.session_id)
        if preserve_run:
            key += (record.run_id,)
        blocks[key].append(record)

    generator = random.Random(seed)
    mapping: dict[str, str] = {}
    unmatched: list[str] = []
    for key in sorted(blocks):
        block_mapping, block_unmatched = _match_block(
            blocks[key], adjacency_radius, generator
        )
        mapping.update(block_mapping)
        unmatched.extend(block_unmatched)
    return mapping, tuple(unmatched)


def apply_brain_control(
    brain: torch.Tensor,
    control: str,
    *,
    mapping: Mapping[str, str] | None = None,
    trial_ids: Sequence[str] | None = None,
    seed: int = 0,
) -> torch.Tensor:
    """Return a deterministic control version of normalized trial brain data.

    ``shuffle``/``matched-wrong`` and ``wrong-subject`` require an externally
    precomputed ID mapping so that matching is auditable before tensor access.
    ``covariance-noise`` draws empirical-covariance noise from the supplied,
    normalized trial tensor without fitting a latent subspace.
    """
    if not isinstance(brain, torch.Tensor) or brain.ndim < 2:
        raise ValueError("brain must be a tensor with trials on its first dimension")
    if control == "correct":
        return brain
    if control == "zero":
        return torch.zeros_like(brain)
    if control == "covariance-noise":
        return _covariance_noise(brain, seed)
    if control not in {"shuffle", "matched-wrong", "wrong-subject"}:
        raise ValueError(f"unknown brain control {control!r}")
    if mapping is None or trial_ids is None:
        raise ValueError(f"{control} requires a precomputed mapping and trial_ids")
    return _mapped_brain(brain, mapping, trial_ids)


def _match_block(
    block: Sequence[TrialRecord], adjacency_radius: int, generator: random.Random
) -> tuple[dict[str, str], list[str]]:
    by_id = {record.trial_id: record for record in block}
    sources = list(by_id)
    generator.shuffle(sources)
    candidates: dict[str, list[str]] = {}
    for source_id in sources:
        source = by_id[source_id]
        choices = [
            candidate_id
            for candidate_id, candidate in by_id.items()
            if candidate_id != source_id
            and candidate.image_id != source.image_id
            and abs(candidate.trial_index - source.trial_index) > adjacency_radius
        ]
        generator.shuffle(choices)
        candidates[source_id] = choices

    target_to_source: dict[str, str] = {}

    def assign(source_id: str, visited: set[str]) -> bool:
        for target_id in candidates[source_id]:
            if target_id in visited:
                continue
            visited.add(target_id)
            occupying_source = target_to_source.get(target_id)
            if occupying_source is None or assign(occupying_source, visited):
                target_to_source[target_id] = source_id
                return True
        return False

    for source_id in sources:
        assign(source_id, set())
    mapping = {source_id: target_id for target_id, source_id in target_to_source.items()}
    if len(mapping) != len(block) or set(mapping) != set(mapping.values()):
        return {}, [record.trial_id for record in block]
    return mapping, []


def _mapped_brain(
    brain: torch.Tensor, mapping: Mapping[str, str], trial_ids: Sequence[str]
) -> torch.Tensor:
    if len(trial_ids) != brain.shape[0]:
        raise ValueError("trial_ids must have one ID for each brain trial")
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError("trial_ids must be unique")
    index_by_id = {trial_id: index for index, trial_id in enumerate(trial_ids)}
    matched_indices: list[int] = []
    for trial_id in trial_ids:
        matched_id = mapping.get(trial_id)
        if matched_id is None:
            raise ValueError(f"missing matched-wrong reference for {trial_id!r}")
        if matched_id not in index_by_id:
            raise ValueError(f"matched-wrong reference {matched_id!r} does not exist")
        matched_indices.append(index_by_id[matched_id])
    return brain.index_select(
        0, torch.tensor(matched_indices, device=brain.device, dtype=torch.long)
    )


def _covariance_noise(brain: torch.Tensor, seed: int) -> torch.Tensor:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not brain.is_floating_point():
        raise ValueError("covariance-noise requires a floating-point brain tensor")
    if brain.shape[0] < 2:
        raise ValueError("covariance-noise requires at least two trials")
    flattened = brain.reshape(brain.shape[0], -1)
    centered = flattened - flattened.mean(dim=0, keepdim=True)
    generator = torch.Generator(device=brain.device)
    generator.manual_seed(seed)
    weights = torch.randn(
        (brain.shape[0], brain.shape[0]),
        dtype=brain.dtype,
        device=brain.device,
        generator=generator,
    )
    noise = weights.matmul(centered) / (brain.shape[0] - 1) ** 0.5
    return noise.reshape_as(brain)
