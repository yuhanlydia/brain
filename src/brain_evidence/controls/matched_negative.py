"""Deterministic within-session matched-negative sampling."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from ..pronunciation import Pronunciation, canonicalize_pronunciation


def _require_nonempty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{name} must not have leading or trailing whitespace")


@dataclass(frozen=True)
class TrialMetadata:
    """Metadata needed to decide whether two neural trials are comparable."""

    trial_id: str
    subject_id: str
    session_id: str
    split: str
    fold_id: str
    pronunciation: Pronunciation
    duration: float
    reliability_stratum: str

    def __post_init__(self) -> None:
        for name in (
            "trial_id",
            "subject_id",
            "session_id",
            "split",
            "fold_id",
            "reliability_stratum",
        ):
            _require_nonempty_string(getattr(self, name), name)
        canonicalize_pronunciation(self.pronunciation)
        if isinstance(self.duration, bool):
            raise TypeError("duration must be a finite number")
        try:
            finite_duration = math.isfinite(self.duration)
        except TypeError as error:
            raise TypeError("duration must be a finite number") from error
        if not finite_duration:
            raise ValueError("duration must be finite")


@dataclass(frozen=True)
class MatchedNegativeResult:
    """Indices selected from the input sequence and sampler diagnostics."""

    indices: tuple[int, ...]
    requested_count: int
    eligible_count: int

    @property
    def shortfall(self) -> int:
        """Number of requested negatives that could not be sampled safely."""

        return self.requested_count - len(self.indices)

    @property
    def complete(self) -> bool:
        """Whether the sampler satisfied the full request."""

        return self.shortfall == 0

    @property
    def valid_fraction(self) -> float:
        """Fraction of the requested slots filled with valid negatives."""

        if self.requested_count == 0:
            return 1.0
        return len(self.indices) / self.requested_count


@dataclass(frozen=True)
class AnchorLocalControlPair:
    """One anchor-local negative-pronunciation/control-brain construction."""

    anchor_index: int
    negative_index: int
    control_brain_index: int
    anchor: TrialMetadata
    negative: TrialMetadata
    control_brain: TrialMetadata


@dataclass(frozen=True)
class AnchorLocalControlResult:
    """Valid anchor-local pairs plus effective-K and sampling provenance."""

    pairs: tuple[AnchorLocalControlPair, ...]
    anchor_index: int
    requested_count: int
    eligible_pronunciation_count: int
    seed: int
    duration_tolerance: float
    confirmatory: bool

    @property
    def effective_k(self) -> int:
        return len(self.pairs)

    @property
    def shortfall(self) -> int:
        return self.requested_count - self.effective_k

    @property
    def complete(self) -> bool:
        return self.shortfall == 0

    @property
    def should_skip(self) -> bool:
        """Whether confirmatory analysis must skip this incomplete anchor."""

        return self.confirmatory and not self.complete

    @property
    def valid_fraction(self) -> float:
        if self.requested_count == 0:
            return 1.0
        return self.effective_k / self.requested_count


def sample_matched_negatives(
    trials: Sequence[TrialMetadata],
    *,
    anchor_index: int,
    k: int,
    duration_tolerance: float,
    seed: int,
    prefer_phonetic_hard_negatives: bool = False,
) -> MatchedNegativeResult:
    """Sample distinct, constraint-preserving negatives for one anchor.

    ``duration_tolerance`` is an absolute tolerance expressed in the same units
    as ``TrialMetadata.duration``. In hard-negative mode candidates with the
    smallest normalized edit distance to the anchor pronunciation are chosen
    first; the seed deterministically resolves equal-similarity ties.
    """

    _validate_request(
        trials=trials,
        anchor_index=anchor_index,
        k=k,
        duration_tolerance=duration_tolerance,
        seed=seed,
    )
    anchor = trials[anchor_index]
    anchor_pronunciation = canonicalize_pronunciation(anchor.pronunciation)
    eligible = [
        index
        for index, candidate in enumerate(trials)
        if index != anchor_index
        and candidate.subject_id == anchor.subject_id
        and candidate.session_id == anchor.session_id
        and candidate.split == anchor.split
        and candidate.fold_id == anchor.fold_id
        and candidate.reliability_stratum == anchor.reliability_stratum
        and canonicalize_pronunciation(candidate.pronunciation) != anchor_pronunciation
        and abs(candidate.duration - anchor.duration) <= duration_tolerance
    ]

    rng = random.Random(seed)
    groups = _group_by_pronunciation(trials, eligible)
    pronunciation_keys = list(groups)
    count = min(k, len(pronunciation_keys))
    if prefer_phonetic_hard_negatives:
        rng.shuffle(pronunciation_keys)
        pronunciation_keys.sort(
            key=lambda key: _phonetic_similarity(
                anchor.pronunciation,
                key,
            ),
            reverse=True,
        )
        selected_keys = pronunciation_keys[:count]
    else:
        selected_keys = rng.sample(pronunciation_keys, count)
    selected = [rng.choice(groups[key]) for key in selected_keys]

    return MatchedNegativeResult(
        indices=tuple(selected),
        requested_count=k,
        eligible_count=len(pronunciation_keys),
    )


def sample_anchor_local_controls(
    trials: Sequence[TrialMetadata],
    *,
    anchor_index: int,
    k: int,
    duration_tolerance: float,
    seed: int,
    prefer_phonetic_hard_negatives: bool = False,
    confirmatory: bool = False,
) -> AnchorLocalControlResult:
    """Construct anchor-local factorial controls without matched-cell leakage.

    For every negative pronunciation trial ``j``, the control brain trial
    ``c`` has a true pronunciation distinct from both the anchor's ``p_i`` and
    the negative's ``p_j``. Effective K counts distinct negative
    pronunciations, not merely distinct trial rows.
    """

    _validate_request(
        trials=trials,
        anchor_index=anchor_index,
        k=k,
        duration_tolerance=duration_tolerance,
        seed=seed,
    )
    if not isinstance(confirmatory, bool):
        raise TypeError("confirmatory must be a bool")
    if confirmatory and k < 4:
        raise ValueError("confirmatory sampling requires at least 4 negatives")

    anchor = trials[anchor_index]
    anchor_pronunciation = canonicalize_pronunciation(anchor.pronunciation)
    candidate_indices = [
        index
        for index, candidate in enumerate(trials)
        if index != anchor_index
        and candidate.subject_id == anchor.subject_id
        and candidate.session_id == anchor.session_id
        and candidate.split == anchor.split
        and candidate.fold_id == anchor.fold_id
        and candidate.reliability_stratum == anchor.reliability_stratum
        and canonicalize_pronunciation(candidate.pronunciation) != anchor_pronunciation
        and abs(candidate.duration - anchor.duration) <= duration_tolerance
    ]
    candidate_groups = _group_by_pronunciation(trials, candidate_indices)
    viable_groups = {
        pronunciation: indices
        for pronunciation, indices in candidate_groups.items()
        if _control_candidates(
            trials,
            anchor=anchor,
            anchor_index=anchor_index,
            negative_pronunciation=pronunciation,
            duration_tolerance=duration_tolerance,
        )
    }

    rng = random.Random(seed)
    pronunciation_keys = list(viable_groups)
    selection_count = min(k, len(pronunciation_keys))
    if prefer_phonetic_hard_negatives:
        rng.shuffle(pronunciation_keys)
        pronunciation_keys.sort(
            key=lambda key: _phonetic_similarity(anchor.pronunciation, key),
            reverse=True,
        )
        selected_pronunciations = pronunciation_keys[:selection_count]
    else:
        selected_pronunciations = rng.sample(
            pronunciation_keys,
            selection_count,
        )

    pairs: list[AnchorLocalControlPair] = []
    for negative_pronunciation in selected_pronunciations:
        negative_index = rng.choice(viable_groups[negative_pronunciation])
        control_index = rng.choice(
            _control_candidates(
                trials,
                anchor=anchor,
                anchor_index=anchor_index,
                negative_pronunciation=negative_pronunciation,
                duration_tolerance=duration_tolerance,
            )
        )
        pairs.append(
            AnchorLocalControlPair(
                anchor_index=anchor_index,
                negative_index=negative_index,
                control_brain_index=control_index,
                anchor=anchor,
                negative=trials[negative_index],
                control_brain=trials[control_index],
            )
        )

    return AnchorLocalControlResult(
        pairs=tuple(pairs),
        anchor_index=anchor_index,
        requested_count=k,
        eligible_pronunciation_count=len(viable_groups),
        seed=seed,
        duration_tolerance=duration_tolerance,
        confirmatory=confirmatory,
    )


def _validate_request(
    *,
    trials: Sequence[TrialMetadata],
    anchor_index: int,
    k: int,
    duration_tolerance: float,
    seed: int,
) -> None:
    if any(not isinstance(trial, TrialMetadata) for trial in trials):
        raise TypeError("trials must contain TrialMetadata values")
    if not isinstance(anchor_index, int) or isinstance(anchor_index, bool):
        raise TypeError("anchor_index must be an integer")
    if not 0 <= anchor_index < len(trials):
        raise IndexError("anchor_index is outside the trial sequence")
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError("k must be an integer")
    if k < 0:
        raise ValueError("k must be non-negative")
    if not math.isfinite(duration_tolerance) or duration_tolerance < 0:
        raise ValueError("duration_tolerance must be finite and non-negative")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if not math.isfinite(trials[anchor_index].duration):
        raise ValueError("anchor duration must be finite")
    if any(not math.isfinite(trial.duration) for trial in trials):
        raise ValueError("trial durations must be finite")
    if any(
        not isinstance(trial.reliability_stratum, str) or not trial.reliability_stratum
        for trial in trials
    ):
        raise ValueError("trial reliability_stratum values must be non-empty strings")
    normalized_trial_ids = [trial.trial_id.strip() for trial in trials]
    if len(normalized_trial_ids) != len(set(normalized_trial_ids)):
        raise ValueError("trials must not contain duplicate trial_id provenance")
    for trial in trials:
        canonicalize_pronunciation(trial.pronunciation)


def _group_by_pronunciation(
    trials: Sequence[TrialMetadata],
    indices: Sequence[int],
) -> dict[tuple[str, ...], list[int]]:
    groups: dict[tuple[str, ...], list[int]] = {}
    for index in indices:
        key = canonicalize_pronunciation(trials[index].pronunciation)
        groups.setdefault(key, []).append(index)
    return groups


def _control_candidates(
    trials: Sequence[TrialMetadata],
    *,
    anchor: TrialMetadata,
    anchor_index: int,
    negative_pronunciation: tuple[str, ...],
    duration_tolerance: float,
) -> list[int]:
    anchor_pronunciation = canonicalize_pronunciation(anchor.pronunciation)
    return [
        index
        for index, candidate in enumerate(trials)
        if index != anchor_index
        and candidate.subject_id == anchor.subject_id
        and candidate.session_id == anchor.session_id
        and candidate.split == anchor.split
        and candidate.fold_id == anchor.fold_id
        and candidate.reliability_stratum == anchor.reliability_stratum
        and canonicalize_pronunciation(candidate.pronunciation)
        not in {anchor_pronunciation, negative_pronunciation}
        and abs(candidate.duration - anchor.duration) <= duration_tolerance
    ]


def _phonetic_similarity(left: Pronunciation, right: Pronunciation) -> float:
    left_tokens = canonicalize_pronunciation(left)
    right_tokens = canonicalize_pronunciation(right)
    scale = max(len(left_tokens), len(right_tokens))
    if scale == 0:
        return 1.0
    return 1.0 - _edit_distance(left_tokens, right_tokens) / scale


def _edit_distance(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]
