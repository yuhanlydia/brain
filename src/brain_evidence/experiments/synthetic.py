"""Deterministic CPU smoke experiments for the public training primitives.

These experiments prove that the code paths execute and that target-side
gradients remain detached.  They use invented tensors and are deliberately not
reported as benchmark evidence.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from brain_evidence.gates import legacy_dual_cosine_gate
from brain_evidence.objectives import (
    forward_kl,
    group_relative_standardize,
    wer_reward,
)
from brain_evidence.recoverability import (
    AnchorLocalValidity,
    NRAEvidence,
    NRAInteractionBatch,
    aggregate_interactions,
    anchor_local_interaction,
    fisher_center,
    fisher_inner,
    fisher_norm_sq,
    recoverable_target,
)

SYNTHETIC_METHODS = (
    "ce",
    "vanilla_opsd",
    "legacy_dual_cosine",
    "nra_opsd",
    "rlvr",
)

_DTYPE = torch.float64
_VOCABULARY_SIZE = 4
_IDENTITY_COUNT = 101
_CURRENT_SUPPORTED_IDENTITY = 0
_COPIER_ONLY_IDENTITY = 3
_CONTROL_IDENTITIES = (99, 100)
_START_TOKEN_ID = 0
_NULL_ALPHA = 0.05
_NULL_PERMUTATION_COUNT = 99
_NULL_PRONUNCIATION_COUNT = 99
_NULL_SAMPLER = "synthetic-anchor-local-v1"
_STATISTIC_EPSILON = 1e-8
_AGGREGATION_POLICY = "mean"
_WEIGHT_POLICY = "uniform"
_PREFIX_POLICY = "synthetic-student-rollout-v1"
_ALIGNMENT_POLICY = "synthetic-token-index-v1"
_SUPPORT_POLICY = "synthetic-full-vocabulary-v1"
_CHECKPOINT_ID = "synthetic-frozen-ce-v1"
_SUBJECT_ID = "synthetic-subject-0"
_TRAIN_SPLIT = "train"
_RELIABILITY_STRATUM = "synthetic-high-reliability"
_CURRENT_FOLD_ID = "synthetic-current-fold"
_CURRENT_SESSION_ID = "synthetic-current-session"
_CALIBRATION_FOLD_ID = "synthetic-calibration-fold"
_CALIBRATION_SESSION_ID = "synthetic-calibration-session"


class _TinyStudent(nn.Module):
    """A minimal brain-and-prefix model whose only parameters are trainable."""

    def __init__(self) -> None:
        super().__init__()
        self.brain_head = nn.Linear(
            _IDENTITY_COUNT,
            _VOCABULARY_SIZE,
            bias=False,
            dtype=_DTYPE,
        )
        self.prefix_embedding = nn.Embedding(
            _VOCABULARY_SIZE,
            _VOCABULARY_SIZE,
            dtype=_DTYPE,
        )

    def initialize(self, generator: torch.Generator) -> None:
        with torch.no_grad():
            self.brain_head.weight.normal_(std=0.04, generator=generator)
            self.prefix_embedding.weight.normal_(std=0.04, generator=generator)

    def student_logits(self, brain: Tensor, prefix: Tensor) -> Tensor:
        return self.brain_head(brain).unsqueeze(1) + self.prefix_embedding(prefix)


class _FrozenTeacher(nn.Module):
    """A frozen CE copy plus controlled privilege effects.

    Privilege adds three separable ingredients: a brain-independent copier, a
    brain-modulated copier nuisance, and an effect present only when the brain's
    encoded pronunciation matches the privileged pronunciation.
    """

    def __init__(self, student: _TinyStudent) -> None:
        super().__init__()
        self.base = copy.deepcopy(student)
        self.base.requires_grad_(False)
        copy_direction = torch.tensor([0.0, 0.0, 1.0, -1.0], dtype=_DTYPE)
        copy_coefficients = torch.tensor(
            [(1.0, -1.0, -1.0)[identity % 3] for identity in range(_IDENTITY_COUNT)],
            dtype=_DTYPE,
        )
        self.register_buffer(
            "copy_vectors",
            copy_coefficients.unsqueeze(-1) * copy_direction,
        )
        matching_vectors = torch.zeros(
            (_IDENTITY_COUNT, _VOCABULARY_SIZE),
            dtype=_DTYPE,
        )
        matching_vectors[_CURRENT_SUPPORTED_IDENTITY] = torch.tensor(
            [1.25, -1.25, 0.0, 0.0],
            dtype=_DTYPE,
        )
        self.register_buffer(
            "matching_vectors",
            matching_vectors,
        )
        brain_copy_gains = torch.full((_IDENTITY_COUNT,), 0.45, dtype=_DTYPE)
        brain_copy_gains[_CONTROL_IDENTITIES[0]] = -0.35
        brain_copy_gains[_CONTROL_IDENTITIES[1]] = -0.15
        self.register_buffer(
            "brain_copy_gains",
            brain_copy_gains,
        )

    def teacher_logits(
        self,
        brain: Tensor,
        privilege: Tensor,
        prefix: Tensor,
    ) -> Tensor:
        base_logits = self.base.student_logits(brain, prefix)
        copied = self.copy_vectors[privilege]
        gain = brain @ self.brain_copy_gains
        logits = base_logits + 0.6 * copied.unsqueeze(1)
        logits = logits + gain[:, None, None] * copied.unsqueeze(1)
        matching = brain.gather(1, privilege.unsqueeze(1)).squeeze(1)
        matched_effect = matching.unsqueeze(-1) * self.matching_vectors[privilege]
        return logits + matched_effect.unsqueeze(1)


class _FrozenCrossfitBrainSupport(nn.Module):
    """Fixed brain-only logits standing in for a held-out CE checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        support = torch.zeros(
            (_IDENTITY_COUNT, _VOCABULARY_SIZE),
            dtype=_DTYPE,
        )
        support[_CURRENT_SUPPORTED_IDENTITY] = torch.tensor(
            [1.0, -1.0, 0.0, 0.0],
            dtype=_DTYPE,
        )
        support[_COPIER_ONLY_IDENTITY] = torch.tensor(
            [1.0, -1.0, 0.0, 0.0],
            dtype=_DTYPE,
        )
        self.register_buffer(
            "support_by_pronunciation",
            support,
        )

    def log_probs(self, brain: Tensor, prefix: Tensor) -> Tensor:
        logits = brain @ self.support_by_pronunciation
        return torch.log_softmax(
            logits.unsqueeze(1).expand(-1, prefix.shape[1], -1),
            dim=-1,
        ).detach()


@dataclass(frozen=True)
class _SyntheticPronunciationRecord:
    """An immutable identity-to-label entry shared by all synthetic paths."""

    identity: int
    pronunciation_id: str

    def __post_init__(self) -> None:
        if not 0 <= self.identity < _IDENTITY_COUNT:
            raise ValueError("pronunciation identity is outside the registered table")
        if not self.pronunciation_id:
            raise ValueError("pronunciation_id must be non-empty")


@dataclass(frozen=True)
class _SyntheticProvenanceRecord:
    """A concrete calibration/current record identity."""

    subject_id: str
    split: str
    fold_id: str
    session_id: str
    trial_id: str
    reliability_stratum: str

    def __post_init__(self) -> None:
        for name in (
            "subject_id",
            "split",
            "fold_id",
            "session_id",
            "trial_id",
            "reliability_stratum",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if self.split != _TRAIN_SPLIT:
            raise ValueError("split must be exactly 'train'")


@dataclass(frozen=True)
class _SyntheticPronunciationDonorRecord(_SyntheticProvenanceRecord):
    """A calibration-only pronunciation label with source provenance."""

    pronunciation: _SyntheticPronunciationRecord

    @property
    def identity(self) -> int:
        return self.pronunciation.identity

    @property
    def pronunciation_id(self) -> str:
        return self.pronunciation.pronunciation_id


@dataclass(frozen=True)
class _SyntheticTrialRecord(_SyntheticPronunciationDonorRecord):
    """A provenance record that also carries a scored brain tensor."""

    brain: Tensor

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.brain.ndim != 2 or self.brain.shape[1] != _IDENTITY_COUNT:
            raise ValueError(f"brain must have shape [batch, {_IDENTITY_COUNT}]")
        if not self.brain.is_floating_point() or not bool(
            torch.isfinite(self.brain).all()
        ):
            raise ValueError("brain must be a finite floating-point tensor")


@dataclass(frozen=True)
class _SyntheticCalibrationRecords:
    """Three held-out neural trials plus 99 pronunciation-label donors."""

    control_records: tuple[_SyntheticTrialRecord, _SyntheticTrialRecord]
    pronunciation_donors: tuple[_SyntheticPronunciationDonorRecord, ...]

    def __post_init__(self) -> None:
        if len(self.control_records) != 2:
            raise ValueError("calibration requires exactly two control records")
        if len(self.pronunciation_donors) != _NULL_PRONUNCIATION_COUNT:
            raise ValueError("calibration requires exactly 99 pronunciation donors")
        donor_identities = tuple(donor.identity for donor in self.pronunciation_donors)
        if set(donor_identities) != set(range(_NULL_PRONUNCIATION_COUNT)):
            raise ValueError(
                "calibration donor identities must be exactly 0 through 98"
            )
        neural_donors = tuple(
            donor
            for donor in self.pronunciation_donors
            if isinstance(donor, _SyntheticTrialRecord)
        )
        if (
            len(neural_donors) != 1
            or neural_donors[0] is not self.pronunciation_donors[0]
        ):
            raise ValueError("only the first donor may carry the calibration brain")
        if (
            tuple(record.pronunciation.identity for record in self.control_records)
            != _CONTROL_IDENTITIES
        ):
            raise ValueError("calibration control identities must be 99 and 100")
        for field in (
            "subject_id",
            "split",
            "fold_id",
            "session_id",
            "reliability_stratum",
        ):
            values = {getattr(record, field) for record in self.contributor_records}
            if len(values) != 1:
                raise ValueError(f"calibration contributors must share {field}")
        trial_ids = tuple(record.trial_id for record in self.contributor_records)
        if len(trial_ids) != len(set(trial_ids)):
            raise ValueError("calibration trial_id values must be distinct")

    @property
    def anchor_record(self) -> _SyntheticTrialRecord:
        anchor = self.pronunciation_donors[0]
        if not isinstance(anchor, _SyntheticTrialRecord):
            raise TypeError("validated calibration anchor must carry brain data")
        return anchor

    @property
    def neural_trials(
        self,
    ) -> tuple[_SyntheticTrialRecord, _SyntheticTrialRecord, _SyntheticTrialRecord]:
        return (self.anchor_record, *self.control_records)

    @property
    def contributor_records(self) -> tuple[_SyntheticProvenanceRecord, ...]:
        return (*self.pronunciation_donors, *self.control_records)


@dataclass(frozen=True)
class _SyntheticFixture:
    seed: int
    student: _TinyStudent
    teacher: _FrozenTeacher
    brain_support: _FrozenCrossfitBrainSupport
    pronunciations: tuple[_SyntheticPronunciationRecord, ...]
    current_anchor_record: _SyntheticTrialRecord
    current_control_records: tuple[_SyntheticTrialRecord, _SyntheticTrialRecord]
    calibration_records: _SyntheticCalibrationRecords
    negative_pronunciations: tuple[
        _SyntheticPronunciationRecord,
        _SyntheticPronunciationRecord,
    ]
    prefix: Tensor
    labels: Tensor

    def __post_init__(self) -> None:
        if len(self.pronunciations) != _IDENTITY_COUNT:
            raise ValueError("fixture requires the complete 101-identity table")
        if tuple(record.identity for record in self.pronunciations) != tuple(
            range(_IDENTITY_COUNT)
        ):
            raise ValueError(
                "pronunciation table identities must be ordered 0 through 100"
            )
        if len(self.current_control_records) != 2:
            raise ValueError("fixture requires exactly two current control records")
        if (
            tuple(
                record.pronunciation.identity for record in self.current_control_records
            )
            != _CONTROL_IDENTITIES
        ):
            raise ValueError("current control identities must be 99 and 100")
        if tuple(
            record.pronunciation.identity
            for record in self.calibration_records.control_records
        ) != tuple(
            record.pronunciation.identity for record in self.current_control_records
        ):
            raise ValueError("current and calibration control identities must match")
        if (
            self.calibration_records.anchor_record.pronunciation.identity
            != self.current_anchor_record.pronunciation.identity
        ):
            raise ValueError("current and calibration anchor identities must match")
        current_records = (self.current_anchor_record, *self.current_control_records)
        calibration_records = self.calibration_records.contributor_records
        for field in (
            "subject_id",
            "split",
            "fold_id",
            "session_id",
            "reliability_stratum",
        ):
            current_values = {getattr(record, field) for record in current_records}
            if len(current_values) != 1:
                raise ValueError(f"current records must share {field}")
        current_trial_ids = tuple(record.trial_id for record in current_records)
        if len(current_trial_ids) != len(set(current_trial_ids)):
            raise ValueError("current trial_id values must be distinct")
        for field in ("subject_id", "split", "reliability_stratum"):
            if getattr(current_records[0], field) != getattr(
                calibration_records[0], field
            ):
                raise ValueError(f"current and calibration records must share {field}")
        for identifier in ("fold_id", "session_id", "trial_id"):
            current_ids = {getattr(record, identifier) for record in current_records}
            calibration_ids = {
                getattr(record, identifier) for record in calibration_records
            }
            if current_ids & calibration_ids:
                raise ValueError(
                    f"{identifier} sets overlap across current/calibration"
                )
        current_storages = {
            record.brain.untyped_storage().data_ptr() for record in current_records
        }
        calibration_storages = {
            record.brain.untyped_storage().data_ptr()
            for record in self.calibration_records.neural_trials
        }
        if current_storages & calibration_storages:
            raise ValueError("brain storage sets overlap across current/calibration")

    @property
    def anchor_brain(self) -> Tensor:
        return self.current_anchor_record.brain

    @property
    def control_brains(self) -> tuple[Tensor, Tensor]:
        return tuple(record.brain for record in self.current_control_records)

    @property
    def anchor_privilege(self) -> Tensor:
        return _pronunciation_batch(
            self.current_anchor_record.pronunciation,
            self.anchor_brain,
        )

    @property
    def negative_privileges(self) -> tuple[Tensor, Tensor]:
        return tuple(
            _pronunciation_batch(pronunciation, self.anchor_brain)
            for pronunciation in self.negative_pronunciations
        )

    @property
    def interaction_validities(
        self,
    ) -> tuple[AnchorLocalValidity, AnchorLocalValidity]:
        return tuple(
            AnchorLocalValidity(
                anchor_pronunciation=self.pronunciation_id,
                negative_pronunciation=negative.pronunciation_id,
                control_pronunciation=control.pronunciation.pronunciation_id,
            )
            for control, negative in zip(
                self.current_control_records,
                self.negative_pronunciations,
                strict=True,
            )
        )

    @property
    def subject_id(self) -> str:
        return self.current_anchor_record.subject_id

    @property
    def pronunciation_id(self) -> str:
        return self.current_anchor_record.pronunciation.pronunciation_id

    @property
    def reliability_stratum(self) -> str:
        return self.current_anchor_record.reliability_stratum


@dataclass(frozen=True)
class _SyntheticNullCalibration:
    """Deterministic held-out cyclic-permutation calibration artifact."""

    threshold: Tensor
    statistics: Tensor
    permutations: Tensor
    anchor_match_draw_count: int
    any_match_draw_count: int
    pairing_broken_draw_count: int
    alpha: float
    permutation_count: int
    seed: int
    split: str
    statistic_id: str
    statistic_epsilon: float
    phase: Literal["pilot"]
    k: int
    sampler: str
    aggregation_policy: str
    weight_policy: str
    checkpoint_id: str
    prefix_policy: str
    alignment_policy: str
    support_policy: str
    producer_fold_ids: tuple[str, ...]
    producer_session_ids: tuple[str, ...]
    producer_trial_ids: tuple[str, ...]


def _pronunciation_batch(
    pronunciation: _SyntheticPronunciationRecord,
    brain: Tensor,
) -> Tensor:
    return torch.full(
        (brain.shape[0],),
        pronunciation.identity,
        dtype=torch.long,
        device=brain.device,
    )


def _brain_for_identity(identity: int, *, batch_size: int) -> Tensor:
    brain = torch.zeros((batch_size, _IDENTITY_COUNT), dtype=_DTYPE)
    brain[:, identity] = 1.0
    return brain


def _trial_record(
    *,
    role: str,
    cohort: Literal["current", "calibration"],
    pronunciation: _SyntheticPronunciationRecord,
    batch_size: int,
) -> _SyntheticTrialRecord:
    fold_id = _CURRENT_FOLD_ID if cohort == "current" else _CALIBRATION_FOLD_ID
    session_id = _CURRENT_SESSION_ID if cohort == "current" else _CALIBRATION_SESSION_ID
    return _SyntheticTrialRecord(
        brain=_brain_for_identity(pronunciation.identity, batch_size=batch_size),
        pronunciation=pronunciation,
        subject_id=_SUBJECT_ID,
        split=_TRAIN_SPLIT,
        fold_id=fold_id,
        session_id=session_id,
        trial_id=f"synthetic-{cohort}-trial-{role}",
        reliability_stratum=_RELIABILITY_STRATUM,
    )


def _donor_record(
    *,
    pronunciation: _SyntheticPronunciationRecord,
) -> _SyntheticPronunciationDonorRecord:
    role = f"donor-{pronunciation.identity:03d}"
    return _SyntheticPronunciationDonorRecord(
        pronunciation=pronunciation,
        subject_id=_SUBJECT_ID,
        split=_TRAIN_SPLIT,
        fold_id=_CALIBRATION_FOLD_ID,
        session_id=_CALIBRATION_SESSION_ID,
        trial_id=f"synthetic-calibration-trial-{role}",
        reliability_stratum=_RELIABILITY_STRATUM,
    )


def _build_fixture(
    seed: int,
    *,
    anchor_identity: int = _CURRENT_SUPPORTED_IDENTITY,
) -> _SyntheticFixture:
    if anchor_identity not in {_CURRENT_SUPPORTED_IDENTITY, _COPIER_ONLY_IDENTITY}:
        raise ValueError("anchor_identity must select a registered smoke anchor")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    student = _TinyStudent()
    student.initialize(generator)
    teacher = _FrozenTeacher(student)
    brain_support = _FrozenCrossfitBrainSupport()

    pronunciations = tuple(
        _SyntheticPronunciationRecord(
            identity=identity,
            pronunciation_id=f"synthetic-pronunciation-{identity:03d}",
        )
        for identity in range(_IDENTITY_COUNT)
    )
    donor_order = tuple(
        pronunciations[(anchor_identity + offset) % _NULL_PRONUNCIATION_COUNT]
        for offset in range(_NULL_PRONUNCIATION_COUNT)
    )
    batch_size = 2
    current_anchor = _trial_record(
        role="anchor",
        cohort="current",
        pronunciation=pronunciations[anchor_identity],
        batch_size=batch_size,
    )
    current_controls = tuple(
        _trial_record(
            role=f"control-{index}",
            cohort="current",
            pronunciation=pronunciations[identity],
            batch_size=batch_size,
        )
        for index, identity in enumerate(_CONTROL_IDENTITIES)
    )
    calibration_anchor = _trial_record(
        role=f"donor-{anchor_identity:03d}",
        cohort="calibration",
        pronunciation=pronunciations[anchor_identity],
        batch_size=batch_size,
    )
    calibration_controls = tuple(
        _trial_record(
            role=f"control-{index}",
            cohort="calibration",
            pronunciation=pronunciations[identity],
            batch_size=batch_size,
        )
        for index, identity in enumerate(_CONTROL_IDENTITIES)
    )
    calibration_donors = (
        calibration_anchor,
        *tuple(
            _donor_record(pronunciation=pronunciation)
            for pronunciation in donor_order[1:]
        ),
    )
    labels = torch.tensor([[1, 2, 0], [3, 0, 1]], dtype=torch.long)
    start = torch.full(
        (batch_size, 1),
        _START_TOKEN_ID,
        dtype=torch.long,
    )
    prefix = torch.cat((start, labels[:, :-1]), dim=1)
    return _SyntheticFixture(
        seed=seed,
        student=student,
        teacher=teacher,
        brain_support=brain_support,
        pronunciations=pronunciations,
        current_anchor_record=current_anchor,
        current_control_records=current_controls,
        calibration_records=_SyntheticCalibrationRecords(
            control_records=calibration_controls,
            pronunciation_donors=calibration_donors,
        ),
        negative_pronunciations=(donor_order[1], donor_order[2]),
        prefix=prefix,
        labels=labels,
    )


def _teacher_log_probs(
    fixture: _SyntheticFixture,
    brain: Tensor,
    privilege: Tensor,
    prefix: Tensor,
) -> Tensor:
    logits = fixture.teacher.teacher_logits(brain, privilege, prefix)
    return torch.log_softmax(logits, dim=-1).detach()


def _greedy_student_prefix(
    student: _TinyStudent,
    brain: Tensor,
    *,
    token_count: int,
    start_token: int = _START_TOKEN_ID,
) -> Tensor:
    """Generate a detached greedy prefix from the current brain-only student."""

    if token_count <= 0:
        raise ValueError("token_count must be positive")
    if not 0 <= start_token < _VOCABULARY_SIZE:
        raise ValueError("start_token must be in the synthetic vocabulary")
    with torch.no_grad():
        prefix = torch.full(
            (brain.shape[0], 1),
            start_token,
            dtype=torch.long,
            device=brain.device,
        )
        while prefix.shape[1] < token_count:
            next_token = student.student_logits(brain, prefix)[:, -1].argmax(
                dim=-1,
                keepdim=True,
            )
            prefix = torch.cat((prefix, next_token), dim=1)
    return prefix.detach()


def _interactions_from_records(
    fixture: _SyntheticFixture,
    anchor_record: _SyntheticTrialRecord,
    control_records: tuple[_SyntheticTrialRecord, _SyntheticTrialRecord],
    privileged_pronunciation: _SyntheticPronunciationRecord,
    negative_pronunciations: tuple[
        _SyntheticPronunciationRecord,
        _SyntheticPronunciationRecord,
    ],
    student_probs: Tensor,
    prefix: Tensor,
) -> Tensor:
    interactions: list[Tensor] = []
    privileged = _pronunciation_batch(privileged_pronunciation, anchor_record.brain)
    for control_record, negative_pronunciation in zip(
        control_records,
        negative_pronunciations,
        strict=True,
    ):
        negative = _pronunciation_batch(negative_pronunciation, anchor_record.brain)
        xi_pi = _teacher_log_probs(
            fixture,
            anchor_record.brain,
            privileged,
            prefix,
        )
        xi_pj = _teacher_log_probs(
            fixture,
            anchor_record.brain,
            negative,
            prefix,
        )
        xc_pi = _teacher_log_probs(
            fixture,
            control_record.brain,
            privileged,
            prefix,
        )
        xc_pj = _teacher_log_probs(
            fixture,
            control_record.brain,
            negative,
            prefix,
        )
        interactions.append(
            anchor_local_interaction(
                xi_pi,
                xi_pj,
                xc_pi,
                xc_pj,
                student_probs,
                validity=AnchorLocalValidity(
                    anchor_pronunciation=(privileged_pronunciation.pronunciation_id),
                    negative_pronunciation=(negative_pronunciation.pronunciation_id),
                    control_pronunciation=(
                        control_record.pronunciation.pronunciation_id
                    ),
                ),
            )
        )
    return torch.stack(interactions, dim=-2)


def _anchor_interactions(
    fixture: _SyntheticFixture,
    student_probs: Tensor,
    prefix: Tensor,
) -> Tensor:
    return _interactions_from_records(
        fixture,
        fixture.current_anchor_record,
        fixture.current_control_records,
        fixture.current_anchor_record.pronunciation,
        fixture.negative_pronunciations,
        student_probs,
        prefix,
    )


def _signed_fisher_alignment(
    interactions: Tensor,
    brain_only_direction: Tensor,
    student_probs: Tensor,
    *,
    statistic_epsilon: float = _STATISTIC_EPSILON,
) -> Tensor:
    """Compute the registered statistic with the public production geometry."""

    expanded_student_probs = student_probs.unsqueeze(-2).expand_as(interactions)
    centered_interactions = fisher_center(interactions, expanded_student_probs)
    aggregate = aggregate_interactions(
        centered_interactions,
        student_probs,
        method=_AGGREGATION_POLICY,
    )
    centered_direction = fisher_center(brain_only_direction, student_probs)
    numerator = fisher_inner(aggregate, centered_direction, student_probs)
    aggregate_norm_sq = fisher_norm_sq(aggregate, student_probs)
    direction_norm_sq = fisher_norm_sq(centered_direction, student_probs)
    epsilon = student_probs.new_tensor(statistic_epsilon)
    denominator = (
        aggregate_norm_sq.clamp_min(0).sqrt() * direction_norm_sq.clamp_min(0).sqrt()
        + epsilon
    )
    safe_denominator = torch.where(
        denominator > 0,
        denominator,
        torch.ones_like(denominator),
    )
    score = numerator / safe_denominator
    supported = (aggregate_norm_sq > 0) & (direction_norm_sq > 0)
    return (
        torch.where(supported, score, torch.zeros_like(score))
        .clamp(
            min=-1.0,
            max=1.0,
        )
        .detach()
    )


def _permuted_null_interactions(
    fixture: _SyntheticFixture,
    student_probs: Tensor,
    prefix: Tensor,
    permutation: Tensor,
) -> Tensor:
    """Score one donor reassignment on the three held-out neural records."""

    if permutation.shape != (_NULL_PRONUNCIATION_COUNT,):
        raise ValueError("permutation must contain exactly 99 donor identities")
    identities = tuple(int(value) for value in permutation.tolist())
    if set(identities) != set(range(_NULL_PRONUNCIATION_COUNT)):
        raise ValueError("permutation must reorder donor identities 0 through 98")
    donors_by_identity = {
        donor.identity: donor.pronunciation
        for donor in fixture.calibration_records.pronunciation_donors
    }
    assigned_pronunciations = tuple(
        donors_by_identity[identity] for identity in identities[:3]
    )
    return _interactions_from_records(
        fixture,
        fixture.calibration_records.anchor_record,
        fixture.calibration_records.control_records,
        assigned_pronunciations[0],
        (assigned_pronunciations[1], assigned_pronunciations[2]),
        student_probs,
        prefix,
    )


def _calibrate_nra_null(
    fixture: _SyntheticFixture,
    student_probs: Tensor,
    prefix: Tensor,
    brain_only_direction: Tensor,
) -> _SyntheticNullCalibration:
    """Calibrate the exact signed statistic over 99 held-out cyclic assignments.

    The three scored brain tensors are held-out records; 99 separate donor-label
    records supply the cyclic assignments.  The anchor label reaches the
    positive cell once and a negative cell twice, so all 99 draws keep the same
    neural/support context while 98 break the anchor-label pairing.
    """

    generator = torch.Generator(device="cpu")
    generator.manual_seed(fixture.seed)
    donor_identities = torch.tensor(
        [donor.identity for donor in fixture.calibration_records.pronunciation_donors],
        dtype=torch.long,
    )
    positions = torch.arange(_NULL_PRONUNCIATION_COUNT, dtype=torch.long)
    shifts = torch.randperm(
        _NULL_PERMUTATION_COUNT,
        generator=generator,
    )
    permutations = torch.stack(
        [
            donor_identities[(positions + shift) % _NULL_PRONUNCIATION_COUNT]
            for shift in shifts
        ],
        dim=0,
    )
    statistics = torch.stack(
        [
            _signed_fisher_alignment(
                _permuted_null_interactions(
                    fixture,
                    student_probs,
                    prefix,
                    permutation,
                ),
                brain_only_direction,
                student_probs,
                statistic_epsilon=_STATISTIC_EPSILON,
            )
            for permutation in permutations
        ],
        dim=0,
    ).detach()
    threshold = torch.quantile(
        statistics,
        1.0 - _NULL_ALPHA,
        dim=0,
        interpolation="higher",
    ).detach()
    anchor_identity = fixture.calibration_records.anchor_record.pronunciation.identity
    producer_records = fixture.calibration_records.contributor_records
    return _SyntheticNullCalibration(
        threshold=threshold,
        statistics=statistics,
        permutations=permutations.detach(),
        anchor_match_draw_count=int(
            (permutations[:, 0] == anchor_identity).sum().item()
        ),
        any_match_draw_count=int(
            (permutations[:, :3] == anchor_identity).any(dim=1).sum().item()
        ),
        pairing_broken_draw_count=int(
            (permutations[:, 0] != anchor_identity).sum().item()
        ),
        alpha=_NULL_ALPHA,
        permutation_count=_NULL_PERMUTATION_COUNT,
        seed=fixture.seed,
        split=producer_records[0].split,
        statistic_id="signed_fisher_alignment",
        statistic_epsilon=_STATISTIC_EPSILON,
        phase="pilot",
        k=len(fixture.interaction_validities),
        sampler=_NULL_SAMPLER,
        aggregation_policy=_AGGREGATION_POLICY,
        weight_policy=_WEIGHT_POLICY,
        checkpoint_id=_CHECKPOINT_ID,
        prefix_policy=_PREFIX_POLICY,
        alignment_policy=_ALIGNMENT_POLICY,
        support_policy=_SUPPORT_POLICY,
        producer_fold_ids=tuple(
            dict.fromkeys(record.fold_id for record in producer_records)
        ),
        producer_session_ids=tuple(
            dict.fromkeys(record.session_id for record in producer_records)
        ),
        producer_trial_ids=tuple(record.trial_id for record in producer_records),
    )


def _brain_only_direction(fixture: _SyntheticFixture, prefix: Tensor) -> Tensor:
    anchor = fixture.brain_support.log_probs(
        fixture.anchor_brain,
        prefix,
    )
    controls = torch.stack(
        [
            fixture.brain_support.log_probs(control, prefix)
            for control in fixture.control_brains
        ],
        dim=0,
    ).mean(dim=0)
    return (anchor - controls).detach()


def _nra_evidence(
    fixture: _SyntheticFixture,
    brain_only_direction: Tensor,
    calibration: _SyntheticNullCalibration,
) -> NRAEvidence:
    """Bind synthetic support to valid train-only cross-fit provenance."""

    return NRAEvidence(
        brain_only_direction=brain_only_direction,
        null_threshold=calibration.threshold,
        source="frozen_crossfit_ce",
        split=calibration.split,
        subject_id=fixture.subject_id,
        pronunciation_id=fixture.pronunciation_id,
        reliability_stratum=fixture.reliability_stratum,
        producer_fold_ids=calibration.producer_fold_ids,
        current_fold_id=fixture.current_anchor_record.fold_id,
        producer_session_ids=calibration.producer_session_ids,
        current_session_id=fixture.current_anchor_record.session_id,
        producer_trial_ids=calibration.producer_trial_ids,
        current_trial_id=fixture.current_anchor_record.trial_id,
        checkpoint_id=calibration.checkpoint_id,
        statistic_id=calibration.statistic_id,
        statistic_epsilon=calibration.statistic_epsilon,
        alpha=calibration.alpha,
        phase=calibration.phase,
        permutation_count=calibration.permutation_count,
        k=calibration.k,
        sampler=calibration.sampler,
        aggregation_policy=calibration.aggregation_policy,
        weight_policy=calibration.weight_policy,
        seed=calibration.seed,
        producer_prefix_policy=calibration.prefix_policy,
        current_prefix_policy=calibration.prefix_policy,
        producer_alignment_policy=calibration.alignment_policy,
        current_alignment_policy=calibration.alignment_policy,
        producer_support_policy=calibration.support_policy,
        current_support_policy=calibration.support_policy,
        control_policy="anchor_local_three_mismatch",
    )


def _nra_target(
    fixture: _SyntheticFixture,
    student_log_probs: Tensor,
    prefix: Tensor,
    *,
    brain_only_direction: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    student_probs = student_log_probs.detach().exp()
    interactions = _anchor_interactions(fixture, student_probs, prefix)
    teacher_log_probs = _teacher_log_probs(
        fixture,
        fixture.anchor_brain,
        fixture.anchor_privilege,
        prefix,
    )
    support = (
        _brain_only_direction(fixture, prefix)
        if brain_only_direction is None
        else brain_only_direction
    )
    calibration = _calibrate_nra_null(
        fixture,
        student_probs,
        prefix,
        support,
    )
    evidence = _nra_evidence(fixture, support, calibration)
    interaction_batch = NRAInteractionBatch(
        values=interactions,
        subject_id=evidence.subject_id,
        pronunciation_id=evidence.pronunciation_id,
        reliability_stratum=evidence.reliability_stratum,
        current_fold_id=evidence.current_fold_id,
        current_session_id=evidence.current_session_id,
        current_trial_id=evidence.current_trial_id,
        checkpoint_id=evidence.checkpoint_id,
        prefix_policy=evidence.current_prefix_policy,
        alignment_policy=evidence.current_alignment_policy,
        support_policy=evidence.current_support_policy,
        sampler=evidence.sampler,
        aggregation_policy=evidence.aggregation_policy,
        weight_policy=evidence.weight_policy,
        seed=evidence.seed,
        weights=None,
        control_policy=evidence.control_policy,
    )
    target, diagnostics = recoverable_target(
        mode="nra",
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        anchor_interactions=interaction_batch,
        nra_evidence=evidence,
        neural_reliability=0.90,
        session_stability=0.80,
        aggregation=_AGGREGATION_POLICY,
        alpha=1.0,
        clip_value=2.0,
    )
    result_diagnostics = dict(diagnostics)
    result_diagnostics.update(
        {
            "null_permutation_statistics": calibration.statistics,
            "null_anchor_match_draw_count": student_probs.new_tensor(
                calibration.anchor_match_draw_count
            ),
            "null_any_match_draw_count": student_probs.new_tensor(
                calibration.any_match_draw_count
            ),
            "null_pairing_broken_draw_count": student_probs.new_tensor(
                calibration.pairing_broken_draw_count
            ),
            "null_alpha": student_probs.new_tensor(calibration.alpha),
            "null_permutation_count": student_probs.new_tensor(
                calibration.permutation_count
            ),
            "null_seed": student_probs.new_tensor(calibration.seed),
        }
    )
    return target, result_diagnostics


def synthetic_target_checks(
    *,
    seed: int = 0,
) -> dict[str, bool | float | int | list[float]]:
    """Exercise the NRA support/null boundary on controlled synthetic logits."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(seed)
        reference_probabilities = torch.tensor(
            [0.1, 0.7, 0.19, 0.01],
            dtype=_DTYPE,
        )

        def reference_inputs(
            fixture: _SyntheticFixture,
        ) -> tuple[Tensor, Tensor, Tensor]:
            fixture.student.brain_head.weight.zero_()
            fixture.student.prefix_embedding.weight.copy_(
                reference_probabilities.log().expand_as(
                    fixture.student.prefix_embedding.weight
                )
            )
            fixture.teacher.base.load_state_dict(fixture.student.state_dict())
            prefix = _greedy_student_prefix(
                fixture.student,
                fixture.anchor_brain,
                token_count=fixture.prefix.shape[1],
            )
            student_log_probs = torch.log_softmax(
                fixture.student.student_logits(fixture.anchor_brain, prefix),
                dim=-1,
            )
            return prefix, student_log_probs, student_log_probs.exp()

        copier_fixture = _build_fixture(
            seed=seed,
            anchor_identity=_COPIER_ONLY_IDENTITY,
        )
        copier_prefix, copier_student_log_probs, copier_student_probs = (
            reference_inputs(copier_fixture)
        )
        copier_target, copier_diagnostics = _nra_target(
            copier_fixture,
            copier_student_log_probs,
            copier_prefix,
        )
        below_null_target = copier_target

        fixture = _build_fixture(seed=seed)
        prefix, student_log_probs, student_probs = reference_inputs(fixture)
        unsupported_direction = -_brain_only_direction(fixture, prefix)
        unsupported_target, _ = _nra_target(
            fixture,
            student_log_probs,
            prefix,
            brain_only_direction=unsupported_direction,
        )
        supported_target, supported_diagnostics = _nra_target(
            fixture,
            student_log_probs,
            prefix,
        )

        copier_delta = (copier_target - copier_student_probs).abs().max().item()
        below_null_delta = (below_null_target - copier_student_probs).abs().max().item()
        unsupported_delta = (unsupported_target - student_probs).abs().max().item()
        supported_delta = (supported_target - student_probs).abs().max().item()
        return {
            "student_reference_probabilities": [
                float(value) for value in copier_student_probs[0, 0].tolist()
            ],
            "copier_only_target_matches_student": copier_delta <= 1e-7,
            "copier_only_max_abs_delta": float(copier_delta),
            "below_null_target_matches_student": below_null_delta <= 1e-7,
            "below_null_max_abs_delta": float(below_null_delta),
            "copier_alignment_max": float(
                copier_diagnostics["alignment_score"].max().item()
            ),
            "null_threshold_min": float(
                copier_diagnostics["null_threshold"].min().item()
            ),
            "null_threshold_max": float(
                copier_diagnostics["null_threshold"].max().item()
            ),
            "null_permutation_count": int(
                copier_diagnostics["null_permutation_count"].item()
            ),
            "null_anchor_match_draw_count": int(
                copier_diagnostics["null_anchor_match_draw_count"].item()
            ),
            "null_any_match_draw_count": int(
                copier_diagnostics["null_any_match_draw_count"].item()
            ),
            "null_pairing_broken_draw_count": int(
                copier_diagnostics["null_pairing_broken_draw_count"].item()
            ),
            "null_alpha": float(copier_diagnostics["null_alpha"].item()),
            "null_seed": int(copier_diagnostics["null_seed"].item()),
            "null_neural_trial_count": len(
                copier_fixture.calibration_records.neural_trials
            ),
            "null_pronunciation_donor_count": len(
                copier_fixture.calibration_records.pronunciation_donors
            ),
            "unsupported_target_matches_student": unsupported_delta <= 1e-7,
            "unsupported_max_abs_delta": float(unsupported_delta),
            "supported_target_changes_student": supported_delta > 1e-4,
            "supported_target_max_abs_delta": float(supported_delta),
            "supported_alignment_min": float(
                supported_diagnostics["alignment_score"].min().item()
            ),
            "supported_coverage": float(
                supported_diagnostics["coverage"].mean().item()
            ),
            "supported_kappa_min": float(supported_diagnostics["kappa"].min().item()),
        }


def _legacy_teacher_and_gate(
    fixture: _SyntheticFixture,
    student_log_probs: Tensor,
    prefix: Tensor,
) -> tuple[Tensor, Tensor]:
    student_probs = student_log_probs.detach().exp()
    teacher_log_probs = _teacher_log_probs(
        fixture,
        fixture.anchor_brain,
        fixture.anchor_privilege,
        prefix,
    )
    negative_log_probs = _teacher_log_probs(
        fixture,
        fixture.anchor_brain,
        fixture.negative_privileges[0],
        prefix,
    )
    teacher_contrast = teacher_log_probs - negative_log_probs
    brain_contrast = _brain_only_direction(fixture, prefix)
    gate = legacy_dual_cosine_gate(
        teacher_contrast,
        brain_contrast,
        student_probs,
        reliability=0.85,
    )
    return teacher_log_probs.exp(), gate


def _rlvr_loss(student_log_probs: Tensor) -> tuple[Tensor, float]:
    candidates = ("alpha", "beta", "gamma", "delta")
    rewards = torch.tensor(
        [
            [wer_reward("alpha", candidate) for candidate in candidates],
            [wer_reward("omega", candidate) for candidate in candidates],
        ],
        dtype=student_log_probs.dtype,
    )
    standardized = group_relative_standardize(rewards)
    action_log_probs = student_log_probs[:, -1, :]
    loss = -(standardized.advantages * action_log_probs).sum(dim=-1).mean()
    return loss, standardized.skip_rate


def _method_loss(
    method: str,
    fixture: _SyntheticFixture,
) -> tuple[Tensor, dict[str, float | str]]:
    if method == "ce":
        prefix = fixture.prefix
        prefix_source = "gold_prefix"
    else:
        prefix = _greedy_student_prefix(
            fixture.student,
            fixture.anchor_brain,
            token_count=fixture.prefix.shape[1],
        )
        prefix_source = "current_student_greedy_rollout"
    student_logits = fixture.student.student_logits(
        fixture.anchor_brain,
        prefix,
    )
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    if method == "ce":
        return (
            F.cross_entropy(
                student_logits.reshape(-1, _VOCABULARY_SIZE),
                fixture.labels.reshape(-1),
            ),
            {"prefix_source": prefix_source},
        )
    if method == "vanilla_opsd":
        teacher_probs = _teacher_log_probs(
            fixture,
            fixture.anchor_brain,
            fixture.anchor_privilege,
            prefix,
        ).exp()
        return forward_kl(student_log_probs, teacher_probs), {
            "prefix_source": prefix_source
        }
    if method == "legacy_dual_cosine":
        teacher_probs, gate = _legacy_teacher_and_gate(
            fixture,
            student_log_probs,
            prefix,
        )
        return forward_kl(
            student_log_probs,
            teacher_probs,
            token_weights=gate,
        ), {
            "legacy_gate_mean": float(gate.mean().item()),
            "prefix_source": prefix_source,
        }
    if method == "nra_opsd":
        target, diagnostics = _nra_target(fixture, student_log_probs, prefix)
        return forward_kl(student_log_probs, target), {
            "coverage": float(diagnostics["coverage"].mean().item()),
            "target_kl": float(diagnostics["target_kl"].mean().item()),
            "prefix_source": prefix_source,
        }
    if method == "rlvr":
        loss, skip_rate = _rlvr_loss(student_log_probs)
        return loss, {
            "reward_zero_std_skip_rate": float(skip_rate),
            "prefix_source": prefix_source,
        }
    raise AssertionError(f"unhandled registered method {method!r}")


def run_synthetic_experiment(
    method: str,
    *,
    seed: int = 0,
    steps: int = 3,
) -> dict[str, Any]:
    """Run a finite deterministic CPU training path and return JSON-safe data."""

    if method not in SYNTHETIC_METHODS:
        expected = ", ".join(SYNTHETIC_METHODS)
        raise ValueError(
            f"unknown synthetic method {method!r}; choose one of {expected}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be positive")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        fixture = _build_fixture(seed)
        optimizer = torch.optim.SGD(fixture.student.parameters(), lr=0.12)
        initial_parameters = [
            parameter.detach().clone() for parameter in fixture.student.parameters()
        ]
        losses: list[float] = []
        student_gradients_present = False
        teacher_gradients_present = False
        final_diagnostics: dict[str, float | str] = {}

        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss, final_diagnostics = _method_loss(method, fixture)
            if not bool(torch.isfinite(loss.detach())):
                raise RuntimeError(f"{method} produced a non-finite synthetic loss")
            loss.backward()
            student_gradients = [
                parameter.grad for parameter in fixture.student.parameters()
            ]
            if any(
                gradient is not None and bool((gradient.detach() != 0).any())
                for gradient in student_gradients
            ):
                student_gradients_present = True
            if any(
                gradient is not None and not bool(torch.isfinite(gradient).all())
                for gradient in student_gradients
            ):
                raise RuntimeError(f"{method} produced a non-finite student gradient")
            teacher_gradients_present = teacher_gradients_present or any(
                parameter.grad is not None for parameter in fixture.teacher.parameters()
            )
            optimizer.step()
            losses.append(float(loss.detach().item()))

        student_updated = any(
            not torch.equal(before, after.detach())
            for before, after in zip(
                initial_parameters,
                fixture.student.parameters(),
                strict=True,
            )
        )
        return {
            "method": method,
            "seed": seed,
            "steps": steps,
            "device": "cpu",
            "losses": losses,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "finite": all(math.isfinite(value) for value in losses),
            "student_updated": student_updated,
            "student_gradients_present": student_gradients_present,
            "teacher_frozen": all(
                not parameter.requires_grad
                for parameter in fixture.teacher.parameters()
            ),
            "teacher_gradients_present": teacher_gradients_present,
            "diagnostics": final_diagnostics,
            "claim_scope": "synthetic_execution_only",
            "benchmark_claim": False,
        }


def run_synthetic_suite(*, seed: int = 0, steps: int = 3) -> dict[str, Any]:
    """Run every smoke method with matched seed and finite step count."""

    return {
        "claim_scope": "synthetic_execution_only",
        "benchmark_claim": False,
        "seed": seed,
        "steps": steps,
        "runs": {
            method: run_synthetic_experiment(method, seed=seed, steps=steps)
            for method in SYNTHETIC_METHODS
        },
        "target_checks": synthetic_target_checks(seed=seed),
    }


run_synthetic = run_synthetic_experiment


__all__ = [
    "SYNTHETIC_METHODS",
    "run_synthetic",
    "run_synthetic_experiment",
    "run_synthetic_suite",
    "synthetic_target_checks",
]
