"""Fail-closed calibration of neural posterior confidence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import torch
from torch import Tensor


@dataclass(frozen=True)
class CalibrationProvenance:
    """Generic producer and data provenance for a fitted calibration."""

    split: str
    fold_id: str
    subject_id: str
    session_id: str
    trial_ids: tuple[str, ...]
    checkpoint_id: str
    pronunciation_policy: str
    pronunciation_id: str

    def __post_init__(self) -> None:
        for name in (
            "split",
            "fold_id",
            "subject_id",
            "session_id",
            "checkpoint_id",
            "pronunciation_policy",
            "pronunciation_id",
        ):
            _require_nonempty_string(getattr(self, name), name)
        if not isinstance(self.trial_ids, tuple) or not self.trial_ids:
            raise ValueError("trial_ids must be a non-empty tuple")
        if any(
            not isinstance(trial_id, str)
            or not trial_id
            or trial_id != trial_id.strip()
            for trial_id in self.trial_ids
        ):
            raise ValueError(
                "trial_ids must contain non-empty strings without boundary whitespace"
            )
        if len(self.trial_ids) != len(set(self.trial_ids)):
            raise ValueError("trial_ids must not contain duplicates")

    def validate_for_use(
        self,
        *,
        checkpoint_id: str,
        current_fold_id: str,
        current_subject_id: str,
        current_session_id: str,
        current_trial_id: str,
        pronunciation_policy: str,
        current_pronunciation_id: str,
    ) -> None:
        """Reject producer mismatch and fold/session/trial leakage."""

        context = {
            "checkpoint_id": checkpoint_id,
            "current_fold_id": current_fold_id,
            "current_subject_id": current_subject_id,
            "current_session_id": current_session_id,
            "current_trial_id": current_trial_id,
            "pronunciation_policy": pronunciation_policy,
            "current_pronunciation_id": current_pronunciation_id,
        }
        for name, value in context.items():
            _require_nonempty_string(value, name)

        if self.split != "train":
            raise ValueError("calibration provenance must use the training split")
        if self.checkpoint_id != checkpoint_id:
            raise ValueError(
                "calibration checkpoint does not match the active checkpoint"
            )
        if self.fold_id == current_fold_id:
            raise ValueError("calibration must leave the current fold out")
        if self.subject_id != current_subject_id:
            raise ValueError("calibration subject does not match the current subject")
        if self.session_id == current_session_id:
            raise ValueError("calibration must leave the current session out")
        if current_trial_id in self.trial_ids:
            raise ValueError("calibration must leave the current trial out")
        if self.pronunciation_policy != pronunciation_policy:
            raise ValueError("calibration pronunciation policy is incompatible")
        if self.pronunciation_id != current_pronunciation_id:
            raise ValueError(
                "calibration pronunciation does not match the current pronunciation"
            )


@dataclass(frozen=True)
class ReliabilityCalibration:
    """Temperature and policies fitted for neural reliability calibration."""

    temperature: float
    provenance: CalibrationProvenance
    prefix_policy: str
    token_alignment_policy: str
    support_policy: str

    def __post_init__(self) -> None:
        if not isinstance(self.temperature, Real) or isinstance(
            self.temperature,
            bool,
        ):
            raise TypeError("temperature must be a real number")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and greater than zero")
        if not isinstance(self.provenance, CalibrationProvenance):
            raise TypeError("calibration provenance must be CalibrationProvenance")
        for name in (
            "prefix_policy",
            "token_alignment_policy",
            "support_policy",
        ):
            _require_nonempty_string(getattr(self, name), name)

    @property
    def pronunciation_id(self) -> str:
        """Concrete trial pronunciation recorded by the fitted artifact."""

        return self.provenance.pronunciation_id

    def validate_for_use(
        self,
        *,
        checkpoint_id: str,
        current_fold_id: str,
        current_subject_id: str,
        current_session_id: str,
        current_trial_id: str,
        pronunciation_policy: str,
        current_pronunciation_id: str,
        prefix_policy: str,
        token_alignment_policy: str,
        support_policy: str,
    ) -> None:
        self.provenance.validate_for_use(
            checkpoint_id=checkpoint_id,
            current_fold_id=current_fold_id,
            current_subject_id=current_subject_id,
            current_session_id=current_session_id,
            current_trial_id=current_trial_id,
            pronunciation_policy=pronunciation_policy,
            current_pronunciation_id=current_pronunciation_id,
        )
        context = {
            "prefix_policy": prefix_policy,
            "token_alignment_policy": token_alignment_policy,
            "support_policy": support_policy,
        }
        for name, value in context.items():
            _require_nonempty_string(value, name)
        if self.prefix_policy != prefix_policy:
            raise ValueError("reliability calibration prefix policy is incompatible")
        if self.token_alignment_policy != token_alignment_policy:
            raise ValueError(
                "reliability calibration token alignment policy is incompatible"
            )
        if self.support_policy != support_policy:
            raise ValueError("reliability calibration support policy is incompatible")


@dataclass(frozen=True)
class NullCalibration:
    """Registered signed-Fisher null threshold and complete design identity."""

    statistic_id: str
    threshold: float
    alpha: float
    permutation_count: int
    phase: Literal["pilot", "confirmatory"]
    provenance: CalibrationProvenance
    k: int
    sampler: str
    aggregation_policy: str
    weight_policy: str
    alignment_policy: str
    prefix_policy: str
    support_policy: str
    seed: int

    def __post_init__(self) -> None:
        if self.statistic_id != "signed_fisher_alignment":
            raise ValueError("statistic_id must be 'signed_fisher_alignment'")
        if not isinstance(self.threshold, Real) or isinstance(self.threshold, bool):
            raise TypeError("null-calibration threshold must be a real number")
        if not math.isfinite(self.threshold) or not -1.0 <= self.threshold <= 1.0:
            raise ValueError(
                "null-calibration threshold must be finite and lie in [-1, 1]"
            )
        if self.alpha != 0.05:
            raise ValueError("null-calibration alpha must be exactly 0.05")
        if self.phase not in {"pilot", "confirmatory"}:
            raise ValueError("null-calibration phase must be pilot or confirmatory")
        if not isinstance(self.permutation_count, int) or isinstance(
            self.permutation_count,
            bool,
        ):
            raise TypeError("permutation_count must be an integer")
        minimum = 99 if self.phase == "pilot" else 999
        if self.permutation_count < minimum:
            raise ValueError(
                f"{self.phase} null calibration requires at least {minimum} "
                "permutations"
            )
        if not isinstance(self.provenance, CalibrationProvenance):
            raise TypeError("calibration provenance must be CalibrationProvenance")
        if not isinstance(self.k, int) or isinstance(self.k, bool) or self.k <= 0:
            raise ValueError("k must be a positive integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        for name in (
            "sampler",
            "aggregation_policy",
            "weight_policy",
            "alignment_policy",
            "prefix_policy",
            "support_policy",
        ):
            _require_nonempty_string(getattr(self, name), name)

    @property
    def permutations(self) -> int:
        """Compatibility spelling for diagnostics and serialized reports."""

        return self.permutation_count

    def validate_for_use(
        self,
        *,
        checkpoint_id: str,
        current_fold_id: str,
        current_subject_id: str,
        current_session_id: str,
        current_trial_id: str,
        pronunciation_policy: str,
        current_pronunciation_id: str,
        k: int,
        sampler: str,
        aggregation_policy: str,
        weight_policy: str,
        alignment_policy: str,
        prefix_policy: str,
        support_policy: str,
        seed: int,
    ) -> None:
        """Validate generic provenance and exact statistic design together."""

        self.provenance.validate_for_use(
            checkpoint_id=checkpoint_id,
            current_fold_id=current_fold_id,
            current_subject_id=current_subject_id,
            current_session_id=current_session_id,
            current_trial_id=current_trial_id,
            pronunciation_policy=pronunciation_policy,
            current_pronunciation_id=current_pronunciation_id,
        )
        context = {
            "sampler": sampler,
            "aggregation_policy": aggregation_policy,
            "weight_policy": weight_policy,
            "alignment_policy": alignment_policy,
            "prefix_policy": prefix_policy,
            "support_policy": support_policy,
        }
        for name, value in context.items():
            _require_nonempty_string(value, name)
        if not isinstance(k, int) or isinstance(k, bool):
            raise TypeError("k must be an integer")
        if k <= 0:
            raise ValueError("k must be a positive integer")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        expected = {
            "k": self.k,
            "sampler": self.sampler,
            "aggregation policy": self.aggregation_policy,
            "weight policy": self.weight_policy,
            "alignment policy": self.alignment_policy,
            "prefix policy": self.prefix_policy,
            "support policy": self.support_policy,
            "seed": self.seed,
        }
        actual = {
            "k": k,
            "sampler": sampler,
            "aggregation policy": aggregation_policy,
            "weight policy": weight_policy,
            "alignment policy": alignment_policy,
            "prefix policy": prefix_policy,
            "support policy": support_policy,
            "seed": seed,
        }
        for name, expected_value in expected.items():
            if actual[name] != expected_value:
                raise ValueError(f"null-calibration {name} does not match")


def calibrated_reliability(
    posterior_confidence: Tensor,
    validity_mask: Tensor | None = None,
    *,
    calibration: ReliabilityCalibration,
    checkpoint_id: str,
    current_fold_id: str,
    current_subject_id: str,
    current_session_id: str,
    current_trial_id: str,
    pronunciation_policy: str,
    current_pronunciation_id: str,
    prefix_policy: str,
    token_alignment_policy: str,
    support_policy: str,
) -> Tensor:
    """Apply a provenance-validated temperature calibration, fail closed."""

    if not isinstance(calibration, ReliabilityCalibration):
        raise TypeError(
            "calibration must be a complete ReliabilityCalibration artifact"
        )
    calibration.validate_for_use(
        checkpoint_id=checkpoint_id,
        current_fold_id=current_fold_id,
        current_subject_id=current_subject_id,
        current_session_id=current_session_id,
        current_trial_id=current_trial_id,
        pronunciation_policy=pronunciation_policy,
        current_pronunciation_id=current_pronunciation_id,
        prefix_policy=prefix_policy,
        token_alignment_policy=token_alignment_policy,
        support_policy=support_policy,
    )
    return _map_reliability(
        posterior_confidence,
        validity_mask,
        temperature=calibration.temperature,
    )


def uncalibrated_reliability(
    posterior_confidence: Tensor,
    validity_mask: Tensor | None = None,
) -> Tensor:
    """Explicit identity-confidence mode that uses no fitted calibration."""

    return _map_reliability(
        posterior_confidence,
        validity_mask,
        temperature=1.0,
    )


def neural_reliability(
    posterior_confidence: Tensor,
    validity_mask: Tensor | None = None,
    **calibration_context: object,
) -> Tensor:
    """NRA-OPSD spelling for the fail-closed calibrated path."""

    return calibrated_reliability(
        posterior_confidence,
        validity_mask,
        **calibration_context,
    )


def _map_reliability(
    posterior_confidence: Tensor,
    validity_mask: Tensor | None,
    *,
    temperature: float,
) -> Tensor:
    if not isinstance(posterior_confidence, Tensor):
        raise TypeError("posterior_confidence must be a torch.Tensor")
    if not posterior_confidence.is_floating_point():
        raise TypeError("posterior_confidence must have a floating dtype")

    confidence = posterior_confidence.detach()
    if not bool(torch.isfinite(confidence).all()):
        raise ValueError("posterior_confidence must be finite")
    if not bool(((confidence >= 0) & (confidence <= 1)).all()):
        raise ValueError("posterior_confidence must lie in [0, 1]")

    with torch.no_grad():
        if temperature == 1.0:
            reliability = confidence.clone()
        else:
            reliability = torch.sigmoid(torch.logit(confidence) / temperature)

        if validity_mask is not None:
            if not isinstance(validity_mask, Tensor):
                raise TypeError("validity_mask must be a torch.Tensor")
            valid = validity_mask.detach()
            if valid.dtype != torch.bool and (
                valid.is_complex()
                or not bool(torch.isfinite(valid).all())
                or not bool(((valid == 0) | (valid == 1)).all())
            ):
                raise ValueError("validity_mask must be boolean or finite and binary")
            try:
                reliability, valid = torch.broadcast_tensors(
                    reliability,
                    valid.to(device=reliability.device),
                )
            except RuntimeError as error:
                raise ValueError(
                    "validity_mask must be broadcastable to posterior_confidence"
                ) from error
            reliability = torch.where(
                valid.to(dtype=torch.bool),
                reliability,
                torch.zeros_like(reliability),
            )

    return reliability.detach()


def _require_nonempty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            f"{name} must be a non-empty string without boundary whitespace"
        )
