from __future__ import annotations

import pytest
import torch

from brain_evidence.gates.reliability import (
    CalibrationProvenance,
    NullCalibration,
    ReliabilityCalibration,
    calibrated_reliability,
    uncalibrated_reliability,
)


def _provenance(**overrides: object) -> CalibrationProvenance:
    values = {
        "split": "train",
        "fold_id": "fold-fit",
        "subject_id": "subject-1",
        "session_id": "session-fit",
        "trial_ids": ("fit-1", "fit-2"),
        "checkpoint_id": "checkpoint-a",
        "pronunciation_policy": "arpabet-v1",
        "pronunciation_id": "pronunciation-cat",
    }
    values.update(overrides)
    return CalibrationProvenance(**values)


def _use_context(**overrides: object) -> dict[str, object]:
    values = {
        "checkpoint_id": "checkpoint-a",
        "current_fold_id": "fold-eval",
        "current_subject_id": "subject-1",
        "current_session_id": "session-eval",
        "current_trial_id": "eval-1",
        "pronunciation_policy": "arpabet-v1",
        "current_pronunciation_id": "pronunciation-cat",
        "prefix_policy": "student-rollout",
        "token_alignment_policy": "token-index-v1",
        "support_policy": "full-vocabulary",
    }
    values.update(overrides)
    return values


def _reliability_calibration(
    *,
    temperature: float = 2.0,
    **overrides: object,
) -> ReliabilityCalibration:
    values = {
        "temperature": temperature,
        "provenance": _provenance(),
        "prefix_policy": "student-rollout",
        "token_alignment_policy": "token-index-v1",
        "support_policy": "full-vocabulary",
    }
    values.update(overrides)
    return ReliabilityCalibration(**values)


def _null_calibration(**overrides: object) -> NullCalibration:
    values = {
        "statistic_id": "signed_fisher_alignment",
        "threshold": 0.2,
        "alpha": 0.05,
        "permutation_count": 99,
        "phase": "pilot",
        "provenance": _provenance(),
        "k": 2,
        "sampler": "anchor-local-v1",
        "aggregation_policy": "mean",
        "weight_policy": "uniform",
        "alignment_policy": "token-index-v1",
        "prefix_policy": "student-rollout",
        "support_policy": "full-vocabulary",
        "seed": 41,
    }
    values.update(overrides)
    return NullCalibration(**values)


def _null_use_context(**overrides: object) -> dict[str, object]:
    values = _use_context()
    values["alignment_policy"] = values.pop("token_alignment_policy")
    values.update(
        {
            "k": 2,
            "sampler": "anchor-local-v1",
            "aggregation_policy": "mean",
            "weight_policy": "uniform",
            "seed": 41,
        }
    )
    values.update(overrides)
    return values


def test_reliability_applies_calibration_and_validity_mask() -> None:
    confidence = torch.tensor([0.2, 0.8, 0.5], requires_grad=True)
    valid = torch.tensor([True, True, False])

    reliability = calibrated_reliability(
        confidence,
        valid,
        calibration=_reliability_calibration(temperature=2.0),
        **_use_context(),
    )

    torch.testing.assert_close(
        reliability,
        torch.tensor([1 / 3, 2 / 3, 0.0]),
    )
    assert torch.all((0.0 <= reliability) & (reliability <= 1.0))
    assert not reliability.requires_grad


def test_uncalibrated_mode_is_explicit_and_leaves_confidence_unchanged() -> None:
    confidence = torch.tensor([[0.0, 0.25, 1.0]], requires_grad=True)

    reliability = uncalibrated_reliability(confidence)

    torch.testing.assert_close(reliability, confidence.detach())
    assert not reliability.requires_grad


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_reliability_accepts_finite_binary_numeric_validity_mask(
    dtype: torch.dtype,
) -> None:
    reliability = uncalibrated_reliability(
        torch.tensor([[0.2, 0.8], [0.6, 0.4]]),
        torch.tensor([1, 0], dtype=dtype),
    )

    torch.testing.assert_close(
        reliability,
        torch.tensor([[0.2, 0.0], [0.6, 0.0]]),
    )


@pytest.mark.parametrize("mask_value", [-1.0, 0.5, float("nan"), float("inf")])
def test_reliability_rejects_nonbinary_or_nonfinite_validity_mask(
    mask_value: float,
) -> None:
    with pytest.raises(ValueError, match="validity_mask"):
        uncalibrated_reliability(
            torch.tensor([0.8]),
            torch.tensor([mask_value]),
        )


def test_reliability_rejects_complex_validity_mask() -> None:
    with pytest.raises(ValueError, match="validity_mask"):
        uncalibrated_reliability(
            torch.tensor([0.8]),
            torch.tensor([1 + 0j]),
        )


def test_reliability_rejects_nonbroadcastable_validity_mask() -> None:
    with pytest.raises(ValueError, match="broadcastable"):
        uncalibrated_reliability(
            torch.tensor([0.2, 0.8]),
            torch.tensor([True, False, True]),
        )


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("inf")])
def test_reliability_calibration_rejects_invalid_temperature(
    temperature: float,
) -> None:
    with pytest.raises(ValueError, match="temperature"):
        _reliability_calibration(temperature=temperature)


@pytest.mark.parametrize("temperature", [True, 1 + 0j, "1.0"])
def test_reliability_calibration_rejects_bool_or_nonreal_temperature(
    temperature: object,
) -> None:
    with pytest.raises(TypeError, match="temperature"):
        _reliability_calibration(temperature=temperature)


def test_reliability_rejects_values_that_are_not_probabilities() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        uncalibrated_reliability(torch.tensor([1.01]))


def test_calibration_provenance_validates_all_registered_boundaries() -> None:
    _provenance().validate_for_use(
        checkpoint_id="checkpoint-a",
        current_fold_id="fold-eval",
        current_subject_id="subject-1",
        current_session_id="session-eval",
        current_trial_id="eval-1",
        pronunciation_policy="arpabet-v1",
        current_pronunciation_id="pronunciation-cat",
    )


def test_concrete_pronunciation_is_recorded_without_class_exclusion() -> None:
    calibration = _reliability_calibration()

    assert calibration.pronunciation_id == "pronunciation-cat"
    reliability = calibrated_reliability(
        torch.tensor([0.8]),
        calibration=calibration,
        **_use_context(current_pronunciation_id="pronunciation-cat"),
    )

    torch.testing.assert_close(reliability, torch.tensor([2 / 3]))


def test_reliability_calibration_rejects_pronunciation_mismatch() -> None:
    with pytest.raises(ValueError, match="pronunciation"):
        calibrated_reliability(
            torch.tensor([0.8]),
            calibration=_reliability_calibration(),
            **_use_context(current_pronunciation_id="pronunciation-ship"),
        )


def test_null_calibration_rejects_pronunciation_mismatch() -> None:
    calibration = _null_calibration()
    generic_context = _use_context(
        current_pronunciation_id="pronunciation-ship",
    )
    for key in ("prefix_policy", "token_alignment_policy", "support_policy"):
        generic_context.pop(key)

    with pytest.raises(ValueError, match="pronunciation"):
        calibration.validate_for_use(
            **generic_context,
            k=2,
            sampler="anchor-local-v1",
            aggregation_policy="mean",
            weight_policy="uniform",
            alignment_policy="token-index-v1",
            prefix_policy="student-rollout",
            support_policy="full-vocabulary",
            seed=41,
        )


@pytest.mark.parametrize("current_pronunciation_id", ["", " \t", None, 7])
def test_calibration_provenance_rejects_invalid_current_pronunciation_id(
    current_pronunciation_id: object,
) -> None:
    use = _use_context(current_pronunciation_id=current_pronunciation_id)
    for key in ("prefix_policy", "token_alignment_policy", "support_policy"):
        use.pop(key)

    with pytest.raises(ValueError, match="current_pronunciation_id"):
        _provenance().validate_for_use(**use)


@pytest.mark.parametrize(
    "field",
    [
        "checkpoint_id",
        "current_fold_id",
        "current_subject_id",
        "current_session_id",
        "current_trial_id",
        "pronunciation_policy",
    ],
)
def test_calibration_provenance_rejects_whitespace_only_use_context(
    field: str,
) -> None:
    use = _use_context(**{field: " \t"})
    for key in ("prefix_policy", "token_alignment_policy", "support_policy"):
        use.pop(key)

    with pytest.raises(ValueError):
        _provenance().validate_for_use(**use)


@pytest.mark.parametrize("pronunciation_id", ["", " \t"])
def test_calibration_rejects_empty_concrete_pronunciation_id(
    pronunciation_id: str,
) -> None:
    with pytest.raises(ValueError, match="pronunciation_id"):
        _provenance(pronunciation_id=pronunciation_id)


@pytest.mark.parametrize(
    "field",
    [
        "split",
        "fold_id",
        "subject_id",
        "session_id",
        "checkpoint_id",
        "pronunciation_policy",
    ],
)
def test_calibration_provenance_rejects_whitespace_only_fields(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _provenance(**{field: " \t"})


def test_calibration_provenance_rejects_whitespace_only_trial_id() -> None:
    with pytest.raises(ValueError, match="trial_ids"):
        _provenance(trial_ids=("fit-1", " \t"))


def test_calibration_artifacts_reject_whitespace_only_policies() -> None:
    with pytest.raises(ValueError, match="prefix_policy"):
        _reliability_calibration(prefix_policy=" \t")

    with pytest.raises(ValueError, match="sampler"):
        _null_calibration(sampler=" \t")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("split", "train"),
        ("fold_id", "fold-fit"),
        ("subject_id", "subject-1"),
        ("session_id", "session-fit"),
        ("checkpoint_id", "checkpoint-a"),
        ("pronunciation_policy", "arpabet-v1"),
        ("pronunciation_id", "pronunciation-cat"),
    ],
)
@pytest.mark.parametrize("side", ["leading", "trailing"])
def test_calibration_provenance_rejects_boundary_whitespace(
    field: str,
    value: str,
    side: str,
) -> None:
    padded_value = f" {value}" if side == "leading" else f"{value} "

    with pytest.raises(ValueError, match=field):
        _provenance(**{field: padded_value})


@pytest.mark.parametrize("trial_id", [" fit-1", "fit-1 "])
def test_calibration_provenance_rejects_trial_id_boundary_whitespace(
    trial_id: str,
) -> None:
    with pytest.raises(ValueError, match="trial_ids"):
        _provenance(trial_ids=(trial_id, "fit-2"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checkpoint_id", "checkpoint-a"),
        ("current_fold_id", "fold-fit"),
        ("current_subject_id", "subject-1"),
        ("current_session_id", "session-fit"),
        ("current_trial_id", "fit-1"),
        ("pronunciation_policy", "arpabet-v1"),
        ("current_pronunciation_id", "pronunciation-cat"),
    ],
)
@pytest.mark.parametrize("side", ["leading", "trailing"])
def test_calibration_provenance_rejects_use_context_boundary_whitespace(
    field: str,
    value: str,
    side: str,
) -> None:
    padded_value = f" {value}" if side == "leading" else f"{value} "
    use = _use_context(**{field: padded_value})
    for policy_field in (
        "prefix_policy",
        "token_alignment_policy",
        "support_policy",
    ):
        use.pop(policy_field)

    with pytest.raises(ValueError, match=rf"{field} must be a non-empty string"):
        _provenance().validate_for_use(**use)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prefix_policy", "student-rollout"),
        ("token_alignment_policy", "token-index-v1"),
        ("support_policy", "full-vocabulary"),
    ],
)
@pytest.mark.parametrize("side", ["leading", "trailing"])
def test_reliability_calibration_rejects_stored_policy_boundary_whitespace(
    field: str,
    value: str,
    side: str,
) -> None:
    padded_value = f" {value}" if side == "leading" else f"{value} "

    with pytest.raises(ValueError, match=field):
        _reliability_calibration(**{field: padded_value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prefix_policy", "student-rollout"),
        ("token_alignment_policy", "token-index-v1"),
        ("support_policy", "full-vocabulary"),
    ],
)
@pytest.mark.parametrize("side", ["leading", "trailing"])
def test_reliability_calibration_rejects_use_policy_boundary_whitespace(
    field: str,
    value: str,
    side: str,
) -> None:
    padded_value = f" {value}" if side == "leading" else f"{value} "

    with pytest.raises(ValueError, match=rf"{field} must be a non-empty string"):
        calibrated_reliability(
            torch.tensor([0.8]),
            calibration=_reliability_calibration(),
            **_use_context(**{field: padded_value}),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sampler", "anchor-local-v1"),
        ("aggregation_policy", "mean"),
        ("weight_policy", "uniform"),
        ("alignment_policy", "token-index-v1"),
        ("prefix_policy", "student-rollout"),
        ("support_policy", "full-vocabulary"),
    ],
)
@pytest.mark.parametrize("side", ["leading", "trailing"])
def test_null_calibration_rejects_stored_policy_boundary_whitespace(
    field: str,
    value: str,
    side: str,
) -> None:
    padded_value = f" {value}" if side == "leading" else f"{value} "

    with pytest.raises(ValueError, match=field):
        _null_calibration(**{field: padded_value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sampler", "anchor-local-v1"),
        ("aggregation_policy", "mean"),
        ("weight_policy", "uniform"),
        ("alignment_policy", "token-index-v1"),
        ("prefix_policy", "student-rollout"),
        ("support_policy", "full-vocabulary"),
    ],
)
@pytest.mark.parametrize("side", ["leading", "trailing"])
def test_null_calibration_rejects_use_policy_boundary_whitespace(
    field: str,
    value: str,
    side: str,
) -> None:
    padded_value = f" {value}" if side == "leading" else f"{value} "

    with pytest.raises(ValueError, match=rf"{field} must be a non-empty string"):
        _null_calibration().validate_for_use(
            **_null_use_context(**{field: padded_value})
        )


@pytest.mark.parametrize(
    ("override", "use_override", "message"),
    [
        ({"split": "validation"}, {}, "training split"),
        ({}, {"checkpoint_id": "checkpoint-b"}, "checkpoint"),
        ({}, {"current_fold_id": "fold-fit"}, "fold"),
        ({}, {"current_session_id": "session-fit"}, "session"),
        ({}, {"current_trial_id": "fit-1"}, "trial"),
        ({}, {"current_subject_id": "subject-2"}, "subject"),
        ({}, {"pronunciation_policy": "ipa-v1"}, "pronunciation policy"),
    ],
)
def test_calibration_provenance_rejects_leakage_or_mismatch(
    override: dict[str, object],
    use_override: dict[str, object],
    message: str,
) -> None:
    provenance = _provenance(**override)
    use = _use_context()
    for key in ("prefix_policy", "token_alignment_policy", "support_policy"):
        use.pop(key)
    use.update(use_override)

    with pytest.raises(ValueError, match=message):
        provenance.validate_for_use(**use)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"statistic_id": "absolute_cosine"}, "signed_fisher_alignment"),
        ({"alpha": 0.049}, "0.05"),
        ({"k": 0}, "k"),
        ({"sampler": ""}, "sampler"),
    ],
)
def test_null_calibration_rejects_unregistered_design(
    override: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _null_calibration(**override)


@pytest.mark.parametrize("threshold", [-2.0, 2.0])
def test_null_calibration_rejects_out_of_range_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        _null_calibration(threshold=threshold)


@pytest.mark.parametrize("threshold", [-1.0, 1.0])
def test_null_calibration_accepts_threshold_boundaries(threshold: float) -> None:
    assert _null_calibration(threshold=threshold).threshold == threshold


@pytest.mark.parametrize("threshold", [True, 1 + 0j, "0.2"])
def test_null_calibration_rejects_bool_or_nonreal_threshold(
    threshold: object,
) -> None:
    with pytest.raises(TypeError, match="threshold"):
        _null_calibration(threshold=threshold)


@pytest.mark.parametrize(
    ("phase", "permutation_count", "minimum"),
    [("pilot", 98, 99), ("confirmatory", 998, 999)],
)
def test_null_calibration_enforces_phase_specific_permutation_minimum(
    phase: str,
    permutation_count: int,
    minimum: int,
) -> None:
    with pytest.raises(ValueError, match=str(minimum)):
        _null_calibration(
            phase=phase,
            permutation_count=permutation_count,
        )


@pytest.mark.parametrize(
    ("calibration_override", "context_override", "message"),
    [
        ({"k": 1}, {"k": True}, "k"),
        ({"k": 1}, {"k": 1.0}, "k"),
        ({"seed": 1}, {"seed": True}, "seed"),
        ({"seed": 1}, {"seed": 1.0}, "seed"),
    ],
)
def test_null_calibration_rejects_noninteger_use_context(
    calibration_override: dict[str, object],
    context_override: dict[str, object],
    message: str,
) -> None:
    calibration = _null_calibration(**calibration_override)

    with pytest.raises(TypeError, match=message):
        calibration.validate_for_use(**_null_use_context(**context_override))


def test_null_calibration_validates_provenance_and_exact_design() -> None:
    calibration = _null_calibration()
    generic_context = _use_context()
    for key in ("prefix_policy", "token_alignment_policy", "support_policy"):
        generic_context.pop(key)

    calibration.validate_for_use(
        **generic_context,
        k=2,
        sampler="anchor-local-v1",
        aggregation_policy="mean",
        weight_policy="uniform",
        alignment_policy="token-index-v1",
        prefix_policy="student-rollout",
        support_policy="full-vocabulary",
        seed=41,
    )

    with pytest.raises(ValueError, match="aggregation policy"):
        calibration.validate_for_use(
            **generic_context,
            k=2,
            sampler="anchor-local-v1",
            aggregation_policy="median",
            weight_policy="uniform",
            alignment_policy="token-index-v1",
            prefix_policy="student-rollout",
            support_policy="full-vocabulary",
            seed=41,
        )


def test_calibrated_reliability_fails_closed_without_complete_artifact() -> None:
    with pytest.raises((TypeError, ValueError), match="calibration"):
        calibrated_reliability(
            torch.tensor([0.8]),
            calibration=None,
            **_use_context(),
        )


@pytest.mark.parametrize(
    ("context_override", "message"),
    [
        ({"current_trial_id": "fit-1"}, "trial"),
        ({"current_fold_id": "fold-fit"}, "fold"),
        ({"current_session_id": "session-fit"}, "session"),
        ({"prefix_policy": "gold-prefix"}, "prefix policy"),
        ({"token_alignment_policy": "word-aligned"}, "token alignment"),
        ({"support_policy": "top-k"}, "support policy"),
    ],
)
def test_calibrated_reliability_rejects_leakage_or_policy_mismatch(
    context_override: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        calibrated_reliability(
            torch.tensor([0.8]),
            calibration=_reliability_calibration(),
            **_use_context(**context_override),
        )
