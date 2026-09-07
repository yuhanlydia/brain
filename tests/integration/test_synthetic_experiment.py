from __future__ import annotations

import inspect
import math
from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from brain_evidence.experiments import synthetic
from brain_evidence.experiments.synthetic import (
    SYNTHETIC_METHODS,
    run_synthetic_experiment,
    run_synthetic_suite,
    synthetic_target_checks,
)
from brain_evidence.gates import legacy_dual_cosine_gate
from brain_evidence.objectives import forward_kl


@pytest.mark.parametrize("method", SYNTHETIC_METHODS)
def test_each_synthetic_method_runs_finite_student_only_steps(method: str) -> None:
    result = run_synthetic_experiment(method, seed=17, steps=3)

    assert result["method"] == method
    assert result["device"] == "cpu"
    assert result["steps"] == 3
    assert len(result["losses"]) == 3
    assert all(math.isfinite(loss) for loss in result["losses"])
    assert result["finite"] is True
    assert result["student_updated"] is True
    assert result["student_gradients_present"] is True
    assert result["teacher_frozen"] is True
    assert result["teacher_gradients_present"] is False
    assert result["claim_scope"] == "synthetic_execution_only"
    assert result["benchmark_claim"] is False


def test_synthetic_experiment_is_exactly_deterministic_for_a_seed() -> None:
    first = run_synthetic_experiment("nra_opsd", seed=23, steps=2)
    second = run_synthetic_experiment("nra_opsd", seed=23, steps=2)

    assert first == second


def test_synthetic_nra_checks_block_copying_and_unsupported_directions() -> None:
    checks = synthetic_target_checks()

    assert checks["student_reference_probabilities"] == pytest.approx(
        [0.1, 0.7, 0.19, 0.01]
    )
    assert checks["copier_only_target_matches_student"] is True
    assert checks["copier_only_max_abs_delta"] == pytest.approx(0.0, abs=1e-7)
    assert checks["below_null_target_matches_student"] is True
    assert checks["copier_alignment_max"] <= checks["null_threshold_min"]
    assert checks["unsupported_target_matches_student"] is True
    assert checks["supported_target_changes_student"] is True
    assert checks["supported_target_max_abs_delta"] > 1e-4
    assert checks["supported_alignment_min"] > checks["null_threshold_max"]
    assert checks["supported_coverage"] == pytest.approx(1.0)
    assert checks["null_permutation_count"] == 99
    assert checks["null_anchor_match_draw_count"] == 1
    assert checks["null_any_match_draw_count"] == 3
    assert checks["null_pairing_broken_draw_count"] == 98
    assert checks["null_alpha"] == pytest.approx(0.05)
    assert checks["null_seed"] == 0


def test_synthetic_suite_runs_every_registered_method() -> None:
    suite = run_synthetic_suite(seed=5, steps=1)

    assert suite["claim_scope"] == "synthetic_execution_only"
    assert suite["benchmark_claim"] is False
    assert tuple(suite["runs"]) == SYNTHETIC_METHODS
    assert all(run["finite"] for run in suite["runs"].values())
    assert suite["target_checks"]["copier_only_target_matches_student"] is True
    assert suite["target_checks"]["null_seed"] == suite["seed"] == 5


def test_synthetic_target_checks_exposes_a_seed_with_zero_default() -> None:
    parameter = inspect.signature(synthetic_target_checks).parameters["seed"]

    assert parameter.default == 0


def test_synthetic_runner_rejects_unknown_methods_and_nonpositive_steps() -> None:
    with pytest.raises(ValueError, match="unknown synthetic method"):
        run_synthetic_experiment("not_a_method")
    with pytest.raises(ValueError, match="steps must be positive"):
        run_synthetic_experiment("ce", steps=0)


def test_greedy_prefix_is_detached_brain_only_and_tracks_current_student() -> None:
    fixture = synthetic._build_fixture(seed=17)

    first = synthetic._greedy_student_prefix(
        fixture.student,
        fixture.anchor_brain,
        token_count=fixture.prefix.shape[1],
    )
    assert first.requires_grad is False
    assert (
        "privilege"
        not in inspect.signature(synthetic._greedy_student_prefix).parameters
    )

    with torch.no_grad():
        fixture.student.brain_head.weight.zero_()
        fixture.student.prefix_embedding.weight.zero_()
        fixture.student.prefix_embedding.weight[:, 2] = 1.0
    second = synthetic._greedy_student_prefix(
        fixture.student,
        fixture.anchor_brain,
        token_count=fixture.prefix.shape[1],
    )

    assert not torch.equal(first, second)
    assert torch.equal(second, torch.tensor([[0, 2, 2], [0, 2, 2]]))


def test_ce_prefix_is_bos_shifted_gold_for_every_example() -> None:
    fixture = synthetic._build_fixture(seed=17)

    assert torch.equal(
        fixture.prefix[:, :1],
        torch.full_like(fixture.prefix[:, :1], synthetic._START_TOKEN_ID),
    )
    assert torch.equal(fixture.prefix[:, 1:], fixture.labels[:, :-1])
    assert torch.equal(fixture.prefix[1], torch.tensor([0, 3, 0]))


@pytest.mark.parametrize(
    ("method", "expects_brain_support", "expects_calibration_teacher"),
    [
        ("vanilla_opsd", False, False),
        ("legacy_dual_cosine", True, False),
        ("nra_opsd", True, True),
    ],
)
def test_opsd_reuses_one_rollout_for_student_teacher_and_support(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    expects_brain_support: bool,
    expects_calibration_teacher: bool,
) -> None:
    fixture = synthetic._build_fixture(seed=17)
    student_prefixes: list[torch.Tensor] = []
    teacher_prefixes: list[torch.Tensor] = []
    support_prefixes: list[torch.Tensor | None] = []
    original_student_logits = fixture.student.student_logits
    original_teacher_logits = fixture.teacher.teacher_logits
    original_support_log_probs = fixture.brain_support.log_probs

    def record_student_logits(
        brain: torch.Tensor, prefix: torch.Tensor
    ) -> torch.Tensor:
        student_prefixes.append(prefix)
        return original_student_logits(brain, prefix)

    def record_teacher_logits(
        brain: torch.Tensor,
        privilege: torch.Tensor,
        prefix: torch.Tensor,
    ) -> torch.Tensor:
        teacher_prefixes.append(prefix)
        return original_teacher_logits(brain, privilege, prefix)

    def record_support_log_probs(
        brain: torch.Tensor,
        prefix: torch.Tensor | None = None,
        *,
        token_count: int | None = None,
    ) -> torch.Tensor:
        support_prefixes.append(prefix)
        if prefix is None:
            assert token_count is not None
            return original_support_log_probs(brain, token_count=token_count)
        return original_support_log_probs(brain, prefix)

    monkeypatch.setattr(fixture.student, "student_logits", record_student_logits)
    monkeypatch.setattr(fixture.teacher, "teacher_logits", record_teacher_logits)
    monkeypatch.setattr(fixture.brain_support, "log_probs", record_support_log_probs)

    synthetic._method_loss(method, fixture)

    assert [prefix.shape[1] for prefix in student_prefixes] == [1, 2, 3]
    rollout = student_prefixes[-1]
    assert rollout.requires_grad is False
    assert all(prefix is rollout for prefix in teacher_prefixes)
    if expects_brain_support:
        assert support_prefixes
        assert all(prefix is rollout for prefix in support_prefixes)
    else:
        assert support_prefixes == []
    if expects_calibration_teacher:
        assert len(teacher_prefixes) > 8


def _reference_nra_inputs(
    fixture: synthetic._SyntheticFixture,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities = torch.tensor([0.1, 0.7, 0.19, 0.01], dtype=synthetic._DTYPE)
    with torch.no_grad():
        fixture.student.brain_head.weight.zero_()
        fixture.student.prefix_embedding.weight.copy_(
            probabilities.log().expand_as(fixture.student.prefix_embedding.weight)
        )
        fixture.teacher.base.load_state_dict(fixture.student.state_dict())
    prefix = synthetic._greedy_student_prefix(
        fixture.student,
        fixture.anchor_brain,
        token_count=fixture.prefix.shape[1],
    )
    student_log_probs = torch.log_softmax(
        fixture.student.student_logits(fixture.anchor_brain, prefix),
        dim=-1,
    )
    student_probs = student_log_probs.detach().exp()
    direction = synthetic._brain_only_direction(fixture, prefix)
    return prefix, student_log_probs, student_probs, direction


def _expected_legacy_loss(
    fixture: synthetic._SyntheticFixture,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = synthetic._greedy_student_prefix(
        fixture.student,
        fixture.anchor_brain,
        token_count=fixture.prefix.shape[1],
    )
    student_log_probs = torch.log_softmax(
        fixture.student.student_logits(fixture.anchor_brain, prefix),
        dim=-1,
    )
    teacher_log_probs = torch.log_softmax(
        fixture.teacher.teacher_logits(
            fixture.anchor_brain,
            fixture.anchor_privilege,
            prefix,
        ),
        dim=-1,
    ).detach()
    negative_log_probs = torch.log_softmax(
        fixture.teacher.teacher_logits(
            fixture.anchor_brain,
            fixture.negative_privileges[0],
            prefix,
        ),
        dim=-1,
    ).detach()
    token_count = prefix.shape[1]
    anchor_support = fixture.brain_support.log_probs(fixture.anchor_brain, prefix)
    control_support = torch.stack(
        [
            fixture.brain_support.log_probs(control, prefix)
            for control in fixture.control_brains
        ],
        dim=0,
    ).mean(dim=0)
    assert anchor_support.shape[1] == token_count
    gate = legacy_dual_cosine_gate(
        teacher_log_probs - negative_log_probs,
        anchor_support - control_support,
        student_log_probs.detach().exp(),
        reliability=0.85,
    )
    loss = forward_kl(
        student_log_probs,
        teacher_log_probs.exp(),
        token_weights=gate,
    )
    return loss, gate


def test_legacy_arm_is_exact_detached_gate_weighted_privileged_teacher_kl() -> None:
    actual_fixture = synthetic._build_fixture(seed=17)
    actual_loss, _ = synthetic._method_loss("legacy_dual_cosine", actual_fixture)
    actual_gradients = torch.autograd.grad(
        actual_loss,
        tuple(actual_fixture.student.parameters()),
    )

    expected_fixture = synthetic._build_fixture(seed=17)
    expected_loss, gate = _expected_legacy_loss(expected_fixture)
    expected_gradients = torch.autograd.grad(
        expected_loss,
        tuple(expected_fixture.student.parameters()),
    )

    assert bool((gate > 0).any())
    assert bool((gate < 1).any())
    torch.testing.assert_close(actual_loss, expected_loss)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected)


def test_nra_calibration_is_seeded_99_draw_higher_quantile_with_disjoint_ids() -> None:
    fixture = synthetic._build_fixture(seed=31)
    prefix = synthetic._greedy_student_prefix(
        fixture.student,
        fixture.anchor_brain,
        token_count=fixture.prefix.shape[1],
    )
    student_log_probs = torch.log_softmax(
        fixture.student.student_logits(fixture.anchor_brain, prefix),
        dim=-1,
    )
    student_probs = student_log_probs.detach().exp()
    direction = synthetic._brain_only_direction(fixture, prefix)

    calibration = synthetic._calibrate_nra_null(
        fixture,
        student_probs,
        prefix,
        direction,
    )
    repeated = synthetic._calibrate_nra_null(
        fixture,
        student_probs,
        prefix,
        direction,
    )

    assert calibration.permutation_count == 99
    assert calibration.alpha == pytest.approx(0.05)
    assert calibration.seed == fixture.seed
    assert calibration.statistics.shape == (99, *student_probs.shape[:-1])
    assert calibration.statistics.requires_grad is False
    assert calibration.threshold.requires_grad is False
    torch.testing.assert_close(calibration.permutations, repeated.permutations)
    torch.testing.assert_close(calibration.statistics, repeated.statistics)
    torch.testing.assert_close(calibration.threshold, repeated.threshold)

    identities = torch.arange(99)
    shifts = calibration.permutations[:, 0]
    assert torch.equal(shifts.sort().values, identities)
    assert not torch.equal(shifts, identities)
    for permutation, shift in zip(
        calibration.permutations,
        shifts,
        strict=True,
    ):
        assert torch.equal(permutation, (identities + shift) % 99)
    assert calibration.anchor_match_draw_count == 1
    assert calibration.any_match_draw_count == 3
    assert calibration.pairing_broken_draw_count == 98
    assert int((calibration.permutations[:, 0] == 0).sum().item()) == 1
    assert int((calibration.permutations[:, :3] == 0).any(dim=1).sum().item()) == 3

    expected_threshold = torch.quantile(
        calibration.statistics,
        0.95,
        dim=0,
        interpolation="higher",
    )
    torch.testing.assert_close(calibration.threshold, expected_threshold)

    evidence = synthetic._nra_evidence(fixture, direction, calibration)
    torch.testing.assert_close(evidence.null_threshold, calibration.threshold)
    assert evidence.alpha == calibration.alpha
    assert evidence.permutation_count == calibration.permutation_count
    assert evidence.seed == calibration.seed
    assert evidence.statistic_id == calibration.statistic_id
    assert evidence.statistic_epsilon == calibration.statistic_epsilon
    assert evidence.k == calibration.k
    assert evidence.sampler == calibration.sampler
    assert evidence.aggregation_policy == calibration.aggregation_policy
    assert evidence.weight_policy == calibration.weight_policy
    assert evidence.producer_trial_ids == calibration.producer_trial_ids
    producer_records = fixture.calibration_records.contributor_records
    assert evidence.producer_fold_ids == tuple(
        dict.fromkeys(record.fold_id for record in producer_records)
    )
    assert evidence.producer_session_ids == tuple(
        dict.fromkeys(record.session_id for record in producer_records)
    )
    assert evidence.producer_trial_ids == tuple(
        record.trial_id for record in producer_records
    )
    assert evidence.split == producer_records[0].split == "train"
    assert evidence.current_fold_id == fixture.current_anchor_record.fold_id
    assert evidence.current_session_id == fixture.current_anchor_record.session_id
    assert evidence.current_trial_id == fixture.current_anchor_record.trial_id
    assert evidence.current_fold_id not in evidence.producer_fold_ids
    assert evidence.current_session_id not in evidence.producer_session_ids
    assert evidence.current_trial_id not in evidence.producer_trial_ids


def test_fixture_has_immutable_disjoint_trial_records_and_99_label_donors() -> None:
    fixture = synthetic._build_fixture(seed=31)
    calibration = fixture.calibration_records

    assert len(calibration.neural_trials) == 3
    assert len(calibration.pronunciation_donors) == 99
    assert len(calibration.contributor_records) == 101
    assert {donor.identity for donor in calibration.pronunciation_donors} == set(
        range(99)
    )
    assert all(donor.fold_id for donor in calibration.pronunciation_donors)
    assert all(donor.session_id for donor in calibration.pronunciation_donors)
    assert all(donor.trial_id for donor in calibration.pronunciation_donors)
    assert (
        sum(hasattr(donor, "brain") for donor in calibration.pronunciation_donors) == 1
    )
    with pytest.raises(FrozenInstanceError):
        calibration.anchor_record.trial_id = "changed"  # type: ignore[misc]

    current_records = (
        fixture.current_anchor_record,
        *fixture.current_control_records,
    )
    for current in current_records:
        for heldout in calibration.neural_trials:
            assert current is not heldout
            assert current.brain is not heldout.brain
            assert current.brain.untyped_storage().data_ptr() != (
                heldout.brain.untyped_storage().data_ptr()
            )


def test_current_and_calibration_records_have_valid_shared_cohorts() -> None:
    fixture = synthetic._build_fixture(seed=31)
    current = (
        fixture.current_anchor_record,
        *fixture.current_control_records,
    )
    calibration = fixture.calibration_records.contributor_records

    for records in (current, calibration):
        assert {record.subject_id for record in records} == {fixture.subject_id}
        assert {getattr(record, "split", None) for record in records} == {"train"}
        assert len({record.fold_id for record in records}) == 1
        assert len({record.session_id for record in records}) == 1
        assert {getattr(record, "reliability_stratum", None) for record in records} == {
            fixture.reliability_stratum
        }
        assert len({record.trial_id for record in records}) == len(records)

    assert current[0].fold_id != calibration[0].fold_id
    assert current[0].session_id != calibration[0].session_id
    assert len({record.trial_id for record in (*current, *calibration)}) == 104


@pytest.mark.parametrize("field", ["session_id", "reliability_stratum"])
def test_calibration_rejects_a_contributor_outside_its_cohort(field: str) -> None:
    fixture = synthetic._build_fixture(seed=7)
    donors = fixture.calibration_records.pronunciation_donors
    changed = replace(donors[1], **{field: f"other-{field}"})

    with pytest.raises(ValueError, match=f"calibration.*share.*{field}"):
        replace(
            fixture.calibration_records,
            pronunciation_donors=(donors[0], changed, *donors[2:]),
        )


@pytest.mark.parametrize("field", ["session_id", "reliability_stratum"])
def test_current_controls_reject_a_record_outside_their_cohort(field: str) -> None:
    fixture = synthetic._build_fixture(seed=7)
    controls = fixture.current_control_records
    changed = replace(controls[0], **{field: f"other-{field}"})

    with pytest.raises(ValueError, match=f"current.*share.*{field}"):
        replace(fixture, current_control_records=(changed, controls[1]))


def test_provenance_records_reject_non_training_split() -> None:
    fixture = synthetic._build_fixture(seed=7)

    with pytest.raises(ValueError, match="split.*train"):
        replace(fixture.current_anchor_record, split="validation")


def test_unified_teacher_table_and_shift_zero_match_application_interactions() -> None:
    fixture = synthetic._build_fixture(seed=19)
    prefix, _, student_probs, _ = _reference_nra_inputs(fixture)
    calibration = fixture.calibration_records

    assert fixture.teacher.copy_vectors.shape == (101, synthetic._VOCABULARY_SIZE)
    coefficients = torch.tensor(
        [1.0, -1.0, -1.0] * 33,
        dtype=synthetic._DTYPE,
    )
    torch.testing.assert_close(
        fixture.teacher.copy_vectors[:99],
        coefficients.unsqueeze(-1) * fixture.teacher.copy_vectors[0],
    )
    assert not hasattr(fixture.teacher, "permutation_logits")

    privilege = fixture.anchor_privilege
    current_logits = fixture.teacher.teacher_logits(
        fixture.current_anchor_record.brain,
        privilege,
        prefix,
    )
    calibration_logits = fixture.teacher.teacher_logits(
        calibration.anchor_record.brain,
        privilege,
        prefix,
    )
    torch.testing.assert_close(calibration_logits, current_logits)

    identity_assignment = torch.tensor(
        [donor.identity for donor in calibration.pronunciation_donors],
        dtype=torch.long,
    )
    current_interactions = synthetic._anchor_interactions(
        fixture,
        student_probs,
        prefix,
    )
    shift_zero_interactions = synthetic._permuted_null_interactions(
        fixture,
        student_probs,
        prefix,
        identity_assignment,
    )
    torch.testing.assert_close(shift_zero_interactions, current_interactions)


def test_calibration_consumes_only_heldout_brain_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = synthetic._build_fixture(seed=13)
    prefix, _, student_probs, direction = _reference_nra_inputs(fixture)
    current_brains = {
        id(record.brain)
        for record in (
            fixture.current_anchor_record,
            *fixture.current_control_records,
        )
    }
    calibration_brains = {
        id(record.brain) for record in fixture.calibration_records.neural_trials
    }
    seen_brains: list[torch.Tensor] = []
    original_teacher_logits = fixture.teacher.teacher_logits

    def record_teacher_logits(
        brain: torch.Tensor,
        privilege: torch.Tensor,
        rollout_prefix: torch.Tensor,
    ) -> torch.Tensor:
        seen_brains.append(brain)
        return original_teacher_logits(brain, privilege, rollout_prefix)

    monkeypatch.setattr(fixture.teacher, "teacher_logits", record_teacher_logits)
    synthetic._calibrate_nra_null(fixture, student_probs, prefix, direction)

    assert seen_brains
    assert not ({id(brain) for brain in seen_brains} & current_brains)
    assert {id(brain) for brain in seen_brains} == calibration_brains


def test_calibration_is_independent_of_current_trial_brain_values() -> None:
    fixture = synthetic._build_fixture(seed=29)
    prefix, _, student_probs, direction = _reference_nra_inputs(fixture)
    expected = synthetic._calibrate_nra_null(
        fixture,
        student_probs,
        prefix,
        direction,
    )
    changed = replace(
        fixture,
        current_anchor_record=replace(
            fixture.current_anchor_record,
            brain=torch.zeros_like(fixture.current_anchor_record.brain),
        ),
        current_control_records=tuple(
            replace(record, brain=torch.zeros_like(record.brain))
            for record in fixture.current_control_records
        ),
    )

    actual = synthetic._calibrate_nra_null(
        changed,
        student_probs,
        prefix,
        direction,
    )

    torch.testing.assert_close(actual.statistics, expected.statistics)
    torch.testing.assert_close(actual.threshold, expected.threshold)


@pytest.mark.parametrize("identifier", ["fold_id", "session_id", "trial_id"])
def test_fixture_rejects_current_calibration_identifier_overlap(
    identifier: str,
) -> None:
    fixture = synthetic._build_fixture(seed=7)
    overlapping_id = getattr(fixture.current_anchor_record, identifier)
    donors = fixture.calibration_records.pronunciation_donors
    controls = fixture.calibration_records.control_records
    if identifier in {"fold_id", "session_id"}:
        bad_donors = tuple(
            replace(record, **{identifier: overlapping_id}) for record in donors
        )
        bad_controls = tuple(
            replace(record, **{identifier: overlapping_id}) for record in controls
        )
    else:
        bad_donors = (
            replace(donors[0], **{identifier: overlapping_id}),
            *donors[1:],
        )
        bad_controls = controls
    bad_calibration = replace(
        fixture.calibration_records,
        pronunciation_donors=bad_donors,
        control_records=bad_controls,
    )

    with pytest.raises(ValueError, match=f"{identifier}.*overlap"):
        replace(fixture, calibration_records=bad_calibration)


@pytest.mark.parametrize("alias_kind", ["same_tensor", "shared_storage_view"])
def test_fixture_rejects_current_calibration_brain_storage_overlap(
    alias_kind: str,
) -> None:
    fixture = synthetic._build_fixture(seed=7)
    current = fixture.current_anchor_record.brain
    aliased_brain = current if alias_kind == "same_tensor" else current[:, :]
    bad_anchor = replace(
        fixture.calibration_records.anchor_record,
        brain=aliased_brain,
    )
    bad_calibration = replace(
        fixture.calibration_records,
        pronunciation_donors=(
            bad_anchor,
            *fixture.calibration_records.pronunciation_donors[1:],
        ),
    )

    with pytest.raises(ValueError, match="brain storage.*overlap"):
        replace(fixture, calibration_records=bad_calibration)


def test_registered_copier_identity_uses_the_real_teacher_and_null_path() -> None:
    copier_fixture = synthetic._build_fixture(
        seed=0,
        anchor_identity=synthetic._COPIER_ONLY_IDENTITY,
    )
    prefix, student_log_probs, student_probs, direction = _reference_nra_inputs(
        copier_fixture
    )

    assert (
        torch.count_nonzero(
            copier_fixture.teacher.matching_vectors[synthetic._COPIER_ONLY_IDENTITY]
        ).item()
        == 0
    )
    assert torch.count_nonzero(direction).item() > 0
    interactions = synthetic._anchor_interactions(
        copier_fixture,
        student_probs,
        prefix,
    )
    expected_alignment = synthetic._signed_fisher_alignment(
        interactions,
        direction,
        student_probs,
    )
    target, diagnostics = synthetic._nra_target(
        copier_fixture,
        student_log_probs,
        prefix,
    )

    torch.testing.assert_close(diagnostics["alignment_score"], expected_alignment)
    torch.testing.assert_close(target, student_probs)
    assert bool((expected_alignment <= diagnostics["null_threshold"]).all())
    assert float(expected_alignment.max().item()) == pytest.approx(
        0.3977044,
        abs=1e-6,
    )
    assert float(diagnostics["null_threshold"].min().item()) == pytest.approx(
        0.3977044,
        abs=1e-6,
    )
    assert not hasattr(synthetic, "_copying_interactions")
    assert (
        "anchor_interactions" not in inspect.signature(synthetic._nra_target).parameters
    )

    supported_fixture = synthetic._build_fixture(seed=0)
    supported_prefix, supported_log_probs, supported_probs, supported_direction = (
        _reference_nra_inputs(supported_fixture)
    )
    supported_target, supported_diagnostics = synthetic._nra_target(
        supported_fixture,
        supported_log_probs,
        supported_prefix,
    )
    assert not torch.allclose(supported_target, supported_probs)
    assert bool(
        (
            supported_diagnostics["alignment_score"]
            > supported_diagnostics["null_threshold"]
        ).all()
    )
    assert float(supported_diagnostics["alignment_score"].min().item()) == (
        pytest.approx(0.8953334, abs=1e-6)
    )
    assert float(supported_diagnostics["kappa"].min().item()) == pytest.approx(
        0.8262206,
        abs=1e-6,
    )
    assert torch.count_nonzero(supported_direction).item() > 0


def test_calibration_statistic_matches_the_nra_application_diagnostic() -> None:
    fixture = synthetic._build_fixture(seed=19)
    prefix = synthetic._greedy_student_prefix(
        fixture.student,
        fixture.anchor_brain,
        token_count=fixture.prefix.shape[1],
    )
    student_log_probs = torch.log_softmax(
        fixture.student.student_logits(fixture.anchor_brain, prefix),
        dim=-1,
    )
    student_probs = student_log_probs.detach().exp()
    interactions = synthetic._anchor_interactions(fixture, student_probs, prefix)
    direction = synthetic._brain_only_direction(fixture, prefix)

    expected = synthetic._signed_fisher_alignment(
        interactions,
        direction,
        student_probs,
    )
    _, diagnostics = synthetic._nra_target(fixture, student_log_probs, prefix)

    torch.testing.assert_close(diagnostics["alignment_score"], expected)
    assert (
        "include_matching_effect"
        not in inspect.signature(synthetic._FrozenTeacher.teacher_logits).parameters
    )
    assert "null_threshold" not in inspect.signature(synthetic._nra_target).parameters
