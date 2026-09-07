import inspect

import pytest
import torch

from brain_evidence.recoverability import NRAInteractionBatch
from brain_evidence.recoverability.projection import NRAEvidence, recoverable_target


def _nra_evidence(**overrides: object) -> NRAEvidence:
    values = {
        "brain_only_direction": torch.tensor([[1.0, -1.0]]),
        "null_threshold": 0.0,
        "source": "frozen_crossfit_ce",
        "split": "train",
        "subject_id": "subject-1",
        "pronunciation_id": "pronunciation-cat",
        "reliability_stratum": "high",
        "producer_fold_ids": ("fold-fit-a", "fold-fit-b"),
        "current_fold_id": "fold-eval",
        "producer_session_ids": ("session-fit-a", "session-fit-b"),
        "current_session_id": "session-eval",
        "producer_trial_ids": ("trial-fit-a", "trial-fit-b"),
        "current_trial_id": "trial-eval",
        "checkpoint_id": "ce-checkpoint-a",
        "statistic_id": "signed_fisher_alignment",
        "statistic_epsilon": 1.0e-8,
        "alpha": 0.05,
        "phase": "pilot",
        "permutation_count": 99,
        "k": 1,
        "sampler": "anchor-local-v1",
        "aggregation_policy": "mean",
        "weight_policy": "uniform",
        "seed": 41,
        "control_policy": "anchor_local_three_mismatch",
        "producer_prefix_policy": "student-rollout",
        "current_prefix_policy": "student-rollout",
        "producer_alignment_policy": "token-index-v1",
        "current_alignment_policy": "token-index-v1",
        "producer_support_policy": "full-vocabulary",
        "current_support_policy": "full-vocabulary",
    }
    values.update(overrides)
    return NRAEvidence(**values)


def _nra_interactions(
    interactions: torch.Tensor, **overrides: object
) -> NRAInteractionBatch:
    values = {
        "values": interactions,
        "subject_id": "subject-1",
        "pronunciation_id": "pronunciation-cat",
        "reliability_stratum": "high",
        "current_fold_id": "fold-eval",
        "current_session_id": "session-eval",
        "current_trial_id": "trial-eval",
        "checkpoint_id": "ce-checkpoint-a",
        "prefix_policy": "student-rollout",
        "alignment_policy": "token-index-v1",
        "support_policy": "full-vocabulary",
        "sampler": "anchor-local-v1",
        "aggregation_policy": "mean",
        "weight_policy": "uniform",
        "seed": 41,
        "control_policy": "anchor_local_three_mismatch",
    }
    values.update(overrides)
    return NRAInteractionBatch(**values)


@pytest.mark.parametrize("keyword", ["anchor_interactions", "factorial_interactions"])
@pytest.mark.parametrize("alternate_context", [False, True])
def test_nra_mode_accepts_a_matching_provenanced_batch(
    keyword: str, alternate_context: bool
) -> None:
    interactions = torch.tensor([[[1.0, -1.0], [3.0, -3.0]]])
    batch_context = {}
    evidence_context = {}
    if alternate_context:
        batch_context = {
            "current_fold_id": "fold-other",
            "current_session_id": "session-other",
            "current_trial_id": "trial-other",
            "checkpoint_id": "ce-checkpoint-b",
            "prefix_policy": "student-rollout-v2",
            "alignment_policy": "token-index-v2",
            "support_policy": "shared-top-k-plus-residual",
        }
        evidence_context = {
            "current_fold_id": "fold-other",
            "current_session_id": "session-other",
            "current_trial_id": "trial-other",
            "checkpoint_id": "ce-checkpoint-b",
            "producer_prefix_policy": "student-rollout-v2",
            "current_prefix_policy": "student-rollout-v2",
            "producer_alignment_policy": "token-index-v2",
            "current_alignment_policy": "token-index-v2",
            "producer_support_policy": "shared-top-k-plus-residual",
            "current_support_policy": "shared-top-k-plus-residual",
        }

    target, diagnostics = recoverable_target(
        mode="nra",
        student_log_probs=torch.tensor([[0.5, 0.5]]).log(),
        teacher_log_probs=torch.log_softmax(torch.tensor([[1.0, -1.0]]), dim=-1),
        nra_evidence=_nra_evidence(k=2, statistic_epsilon=0.0, **evidence_context),
        neural_reliability=1.0,
        session_stability=1.0,
        epsilon=0.0,
        **{keyword: _nra_interactions(interactions, **batch_context)},
    )

    torch.testing.assert_close(target, torch.tensor([[0.8807971, 0.1192029]]))
    torch.testing.assert_close(
        diagnostics["aggregate_interaction"], torch.tensor([[2.0, -2.0]])
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("subject_id", "subject-2"),
        ("pronunciation_id", "pronunciation-dog"),
        ("reliability_stratum", "low"),
        ("sampler", "anchor-local-v2"),
        ("aggregation_policy", "median"),
        ("weight_policy", "reliability"),
        ("seed", 42),
        ("control_policy", "symmetric_diagonal"),
    ],
)
def test_nra_mode_rejects_interaction_identity_mismatch(
    field: str, value: object
) -> None:
    student_log_probs = torch.tensor([[0.5, 0.5]]).log()

    with pytest.raises(ValueError, match=field):
        recoverable_target(
            mode="nra",
            student_log_probs=student_log_probs,
            teacher_log_probs=student_log_probs,
            anchor_interactions=_nra_interactions(
                torch.ones(1, 1, 2), **{field: value}
            ),
            nra_evidence=_nra_evidence(),
            neural_reliability=1.0,
            session_stability=1.0,
        )


@pytest.mark.parametrize("keyword", ["anchor_interactions", "factorial_interactions"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("current_fold_id", "fold-fit-a"),
        ("current_session_id", "session-fit-a"),
        ("current_trial_id", "trial-fit-a"),
        ("checkpoint_id", "ce-checkpoint-b"),
        ("prefix_policy", "student-rollout-v2"),
        ("alignment_policy", "token-index-v2"),
        ("support_policy", "shared-top-k-plus-residual"),
    ],
)
def test_nra_mode_rejects_batch_current_context_mismatch(
    keyword: str, field: str, value: str
) -> None:
    student_log_probs = torch.tensor([[0.5, 0.5]]).log()
    batch = _nra_interactions(torch.tensor([[[1.0, -1.0]]]), **{field: value})

    with pytest.raises(ValueError, match=field):
        recoverable_target(
            mode="nra",
            student_log_probs=student_log_probs,
            teacher_log_probs=torch.log_softmax(torch.tensor([[1.0, -1.0]]), dim=-1),
            nra_evidence=_nra_evidence(),
            neural_reliability=1.0,
            session_stability=1.0,
            **{keyword: batch},
        )


@pytest.mark.parametrize(
    "field",
    [
        "current_fold_id",
        "current_session_id",
        "current_trial_id",
        "checkpoint_id",
        "prefix_policy",
        "alignment_policy",
        "support_policy",
    ],
)
@pytest.mark.parametrize("value", ["", " \t", None, 1])
def test_nra_interaction_batch_rejects_invalid_current_context(
    field: str, value: object
) -> None:
    with pytest.raises((TypeError, ValueError), match=field):
        _nra_interactions(torch.ones(1, 1, 2), **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("subject_id", " subject-1"),
        ("pronunciation_id", "pronunciation-cat "),
        ("reliability_stratum", "high "),
        ("current_fold_id", "fold-eval "),
        ("current_session_id", "session-eval "),
        ("current_trial_id", "trial-eval "),
        ("checkpoint_id", "ce-checkpoint-a "),
        ("prefix_policy", "student-rollout "),
        ("alignment_policy", "token-index-v1 "),
        ("support_policy", "full-vocabulary "),
        ("sampler", "anchor-local-v1 "),
        ("aggregation_policy", "mean "),
        ("weight_policy", "uniform "),
        ("control_policy", "anchor_local_three_mismatch "),
    ],
)
def test_nra_interaction_batch_rejects_surrounding_identity_whitespace(
    field: str,
    value: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        _nra_interactions(torch.ones(1, 1, 2), **{field: value})


@pytest.mark.parametrize("weights", [None, torch.tensor([1.0, 9.0])])
def test_nra_mode_rejects_matching_but_unverifiable_weight_policy(
    weights: torch.Tensor | None,
) -> None:
    student_log_probs = torch.tensor([[0.5, 0.5]]).log()

    with pytest.raises(ValueError, match="weight_policy.*uniform"):
        recoverable_target(
            mode="nra",
            student_log_probs=student_log_probs,
            teacher_log_probs=student_log_probs,
            anchor_interactions=_nra_interactions(
                torch.ones(1, 2, 2), weights=weights, weight_policy="reliability"
            ),
            nra_evidence=_nra_evidence(k=2, weight_policy="reliability"),
            neural_reliability=1.0,
            session_stability=1.0,
        )


@pytest.mark.parametrize("aggregation", ["mean", "median"])
@pytest.mark.parametrize(
    "weights",
    [
        torch.tensor([1.0, 9.0]),
        torch.tensor([[1.0, 9.0]]),
        torch.tensor([[[1.0], [9.0]]]),
        torch.tensor([1.0, 1.0000000001], dtype=torch.float64),
    ],
)
def test_nra_mode_rejects_nonuniform_actual_batch_weights(
    aggregation: str, weights: torch.Tensor
) -> None:
    student_log_probs = torch.tensor([[0.5, 0.5]]).log()

    with pytest.raises(ValueError, match="uniform.*weights|weights.*uniform"):
        recoverable_target(
            mode="nra",
            student_log_probs=student_log_probs,
            teacher_log_probs=student_log_probs,
            anchor_interactions=_nra_interactions(
                torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]]),
                weights=weights,
                aggregation_policy=aggregation,
            ),
            nra_evidence=_nra_evidence(k=2, aggregation_policy=aggregation),
            neural_reliability=1.0,
            session_stability=1.0,
            aggregation=aggregation,
        )


@pytest.mark.parametrize("aggregation", ["mean", "median"])
@pytest.mark.parametrize(
    "weights",
    [
        None,
        torch.tensor([9.0, 9.0]),
        torch.tensor([[2, 2]]),
        torch.tensor([[[3.0], [3.0]]]),
        torch.tensor([1.0e300, 1.0e300], dtype=torch.float64),
        torch.tensor([1.0e-300, 1.0e-300], dtype=torch.float64),
    ],
)
def test_nra_mode_accepts_numerically_uniform_batch_weights(
    aggregation: str, weights: torch.Tensor | None
) -> None:
    target, diagnostics = recoverable_target(
        mode="nra",
        student_log_probs=torch.tensor([[0.5, 0.5]]).log(),
        teacher_log_probs=torch.log_softmax(torch.tensor([[1.0, -1.0]]), dim=-1),
        anchor_interactions=_nra_interactions(
            torch.tensor([[[1.0, -1.0], [3.0, -3.0]]]),
            weights=weights,
            aggregation_policy=aggregation,
        ),
        nra_evidence=_nra_evidence(
            k=2,
            aggregation_policy=aggregation,
            statistic_epsilon=0.0,
        ),
        neural_reliability=1.0,
        session_stability=1.0,
        aggregation=aggregation,
        epsilon=0.0,
    )

    torch.testing.assert_close(target, torch.tensor([[0.8807971, 0.1192029]]))
    torch.testing.assert_close(
        diagnostics["aggregate_interaction"], torch.tensor([[2.0, -2.0]])
    )


@pytest.mark.parametrize("weights", [torch.tensor([1.0]), torch.tensor([9.0])])
def test_nra_mode_rejects_separate_interaction_weights(weights: torch.Tensor) -> None:
    student_log_probs = torch.tensor([[0.5, 0.5]]).log()

    with pytest.raises(ValueError, match="interaction_weights"):
        recoverable_target(
            mode="nra",
            student_log_probs=student_log_probs,
            teacher_log_probs=student_log_probs,
            anchor_interactions=_nra_interactions(torch.ones(1, 1, 2)),
            nra_evidence=_nra_evidence(),
            neural_reliability=1.0,
            session_stability=1.0,
            interaction_weights=weights,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("subject_id", ""),
        ("pronunciation_id", " "),
        ("reliability_stratum", ""),
        ("sampler", ""),
        ("aggregation_policy", ""),
        ("weight_policy", ""),
        ("control_policy", ""),
        ("subject_id", 1),
        ("seed", True),
        ("seed", 41.0),
        ("values", [[[1.0, -1.0]]]),
        ("weights", [1.0]),
    ],
)
def test_nra_interaction_batch_rejects_invalid_typed_identity(
    field: str, value: object
) -> None:
    with pytest.raises((TypeError, ValueError), match=field):
        _nra_interactions(torch.ones(1, 1, 2), **{field: value})


def test_recoverable_target_requires_an_explicit_mode() -> None:
    student_log_probs = torch.tensor([[0.5, 0.5]]).log()

    with pytest.raises(TypeError):
        recoverable_target(
            student_log_probs=student_log_probs,
            teacher_log_probs=student_log_probs,
            anchor_interactions=torch.zeros(1, 1, 2),
            neural_reliability=1.0,
            session_stability=1.0,
        )


def test_recoverable_target_gate_defaults_are_fail_closed() -> None:
    parameters = inspect.signature(recoverable_target).parameters

    assert parameters["neural_reliability"].default is None
    assert parameters["session_stability"].default is None


@pytest.mark.parametrize(
    ("provided_gate", "missing_gate"),
    [
        ({"session_stability": 1.0}, "neural_reliability"),
        ({"neural_reliability": 1.0}, "session_stability"),
    ],
)
def test_nra_mode_requires_each_explicit_transfer_gate(
    provided_gate: dict[str, float],
    missing_gate: str,
) -> None:
    student_log_probs = torch.tensor([[0.5, 0.5]]).log()

    with pytest.raises(ValueError, match=missing_gate):
        recoverable_target(
            mode="nra",
            student_log_probs=student_log_probs,
            teacher_log_probs=student_log_probs,
            anchor_interactions=_nra_interactions(torch.zeros(1, 1, 2)),
            nra_evidence=_nra_evidence(),
            **provided_gate,
        )


def test_interaction_only_mode_defaults_to_neutral_transfer_gates() -> None:
    student_log_probs = torch.tensor([[0.5, 0.5]]).log()
    inputs = {
        "mode": "interaction_only_ablation",
        "student_log_probs": student_log_probs,
        "teacher_log_probs": torch.log_softmax(
            torch.tensor([[1.0, -1.0]]),
            dim=-1,
        ),
        "anchor_interactions": torch.tensor([[[1.0, -1.0]]]),
    }

    default_target, default_diagnostics = recoverable_target(**inputs)
    explicit_target, explicit_diagnostics = recoverable_target(
        **inputs,
        neural_reliability=1.0,
        session_stability=1.0,
    )

    torch.testing.assert_close(default_target, explicit_target)
    torch.testing.assert_close(
        default_diagnostics["reliability"],
        explicit_diagnostics["reliability"],
    )
    torch.testing.assert_close(
        default_diagnostics["stability"],
        explicit_diagnostics["stability"],
    )


def test_nra_mode_requires_a_complete_evidence_artifact() -> None:
    student_log_probs = torch.tensor([[0.5, 0.5]]).log()

    with pytest.raises(ValueError, match="NRAEvidence"):
        recoverable_target(
            mode="nra",
            student_log_probs=student_log_probs,
            teacher_log_probs=student_log_probs,
            anchor_interactions=torch.zeros(1, 1, 2),
            neural_reliability=1.0,
            session_stability=1.0,
        )


def test_nra_mode_rejects_raw_evidence_arguments() -> None:
    student_log_probs = torch.tensor([[0.5, 0.5]]).log()

    with pytest.raises(TypeError, match="brain_only_direction"):
        recoverable_target(
            mode="nra",
            student_log_probs=student_log_probs,
            teacher_log_probs=student_log_probs,
            anchor_interactions=torch.zeros(1, 1, 2),
            brain_only_direction=torch.zeros(1, 2),
            null_threshold=0.0,
            neural_reliability=1.0,
            session_stability=1.0,
        )


@pytest.mark.parametrize("keyword", ["anchor_interactions", "factorial_interactions"])
def test_nra_mode_requires_provenanced_interactions(keyword: str) -> None:
    student_log_probs = torch.tensor([[0.5, 0.5]]).log()

    with pytest.raises(TypeError, match="NRAInteractionBatch"):
        recoverable_target(
            mode="nra",
            student_log_probs=student_log_probs,
            teacher_log_probs=student_log_probs,
            nra_evidence=_nra_evidence(),
            neural_reliability=1.0,
            session_stability=1.0,
            **{keyword: torch.zeros(1, 1, 2)},
        )


def test_nra_mode_cannot_apply_uniform_calibration_to_raw_nonuniform_weights() -> None:
    student_log_probs = torch.tensor([[0.5, 0.5]]).log()

    with pytest.raises((TypeError, ValueError), match="NRAInteractionBatch|weights"):
        recoverable_target(
            mode="nra",
            student_log_probs=student_log_probs,
            teacher_log_probs=student_log_probs,
            anchor_interactions=torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]]),
            nra_evidence=_nra_evidence(k=2),
            neural_reliability=1.0,
            session_stability=1.0,
            interaction_weights=torch.tensor([1.0, 9.0]),
        )


def test_interaction_only_mode_rejects_brain_only_evidence() -> None:
    student_log_probs = torch.tensor([[0.5, 0.5]]).log()

    with pytest.raises(ValueError, match="interaction_only_ablation"):
        recoverable_target(
            mode="interaction_only_ablation",
            student_log_probs=student_log_probs,
            teacher_log_probs=student_log_probs,
            anchor_interactions=torch.zeros(1, 1, 2),
            nra_evidence=_nra_evidence(),
            neural_reliability=1.0,
            session_stability=1.0,
        )


def test_nra_mode_rejects_k_mismatched_to_current_interactions() -> None:
    student_log_probs = torch.tensor([[0.5, 0.5]]).log()

    with pytest.raises(ValueError, match="k does not match"):
        recoverable_target(
            mode="nra",
            student_log_probs=student_log_probs,
            teacher_log_probs=student_log_probs,
            anchor_interactions=_nra_interactions(torch.zeros(1, 1, 2)),
            nra_evidence=_nra_evidence(k=2),
            neural_reliability=1.0,
            session_stability=1.0,
        )


def test_nra_mode_rejects_aggregation_policy_mismatch() -> None:
    student_log_probs = torch.tensor([[0.5, 0.5]]).log()

    with pytest.raises(ValueError, match="aggregation policy"):
        recoverable_target(
            mode="nra",
            student_log_probs=student_log_probs,
            teacher_log_probs=student_log_probs,
            anchor_interactions=_nra_interactions(
                torch.zeros(1, 1, 2), aggregation_policy="median"
            ),
            nra_evidence=_nra_evidence(aggregation_policy="median"),
            neural_reliability=1.0,
            session_stability=1.0,
            aggregation="mean",
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"source": "current_student"}, "frozen_crossfit_ce"),
        ({"split": "validation"}, "training split"),
        ({"producer_fold_ids": ("fold-fit", "fold-eval")}, "fold"),
        ({"producer_session_ids": ("session-fit", "session-eval")}, "session"),
        ({"producer_trial_ids": ("trial-fit", "trial-eval")}, "trial"),
        ({"checkpoint_id": ""}, "checkpoint_id"),
        ({"statistic_id": "absolute_cosine"}, "signed_fisher_alignment"),
        ({"alpha": 0.049}, "0.05"),
        ({"control_policy": "symmetric_diagonal"}, "three_mismatch"),
        ({"producer_prefix_policy": "gold-prefix"}, "prefix policy"),
        ({"producer_alignment_policy": "word-aligned"}, "alignment policy"),
        ({"producer_support_policy": "top-k"}, "support policy"),
    ],
)
def test_nra_evidence_rejects_leakage_or_unregistered_provenance(
    override: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _nra_evidence(**override)


@pytest.mark.parametrize(
    "field",
    [
        "subject_id",
        "pronunciation_id",
        "reliability_stratum",
        "sampler",
        "aggregation_policy",
        "weight_policy",
        "current_fold_id",
        "current_session_id",
        "current_trial_id",
    ],
)
def test_nra_evidence_rejects_empty_calibration_identity(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _nra_evidence(**{field: ""})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("subject_id", " subject-1"),
        ("pronunciation_id", "pronunciation-cat "),
        ("reliability_stratum", "high "),
        ("current_fold_id", "fold-eval "),
        ("current_session_id", "session-eval "),
        ("current_trial_id", "trial-eval "),
        ("checkpoint_id", "ce-checkpoint-a "),
        ("sampler", "anchor-local-v1 "),
        ("aggregation_policy", "mean "),
        ("weight_policy", "uniform "),
    ],
)
def test_nra_evidence_rejects_surrounding_identity_whitespace(
    field: str,
    value: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        _nra_evidence(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("producer_fold_ids", (" fold-fit-a", "fold-fit-b")),
        ("producer_session_ids", ("session-fit-a", "session-eval ")),
        ("producer_trial_ids", ("trial-fit-a", "trial-eval ")),
    ],
)
def test_nra_evidence_rejects_surrounding_whitespace_in_producer_ids(
    field: str,
    value: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match=field):
        _nra_evidence(**{field: value})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "producer_prefix_policy": " student-rollout",
                "current_prefix_policy": " student-rollout",
            },
            "producer_prefix_policy",
        ),
        (
            {
                "producer_alignment_policy": "token-index-v1 ",
                "current_alignment_policy": "token-index-v1 ",
            },
            "producer_alignment_policy",
        ),
        (
            {
                "producer_support_policy": "full-vocabulary ",
                "current_support_policy": "full-vocabulary ",
            },
            "producer_support_policy",
        ),
    ],
)
def test_nra_evidence_rejects_matching_policies_with_surrounding_whitespace(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _nra_evidence(**overrides)


@pytest.mark.parametrize(
    "field",
    ["producer_fold_ids", "producer_session_ids", "producer_trial_ids"],
)
def test_nra_evidence_requires_nonempty_tuple_membership_sets(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _nra_evidence(**{field: ()})

    with pytest.raises(TypeError, match=field):
        _nra_evidence(**{field: ["fit-a"]})


@pytest.mark.parametrize(
    "field",
    ["producer_fold_ids", "producer_session_ids", "producer_trial_ids"],
)
def test_nra_evidence_requires_distinct_nonempty_producer_ids(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _nra_evidence(**{field: ("fit-a", "fit-a")})

    with pytest.raises(ValueError, match=field):
        _nra_evidence(**{field: ("fit-a", "")})

    with pytest.raises(TypeError, match=field):
        _nra_evidence(**{field: ("fit-a", 3)})


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"k": True}, "k"),
        ({"k": 1.5}, "k"),
        ({"k": 0}, "k"),
        ({"seed": True}, "seed"),
        ({"seed": 41.5}, "seed"),
        ({"permutation_count": True}, "permutation_count"),
    ],
)
def test_nra_evidence_rejects_invalid_numeric_identity(
    override: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _nra_evidence(**override)


@pytest.mark.parametrize(
    "statistic_epsilon",
    [True, False, "1e-8", -1.0e-8, float("nan"), float("inf")],
)
def test_nra_evidence_rejects_invalid_statistic_epsilon(
    statistic_epsilon: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="statistic_epsilon"):
        _nra_evidence(statistic_epsilon=statistic_epsilon)


@pytest.mark.parametrize(
    ("statistic_epsilon", "use_epsilon"),
    [(0.0, 1.0e-8), (1.0e-8, 0.0)],
)
def test_nra_mode_rejects_epsilon_not_bound_to_exact_statistic(
    statistic_epsilon: float,
    use_epsilon: float,
) -> None:
    """A tiny direction makes an epsilon change scientifically material."""
    tiny_direction = torch.tensor([[1.0e-6, -1.0e-6]], dtype=torch.float64)

    with pytest.raises(ValueError, match="epsilon.*NRA evidence|NRA evidence.*epsilon"):
        recoverable_target(
            mode="nra",
            student_log_probs=torch.tensor([[0.5, 0.5]], dtype=torch.float64).log(),
            teacher_log_probs=torch.log_softmax(tiny_direction, dim=-1),
            anchor_interactions=_nra_interactions(tiny_direction.unsqueeze(-2)),
            nra_evidence=_nra_evidence(
                brain_only_direction=tiny_direction,
                statistic_epsilon=statistic_epsilon,
            ),
            neural_reliability=1.0,
            session_stability=1.0,
            epsilon=use_epsilon,
        )


def test_recoverable_target_rejects_boolean_epsilon() -> None:
    student_log_probs = torch.tensor([[0.5, 0.5]]).log()

    with pytest.raises(TypeError, match="epsilon"):
        recoverable_target(
            mode="interaction_only_ablation",
            student_log_probs=student_log_probs,
            teacher_log_probs=student_log_probs,
            anchor_interactions=torch.zeros(1, 1, 2),
            epsilon=True,
        )


@pytest.mark.parametrize(
    ("phase", "permutation_count", "minimum"),
    [("pilot", 98, 99), ("confirmatory", 998, 999)],
)
def test_nra_evidence_enforces_phase_specific_permutation_minimum(
    phase: str,
    permutation_count: int,
    minimum: int,
) -> None:
    with pytest.raises(ValueError, match=str(minimum)):
        _nra_evidence(phase=phase, permutation_count=permutation_count)


def test_nra_evidence_requires_confirmatory_k_of_at_least_four() -> None:
    with pytest.raises(ValueError, match="at least 4"):
        _nra_evidence(phase="confirmatory", permutation_count=999, k=3)


def test_nra_evidence_accepts_pilot_k_of_one() -> None:
    evidence = _nra_evidence(k=1)

    assert evidence.k == 1


def test_nra_evidence_accepts_registered_confirmatory_provenance() -> None:
    evidence = _nra_evidence(
        phase="confirmatory",
        permutation_count=999,
        k=4,
    )

    assert evidence.phase == "confirmatory"
    assert evidence.permutation_count == 999
    assert evidence.k == 4


def test_recoverable_target_reconstructs_an_aligned_correction() -> None:
    student_log_probs = torch.tensor([[0.5, 0.5]]).log()
    teacher_log_probs = torch.log_softmax(torch.tensor([[1.0, -1.0]]), dim=-1)
    interactions = torch.tensor([[[1.0, -1.0], [1.0, -1.0]]])

    target, diagnostics = recoverable_target(
        mode="interaction_only_ablation",
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        anchor_interactions=interactions,
        neural_reliability=torch.tensor([1.0]),
        session_stability=torch.tensor([1.0]),
        epsilon=0.0,
    )

    torch.testing.assert_close(
        target,
        torch.tensor([[0.8807971, 0.1192029]]),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        diagnostics["projection_coefficient"], torch.tensor([1.0])
    )
    torch.testing.assert_close(
        diagnostics["brain_only_filter_applied"], torch.tensor(0.0)
    )


def test_recoverable_target_removes_an_orthogonal_correction() -> None:
    student_log_probs = torch.full((1, 4), 0.25).log()
    teacher_log_probs = torch.log_softmax(
        torch.tensor([[1.0, -1.0, 0.0, 0.0]]),
        dim=-1,
    )
    interactions = torch.tensor([[[0.0, 0.0, 1.0, -1.0]]])

    target, _ = recoverable_target(
        mode="interaction_only_ablation",
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        factorial_interactions=interactions,
        neural_reliability=1.0,
        session_stability=1.0,
        epsilon=0.0,
    )

    torch.testing.assert_close(target, torch.full((1, 4), 0.25))


def test_recoverable_target_rejects_an_oppositely_aligned_correction() -> None:
    student_log_probs = torch.tensor([[0.5, 0.5]]).log()
    teacher_log_probs = torch.log_softmax(torch.tensor([[1.0, -1.0]]), dim=-1)
    interactions = torch.tensor([[[-1.0, 1.0]]])

    target, diagnostics = recoverable_target(
        mode="interaction_only_ablation",
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        factorial_interactions=interactions,
        neural_reliability=1.0,
        session_stability=1.0,
        epsilon=0.0,
    )

    torch.testing.assert_close(target, torch.tensor([[0.5, 0.5]]))
    torch.testing.assert_close(
        diagnostics["projection_coefficient"], torch.tensor([0.0])
    )


def test_recoverable_target_handles_zero_interactions_without_nan() -> None:
    student_log_probs = torch.tensor([[0.5, 0.5]]).log()
    teacher_log_probs = torch.log_softmax(torch.tensor([[1.0, -1.0]]), dim=-1)
    interactions = torch.zeros(1, 2, 2)

    target, diagnostics = recoverable_target(
        mode="interaction_only_ablation",
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        factorial_interactions=interactions,
        neural_reliability=1.0,
        session_stability=1.0,
    )

    torch.testing.assert_close(target, torch.tensor([[0.5, 0.5]]))
    assert torch.isfinite(target).all()
    assert all(torch.isfinite(value).all() for value in diagnostics.values())


def test_recoverable_target_clips_the_projected_correction() -> None:
    student_log_probs = torch.tensor([[0.5, 0.5]]).log()
    teacher_log_probs = torch.log_softmax(
        torch.tensor([[10.0, -10.0]]),
        dim=-1,
    )
    interactions = torch.tensor([[[10.0, -10.0]]])

    target, diagnostics = recoverable_target(
        mode="interaction_only_ablation",
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        factorial_interactions=interactions,
        neural_reliability=1.0,
        session_stability=1.0,
        clip_value=0.25,
        epsilon=0.0,
    )

    torch.testing.assert_close(
        target,
        torch.tensor([[0.62245935, 0.37754068]]),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        diagnostics["clipped_correction"], torch.tensor([[0.25, -0.25]])
    )


def _unequal_four_dimensional_clipping() -> dict[str, torch.Tensor]:
    student_log_probs = torch.full((1, 2, 4), 0.25, dtype=torch.float64).log()
    supported_directions = torch.tensor(
        [[[12.0, 4.0, -6.0, -10.0], [1.0, 0.5, -0.25, -1.25]]],
        dtype=torch.float64,
    )

    _, diagnostics = recoverable_target(
        mode="interaction_only_ablation",
        student_log_probs=student_log_probs,
        teacher_log_probs=torch.log_softmax(supported_directions, dim=-1),
        anchor_interactions=supported_directions.unsqueeze(-2),
        clip_value=3.0,
        epsilon=0.0,
    )
    return dict(diagnostics)


def test_vector_clipping_preserves_each_supported_span() -> None:
    diagnostics = _unequal_four_dimensional_clipping()
    correction = diagnostics["recoverable_correction"]
    clipped = diagnostics["clipped_correction"]

    torch.testing.assert_close(
        clipped,
        correction * torch.tensor([[[0.25], [1.0]]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        clipped[..., :1] * correction[..., 1:],
        correction[..., :1] * clipped[..., 1:],
    )


def test_vector_clipping_bounds_each_token_and_reports_its_scale() -> None:
    diagnostics = _unequal_four_dimensional_clipping()

    assert bool((diagnostics["clipped_correction"].abs().amax(dim=-1) <= 3.0).all())
    torch.testing.assert_close(
        diagnostics["clip_scale"],
        torch.tensor([[0.25, 1.0]], dtype=torch.float64),
    )


def test_recoverable_target_applies_reliability_and_stability() -> None:
    student_log_probs = torch.tensor([[0.5, 0.5]]).log()
    teacher_log_probs = torch.log_softmax(torch.tensor([[1.0, -1.0]]), dim=-1)
    interactions = torch.tensor([[[1.0, -1.0]]])

    target, diagnostics = recoverable_target(
        mode="interaction_only_ablation",
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        factorial_interactions=interactions,
        neural_reliability=torch.tensor([0.5]),
        session_stability=torch.tensor([0.4]),
        epsilon=0.0,
    )

    torch.testing.assert_close(
        target,
        torch.tensor([[0.59868765, 0.40131235]]),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(diagnostics["reliability"], torch.tensor([0.5]))
    torch.testing.assert_close(diagnostics["stability"], torch.tensor([0.4]))


def test_recoverable_target_is_normalized_for_arbitrary_leading_axes() -> None:
    student_log_probs = torch.full((2, 3, 4), 0.25).log()
    teacher_log_probs = student_log_probs.clone()
    interactions = torch.arange(48.0).reshape(2, 3, 2, 4)

    target, _ = recoverable_target(
        mode="interaction_only_ablation",
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        factorial_interactions=interactions,
        neural_reliability=torch.ones(2, 3),
        session_stability=torch.ones(2, 3),
    )

    assert target.shape == (2, 3, 4)
    torch.testing.assert_close(target.sum(dim=-1), torch.ones(2, 3))
    assert bool((target >= 0).all())


@pytest.mark.parametrize(
    ("aggregation", "interactions", "expected"),
    [
        (
            "mean",
            torch.tensor([[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]]),
            torch.tensor([[0.1, 0.1, -0.9]]),
        ),
        (
            "median",
            torch.tensor([[[-2.0, -2.0, 18.0], [-2.0, 0.0, 12.0], [-1.0, -2.0, 12.0]]]),
            torch.tensor([[-1.4, -1.4, 12.6]]),
        ),
    ],
)
def test_recoverable_target_recenters_after_aggregation(
    aggregation: str,
    interactions: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    student_probs = torch.tensor([[0.6, 0.3, 0.1]])

    _, diagnostics = recoverable_target(
        mode="interaction_only_ablation",
        student_log_probs=student_probs.log(),
        teacher_log_probs=student_probs.log(),
        anchor_interactions=interactions,
        neural_reliability=1.0,
        session_stability=1.0,
        aggregation=aggregation,
    )

    torch.testing.assert_close(
        diagnostics["aggregate_interaction"],
        expected,
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        (diagnostics["aggregate_interaction"] * student_probs).sum(dim=-1),
        torch.tensor([0.0]),
        atol=1e-6,
        rtol=0.0,
    )


@pytest.mark.parametrize("keyword", ["anchor_interactions", "factorial_interactions"])
@pytest.mark.parametrize(
    "null_interaction", [torch.zeros(1, 2), torch.tensor([[1.0, -1.0]])]
)
def test_nra_mode_rejects_raw_null_interaction(
    keyword: str, null_interaction: torch.Tensor
) -> None:
    with pytest.raises(ValueError, match="null_interaction"):
        recoverable_target(
            mode="nra",
            student_log_probs=torch.tensor([[0.5, 0.5]]).log(),
            teacher_log_probs=torch.log_softmax(torch.tensor([[1.0, -1.0]]), dim=-1),
            nra_evidence=_nra_evidence(),
            null_interaction=null_interaction,
            neural_reliability=1.0,
            session_stability=1.0,
            **{keyword: _nra_interactions(torch.tensor([[[1.0, -1.0]]]))},
        )


def test_recoverable_target_subtracts_a_null_before_aggregation() -> None:
    student_log_probs = torch.tensor([[0.5, 0.5]]).log()
    teacher_log_probs = torch.log_softmax(torch.tensor([[1.0, -1.0]]), dim=-1)
    interactions = torch.tensor([[[1.0, -1.0], [1.0, -1.0]]])

    target, diagnostics = recoverable_target(
        mode="interaction_only_ablation",
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        anchor_interactions=interactions,
        null_interaction=torch.tensor([[1.0, -1.0]], requires_grad=True),
        neural_reliability=1.0,
        session_stability=1.0,
    )

    torch.testing.assert_close(target, torch.tensor([[0.5, 0.5]]))
    torch.testing.assert_close(diagnostics["aggregate_interaction"], torch.zeros(1, 2))
    assert not target.requires_grad
    assert all(not value.requires_grad for value in diagnostics.values())


def test_recoverable_target_filters_through_brain_only_direction_above_null() -> None:
    student_log_probs = torch.full((1, 4), 0.25).log()
    teacher_log_probs = torch.log_softmax(
        torch.tensor([[1.0, -1.0, 0.0, 0.0]]),
        dim=-1,
    )
    interactions = torch.tensor([[[1.0, -1.0, 1.0, -1.0]]])
    brain_only_direction = torch.tensor([[1.0, -1.0, 0.0, 0.0]])

    target, diagnostics = recoverable_target(
        mode="nra",
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        anchor_interactions=_nra_interactions(interactions),
        nra_evidence=_nra_evidence(
            brain_only_direction=brain_only_direction,
            null_threshold=0.0,
            statistic_epsilon=0.0,
        ),
        neural_reliability=1.0,
        session_stability=1.0,
        epsilon=0.0,
    )

    torch.testing.assert_close(
        target,
        torch.tensor([[0.44858053, 0.10905743, 0.22118102, 0.22118102]]),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        diagnostics["brain_supported_interaction"], brain_only_direction
    )
    torch.testing.assert_close(
        diagnostics["alignment_score"],
        torch.tensor([2.0**-0.5]),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        diagnostics["kappa"],
        torch.tensor([2.0**-0.5]),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        diagnostics["brain_only_filter_applied"], torch.tensor(1.0)
    )


def test_recoverable_target_blocks_alignment_below_the_null_threshold() -> None:
    student_log_probs = torch.full((1, 4), 0.25).log()
    teacher_log_probs = torch.log_softmax(
        torch.tensor([[1.0, -1.0, 0.0, 0.0]]),
        dim=-1,
    )

    target, diagnostics = recoverable_target(
        mode="nra",
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        anchor_interactions=_nra_interactions(torch.tensor([[[1.0, -1.0, 1.0, -1.0]]])),
        nra_evidence=_nra_evidence(
            brain_only_direction=torch.tensor([[1.0, -1.0, 0.0, 0.0]]),
            null_threshold=torch.tensor([0.8]),
            statistic_epsilon=0.0,
        ),
        neural_reliability=1.0,
        session_stability=1.0,
        epsilon=0.0,
    )

    torch.testing.assert_close(target, torch.full((1, 4), 0.25))
    assert diagnostics["alignment_score"].item() > 0.0
    torch.testing.assert_close(diagnostics["kappa"], torch.tensor([0.0]))


def test_null_threshold_broadcasts_over_timestep_axes() -> None:
    student_log_probs = torch.full((1, 2, 4), 0.25).log()
    teacher_logits = torch.tensor([[[1.0, -1.0, 0.0, 0.0]]]).expand(1, 2, 4)
    interaction = torch.tensor([1.0, -1.0, 1.0, -1.0])
    brain_direction = torch.tensor([1.0, -1.0, 0.0, 0.0])

    target, diagnostics = recoverable_target(
        mode="nra",
        student_log_probs=student_log_probs,
        teacher_log_probs=torch.log_softmax(teacher_logits, dim=-1),
        anchor_interactions=_nra_interactions(
            interaction.reshape(1, 1, 1, 4).expand(1, 2, 1, 4)
        ),
        nra_evidence=_nra_evidence(
            brain_only_direction=brain_direction.reshape(1, 1, 4).expand(
                1,
                2,
                4,
            ),
            null_threshold=torch.tensor([0.5, 0.8]),
            statistic_epsilon=0.0,
        ),
        neural_reliability=1.0,
        session_stability=1.0,
        epsilon=0.0,
    )

    torch.testing.assert_close(
        diagnostics["kappa"],
        torch.tensor([[0.41421354, 0.0]]),
        atol=1e-6,
        rtol=0.0,
    )
    assert not torch.allclose(target[:, 0], torch.full((1, 4), 0.25))
    torch.testing.assert_close(target[:, 1], torch.full((1, 4), 0.25))


def test_recoverable_target_rejects_opposite_brain_only_alignment() -> None:
    student_log_probs = torch.tensor([[0.5, 0.5]]).log()
    teacher_log_probs = torch.log_softmax(torch.tensor([[1.0, -1.0]]), dim=-1)

    target, diagnostics = recoverable_target(
        mode="nra",
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        anchor_interactions=_nra_interactions(torch.tensor([[[1.0, -1.0]]])),
        nra_evidence=_nra_evidence(
            brain_only_direction=torch.tensor([[-1.0, 1.0]]),
            null_threshold=0.0,
            statistic_epsilon=0.0,
        ),
        neural_reliability=1.0,
        session_stability=1.0,
        epsilon=0.0,
    )

    torch.testing.assert_close(target, torch.tensor([[0.5, 0.5]]))
    torch.testing.assert_close(
        diagnostics["brain_supported_interaction"], torch.zeros(1, 2)
    )
    torch.testing.assert_close(diagnostics["alignment_score"], torch.tensor([-1.0]))
    torch.testing.assert_close(diagnostics["kappa"], torch.tensor([0.0]))


def test_recoverable_target_is_constant_invariant_with_nonuniform_geometry() -> None:
    student_probs = torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float64)
    teacher_log_probs = torch.log_softmax(
        torch.tensor([[1.0, -0.5, 0.2]], dtype=torch.float64),
        dim=-1,
    )
    interactions = torch.tensor(
        [[[1.5, -0.5, 0.1], [0.8, -0.2, 0.4]]],
        dtype=torch.float64,
    )

    baseline, _ = recoverable_target(
        mode="interaction_only_ablation",
        student_log_probs=student_probs.log(),
        teacher_log_probs=teacher_log_probs,
        anchor_interactions=interactions,
        neural_reliability=1.0,
        session_stability=1.0,
    )
    shifted, _ = recoverable_target(
        mode="interaction_only_ablation",
        student_log_probs=student_probs.log(),
        teacher_log_probs=teacher_log_probs + 17.0,
        anchor_interactions=interactions
        + torch.tensor([[[11.0], [-9.0]]], dtype=torch.float64),
        neural_reliability=1.0,
        session_stability=1.0,
    )

    torch.testing.assert_close(shifted, baseline, atol=1e-12, rtol=0.0)


def test_recoverable_target_and_diagnostics_are_fully_detached() -> None:
    student_logits = torch.tensor([[0.2, -0.2]], requires_grad=True)
    teacher_logits = torch.tensor([[1.0, -1.0]], requires_grad=True)
    interactions = torch.tensor(
        [[[1.0, -1.0], [0.5, -0.5]]],
        requires_grad=True,
    )
    reliability = torch.tensor([0.8], requires_grad=True)
    stability = torch.tensor([0.7], requires_grad=True)
    weights = torch.tensor([[2.0, 2.0]], requires_grad=True)
    brain_only_direction = torch.tensor([[1.0, -1.0]], requires_grad=True)
    null_threshold = torch.tensor([0.2], requires_grad=True)
    student_log_probs = torch.log_softmax(student_logits, dim=-1)

    target, diagnostics = recoverable_target(
        mode="nra",
        student_log_probs=student_log_probs,
        teacher_log_probs=torch.log_softmax(teacher_logits, dim=-1),
        anchor_interactions=_nra_interactions(interactions, weights=weights),
        nra_evidence=_nra_evidence(
            brain_only_direction=brain_only_direction,
            null_threshold=null_threshold,
            k=2,
        ),
        neural_reliability=reliability,
        session_stability=stability,
    )

    assert not target.requires_grad
    assert diagnostics
    assert all(not value.requires_grad for value in diagnostics.values())

    loss = -(target * student_log_probs).sum()
    loss.backward()

    assert student_logits.grad is not None
    assert bool((student_logits.grad != 0).any())
    assert teacher_logits.grad is None
    assert interactions.grad is None
    assert reliability.grad is None
    assert stability.grad is None
    assert weights.grad is None
    assert brain_only_direction.grad is None
    assert null_threshold.grad is None


@pytest.mark.parametrize(
    ("neural_reliability", "session_stability"),
    [(0.0, 1.0), (1.0, 0.0)],
)
def test_coverage_distinguishes_geometric_support_from_effective_transfer(
    neural_reliability: float,
    session_stability: float,
) -> None:
    student_probs = torch.tensor([[0.5, 0.5]], dtype=torch.float64)

    target, diagnostics = recoverable_target(
        mode="interaction_only_ablation",
        student_log_probs=student_probs.log(),
        teacher_log_probs=torch.log_softmax(
            torch.tensor([[1.0, -1.0]], dtype=torch.float64),
            dim=-1,
        ),
        anchor_interactions=torch.tensor(
            [[[1.0, -1.0]]],
            dtype=torch.float64,
        ),
        neural_reliability=neural_reliability,
        session_stability=session_stability,
    )

    torch.testing.assert_close(target, student_probs)
    torch.testing.assert_close(
        diagnostics["geometric_coverage"],
        torch.ones(1, dtype=torch.float64),
    )
    torch.testing.assert_close(
        diagnostics["transfer_coverage"],
        torch.zeros(1, dtype=torch.float64),
    )
    torch.testing.assert_close(
        diagnostics["coverage"],
        diagnostics["transfer_coverage"],
    )


def test_positive_gates_report_geometric_and_transfer_coverage() -> None:
    _, diagnostics = recoverable_target(
        mode="interaction_only_ablation",
        student_log_probs=torch.tensor([[0.5, 0.5]], dtype=torch.float64).log(),
        teacher_log_probs=torch.log_softmax(
            torch.tensor([[1.0, -1.0]], dtype=torch.float64),
            dim=-1,
        ),
        anchor_interactions=torch.tensor(
            [[[1.0, -1.0]]],
            dtype=torch.float64,
        ),
        neural_reliability=0.5,
        session_stability=0.5,
    )

    torch.testing.assert_close(
        diagnostics["geometric_coverage"],
        torch.ones(1, dtype=torch.float64),
    )
    torch.testing.assert_close(
        diagnostics["transfer_coverage"],
        torch.ones(1, dtype=torch.float64),
    )
    torch.testing.assert_close(
        diagnostics["coverage"],
        diagnostics["transfer_coverage"],
    )


def test_recoverable_target_uses_float32_arithmetic_for_float16_inputs() -> None:
    student_log_probs = torch.tensor([[0.5, 0.5]], dtype=torch.float16).log()
    teacher_log_probs = torch.log_softmax(
        torch.tensor([[1.0, -1.0]], dtype=torch.float16),
        dim=-1,
    )
    interactions = torch.tensor(
        [[[1.0e-4, -1.0e-4]]],
        dtype=torch.float16,
    )

    target, diagnostics = recoverable_target(
        mode="interaction_only_ablation",
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        anchor_interactions=interactions,
        neural_reliability=1.0,
        session_stability=1.0,
        epsilon=1.0e-8,
    )

    assert target.dtype == torch.float16
    assert bool(torch.isfinite(target).all())
    torch.testing.assert_close(
        target.float(),
        torch.tensor([[0.7311, 0.2689]]),
        atol=2e-3,
        rtol=0.0,
    )
    assert diagnostics["supported_interaction_norm_sq"].dtype == torch.float32
    assert diagnostics["supported_interaction_norm_sq"].item() > 0.0
    assert all(bool(torch.isfinite(value).all()) for value in diagnostics.values())


@pytest.mark.parametrize("vocabulary_size", [65536, 150000])
def test_recoverable_target_accepts_bfloat16_log_softmax_over_large_vocabulary(
    vocabulary_size: int,
) -> None:
    student_log_probs = torch.log_softmax(
        torch.zeros(1, vocabulary_size, dtype=torch.float64), dim=-1
    ).to(dtype=torch.bfloat16)

    target, diagnostics = recoverable_target(
        mode="interaction_only_ablation",
        student_log_probs=student_log_probs,
        teacher_log_probs=student_log_probs,
        anchor_interactions=torch.zeros(1, 1, vocabulary_size, dtype=torch.bfloat16),
    )

    expected = student_log_probs.float().log_softmax(dim=-1).exp().to(torch.bfloat16)
    torch.testing.assert_close(target, expected, atol=0.0, rtol=0.0)
    assert diagnostics["target_kl"].item() >= 0.0
    assert diagnostics["target_kl"].item() == pytest.approx(0.0, abs=2.0e-6)


def test_recoverable_target_reports_nonnegative_kl_for_nonuniform_large_bfloat16() -> (
    None
):
    generator = torch.Generator().manual_seed(4)
    vocabulary_size = 65536
    student_log_probs = torch.log_softmax(
        torch.randn(
            1,
            vocabulary_size,
            dtype=torch.float64,
            generator=generator,
        ),
        dim=-1,
    ).to(dtype=torch.bfloat16)
    original_student = student_log_probs.clone()

    target, diagnostics = recoverable_target(
        mode="interaction_only_ablation",
        student_log_probs=student_log_probs,
        teacher_log_probs=student_log_probs,
        anchor_interactions=torch.zeros(1, 1, vocabulary_size, dtype=torch.bfloat16),
    )

    torch.testing.assert_close(student_log_probs, original_student, atol=0.0, rtol=0.0)
    expected = student_log_probs.float().log_softmax(dim=-1).exp().to(torch.bfloat16)
    torch.testing.assert_close(target, expected, atol=0.0, rtol=0.0)
    assert diagnostics["target_kl"].item() >= 0.0
    assert diagnostics["target_kl"].item() == pytest.approx(0.0, abs=4.0e-6)


def test_recoverable_target_rejects_materially_unnormalized_bfloat16_logs() -> None:
    vocabulary_size = 65536
    student_log_probs = (
        torch.log_softmax(
            torch.zeros(1, vocabulary_size, dtype=torch.float64), dim=-1
        ).to(dtype=torch.bfloat16)
        + 0.125
    )

    with pytest.raises(ValueError, match="student_log_probs.*normalized"):
        recoverable_target(
            mode="interaction_only_ablation",
            student_log_probs=student_log_probs,
            teacher_log_probs=student_log_probs,
            anchor_interactions=torch.zeros(
                1, 1, vocabulary_size, dtype=torch.bfloat16
            ),
        )


def test_recoverable_target_keeps_float64_log_normalization_strict() -> None:
    student_log_probs = (
        torch.log_softmax(torch.zeros(1, 8, dtype=torch.float64), dim=-1) + 2.0e-6
    )

    with pytest.raises(ValueError, match="student_log_probs.*normalized"):
        recoverable_target(
            mode="interaction_only_ablation",
            student_log_probs=student_log_probs,
            teacher_log_probs=student_log_probs,
            anchor_interactions=torch.zeros(1, 1, 8, dtype=torch.float64),
        )
