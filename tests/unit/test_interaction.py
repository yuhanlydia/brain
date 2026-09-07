import pytest
import torch

from brain_evidence.recoverability.interaction import (
    AnchorLocalValidity,
    aggregate_interactions,
    anchor_local_interaction,
    factorial_interaction,
)

VALID_ANCHOR_LOCAL_CONTROL = AnchorLocalValidity(
    anchor_pronunciation="p_i",
    negative_pronunciation="p_j",
    control_pronunciation="p_c",
)


def test_anchor_local_interaction_cancels_additive_pronunciation_copying() -> None:
    pronunciation_i = torch.tensor([0.7, -0.4, 0.2])
    pronunciation_j = torch.tensor([-0.3, 0.5, 0.1])
    brain_i = torch.tensor([0.2, 0.1, -0.2])
    control_brain = torch.tensor([-0.1, 0.3, 0.4])
    student_probs = torch.tensor([0.2, 0.3, 0.5])

    interaction = anchor_local_interaction(
        brain_i + pronunciation_i,
        brain_i + pronunciation_j,
        control_brain + pronunciation_i,
        control_brain + pronunciation_j,
        student_probs,
        validity=VALID_ANCHOR_LOCAL_CONTROL,
    )

    torch.testing.assert_close(interaction, torch.zeros(3), atol=1e-7, rtol=0.0)


def test_anchor_local_interaction_preserves_the_anchor_matched_effect() -> None:
    matched_effect = torch.tensor([2.0, 0.0, -1.0])
    zeros = torch.zeros(3)
    student_probs = torch.tensor([0.2, 0.3, 0.5])

    interaction = anchor_local_interaction(
        matched_effect,
        zeros,
        zeros,
        zeros,
        student_probs,
        validity=VALID_ANCHOR_LOCAL_CONTROL,
    )

    torch.testing.assert_close(interaction, torch.tensor([2.1, 0.1, -0.9]))


def test_anchor_local_interaction_is_batched_and_detached() -> None:
    xi_pi = torch.tensor(
        [[[2.0, 0.0], [0.0, 2.0]], [[1.0, 0.0], [0.0, 1.0]]],
        requires_grad=True,
    )
    zeros = torch.zeros_like(xi_pi, requires_grad=True)
    student_probs = torch.full_like(xi_pi, 0.5, requires_grad=True)

    interaction = anchor_local_interaction(
        xi_pi,
        zeros,
        zeros,
        zeros,
        student_probs,
        validity=VALID_ANCHOR_LOCAL_CONTROL,
    )

    expected = torch.tensor([[[1.0, -1.0], [-1.0, 1.0]], [[0.5, -0.5], [-0.5, 0.5]]])
    torch.testing.assert_close(interaction, expected)
    assert interaction.shape == (2, 2, 2)
    assert not interaction.requires_grad


def test_anchor_local_interaction_ignores_the_controls_own_unrelated_match() -> None:
    zeros = torch.zeros(3)
    teacher_states = {
        "xi_pi": zeros,
        "xi_pj": zeros,
        "xc_pi": zeros,
        "xc_pj": zeros,
        # p_c differs from both p_i and p_j, so this excluded cell is irrelevant.
        "xc_pc": torch.tensor([4.0, -2.0, 1.0]),
    }

    interaction = anchor_local_interaction(
        teacher_states["xi_pi"],
        teacher_states["xi_pj"],
        teacher_states["xc_pi"],
        teacher_states["xc_pj"],
        torch.tensor([0.2, 0.3, 0.5]),
        validity=VALID_ANCHOR_LOCAL_CONTROL,
    )

    torch.testing.assert_close(interaction, zeros)


def test_anchor_local_interaction_subtracts_x_modulated_copying_null() -> None:
    student_probs = torch.tensor([0.2, 0.3, 0.5])
    pronunciation_i = torch.tensor([1.0, -0.5, 0.2])
    pronunciation_j = torch.tensor([-0.2, 0.4, 0.8])
    brain_i = torch.tensor([0.3, -0.1, 0.2])
    control_brain = torch.tensor([-0.4, 0.2, 0.1])
    modulation_i = 2.0
    modulation_c = 0.5
    null_interaction = (modulation_i - modulation_c) * (
        pronunciation_i - pronunciation_j
    )

    interaction = anchor_local_interaction(
        brain_i + pronunciation_i + modulation_i * pronunciation_i,
        brain_i + pronunciation_j + modulation_i * pronunciation_j,
        control_brain + pronunciation_i + modulation_c * pronunciation_i,
        control_brain + pronunciation_j + modulation_c * pronunciation_j,
        student_probs,
        validity=VALID_ANCHOR_LOCAL_CONTROL,
        null_interaction=null_interaction,
    )

    torch.testing.assert_close(interaction, torch.zeros(3), atol=1e-6, rtol=0.0)


def test_anchor_local_interaction_is_invariant_to_cellwise_additive_constants() -> None:
    student_probs = torch.tensor([0.6, 0.3, 0.1])
    xi_pi = torch.tensor([2.0, 0.0, -1.0])
    xi_pj = torch.tensor([0.1, 0.2, 0.3])
    xc_pi = torch.tensor([-0.2, 0.3, 0.1])
    xc_pj = torch.tensor([0.4, -0.1, 0.2])

    baseline = anchor_local_interaction(
        xi_pi,
        xi_pj,
        xc_pi,
        xc_pj,
        student_probs,
        validity=VALID_ANCHOR_LOCAL_CONTROL,
    )
    shifted = anchor_local_interaction(
        xi_pi + 11.0,
        xi_pj - 7.0,
        xc_pi + 3.0,
        xc_pj + 19.0,
        student_probs,
        validity=VALID_ANCHOR_LOCAL_CONTROL,
    )

    torch.testing.assert_close(shifted, baseline, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        (shifted * student_probs).sum(),
        torch.tensor(0.0),
        atol=1e-6,
        rtol=0.0,
    )


def test_anchor_local_validity_rejects_a_control_matching_the_anchor() -> None:
    with pytest.raises(ValueError, match="control pronunciation"):
        AnchorLocalValidity(
            anchor_pronunciation="same",
            negative_pronunciation="different",
            control_pronunciation="same",
        )


@pytest.mark.parametrize(
    "invalid_pronunciation",
    ["", "   ", (), ("SH", ""), ("SH", " "), ("SH", 7), 7, None],
)
def test_anchor_local_validity_rejects_malformed_pronunciations(
    invalid_pronunciation: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="pronunciation"):
        AnchorLocalValidity(
            anchor_pronunciation=invalid_pronunciation,
            negative_pronunciation="N",
            control_pronunciation="K",
        )


def test_anchor_local_interaction_requires_validity_metadata() -> None:
    values = torch.zeros(2)

    with pytest.raises(TypeError):
        anchor_local_interaction(
            values,
            values,
            values,
            values,
            torch.tensor([0.5, 0.5]),
        )


def test_anchor_local_interaction_rejects_non_floating_teacher_states() -> None:
    integer_values = torch.tensor([1, 2])
    floating_values = torch.tensor([1.0, 2.0])

    with pytest.raises(ValueError, match="floating-point"):
        anchor_local_interaction(
            integer_values,
            floating_values,
            floating_values,
            floating_values,
            torch.tensor([0.5, 0.5]),
            validity=VALID_ANCHOR_LOCAL_CONTROL,
        )


def test_factorial_interaction_remains_an_algebraic_compatibility_alias() -> None:
    student_probs = torch.tensor([0.25, 0.75])
    cells = (
        torch.tensor([1.0, -1.0]),
        torch.tensor([0.2, -0.2]),
        torch.tensor([-0.3, 0.3]),
        torch.tensor([0.1, -0.1]),
    )

    legacy = factorial_interaction(*cells, student_probs)
    anchor_local = anchor_local_interaction(
        *cells,
        student_probs,
        validity=VALID_ANCHOR_LOCAL_CONTROL,
    )

    torch.testing.assert_close(legacy, anchor_local)


def test_aggregate_interactions_computes_mean_over_negative_axis() -> None:
    interactions = torch.tensor([[[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]]])

    probabilities = torch.tensor([[0.25, 0.75]])
    aggregate = aggregate_interactions(
        interactions,
        probabilities,
        method="mean",
    )

    torch.testing.assert_close(aggregate, torch.tensor([[0.75, -0.25]]))
    torch.testing.assert_close(
        (aggregate * probabilities).sum(dim=-1),
        torch.tensor([0.0]),
    )


def test_aggregate_interactions_computes_coordinatewise_median() -> None:
    interactions = torch.tensor([[[1.0, 100.0], [2.0, 0.0], [50.0, 1.0]]])

    aggregate = aggregate_interactions(
        interactions,
        torch.tensor([[0.5, 0.5]]),
        method="median",
    )

    torch.testing.assert_close(aggregate, torch.tensor([[0.5, -0.5]]))


def test_aggregate_interactions_normalizes_non_negative_weights() -> None:
    interactions = torch.tensor([[[1.0, 3.0], [5.0, 7.0]]])
    weights = torch.tensor([[1.0, 3.0]], requires_grad=True)

    aggregate = aggregate_interactions(
        interactions,
        torch.tensor([[0.25, 0.75]]),
        method="mean",
        weights=weights,
    )

    torch.testing.assert_close(aggregate, torch.tensor([[-1.5, 0.5]]))
    assert not aggregate.requires_grad


def test_aggregate_interactions_expands_batch_negative_weights_over_time() -> None:
    interactions = torch.tensor(
        [
            [
                [[1.0, -1.0], [3.0, -3.0]],
                [[2.0, -2.0], [4.0, -4.0]],
                [[3.0, -3.0], [5.0, -5.0]],
            ],
            [
                [[10.0, -10.0], [20.0, -20.0]],
                [[20.0, -20.0], [30.0, -30.0]],
                [[30.0, -30.0], [40.0, -40.0]],
            ],
        ]
    )
    probabilities = torch.full((2, 3, 2), 0.5)
    weights = torch.tensor([[1.0, 3.0], [3.0, 1.0]])

    aggregate = aggregate_interactions(
        interactions,
        probabilities,
        weights=weights,
    )

    expected = torch.tensor(
        [
            [[2.5, -2.5], [3.5, -3.5], [4.5, -4.5]],
            [[12.5, -12.5], [22.5, -22.5], [32.5, -32.5]],
        ]
    )
    torch.testing.assert_close(aggregate, expected)


def test_aggregate_interactions_rejects_ambiguous_timestep_negative_weights() -> None:
    interactions = torch.ones(2, 3, 4, 2)
    probabilities = torch.full((2, 3, 2), 0.5)

    with pytest.raises(ValueError, match="weights"):
        aggregate_interactions(
            interactions,
            probabilities,
            weights=torch.ones(3, 4),
        )


@pytest.mark.parametrize(
    "weights",
    [torch.tensor([[-1.0, 2.0]]), torch.tensor([[0.0, 0.0]])],
)
def test_aggregate_interactions_rejects_invalid_weights(
    weights: torch.Tensor,
) -> None:
    interactions = torch.ones(1, 2, 3)
    probabilities = torch.full((1, 3), 1.0 / 3.0)

    with pytest.raises(ValueError):
        aggregate_interactions(interactions, probabilities, weights=weights)


def test_aggregate_interactions_rejects_unknown_method() -> None:
    with pytest.raises(ValueError):
        aggregate_interactions(
            torch.ones(1, 2, 3),
            torch.full((1, 3), 1.0 / 3.0),
            method="trimmed",
        )
