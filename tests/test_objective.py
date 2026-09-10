import pytest
import torch

from brain_npp.posterior import NeuralPosterior, build_neural_posterior


def test_build_npp_target_matches_exact_posterior_contrastive_fixture():
    """Catches replacing posterior-minus-prior expected logs with another target."""
    from brain_npp.objective import build_npp_target

    teacher_probs = torch.tensor(
        [[[[0.9, 0.1], [0.2, 0.8]]]], dtype=torch.float64
    )
    student_reference_logits = torch.log(
        torch.tensor([[[0.25, 0.75]]], dtype=torch.float64)
    )
    neural_posterior = build_neural_posterior(
        torch.log(torch.tensor([[0.8, 0.2]], dtype=torch.float64)),
        torch.log(torch.tensor([[0.5, 0.5]], dtype=torch.float64)),
    )

    target = build_npp_target(
        teacher_probs.log(), student_reference_logits, neural_posterior
    )

    torch.testing.assert_close(
        target.utility,
        torch.tensor([[[0.451223, -0.623832]]], dtype=torch.float64),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        target.log_probabilities.exp(),
        torch.tensor([[[0.2908, 0.7092]]], dtype=torch.float64),
        atol=1e-4,
        rtol=0,
    )
    torch.testing.assert_close(
        target.log_probabilities.exp().sum(dim=-1),
        torch.ones((1, 1), dtype=torch.float64),
    )


def test_build_npp_target_uses_expected_log_probabilities_not_log_mixtures():
    """Catches substituting log posterior/prior arithmetic mixtures for expectations."""
    from brain_npp.objective import build_npp_target

    teacher_log_probs = torch.log(
        torch.tensor([[[[0.9, 0.1], [0.2, 0.8]]]], dtype=torch.float64)
    )
    posterior = build_neural_posterior(
        torch.log(torch.tensor([[0.8, 0.2]], dtype=torch.float64)),
        torch.log(torch.tensor([[0.5, 0.5]], dtype=torch.float64)),
    )

    target = build_npp_target(
        teacher_log_probs,
        torch.zeros((1, 1, 2), dtype=torch.float64),
        posterior,
    )

    torch.testing.assert_close(
        target.utility,
        torch.tensor([[[0.451223, -0.623832]]], dtype=torch.float64),
        atol=1e-6,
        rtol=0,
    )
    mixture_utility = torch.log(torch.tensor(0.76, dtype=torch.float64)) - torch.log(
        torch.tensor(0.55, dtype=torch.float64)
    )
    assert target.utility[0, 0, 0] != pytest.approx(mixture_utility.item())


def test_build_npp_target_returns_reference_distribution_when_posterior_equals_prior():
    """Catches a nonzero contrastive target at the posterior-prior fixed point."""
    from brain_npp.objective import build_npp_target

    posterior = build_neural_posterior(
        torch.zeros((1, 2), dtype=torch.float64),
        torch.log(torch.tensor([[0.3, 0.7]], dtype=torch.float64)),
    )
    reference_logits = torch.tensor([[[1.0, -1.0, 0.5]]], dtype=torch.float64)
    target = build_npp_target(
        torch.tensor(
            [[[[0.0, -1.0, -2.0], [-3.0, 2.0, 1.0]]]], dtype=torch.float64
        ),
        reference_logits,
        posterior,
    )

    torch.testing.assert_close(target.utility, torch.zeros_like(target.utility))
    torch.testing.assert_close(
        target.log_probabilities, torch.log_softmax(reference_logits, dim=-1)
    )


def test_build_npp_target_excludes_masked_candidates():
    """Catches padded teacher candidates contributing to the contrastive expectation."""
    from brain_npp.objective import build_npp_target

    candidate_mask = torch.tensor([[True, False, True]])
    posterior = build_neural_posterior(
        torch.log(torch.tensor([[0.1, 0.2, 0.7]], dtype=torch.float64)),
        torch.zeros((1, 3), dtype=torch.float64),
        candidate_mask,
    )
    teacher_log_probs = torch.log(
        torch.tensor(
            [[[[0.8, 0.2], [1.0, 0.0], [0.2, 0.8]]]], dtype=torch.float64
        )
    )

    target = build_npp_target(
        teacher_log_probs,
        torch.zeros((1, 1, 2), dtype=torch.float64),
        posterior,
    )

    expected_utility = torch.tensor(
        [[[-0.5198603854, 0.5198603854]]], dtype=torch.float64
    )
    torch.testing.assert_close(target.utility, expected_utility)


def test_build_npp_target_preserves_independent_batch_rows():
    """Catches mixing posterior or teacher values across batch rows."""
    from brain_npp.objective import build_npp_target

    posterior = build_neural_posterior(
        torch.log(torch.tensor([[0.8, 0.2], [0.8, 0.2]], dtype=torch.float64)),
        torch.log(torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float64)),
    )
    teacher_log_probs = torch.log(
        torch.tensor(
            [
                [[[0.9, 0.1], [0.2, 0.8]]],
                [[[0.9, 0.1], [0.2, 0.8]]],
            ],
            dtype=torch.float64,
        )
    )
    reference_logits = torch.tensor(
        [[[0.25, 0.75]], [[0.25, 0.75]]], dtype=torch.float64
    ).log()

    target = build_npp_target(teacher_log_probs, reference_logits, posterior)

    torch.testing.assert_close(target.log_probabilities[0], target.log_probabilities[1])
    torch.testing.assert_close(target.utility[0], target.utility[1])


def test_build_npp_target_clamps_scaled_information_gain_strength():
    """Catches applying alpha_scale or alpha_max incorrectly to information gain."""
    from brain_npp.objective import build_npp_target

    posterior = build_neural_posterior(
        torch.log(torch.tensor([[0.8, 0.2]], dtype=torch.float64)),
        torch.log(torch.tensor([[0.5, 0.5]], dtype=torch.float64)),
    )
    target = build_npp_target(
        torch.zeros((1, 1, 2, 2), dtype=torch.float64),
        torch.zeros((1, 1, 2), dtype=torch.float64),
        posterior,
        alpha_scale=0.1,
        alpha_max=0.5,
    )

    torch.testing.assert_close(target.strength, torch.tensor([0.5], dtype=torch.float64))


@pytest.mark.parametrize(
    ("teacher_log_probs", "student_reference_logits", "message"),
    [
        (torch.zeros((1, 2, 3)), torch.zeros((1, 2, 3)), "teacher_log_probs"),
        (torch.zeros((1, 2, 2, 3)), torch.zeros((1, 2, 3, 1)), "student_reference_logits"),
        (torch.zeros((1, 2, 2, 3)), torch.zeros((2, 2, 3)), "batch"),
    ],
)
def test_build_npp_target_rejects_incompatible_shapes(
    teacher_log_probs, student_reference_logits, message
):
    """Catches accepting tensors outside the documented batch/time/candidate/vocab shapes."""
    from brain_npp.objective import build_npp_target

    posterior = build_neural_posterior(torch.zeros((1, 2)), torch.zeros((1, 2)))

    with pytest.raises(ValueError, match=message):
        build_npp_target(teacher_log_probs, student_reference_logits, posterior)


def test_forward_kl_loss_uses_full_vocabulary_target_expectation():
    """Catches reducing KL to a sampled-token or target-mode loss."""
    from brain_npp.objective import forward_kl_loss

    stats = forward_kl_loss(
        torch.log(torch.tensor([[[0.25, 0.75]]], dtype=torch.float64)),
        torch.zeros((1, 1, 2), dtype=torch.float64),
    )

    assert stats.loss.item() == pytest.approx(0.1308120359)
    assert stats.token_count.item() == 1


def test_sparse_target_forward_kl_has_exact_value_and_only_student_gradient():
    """Catches zero target mass multiplying an infinite log ratio."""
    from brain_npp.objective import forward_kl_loss

    target = torch.tensor([[[0.0, -torch.inf]]], requires_grad=True)
    student = torch.zeros((1, 1, 2), requires_grad=True)
    stats = forward_kl_loss(target, student)
    stats.loss.backward()

    assert stats.loss.item() == pytest.approx(0.69314718056)
    torch.testing.assert_close(student.grad, torch.tensor([[[-0.5, 0.5]]]))
    assert target.grad is None


def test_masked_nonfinite_tokens_do_not_contaminate_forward_kl_or_gradient():
    """Catches multiplying an invalid masked KL by zero after its computation."""
    from brain_npp.objective import forward_kl_loss

    target = torch.tensor([[[0.0, -torch.inf], [torch.nan, torch.nan]]])
    student = torch.tensor([[[0.0, 0.0], [torch.nan, torch.nan]]], requires_grad=True)
    stats = forward_kl_loss(target, student, torch.tensor([[True, False]]))
    stats.loss.backward()

    assert stats.loss.item() == pytest.approx(0.69314718056)
    assert stats.token_count.item() == 1
    torch.testing.assert_close(student.grad, torch.tensor([[[-0.5, 0.5], [0.0, 0.0]]]))


@pytest.mark.parametrize("extreme_teacher", [False, True])
def test_extreme_finite_fp16_target_inputs_are_normalized_in_sufficient_precision(extreme_teacher):
    """Catches fp16 log-softmax overflow and returning attached raw strength."""
    from brain_npp.objective import build_npp_target, forward_kl_loss

    prior = torch.tensor([[-0.693147, -0.693147]], dtype=torch.float16, requires_grad=True)
    posterior = NeuralPosterior(
        prior, prior,
        torch.tensor([0.0], dtype=torch.float16, requires_grad=True),
        torch.ones((1, 2), dtype=torch.bool),
    )
    teacher = torch.zeros((1, 1, 2, 2), dtype=torch.float16)
    if extreme_teacher:
        teacher[..., 0] = 65504.0
        teacher[..., 1] = -65504.0
    teacher.requires_grad_()
    reference = torch.tensor([[[65504.0, -65504.0]]], dtype=torch.float16, requires_grad=True)
    target = build_npp_target(teacher, reference, posterior)

    assert torch.isfinite(target.log_probabilities).all()
    torch.testing.assert_close(target.log_probabilities.float(), torch.tensor([[[0.0, -131008.0]]]))
    assert target.log_probabilities.exp().sum().item() == 1.0
    for output in (target.log_probabilities, target.utility, target.strength):
        assert not output.requires_grad
        assert output.grad_fn is None

    student = torch.zeros((1, 1, 2), requires_grad=True)
    stats = forward_kl_loss(target.log_probabilities, student)
    stats.loss.backward()
    assert stats.loss.item() == pytest.approx(0.69314718056)
    torch.testing.assert_close(student.grad, torch.tensor([[[-0.5, 0.5]]]))
    for source in (prior, posterior.information_gain, teacher, reference):
        assert source.grad is None


def test_forward_kl_loss_mask_controls_numerator_and_denominator():
    """Catches averaging masked tokens into the NPP numerator or denominator."""
    from brain_npp.objective import forward_kl_loss

    target_log_probs = torch.log(
        torch.tensor([[[0.25, 0.75], [0.5, 0.5]]], dtype=torch.float64)
    )
    stats = forward_kl_loss(
        target_log_probs,
        torch.zeros((1, 2, 2), dtype=torch.float64),
        torch.tensor([[True, False]]),
    )

    assert stats.loss.item() == pytest.approx(0.1308120359)
    assert stats.token_count.item() == 1


def test_forward_kl_loss_clips_each_log_ratio_before_target_expectation():
    """Catches clipping a token aggregate instead of its vocabulary log ratios."""
    from brain_npp.objective import forward_kl_loss

    stats = forward_kl_loss(
        torch.log(torch.tensor([[[0.25, 0.75]]], dtype=torch.float64)),
        torch.log(torch.tensor([[[0.1, 0.9]]], dtype=torch.float64)),
        clip_log_ratio=0.1,
    )

    assert stats.loss.item() == pytest.approx(-0.05)


def test_forward_kl_loss_rejects_an_empty_mask():
    """Catches silently producing a divide-by-zero loss for an empty token mask."""
    from brain_npp.objective import forward_kl_loss

    with pytest.raises(ValueError, match="empty"):
        forward_kl_loss(
            torch.zeros((1, 1, 2)),
            torch.zeros((1, 1, 2)),
            torch.tensor([[False]]),
        )


def test_npp_opsd_loss_delegates_to_forward_kl_loss():
    """Catches the Task 4 loss entrypoint diverging from the NPP KL contract."""
    from brain_npp.objective import npp_opsd_loss

    stats = npp_opsd_loss(
        torch.log(torch.tensor([[[0.25, 0.75]]], dtype=torch.float64)),
        torch.zeros((1, 1, 2), dtype=torch.float64),
    )

    assert stats.loss.item() == pytest.approx(0.1308120359)


def test_npp_target_is_detached_and_forward_kl_gradient_is_q_minus_student_probabilities():
    """Catches reverse-KL gradients or target-side gradient flow into teacher inputs."""
    from brain_npp.objective import build_npp_target, forward_kl_loss

    teacher_log_probs = torch.log(
        torch.tensor([[[[0.9, 0.1], [0.2, 0.8]]]], dtype=torch.float64)
    ).detach().requires_grad_()
    reference_logits = torch.log(
        torch.tensor([[[0.25, 0.75]]], dtype=torch.float64)
    ).detach().requires_grad_()
    neural_posterior = NeuralPosterior(
        posterior=torch.log(torch.tensor([[0.8, 0.2]], dtype=torch.float64))
        .detach()
        .requires_grad_(),
        prior=torch.log(torch.tensor([[0.5, 0.5]], dtype=torch.float64))
        .detach()
        .requires_grad_(),
        information_gain=torch.tensor([0.1927447570], dtype=torch.float64, requires_grad=True),
        candidate_mask=torch.tensor([[True, True]]),
    )
    student_logits = torch.tensor(
        [[[0.3, -0.4]]], dtype=torch.float64, requires_grad=True
    )

    target = build_npp_target(teacher_log_probs, reference_logits, neural_posterior)
    stats = forward_kl_loss(target.log_probabilities, student_logits)
    stats.loss.backward()

    torch.testing.assert_close(
        student_logits.grad,
        torch.softmax(student_logits.detach(), dim=-1) - target.log_probabilities.exp(),
    )
    assert not target.log_probabilities.requires_grad
    assert teacher_log_probs.grad is None
    assert reference_logits.grad is None
    assert neural_posterior.posterior.grad is None
    assert neural_posterior.prior.grad is None
    assert neural_posterior.information_gain.grad is None
