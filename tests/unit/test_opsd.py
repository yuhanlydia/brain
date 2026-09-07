import math

import pytest
import torch

from brain_evidence.objectives.opsd import forward_kl


def test_forward_kl_averages_only_unmasked_tokens() -> None:
    student_log_probs = torch.log(
        torch.tensor(
            [[[0.5, 0.5], [0.25, 0.75], [0.9, 0.1]]],
            dtype=torch.float64,
        )
    )
    target_probs = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]],
        dtype=torch.float64,
    )

    loss = forward_kl(
        student_log_probs,
        target_probs,
        token_mask=torch.tensor([[True, True, False]]),
    )

    expected = (math.log(2.0) + math.log(4.0 / 3.0)) / 2.0
    assert loss.item() == pytest.approx(expected)


def test_forward_kl_clips_each_token_before_reduction() -> None:
    student_log_probs = torch.log(
        torch.tensor([[[0.5, 0.5], [0.25, 0.75]]], dtype=torch.float64)
    )
    target_probs = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], dtype=torch.float64)

    loss = forward_kl(student_log_probs, target_probs, pointwise_clip=0.8)

    expected = (math.log(2.0) + 0.8) / 2.0
    assert loss.item() == pytest.approx(expected)


def test_forward_kl_ignores_zero_target_support_at_negative_infinity() -> None:
    student_log_probs = torch.tensor(
        [[0.0, -torch.inf]], dtype=torch.float64, requires_grad=True
    )
    target_probs = torch.tensor([[1.0, 0.0]], dtype=torch.float64)

    loss = forward_kl(student_log_probs, target_probs)
    loss.backward()

    assert loss.item() == pytest.approx(0.0)
    torch.testing.assert_close(
        student_log_probs.grad,
        torch.zeros_like(student_log_probs),
    )


def test_forward_kl_discards_nonfinite_masked_tokens_before_reduction() -> None:
    student_log_probs = torch.tensor(
        [[[-math.log(2.0), -math.log(2.0)], [math.nan, math.nan]]],
        dtype=torch.float64,
    )
    target_probs = torch.tensor([[[0.5, 0.5], [0.5, 0.5]]], dtype=torch.float64)

    loss = forward_kl(
        student_log_probs,
        target_probs,
        token_mask=torch.tensor([[1, 0]]),
    )

    assert loss.item() == pytest.approx(0.0)


@pytest.mark.parametrize("mask_value", [-1.0, 0.5, math.nan, math.inf])
def test_forward_kl_rejects_nonbinary_or_nonfinite_masks(mask_value: float) -> None:
    with pytest.raises(ValueError, match="token_mask"):
        forward_kl(
            torch.tensor([[[0.0, 0.0]]]),
            torch.tensor([[[1.0, 0.0]]]),
            token_mask=torch.tensor([[mask_value]]),
        )


@pytest.mark.parametrize("pointwise_clip", [math.nan, math.inf, -math.inf])
def test_forward_kl_rejects_nonfinite_clipping(pointwise_clip: float) -> None:
    with pytest.raises(ValueError, match="pointwise_clip"):
        forward_kl(
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([[1.0, 0.0]]),
            pointwise_clip=pointwise_clip,
        )


def test_uniform_token_weights_are_identical_to_vanilla_opsd() -> None:
    student_log_probs = torch.log_softmax(
        torch.tensor(
            [[[0.1, 0.4, -0.2], [1.2, -0.3, 0.5], [-0.7, 0.6, 0.2]]],
            dtype=torch.float64,
        ),
        dim=-1,
    )
    target_probs = torch.tensor(
        [[[0.2, 0.5, 0.3], [0.6, 0.1, 0.3], [0.1, 0.2, 0.7]]],
        dtype=torch.float64,
    )
    token_mask = torch.tensor([[True, False, True]])

    vanilla = forward_kl(student_log_probs, target_probs, token_mask=token_mask)
    uniform_weighted = forward_kl(
        student_log_probs,
        target_probs,
        token_mask=token_mask,
        token_weights=torch.ones((1, 3), dtype=torch.float64),
    )

    torch.testing.assert_close(uniform_weighted, vanilla, rtol=0.0, atol=0.0)


def test_token_weights_form_a_normalized_weighted_mean() -> None:
    student_log_probs = torch.log(
        torch.tensor(
            [
                [
                    [math.exp(-1.0), 1.0 - math.exp(-1.0)],
                    [math.exp(-3.0), 1.0 - math.exp(-3.0)],
                ]
            ],
            dtype=torch.float64,
        )
    )
    target_probs = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], dtype=torch.float64)

    loss = forward_kl(
        student_log_probs,
        target_probs,
        token_weights=torch.tensor([[0.025, 0.075]], dtype=torch.float64),
    )

    # Weighted mean: (0.025 * 1 + 0.075 * 3) / 0.1 = 2.5.
    assert loss.item() == pytest.approx(2.5)


def test_forward_kl_detaches_target_but_updates_student() -> None:
    student_logits = torch.log(
        torch.tensor([[[0.6, 0.4], [0.25, 0.75], [0.1, 0.9]]], dtype=torch.float64)
    ).requires_grad_()
    target_probs = torch.tensor(
        [[[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]]],
        dtype=torch.float64,
        requires_grad=True,
    )

    loss = forward_kl(
        torch.log_softmax(student_logits, dim=-1),
        target_probs,
        token_mask=torch.tensor([[True, False, True]]),
    )
    loss.backward()

    torch.testing.assert_close(
        student_logits.grad,
        torch.tensor(
            [[[0.2, -0.2], [0.0, 0.0], [-0.15, 0.15]]],
            dtype=torch.float64,
        ),
        rtol=1e-14,
        atol=1e-14,
    )
    assert target_probs.grad is None


@pytest.mark.parametrize(
    "target",
    [
        [2.0, 0.0],
        [-0.1, 1.1],
        [math.nan, 0.0],
        [math.inf, 0.0],
        [0.0, 0.0],
        [0.25, 0.25],
    ],
)
def test_forward_kl_rejects_invalid_active_target_distributions(target) -> None:
    with pytest.raises(ValueError, match="target_probs"):
        forward_kl(
            torch.tensor([[0.5, 0.5]], dtype=torch.float64).log(),
            torch.tensor([target], dtype=torch.float64),
        )


@pytest.mark.parametrize(
    "student",
    [
        [0.0, 0.0],
        [math.log(0.2), math.log(0.2)],
        [-math.inf, 0.0],
        [math.nan, 0.0],
        [math.inf, -math.inf],
        [-math.inf, -math.inf],
    ],
)
def test_forward_kl_rejects_invalid_active_student_distributions(student) -> None:
    with pytest.raises(ValueError, match="student_log_probs"):
        forward_kl(
            torch.tensor([student], dtype=torch.float64),
            torch.tensor([[0.5, 0.5]], dtype=torch.float64),
        )


@pytest.mark.parametrize("invalid", [math.nan, math.inf])
def test_forward_kl_rejects_nan_or_positive_infinity_off_target_support(
    invalid: float,
) -> None:
    with pytest.raises(ValueError, match="student_log_probs"):
        forward_kl(
            torch.tensor([[0.0, invalid]], dtype=torch.float64),
            torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        )


def test_forward_kl_validates_each_target_row_not_total_batch_mass() -> None:
    with pytest.raises(ValueError, match="target_probs"):
        forward_kl(
            torch.full((2, 2), -math.log(2.0), dtype=torch.float64),
            torch.tensor([[0.5, 0.0], [0.5, 1.0]], dtype=torch.float64),
        )


@pytest.mark.parametrize("argument", ["student_log_probs", "target_probs"])
@pytest.mark.parametrize(
    "invalid",
    [
        [[0.5, 0.5]],
        torch.ones((1, 2), dtype=torch.int64),
        torch.ones((1, 2), dtype=torch.bool),
        torch.ones((1, 2), dtype=torch.complex64),
    ],
)
def test_forward_kl_requires_real_floating_distribution_tensors(
    argument: str,
    invalid,
) -> None:
    arguments = {
        "student_log_probs": torch.full((1, 2), -math.log(2.0)),
        "target_probs": torch.full((1, 2), 0.5),
    }
    arguments[argument] = invalid

    with pytest.raises(TypeError, match=argument):
        forward_kl(**arguments)


def test_forward_kl_requires_a_nonempty_vocabulary_axis() -> None:
    with pytest.raises(ValueError, match="vocabulary"):
        forward_kl(torch.empty(1, 0), torch.empty(1, 0))


def test_forward_kl_rejects_cross_device_distributions_before_arithmetic() -> None:
    with pytest.raises(ValueError, match="same device"):
        forward_kl(torch.zeros(1, 2), torch.empty(1, 2, device="meta"))


@pytest.mark.parametrize("selection", ["mask", "weights"])
@pytest.mark.parametrize(
    "padding",
    [[math.nan, math.inf], [-1.0, 2.0], [0.0, 0.0]],
)
def test_forward_kl_ignores_invalid_padding_with_finite_zero_gradient(
    selection: str,
    padding,
) -> None:
    student = torch.tensor(
        [[-math.log(2.0), -math.log(2.0)], [math.nan, math.inf]],
        dtype=torch.float64,
        requires_grad=True,
    )
    target = torch.tensor(
        [[1.0, 0.0], padding], dtype=torch.float64, requires_grad=True
    )
    kwargs = (
        {"token_mask": torch.tensor([True, False])}
        if selection == "mask"
        else {"token_weights": torch.tensor([1.0, 0.0])}
    )

    loss = forward_kl(student, target, **kwargs)
    loss.backward()

    assert loss.item() == pytest.approx(math.log(2.0))
    torch.testing.assert_close(
        student.grad, torch.tensor([[-0.5, 0.5], [0.0, 0.0]], dtype=torch.float64)
    )
    assert target.grad is None


@pytest.mark.parametrize("argument", ["token_mask", "token_weights"])
@pytest.mark.parametrize("invalid", [[1], torch.tensor([1.0 + 1.0j])])
def test_forward_kl_rejects_non_tensor_or_complex_selectors(
    argument: str,
    invalid,
) -> None:
    with pytest.raises(TypeError, match=argument):
        forward_kl(
            torch.tensor([[0.5, 0.5]]).log(),
            torch.tensor([[0.5, 0.5]]),
            **{argument: invalid},
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_forward_kl_computes_low_precision_distributions_in_float32(dtype) -> None:
    student = torch.tensor([[0.25, 0.75]], dtype=dtype).log().requires_grad_()
    target = torch.tensor([[0.75, 0.25]], dtype=dtype, requires_grad=True)

    loss = forward_kl(student, target)
    loss.backward()

    normalized_student = student.detach().double().log_softmax(dim=-1)
    expected = (
        target.detach().double() * (target.detach().double().log() - normalized_student)
    ).sum()
    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(expected.item(), abs=1e-6)
    torch.testing.assert_close(
        student.grad, (normalized_student.exp() - target.detach().double()).to(dtype)
    )
    assert target.grad is None


@pytest.mark.parametrize("low_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("invalid_argument", ["student_log_probs", "target_probs"])
def test_forward_kl_does_not_relax_float64_validation_for_a_low_precision_peer(
    low_dtype,
    invalid_argument: str,
) -> None:
    arguments = {
        "student_log_probs": torch.tensor([[0.5, 0.5]], dtype=low_dtype).log(),
        "target_probs": torch.tensor([[0.5, 0.5]], dtype=low_dtype),
    }
    if invalid_argument == "student_log_probs":
        arguments[invalid_argument] = torch.tensor(
            [[0.5005, 0.5005]], dtype=torch.float64
        ).log()
    else:
        arguments[invalid_argument] = torch.tensor(
            [[0.5005, 0.5005]], dtype=torch.float64
        )

    with pytest.raises(ValueError, match=invalid_argument):
        forward_kl(**arguments)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_forward_kl_accepts_rounded_log_softmax_over_a_large_vocabulary(dtype) -> None:
    vocabulary_size = 65536
    student = torch.log_softmax(
        torch.zeros(1, vocabulary_size, dtype=torch.float64), dim=-1
    ).to(dtype=dtype)
    target = torch.full_like(student, 1.0 / vocabulary_size)

    loss = forward_kl(student, target)

    assert loss.item() >= -2e-6
    assert loss.item() == pytest.approx(0.0, abs=2e-6)


def test_forward_kl_normalizes_accepted_student_and_target_rounding() -> None:
    distribution = torch.tensor([[0.25, 0.75]], dtype=torch.float64)
    student = (distribution.log() + 5e-7).requires_grad_()
    target = (distribution * (1.0 + 5e-7)).requires_grad_()

    loss = forward_kl(student, target)
    loss.backward()

    assert loss.item() == pytest.approx(0.0, abs=1e-14)
    torch.testing.assert_close(
        student.grad, torch.zeros_like(student), atol=1e-14, rtol=0
    )
    assert target.grad is None


@pytest.mark.parametrize("selector", ["token_mask", "token_weights"])
def test_forward_kl_all_inactive_padding_returns_differentiable_zero(selector) -> None:
    student = torch.tensor([[math.nan, math.inf]], requires_grad=True)
    target = torch.tensor([[math.nan, -1.0]], requires_grad=True)

    loss = forward_kl(student, target, **{selector: torch.zeros(1)})
    loss.backward()

    assert loss.item() == 0.0
    torch.testing.assert_close(student.grad, torch.zeros_like(student))
    assert target.grad is None


def test_forward_kl_detaches_token_weights_while_preserving_weighted_student_gradient():
    student = torch.full(
        (2, 2), -math.log(2.0), dtype=torch.float64, requires_grad=True
    )
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    weights = torch.tensor([1.0, 3.0], dtype=torch.float64, requires_grad=True)

    loss = forward_kl(student, target, token_weights=weights)
    loss.backward()

    assert loss.item() == pytest.approx(math.log(2.0))
    assert loss.dtype == torch.float64
    torch.testing.assert_close(
        student.grad,
        torch.tensor([[-0.125, 0.125], [0.375, -0.375]], dtype=torch.float64),
    )
    assert weights.grad is None


@pytest.mark.parametrize("selector", ["token_mask", "token_weights"])
def test_forward_kl_reports_invalid_selector_before_distribution_values(
    selector,
) -> None:
    with pytest.raises(ValueError, match=selector):
        forward_kl(
            torch.zeros(1, 2),
            torch.zeros(1, 2),
            **{selector: torch.tensor([-1.0])},
        )
