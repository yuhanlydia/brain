import torch

from brain_evidence.gates.counterfactual import legacy_dual_cosine_gate


def test_legacy_gate_uses_centered_fisher_cosines() -> None:
    probabilities = torch.tensor([0.8, 0.1, 0.1])
    teacher_contrast = torch.tensor([4.0, 3.0, 3.0], requires_grad=True)
    brain_contrast = torch.tensor([1.0, 1.0, 0.0], requires_grad=True)

    gate = legacy_dual_cosine_gate(
        teacher_contrast,
        brain_contrast,
        probabilities,
    )

    # The centered Fisher cosine is 2/3 (raw Euclidean cosine is different).
    torch.testing.assert_close(gate, torch.tensor(2 / 3))
    assert not gate.requires_grad


def test_legacy_gate_rejects_unsupported_or_opposite_directions() -> None:
    probabilities = torch.tensor([0.25, 0.25, 0.5])
    correction = torch.tensor([1.0, -1.0, 0.0])
    orthogonal = torch.tensor([1.0, 1.0, -1.0])
    opposite = torch.tensor([-1.0, 1.0, 0.0])

    orthogonal_gate = legacy_dual_cosine_gate(
        correction,
        orthogonal,
        probabilities,
    )
    opposite_gate = legacy_dual_cosine_gate(
        correction,
        opposite,
        probabilities,
    )

    torch.testing.assert_close(orthogonal_gate, torch.tensor(0.0))
    torch.testing.assert_close(opposite_gate, torch.tensor(0.0))


def test_legacy_gate_is_zero_for_a_degenerate_direction() -> None:
    gate = legacy_dual_cosine_gate(
        torch.tensor([0.0, 0.0]),
        torch.tensor([1.0, -1.0]),
        torch.tensor([0.5, 0.5]),
    )

    torch.testing.assert_close(gate, torch.tensor(0.0))


def test_aligned_contrasts_are_not_suppressed_by_full_correction() -> None:
    probabilities = torch.tensor(
        [[0.25, 0.25, 0.5], [0.25, 0.25, 0.5]],
        requires_grad=True,
    )
    teacher_contrast = torch.tensor(
        [[1.0, -1.0, 0.0], [2.0, -2.0, 0.0]],
        requires_grad=True,
    )
    brain_contrast = teacher_contrast.detach().clone().requires_grad_()
    full_correction = torch.tensor([1.0, 1.0, -1.0])
    reliability = torch.tensor([0.2, 0.8], requires_grad=True)
    torch.testing.assert_close(
        (
            probabilities[0].detach() * full_correction * teacher_contrast[0].detach()
        ).sum(),
        torch.tensor(0.0),
    )

    gate = legacy_dual_cosine_gate(
        teacher_contrast,
        brain_contrast,
        probabilities,
        reliability=reliability,
    )

    torch.testing.assert_close(gate, torch.tensor([0.2, 0.8]))
    assert not gate.requires_grad


def test_fp16_legacy_gate_promotes_before_norm_arithmetic() -> None:
    gate = legacy_dual_cosine_gate(
        torch.tensor(
            [2e-7, -2e-7],
            dtype=torch.float16,
            requires_grad=True,
        ),
        torch.tensor([1.0, -1.0], dtype=torch.float16, requires_grad=True),
        torch.tensor([0.5, 0.5], dtype=torch.float16, requires_grad=True),
        reliability=torch.tensor(0.75, dtype=torch.float16, requires_grad=True),
    )

    torch.testing.assert_close(gate, torch.tensor(0.75))
    assert gate.dtype == torch.float32
    assert torch.isfinite(gate)
    assert not gate.requires_grad


def test_float64_legacy_gate_preserves_large_finite_geometry() -> None:
    gate = legacy_dual_cosine_gate(
        torch.tensor([1e150, -1e150], dtype=torch.float64, requires_grad=True),
        torch.tensor([2e150, -2e150], dtype=torch.float64, requires_grad=True),
        torch.tensor([0.5, 0.5], dtype=torch.float64, requires_grad=True),
        reliability=torch.tensor(0.75, dtype=torch.float64, requires_grad=True),
    )

    torch.testing.assert_close(gate, torch.tensor(0.75, dtype=torch.float64))
    assert gate.dtype == torch.float64
    assert torch.isfinite(gate)
    assert not gate.requires_grad
