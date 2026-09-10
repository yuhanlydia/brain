from dataclasses import dataclass

import torch

from .posterior import NeuralPosterior


@dataclass(frozen=True)
class NPPTarget:
    log_probabilities: torch.Tensor
    utility: torch.Tensor
    strength: torch.Tensor


@dataclass(frozen=True)
class NPPLossStats:
    loss: torch.Tensor
    token_count: torch.Tensor


def build_npp_target(
    teacher_log_probs: torch.Tensor,
    student_reference_logits: torch.Tensor,
    neural_posterior: NeuralPosterior,
    *,
    alpha_scale: float = 1.0,
    alpha_max: float | None = None,
) -> NPPTarget:
    """Build a detached posterior-contrastive distribution over each token."""
    _validate_target_inputs(
        teacher_log_probs, student_reference_logits, neural_posterior
    )
    with torch.no_grad():
        # Normalize and take expectations in at least fp32; finite fp16 logits
        # can otherwise overflow when their log-softmax differences exceed 65504.
        target_dtype = torch.float32
        for values in (
            teacher_log_probs, student_reference_logits, neural_posterior.posterior,
            neural_posterior.prior, neural_posterior.information_gain,
        ):
            target_dtype = torch.promote_types(target_dtype, values.dtype)
        normalized_teacher_log_probs = torch.log_softmax(
            teacher_log_probs.to(target_dtype), dim=-1
        )
        candidate_mask = neural_posterior.candidate_mask[:, None, :, None]
        normalized_teacher_log_probs = torch.where(
            candidate_mask,
            normalized_teacher_log_probs,
            torch.zeros_like(normalized_teacher_log_probs),
        )
        posterior_expectation = (
            neural_posterior.posterior.to(target_dtype).exp()[:, None, :, None]
            * normalized_teacher_log_probs
        ).sum(dim=2)
        prior_expectation = (
            neural_posterior.prior.to(target_dtype).exp()[:, None, :, None]
            * normalized_teacher_log_probs
        ).sum(dim=2)
        utility = posterior_expectation - prior_expectation
        strength = neural_posterior.information_gain.detach().to(target_dtype)
        if alpha_max is not None:
            strength = torch.clamp(strength / alpha_scale, max=alpha_max)
        log_probabilities = torch.log_softmax(
            student_reference_logits.to(target_dtype) + strength[:, None, None] * utility,
            dim=-1,
        )
        return NPPTarget(log_probabilities, utility, strength)


def _validate_target_inputs(
    teacher_log_probs: torch.Tensor,
    student_reference_logits: torch.Tensor,
    neural_posterior: NeuralPosterior,
) -> None:
    if teacher_log_probs.ndim != 4:
        raise ValueError(
            "teacher_log_probs must have rank 4 [batch, time, candidates, vocab]"
        )
    if student_reference_logits.ndim != 3:
        raise ValueError(
            "student_reference_logits must have rank 3 [batch, time, vocab]"
        )

    batch, time, candidates, vocab = teacher_log_probs.shape
    if student_reference_logits.shape != (batch, time, vocab):
        raise ValueError(
            "teacher_log_probs and student_reference_logits must share batch, time, and vocab shapes"
        )
    if neural_posterior.posterior.shape != (batch, candidates):
        raise ValueError("neural_posterior.posterior must have shape [batch, candidates]")
    if neural_posterior.prior.shape != (batch, candidates):
        raise ValueError("neural_posterior.prior must have shape [batch, candidates]")
    if neural_posterior.information_gain.shape != (batch,):
        raise ValueError("neural_posterior.information_gain must have shape [batch]")
    if neural_posterior.candidate_mask.shape != (batch, candidates):
        raise ValueError(
            "neural_posterior.candidate_mask must have shape [batch, candidates]"
        )


def forward_kl_loss(
    target_log_probs: torch.Tensor,
    student_logits: torch.Tensor,
    token_mask: torch.Tensor | None = None,
    *,
    clip_log_ratio: float | None = None,
) -> NPPLossStats:
    """Return the masked mean of ``KL(target || student)`` over token positions."""
    _validate_loss_inputs(target_log_probs, student_logits, token_mask)
    if clip_log_ratio is not None and clip_log_ratio < 0:
        raise ValueError("clip_log_ratio must be non-negative")

    if token_mask is None:
        mask = torch.ones_like(target_log_probs[..., 0], dtype=torch.bool)
    else:
        mask = token_mask
    token_count = mask.sum()
    if token_count.item() == 0:
        raise ValueError("token_mask must contain at least one unmasked token; received empty mask")

    with torch.no_grad():
        detached_target_log_probs = target_log_probs.detach()[mask]
        target_probabilities = detached_target_log_probs.exp()

    # Select valid tokens before arithmetic, so invalid padded values cannot
    # contaminate either the reduction or log-softmax's backward pass.
    student_log_probs = torch.log_softmax(student_logits[mask], dim=-1)
    log_ratio = torch.where(
        target_probabilities > 0,
        detached_target_log_probs - student_log_probs,
        torch.zeros_like(student_log_probs),
    )
    if clip_log_ratio is not None:
        log_ratio = log_ratio.clamp(-clip_log_ratio, clip_log_ratio)
    per_token_loss = (target_probabilities * log_ratio).sum(dim=-1)

    loss = per_token_loss.sum() / token_count
    return NPPLossStats(loss=loss, token_count=token_count)


def npp_opsd_loss(
    target_log_probs: torch.Tensor,
    student_logits: torch.Tensor,
    token_mask: torch.Tensor | None = None,
    *,
    clip_log_ratio: float | None = None,
) -> NPPLossStats:
    """Task 4 entrypoint for the differentiable NPP forward-KL objective."""
    return forward_kl_loss(
        target_log_probs,
        student_logits,
        token_mask,
        clip_log_ratio=clip_log_ratio,
    )


def _validate_loss_inputs(
    target_log_probs: torch.Tensor,
    student_logits: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> None:
    if target_log_probs.ndim != 3:
        raise ValueError("target_log_probs must have rank 3 [batch, time, vocab]")
    if student_logits.ndim != 3:
        raise ValueError("student_logits must have rank 3 [batch, time, vocab]")
    if student_logits.shape != target_log_probs.shape:
        raise ValueError("target_log_probs and student_logits must have matching shapes")
    if token_mask is not None:
        if token_mask.ndim != 2 or token_mask.shape != target_log_probs.shape[:2]:
            raise ValueError("token_mask must have shape [batch, time]")
        if token_mask.dtype != torch.bool:
            raise ValueError("token_mask must be a boolean tensor")
