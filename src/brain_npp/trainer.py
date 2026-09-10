"""One-step NPP-OPSD optimization orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .objective import build_npp_target, npp_opsd_loss
from .posterior import build_neural_posterior
from .protocols import PosteriorTeacherStudent


class NPPTrainer:
    """Optimize a student against on-policy CE and neural-posterior targets."""

    def __init__(
        self,
        adapter: PosteriorTeacherStudent,
        optimizer: torch.optim.Optimizer,
        *,
        generation_config: Any = None,
        ce_weight: float = 1.0,
        npp_weight: float = 1.0,
        alpha_scale: float = 1.0,
        alpha_max: float | None = 1.0,
        clip_log_ratio: float | None = None,
    ) -> None:
        if ce_weight < 0 or npp_weight < 0 or ce_weight + npp_weight == 0:
            raise ValueError("ce_weight and npp_weight must be non-negative with a positive sum")
        self.adapter = adapter
        self.optimizer = optimizer
        self.generation_config = {} if generation_config is None else generation_config
        self.ce_weight = ce_weight
        self.npp_weight = npp_weight
        self.alpha_scale = alpha_scale
        self.alpha_max = alpha_max
        self.clip_log_ratio = clip_log_ratio
        self.last_rollout_ids: torch.Tensor | None = None

    def step(self, batch: Any) -> dict[str, float | int]:
        """Generate one fixed rollout, construct its target, and update the student."""
        generated = self.adapter.generate_student(batch, self.generation_config)
        if not isinstance(generated, torch.Tensor) or generated.ndim != 2:
            raise ValueError("generate_student must return rollout_ids with shape [batch, time]")
        fixed_rollout = generated.detach().clone()
        self.last_rollout_ids = fixed_rollout.clone()

        student_logits = self.adapter.student_logits(batch, fixed_rollout.clone())
        token_mask = _batch_value(batch, "token_mask")
        ce_loss, ce_token_count = _on_policy_cross_entropy(
            student_logits, fixed_rollout, token_mask
        )

        with torch.no_grad():
            reference_logits = self.adapter.student_reference_logits(
                batch, fixed_rollout.clone()
            ).detach()
            teacher_log_probs = self.adapter.teacher_candidate_log_probs(
                batch, fixed_rollout.clone()
            ).detach()
            brain_log_likelihoods = self.adapter.brain_candidate_log_likelihoods(
                batch
            ).detach()
            candidate_mask = _batch_value(batch, "candidate_mask")
            log_prior = _batch_value(batch, "candidate_log_prior")
            if log_prior is None:
                log_prior = torch.zeros_like(brain_log_likelihoods)
            else:
                log_prior = torch.as_tensor(
                    log_prior,
                    dtype=brain_log_likelihoods.dtype,
                    device=brain_log_likelihoods.device,
                ).detach()
            posterior = build_neural_posterior(
                brain_log_likelihoods, log_prior, candidate_mask
            )
            target = build_npp_target(
                teacher_log_probs,
                reference_logits,
                posterior,
                alpha_scale=self.alpha_scale,
                alpha_max=self.alpha_max,
            )

        npp_stats = npp_opsd_loss(
            target.log_probabilities,
            student_logits,
            token_mask,
            clip_log_ratio=self.clip_log_ratio,
        )
        loss = self.ce_weight * ce_loss + self.npp_weight * npp_stats.loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = _gradient_norm(self.optimizer)
        self.optimizer.step()

        with torch.no_grad():
            posterior_probabilities = posterior.posterior.exp()
            posterior_log = torch.where(
                posterior_probabilities > 0,
                posterior.posterior,
                torch.zeros_like(posterior.posterior),
            )
            posterior_entropy = -(
                posterior_probabilities * posterior_log
            ).sum(dim=-1).mean()
            target_shift = 0.5 * (
                target.log_probabilities.exp()
                - torch.softmax(reference_logits, dim=-1)
            ).abs().sum(dim=-1).mean()

        return {
            "loss": float(loss.detach()),
            "ce_loss": float(ce_loss.detach()),
            "npp_loss": float(npp_stats.loss.detach()),
            "information_gain": float(posterior.information_gain.mean()),
            "posterior_entropy": float(posterior_entropy),
            "target_shift": float(target_shift),
            "gradient_norm": gradient_norm,
            "token_count": int(ce_token_count),
        }


def _batch_value(batch: Any, name: str) -> Any:
    if isinstance(batch, Mapping):
        return batch.get(name)
    return getattr(batch, name, None)


def _on_policy_cross_entropy(
    student_logits: torch.Tensor,
    rollout_ids: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if student_logits.ndim != 3 or student_logits.shape[:2] != rollout_ids.shape:
        raise ValueError("student_logits must have shape [batch, time, vocab]")
    if rollout_ids.dtype == torch.bool or rollout_ids.is_floating_point():
        raise ValueError("rollout_ids must contain integer token IDs")
    if token_mask is None:
        mask = torch.ones_like(rollout_ids, dtype=torch.bool)
    else:
        if token_mask.shape != rollout_ids.shape or token_mask.dtype != torch.bool:
            raise ValueError("token_mask must be boolean with shape [batch, time]")
        mask = token_mask
    token_count = mask.sum()
    if token_count.item() == 0:
        raise ValueError("token_mask must contain at least one unmasked token")
    per_token = -torch.log_softmax(student_logits, dim=-1).gather(
        -1, rollout_ids.to(torch.long).unsqueeze(-1)
    ).squeeze(-1)
    return (per_token * mask).sum() / token_count, token_count


def _gradient_norm(optimizer: torch.optim.Optimizer) -> float:
    squared_norm = 0.0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is not None:
                squared_norm += float(parameter.grad.detach().square().sum())
    return squared_norm**0.5
