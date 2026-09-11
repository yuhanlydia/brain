"""Neural evidence posterior with explicit score semantics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Literal

import torch


ScoreSemantics = Literal["log_likelihood", "posterior_logits", "compatibility"]


@dataclass(frozen=True)
class NeuralPosterior:
    """Normalized log posterior/reference weights and KL(w || r)."""

    posterior: torch.Tensor
    prior: torch.Tensor
    information_gain: torch.Tensor
    candidate_mask: torch.Tensor

    @property
    def log_posterior_weights(self) -> torch.Tensor:
        return self.posterior

    @property
    def log_prior_weights(self) -> torch.Tensor:
        return self.prior

    @property
    def kl_posterior_prior(self) -> torch.Tensor:
        return self.information_gain


def normalize_candidate_prior(
    log_prior: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalize candidate log weights in fp32 or better."""
    _validate_rank2_float("log_prior", log_prior)
    mask = _candidate_mask(log_prior, candidate_mask)
    values = log_prior.detach().to(_compute_dtype(log_prior))
    active = values[mask]
    if torch.isnan(active).any() or torch.isposinf(active).any():
        raise ValueError("active log_prior must not contain NaN or +inf")
    masked = values.masked_fill(~mask, -torch.inf)
    if not torch.isfinite(masked).any(dim=-1).all():
        raise ValueError("each row must have prior support")
    return masked - torch.logsumexp(masked, dim=-1, keepdim=True)


def build_neural_posterior(
    candidate_scores: torch.Tensor,
    log_prior: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
    *,
    score_semantics: ScoreSemantics = "log_likelihood",
    log_proposal: torch.Tensor | None = None,
    compatibility_temperature: float | None = None,
) -> NeuralPosterior:
    """Build w(I|b), applying the candidate prior exactly once.

    ``log_likelihood`` means scores proportional to log p(b|I).
    ``posterior_logits`` means scores already parameterize w and therefore
    are normalized directly. ``compatibility`` is an explicitly calibrated
    energy, divided by ``compatibility_temperature`` before Bayes updating.
    A supplied proposal distribution is importance-corrected once.
    """
    _validate_rank2_float("candidate_scores", candidate_scores)
    _validate_rank2_float("log_prior", log_prior)
    if candidate_scores.shape != log_prior.shape:
        raise ValueError("candidate_scores and log_prior must have matching shapes")
    if candidate_scores.device != log_prior.device:
        raise ValueError("candidate_scores and log_prior must share a device")
    if score_semantics not in {"log_likelihood", "posterior_logits", "compatibility"}:
        raise ValueError(
            "score_semantics must be one of: compatibility, log_likelihood, posterior_logits"
        )
    if score_semantics == "compatibility":
        _validate_positive("compatibility_temperature", compatibility_temperature)
    elif compatibility_temperature is not None:
        raise ValueError(
            "compatibility_temperature is only valid for compatibility scores"
        )
    if score_semantics == "posterior_logits" and log_proposal is not None:
        raise ValueError("log_proposal is not valid for posterior_logits scores")

    mask = _candidate_mask(candidate_scores, candidate_mask)
    dtype = _compute_dtype(candidate_scores, log_prior, log_proposal)
    scores = candidate_scores.detach().to(dtype)
    active_scores = scores[mask]
    if torch.isnan(active_scores).any() or torch.isposinf(active_scores).any():
        raise ValueError("active candidate_scores must not contain NaN or +inf")
    prior = normalize_candidate_prior(log_prior, mask).to(dtype)

    if score_semantics == "posterior_logits":
        posterior_unnormalized = scores.masked_fill(~mask, -torch.inf)
    else:
        evidence = scores
        if score_semantics == "compatibility":
            evidence = evidence / float(compatibility_temperature)
        posterior_unnormalized = prior + evidence
        if log_proposal is not None:
            _validate_rank2_float("log_proposal", log_proposal)
            if log_proposal.shape != scores.shape or log_proposal.device != scores.device:
                raise ValueError("log_proposal must match candidate_scores shape and device")
            proposal = normalize_candidate_prior(log_proposal, mask).to(dtype)
            prior_support = torch.isfinite(prior)
            if (prior_support & torch.isneginf(proposal)).any():
                raise ValueError("proposal support must contain prior support")
            posterior_unnormalized = torch.where(
                prior_support,
                posterior_unnormalized - proposal,
                torch.full_like(posterior_unnormalized, -torch.inf),
            )
        posterior_unnormalized = posterior_unnormalized.masked_fill(~mask, -torch.inf)

    if not torch.isfinite(posterior_unnormalized).any(dim=-1).all():
        raise ValueError("each row must have posterior support")
    posterior = posterior_unnormalized - torch.logsumexp(
        posterior_unnormalized, dim=-1, keepdim=True
    )
    posterior_support = mask & torch.isfinite(posterior)
    if (posterior_support & torch.isneginf(prior)).any():
        raise ValueError("posterior support must be contained in prior support")

    probabilities = posterior.exp()
    log_ratio = torch.where(
        probabilities > 0, posterior - prior, torch.zeros_like(posterior)
    )
    information_gain = (probabilities * log_ratio).sum(dim=-1).clamp_min(0.0)
    if not torch.isfinite(information_gain).all():
        raise ValueError("information gain must be finite")
    return NeuralPosterior(posterior, prior, information_gain, mask.detach().clone())


def _candidate_mask(
    values: torch.Tensor, candidate_mask: torch.Tensor | None
) -> torch.Tensor:
    if values.shape[1] < 1:
        raise ValueError("candidate bank must contain at least one candidate")
    if candidate_mask is None:
        return torch.ones_like(values, dtype=torch.bool)
    if not isinstance(candidate_mask, torch.Tensor):
        raise ValueError("candidate_mask must be a tensor")
    if candidate_mask.ndim != 2 or candidate_mask.shape != values.shape:
        raise ValueError("candidate_mask must have the same rank 2 shape as the inputs")
    if candidate_mask.dtype != torch.bool:
        raise ValueError("candidate_mask must be a boolean tensor")
    if candidate_mask.device != values.device:
        raise ValueError("candidate_mask must be on the same device as the inputs")
    if not candidate_mask.any(dim=-1).all():
        raise ValueError("each row must have at least one valid candidate")
    return candidate_mask.detach().clone()


def _validate_rank2_float(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if value.ndim != 2:
        raise ValueError(f"{name} must have rank 2 [batch, candidates]")
    if value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError(f"{name} axes must be non-empty")
    if not value.is_floating_point():
        raise ValueError(f"{name} must be floating-point")


def _compute_dtype(*values: torch.Tensor | None) -> torch.dtype:
    dtype = torch.float32
    for value in values:
        if isinstance(value, torch.Tensor):
            dtype = torch.promote_types(dtype, value.dtype)
    return dtype


def _validate_positive(name: str, value: float | None) -> None:
    if (
        value is None
        or isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{name} must be a finite positive real number")


__all__ = [
    "NeuralPosterior",
    "ScoreSemantics",
    "build_neural_posterior",
    "normalize_candidate_prior",
]
