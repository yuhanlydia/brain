from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NeuralPosterior:
    posterior: torch.Tensor
    prior: torch.Tensor
    information_gain: torch.Tensor
    candidate_mask: torch.Tensor


def normalize_candidate_prior(
    log_prior: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if log_prior.ndim != 2:
        raise ValueError("log_prior must have rank 2 [batch, candidates]")

    mask = _candidate_mask(log_prior, candidate_mask)
    if log_prior.dtype in (torch.float16, torch.bfloat16):
        log_prior = log_prior.float()
    masked_log_prior = log_prior.masked_fill(~mask, -torch.inf)
    prior_max = masked_log_prior.max(dim=-1, keepdim=True).values
    if not torch.isfinite(prior_max).all():
        raise ValueError("each row must have prior support")
    centered_log_prior = masked_log_prior - prior_max
    return centered_log_prior - torch.logsumexp(
        centered_log_prior, dim=-1, keepdim=True
    )


def build_neural_posterior(
    log_likelihood: torch.Tensor,
    log_prior: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
) -> NeuralPosterior:
    if log_likelihood.ndim != 2 or log_prior.ndim != 2:
        raise ValueError("log_likelihood and log_prior must have rank 2 [batch, candidates]")
    if log_likelihood.shape != log_prior.shape:
        raise ValueError("log_likelihood and log_prior must have matching shapes")

    mask = _candidate_mask(log_prior, candidate_mask)
    prior = normalize_candidate_prior(log_prior, mask)
    masked_log_likelihood = log_likelihood.masked_fill(~mask, -torch.inf)
    candidate_receives_posterior_support = mask & torch.isfinite(masked_log_likelihood)
    if (candidate_receives_posterior_support & torch.isneginf(prior)).any():
        raise ValueError("prior support is absent for a candidate receiving posterior support")

    likelihood_max = masked_log_likelihood.max(dim=-1, keepdim=True).values
    if not torch.isfinite(likelihood_max).all():
        raise ValueError("each row must have posterior support")
    stabilized_likelihood = masked_log_likelihood - likelihood_max
    posterior = prior + stabilized_likelihood
    posterior = posterior - torch.logsumexp(posterior, dim=-1, keepdim=True)
    posterior_probabilities = posterior.exp()
    log_ratio = torch.where(
        posterior_probabilities > 0,
        posterior - prior,
        torch.zeros_like(posterior),
    )
    information_gain = (posterior_probabilities * log_ratio).sum(dim=-1)
    return NeuralPosterior(posterior, prior, information_gain, mask)


def _candidate_mask(
    values: torch.Tensor,
    candidate_mask: torch.Tensor | None,
) -> torch.Tensor:
    if candidate_mask is None:
        return torch.ones_like(values, dtype=torch.bool)
    if candidate_mask.ndim != 2 or candidate_mask.shape != values.shape:
        raise ValueError("candidate_mask must have the same rank 2 shape as the inputs")
    if candidate_mask.dtype != torch.bool:
        raise ValueError("candidate_mask must be a boolean tensor")
    if not candidate_mask.any(dim=-1).all():
        raise ValueError("each row must have at least one valid candidate")
    return candidate_mask
