"""Fixed-rollout optimization for arithmetic NPP-OPSD."""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real
from typing import Any

import torch

from .objective import (
    NPPLossStats,
    NPPTeacherCorrection,
    Pooling,
    StrengthMode,
    build_teacher_correction,
    compose_npp_target,
    npp_opsd_loss,
)
from .posterior import NeuralPosterior, ScoreSemantics, build_neural_posterior
from .protocols import PosteriorTeacherStudent, Rollout


class NPPTrainer:
    """Train against a detached target built on one immutable rollout.

    Gradient accumulation averages complete per-microbatch scalar objectives;
    it does not pretend to be a concatenated-token mean across microbatches.
    """

    def __init__(
        self,
        adapter: PosteriorTeacherStudent,
        optimizer: torch.optim.Optimizer,
        *,
        generation_config: Any = None,
        ce_weight: float = 0.0,
        npp_weight: float = 1.0,
        pooling: Pooling = "arithmetic",
        ratio_strength: float = 1.0,
        strength_mode: StrengthMode = "constant",
        strength_max: float | None = None,
        score_semantics: ScoreSemantics = "log_likelihood",
        compatibility_temperature: float | None = None,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        _validate_nonnegative("ce_weight", ce_weight)
        _validate_nonnegative("npp_weight", npp_weight)
        if float(ce_weight) + float(npp_weight) == 0:
            raise ValueError("ce_weight and npp_weight must have a positive sum")
        if pooling not in {"arithmetic", "geometric"}:
            raise ValueError("pooling must be arithmetic or geometric")
        if strength_mode not in {"constant", "sqrt_information_gain", "information_gain"}:
            raise ValueError("invalid strength_mode")
        _validate_nonnegative("ratio_strength", ratio_strength)
        if strength_max is not None:
            _validate_nonnegative("strength_max", strength_max)
        if score_semantics not in {"log_likelihood", "posterior_logits", "compatibility"}:
            raise ValueError("invalid score_semantics")
        if score_semantics == "compatibility":
            if compatibility_temperature is None or float(compatibility_temperature) <= 0:
                raise ValueError("compatibility scores require a positive temperature")
        elif compatibility_temperature is not None:
            raise ValueError("compatibility_temperature is only valid for compatibility scores")
        if isinstance(gradient_accumulation_steps, bool) or not isinstance(gradient_accumulation_steps, int) or gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be a positive integer")

        self.adapter = adapter
        self.optimizer = optimizer
        self.generation_config = {} if generation_config is None else generation_config
        self.ce_weight = float(ce_weight)
        self.npp_weight = float(npp_weight)
        self.pooling = pooling
        self.ratio_strength = float(ratio_strength)
        self.strength_mode = strength_mode
        self.strength_max = strength_max
        self.score_semantics = score_semantics
        self.compatibility_temperature = compatibility_temperature
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.last_rollout_ids: torch.Tensor | None = None
        self.last_rollout_mask: torch.Tensor | None = None
        self._pending_microbatches = 0
        self.optimizer.zero_grad(set_to_none=True)

    def step(self, batch: Any) -> dict[str, float | int]:
        """Accumulate one scalar objective and update on a full window."""
        try:
            rollout, posterior, correction = self._detached_teacher_stage(batch)
            student_logits = self.adapter.student_logits(batch, rollout.detached_clone())
            if not isinstance(student_logits, torch.Tensor) or student_logits.ndim != 3:
                raise ValueError("student_logits must return [batch,time,vocab]")
            if student_logits.shape[:2] != rollout.token_ids.shape:
                raise ValueError("student logits must match the fixed rollout")

            with torch.no_grad():
                target = compose_npp_target(
                    correction,
                    student_logits.detach(),
                    posterior,
                    ratio_strength=self.ratio_strength,
                    strength_mode=self.strength_mode,
                    strength_max=self.strength_max,
                )
            npp = npp_opsd_loss(target.log_probabilities, student_logits, rollout.token_mask)
            ce = self._supervised_loss(batch, student_logits)
            loss = self.ce_weight * ce.loss + self.npp_weight * npp.loss
            if not torch.isfinite(loss):
                raise ValueError("combined training loss must be finite")

            # Finish all fallible diagnostics before an irreversible optimizer step.
            diagnostics = _diagnostics(loss, ce, npp, posterior, target.log_probabilities, student_logits, rollout.token_mask)
            loss.backward()
            self._pending_microbatches += 1
            contributing = self._pending_microbatches
            gradient_norm = _gradient_norm(self.optimizer, divisor=contributing)
            if not math.isfinite(gradient_norm):
                raise ValueError("accumulated gradient must be finite")
            stepped = 0
            if contributing == self.gradient_accumulation_steps:
                gradient_norm = self._apply_pending_update()
                stepped = 1
            diagnostics.update(
                gradient_norm=gradient_norm,
                optimizer_step=stepped,
                accumulation_count=contributing,
            )
            return diagnostics
        except Exception:
            self._abort_pending_window()
            raise

    def flush(self) -> dict[str, float | int]:
        if self._pending_microbatches == 0:
            return {"optimizer_step": 0, "accumulation_count": 0, "gradient_norm": 0.0}
        count = self._pending_microbatches
        try:
            norm = self._apply_pending_update()
        except Exception:
            self._abort_pending_window()
            raise
        return {"optimizer_step": 1, "accumulation_count": count, "gradient_norm": norm}

    def _detached_teacher_stage(
        self, batch: Any
    ) -> tuple[Rollout, NeuralPosterior, NPPTeacherCorrection]:
        with torch.no_grad():
            generated = self.adapter.generate_student(batch, self.generation_config)
            if not isinstance(generated, Rollout):
                raise ValueError("generate_student must return a Rollout")
            rollout = generated.detached_clone()
            self.last_rollout_ids = rollout.token_ids.detach().clone()
            self.last_rollout_mask = rollout.token_mask.detach().clone()
            scores = self.adapter.brain_candidate_scores(batch)
            if not isinstance(scores, torch.Tensor):
                raise ValueError("brain_candidate_scores must return a tensor")
            scores = scores.detach()
            prior = _candidate_value(batch, "candidate_log_prior", scores, default_zero=True)
            proposal = _candidate_value(batch, "candidate_log_proposal", scores, default_zero=False)
            posterior = build_neural_posterior(
                scores,
                prior,
                _batch_value(batch, "candidate_mask"),
                score_semantics=self.score_semantics,
                log_proposal=proposal,
                compatibility_temperature=self.compatibility_temperature,
            )
            source = self.adapter.teacher_candidate_logits(batch, rollout.detached_clone())
            correction = build_teacher_correction(source, posterior, pooling=self.pooling)
            if correction.utility.shape[:2] != rollout.token_ids.shape:
                raise ValueError("teacher correction must match rollout batch/time axes")
            return rollout, posterior, correction

    def _supervised_loss(self, batch: Any, anchor: torch.Tensor) -> NPPLossStats:
        if self.ce_weight == 0:
            return NPPLossStats(
                anchor.new_zeros((), dtype=torch.float32),
                torch.zeros((), dtype=torch.long, device=anchor.device),
            )
        target_ids = _batch_value(batch, "target_ids")
        target_mask = _batch_value(batch, "target_mask")
        _validate_targets(target_ids, target_mask)
        logits = self.adapter.student_supervised_logits(
            batch, target_ids.detach().clone(), target_mask.detach().clone()
        )
        return _supervised_cross_entropy(logits, target_ids, target_mask)

    def _apply_pending_update(self) -> float:
        if self._pending_microbatches < 1:
            raise RuntimeError("no accumulated gradient")
        _divide_gradients(self.optimizer, self._pending_microbatches)
        norm = _gradient_norm(self.optimizer)
        if not math.isfinite(norm):
            raise ValueError("effective averaged gradient must be finite")
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._pending_microbatches = 0
        return norm

    def _abort_pending_window(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        self._pending_microbatches = 0


def _diagnostics(loss, ce, npp, posterior, target_log_probs, logits, token_mask):
    with torch.no_grad():
        w = posterior.posterior.exp()
        safe_log_w = torch.where(w > 0, posterior.posterior, torch.zeros_like(w))
        entropy = -(w * safe_log_w).sum(dim=-1).mean()
        anchor = torch.softmax(logits.detach().to(target_log_probs.dtype), dim=-1)
        shift = 0.5 * (target_log_probs.exp() - anchor).abs().sum(dim=-1)
        shift = shift[token_mask].mean()
        return {
            "loss": float(loss.detach()),
            "ce_loss": float(ce.loss.detach()),
            "npp_loss": float(npp.loss.detach()),
            "information_gain": float(posterior.information_gain.mean()),
            "posterior_entropy": float(entropy),
            "target_shift": float(shift),
            "token_count": int(npp.token_count),
            "ce_token_count": int(ce.token_count),
        }


def _supervised_cross_entropy(logits, target_ids, target_mask) -> NPPLossStats:
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or not logits.is_floating_point():
        raise ValueError("student supervised logits must be floating-point [batch,time,vocab]")
    _validate_targets(target_ids, target_mask)
    if logits.shape[:2] != target_ids.shape:
        raise ValueError("supervised logits and targets must match")
    if not (logits.device == target_ids.device == target_mask.device):
        raise ValueError("supervised tensors must share a device")
    active_logits = logits[target_mask].to(torch.promote_types(torch.float32, logits.dtype))
    active_ids = target_ids[target_mask].to(torch.long)
    if (active_ids < 0).any() or (active_ids >= logits.shape[-1]).any():
        raise ValueError("active target_ids must be in vocabulary range")
    if not torch.isfinite(active_logits).all():
        raise ValueError("active supervised logits must be finite")
    loss = -torch.log_softmax(active_logits, dim=-1).gather(-1, active_ids[:, None]).mean()
    return NPPLossStats(loss, target_mask.sum())


def _validate_targets(target_ids, target_mask) -> None:
    if not isinstance(target_ids, torch.Tensor) or not isinstance(target_mask, torch.Tensor):
        raise ValueError("ce_weight > 0 requires tensor target_ids and target_mask")
    if target_ids.ndim != 2 or target_mask.ndim != 2 or target_ids.shape != target_mask.shape:
        raise ValueError("target_ids and target_mask must have matching rank-2 shapes")
    if target_ids.dtype == torch.bool or target_ids.is_floating_point():
        raise ValueError("target_ids must use an integer dtype")
    if target_mask.dtype != torch.bool or not target_mask.any():
        raise ValueError("target_mask must be boolean and contain an active token")


def _batch_value(batch: Any, name: str) -> Any:
    return batch.get(name) if isinstance(batch, Mapping) else getattr(batch, name, None)


def _candidate_value(batch, name, scores, *, default_zero):
    value = _batch_value(batch, name)
    if value is None:
        return torch.zeros_like(scores) if default_zero else None
    if isinstance(value, torch.Tensor):
        return value.detach()
    return torch.as_tensor(value, dtype=scores.dtype, device=scores.device).detach()


def _divide_gradients(optimizer, divisor):
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            if parameter.grad.is_sparse:
                parameter.grad._values().div_(divisor)
            else:
                parameter.grad.div_(divisor)


def _gradient_norm(optimizer, *, divisor=1) -> float:
    squared = 0.0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is not None:
                values = parameter.grad.coalesce().values() if parameter.grad.is_sparse else parameter.grad
                squared += float((values.detach().to(torch.float64) / divisor).square().sum())
    return squared**0.5


def _validate_nonnegative(name, value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} must be a finite non-negative real number")


__all__ = ["NPPTrainer", "_supervised_cross_entropy"]
