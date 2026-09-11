"""Deterministic CPU adapter for the arithmetic NPP smoke experiment."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .protocols import Rollout


class ToyAdapter(torch.nn.Module):
    def __init__(self, *, seed=0, batch_size=4, time_steps=3, candidates=4, vocab_size=5):
        super().__init__()
        if min(batch_size, time_steps, candidates) < 1 or vocab_size < 2:
            raise ValueError("toy dimensions must be positive and vocab_size at least two")
        self.seed, self.batch_size, self.time_steps = seed, batch_size, time_steps
        self.candidates, self.vocab_size = candidates, vocab_size
        self.student = torch.nn.Linear(candidates, time_steps * vocab_size)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            self.student.weight.normal_(0.0, 0.02, generator=generator)
            self.student.bias.zero_()
        self._generated_ids = self._generated_mask = None
        self._teacher_rollouts: list[Rollout] = []

    def make_batch(self) -> dict[str, torch.Tensor]:
        repeats = (self.batch_size + self.candidates - 1) // self.candidates
        brain = torch.eye(self.candidates).repeat((repeats, 1))[: self.batch_size]
        return {
            "brain": brain,
            "candidate_log_prior": torch.zeros(self.batch_size, self.candidates),
        }

    def generate_student(self, batch, generation_config: Any) -> Rollout:
        if not isinstance(generation_config, Mapping):
            raise ValueError("generation_config must be a mapping")
        steps = generation_config.get("max_new_tokens", self.time_steps)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        steps = min(steps, self.time_steps)
        with torch.no_grad():
            logits = self._all_logits(batch)[..., :steps, :]
            ids = logits.argmax(dim=-1).detach().clone()
            mask = torch.ones_like(ids, dtype=torch.bool)
        self._generated_ids, self._generated_mask = ids.clone(), mask.clone()
        self._teacher_rollouts.clear()
        return Rollout(ids, mask)

    def student_logits(self, batch, rollout: Rollout) -> torch.Tensor:
        return self._all_logits(batch)[:, : rollout.token_ids.shape[1]]

    def student_supervised_logits(self, batch, target_ids, target_mask) -> torch.Tensor:
        del target_mask
        return self._all_logits(batch)[:, : target_ids.shape[1]]

    def teacher_candidate_logits(self, batch, rollout: Rollout):
        del batch
        self._teacher_rollouts.append(rollout.detached_clone())
        batch_size, time_steps = rollout.token_ids.shape
        low = 0.15 / (self.vocab_size - 1)
        probabilities = torch.full(
            (batch_size, time_steps, self.candidates, self.vocab_size),
            low,
            dtype=self.student.weight.dtype,
            device=rollout.token_ids.device,
        )
        prefix = torch.zeros_like(rollout.token_ids)
        prefix[:, 1:] = rollout.token_ids[:, :-1]
        preferred = (prefix[:, :, None] + torch.arange(self.candidates, device=prefix.device)).remainder(self.vocab_size)
        probabilities.scatter_(-1, preferred[..., None], 0.85)
        return probabilities.log()

    def brain_candidate_scores(self, batch):
        scores = torch.linspace(2.5, -1.0, self.candidates, dtype=self.student.weight.dtype, device=batch["brain"].device)
        return scores.expand(batch["brain"].shape[0], -1)

    def _all_logits(self, batch):
        return self.student(batch["brain"]).reshape(-1, self.time_steps, self.vocab_size)

    @property
    def prefix_consistent(self) -> bool:
        return (
            self._generated_ids is not None
            and bool(self._teacher_rollouts)
            and all(torch.equal(r.token_ids, self._generated_ids) and torch.equal(r.token_mask, self._generated_mask) for r in self._teacher_rollouts)
            and all(r.token_ids.data_ptr() != self._generated_ids.data_ptr() for r in self._teacher_rollouts)
        )


ToyPosteriorTeacherStudent = ToyAdapter

__all__ = ["ToyAdapter", "ToyPosteriorTeacherStudent"]
