"""Deterministic CPU adapter used by the executable optimization smoke test."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


class ToyAdapter(torch.nn.Module):
    """A tiny linear student with fixed teachers and fixed neural evidence."""

    def __init__(
        self,
        *,
        seed: int = 0,
        batch_size: int = 4,
        time_steps: int = 3,
        candidates: int = 4,
        vocab_size: int = 5,
    ) -> None:
        super().__init__()
        if min(batch_size, time_steps, candidates) < 1 or vocab_size < 2:
            raise ValueError("toy dimensions must be positive and vocab_size at least two")
        self.seed = seed
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.candidates = candidates
        self.vocab_size = vocab_size
        self.student = torch.nn.Linear(candidates, time_steps * vocab_size)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            self.student.weight.normal_(mean=0.0, std=0.02, generator=generator)
            self.student.bias.zero_()
        self._generated_value: torch.Tensor | None = None
        self._generated_pointer: int | None = None
        self._teacher_prefixes: list[torch.Tensor] = []
        self._teacher_pointers: list[int] = []

    def make_batch(self) -> dict[str, torch.Tensor]:
        """Return one deterministic in-memory batch; no data is downloaded."""
        brain = torch.eye(self.candidates)[: self.batch_size]
        if brain.shape[0] < self.batch_size:
            repeats = (self.batch_size + self.candidates - 1) // self.candidates
            brain = torch.eye(self.candidates).repeat((repeats, 1))[: self.batch_size]
        return {
            "brain": brain,
            "candidate_log_prior": torch.full(
                (self.batch_size, self.candidates),
                -torch.log(torch.tensor(float(self.candidates))).item(),
            ),
        }

    def generate_student(
        self, batch: dict[str, torch.Tensor], generation_config: Any
    ) -> torch.Tensor:
        if not isinstance(generation_config, Mapping):
            raise ValueError("generation_config must be a mapping")
        max_new_tokens = generation_config.get("max_new_tokens", self.time_steps)
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens < 1
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        rollout_steps = min(max_new_tokens, self.time_steps)
        with torch.no_grad():
            logits = self.student(batch["brain"]).reshape(
                batch["brain"].shape[0], self.time_steps, self.vocab_size
            )
            rollout_ids = logits[:, :rollout_steps].argmax(dim=-1).detach().clone()
        self._generated_value = rollout_ids.detach().clone()
        self._generated_pointer = rollout_ids.data_ptr()
        self._teacher_prefixes.clear()
        self._teacher_pointers.clear()
        return rollout_ids

    def student_logits(
        self, batch: dict[str, torch.Tensor], rollout_ids: torch.Tensor
    ) -> torch.Tensor:
        batch_size, time_steps = rollout_ids.shape
        if not 1 <= time_steps <= self.time_steps:
            raise ValueError("toy rollout time dimension exceeds the adapter bound")
        return self.student(batch["brain"]).reshape(
            batch_size, self.time_steps, self.vocab_size
        )[:, :time_steps]

    def student_reference_logits(
        self, batch: dict[str, torch.Tensor], rollout_ids: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.student_logits(batch, rollout_ids).detach().clone()

    def teacher_candidate_log_probs(
        self, batch: dict[str, torch.Tensor], rollout_ids: torch.Tensor
    ) -> torch.Tensor:
        del batch
        self._teacher_pointers.extend(
            rollout_ids.data_ptr() for _ in range(self.candidates)
        )
        self._teacher_prefixes.extend(
            rollout_ids.detach().clone() for _ in range(self.candidates)
        )
        batch_size, time_steps = rollout_ids.shape
        low = 0.15 / (self.vocab_size - 1)
        probabilities = torch.full(
            (batch_size, time_steps, self.candidates, self.vocab_size),
            low,
            dtype=self.student.weight.dtype,
            device=rollout_ids.device,
        )
        # Token zero is the fixed BOS convention. At t > 0 only the previous
        # student token is used, never the current target or a future token.
        prefix_token = torch.zeros_like(rollout_ids)
        prefix_token[:, 1:] = rollout_ids[:, :-1]
        preferred = (
            prefix_token[:, :, None]
            + torch.arange(self.candidates, device=rollout_ids.device)
        ).remainder(self.vocab_size)
        probabilities.scatter_(-1, preferred[..., None], 0.85)
        return probabilities.log()

    def brain_candidate_log_likelihoods(
        self, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        scores = torch.linspace(
            2.5,
            -1.0,
            self.candidates,
            dtype=self.student.weight.dtype,
            device=batch["brain"].device,
        )
        return scores.expand(batch["brain"].shape[0], -1)

    @property
    def prefix_consistent(self) -> bool:
        """Whether every teacher saw a cloned value of the generated prefix."""
        return (
            self._generated_value is not None
            and bool(self._teacher_prefixes)
            and all(
                torch.equal(prefix, self._generated_value)
                for prefix in self._teacher_prefixes
            )
            and all(pointer != self._generated_pointer for pointer in self._teacher_pointers)
        )


ToyPosteriorTeacherStudent = ToyAdapter


__all__ = ["ToyAdapter", "ToyPosteriorTeacherStudent"]
