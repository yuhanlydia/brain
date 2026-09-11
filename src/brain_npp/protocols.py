"""Explicit adapter contract for NPP-OPSD backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch

from .objective import TeacherCandidateSource


@dataclass(frozen=True)
class Rollout:
    """Fixed on-policy token IDs together with their authoritative mask."""

    token_ids: torch.Tensor
    token_mask: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.token_ids, torch.Tensor) or not isinstance(self.token_mask, torch.Tensor):
            raise ValueError("rollout token_ids and token_mask must be tensors")
        if self.token_ids.ndim != 2 or self.token_mask.ndim != 2:
            raise ValueError("rollout token_ids and token_mask must have rank 2")
        if self.token_ids.shape != self.token_mask.shape:
            raise ValueError("rollout token_ids and token_mask must have the same shape")
        if min(self.token_ids.shape) < 1:
            raise ValueError("rollout axes must be non-empty")
        if self.token_ids.dtype == torch.bool or self.token_ids.is_floating_point():
            raise ValueError("rollout token_ids must use an integer dtype")
        if self.token_mask.dtype != torch.bool:
            raise ValueError("rollout token_mask must be boolean")
        if self.token_ids.device != self.token_mask.device:
            raise ValueError("rollout tensors must share a device")
        if not self.token_mask.any(dim=-1).all():
            raise ValueError("each rollout row must contain an active token")

    def detached_clone(self) -> Rollout:
        return Rollout(self.token_ids.detach().clone(), self.token_mask.detach().clone())


class PosteriorTeacherStudent(Protocol):
    def generate_student(self, batch: Any, generation_config: Any) -> Rollout: ...

    def student_logits(self, batch: Any, rollout: Rollout) -> torch.Tensor: ...

    def teacher_candidate_logits(
        self, batch: Any, rollout: Rollout
    ) -> TeacherCandidateSource: ...

    def brain_candidate_scores(self, batch: Any) -> torch.Tensor: ...

    def student_supervised_logits(
        self,
        batch: Any,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor: ...


__all__ = ["PosteriorTeacherStudent", "Rollout"]
