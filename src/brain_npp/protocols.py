"""Public adapter contract for NPP-OPSD training backends."""

from __future__ import annotations

from typing import Any, Protocol

import torch


class PosteriorTeacherStudent(Protocol):
    """Operations required by :class:`brain_npp.trainer.NPPTrainer`."""

    def generate_student(self, batch: Any, generation_config: Any) -> torch.Tensor: ...

    def student_logits(self, batch: Any, rollout_ids: torch.Tensor) -> torch.Tensor: ...

    def student_reference_logits(
        self, batch: Any, rollout_ids: torch.Tensor
    ) -> torch.Tensor: ...

    def teacher_candidate_log_probs(
        self, batch: Any, rollout_ids: torch.Tensor
    ) -> torch.Tensor: ...

    def brain_candidate_log_likelihoods(self, batch: Any) -> torch.Tensor: ...
