"""Typed contracts shared by brain-conditioned model adapters.

The student contract contains only neural evidence and an autoregressive
prefix.  Privileged inputs exist only on the frozen-teacher path.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class BrainOnlyBatch:
    """Inputs available to the student during training and inference."""

    brain: Tensor
    prefix: Tensor
    brain_mask: Tensor | None = None
    prefix_mask: Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def prefix_tokens(self) -> Tensor:
        """Compatibility name emphasizing that the prefix contains token IDs."""

        return self.prefix


@dataclass(frozen=True, slots=True)
class PrivilegedTeacherBatch:
    """Frozen-teacher inputs; ``privilege`` must never reach the student."""

    brain: Tensor
    privilege: Tensor
    prefix: Tensor
    brain_mask: Tensor | None = None
    privilege_mask: Tensor | None = None
    prefix_mask: Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def prefix_tokens(self) -> Tensor:
        return self.prefix

    @property
    def privilege_tokens(self) -> Tensor:
        return self.privilege


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Model-family-independent generation settings."""

    max_new_tokens: int
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0


@runtime_checkable
class BrainConditionedModel(Protocol):
    """Minimal model surface required by the training objectives."""

    def student_logits(self, brain: Tensor, prefix: Tensor) -> Tensor:
        """Return brain-only next-token logits."""

        ...

    def teacher_logits(
        self, brain: Tensor, privilege: Tensor, prefix: Tensor
    ) -> Tensor:
        """Return frozen privileged-teacher next-token logits."""

        ...

    def generate(
        self, brain: Tensor, generation_config: GenerationConfig | Mapping[str, Any]
    ) -> Tensor:
        """Generate without privileged teacher inputs."""

        ...

    def trainable_parameters(self) -> Iterable[torch.nn.Parameter]:
        """Yield only the parameters optimized on the student path."""

        ...


# Concise aliases for callers that prefer role-oriented names.
StudentBatch = BrainOnlyBatch
TeacherBatch = PrivilegedTeacherBatch


__all__ = [
    "BrainConditionedModel",
    "BrainOnlyBatch",
    "GenerationConfig",
    "PrivilegedTeacherBatch",
    "StudentBatch",
    "TeacherBatch",
]
