"""Validation-only adapter boundary for UMBRAE/BrainHub features."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from torch import Tensor

from brain_evidence.protocols import BrainOnlyBatch, PrivilegedTeacherBatch

from .bit import (
    ExternalManifest,
    ManifestSource,
    _load_external_manifest,
    _validate_feature_tensor,
    _validate_mask,
    _validate_shared_inputs,
)


@dataclass(frozen=True, slots=True)
class BrainHubManifest(ExternalManifest):
    """Validated local UMBRAE/BrainHub input manifest."""


@dataclass(frozen=True, slots=True)
class BrainHubAdapter:
    """Shape and manifest boundary for brain/image-token grounding models."""

    manifest: BrainHubManifest | None = None
    brain_feature_dim: int | None = None
    privilege_feature_dim: int | None = None

    def __post_init__(self) -> None:
        if self.brain_feature_dim is not None and self.brain_feature_dim <= 0:
            raise ValueError("brain_feature_dim must be positive")
        if self.privilege_feature_dim is not None and self.privilege_feature_dim <= 0:
            raise ValueError("privilege_feature_dim must be positive")

    @classmethod
    def from_manifest(
        cls,
        source: ManifestSource,
        *,
        brain_feature_dim: int | None = None,
        privilege_feature_dim: int | None = None,
    ) -> BrainHubAdapter:
        return cls(
            manifest=cls.validate_manifest(source),
            brain_feature_dim=brain_feature_dim,
            privilege_feature_dim=privilege_feature_dim,
        )

    @staticmethod
    def validate_manifest(source: ManifestSource) -> BrainHubManifest:
        return _load_external_manifest(
            source,
            adapter_name="BrainHub",
            accepted_backends=frozenset({"brainhub", "umbrae"}),
            manifest_type=BrainHubManifest,
        )

    def student_batch(
        self,
        brain: Tensor,
        prefix: Tensor,
        *,
        brain_mask: Tensor | None = None,
        prefix_mask: Tensor | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> BrainOnlyBatch:
        _validate_shared_inputs(
            brain,
            prefix,
            brain_feature_dim=self.brain_feature_dim,
            brain_mask=brain_mask,
            prefix_mask=prefix_mask,
        )
        return BrainOnlyBatch(
            brain=brain,
            prefix=prefix,
            brain_mask=brain_mask,
            prefix_mask=prefix_mask,
            metadata={} if metadata is None else dict(metadata),
        )

    def teacher_batch(
        self,
        brain: Tensor,
        privilege: Tensor,
        prefix: Tensor,
        *,
        brain_mask: Tensor | None = None,
        privilege_mask: Tensor | None = None,
        prefix_mask: Tensor | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PrivilegedTeacherBatch:
        _validate_shared_inputs(
            brain,
            prefix,
            brain_feature_dim=self.brain_feature_dim,
            brain_mask=brain_mask,
            prefix_mask=prefix_mask,
        )
        expected_privilege_dim = (
            self.privilege_feature_dim
            if self.privilege_feature_dim is not None
            else self.brain_feature_dim
        )
        _validate_feature_tensor(
            privilege,
            name="privilege",
            expected_feature_dim=expected_privilege_dim,
        )
        if privilege.shape[0] != brain.shape[0]:
            raise ValueError(
                "privilege and brain batch sizes must match; "
                f"got {privilege.shape[0]} and {brain.shape[0]}"
            )
        _validate_mask(
            privilege_mask,
            name="privilege_mask",
            shape=(privilege.shape[0], privilege.shape[1]),
        )
        return PrivilegedTeacherBatch(
            brain=brain,
            privilege=privilege,
            prefix=prefix,
            brain_mask=brain_mask,
            privilege_mask=privilege_mask,
            prefix_mask=prefix_mask,
            metadata={} if metadata is None else dict(metadata),
        )


def validate_manifest(source: ManifestSource) -> BrainHubManifest:
    """Validate and resolve a BrainHub manifest without loading artifacts."""

    return BrainHubAdapter.validate_manifest(source)


__all__ = ["BrainHubAdapter", "BrainHubManifest", "validate_manifest"]
