"""Validation-only adapter boundary for Brain-to-Text inputs.

This module deliberately does not download datasets or checkpoints.  It turns
already prepared, user-supplied tensors into the package's typed contracts.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlsplit

import torch
import yaml
from torch import Tensor

from brain_evidence.protocols import BrainOnlyBatch, PrivilegedTeacherBatch

ManifestSource = str | Path | Mapping[str, Any]
_ManifestT = TypeVar("_ManifestT", bound="ExternalManifest")
_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


@dataclass(frozen=True, slots=True)
class ExternalManifest:
    """Resolved local paths for an external dataset and checkpoint."""

    backend: str
    dataset_path: Path
    checkpoint_path: Path
    revision: str
    checkpoint_sha256: str
    manifest_path: Path | None = None


@dataclass(frozen=True, slots=True)
class BITManifest(ExternalManifest):
    """Validated Brain-to-Text external input manifest."""


def _read_manifest_payload(
    source: ManifestSource,
) -> tuple[dict[str, Any], Path, Path | None]:
    if isinstance(source, Mapping):
        return dict(source), Path.cwd(), None

    manifest_path = Path(source).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest {manifest_path} was not found. Provide a user-supplied "
            "local manifest; brain-evidence never downloads private inputs."
        )
    try:
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Manifest {manifest_path} is not valid YAML: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise TypeError(f"Manifest {manifest_path} must contain a YAML mapping")
    return dict(payload), manifest_path.parent, manifest_path


def _local_existing_path(
    raw_value: Any,
    *,
    field_name: str,
    base_dir: Path,
    adapter_name: str,
    expected_kind: str,
) -> Path:
    if not isinstance(raw_value, (str, Path)) or not str(raw_value).strip():
        raise ValueError(
            f"{adapter_name} manifest requires non-empty {field_name}. Provide the "
            "user-supplied local path explicitly; brain-evidence never downloads "
            "private inputs."
        )

    raw_text = str(raw_value)
    parsed = urlsplit(raw_text)
    if parsed.scheme or parsed.netloc:
        raise ValueError(
            f"{adapter_name} {field_name} must be a local path; brain-evidence "
            "never downloads private inputs"
        )

    path = Path(raw_text).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"{adapter_name} {field_name} does not exist at {path}. Provide the "
            "user-supplied dataset/checkpoint explicitly; brain-evidence never "
            "downloads private inputs."
        )
    if expected_kind == "directory" and not path.is_dir():
        raise ValueError(f"{adapter_name} {field_name} must be a directory")
    if expected_kind == "file" and not path.is_file():
        raise ValueError(f"{adapter_name} {field_name} must be a file")
    return path


def _checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_external_manifest(
    source: ManifestSource,
    *,
    adapter_name: str,
    accepted_backends: frozenset[str],
    manifest_type: type[_ManifestT],
) -> _ManifestT:
    payload, base_dir, manifest_path = _read_manifest_payload(source)
    backend = payload.get("backend")
    if not isinstance(backend, str) or backend.lower() not in accepted_backends:
        expected = ", ".join(sorted(accepted_backends))
        raise ValueError(
            f"{adapter_name} manifest backend must be one of [{expected}]; "
            f"got {backend!r}"
        )

    revision = payload.get("revision")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError(f"{adapter_name} manifest revision must be non-empty")

    checkpoint_sha256 = payload.get("checkpoint_sha256")
    if (
        not isinstance(checkpoint_sha256, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", checkpoint_sha256) is None
    ):
        raise ValueError(
            f"{adapter_name} manifest checkpoint_sha256 must contain 64 "
            "hexadecimal characters"
        )
    checkpoint_sha256 = checkpoint_sha256.lower()

    dataset_path = _local_existing_path(
        payload.get("dataset_path"),
        field_name="dataset_path",
        base_dir=base_dir,
        adapter_name=adapter_name,
        expected_kind="directory",
    )
    checkpoint_path = _local_existing_path(
        payload.get("checkpoint_path"),
        field_name="checkpoint_path",
        base_dir=base_dir,
        adapter_name=adapter_name,
        expected_kind="file",
    )
    actual_checkpoint_sha256 = _checkpoint_digest(checkpoint_path)
    if not hmac.compare_digest(checkpoint_sha256, actual_checkpoint_sha256):
        raise ValueError(
            f"{adapter_name} manifest checkpoint_sha256 does not match "
            f"{checkpoint_path}"
        )
    return manifest_type(
        backend=backend.lower(),
        dataset_path=dataset_path,
        checkpoint_path=checkpoint_path,
        revision=revision.strip(),
        checkpoint_sha256=checkpoint_sha256,
        manifest_path=manifest_path,
    )


def _validate_feature_tensor(
    tensor: Tensor,
    *,
    name: str,
    expected_feature_dim: int | None,
) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have rank 3 [batch, sequence, feature]")
    if any(size <= 0 for size in tensor.shape):
        raise ValueError(f"{name} dimensions must all be non-zero")
    if expected_feature_dim is not None and tensor.shape[-1] != expected_feature_dim:
        raise ValueError(
            f"{name} feature dimension must be {expected_feature_dim}, "
            f"got {tensor.shape[-1]}"
        )


def _validate_token_tensor(tensor: Tensor, *, name: str) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have rank 2 [batch, sequence]")
    if any(size <= 0 for size in tensor.shape):
        raise ValueError(f"{name} dimensions must all be non-zero")
    if tensor.dtype not in _INTEGER_DTYPES:
        raise ValueError(f"{name} must contain integer token ids")


def _validate_mask(mask: Tensor | None, *, name: str, shape: tuple[int, int]) -> None:
    if mask is None:
        return
    if not isinstance(mask, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(mask.shape) != shape:
        raise ValueError(f"{name} shape must be {shape}, got {tuple(mask.shape)}")
    if mask.dtype != torch.bool:
        raise ValueError(f"{name} must have boolean dtype")


def _validate_shared_inputs(
    brain: Tensor,
    prefix: Tensor,
    *,
    brain_feature_dim: int | None,
    brain_mask: Tensor | None,
    prefix_mask: Tensor | None,
) -> None:
    _validate_feature_tensor(
        brain, name="brain", expected_feature_dim=brain_feature_dim
    )
    _validate_token_tensor(prefix, name="prefix")
    if brain.shape[0] != prefix.shape[0]:
        raise ValueError(
            "brain and prefix batch sizes must match; "
            f"got {brain.shape[0]} and {prefix.shape[0]}"
        )
    _validate_mask(
        brain_mask, name="brain_mask", shape=(brain.shape[0], brain.shape[1])
    )
    _validate_mask(
        prefix_mask, name="prefix_mask", shape=(prefix.shape[0], prefix.shape[1])
    )


@dataclass(frozen=True, slots=True)
class BITAdapter:
    """Shape and manifest boundary for BIT '24/'25 model integrations."""

    manifest: BITManifest | None = None
    brain_feature_dim: int | None = None

    def __post_init__(self) -> None:
        if self.brain_feature_dim is not None and self.brain_feature_dim <= 0:
            raise ValueError("brain_feature_dim must be positive")

    @classmethod
    def from_manifest(
        cls,
        source: ManifestSource,
        *,
        brain_feature_dim: int | None = None,
    ) -> BITAdapter:
        return cls(
            manifest=cls.validate_manifest(source),
            brain_feature_dim=brain_feature_dim,
        )

    @staticmethod
    def validate_manifest(source: ManifestSource) -> BITManifest:
        return _load_external_manifest(
            source,
            adapter_name="BIT",
            accepted_backends=frozenset({"bit", "b2t24", "b2t25", "brain-to-text"}),
            manifest_type=BITManifest,
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
        _validate_token_tensor(privilege, name="privilege")
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


def validate_manifest(source: ManifestSource) -> BITManifest:
    """Validate and resolve a BIT manifest without loading either artifact."""

    return BITAdapter.validate_manifest(source)


BitAdapter = BITAdapter


__all__ = [
    "BITAdapter",
    "BITManifest",
    "BitAdapter",
    "ExternalManifest",
    "ManifestSource",
    "validate_manifest",
]
