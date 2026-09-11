"""Safe, lazy loading and subject routing for the released UMBRAE BrainX."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys


SUBJECT_NUMBERS = {"subj01": 1, "subj02": 2, "subj05": 5, "subj07": 7}


def source_provenance(model_source):
    source = Path(model_source)
    dependency = source.with_name("perceiver.py")
    if not dependency.is_file():
        raise ValueError(f"BrainX dependency missing: {dependency}")
    return {source.name: hashlib.sha256(source.read_bytes()).hexdigest(),
            dependency.name: hashlib.sha256(dependency.read_bytes()).hexdigest()}


def restricted_scheduler_getattr(owner, name):
    """Replacement for the one historical pickle ``getattr`` invocation."""
    from torch.optim.lr_scheduler import OneCycleLR
    if owner is not OneCycleLR or name != "_annealing_cos":
        raise ValueError("safe checkpoint loader refused serialized getattr")
    return OneCycleLR._annealing_cos


def encode_subject_brain(model, brain, subject_id: str):
    """Route an unmodified raw brain tensor through its BrainX subject branch."""
    import torch
    if subject_id not in SUBJECT_NUMBERS:
        raise ValueError(f"unsupported BrainX subject {subject_id!r}")
    if not isinstance(brain, torch.Tensor) or brain.ndim != 2:
        raise ValueError("brain must have shape [batch, voxels]")
    number = SUBJECT_NUMBERS[subject_id]
    expected = model.num_voxels[number]
    if brain.shape[1] != expected:
        raise ValueError(f"{subject_id} expects {expected} voxels; got {brain.shape[1]}")
    return model(brain, modal=f"fmri{number}")


def _load_model_module(source_path: Path):
    specification = importlib.util.spec_from_file_location("brain_npp_umbrae_model", source_path)
    if specification is None or specification.loader is None:
        raise ValueError(f"cannot import BrainX source {source_path}")
    module = importlib.util.module_from_spec(specification)
    sys.path.insert(0, str(source_path.parent))
    try:
        specification.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def load_brainx(checkpoint_path, *, model_source="/root/UMBRAE/umbrae/model.py",
                device="cpu", expected_checkpoint_sha256=None):
    """Load the released four-subject BrainX without unrestricted pickle."""
    import torch
    checkpoint_path, source_path = Path(checkpoint_path), Path(model_source)
    actual_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if expected_checkpoint_sha256 is not None and actual_hash != expected_checkpoint_sha256:
        raise ValueError("BrainX checkpoint sha256 mismatch")
    module = _load_model_module(source_path)
    model = module.BrainX(hidden_dim=1024, out_dim=1024, num_latents=256)
    from torch.optim.lr_scheduler import OneCycleLR
    safe_alias = (restricted_scheduler_getattr, "builtins.getattr")
    with torch.serialization.safe_globals([safe_alias, OneCycleLR]):
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    state = saved.get("model_state_dict") if isinstance(saved, dict) else None
    if not isinstance(state, dict):
        raise ValueError("BrainX checkpoint lacks model_state_dict")
    if len(state) != 90:
        raise ValueError(f"BrainX checkpoint expected 90 tensors; got {len(state)}")
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval().to(device)
    model.checkpoint_provenance = {"path": str(checkpoint_path), "sha256": actual_hash,
                                   "architecture": "UMBRAE BrainX", "strict_tensor_count": len(state),
                                   "source_sha256": source_provenance(source_path)}
    return model
