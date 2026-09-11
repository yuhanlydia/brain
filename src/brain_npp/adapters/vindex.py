"""VINDEX/LLaVA adapter primitives for the bounded real P1 run.

This module keeps transformers, PEFT, and bitsandbytes out of package import
time.  The external factory owns construction of those optional components.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from brain_npp.calibration import GaussianEncodingLikelihood
from brain_npp.protocols import Rollout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def causal_rollout_logits(logits: torch.Tensor, prompt_length: int, rollout_length: int) -> torch.Tensor:
    """Select next-token logits aligned with a fixed generated continuation."""
    if logits.ndim != 3 or prompt_length < 1 or rollout_length < 1:
        raise ValueError("logits must be [batch,time,vocab] with positive lengths")
    result = logits[:, prompt_length - 1 : prompt_length - 1 + rollout_length]
    if result.shape[1] != rollout_length:
        raise ValueError("model logits are too short for the requested rollout")
    return result


def rollout_mask(token_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """Include the first EOS and mask only tokens following it."""
    if token_ids.ndim != 2:
        raise ValueError("token_ids must be rank 2")
    eos = token_ids.eq(eos_token_id)
    return eos.cumsum(dim=-1).le(1)


def frozen_snapshot(module: torch.nn.Module) -> torch.nn.Module:
    snapshot = copy.deepcopy(module)
    snapshot.requires_grad_(False)
    snapshot.eval()
    return snapshot


@dataclass
class P1Artifacts:
    arrays: dict[str, np.ndarray]
    candidate_ids: list[str]
    fold_ids: np.ndarray
    log_likelihood: np.ndarray
    hashes: dict[str, str]


def save_training_state(path, named_parameters, optimizer, *, optimizer_steps, input_hash, extra=None):
    """Save live trainables, optimizer tensors, and all local RNG streams."""
    state = {"optimizer_steps": int(optimizer_steps), "input_hash": input_hash,
        "trainable": {name: value.detach().cpu().clone() for name, value in named_parameters.items()},
        "optimizer": optimizer.state_dict(), "rng": {"python": random.getstate(),
        "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, "extra": extra}
    torch.save(state, path)


def load_training_state(path, named_parameters, optimizer, *, expected_input_hash):
    """Restore a checkpoint into fresh live parameters/optimizer and RNG streams."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("input_hash") != expected_input_hash:
        raise ValueError("checkpoint input hash does not match current inputs")
    if set(state["trainable"]) != set(named_parameters):
        raise ValueError("checkpoint trainable parameter names do not match")
    with torch.no_grad():
        for name, parameter in named_parameters.items(): parameter.copy_(state["trainable"][name].to(parameter.device))
    optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["rng"]["python"]); np.random.set_state(state["rng"]["numpy"]); torch.set_rng_state(state["rng"]["torch"])
    if torch.cuda.is_available() and state["rng"]["cuda"]: torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    return {"optimizer_steps": state["optimizer_steps"], "extra": state.get("extra")}


def load_validated_p1_artifacts(features_path: str | Path, calibration_dir: str | Path) -> P1Artifacts:
    """Validate hashes/IDs/fold isolation and recompute all candidate scores."""
    features_path, calibration_dir = Path(features_path), Path(calibration_dir)
    feature_provenance = json.loads(features_path.with_name("features.provenance.json").read_text())
    feature_hash = _sha256(features_path)
    if feature_provenance.get("artifact_sha256") != feature_hash:
        raise ValueError("features artifact_sha256 mismatch")
    provenance = json.loads((calibration_dir / "provenance.json").read_text())
    if provenance.get("input_features_sha256") != feature_hash:
        raise ValueError("calibration input_features_sha256 mismatch")
    with np.load(features_path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    with np.load(calibration_dir / "scores.npz", allow_pickle=False) as scores:
        trial_ids = scores["trial_ids"]
        candidate_ids = [str(value) for value in scores["candidate_ids"]]
        fold_ids = scores["fold_ids"].astype(np.int64)
    if not np.array_equal(trial_ids, arrays["trial_ids"]):
        raise ValueError("calibration trial_ids do not match features")
    if candidate_ids != [str(value) for value in provenance.get("candidate_ids", [])]:
        raise ValueError("candidate_ids do not match calibration provenance")
    image_lookup = {str(value): index for index, value in enumerate(arrays["image_ids"])}
    try:
        candidate_features = arrays["image_pooled"][[image_lookup[value] for value in candidate_ids]]
    except KeyError as error:
        raise ValueError(f"candidate image missing from features: {error.args[0]}") from error
    recomputed = np.empty((len(trial_ids), len(candidate_ids)), dtype=np.float64)
    hashes = {"features": feature_hash, "scores": _sha256(calibration_dir / "scores.npz")}
    for fold in np.unique(fold_ids):
        path = calibration_dir / f"fold{int(fold)}.npz"
        actual_hash = _sha256(path)
        expected = provenance.get("folds", {}).get(str(int(fold)), {}).get("sha256")
        if actual_hash != expected:
            raise ValueError(f"fold{int(fold)} sha256 mismatch")
        model = GaussianEncodingLikelihood.load(path)
        selected = fold_ids == fold
        held_out_images = set(str(value) for value in arrays["trial_image_ids"][selected])
        if held_out_images & set(model.provenance.get("train_image_ids", [])):
            raise ValueError(f"fold{int(fold)} train_image_ids contains a selected trial image")
        recomputed[selected] = model.score(arrays["brains"][selected], candidate_features)
        hashes[f"fold{int(fold)}"] = actual_hash
    if not np.isfinite(recomputed).all():
        raise ValueError("recomputed candidate likelihood is nonfinite")
    return P1Artifacts(arrays, candidate_ids, fold_ids, recomputed, hashes)


class VindexP1Adapter(torch.nn.Module):
    """Student brain-prefix model and frozen image-prefix teacher on one rollout."""

    def __init__(self, model, tokenizer, projector, teacher_projector, pre_ids, post_ids, device):
        super().__init__()
        self.model, self.tokenizer = model, tokenizer
        self.projector, self.teacher_projector = projector, teacher_projector
        self.register_buffer("pre_ids", pre_ids.to(device), persistent=False)
        self.register_buffer("post_ids", post_ids.to(device), persistent=False)
        self.device = torch.device(device)

    def _prefix(self, features, projector):
        embeddings = self.model.get_input_embeddings()
        pre = embeddings(self.pre_ids).expand(features.shape[0], -1, -1)
        post = embeddings(self.post_ids).expand(features.shape[0], -1, -1)
        visual = projector(features.to(self.device)).to(embeddings.weight.dtype)
        return torch.cat((pre, visual, post), dim=1)

    def generate_student(self, batch, generation_config):
        self.model.eval()
        with torch.no_grad():
            prefix = self._prefix(batch["brain_features"], self.projector)
            kwargs = dict(generation_config)
            kwargs.setdefault("do_sample", False)
            kwargs.setdefault("pad_token_id", self.tokenizer.eos_token_id)
            ids = self.model.generate(inputs_embeds=prefix,
                attention_mask=torch.ones(prefix.shape[:2], dtype=torch.long, device=self.device), **kwargs)
        return Rollout(ids, rollout_mask(ids, self.tokenizer.eos_token_id))

    def _logits(self, features, projector, rollout, disable_adapters=False):
        context = self.model.disable_adapter() if disable_adapters else nullcontext()
        with context:
            prefix = self._prefix(features, projector)
            tokens = self.model.get_input_embeddings()(rollout.token_ids)
            logits = self.model(inputs_embeds=torch.cat((prefix, tokens), 1), use_cache=False).logits
        return causal_rollout_logits(logits, prefix.shape[1], rollout.token_ids.shape[1])

    def student_logits(self, batch, rollout):
        self.model.train()
        return self._logits(batch["brain_features"], self.projector, rollout)

    def teacher_candidate_logits(self, batch, rollout):
        def stream():
            with torch.no_grad():
                for index in range(batch["candidate_image_features"].shape[1]):
                    yield self._logits(batch["candidate_image_features"][:, index],
                                       self.teacher_projector, rollout, True)
        return stream()

    def brain_candidate_scores(self, batch):
        return batch["candidate_scores"].to(self.device)

    def student_supervised_logits(self, batch, target_ids, target_mask):
        return self._logits(batch["brain_features"], self.projector,
                            Rollout(target_ids, target_mask))
