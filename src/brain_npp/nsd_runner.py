"""Resumable protocol primitives for the matched real-NSD method matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .baselines import (credit_image_contrastive_loss, dapd_loss,
                        full_vocabulary_teacher_target, masked_cross_entropy,
                        vad_budgeted_asvc_loss)
from .objective import build_npp_target, forward_kl_loss
from .posterior import build_neural_posterior


class Method(str, Enum):
    CE = "ce"
    EXACT = "exact-image-opsd"
    MAP = "map-image-opsd"
    UNIFORM = "uniform-mixture"
    POSTERIOR = "plain-posterior-mixture"
    CREDIT = "credit-style-image-contrastive"
    DAPD = "dapd-brain-text-reference"
    VAD = "vad-style-full-image"
    NPP = "npp-opsd"
    NPP_GEOMETRIC = "npp-geometric"
    NPP_IG = "npp-information-gain"
    NPP_SQRT_IG = "npp-sqrt-information-gain"


@dataclass(frozen=True)
class ViewPlan:
    rollouts: int
    live_views: tuple[str, ...]
    teacher_views: tuple[str, ...]
    anchor_views: tuple[str, ...] = ()

    @property
    def sequence_views(self) -> int:
        return len(self.live_views) + len(self.teacher_views) + len(self.anchor_views)


def method_view_plan(method: Method | str) -> ViewPlan:
    method = Method(method)
    if method is Method.DAPD:
        return ViewPlan(1, ("rollout_none", "reference_reference", "rollout_rollout", "reference_none"), (),
                        ("entangled_rollout", "inference_reference", "privileged_rollout",
                         "entangled_reference", "inference_rollout", "privileged_reference"))
    if method is Method.VAD:
        return ViewPlan(1, ("rollout",), ("clear", "degraded"))
    if method is Method.CREDIT:
        return ViewPlan(1, ("rollout",), ("positive_image", "mismatched_image"))
    if method is Method.CE:
        return ViewPlan(0, ("reference",), ())
    if method in {Method.EXACT, Method.MAP, Method.UNIFORM, Method.POSTERIOR}:
        return ViewPlan(1, ("rollout",), ("candidate_images",))
    return ViewPlan(1, ("rollout",), ("candidate_images",))


def method_loss(method: Method | str, *, student_logits: torch.Tensor, mask: torch.Tensor,
                gold_tokens: torch.Tensor, teacher_probabilities: torch.Tensor,
                posterior_weights: torch.Tensor, prior_weights: torch.Tensor,
                exact_index: int = 0, **views) -> torch.Tensor:
    """Dispatch all preregistered objectives over fixed, caller-constructed views."""
    method = Method(method)
    if method is Method.CE:
        return masked_cross_entropy(student_logits, gold_tokens, mask)
    if method in {Method.EXACT, Method.MAP, Method.UNIFORM, Method.POSTERIOR}:
        mode = {Method.EXACT: "exact", Method.MAP: "map", Method.UNIFORM: "uniform",
                Method.POSTERIOR: "posterior"}[method]
        weights = posterior_weights if mode in {"map", "posterior"} else None
        target = full_vocabulary_teacher_target(teacher_probabilities, mode=mode,
                                                weights=weights, exact_index=exact_index)
        return forward_kl_loss(target.clamp_min(torch.finfo(target.dtype).tiny).log(), student_logits, mask).loss
    if method is Method.CREDIT:
        return credit_image_contrastive_loss(student_logits, views["positive_teacher_probabilities"],
                                             views["negative_teacher_probabilities"], mask)
    if method is Method.VAD:
        return vad_budgeted_asvc_loss(student_logits, views["clear_teacher_probabilities"],
                                      views["degraded_teacher_probabilities"], mask,
                                      sampled_log_probs=views.get("sampled_log_probs"),
                                      old_log_probs=views.get("old_log_probs")).loss
    if method is Method.DAPD:
        return dapd_loss(views["live_logits"], views["anchor_logits"], views["completion_masks"]).loss
    posterior = build_neural_posterior(views["candidate_scores"], views["log_prior"])
    pooling = "geometric" if method is Method.NPP_GEOMETRIC else "arithmetic"
    strength = "information_gain" if method is Method.NPP_IG else (
        "sqrt_information_gain" if method is Method.NPP_SQRT_IG else "constant")
    target = build_npp_target(teacher_probabilities.clamp_min(torch.finfo(teacher_probabilities.dtype).tiny).log(),
                              student_logits.detach(), posterior, pooling=pooling, strength_mode=strength)
    return forward_kl_loss(target.log_probabilities, student_logits, mask).loss


def accumulation_windows(example_count: int, accumulation_steps: int, *, max_updates: int) -> Iterable[tuple[int, int]]:
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in (example_count, accumulation_steps, max_updates)):
        raise ValueError("counts must be positive integers")
    for update, start in enumerate(range(0, example_count, accumulation_steps)):
        if update >= max_updates:
            break
        yield start, min(start + accumulation_steps, example_count)


def select_question_ids(available: Mapping[str, Sequence[int]], *, seed: int) -> dict[str, int]:
    """Select one training QA per trial without consulting method, control, or answer."""
    result = {}
    for trial_id, question_ids in available.items():
        if not question_ids: raise ValueError(f"trial {trial_id} has no questions")
        digest=hashlib.sha256(f"qa-v1:{seed}:{trial_id}".encode()).digest()
        result[trial_id]=int(question_ids[int.from_bytes(digest[:8],"big") % len(question_ids)])
    return result


def select_image_group_fraction(trial_images: Mapping[str, str], *, fraction: float, seed: int) -> tuple[str, ...]:
    """Select a deterministic image-group population and retain every group trial."""
    if not 0 < fraction <= 1 or not trial_images: raise ValueError("fraction and trial_images are invalid")
    images=sorted(set(trial_images.values())); random.Random(seed).shuffle(images)
    count=max(1,round(len(images)*fraction)); chosen=set(images[:count])
    return tuple(trial_id for trial_id,image_id in trial_images.items() if image_id in chosen)


def run_update_window(model: torch.nn.Module, optimizer: torch.optim.Optimizer, examples: Sequence[Any],
                      loss_fn, *, gradient_clip: float | None) -> dict[str, float | int]:
    """Apply one optimizer update, scaling even a partial final window as a mean."""
    if not examples: raise ValueError("update window must contain examples")
    if torch.cuda.is_available(): torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter();optimizer.zero_grad(set_to_none=True); losses = []
    for example in examples:
        loss = loss_fn(model, example)
        if loss.ndim != 0 or not torch.isfinite(loss): raise ValueError("micro loss must be finite scalar")
        (loss / len(examples)).backward(); losses.append(float(loss.detach()))
    parameters = [value for value in model.parameters() if value.requires_grad]
    if not parameters or not any(value.grad is not None for value in parameters):
        raise ValueError("update produced no trainable gradients")
    squared = sum(float(value.grad.detach().float().square().sum()) for value in parameters if value.grad is not None)
    if not math.isfinite(squared): raise ValueError("update produced nonfinite gradients")
    if gradient_clip is not None: torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)
    optimizer.step()
    if torch.cuda.is_available():torch.cuda.synchronize()
    return {"loss": sum(losses)/len(losses), "micro_examples": len(examples), "gradient_norm": math.sqrt(squared),
            "update_seconds":time.perf_counter()-started,"peak_cuda_bytes":torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class MatchedSchedule:
    schema_version: int
    seed: int
    trial_ids: tuple[str, ...]
    accumulation_steps: int
    max_updates: int
    schedule_hash: str

    @classmethod
    def create(cls, trial_ids: Sequence[str], *, seed: int, max_examples: int,
               accumulation_steps: int, max_updates: int) -> "MatchedSchedule":
        if len(set(trial_ids)) != len(trial_ids):
            raise ValueError("trial_ids must be unique")
        ordered = list(trial_ids)
        random.Random(seed).shuffle(ordered)
        selected = tuple(ordered[:max_examples])
        base = {"schema_version": 1, "seed": seed, "trial_ids": list(selected),
                "accumulation_steps": accumulation_steps, "max_updates": max_updates}
        return cls(schema_version=1, seed=seed, trial_ids=selected,
                   accumulation_steps=accumulation_steps, max_updates=max_updates,
                   schedule_hash=_canonical_hash(base))

    @classmethod
    def ordered(cls, trial_ids: Sequence[str], *, seed: int, accumulation_steps: int,
                max_updates: int) -> "MatchedSchedule":
        if len(set(trial_ids)) != len(trial_ids) or not trial_ids: raise ValueError("ordered trial_ids must be nonempty and unique")
        selected=tuple(trial_ids);base={"schema_version":1,"seed":seed,"trial_ids":list(selected),"accumulation_steps":accumulation_steps,"max_updates":max_updates}
        return cls(1,seed,selected,accumulation_steps,max_updates,_canonical_hash(base))

    def save(self, path: str | Path) -> None:
        _atomic_json(Path(path), {**asdict(self), "trial_ids": list(self.trial_ids)})

    @classmethod
    def load(cls, path: str | Path) -> "MatchedSchedule":
        value = json.loads(Path(path).read_text())
        supplied = value.pop("schedule_hash", None)
        if supplied != _canonical_hash(value):
            raise ValueError("schedule hash mismatch")
        value["trial_ids"] = tuple(value["trial_ids"])
        return cls(**value, schedule_hash=supplied)


def credit_negative_mapping(image_ids: Sequence[str], *, seed: int) -> dict[int, int]:
    """Create a saved index derangement whose paired image identity always differs."""
    if len(set(image_ids)) < 2:
        raise ValueError("CREDIT negatives require at least two image identities")
    groups: dict[str, list[int]] = {}
    for index, identity in enumerate(image_ids):
        groups.setdefault(identity, []).append(index)
    largest = max(map(len, groups.values()))
    if largest * 2 > len(image_ids):
        raise ValueError("image identities cannot be deranged without a self-identity pair")
    rng = random.Random(seed)
    source = list(range(len(image_ids)))
    for _ in range(10_000):
        target = source.copy(); rng.shuffle(target)
        if all(image_ids[i] != image_ids[j] for i, j in zip(source, target)):
            return dict(zip(source, target))
    raise ValueError("image identities could not be deranged")


def degrade_vad_image(image: torch.Tensor) -> torch.Tensor:
    """Released full-image degradation: RGB, 0.10 bilinear down, nearest up."""
    if image.ndim not in (3, 4) or image.shape[-3] != 3:
        raise ValueError("VAD degradation requires CHW or BCHW RGB input")
    batched = image.unsqueeze(0) if image.ndim == 3 else image
    height, width = batched.shape[-2:]
    reduced = (max(1, round(height * .10)), max(1, round(width * .10)))
    result = F.interpolate(F.interpolate(batched, reduced, mode="bilinear", align_corners=False),
                           (height, width), mode="nearest")
    return result[0] if image.ndim == 3 else result


@dataclass
class ComputeAccounting:
    student_tokens: int = 0
    teacher_tokens: int = 0
    generated_tokens: int = 0
    physical_forwards: int = 0
    sequence_views: int = 0

    def add(self, **counts: int) -> None:
        for name, value in counts.items():
            if name not in asdict(self) or isinstance(value, bool) or value < 0:
                raise ValueError(f"invalid compute count {name}")
            setattr(self, name, getattr(self, name) + int(value))


@dataclass(frozen=True)
class DAPDSnapshot:
    trainable: dict[str, torch.Tensor]
    update_index: int

    @classmethod
    def capture(cls, named_parameters: Mapping[str, torch.Tensor], *, update_index: int) -> "DAPDSnapshot":
        return cls({name: value.detach().cpu().clone() for name, value in named_parameters.items()}, update_index)

    def apply(self, named_parameters: Mapping[str, torch.Tensor]) -> None:
        if set(named_parameters) != set(self.trainable):
            raise ValueError("DAPD snapshot trainable names mismatch")
        with torch.no_grad():
            for name, parameter in named_parameters.items():
                parameter.copy_(self.trainable[name].to(parameter.device, parameter.dtype))


@dataclass(frozen=True)
class RestoredRunState:
    cursor: int
    optimizer_steps: int
    snapshot: DAPDSnapshot | None
    accounting: ComputeAccounting
    extra: Any


class AtomicRunState:
    def __init__(self, path: str | Path, *, identity: Mapping[str, Any]):
        self.path, self.identity = Path(path), dict(identity)
        self.identity_hash = _canonical_hash(self.identity)

    def save(self, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer, cursor: int,
             optimizer_steps: int, snapshot: DAPDSnapshot | None, accounting: ComputeAccounting,
             extra: Any = None) -> None:
        trainable = {name: value.detach().cpu().clone() for name, value in model.named_parameters() if value.requires_grad}
        if not trainable: raise ValueError("checkpoint model has no named trainables")
        state = {"schema_version": 1, "identity": self.identity, "identity_hash": self.identity_hash,
                 "trainable": trainable, "optimizer": optimizer.state_dict(), "cursor": cursor,
                 "optimizer_steps": optimizer_steps, "snapshot": snapshot,
                 "accounting": asdict(accounting), "extra": extra,
                 "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                         "torch": torch.get_rng_state(),
                         "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        torch.save(state, temporary); os.replace(temporary, self.path)

    def restore(self, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> RestoredRunState:
        state = torch.load(self.path, map_location="cpu", weights_only=False)
        if state.get("identity_hash") != self.identity_hash or state.get("identity") != self.identity:
            raise ValueError("checkpoint identity mismatch")
        named = {name: value for name, value in model.named_parameters() if value.requires_grad}
        if set(named) != set(state["trainable"]): raise ValueError("checkpoint trainable names mismatch")
        with torch.no_grad():
            for name, parameter in named.items(): parameter.copy_(state["trainable"][name].to(parameter.device, parameter.dtype))
        optimizer.load_state_dict(state["optimizer"])
        rng = state["rng"]; random.setstate(rng["python"]); np.random.set_state(rng["numpy"]); torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng["cuda"]:
            torch.cuda.set_rng_state_all(rng["cuda"])
        return RestoredRunState(state["cursor"], state["optimizer_steps"], state["snapshot"],
                                ComputeAccounting(**state["accounting"]), state.get("extra"))


class MatrixExperiment:
    """Backend-neutral matched schedule executor; checkpoints only update boundaries."""
    def __init__(self, backend, schedule: MatchedSchedule, method: Method | str, output_dir: str | Path,
                 *, identity: Mapping[str, Any], gradient_clip: float | None = None):
        self.backend, self.schedule, self.method = backend, schedule, Method(method)
        self.output_dir = Path(output_dir); self.gradient_clip = gradient_clip
        bound = {**dict(identity), "schedule_hash": schedule.schedule_hash, "method": self.method.value}
        self.state = AtomicRunState(self.output_dir / "checkpoint.pt", identity=bound)

    def train(self) -> dict:
        model, optimizer = self.backend.model, self.backend.optimizer
        cursor = optimizer_steps = 0; accounting = getattr(self.backend, "accounting", ComputeAccounting()); history = []; snapshot = None
        was_resumed = self.state.path.exists()
        if was_resumed:
            restored = self.state.restore(model=model, optimizer=optimizer)
            cursor, optimizer_steps, accounting, snapshot = (restored.cursor, restored.optimizer_steps,
                                                               restored.accounting, restored.snapshot)
            history = list((restored.extra or {}).get("history", ()))
            self.backend.accounting=accounting
            contexts = getattr(self.backend, "contexts", None)
            if snapshot is not None and contexts is not None: contexts.snapshot_state = snapshot
        contexts = getattr(self.backend, "contexts", None)
        if self.method is Method.DAPD and contexts is not None and contexts.snapshot_state is None:
            snapshot = contexts.capture_snapshot(update_index=optimizer_steps)
        for start, end in accumulation_windows(len(self.schedule.trial_ids), self.schedule.accumulation_steps,
                                               max_updates=self.schedule.max_updates):
            if end <= cursor: continue
            if start < cursor: raise ValueError("checkpoint cursor is not an optimizer boundary")
            ids = self.schedule.trial_ids[start:end]
            diagnostic = run_update_window(model, optimizer, ids,
                lambda _model, trial_id: self.backend.loss(trial_id, self.method), gradient_clip=self.gradient_clip)
            cursor, optimizer_steps = end, optimizer_steps + 1; history.append(diagnostic)
            if self.method is Method.DAPD and contexts is not None:
                contexts.refresh_snapshot_if_due(optimizer_steps, interval=100); snapshot = contexts.snapshot_state
            self.state.save(model=model, optimizer=optimizer, cursor=cursor, optimizer_steps=optimizer_steps,
                            snapshot=snapshot, accounting=accounting, extra={"history": history})
        return {"method": self.method.value, "cursor": cursor, "optimizer_steps": optimizer_steps,
                "history": history, "resumed": was_resumed, "accounting": asdict(accounting)}


class PredictionJournal:
    """Append-only, fsynced per-example predictions with identity sidecar."""
    def __init__(self, path: str | Path, *, identity: Mapping[str, Any]):
        self.path = Path(path); self.identity = dict(identity)
        self.meta = self.path.with_suffix(self.path.suffix + ".identity.json")
        identity_payload = {"schema_version": 1, "identity": self.identity, "identity_hash": _canonical_hash(self.identity)}
        if self.meta.exists() and json.loads(self.meta.read_text()) != identity_payload:
            raise ValueError("prediction journal identity mismatch")
        if not self.meta.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True); _atomic_json(self.meta, identity_payload)
        self.rows: dict[str, dict] = {}
        if self.path.exists():
            raw=self.path.read_bytes(); lines=raw.splitlines(keepends=True); offset=0
            for index,line in enumerate(lines):
              try: row=json.loads(line)
              except (json.JSONDecodeError, UnicodeDecodeError) as error:
                if index==len(lines)-1 and not line.endswith(b"\n"):
                    with self.path.open("r+b") as stream: stream.truncate(offset)
                    break
                raise ValueError("malformed prediction journal record") from error
              key = str(row["example_id"])
              offset += len(line)
              if key in self.rows and self.rows[key] != row:
                  raise ValueError("conflicting duplicate prediction in journal")
              self.rows[key] = row
            if raw and not raw.endswith(b"\n") and offset==len(raw):
                with self.path.open("ab") as stream: stream.write(b"\n");stream.flush();os.fsync(stream.fileno())

    @property
    def completed_ids(self) -> set[str]:
        return set(self.rows)

    def append(self, row: Mapping[str, Any]) -> bool:
        value = dict(row); key = str(value.get("example_id", ""))
        required = {"example_id", "token_ids", "text", "source_image_id"}
        if not key or not required <= value.keys():
            raise ValueError("prediction row is missing required provenance")
        if key in self.rows:
            if self.rows[key] != value:
                raise ValueError("conflicting prediction for completed example")
            return False
        encoded = json.dumps(value, sort_keys=True, allow_nan=False) + "\n"
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
        self.rows[key] = value
        return True


def validate_brain_array(path: str | Path, *, expected_sha256: str) -> torch.Tensor:
    """Load a brain array and bind the emitted numeric bytes used by preparation."""
    value = np.load(path, allow_pickle=False)
    digest = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError("brain array sha256 mismatch")
    if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
        raise ValueError("brain array must be finite floating point")
    return torch.from_numpy(np.asarray(value))


@dataclass(frozen=True)
class ValidatedControl:
    control: str
    mapping: dict[str, str]
    unmatched_ids: tuple[str, ...]
    selected_unmatched: tuple[str, ...]
    artifact_sha256: str


def validate_control_artifact(path: str | Path, *, expected_control: str, manifest_sha256: str,
                              records: Mapping[str, tuple[str, str, int]],
                              selected_source_ids: Sequence[str]) -> ValidatedControl:
    """Validate a full-population mapping before selecting evaluation sources."""
    path = Path(path); payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1 or payload.get("control") != expected_control:
        raise ValueError("control artifact kind mismatch")
    if payload.get("manifest_sha256") != manifest_sha256:
        raise ValueError("control artifact manifest hash mismatch")
    mapping, unmatched = payload.get("mapping"), tuple(payload.get("unmatched_ids", ()))
    if not isinstance(mapping, dict) or set(mapping) & set(unmatched):
        raise ValueError("control mapping and unmatched IDs are invalid")
    for source, target in mapping.items():
        if source not in records or target not in records or source == target:
            raise ValueError("control mapping references invalid trial IDs")
        ss, si, sr = records[source]; ts, ti, tr = records[target]
        if expected_control == "shuffled" and (ss != ts or si == ti):
            raise ValueError("shuffled mapping must preserve subject and change image")
        if expected_control == "wrong-subject" and (ss == ts or si != ti or sr != tr):
            raise ValueError("wrong-subject mapping must change subject and preserve image/repeat")
    missing = [key for key in selected_source_ids if key not in mapping and key not in unmatched]
    if missing:
        raise ValueError(f"selected source absent from full control mapping: {missing[0]}")
    return ValidatedControl(expected_control, dict(mapping), unmatched,
                            tuple(key for key in selected_source_ids if key in unmatched),
                            hashlib.sha256(path.read_bytes()).hexdigest())


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)
