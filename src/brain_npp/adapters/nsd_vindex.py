"""VINDEX matrix adapter utilities with explicit live/base/snapshot contexts."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, ContextManager, Mapping, Sequence

import torch

from brain_npp.nsd_runner import DAPDSnapshot


@dataclass(frozen=True)
class CompletionBatch:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    logit_indices: torch.Tensor
    completion_mask: torch.Tensor


def causal_completion_batch(prefixes: Sequence[torch.Tensor], completions: Sequence[torch.Tensor],
                            embedding: torch.nn.Module, *, pad_token_id: int) -> CompletionBatch:
    """Pad variable prompts/completions while retaining exact causal logit positions."""
    if len(prefixes) != len(completions) or not prefixes:
        raise ValueError("prefixes and completions must be matching nonempty sequences")
    embedded, positions, masks = [], [], []
    for prefix, tokens in zip(prefixes, completions):
        if prefix.ndim != 2 or tokens.ndim != 1 or tokens.numel() < 1:
            raise ValueError("each prefix is [prompt,hidden] and completion is nonempty [tokens]")
        values = torch.cat((prefix, embedding(tokens)), dim=0); embedded.append(values)
        positions.append(torch.arange(tokens.numel(), device=tokens.device) + prefix.shape[0] - 1)
    max_sequence = max(x.shape[0] for x in embedded); max_completion = max(x.numel() for x in completions)
    hidden = embedded[0].shape[-1]; device, dtype = embedded[0].device, embedded[0].dtype
    inputs = torch.zeros((len(embedded), max_sequence, hidden), device=device, dtype=dtype)
    attention = torch.zeros((len(embedded), max_sequence), device=device, dtype=torch.long)
    indices = torch.full((len(embedded), max_completion), -1, device=device, dtype=torch.long)
    completion_mask = torch.zeros_like(indices, dtype=torch.bool)
    for row, (values, position) in enumerate(zip(embedded, positions)):
        inputs[row, :values.shape[0]] = values; attention[row, :values.shape[0]] = 1
        indices[row, :position.numel()] = position; completion_mask[row, :position.numel()] = True
    return CompletionBatch(inputs, attention, indices, completion_mask)


def aligned_completion_logits(logits: torch.Tensor, batch: CompletionBatch) -> torch.Tensor:
    if logits.shape[:2] != batch.attention_mask.shape:
        raise ValueError("model logits do not match completion batch")
    safe = batch.logit_indices.clamp_min(0)
    gathered = logits.gather(1, safe[..., None].expand(-1, -1, logits.shape[-1]))
    return gathered.masked_fill(~batch.completion_mask[..., None], 0)


class ParameterContexts:
    """One base model with reversible projector/LoRA live, base and snapshot views."""
    def __init__(self, named_trainables: Mapping[str, torch.Tensor], *, projector_names: set[str],
                 adapter_context: Callable[[], ContextManager]):
        self.named = dict(named_trainables); self.projector_names = set(projector_names)
        if not self.projector_names <= self.named.keys():
            raise ValueError("projector names must be trainable parameter names")
        self.adapter_context = adapter_context
        self.original_projector = {name: self.named[name].detach().cpu().clone() for name in self.projector_names}
        self.snapshot_state: DAPDSnapshot | None = None

    def capture_snapshot(self, *, update_index: int) -> DAPDSnapshot:
        self.snapshot_state = DAPDSnapshot.capture(self.named, update_index=update_index)
        return self.snapshot_state

    @contextmanager
    def _temporary(self, values: Mapping[str, torch.Tensor]):
        current = {name: self.named[name].detach().cpu().clone() for name in values}
        try:
            with torch.no_grad():
                for name, value in values.items(): self.named[name].copy_(value.to(self.named[name].device, self.named[name].dtype))
            yield
        finally:
            with torch.no_grad():
                for name, value in current.items(): self.named[name].copy_(value.to(self.named[name].device, self.named[name].dtype))

    @contextmanager
    def base(self):
        with self.adapter_context():
            with self._temporary(self.original_projector):
                yield

    @contextmanager
    def snapshot(self):
        if self.snapshot_state is None:
            raise ValueError("DAPD snapshot has not been captured")
        with self._temporary(self.snapshot_state.trainable):
            yield self.snapshot_state.update_index

    def refresh_snapshot_if_due(self, optimizer_steps: int, *, interval: int = 100) -> bool:
        if optimizer_steps < 0 or interval < 1: raise ValueError("snapshot indices are invalid")
        if self.snapshot_state is None or optimizer_steps >= self.snapshot_state.update_index + interval:
            self.capture_snapshot(update_index=optimizer_steps); return True
        return False


class PatchCache:
    """Bounded authenticated on-disk cache for CLIP layer-2 patch tensors."""
    def __init__(self, root: str | Path, *, max_bytes: int, model_hash: str, preprocessing_hash: str):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True); self.max_bytes=max_bytes
        self.model_hash=model_hash; self.preprocessing_hash=preprocessing_hash
        if max_bytes < 1: raise ValueError("patch cache max_bytes must be positive")

    def _paths(self,image_id,view):
        key=hashlib.sha256(f"{image_id}:{view}:{self.model_hash}:{self.preprocessing_hash}".encode()).hexdigest()
        return self.root/(key+".pt"),self.root/(key+".json")

    @staticmethod
    def _tensor_hash(value):
        return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()

    def get_or_compute(self,image_id: str,view: str,image_hash: str,compute):
        if view not in {"clear","degraded"}: raise ValueError("patch cache view must be clear or degraded")
        tensor_path,meta_path=self._paths(image_id,view)
        if tensor_path.exists() or meta_path.exists():
          try:
            if not tensor_path.exists() or not meta_path.exists(): raise RuntimeError("incomplete")
            meta=json.loads(meta_path.read_text())
            if meta.get("image_hash") != image_hash: raise ValueError("patch cache image hash mismatch")
            if meta.get("model_hash") != self.model_hash or meta.get("preprocessing_hash") != self.preprocessing_hash:
                raise ValueError("patch cache model/preprocessing hash mismatch")
            value=torch.load(tensor_path,map_location="cpu",weights_only=True)
            if (list(value.shape)!=meta.get("shape") or str(value.dtype)!=meta.get("dtype") or
                not torch.isfinite(value).all() or self._tensor_hash(value)!=meta.get("tensor_sha256")):
                raise RuntimeError("corrupt tensor")
            os.utime(tensor_path); os.utime(meta_path)
            return value
          except ValueError:
            raise
          except Exception:
            tensor_path.unlink(missing_ok=True); meta_path.unlink(missing_ok=True)
        value=compute().detach().cpu()
        if value.ndim != 3 or value.shape[1] < 1 or value.shape[2] < 1 or not torch.isfinite(value).all():
            raise ValueError("CLIP patch cache expects finite [batch,patch,hidden]")
        temporary=tensor_path.with_suffix(".tmp"); torch.save(value,temporary); os.replace(temporary,tensor_path)
        meta={"schema_version":1,"image_id":image_id,"view":view,"image_hash":image_hash,
              "model_hash":self.model_hash,"preprocessing_hash":self.preprocessing_hash,
              "shape":list(value.shape),"dtype":str(value.dtype),"tensor_sha256":self._tensor_hash(value)}
        temp_meta=meta_path.with_suffix(".tmp"); temp_meta.write_text(json.dumps(meta,sort_keys=True)+"\n"); os.replace(temp_meta,meta_path)
        while sum(p.stat().st_size for p in self.root.glob("*.*") if p.suffix in {".pt",".json"}) > self.max_bytes:
            candidates=sorted((p for p in self.root.glob("*.pt") if p != tensor_path),key=lambda p:p.stat().st_mtime)
            if not candidates:
                tensor_path.unlink(missing_ok=True);meta_path.unlink(missing_ok=True);break
            victim=candidates[0]; victim.with_suffix(".json").unlink(missing_ok=True);victim.unlink(missing_ok=True)
        return value
