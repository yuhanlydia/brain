"""Validated full-NSD examples, galleries, calibration routes, and controls."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from .data import TrialRecord, assign_grouped_folds, load_jsonl_manifest, validate_manifest


@dataclass(frozen=True)
class VQAExample:
    trial: TrialRecord
    question_id: int
    answer_id: int
    question: str
    answers: tuple[str, ...]
    category: str


@dataclass(frozen=True)
class CaptionReference:
    caption_id: int
    caption: str


@dataclass(frozen=True)
class CaptionExample:
    trial: TrialRecord
    references: tuple[CaptionReference, ...]


@dataclass(frozen=True)
class CandidateGallery:
    image_ids: tuple[str, ...]
    log_prior: tuple[float, ...]
    policy: str = "uniform-subject-train-bank"
    conditioning: str = "sampled-gallery"

    def coverage(self, true_image_id: str) -> bool:
        return true_image_id in self.image_ids


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict]:
    result = []
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at line {number} in {path}") from error
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {number} in {path} must be an object")
            result.append(row)
    return result


class FullNSDData:
    """Lightweight index over full NSD metadata and memory-mapped image features."""

    def __init__(self, records, image_ids, features, vqa, captions, provenance):
        self.records = tuple(records)
        self.image_ids = tuple(image_ids)
        self.features = features
        self.feature_row = {value: index for index, value in enumerate(self.image_ids)}
        self._vqa, self._captions, self.provenance = vqa, captions, provenance

    @classmethod
    def load(cls, manifest_path, *, features_path, image_ids_path,
             feature_provenance_path, vqa_paths=None, caption_paths=None,
             row_mapping_binding_path=None):
        manifest_path, features_path = Path(manifest_path), Path(features_path)
        records = load_jsonl_manifest(manifest_path)
        image_ids = json.loads(Path(image_ids_path).read_text(encoding="utf-8"))
        if not isinstance(image_ids, list) or len(set(image_ids)) != len(image_ids) or not all(isinstance(x, str) for x in image_ids):
            raise ValueError("image_ids must be a unique JSON string list")
        features = np.load(features_path, mmap_mode="r", allow_pickle=False)
        provenance = json.loads(Path(feature_provenance_path).read_text(encoding="utf-8"))
        if provenance.get("feature_sha256") != _sha256(features_path):
            raise ValueError("feature sha256 does not match provenance")
        image_ids_hash = _sha256(Path(image_ids_path))
        binding = None
        if provenance.get("image_ids_sha256") != image_ids_hash:
            if provenance.get("image_ids_sha256") is not None or row_mapping_binding_path is None:
                raise ValueError("image_ids sha256 does not match feature provenance")
            binding = json.loads(Path(row_mapping_binding_path).read_text(encoding="utf-8"))
            expected_binding = {
                "schema_version": 1, "kind": "full-nsd-feature-row-binding",
                "manifest_sha256": _sha256(manifest_path),
                "image_features_sha256": _sha256(features_path),
                "image_ids_sha256": image_ids_hash,
                "original_feature_provenance_sha256": _sha256(Path(feature_provenance_path)),
                "ordering": "sorted-unique-manifest-image-ids", "count": len(image_ids),
            }
            for key, value in expected_binding.items():
                if binding.get(key) != value:
                    raise ValueError(f"legacy row mapping binding {key} mismatch")
            evidence_hash = binding.get("evidence_sha256")
            if not isinstance(evidence_hash, str) or len(evidence_hash) != 64 or any(x not in "0123456789abcdef" for x in evidence_hash):
                raise ValueError("legacy row mapping binding evidence_sha256 is invalid")
            if image_ids != sorted({r.image_id for r in records}):
                raise ValueError("legacy row mapping is not sorted unique manifest image IDs")
        if features.ndim != 2 or features.shape[0] != len(image_ids):
            raise ValueError("feature rows must match image_ids")
        missing = sorted({r.image_id for r in records} - set(image_ids))
        if missing:
            raise ValueError(f"manifest image missing from features: {missing[0]}")
        vqa = {split: _rows(Path(path)) for split, path in (vqa_paths or {}).items()}
        captions = {split: _rows(Path(path)) for split, path in (caption_paths or {}).items()}
        return cls(records, image_ids, features, vqa, captions,
                   {"manifest_sha256": _sha256(manifest_path), "feature_sha256": _sha256(features_path),
                    "image_ids_sha256": image_ids_hash, "row_mapping_binding": binding})

    def image_features(self, image_ids: Sequence[str]) -> np.ndarray:
        try:
            return np.asarray(self.features[[self.feature_row[x] for x in image_ids]])
        except KeyError as error:
            raise ValueError(f"unknown image_id {error.args[0]!r}") from error

    def _trials(self, split, subject_id):
        return [r for r in self.records if r.split == split and (subject_id is None or r.subject_id == subject_id)]

    def vqa_examples(self, split: str, *, subject_id: str | None = None):
        annotations = {}
        for row in self._vqa.get(split, []):
            image_id = f"nsd{int(row['nsd_image_id']):05d}"
            annotations.setdefault(image_id, []).append(row)
        result = []
        for record in self._trials(split, subject_id):
            for row in annotations.get(record.image_id, []):
                result.append(VQAExample(record, int(row["question_id"]), int(row["answer_id"]),
                    str(row["question"]), tuple(str(x) for x in row["answers"]), str(row["category"])))
        return tuple(result)

    def caption_examples(self, split: str, *, subject_id: str | None = None):
        annotations = {str(row["image_id"]): row for row in self._captions.get(split, [])}
        result = []
        for record in self._trials(split, subject_id):
            if record.image_id in annotations:
                refs = tuple(CaptionReference(int(x["caption_id"]), str(x["caption"]))
                             for x in annotations[record.image_id]["references"])
                result.append(CaptionExample(record, refs))
        return tuple(result)

    def sample_gallery(self, *, subject_id: str, experiment_seed: int, data_cursor: int, count: int = 4):
        if any(isinstance(x, bool) or not isinstance(x, int) for x in (experiment_seed, data_cursor, count)) or count < 1:
            raise ValueError("experiment_seed, data_cursor, and positive count must be integers")
        bank = sorted({r.image_id for r in self.records if r.subject_id == subject_id and r.split == "train"})
        if count > len(bank):
            raise ValueError("candidate count exceeds subject training image bank")
        digest = hashlib.sha256(f"gallery-v1:{experiment_seed}:{data_cursor}:{subject_id}".encode()).digest()
        chosen = random.Random(int.from_bytes(digest[:8], "big")).sample(bank, count)
        return CandidateGallery(tuple(chosen), tuple([-math.log(count)] * count))

    @staticmethod
    def calibration_name(record: TrialRecord, *, folds: Mapping[str, int]) -> str:
        if record.split != "train":
            return "final"
        if record.image_id not in folds:
            raise ValueError(f"training image {record.image_id!r} has no outer fold")
        return str(folds[record.image_id])

    @staticmethod
    def validate_calibration_exclusion(record: TrialRecord, provenance: Mapping) -> None:
        if record.split == "train" and record.image_id in set(provenance.get("train_image_ids", ())):
            raise ValueError(f"held-out image {record.image_id!r} appears in calibration training images")


class FullCalibrationStore:
    """Hash-validated router for train OOF and validation/test final fits."""

    def __init__(self, root, data: FullNSDData, *, fold_seed: int = 1731, outer_folds: int = 5):
        self.root, self.data = Path(root), data
        training = [r for r in data.records if r.split == "train"]
        self.folds = assign_grouped_folds(training, outer_folds, fold_seed)
        self._inputs = {}
        for subject in sorted({r.subject_id for r in data.records}):
            path = self.root / subject / "inputs.json"
            if not path.exists():
                continue
            inputs = json.loads(path.read_text(encoding="utf-8"))
            expected = {"manifest_sha256": data.provenance["manifest_sha256"],
                        "image_features_sha256": data.provenance["feature_sha256"],
                        "image_ids_sha256": data.provenance["image_ids_sha256"],
                        "fold_seed": fold_seed, "outer_folds": outer_folds}
            for key, value in expected.items():
                if key == "image_ids_sha256" and key not in inputs and data.provenance["row_mapping_binding"] is not None:
                    continue
                if inputs.get(key) != value:
                    raise ValueError(f"calibration {subject} {key} mismatch")
            self._inputs[subject] = inputs

    def model_for(self, record: TrialRecord):
        """Load and validate exactly the fit allowed for this trial."""
        from .calibration import GaussianEncodingLikelihood
        if record.subject_id not in self._inputs:
            raise ValueError(f"no calibration inputs for {record.subject_id}")
        name = self.data.calibration_name(record, folds=self.folds)
        path = self.root / record.subject_id / f"{name}.npz"
        provenance_path = self.root / record.subject_id / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        expected_hash = provenance.get("artifacts", {}).get(name, {}).get("sha256")
        binding = self.data.provenance["row_mapping_binding"]
        if binding is not None:
            bound_hash = binding.get("calibration_artifacts", {}).get(record.subject_id, {}).get(name)
            if bound_hash != expected_hash:
                raise ValueError(f"legacy calibration binding {record.subject_id}/{name} mismatch")
        if expected_hash != _sha256(path):
            raise ValueError(f"calibration artifact {name} sha256 mismatch")
        model = GaussianEncodingLikelihood.load(path)
        self.data.validate_calibration_exclusion(record, model.provenance)
        return model

    def score(self, record: TrialRecord, brain, candidate_features):
        return self.model_for(record).score(brain, candidate_features)


def build_wrong_subject_mapping(records: Sequence[TrialRecord], *, subject_order=("subj01", "subj02", "subj05", "subj07")):
    records = validate_manifest(records)
    order = tuple(subject_order)
    if len(set(order)) != len(order):
        raise ValueError("subject_order must contain unique subjects")
    by_key = {(r.subject_id, r.image_id, r.repeat_id): r for r in records}
    mapping, unmatched = {}, []
    for record in records:
        if record.subject_id not in order:
            raise ValueError(f"subject {record.subject_id!r} missing from subject_order")
        target_subject = order[(order.index(record.subject_id) + 1) % len(order)]
        target = by_key.get((target_subject, record.image_id, record.repeat_id))
        if target is None:
            unmatched.append(record.trial_id)
        elif target.subject_id == record.subject_id or target.image_id != record.image_id or target.repeat_id != record.repeat_id:
            raise ValueError("invalid wrong-subject match metadata")
        else:
            mapping[record.trial_id] = target.trial_id
    return mapping, tuple(unmatched)


def save_control_mapping(path, mapping, unmatched, *, control, manifest_sha256):
    payload = {"schema_version": 1, "control": control, "manifest_sha256": manifest_sha256,
               "mapping": dict(mapping), "unmatched_ids": list(unmatched)}
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_control_mapping(path, *, expected_control=None, manifest_sha256=None):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("mapping"), dict):
        raise ValueError("invalid control mapping artifact")
    if expected_control is not None and payload.get("control") != expected_control:
        raise ValueError("control mapping type mismatch")
    if manifest_sha256 is not None and payload.get("manifest_sha256") != manifest_sha256:
        raise ValueError("control mapping manifest mismatch")
    return payload["mapping"], tuple(payload.get("unmatched_ids", ()))


@dataclass(frozen=True)
class ResolvedWrongSubjectBrain:
    source_subject_id: str
    target_trial_id: str
    brain: np.ndarray


class WrongSubjectResolver:
    def __init__(self, mapping, records, voxel_counts):
        records = validate_manifest(records)
        self.mapping, self.records = mapping, {r.trial_id: r for r in records}
        self.voxel_counts = voxel_counts
        for source_id, target_id in mapping.items():
            if source_id not in self.records or target_id not in self.records:
                raise ValueError("wrong-subject mapping contains unknown trial ID")
            source, target = self.records[source_id], self.records[target_id]
            if source.image_id != target.image_id or source.repeat_id != target.repeat_id:
                raise ValueError("wrong-subject target must have same image and repeat")
            if source.subject_id == target.subject_id:
                raise ValueError("wrong-subject target must have different subject")

    @classmethod
    def from_artifact(cls, path, records, *, manifest_sha256,
                      voxel_counts=None):
        mapping, unmatched = load_control_mapping(path, expected_control="wrong-subject", manifest_sha256=manifest_sha256)
        record_ids = {r.trial_id for r in records}
        if set(mapping) & set(unmatched) or set(mapping) | set(unmatched) != record_ids:
            raise ValueError("wrong-subject mapping and unmatched IDs must partition the manifest records")
        native = voxel_counts or {"subj01": 15724, "subj02": 14278, "subj05": 13039, "subj07": 12682}
        return cls(mapping, records, native)

    def resolve(self, source_trial_id, *, loader=np.load):
        if source_trial_id not in self.mapping:
            raise ValueError(f"unmatched wrong-subject trial {source_trial_id!r}")
        target = self.records[self.mapping[source_trial_id]]
        brain = np.asarray(loader(target.brain_path))
        if brain.ndim != 1 or not np.isfinite(brain).all():
            raise ValueError("wrong-subject brain must be a finite native voxel vector")
        if target.subject_id not in self.voxel_counts:
            raise ValueError(f"no native voxel dimension declared for {target.subject_id}")
        if brain.shape[0] != self.voxel_counts[target.subject_id]:
            raise ValueError(f"{target.subject_id} brain has non-native voxel dimension")
        return ResolvedWrongSubjectBrain(target.subject_id, target.trial_id, brain)


@dataclass(frozen=True)
class EmpiricalCovarianceNoise:
    centered_reference: torch.Tensor
    provenance: dict

    @classmethod
    def from_records(cls, records: Sequence[TrialRecord], *, subject_id: str, manifest_sha256: str):
        validated = validate_manifest(records)
        if not validated or any(r.split != "train" for r in validated):
            raise ValueError("covariance references must all be training records")
        if any(r.subject_id != subject_id for r in validated):
            raise ValueError("covariance references must belong to one requested subject")
        hashes = {r.trial_id: _sha256(Path(r.brain_path)) for r in validated}
        reference = torch.from_numpy(np.stack([np.load(r.brain_path) for r in validated]))
        if not isinstance(reference, torch.Tensor) or reference.ndim < 2 or not reference.is_floating_point() or reference.shape[0] < 2:
            raise ValueError("training reference must be a floating tensor with at least two trials")
        flat = reference.detach().reshape(reference.shape[0], -1).clone()
        if not torch.isfinite(flat).all():
            raise ValueError("training reference must be finite")
        provenance = {"manifest_sha256": manifest_sha256, "subject_id": subject_id,
                      "split": "train", "trial_ids": [r.trial_id for r in validated], "brain_sha256": hashes}
        return cls(flat - flat.mean(0, keepdim=True), provenance)

    def sample(self, count: int, *, seed: int):
        if count < 1:
            raise ValueError("count must be positive")
        generator = torch.Generator(device=self.centered_reference.device).manual_seed(seed)
        weights = torch.randn((count, self.centered_reference.shape[0]), generator=generator,
                              device=self.centered_reference.device, dtype=self.centered_reference.dtype)
        return weights @ self.centered_reference / math.sqrt(self.centered_reference.shape[0] - 1)
