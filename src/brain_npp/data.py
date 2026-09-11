"""NSD trial-manifest records and image-identity split safeguards."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class TrialRecord:
    trial_id: str
    subject_id: str
    session_id: str
    run_id: str
    trial_index: int
    image_id: str
    repeat_id: int
    split: str
    brain_path: str
    image_path: str


_REQUIRED_FIELDS = tuple(TrialRecord.__dataclass_fields__)


def load_jsonl_manifest(path: str | Path) -> tuple[TrialRecord, ...]:
    """Load and validate a JSONL manifest of one record per NSD trial."""
    records: list[TrialRecord] = []
    with Path(path).open("r", encoding="utf-8") as manifest:
        for line_number, line in enumerate(manifest, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at line {line_number}") from error
            if not isinstance(payload, dict):
                raise ValueError(f"manifest line {line_number} must be an object")
            missing = [field for field in _REQUIRED_FIELDS if field not in payload]
            if missing:
                raise ValueError(
                    f"manifest line {line_number} is missing required field {missing[0]!r}"
                )
            unexpected = set(payload) - set(_REQUIRED_FIELDS)
            if unexpected:
                raise ValueError(
                    f"manifest line {line_number} contains unexpected field {sorted(unexpected)[0]!r}"
                )
            try:
                records.append(TrialRecord(**payload))
            except TypeError as error:
                raise ValueError(f"invalid trial record at line {line_number}") from error
    return validate_manifest(records)


def validate_manifest(records: Iterable[TrialRecord]) -> tuple[TrialRecord, ...]:
    """Validate single-trial metadata and prohibit image identity split leakage."""
    validated = tuple(records)
    seen_trial_ids: set[str] = set()
    image_splits: dict[str, str] = {}
    for record in validated:
        _validate_record(record)
        if record.trial_id in seen_trial_ids:
            raise ValueError(f"duplicate trial_id {record.trial_id!r}")
        seen_trial_ids.add(record.trial_id)

        previous_split = image_splits.setdefault(record.image_id, record.split)
        if previous_split != record.split:
            raise ValueError(
                f"image_id {record.image_id!r} appears in both {previous_split!r} and {record.split!r} splits"
            )
    return validated


def assign_grouped_folds(
    records: Sequence[TrialRecord], folds: int, seed: int
) -> dict[str, int]:
    """Assign each image identity to one deterministic fold, keeping repeats together."""
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError("folds must be an integer of at least 2")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    validated = validate_manifest(records)
    image_ids = sorted({record.image_id for record in validated})
    shuffled = sorted(
        image_ids,
        key=lambda image_id: hashlib.sha256(
            f"{seed}:{image_id}".encode("utf-8")
        ).digest(),
    )
    return {image_id: position % folds for position, image_id in enumerate(shuffled)}


def _validate_record(record: TrialRecord) -> None:
    if not isinstance(record, TrialRecord):
        raise ValueError("manifest entries must be TrialRecord instances")
    for field in (
        "trial_id",
        "subject_id",
        "session_id",
        "run_id",
        "image_id",
        "split",
        "brain_path",
        "image_path",
    ):
        value = getattr(record, field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} must be a non-empty string")
    if isinstance(record.trial_index, bool) or not isinstance(record.trial_index, int):
        raise ValueError("trial_index must be an integer")
    if isinstance(record.repeat_id, bool) or not isinstance(record.repeat_id, int):
        raise ValueError("repeat_id must be an integer")
    if record.repeat_id < 0:
        raise ValueError("repeat_id must be non-negative")
