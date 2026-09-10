from __future__ import annotations

import json

import pytest
import torch

from brain_npp.data import (
    TrialRecord,
    assign_grouped_folds,
    load_jsonl_manifest,
    validate_manifest,
)
from brain_npp.controls import apply_brain_control, build_block_derangement


def _trial(
    trial_id: str,
    image_id: str,
    *,
    repeat_id: int = 0,
    split: str = "train",
    trial_index: int = 0,
    subject_id: str = "sub-01",
    session_id: str = "ses-01",
    run_id: str = "run-01",
) -> TrialRecord:
    return TrialRecord(
        trial_id=trial_id,
        subject_id=subject_id,
        session_id=session_id,
        run_id=run_id,
        trial_index=trial_index,
        image_id=image_id,
        repeat_id=repeat_id,
        split=split,
        brain_path=f"brain/{trial_id}.pt",
        image_path=f"image/{image_id}.jpg",
    )


def test_manifest_accepts_repeats_in_one_split_and_grouped_folds_keep_them_together() -> None:
    records = [_trial("trial-a", "image-1"), _trial("trial-b", "image-1", repeat_id=1)]

    validated = validate_manifest(records)
    folds = assign_grouped_folds(validated, folds=2, seed=7)

    assert validated == tuple(records)
    assert folds == {"image-1": 0}


def test_manifest_rejects_image_leakage_between_splits() -> None:
    records = [_trial("trial-a", "image-1", split="train"), _trial("trial-b", "image-1", repeat_id=1, split="test")]

    with pytest.raises(ValueError, match="image-1"):
        validate_manifest(records)


def test_jsonl_loader_rejects_missing_single_trial_field(tmp_path) -> None:
    payload = _trial("trial-a", "image-1").__dict__.copy()
    del payload["brain_path"]
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="brain_path"):
        load_jsonl_manifest(path)


def test_manifest_rejects_duplicate_trial_ids() -> None:
    records = [_trial("trial-a", "image-1"), _trial("trial-a", "image-2")]

    with pytest.raises(ValueError, match="duplicate trial_id.*trial-a"):
        validate_manifest(records)


@pytest.mark.parametrize("repeat_id", [-1, "0"])
def test_manifest_rejects_invalid_repeat_ids(repeat_id) -> None:
    record = _trial("trial-a", "image-1")
    payload = record.__dict__.copy()
    payload["repeat_id"] = repeat_id

    with pytest.raises(ValueError, match="repeat_id"):
        validate_manifest([TrialRecord(**payload)])


def test_block_derangement_is_seeded_and_preserves_strata_without_self_or_image_matches() -> None:
    records = [
        _trial("trial-a", "image-a", trial_index=0),
        _trial("trial-b", "image-b", trial_index=3),
        _trial("trial-c", "image-c", trial_index=6),
        _trial("trial-d", "image-d", trial_index=9),
    ]

    mapping, unmatched = build_block_derangement(
        records, seed=13, preserve_run=True, adjacency_radius=1
    )

    assert (mapping, unmatched) == build_block_derangement(
        records, seed=13, preserve_run=True, adjacency_radius=1
    )
    assert unmatched == ()
    by_id = {record.trial_id: record for record in records}
    for source_id, matched_id in mapping.items():
        source, matched = by_id[source_id], by_id[matched_id]
        assert source_id != matched_id
        assert source.image_id != matched.image_id
        assert (source.subject_id, source.session_id, source.run_id) == (
            matched.subject_id,
            matched.session_id,
            matched.run_id,
        )
        assert abs(source.trial_index - matched.trial_index) > 1


def test_block_derangement_returns_unmatched_ids_when_run_blocks_are_impossible() -> None:
    records = [
        _trial("trial-a", "image-a", run_id="run-01"),
        _trial("trial-b", "image-b", run_id="run-02"),
    ]

    mapping, unmatched = build_block_derangement(records, seed=0, preserve_run=True)

    assert mapping == {}
    assert unmatched == ("trial-a", "trial-b")


def test_block_derangement_does_not_return_a_chain_into_an_unmatched_trial() -> None:
    records = [
        _trial("trial-a", "image-x", trial_index=0),
        _trial("trial-b", "image-y", trial_index=2),
        _trial("trial-c", "image-x", trial_index=4, repeat_id=1),
    ]
    brain = torch.tensor([[-1.0], [0.0], [1.0]])

    mapping, unmatched = build_block_derangement(records, seed=0)

    assert mapping == {}
    assert unmatched == ("trial-a", "trial-b", "trial-c")
    retained_ids = tuple(record.trial_id for record in records if record.trial_id not in unmatched)
    retained_brain = brain[: len(retained_ids)]
    assert torch.equal(
        apply_brain_control(
            retained_brain, "shuffle", mapping=mapping, trial_ids=retained_ids
        ),
        retained_brain,
    )


def test_brain_controls_apply_precomputed_mappings_and_reject_missing_references() -> None:
    brain = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
    trial_ids = ("trial-a", "trial-b")
    mapping = {"trial-a": "trial-b", "trial-b": "trial-a"}

    assert torch.equal(apply_brain_control(brain, "correct"), brain)
    assert torch.equal(apply_brain_control(brain, "zero"), torch.zeros_like(brain))
    assert torch.equal(
        apply_brain_control(brain, "shuffle", mapping=mapping, trial_ids=trial_ids),
        brain.flip(0),
    )
    assert torch.equal(
        apply_brain_control(brain, "wrong-subject", mapping=mapping, trial_ids=trial_ids),
        brain.flip(0),
    )
    with pytest.raises(ValueError, match="missing"):
        apply_brain_control(
            brain,
            "shuffle",
            mapping={"trial-a": "missing", "trial-b": "trial-a"},
            trial_ids=trial_ids,
        )


def test_covariance_noise_is_seeded_and_preserves_normalized_tensor_shape() -> None:
    brain = torch.tensor([[-1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [1.0, 1.0]])

    first = apply_brain_control(brain, "covariance-noise", seed=23)
    second = apply_brain_control(brain, "covariance-noise", seed=23)

    assert first.shape == brain.shape
    assert torch.equal(first, second)
    assert not torch.equal(first, brain)
