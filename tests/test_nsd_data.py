from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from brain_npp.data import TrialRecord, assign_grouped_folds
from brain_npp.nsd_data import (
    EmpiricalCovarianceNoise,
    FullCalibrationStore,
    FullNSDData,
    build_wrong_subject_mapping,
    load_control_mapping,
    WrongSubjectResolver,
    save_control_mapping,
)


def trial(tid, image, *, subject="subj01", split="train", index=0, repeat=0):
    return TrialRecord(tid, subject, "session01", "run01", index, image, repeat,
                       split, f"brain/{tid}.npy", f"images/{image}.jpg")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def fixture(tmp_path):
    records = [trial("a", "nsd00001"), trial("b", "nsd00002"),
               trial("c", "nsd00003", subject="subj02"),
               trial("d", "nsd00001", subject="subj02")]
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, [r.__dict__ for r in records])
    ids = ["nsd00001", "nsd00002", "nsd00003"]
    np.save(tmp_path / "features.npy", np.arange(6, dtype=np.float32).reshape(3, 2))
    (tmp_path / "image_ids.json").write_text(json.dumps(ids))
    provenance = {"feature_sha256": hashlib.sha256((tmp_path / "features.npy").read_bytes()).hexdigest(),
                  "image_ids_sha256": hashlib.sha256((tmp_path / "image_ids.json").read_bytes()).hexdigest(),
                  "all_finite": True, "completed_images": 3, "total_images": 3}
    (tmp_path / "feature_provenance.json").write_text(json.dumps(provenance))
    write_jsonl(tmp_path / "vqa_train.jsonl", [{"nsd_image_id": 1, "coco_image_id": 9,
        "question_id": 11, "answer_id": 12, "question": "where?", "answers": ["here"], "category": "place"}])
    write_jsonl(tmp_path / "captions_train.jsonl", [{"image_id": "nsd00001", "nsd_image_id": 1,
        "coco_image_id": 9, "references": [{"caption_id": 21, "caption": "caption"}]}])
    return records, manifest


def test_annotation_join_preserves_question_answer_and_caption_ids(tmp_path):
    _, manifest = fixture(tmp_path)
    data = FullNSDData.load(manifest, features_path=tmp_path / "features.npy",
        image_ids_path=tmp_path / "image_ids.json", feature_provenance_path=tmp_path / "feature_provenance.json",
        vqa_paths={"train": tmp_path / "vqa_train.jsonl"}, caption_paths={"train": tmp_path / "captions_train.jsonl"})
    vqa = data.vqa_examples("train", subject_id="subj01")
    captions = data.caption_examples("train", subject_id="subj01")
    assert [(x.trial.trial_id, x.question_id, x.answer_id, x.answers) for x in vqa] == [("a", 11, 12, ("here",))]
    assert [(x.trial.trial_id, x.references[0].caption_id) for x in captions] == [("a", 21)]


def test_gallery_depends_only_on_seed_cursor_subject_train_bank(tmp_path):
    _, manifest = fixture(tmp_path)
    data = FullNSDData.load(manifest, features_path=tmp_path / "features.npy",
        image_ids_path=tmp_path / "image_ids.json", feature_provenance_path=tmp_path / "feature_provenance.json")
    first = data.sample_gallery(subject_id="subj01", experiment_seed=41, data_cursor=8, count=2)
    assert first.image_ids == data.sample_gallery(subject_id="subj01", experiment_seed=41, data_cursor=8, count=2).image_ids
    assert len(set(first.image_ids)) == 2 and first.log_prior == (-np.log(2), -np.log(2))
    assert first.policy == "uniform-subject-train-bank" and first.conditioning == "sampled-gallery"


def test_feature_provenance_mismatch_fails_closed(tmp_path):
    _, manifest = fixture(tmp_path)
    (tmp_path / "feature_provenance.json").write_text(json.dumps({"feature_sha256": "bad"}))
    with pytest.raises(ValueError, match="feature.*sha256"):
        FullNSDData.load(manifest, features_path=tmp_path / "features.npy",
            image_ids_path=tmp_path / "image_ids.json", feature_provenance_path=tmp_path / "feature_provenance.json")


def test_swapped_image_ids_fail_feature_provenance_before_row_lookup(tmp_path):
    _, manifest = fixture(tmp_path)
    (tmp_path / "image_ids.json").write_text(json.dumps(["nsd00002", "nsd00001", "nsd00003"]))
    with pytest.raises(ValueError, match="image_ids sha256"):
        FullNSDData.load(manifest, features_path=tmp_path / "features.npy",
            image_ids_path=tmp_path / "image_ids.json", feature_provenance_path=tmp_path / "feature_provenance.json")


def test_legacy_feature_provenance_requires_explicit_exact_row_binding(tmp_path):
    _, manifest = fixture(tmp_path)
    provenance_path = tmp_path / "feature_provenance.json"
    provenance = json.loads(provenance_path.read_text()); del provenance["image_ids_sha256"]
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="image_ids sha256"):
        FullNSDData.load(manifest, features_path=tmp_path / "features.npy",
            image_ids_path=tmp_path / "image_ids.json", feature_provenance_path=provenance_path)
    binding = {"schema_version": 1, "kind": "full-nsd-feature-row-binding",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "image_features_sha256": hashlib.sha256((tmp_path / "features.npy").read_bytes()).hexdigest(),
        "image_ids_sha256": hashlib.sha256((tmp_path / "image_ids.json").read_bytes()).hexdigest(),
        "original_feature_provenance_sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
        "ordering": "sorted-unique-manifest-image-ids", "count": 3, "calibration_artifacts": {}}
    binding_path = tmp_path / "binding.json"; binding_path.write_text(json.dumps(binding))
    with pytest.raises(ValueError, match="evidence_sha256"):
        FullNSDData.load(manifest, features_path=tmp_path / "features.npy",
            image_ids_path=tmp_path / "image_ids.json", feature_provenance_path=provenance_path,
            row_mapping_binding_path=binding_path)
    binding["evidence_sha256"] = "a" * 64; binding_path.write_text(json.dumps(binding))
    loaded = FullNSDData.load(manifest, features_path=tmp_path / "features.npy",
        image_ids_path=tmp_path / "image_ids.json", feature_provenance_path=provenance_path,
        row_mapping_binding_path=binding_path)
    assert loaded.provenance["row_mapping_binding"]["image_ids_sha256"] == binding["image_ids_sha256"]


def test_calibration_route_uses_oof_for_train_and_final_for_eval_with_exclusion(tmp_path):
    records, manifest = fixture(tmp_path)
    data = FullNSDData.load(manifest, features_path=tmp_path / "features.npy",
        image_ids_path=tmp_path / "image_ids.json", feature_provenance_path=tmp_path / "feature_provenance.json")
    folds = assign_grouped_folds([r for r in records if r.subject_id == "subj01"], 2, 1731)
    assert data.calibration_name(records[0], folds=folds) == str(folds[records[0].image_id])
    assert data.calibration_name(TrialRecord(**{**records[0].__dict__, "split": "val"}), folds=folds) == "final"
    with pytest.raises(ValueError, match="held-out image"):
        data.validate_calibration_exclusion(records[0], {"train_image_ids": [records[0].image_id]})


def test_calibration_store_rejects_inputs_from_other_manifest_or_features(tmp_path):
    _, manifest = fixture(tmp_path)
    data = FullNSDData.load(manifest, features_path=tmp_path / "features.npy",
        image_ids_path=tmp_path / "image_ids.json", feature_provenance_path=tmp_path / "feature_provenance.json")
    directory = tmp_path / "calibration" / "subj01"; directory.mkdir(parents=True)
    (directory / "inputs.json").write_text(json.dumps({"manifest_sha256": "wrong",
        "image_features_sha256": data.provenance["feature_sha256"], "fold_seed": 1731, "outer_folds": 5}))
    with pytest.raises(ValueError, match="manifest_sha256"):
        FullCalibrationStore(tmp_path / "calibration", data)


def test_wrong_subject_mapping_requires_same_image_repeat_and_routes_source_dimensions(tmp_path):
    records, _ = fixture(tmp_path)
    mapping, unmatched = build_wrong_subject_mapping(records, subject_order=("subj01", "subj02"))
    assert mapping == {"a": "d", "d": "a"}
    assert unmatched == ("b", "c")
    source = {r.trial_id: r for r in records}
    assert source[mapping["a"]].subject_id == "subj02" and source[mapping["a"]].image_id == source["a"].image_id
    path = tmp_path / "mapping.json"
    save_control_mapping(path, mapping, unmatched, control="wrong-subject", manifest_sha256="manifest-hash")
    assert load_control_mapping(path, expected_control="wrong-subject", manifest_sha256="manifest-hash") == (mapping, unmatched)


def test_wrong_subject_resolver_revalidates_mapping_and_loads_native_target(tmp_path):
    records, _ = fixture(tmp_path)
    for record, values in zip(records, ([1, 2, 3], [4, 5, 6], [7, 8], [9, 10])):
        path = tmp_path / f"{record.trial_id}.npy"; np.save(path, np.asarray(values, dtype=np.float32))
    records = [TrialRecord(**{**r.__dict__, "brain_path": str(tmp_path / f"{r.trial_id}.npy")}) for r in records]
    path = tmp_path / "mapping.json"
    save_control_mapping(path, {"a": "d", "d": "a"}, ("b", "c"), control="wrong-subject", manifest_sha256="manifest-hash")
    resolver = WrongSubjectResolver.from_artifact(path, records, manifest_sha256="manifest-hash",
                                                  voxel_counts={"subj01": 3, "subj02": 2})
    resolved = resolver.resolve("a")
    assert resolved.source_subject_id == "subj02"
    assert resolved.target_trial_id == "d"
    np.testing.assert_array_equal(resolved.brain, [9, 10])
    payload = json.loads(path.read_text()); payload["mapping"]["a"] = "c"; path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="same image"):
        WrongSubjectResolver.from_artifact(path, records, manifest_sha256="manifest-hash",
                                           voxel_counts={"subj01": 3, "subj02": 2})


def test_covariance_noise_loads_validated_training_records_and_records_sources(tmp_path):
    records = [trial(f"t{x}", f"nsd{x:05d}", index=x) for x in range(4)]
    for x, record in enumerate(records):
        path = tmp_path / f"{record.trial_id}.npy"; np.save(path, np.asarray([x, x * 2], dtype=np.float32))
    records = [TrialRecord(**{**r.__dict__, "brain_path": str(tmp_path / f"{r.trial_id}.npy")}) for r in records]
    sampler = EmpiricalCovarianceNoise.from_records(records, subject_id="subj01", manifest_sha256="manifest-hash")
    first = sampler.sample(2, seed=7)
    second = sampler.sample(2, seed=7)
    assert torch.equal(first, second) and first.shape == (2, 2)
    assert sampler.provenance["trial_ids"] == ["t0", "t1", "t2", "t3"]
    assert sampler.provenance["manifest_sha256"] == "manifest-hash"
    bad = [TrialRecord(**{**records[0].__dict__, "split": "val"}), *records[1:]]
    with pytest.raises(ValueError, match="training"):
        EmpiricalCovarianceNoise.from_records(bad, subject_id="subj01", manifest_sha256="manifest-hash")
