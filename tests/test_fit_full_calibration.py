from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

from brain_npp.nsd_data import FullCalibrationStore, FullNSDData

_SPEC = importlib.util.spec_from_file_location("fit_full_calibration", Path(__file__).parents[1] / "scripts/fit_full_calibration.py")
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC); _SPEC.loader.exec_module(_MODULE)


def test_calibration_identity_rejects_image_ids_not_bound_by_feature_provenance(tmp_path):
    features = tmp_path / "features.npy"; features.write_bytes(b"features")
    ids = tmp_path / "image_ids.json"; ids.write_text('["b", "a"]')
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({"feature_sha256": hashlib.sha256(features.read_bytes()).hexdigest(),
        "image_ids_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="image_ids_sha256"):
        _MODULE.validated_feature_identity(features, ids, provenance)


def _tiny_inputs(tmp_path):
    images = [f"nsd{x:05d}" for x in range(1, 9)]
    feature_path = tmp_path / "features.npy"
    np.save(feature_path, np.arange(16, dtype=np.float64).reshape(8, 2))
    ids_path = tmp_path / "image_ids.json"; ids_path.write_text(json.dumps(images))
    feature_provenance = tmp_path / "feature-provenance.json"
    feature_provenance.write_text(json.dumps({"feature_sha256": _MODULE.sha256(feature_path),
        "image_ids_sha256": _MODULE.sha256(ids_path)}))
    rows = []
    for index, image_id in enumerate(images):
        brain_path = tmp_path / f"brain-{index}.npy"
        np.save(brain_path, np.asarray([index + 0.1, 2 * index - 0.2], dtype=np.float64))
        rows.append({"trial_id": f"trial-{index}", "subject_id": "subj01", "session_id": "s1",
            "run_id": "r1", "trial_index": index, "image_id": image_id, "repeat_id": 0,
            "split": "train", "brain_path": str(brain_path), "image_path": f"{image_id}.jpg"})
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return manifest, feature_path, ids_path, feature_provenance, Path(rows[0]["brain_path"])


def test_real_fit_reuse_rejects_changed_brain_and_calibration_source(tmp_path, monkeypatch):
    manifest, features, ids, provenance, brain = _tiny_inputs(tmp_path)
    source_copy = tmp_path / "calibration.py"; shutil.copyfile(_MODULE.CALIBRATION_SOURCE_PATH, source_copy)
    monkeypatch.setattr(_MODULE, "CALIBRATION_SOURCE_PATH", source_copy)
    argv = ["fit_full_calibration.py", "--manifest", str(manifest), "--features", str(features),
            "--image-ids", str(ids), "--feature-provenance", str(provenance), "--output", str(tmp_path / "out"),
            "--subjects", "subj01", "--folds", "2"]
    monkeypatch.setattr(sys, "argv", argv); _MODULE.main()
    monkeypatch.setattr(sys, "argv", argv); _MODULE.main()
    np.save(brain, np.asarray([999.0, 999.0]))
    with pytest.raises(ValueError, match="incompatible resume inputs"):
        monkeypatch.setattr(sys, "argv", argv); _MODULE.main()
    np.save(brain, np.asarray([0.1, -0.2])); source_copy.write_text(source_copy.read_text() + "\n# mutation\n")
    with pytest.raises(ValueError, match="incompatible resume inputs"):
        monkeypatch.setattr(sys, "argv", argv); _MODULE.main()


def test_cli_artifacts_load_through_full_calibration_store(tmp_path, monkeypatch):
    manifest, features, ids, provenance, _ = _tiny_inputs(tmp_path)
    output = tmp_path / "out"
    argv = ["fit_full_calibration.py", "--manifest", str(manifest), "--features", str(features),
            "--image-ids", str(ids), "--feature-provenance", str(provenance), "--output", str(output),
            "--subjects", "subj01", "--folds", "2"]
    monkeypatch.setattr(sys, "argv", argv); _MODULE.main()
    data = FullNSDData.load(manifest, features_path=features, image_ids_path=ids,
                            feature_provenance_path=provenance)
    store = FullCalibrationStore(output, data, fold_seed=1731, outer_folds=2)
    record = data.records[0]
    model = store.model_for(record)
    assert record.image_id not in model.provenance["train_image_ids"]
