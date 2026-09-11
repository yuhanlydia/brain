#!/usr/bin/env python3
"""Resume full subject-wise, training-only Gaussian calibration fits."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from brain_npp import calibration as calibration_module
from brain_npp.calibration import GaussianEncodingLikelihood
from brain_npp.data import assign_grouped_folds, load_jsonl_manifest


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


CALIBRATION_SOURCE_PATH = Path(inspect.getsourcefile(calibration_module) or calibration_module.__file__)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def training_brain_identity(records):
    sources = [{"trial_id": record.trial_id, "path": str(Path(record.brain_path).resolve()),
                "sha256": sha256(record.brain_path)} for record in records]
    return {"train_brain_sources": sources,
            "train_brain_identity_sha256": hashlib.sha256(_canonical(sources).encode()).hexdigest()}


def fit_implementation_identity(*, outer_folds, inner_folds, fold_seed):
    hyperparameters = {"outer_folds": outer_folds, "inner_folds": inner_folds, "fold_seed": fold_seed,
        "projection_seed": calibration_module.PROJECTION_SEED,
        "projection_dimension": calibration_module.PROJECTION_DIMENSION,
        "ridge_alpha": calibration_module.RIDGE_ALPHA,
        "covariance_shrinkage": calibration_module.COVARIANCE_SHRINKAGE,
        "covariance_jitter": calibration_module.COVARIANCE_JITTER}
    return {"calibration_source_sha256": sha256(CALIBRATION_SOURCE_PATH),
            "fit_hyperparameters": hyperparameters}


def validated_feature_identity(features_path, image_ids_path, provenance_path):
    provenance = json.loads(Path(provenance_path).read_text())
    identity = {"image_features_sha256": sha256(features_path), "image_ids_sha256": sha256(image_ids_path)}
    expected = {"image_features_sha256": provenance.get("feature_sha256"),
                "image_ids_sha256": provenance.get("image_ids_sha256")}
    for key, value in identity.items():
        if expected[key] != value:
            raise ValueError(f"feature provenance {key} mismatch")
    return identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--image-ids", type=Path, required=True)
    parser.add_argument("--feature-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=["subj01", "subj02", "subj05", "subj07"])
    parser.add_argument("--folds", type=int, default=5); parser.add_argument("--fold-seed", type=int, default=1731)
    args = parser.parse_args()
    records = load_jsonl_manifest(args.manifest)
    features = np.load(args.features, mmap_mode="r", allow_pickle=False)
    image_ids = json.loads(args.image_ids.read_text()); feature_row = {x: i for i, x in enumerate(image_ids)}
    inputs_base = {"manifest_sha256": sha256(args.manifest),
                   **validated_feature_identity(args.features, args.image_ids, args.feature_provenance),
                   **fit_implementation_identity(outer_folds=args.folds, inner_folds=args.folds,
                                                 fold_seed=args.fold_seed),
                   "fold_seed": args.fold_seed, "outer_folds": args.folds}
    training = [r for r in records if r.split == "train"]
    global_folds = assign_grouped_folds(training, args.folds, args.fold_seed)
    for subject in args.subjects:
        selected = [r for r in records if r.subject_id == subject and r.split == "train"]
        folds = {r.image_id: global_folds[r.image_id] for r in selected}
        subject_dir = args.output / subject; subject_dir.mkdir(parents=True, exist_ok=True)
        inputs = {**inputs_base, **training_brain_identity(selected), "subject": subject}; inputs_path = subject_dir / "inputs.json"
        if inputs_path.exists() and json.loads(inputs_path.read_text()) != inputs: raise ValueError(f"incompatible resume inputs for {subject}")
        inputs_path.write_text(json.dumps(inputs, indent=2) + "\n")
        brains = np.stack([np.load(r.brain_path) for r in selected]); x = np.asarray(features[[feature_row[r.image_id] for r in selected]])
        feature_fit_provenance = _canonical({key: inputs[key] for key in
            ("image_features_sha256", "image_ids_sha256", "calibration_source_sha256", "fit_hyperparameters")})
        brain_fit_provenance = _canonical({key: inputs[key] for key in
            ("manifest_sha256", "subject", "train_brain_identity_sha256")})
        artifacts = {}; scores = np.empty(len(selected), dtype=np.float64)
        for name in [str(x) for x in range(args.folds)] + ["final"]:
            path = subject_dir / f"{name}.npz"
            held = None if name == "final" else int(name)
            mask = np.ones(len(selected), dtype=bool) if held is None else np.array([folds[r.image_id] != held for r in selected])
            if path.exists():
                model = GaussianEncodingLikelihood.load(path); reused = True
                expected_samples = [r.trial_id for r, keep in zip(selected, mask) if keep]
                if model.provenance.get("train_sample_ids") != expected_samples: raise ValueError(f"incompatible saved fit {path}")
                if model.provenance.get("feature_provenance") != feature_fit_provenance or model.provenance.get("brain_provenance") != brain_fit_provenance:
                    raise ValueError(f"incompatible cached fit provenance {path}")
            else:
                model = GaussianEncodingLikelihood.fit(x[mask], brains[mask], image_ids=[r.image_id for r, keep in zip(selected, mask) if keep],
                    sample_ids=[r.trial_id for r, keep in zip(selected, mask) if keep], feature_provenance=feature_fit_provenance,
                    brain_provenance=brain_fit_provenance, inner_folds=args.folds)
                temporary = path.with_suffix(".tmp.npz"); model.save(temporary); temporary.replace(path); reused = False
            if held is not None:
                held_mask = ~mask
                if set(r.image_id for r, keep in zip(selected, held_mask) if keep) & set(model.provenance["train_image_ids"]):
                    raise ValueError(f"fold {name} contains held-out image")
                scores[held_mask] = model.score_paired(brains[held_mask], x[held_mask])
            artifacts[name] = {"sha256": sha256(path), "train_images": len(model.provenance["train_image_ids"]), "reused": reused}
        np.savez(subject_dir / "oof_density.npz", trial_ids=[r.trial_id for r in selected],
                 image_ids=[r.image_id for r in selected], fold_ids=[folds[r.image_id] for r in selected], paired_log_density=scores)
        provenance = {**inputs, "purpose": "full training-only likelihood fitting", "validation_test_used_in_fitting": False,
                      "all_oof_scores_finite": bool(np.isfinite(scores).all()), "artifacts": artifacts}
        (subject_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__": main()
