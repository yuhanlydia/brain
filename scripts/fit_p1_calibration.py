#!/usr/bin/env python3
"""Fit and serialize cross-fitted Gaussian P1 likelihood artifacts."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from brain_npp.calibration import GaussianEncodingLikelihood, crossfit_gaussian_encoding

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--features", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); data = np.load(args.features, allow_pickle=False)
    lookup = {str(value): index for index, value in enumerate(data["image_ids"])}
    rows = np.array([lookup[str(value)] for value in data["trial_image_ids"]])
    result = crossfit_gaussian_encoding(data["image_pooled"][rows], data["brains"], image_ids=data["trial_image_ids"],
        sample_ids=data["trial_ids"], feature_provenance="CLIP224 pooler_output revision 32bd64288804d66eefd0ccbe215aa642df71cc41",
        brain_provenance="subject01 single-trial betas; MindEyeV2 revision 26421f100e4c6012a35ecadb272a0ec1d999202d", outer_folds=5, inner_folds=5)
    gallery_rows = np.random.default_rng(41).permutation(len(data["image_ids"]))[:4]; gallery = data["image_ids"][gallery_rows]
    scores = np.empty((len(data["trial_ids"]), 4)); folds = {}; args.output.mkdir(parents=True, exist_ok=True)
    for fold, model in result.models.items():
        held = result.fold_ids == fold
        if set(data["trial_image_ids"][held]) & set(model.provenance["train_image_ids"]): raise ValueError("fold image leakage")
        path = args.output / f"fold{fold}.npz"; model.save(path); loaded = GaussianEncodingLikelihood.load(path)
        scores[held] = loaded.score(data["brains"][held], data["image_pooled"][gallery_rows])
        folds[str(fold)] = {"artifact": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    np.savez(args.output / "scores.npz", trial_ids=data["trial_ids"], candidate_ids=gallery, fold_ids=result.fold_ids, log_likelihood=scores)
    provenance = {"purpose": "P1 real-data calibration smoke, not benchmark performance",
        "input_features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "candidate_gallery_policy": "seed41 uniform image-only fixed gallery; no true-image insertion",
        "candidate_ids": gallery.tolist(), "all_scores_finite": bool(np.isfinite(scores).all()), "folds": folds}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n"); print(json.dumps(provenance, indent=2))

if __name__ == "__main__": main()
