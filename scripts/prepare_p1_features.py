#!/usr/bin/env python3
"""Reproducibly extract frozen CLIP224 layer -2 and strict BrainXS features."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from brain_npp.provenance import git_revision, manifest_hash, sha256_file, sha256_tree

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--umbrae", type=Path, required=True)
    parser.add_argument("--vindex", type=Path, required=True)
    parser.add_argument("--umbrae-revision", required=True)
    parser.add_argument("--vindex-revision", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verified_umbrae = git_revision(args.umbrae, args.umbrae_revision)
    verified_vindex = git_revision(args.vindex, args.vindex_revision)
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPVisionModel
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    image_paths = {row["image_id"]: row["image_path"] for row in rows}; image_ids = sorted(image_paths)
    clip_path = args.assets / "clip-vit-large-patch14"
    processor = CLIPImageProcessor.from_pretrained(clip_path, local_files_only=True)
    clip = CLIPVisionModel.from_pretrained(clip_path, local_files_only=True).eval().cuda()
    patches, pooled = [], []
    with torch.inference_mode():
        for image_id in image_ids:
            with Image.open(image_paths[image_id]) as image:
                pixels = processor(images=image.convert("RGB"), return_tensors="pt").pixel_values.cuda()
            output = clip(pixels, output_hidden_states=True)
            patches.append(output.hidden_states[-2][0, 1:].float().cpu().numpy())
            pooled.append(output.pooler_output[0].float().cpu().numpy())
    del clip; torch.cuda.empty_cache(); sys.path.insert(0, str(args.umbrae / "umbrae"))
    from model import BrainXS
    encoder = BrainXS(in_dim=15724, hidden_dim=1024, out_dim=1024, num_latents=256)
    checkpoint = args.assets / "vindex/train_logs/specific_sub/sub01_dim1024/clip_224/last.pth"
    encoder.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)["model_state_dict"], strict=True)
    encoder.eval().cuda(); brains = np.stack([np.load(row["brain_path"], allow_pickle=False) for row in rows])
    with torch.inference_mode():
        brain_features = np.stack([encoder(torch.from_numpy(value).unsqueeze(0).cuda())[0].float().cpu().numpy() for value in brains])
    arrays = {"image_ids": np.array(image_ids), "trial_ids": np.array([r["trial_id"] for r in rows]),
        "trial_image_ids": np.array([r["image_id"] for r in rows]), "brains": brains,
        "image_patches": np.stack(patches), "image_pooled": np.stack(pooled), "brain_features": brain_features}
    if arrays["image_patches"].shape != (len(image_ids), 256, 1024) or any(not np.isfinite(v).all() for v in arrays.values() if v.dtype.kind == "f"):
        raise ValueError("prepared feature shape/finite validation failed")
    args.output.parent.mkdir(parents=True, exist_ok=True); np.savez(args.output, **arrays)
    source_files = {"manifest": sha256_file(args.manifest), "brainxs_checkpoint": sha256_file(checkpoint),
        "clip_checkpoint_tree": sha256_tree(clip_path), "umbrae_revision": verified_umbrae, "vindex_revision": verified_vindex,
        "brain_files": {row["trial_id"]: sha256_file(row["brain_path"]) for row in rows},
        "image_files": {image_id: sha256_file(path) for image_id, path in image_paths.items()}}
    metadata = {"purpose": "P1 preparation only; no accuracy claim",
        "umbrae_revision": verified_umbrae, "vindex_revision": verified_vindex,
        "source_manifest": source_files, "source_manifest_sha256": manifest_hash(source_files),
        "image_teacher_features": "CLIP224 hidden_states[-2] patch tokens", "likelihood_image_features": "CLIP224 pooler_output",
        "brain_inputs": "subject01 single-trial float32 betas; no repetition averaging",
        "manifest_sha256": sha256_file(args.manifest), "artifact_sha256": sha256_file(args.output),
        "shapes": {key: list(value.shape) for key, value in arrays.items()}}
    args.output.with_name("features.provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__": main()
