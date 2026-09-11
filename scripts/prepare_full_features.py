#!/usr/bin/env python3
"""Resume frozen CLIP pooled-feature extraction for a validated NSD image bank."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_chunk_hash(values, start, end):
    return hashlib.sha256(np.ascontiguousarray(values[start:end]).tobytes()).hexdigest()


def completed_chunks(values, completed, chunk_size):
    return [{"start": start, "end": min(start + chunk_size, completed),
             "sha256": _array_chunk_hash(values, start, min(start + chunk_size, completed))}
            for start in range(0, completed, chunk_size)]


def validate_resume(progress, features, *, total_images):
    if progress.get("schema_version") != 2:
        raise ValueError("unsupported feature progress schema")
    completed = progress.get("completed_images")
    if isinstance(completed, bool) or not isinstance(completed, int) or not 0 <= completed <= total_images:
        raise ValueError("completed_images is outside feature bank bounds")
    if progress.get("total_images") != total_images:
        raise ValueError("progress total_images mismatch")
    if progress.get("array_shape") != list(features.shape) or progress.get("array_dtype") != str(features.dtype):
        raise ValueError("feature resume array schema mismatch")
    cursor = 0
    for chunk in progress.get("chunks", []):
        if chunk.get("start") != cursor or not cursor < chunk.get("end", -1) <= completed:
            raise ValueError("feature resume chunks are not a contiguous completed prefix")
        if chunk.get("sha256") != _array_chunk_hash(features, cursor, chunk["end"]):
            raise ValueError("completed feature chunk hash mismatch")
        cursor = chunk["end"]
    if cursor != completed:
        raise ValueError("feature resume chunks do not cover completed_images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-provenance", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    # Heavy optional dependencies remain out of module import time.
    import torch
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPVisionModel
    records = sorted((json.loads(x) for x in args.image_provenance.read_text().splitlines()), key=lambda x: x["image_id"])
    inputs = {"image_provenance": sha256(args.image_provenance)}
    for name in ("config.json", "preprocessor_config.json", "pytorch_model.bin"):
        inputs[name] = sha256(args.model / name)
    for row in records:
        if sha256(args.images / f"{row['image_id']}.jpg") != row["sha256"]:
            raise ValueError(f"source image checksum mismatch: {row['image_id']}")
    args.output.mkdir(parents=True, exist_ok=True)
    progress_path = args.output / "clip_progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else None
    if progress and progress["input_hashes"] != inputs:
        raise ValueError("incompatible feature resume inputs")
    start = int(progress["completed_images"]) if progress else 0
    feature_path = args.output / "image_pooled.npy"
    features = np.lib.format.open_memmap(feature_path, mode="r+" if start else "w+", dtype=np.float32, shape=(len(records), 1024))
    if progress: validate_resume(progress, features, total_images=len(records))
    image_ids_path = args.output / "image_ids.json"
    image_ids_path.write_text(json.dumps([x["image_id"] for x in records]) + "\n")
    processor = CLIPImageProcessor.from_pretrained(args.model, local_files_only=True)
    model = CLIPVisionModel.from_pretrained(args.model, local_files_only=True).eval().cuda()
    with torch.inference_mode():
        for offset in range(start, len(records), args.batch_size):
            batch = records[offset:offset + args.batch_size]
            images = []
            for row in batch:
                with Image.open(args.images / f"{row['image_id']}.jpg") as image:
                    images.append(image.convert("RGB"))
            values = model(processor(images=images, return_tensors="pt").pixel_values.cuda()).pooler_output.float().cpu().numpy()
            if not np.isfinite(values).all(): raise ValueError("nonfinite CLIP features")
            features[offset:offset + len(batch)] = values; features.flush()
            completed = offset + len(batch)
            chunks = list(progress.get("chunks", [])) if progress else []
            chunks.append({"start": offset, "end": completed, "sha256": _array_chunk_hash(features, offset, completed)})
            progress = {"schema_version": 2, "input_hashes": inputs, "completed_images": completed,
                        "total_images": len(records), "array_shape": list(features.shape), "array_dtype": str(features.dtype),
                        "chunks": chunks, "purpose": "fixed image features only; no brain/label fitting"}
            temporary = progress_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(progress, indent=2) + "\n"); temporary.replace(progress_path)
    progress.update(feature_sha256=sha256(feature_path), image_ids_sha256=sha256(image_ids_path),
                    all_finite=bool(np.isfinite(features).all()))
    (args.output / "clip_provenance.json").write_text(json.dumps(progress, indent=2) + "\n")


if __name__ == "__main__": main()
