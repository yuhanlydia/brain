"""Create a BIT manifest from explicit local user-provided paths."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import yaml

from brain_evidence.adapters.bit import BITManifest, validate_manifest

_BACKENDS = ("bit", "b2t24", "b2t25", "brain-to-text")


def _validated_output_path(output_path: str | Path, checkpoint_path: Path) -> Path:
    """Reject checkpoint aliases before creating directories or writing output."""
    destination = Path(output_path).expanduser().resolve()
    if destination == checkpoint_path or (
        destination.exists() and destination.samefile(checkpoint_path)
    ):
        raise ValueError("output path must not alias the checkpoint path")
    return destination


def prepare_manifest(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    revision: str,
    checkpoint_sha256: str,
    backend: str = "bit",
) -> BITManifest:
    """Validate local inputs, write their manifest, and validate the result."""

    requested = {
        "backend": backend,
        "dataset_path": str(dataset_path),
        "checkpoint_path": str(checkpoint_path),
        "revision": revision,
        "checkpoint_sha256": checkpoint_sha256,
    }
    validated_inputs = validate_manifest(requested)
    payload = {
        "backend": validated_inputs.backend,
        "dataset_path": str(validated_inputs.dataset_path),
        "checkpoint_path": str(validated_inputs.checkpoint_path),
        "revision": validated_inputs.revision,
        "checkpoint_sha256": validated_inputs.checkpoint_sha256,
    }
    destination = _validated_output_path(output_path, validated_inputs.checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return validate_manifest(destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        "--dataset",
        dest="dataset_path",
        required=True,
    )
    parser.add_argument(
        "--checkpoint-path",
        "--checkpoint",
        dest="checkpoint_path",
        required=True,
    )
    parser.add_argument("--revision", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=_BACKENDS, default="bit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        manifest = prepare_manifest(
            arguments.dataset_path,
            arguments.checkpoint_path,
            arguments.output,
            revision=arguments.revision,
            checkpoint_sha256=arguments.checkpoint_sha256,
            backend=arguments.backend,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "status": "manifest_written",
                "downloaded": False,
                "backend": manifest.backend,
                "manifest": str(manifest.manifest_path),
                "dataset_path": str(manifest.dataset_path),
                "checkpoint_path": str(manifest.checkpoint_path),
                "revision": manifest.revision,
                "checkpoint_sha256": manifest.checkpoint_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
