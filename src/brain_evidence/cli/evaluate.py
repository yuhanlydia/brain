"""Evaluation configuration dry runs without model execution or updates."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from brain_evidence.cli import dry_run_payload, emit_json
from brain_evidence.experiments import ConfigError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="brain-evaluate")
    parser.add_argument(
        "--config",
        required=True,
        help="local YAML experiment configuration",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve and validate configuration without loading private inputs",
    )
    parser.add_argument("--hardware-profile")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    if not arguments.dry_run:
        parser.error(
            "real-data evaluation is adapter-owned; use --dry-run to resolve the "
            "configuration without private data"
        )
    try:
        payload = dry_run_payload(
            "evaluate",
            arguments.config,
            hardware_profile=arguments.hardware_profile,
        )
    except (ConfigError, OSError) as error:
        parser.error(str(error))
    emit_json(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
