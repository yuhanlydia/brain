"""Training command for synthetic execution and real-config dry runs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from brain_evidence.cli import dry_run_payload, emit_json
from brain_evidence.experiments import ConfigError
from brain_evidence.experiments.synthetic import (
    SYNTHETIC_METHODS,
    run_synthetic_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="brain-train")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--config",
        help="local YAML experiment configuration",
    )
    source.add_argument(
        "--synthetic",
        action="store_true",
        help="run an invented CPU smoke experiment (not a benchmark)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve and validate configuration without loading private inputs",
    )
    parser.add_argument(
        "--hardware-profile",
        help="optional local hardware YAML overriding the configured profile",
    )
    parser.add_argument(
        "--method",
        choices=SYNTHETIC_METHODS,
        help="synthetic training path (default: nra_opsd)",
    )
    parser.add_argument("--seed", type=int, help="synthetic seed (default: 0)")
    parser.add_argument(
        "--steps", type=int, help="synthetic training steps (default: 3)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    if arguments.synthetic:
        if arguments.dry_run:
            parser.error("--dry-run applies to --config, not --synthetic")
        if arguments.hardware_profile is not None:
            parser.error("--hardware-profile applies only to --config")
        try:
            result = run_synthetic_experiment(
                arguments.method if arguments.method is not None else "nra_opsd",
                seed=arguments.seed if arguments.seed is not None else 0,
                steps=arguments.steps if arguments.steps is not None else 3,
            )
        except ValueError as error:
            parser.error(str(error))
        emit_json({"command": "train", "result": result})
        return 0

    for option in ("method", "seed", "steps"):
        if getattr(arguments, option) is not None:
            parser.error(f"--{option} applies only to --synthetic")

    if not arguments.dry_run:
        parser.error(
            "real-data execution is adapter-owned; use --dry-run to resolve the "
            "configuration without private data"
        )
    try:
        payload = dry_run_payload(
            "train",
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
