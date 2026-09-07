"""Command-line helpers for configuration and synthetic smoke runs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from brain_evidence.experiments import resolve_config

ConfigSource = str | Path | Mapping[str, Any]


def emit_json(payload: Mapping[str, Any]) -> None:
    """Print stable, machine-readable CLI output."""

    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def dry_run_payload(
    command: str,
    config: ConfigSource,
    *,
    hardware_profile: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve a run without constructing an adapter or touching private data."""

    resolved = resolve_config(config, hardware_profile=hardware_profile)
    return {
        "command": command,
        "status": "resolved_only",
        "dry_run": True,
        "private_data_accessed": False,
        "config": resolved,
    }


def train_main(argv: Sequence[str] | None = None) -> int:
    """Lazy public wrapper for :func:`brain_evidence.cli.train.main`."""

    from .train import main

    return main(argv)


def evaluate_main(argv: Sequence[str] | None = None) -> int:
    """Lazy public wrapper for :func:`brain_evidence.cli.evaluate.main`."""

    from .evaluate import main

    return main(argv)


__all__ = ["dry_run_payload", "emit_json", "evaluate_main", "train_main"]
