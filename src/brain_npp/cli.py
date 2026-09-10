"""Command-line entry points for local NPP-OPSD validation and smoke runs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
import yaml

from .adapters.external import ExternalBackendUnavailable, run_external_training
from .data import load_jsonl_manifest
from .toy import ToyAdapter
from .trainer import NPPTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="brain-npp")
    subcommands = parser.add_subparsers(dest="command", required=True)

    smoke = subcommands.add_parser("smoke", help="run deterministic CPU optimization")
    smoke.add_argument("--config", required=True)

    manifest = subcommands.add_parser(
        "validate-manifest", help="validate a local JSONL trial manifest"
    )
    manifest.add_argument("path")

    train = subcommands.add_parser("train", help="train from a local backend config")
    train.add_argument("--config", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "validate-manifest":
            records = load_jsonl_manifest(arguments.path)
            _print_json({"valid": True, "trials": len(records)})
            return 0
        config = load_config(arguments.config)
        if arguments.command == "smoke":
            _print_json(run_smoke(config))
            return 0
        return _run_train(config)
    except (ExternalBackendUnavailable, OSError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, Mapping):
        raise ValueError("config must contain a YAML mapping")
    return dict(payload)


def run_smoke(config: Mapping[str, Any]) -> dict[str, Any]:
    """Execute real torch updates and return machine-readable diagnostics."""
    backend = config.get("backend")
    if backend != "toy":
        raise ValueError("smoke requires backend: toy; external backends never fall back")
    seed = _integer(config, "seed", 314159)
    steps = _integer(config, "steps", 30)
    if steps < 5:
        raise ValueError("smoke steps must be at least 5")
    torch.manual_seed(seed)
    toy_config = config.get("toy", {})
    if not isinstance(toy_config, Mapping):
        raise ValueError("toy must be a mapping")
    adapter = ToyAdapter(seed=seed, **dict(toy_config))
    optimizer_config = config.get("optimizer", {})
    if not isinstance(optimizer_config, Mapping):
        raise ValueError("optimizer must be a mapping")
    learning_rate = float(optimizer_config.get("lr", 0.08))
    optimizer = torch.optim.Adam(adapter.parameters(), lr=learning_rate)
    trainer_config = config.get("trainer", {})
    if not isinstance(trainer_config, Mapping):
        raise ValueError("trainer must be a mapping")
    trainer = NPPTrainer(
        adapter,
        optimizer,
        generation_config=config.get("generation", {}),
        **dict(trainer_config),
    )
    batch = adapter.make_batch()
    diagnostics = [trainer.step(batch) for _ in range(steps)]
    losses = [step["npp_loss"] for step in diagnostics]
    if not all(math.isfinite(loss) for loss in losses):
        raise ValueError("smoke produced a non-finite NPP loss")
    window = min(5, steps)
    initial_mean = sum(losses[:window]) / window
    final_mean = sum(losses[-window:]) / window
    if final_mean >= initial_mean:
        raise ValueError(
            "smoke did not reduce final mean NPP loss below the initial mean"
        )
    return {
        "method": "NPP-OPSD",
        "steps": steps,
        "npp_losses": losses,
        "initial_npp_loss_mean": initial_mean,
        "final_npp_loss_mean": final_mean,
        "prefix_consistent": adapter.prefix_consistent,
        "final_diagnostics": diagnostics[-1],
    }


def _run_train(config: Mapping[str, Any]) -> int:
    backend = config.get("backend")
    if backend == "toy":
        _print_json(run_smoke(config))
        return 0
    if backend not in {"vindex", "brainjanus"}:
        raise ValueError("backend must be one of: toy, vindex, brainjanus")
    _print_json(run_external_training(config))
    return 0


def _print_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


def _integer(config: Mapping[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
