from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from brain_evidence.experiments.config import resolve_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _python_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
    return environment


@pytest.mark.parametrize(
    "command",
    [
        (sys.executable, "-m", "brain_evidence.cli.train", "--help"),
        (sys.executable, "-m", "brain_evidence.cli.evaluate", "--help"),
        (sys.executable, "scripts/prepare_b2t.py", "--help"),
        (sys.executable, "scripts/prepare_brainhub.py", "--help"),
    ],
)
def test_documented_command_targets_offer_help(command: tuple[str, ...]) -> None:
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=_python_environment(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("script_name", "backend"),
    [
        ("prepare_b2t.py", "b2t24"),
        ("prepare_b2t.py", "b2t25"),
        ("prepare_brainhub.py", "umbrae"),
    ],
)
def test_documented_preparation_commands_pin_existing_local_inputs(
    tmp_path: Path,
    script_name: str,
    backend: str,
) -> None:
    dataset = tmp_path / f"{backend}-dataset"
    checkpoint = tmp_path / f"{backend}-checkpoint.pt"
    manifest_path = tmp_path / f"{backend}.yaml"
    dataset.mkdir()
    checkpoint.write_bytes(f"{backend} checkpoint".encode())
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    revision = f"{backend}-pinned-revision"

    completed = subprocess.run(
        (
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / script_name),
            "--dataset-path",
            str(dataset),
            "--checkpoint-path",
            str(checkpoint),
            "--revision",
            revision,
            "--checkpoint-sha256",
            checkpoint_sha256,
            "--backend",
            backend,
            "--output",
            str(manifest_path),
        ),
        cwd=REPOSITORY_ROOT,
        env=_python_environment(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert checkpoint.is_file()
    assert result["downloaded"] is False
    assert result["revision"] == revision
    assert result["checkpoint_sha256"] == checkpoint_sha256
    assert manifest == {
        "backend": backend,
        "dataset_path": str(dataset.resolve()),
        "checkpoint_path": str(checkpoint.resolve()),
        "revision": revision,
        "checkpoint_sha256": checkpoint_sha256,
    }


@pytest.mark.parametrize(
    ("relative_config", "benchmark", "task", "adapter", "memory_limit"),
    [
        (
            "configs/b2t24/nra_opsd.yaml",
            "brain_to_text_2024",
            "brain_to_text",
            "bit",
            14.5,
        ),
        (
            "configs/b2t25/nra_opsd.yaml",
            "brain_to_text_2025",
            "brain_to_text",
            "bit",
            14.5,
        ),
        (
            "configs/brainhub/umbrae_grounding.yaml",
            "brainhub",
            "visual_grounding",
            "brainhub",
            22.0,
        ),
    ],
)
def test_documented_experiment_configs_resolve_through_the_public_api(
    relative_config: str,
    benchmark: str,
    task: str,
    adapter: str,
    memory_limit: float,
) -> None:
    resolved = resolve_config(REPOSITORY_ROOT / relative_config)

    assert resolved["experiment"]["benchmark"] == benchmark
    assert resolved["task"] == task
    assert resolved["adapter"]["name"] == adapter
    assert resolved["seeds"] == [41, 42, 43]
    assert resolved["negative_sampling"]["k"] == 4
    assert resolved["negative_sampling"]["distinct_pronunciations"] is True
    assert resolved["controls"]["null_permutations"] == 999
    assert resolved["controls"]["brain_only_source"] == "frozen_crossfit_ce"
    assert resolved["resources"]["max_peak_allocation_gib"] == memory_limit


@pytest.mark.parametrize(
    ("relative_config", "memory_limit"),
    [
        ("configs/b2t24/nra_opsd.yaml", 14.5),
        ("configs/b2t25/nra_opsd.yaml", 14.5),
        ("configs/brainhub/umbrae_grounding.yaml", 22.0),
    ],
)
@pytest.mark.parametrize("command_module", ["train", "evaluate"])
def test_documented_config_dry_runs_do_not_access_private_data(
    relative_config: str,
    memory_limit: float,
    command_module: str,
) -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            f"brain_evidence.cli.{command_module}",
            "--config",
            relative_config,
            "--dry-run",
        ),
        cwd=REPOSITORY_ROOT,
        env=_python_environment(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "resolved_only"
    assert payload["private_data_accessed"] is False
    assert payload["config"]["adapter"]["manifest"] is None
    assert payload["config"]["resources"]["max_peak_allocation_gib"] == memory_limit


def test_documented_smoke_target_executes_on_cpu() -> None:
    completed = subprocess.run(
        (sys.executable, "scripts/smoke_test.py"),
        cwd=REPOSITORY_ROOT,
        env=_python_environment(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["claim_scope"] == "synthetic_execution_only"
    assert payload["benchmark_claim"] is False


def test_local_manifests_are_ignored_to_protect_private_paths() -> None:
    completed = subprocess.run(
        ("git", "check-ignore", "local/manifests/private.yaml"),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "local/manifests/private.yaml"
