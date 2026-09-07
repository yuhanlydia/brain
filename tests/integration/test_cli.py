from __future__ import annotations

import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from brain_evidence.adapters.bit import validate_manifest as validate_b2t_manifest
from brain_evidence.adapters.brainhub import (
    validate_manifest as validate_brainhub_manifest,
)
from brain_evidence.cli.evaluate import build_parser as build_evaluate_parser
from brain_evidence.cli.evaluate import main as evaluate_main
from brain_evidence.cli.train import main as train_main

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
prepare_b2t_manifest = runpy.run_path(
    str(REPOSITORY_ROOT / "scripts" / "prepare_b2t.py")
)["prepare_manifest"]
prepare_brainhub_manifest = runpy.run_path(
    str(REPOSITORY_ROOT / "scripts" / "prepare_brainhub.py")
)["prepare_manifest"]


def test_train_synthetic_cli_prints_deterministic_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments = ["--synthetic", "--method", "nra_opsd", "--seed", "13", "--steps", "2"]

    assert train_main(arguments) == 0
    first = json.loads(capsys.readouterr().out)
    assert train_main(arguments) == 0
    second = json.loads(capsys.readouterr().out)

    assert first == second
    assert first["command"] == "train"
    assert first["result"]["method"] == "nra_opsd"
    assert first["result"]["claim_scope"] == "synthetic_execution_only"


@pytest.mark.parametrize(
    "method", ["ce", "vanilla_opsd", "legacy_dual_cosine", "nra_opsd", "rlvr"]
)
def test_train_synthetic_cli_runs_each_registered_method_with_explicit_options(
    capsys: pytest.CaptureFixture[str], method: str
) -> None:
    assert (
        train_main(["--synthetic", "--method", method, "--seed", "13", "--steps", "1"])
        == 0
    )
    result = json.loads(capsys.readouterr().out)["result"]

    assert result["method"] == method
    assert result["seed"] == 13
    assert result["steps"] == 1
    assert result["finite"] is True


def test_train_synthetic_cli_retains_defaults_when_options_are_omitted(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert train_main(["--synthetic"]) == 0
    result = json.loads(capsys.readouterr().out)["result"]

    assert result["method"] == "nra_opsd"
    assert result["seed"] == 0
    assert result["steps"] == 3


@pytest.mark.parametrize(
    ("option", "value"),
    [("--method", "invented"), ("--seed", "invalid"), ("--steps", "0")],
)
def test_train_synthetic_cli_rejects_invalid_option_values(
    capsys: pytest.CaptureFixture[str], option: str, value: str
) -> None:
    with pytest.raises(SystemExit) as error:
        train_main(["--synthetic", option, value])

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert option.removeprefix("--") in captured.err
    assert not captured.out


def test_train_dry_run_resolves_config_without_reading_private_inputs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = REPOSITORY_ROOT / "configs" / "b2t24" / "nra_opsd.yaml"

    assert train_main(["--config", str(config_path), "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["command"] == "train"
    assert payload["status"] == "resolved_only"
    assert payload["private_data_accessed"] is False
    assert payload["config"]["method"] == "interaction_projection"
    assert payload["config"]["adapter"]["manifest"] is None
    assert payload["config"]["resources"]["max_peak_allocation_gib"] == 14.5


def test_evaluate_dry_run_uses_the_same_public_config_resolution(
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = REPOSITORY_ROOT / "configs" / "brainhub" / "umbrae_grounding.yaml"

    assert evaluate_main(["--config", str(config_path), "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["command"] == "evaluate"
    assert payload["status"] == "resolved_only"
    assert payload["private_data_accessed"] is False
    assert payload["config"]["method"] == "evidence_iou_rlvr"
    assert payload["config"]["adapter"]["manifest"] is None


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--method", "ce"),
        ("--method", "vanilla_opsd"),
        ("--method", "legacy_dual_cosine"),
        ("--method", "nra_opsd"),
        ("--method", "rlvr"),
        ("--seed", "0"),
        ("--seed", "41"),
        ("--steps", "1"),
        ("--steps", "3"),
    ],
)
def test_train_config_cli_rejects_explicit_synthetic_options(
    capsys: pytest.CaptureFixture[str],
    option: str,
    value: str,
) -> None:
    config_path = REPOSITORY_ROOT / "configs" / "b2t24" / "nra_opsd.yaml"

    with pytest.raises(SystemExit) as error:
        train_main(["--config", str(config_path), "--dry-run", option, value])

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert option in captured.err
    assert "--synthetic" in captured.err
    assert not captured.out


@pytest.mark.parametrize(
    "arguments",
    [
        ["--synthetic"],
        ["--synthetic", "--method", "ce", "--seed", "13", "--steps", "1"],
    ],
)
def test_evaluate_cli_rejects_synthetic_execution(
    capsys: pytest.CaptureFixture[str], arguments: list[str]
) -> None:
    with pytest.raises(SystemExit) as error:
        evaluate_main(arguments)

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.err
    assert not captured.out


@pytest.mark.parametrize(
    "training_options",
    [["--synthetic"], ["--method", "ce"], ["--seed", "0"], ["--steps", "1"]],
)
def test_evaluate_parser_rejects_training_options(
    capsys: pytest.CaptureFixture[str], training_options: list[str]
) -> None:
    config_path = REPOSITORY_ROOT / "configs" / "b2t24" / "nra_opsd.yaml"

    with pytest.raises(SystemExit) as error:
        build_evaluate_parser().parse_args(
            ["--config", str(config_path), "--dry-run", *training_options]
        )

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert training_options[0] in captured.err
    assert not captured.out


def test_evaluate_cli_requires_dry_run_for_config(
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = REPOSITORY_ROOT / "configs" / "b2t24" / "nra_opsd.yaml"

    with pytest.raises(SystemExit) as error:
        evaluate_main(["--config", str(config_path)])

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert "--dry-run" in captured.err
    assert not captured.out


def test_evaluate_dry_run_accepts_hardware_profile(
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = REPOSITORY_ROOT / "configs" / "b2t24" / "nra_opsd.yaml"
    hardware_path = REPOSITORY_ROOT / "configs" / "hardware" / "gpu24gb.yaml"

    assert (
        evaluate_main(
            [
                "--config",
                str(config_path),
                "--dry-run",
                "--hardware-profile",
                str(hardware_path),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["command"] == "evaluate"
    assert payload["status"] == "resolved_only"
    assert payload["private_data_accessed"] is False
    assert payload["config"]["method"] == "interaction_projection"
    assert payload["config"]["hardware"]["profile"] == "gpu24gb"
    assert payload["config"]["resources"]["max_peak_allocation_gib"] == 22.0


def test_train_module_is_executable_with_python_m() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "brain_evidence.cli.train",
            "--synthetic",
            "--method",
            "ce",
            "--steps",
            "1",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["command"] == "train"
    assert payload["result"]["finite"] is True


@pytest.mark.parametrize(
    ("script_name", "backend", "validator"),
    [
        ("prepare_b2t.py", "b2t25", validate_b2t_manifest),
        ("prepare_brainhub.py", "umbrae", validate_brainhub_manifest),
    ],
)
def test_preparation_scripts_write_adapter_validated_local_manifests(
    tmp_path: Path,
    script_name: str,
    backend: str,
    validator,
) -> None:
    dataset = tmp_path / "user-dataset"
    checkpoint = tmp_path / "user-checkpoint.pt"
    dataset.mkdir()
    checkpoint.write_bytes(b"explicit user checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    revision = "pinned-test-revision"
    output = tmp_path / f"{backend}.yaml"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / script_name),
            "--dataset-path",
            str(dataset),
            "--checkpoint-path",
            str(checkpoint),
            "--backend",
            backend,
            "--revision",
            revision,
            "--checkpoint-sha256",
            checkpoint_sha256,
            "--output",
            str(output),
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    manifest = yaml.safe_load(output.read_text(encoding="utf-8"))
    validated = validator(output)
    assert payload["status"] == "manifest_written"
    assert payload["downloaded"] is False
    assert manifest == {
        "backend": backend,
        "dataset_path": str(dataset.resolve()),
        "checkpoint_path": str(checkpoint.resolve()),
        "revision": revision,
        "checkpoint_sha256": checkpoint_sha256,
    }
    assert validated.dataset_path == dataset.resolve()
    assert validated.checkpoint_path == checkpoint.resolve()
    assert validated.revision == revision
    assert validated.checkpoint_sha256 == checkpoint_sha256
    assert payload["revision"] == revision
    assert payload["checkpoint_sha256"] == checkpoint_sha256


def test_preparation_script_rejects_remote_paths_without_writing_manifest(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"explicit user checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output = tmp_path / "manifest.yaml"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "prepare_b2t.py"),
            "--dataset-path",
            "https://example.invalid/private-data",
            "--checkpoint-path",
            str(checkpoint),
            "--revision",
            "pinned-test-revision",
            "--checkpoint-sha256",
            checkpoint_sha256,
            "--output",
            str(output),
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "local path" in completed.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    "prepare_manifest",
    [prepare_b2t_manifest, prepare_brainhub_manifest],
    ids=["b2t", "brainhub"],
)
@pytest.mark.parametrize("alias_kind", ["exact", "symlink", "hardlink"])
def test_preparation_helpers_reject_checkpoint_output_aliases_without_data_loss(
    tmp_path: Path,
    prepare_manifest,
    alias_kind: str,
) -> None:
    dataset = tmp_path / "user-dataset"
    checkpoint = tmp_path / "user-checkpoint.pt"
    dataset.mkdir()
    checkpoint_bytes = b"checkpoint bytes must survive rejected manifest output"
    checkpoint.write_bytes(checkpoint_bytes)
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()

    if alias_kind == "exact":
        output = checkpoint
    else:
        output = tmp_path / f"{alias_kind}-manifest.yaml"
        if alias_kind == "symlink":
            output.symlink_to(checkpoint)
        else:
            output.hardlink_to(checkpoint)

    with pytest.raises(ValueError) as error:
        prepare_manifest(
            dataset,
            checkpoint,
            output,
            revision="pinned-test-revision",
            checkpoint_sha256=checkpoint_sha256,
        )

    assert checkpoint.read_bytes() == checkpoint_bytes
    assert output.samefile(checkpoint)
    assert "output" in str(error.value)
    assert "checkpoint" in str(error.value)


@pytest.mark.parametrize(
    ("prepare_manifest", "backend", "validator"),
    [
        (prepare_b2t_manifest, "bit", validate_b2t_manifest),
        (prepare_brainhub_manifest, "umbrae", validate_brainhub_manifest),
    ],
    ids=["b2t", "brainhub"],
)
def test_preparation_helpers_can_replace_an_unrelated_existing_manifest(
    tmp_path: Path,
    prepare_manifest,
    backend: str,
    validator,
) -> None:
    dataset = tmp_path / "user-dataset"
    checkpoint = tmp_path / "user-checkpoint.pt"
    output = tmp_path / "manifest.yaml"
    dataset.mkdir()
    checkpoint_bytes = b"explicit user checkpoint"
    checkpoint.write_bytes(checkpoint_bytes)
    output.write_text("previous: manifest\n", encoding="utf-8")
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()

    prepared = prepare_manifest(
        dataset,
        checkpoint,
        output,
        revision="new-pinned-revision",
        checkpoint_sha256=checkpoint_sha256,
    )

    assert checkpoint.read_bytes() == checkpoint_bytes
    assert prepared == validator(output)
    assert yaml.safe_load(output.read_text(encoding="utf-8")) == {
        "backend": backend,
        "dataset_path": str(dataset.resolve()),
        "checkpoint_path": str(checkpoint.resolve()),
        "revision": "new-pinned-revision",
        "checkpoint_sha256": checkpoint_sha256,
    }
