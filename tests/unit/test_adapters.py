from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
import yaml

from brain_evidence.adapters import BITAdapter, BrainHubAdapter
from brain_evidence.protocols import BrainOnlyBatch, PrivilegedTeacherBatch


def _manifest(tmp_path: Path, *, backend: str) -> Path:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"test checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "backend": backend,
                "dataset_path": str(dataset_path),
                "checkpoint_path": str(checkpoint_path),
                "revision": "test-revision",
                "checkpoint_sha256": checkpoint_sha256,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_bit_adapter_builds_typed_student_and_teacher_batches(
    tmp_path: Path,
) -> None:
    adapter = BITAdapter.from_manifest(
        _manifest(tmp_path, backend="bit"), brain_feature_dim=8
    )
    brain = torch.zeros(2, 5, 8)
    prefix = torch.ones(2, 4, dtype=torch.long)
    brain_mask = torch.ones(2, 5, dtype=torch.bool)
    privilege = torch.ones(2, 7, dtype=torch.long)

    student = adapter.student_batch(brain, prefix, brain_mask=brain_mask)
    teacher = adapter.teacher_batch(
        brain,
        privilege,
        prefix,
        brain_mask=brain_mask,
    )

    assert isinstance(student, BrainOnlyBatch)
    assert isinstance(teacher, PrivilegedTeacherBatch)
    assert student.brain is brain
    assert teacher.privilege is privilege
    assert adapter.manifest is not None
    assert adapter.manifest.revision == "test-revision"
    assert (
        adapter.manifest.checkpoint_sha256
        == hashlib.sha256(b"test checkpoint").hexdigest()
    )


@pytest.mark.parametrize(
    ("brain", "prefix", "message"),
    [
        (torch.zeros(2, 8), torch.ones(2, 4, dtype=torch.long), "brain.*rank 3"),
        (
            torch.zeros(2, 5, 8),
            torch.ones(3, 4, dtype=torch.long),
            "batch size",
        ),
        (torch.zeros(2, 5, 7), torch.ones(2, 4, dtype=torch.long), "feature.*8"),
        (torch.zeros(2, 5, 8), torch.ones(2, 4), "integer token ids"),
    ],
)
def test_bit_adapter_rejects_invalid_student_shapes(
    tmp_path: Path,
    brain: torch.Tensor,
    prefix: torch.Tensor,
    message: str,
) -> None:
    adapter = BITAdapter.from_manifest(
        _manifest(tmp_path, backend="bit"), brain_feature_dim=8
    )

    with pytest.raises(ValueError, match=message):
        adapter.student_batch(brain, prefix)


def test_bit_adapter_rejects_teacher_privilege_batch_mismatch(
    tmp_path: Path,
) -> None:
    adapter = BITAdapter.from_manifest(
        _manifest(tmp_path, backend="bit"), brain_feature_dim=8
    )

    with pytest.raises(ValueError, match="privilege.*batch size"):
        adapter.teacher_batch(
            torch.zeros(2, 5, 8),
            torch.ones(3, 7, dtype=torch.long),
            torch.ones(2, 4, dtype=torch.long),
        )


def test_brainhub_adapter_validates_brain_and_privileged_feature_shapes(
    tmp_path: Path,
) -> None:
    adapter = BrainHubAdapter.from_manifest(
        _manifest(tmp_path, backend="umbrae"), brain_feature_dim=16
    )
    prefix = torch.ones(2, 4, dtype=torch.long)

    student = adapter.student_batch(torch.zeros(2, 6, 16), prefix)
    teacher = adapter.teacher_batch(
        torch.zeros(2, 6, 16),
        torch.zeros(2, 5, 16),
        prefix,
    )

    assert isinstance(student, BrainOnlyBatch)
    assert isinstance(teacher, PrivilegedTeacherBatch)

    with pytest.raises(ValueError, match="privilege.*feature.*16"):
        adapter.teacher_batch(
            torch.zeros(2, 6, 16),
            torch.zeros(2, 5, 8),
            prefix,
        )


def test_adapter_rejects_missing_user_supplied_private_inputs(tmp_path: Path) -> None:
    missing_dataset = tmp_path / "private-dataset"
    missing_checkpoint = tmp_path / "private-checkpoint.pt"
    manifest_path = tmp_path / "missing.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "backend": "bit",
                "dataset_path": str(missing_dataset),
                "checkpoint_path": str(missing_checkpoint),
                "revision": "test-revision",
                "checkpoint_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        FileNotFoundError,
        match="user-supplied.*never downloads private inputs",
    ):
        BITAdapter.from_manifest(manifest_path)

    assert not missing_dataset.exists()
    assert not missing_checkpoint.exists()


def test_adapter_reports_an_actionable_missing_manifest_field(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    manifest_path = tmp_path / "incomplete.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "backend": "bit",
                "checkpoint_path": str(checkpoint_path),
                "revision": "test-revision",
                "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="user-supplied.*never downloads"):
        BITAdapter.from_manifest(manifest_path)


def test_adapter_rejects_remote_manifest_paths_without_downloading(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    manifest_path = tmp_path / "remote.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "backend": "bit",
                "dataset_path": "https://example.test/private-data.tar",
                "checkpoint_path": str(checkpoint_path),
                "revision": "test-revision",
                "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="local path.*never downloads"):
        BITAdapter.from_manifest(manifest_path)


def test_adapter_rejects_manifest_for_the_wrong_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="backend.*umbrae"):
        BITAdapter.from_manifest(_manifest(tmp_path, backend="umbrae"))


@pytest.mark.parametrize(
    ("bad_field", "message"),
    [
        ("dataset_path", "dataset_path.*directory"),
        ("checkpoint_path", "checkpoint_path.*file"),
    ],
)
def test_adapter_validates_manifest_path_kinds(
    tmp_path: Path,
    bad_field: str,
    message: str,
) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    payload = {
        "backend": "bit",
        "dataset_path": str(dataset_path),
        "checkpoint_path": str(checkpoint_path),
        "revision": "test-revision",
        "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
    }
    if bad_field == "dataset_path":
        not_a_directory = tmp_path / "dataset.bin"
        not_a_directory.touch()
        payload[bad_field] = str(not_a_directory)
    else:
        not_a_file = tmp_path / "checkpoint-directory"
        not_a_file.mkdir()
        payload[bad_field] = str(not_a_file)
    manifest_path = tmp_path / f"bad-{bad_field}.yaml"
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        BITAdapter.from_manifest(manifest_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("revision", "", "revision.*non-empty"),
        ("checkpoint_sha256", "abc", "checkpoint_sha256.*64"),
        ("checkpoint_sha256", "f" * 64, "checkpoint_sha256.*does not match"),
    ],
)
def test_adapter_validates_manifest_revision_and_checkpoint_digest(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    manifest_path = _manifest(tmp_path, backend="bit")
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    payload[field] = value
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        BITAdapter.from_manifest(manifest_path)
