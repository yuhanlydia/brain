from __future__ import annotations

import csv
from pathlib import Path
import resource
import signal
import subprocess
import sys

import pytest

from brain_npp.data import load_jsonl_manifest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "prepare_nsd_manifest.py"
FIELDS = (
    "trial_id",
    "subject_id",
    "session_id",
    "run_id",
    "trial_index",
    "image_id",
    "repeat_id",
    "split",
    "brain_path",
    "image_path",
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _row(
    trial_id: str,
    image_id: str,
    repeat_id: int,
    *,
    split: str = "train",
) -> dict[str, str]:
    return {
        "trial_id": trial_id,
        "subject_id": "subj01",
        "session_id": "session01",
        "run_id": "run01",
        "trial_index": str(repeat_id),
        "image_id": image_id,
        "repeat_id": str(repeat_id),
        "split": split,
        "brain_path": f"brain/{trial_id}.npy",
        "image_path": f"images/{image_id}.png",
    }


def _set_tiny_file_size_limit() -> None:
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    resource.setrlimit(resource.RLIMIT_FSIZE, (64, 64))


def _run_prepare(
    source: Path,
    destination: Path,
    *,
    limit_output_size: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-B",
            str(SCRIPT),
            "--input-csv",
            str(source),
            "--output-jsonl",
            str(destination),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        preexec_fn=_set_tiny_file_size_limit if limit_output_size else None,
    )


def test_prepare_manifest_preserves_single_trials_and_repeat_groups(tmp_path: Path) -> None:
    """Catches collapsing repeated presentations or changing their trial identifiers."""
    source = tmp_path / "trials.csv"
    destination = tmp_path / "manifest.jsonl"
    rows = [
        _row("trial-a0", "image-a", 0),
        _row("trial-b0", "image-b", 0, split="validation"),
        _row("trial-a1", "image-a", 1),
        _row("trial-b1", "image-b", 1, split="validation"),
    ]
    _write_csv(source, rows)

    result = _run_prepare(source, destination)

    assert result.returncode == 0, result.stderr
    records = load_jsonl_manifest(destination)
    assert [record.trial_id for record in records] == [row["trial_id"] for row in rows]
    by_image = {
        image_id: [record.repeat_id for record in records if record.image_id == image_id]
        for image_id in ("image-a", "image-b")
    }
    assert by_image == {"image-a": [0, 1], "image-b": [0, 1]}


def test_prepare_manifest_rejects_conflicting_repeat_splits(tmp_path: Path) -> None:
    """Catches writing a manifest whose repeated image identity leaks across splits."""
    source = tmp_path / "conflicting.csv"
    destination = tmp_path / "manifest.jsonl"
    _write_csv(
        source,
        [
            _row("trial-a0", "image-a", 0, split="train"),
            _row("trial-a1", "image-a", 1, split="test"),
        ],
    )

    result = _run_prepare(source, destination)

    assert result.returncode != 0
    assert "image-a" in result.stderr
    assert "train" in result.stderr
    assert "test" in result.stderr
    assert not destination.exists()


@pytest.mark.parametrize("preexisting", [False, True])
def test_prepare_manifest_write_failure_never_leaves_partial_output(
    tmp_path: Path, preexisting: bool
) -> None:
    """Catches direct writes that expose a partial file when flush or close fails."""
    source = tmp_path / "trials.csv"
    destination = tmp_path / "manifest.jsonl"
    _write_csv(source, [_row("trial-a0", "image-a", 0)])
    original = b"existing validated manifest\n"
    if preexisting:
        destination.write_bytes(original)

    result = _run_prepare(source, destination, limit_output_size=True)

    assert result.returncode != 0
    assert "File too large" in result.stderr
    if preexisting:
        assert destination.read_bytes() == original
    else:
        assert not destination.exists()
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []


def test_prepare_manifest_rejects_duplicate_header_without_output(tmp_path: Path) -> None:
    """Catches DictReader silently replacing a value under a duplicate legal header."""
    source = tmp_path / "duplicate-header.csv"
    destination = tmp_path / "manifest.jsonl"
    row = _row("trial-a0", "image-a", 0)
    source.write_text(
        ",".join((*FIELDS, "trial_id"))
        + "\n"
        + ",".join((*[row[field] for field in FIELDS], "shadow-trial"))
        + "\n",
        encoding="utf-8",
    )

    result = _run_prepare(source, destination)

    assert result.returncode != 0
    assert "duplicate CSV header" in result.stderr
    assert "trial_id" in result.stderr
    assert not destination.exists()


@pytest.mark.parametrize(
    ("values_delta", "actual_columns"),
    [
        (["unexpected"], 11),
        ([-1], 9),
    ],
)
def test_prepare_manifest_rejects_wrong_row_width_without_output(
    tmp_path: Path, values_delta: list[str | int], actual_columns: int
) -> None:
    """Catches silently dropping extra cells or obscuring a missing cell."""
    source = tmp_path / f"row-{actual_columns}.csv"
    destination = tmp_path / "manifest.jsonl"
    row = _row("trial-a0", "image-a", 0)
    values = [row[field] for field in FIELDS]
    if values_delta == [-1]:
        values = values[:-1]
    else:
        values.extend(values_delta)
    source.write_text(
        ",".join(FIELDS) + "\n" + ",".join(values) + "\n", encoding="utf-8"
    )

    result = _run_prepare(source, destination)

    assert result.returncode != 0
    assert "CSV row 2" in result.stderr
    assert f"has {actual_columns} columns; expected 10" in result.stderr
    assert not destination.exists()


def test_prepare_manifest_wraps_csv_error_without_output(tmp_path: Path) -> None:
    """Catches leaking a parser traceback or accepting malformed quoted CSV."""
    source = tmp_path / "malformed.csv"
    destination = tmp_path / "manifest.jsonl"
    source.write_text(",".join(FIELDS) + '\n"unterminated\n', encoding="utf-8")

    result = _run_prepare(source, destination)

    assert result.returncode != 0
    assert "CSV parse error" in result.stderr
    assert "unexpected end of data" in result.stderr
    assert "Traceback" not in result.stderr
    assert not destination.exists()
