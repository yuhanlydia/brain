#!/usr/bin/env python3
"""Convert a single-trial NSD CSV into a validated JSONL manifest."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, fields
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence

from brain_npp.data import TrialRecord, validate_manifest


FIELD_NAMES = tuple(field.name for field in fields(TrialRecord))
INTEGER_FIELDS = ("trial_index", "repeat_id")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert an NSD single-trial CSV to validated JSONL."
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    return parser


def prepare_manifest(input_csv: Path, output_jsonl: Path) -> int:
    records = _read_csv(input_csv)
    validated = validate_manifest(records)
    _write_jsonl_atomically(validated, output_jsonl)
    return len(validated)


def _read_csv(path: Path) -> list[TrialRecord]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("input CSV must contain a header row")
        except csv.Error as error:
            raise _csv_error(reader, error) from error

        duplicates = sorted({name for name in header if header.count(name) > 1})
        if duplicates:
            raise ValueError(f"input CSV contains duplicate CSV header {duplicates[0]!r}")
        missing = [name for name in FIELD_NAMES if name not in header]
        if missing:
            raise ValueError(f"input CSV is missing required column {missing[0]!r}")
        unexpected = [name for name in header if name not in FIELD_NAMES]
        if unexpected:
            raise ValueError(f"input CSV contains unexpected column {unexpected[0]!r}")

        records: list[TrialRecord] = []
        try:
            for row_number, values in enumerate(reader, start=2):
                if len(values) != len(header):
                    raise ValueError(
                        f"CSV row {row_number} has {len(values)} columns; "
                        f"expected {len(header)}"
                    )
                payload = dict(zip(header, values))
                for name in INTEGER_FIELDS:
                    try:
                        payload[name] = int(payload[name])
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"CSV row {row_number} column {name!r} must be an integer"
                        ) from error
                records.append(TrialRecord(**payload))
        except csv.Error as error:
            raise _csv_error(reader, error) from error
    return records


def _csv_error(reader: csv.reader, error: csv.Error) -> ValueError:
    return ValueError(f"CSV parse error at row {reader.line_num}: {error}")


def _write_jsonl_atomically(
    records: Sequence[TrialRecord], output_jsonl: Path
) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=output_jsonl.parent,
            prefix=f".{output_jsonl.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            for record in records:
                stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output_jsonl)
        temporary_path = None
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        count = prepare_manifest(arguments.input_csv, arguments.output_jsonl)
    except (OSError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(arguments.output_jsonl), "trials": count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
