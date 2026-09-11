"""Deterministic content and realized-environment provenance helpers."""
from __future__ import annotations
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import torch

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()

def sha256_tree(path: str | Path) -> str:
    root = Path(path); digest = hashlib.sha256()
    excluded = {".git", "__pycache__", ".pytest_cache"}
    for item in sorted(value for value in root.rglob("*") if value.is_file() and not excluded.intersection(value.relative_to(root).parts) and value.suffix != ".pyc"):
        digest.update(item.relative_to(root).as_posix().encode()); digest.update(b"\0"); digest.update(sha256_file(item).encode()); digest.update(b"\n")
    return digest.hexdigest()

def git_revision(path: str | Path, expected: str | None = None) -> str:
    revision = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()
    if expected is not None and revision != expected: raise ValueError(f"checkout {path} revision {revision} does not match {expected}")
    return revision

def realized_environment(packages=()) -> dict:
    versions = {}
    for package in packages:
        try: versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError: versions[package] = None
    return {"python": platform.python_version(), "packages": versions, "torch_cuda": torch.version.cuda,
        "cuda_runtime": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cudnn": torch.backends.cudnn.version()}

def manifest_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
