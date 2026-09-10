"""Explicit, local-only integration boundary for VINDEX and BrainJanus."""

from __future__ import annotations

from collections.abc import Mapping
import importlib.util
from pathlib import Path
import sys
from typing import Any, Protocol, cast


class ExternalBackendUnavailable(RuntimeError):
    """Raised when an external backend cannot be created from local assets."""


class ExternalTrainingRunner(Protocol):
    """Minimal contract returned by a pinned sibling backend factory."""

    def train(self, config: Mapping[str, Any]) -> Mapping[str, Any]: ...


_CHECKOUT_KEYS = {
    "vindex": "paths.vindex_checkout",
    "brainjanus": "paths.brainjanus_checkout",
}


def required_local_path_keys(backend: str) -> tuple[str, str, str]:
    """Return the required config keys in actionable validation order."""
    try:
        checkout_key = _CHECKOUT_KEYS[backend]
    except KeyError as error:
        raise ExternalBackendUnavailable(
            f"unsupported external backend {backend!r}; expected vindex or brainjanus"
        ) from error
    return checkout_key, "paths.model", "paths.data"


def validate_external_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    """Validate that every configured external dependency already exists locally."""
    backend = config.get("backend")
    if not isinstance(backend, str):
        raise ExternalBackendUnavailable("missing required config key: backend")
    required = required_local_path_keys(backend)
    paths = config.get("paths")
    paths = paths if isinstance(paths, Mapping) else {}
    values = {
        key: paths.get(key.removeprefix("paths."))
        for key in required
    }
    missing = [
        key
        for key, value in values.items()
        if not isinstance(value, str) or not value.strip()
    ]
    if missing:
        raise ExternalBackendUnavailable(
            f"{backend}: missing required local paths: {', '.join(missing)}"
        )
    resolved = {key: Path(value).expanduser() for key, value in values.items()}
    unavailable = [key for key, path in resolved.items() if not path.exists()]
    if unavailable:
        details = ", ".join(f"{key}={resolved[key]}" for key in unavailable)
        raise ExternalBackendUnavailable(
            f"{backend}: configured local paths do not exist: {details}"
        )
    return resolved


def integration_command(backend: str, config_path: str | Path) -> tuple[str, ...]:
    """Return the package command used after local paths and a factory are supplied."""
    required_local_path_keys(backend)
    return ("brain-npp", "train", "--config", str(config_path))


def create_external_adapter(config: Mapping[str, Any]) -> ExternalTrainingRunner:
    """Load a training runner from a pinned local sibling checkout, if supplied."""
    backend = config.get("backend")
    resolved = validate_external_paths(config)
    checkout_key = _CHECKOUT_KEYS[str(backend)]
    factory_path = resolved[checkout_key] / "brain_npp_factory.py"
    if not factory_path.is_file():
        raise ExternalBackendUnavailable(
            f"{backend}: pinned sibling checkout must supply {factory_path} "
            "with create_adapter(config); no toy fallback is permitted"
        )
    specification = importlib.util.spec_from_file_location(
        f"brain_npp_external_{backend}", factory_path
    )
    if specification is None or specification.loader is None:
        raise ExternalBackendUnavailable(f"{backend}: cannot load factory {factory_path}")
    module = importlib.util.module_from_spec(specification)
    missing = object()
    previous = sys.modules.get(specification.name, missing)
    sys.modules[specification.name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        # Match normal import lifecycle, including interrupted module execution.
        if previous is missing:
            sys.modules.pop(specification.name, None)
        else:
            sys.modules[specification.name] = previous
        raise
    factory = getattr(module, "create_adapter", None)
    if not callable(factory):
        raise ExternalBackendUnavailable(
            f"{backend}: {factory_path} must define callable create_adapter(config)"
        )
    runner = factory(config)
    if not callable(getattr(runner, "train", None)):
        raise ExternalBackendUnavailable(
            f"{backend}: create_adapter(config) must return an object with train(config)"
        )
    return cast(ExternalTrainingRunner, runner)


def run_external_training(config: Mapping[str, Any]) -> dict[str, Any]:
    """Create the external runner and execute its one explicit training entrypoint."""
    backend = config.get("backend")
    result = create_external_adapter(config).train(config)
    if not isinstance(result, Mapping):
        raise ExternalBackendUnavailable(
            f"{backend}: runner train(config) must return a result mapping"
        )
    return dict(result)
