"""Layered YAML experiment configuration with registration-time guards."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ConfigSource = str | Path | Mapping[str, Any]


class ConfigError(ValueError):
    """Raised when a configuration is malformed or violates a run gate."""


_BRAIN_TO_TEXT_METHODS = frozenset(
    {
        "ce",
        "scheduled_sampling",
        "off_policy_kd",
        "mwer",
        "vanilla_opsd",
        "uniform",
        "reliability",
        "teacher_contrast",
        "brain_contrast",
        "legacy_dual_cosine",
        "interaction_projection",
        "nra_opsd",
        "rlvr",
        "nra_opsd_then_rlvr",
    }
)
_VISUAL_GROUNDING_METHODS = frozenset(
    {
        "coordinate_ce",
        "smooth_l1_giou",
        "iou_rlvr",
        "evidence_iou_rlvr",
        "image_privileged_opsd",
        "image_privileged_evidence_opsd",
    }
)
_GROUNDING_REWARD_METHODS = frozenset({"iou_rlvr", "evidence_iou_rlvr"})
SUPPORTED_METHODS = _BRAIN_TO_TEXT_METHODS | _VISUAL_GROUNDING_METHODS
ALLOWED_METHODS = SUPPORTED_METHODS
SUPPORTED_MODES = frozenset({"ablation", "confirmatory", "pilot", "smoke"})
NRA_METHODS = frozenset({"interaction_projection", "nra_opsd", "nra_opsd_then_rlvr"})
_BENCHMARK_TASKS = {
    "brain_to_text_2024": "brain_to_text",
    "brain_to_text_2025": "brain_to_text",
    "brainhub": "visual_grounding",
}
_TASK_ADAPTERS = {
    "brain_to_text": "bit",
    "visual_grounding": "brainhub",
}
_TASK_METHODS = {
    "brain_to_text": _BRAIN_TO_TEXT_METHODS,
    "visual_grounding": _VISUAL_GROUNDING_METHODS,
}
_REGISTERED_ADAPTERS = frozenset(_TASK_ADAPTERS.values())
_RELIABILITY_METHODS = NRA_METHODS | {"legacy_dual_cosine", "reliability"}
_BRAIN_SUPPORT_METHODS = NRA_METHODS | {"brain_contrast", "legacy_dual_cosine"}
_TEACHER_MATCH_METHODS = NRA_METHODS | {"legacy_dual_cosine", "teacher_contrast"}
_CONTROL_BRAIN_MATCH_METHODS = NRA_METHODS | {
    "brain_contrast",
    "legacy_dual_cosine",
}
_HARDWARE_OVERLAY_KEYS = frozenset({"hardware", "resources", "training"})
_HARDWARE_RESOURCE_KEYS = frozenset(
    {"max_peak_allocation_gib", "max_peak_gib", "peak_limit_gib"}
)
_HARDWARE_TRAINING_KEYS = frozenset(
    {
        "activation_checkpointing",
        "gradient_accumulation_steps",
        "max_sequence_length",
        "micro_batch_size",
        "precision",
        "quantization_bits",
        "rollout_group_size",
    }
)
_PROFILE_ALLOCATION_CAPS_GIB = {"gpu16gb": 14.5, "gpu24gb": 22.0}


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""

    merged: dict[str, Any] = deepcopy(dict(base))
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"configuration file does not exist: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ConfigError(f"configuration {path} must contain a YAML mapping")
    return dict(payload)


def _resolve_reference(reference: str | Path, *, base_dir: Path) -> Path:
    path = Path(reference).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _load_layered_path(path: Path, *, active: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path in active:
        chain = " -> ".join(str(item) for item in (*active, path))
        raise ConfigError(f"cyclic config inheritance: {chain}")

    payload = _read_yaml(path)
    extends = payload.pop("extends", None)
    if extends is None:
        return payload

    if isinstance(extends, (str, Path)):
        references: list[str | Path] = [extends]
    elif isinstance(extends, list) and all(
        isinstance(item, (str, Path)) for item in extends
    ):
        references = extends
    else:
        raise ConfigError(f"extends in {path} must be a path or list of paths")

    merged: dict[str, Any] = {}
    for reference in references:
        parent_path = _resolve_reference(reference, base_dir=path.parent)
        parent = _load_layered_path(parent_path, active=(*active, path))
        merged = deep_merge(merged, parent)
    return deep_merge(merged, payload)


def _load_source(
    source: ConfigSource, *, base_dir: Path
) -> tuple[dict[str, Any], Path]:
    if isinstance(source, Mapping):
        return deepcopy(dict(source)), base_dir
    path = _resolve_reference(source, base_dir=base_dir)
    return _load_layered_path(path), path.parent


def _profile_source(
    profile: str | Path,
    *,
    base_dir: Path,
) -> Path:
    direct = _resolve_reference(profile, base_dir=base_dir)
    if direct.is_file():
        return direct

    profile_path = Path(profile)
    profile_name = profile_path.name
    if not profile_path.suffix:
        profile_name = f"{profile_name}.yaml"

    candidates = [Path.cwd() / "configs" / "hardware" / profile_name]
    for ancestor in (base_dir, *base_dir.parents):
        candidates.append(ancestor / "configs" / "hardware" / profile_name)
        if ancestor.name == "configs":
            candidates.append(ancestor / "hardware" / profile_name)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise ConfigError(f"hardware profile does not exist: {profile}")


def _mapping_at(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key} must be a mapping")
    return value


def _mode(config: Mapping[str, Any]) -> Any:
    mode = config.get("mode")
    if mode is None:
        mode = _mapping_at(config, "experiment").get("mode")
    return mode


def _validate_mode(config: Mapping[str, Any]) -> None:
    mode = _mode(config)
    if not isinstance(mode, str) or mode not in SUPPORTED_MODES:
        raise ConfigError(f"unknown mode {mode!r}")


def _registered_methods(config: Mapping[str, Any]) -> tuple[Any, ...]:
    registered: list[Any] = []
    if "method" in config:
        registered.append(config["method"])
    methods = config.get("methods")
    if isinstance(methods, list):
        registered.extend(methods)
    return tuple(registered)


def _validate_methods(config: Mapping[str, Any]) -> None:
    registered = _registered_methods(config)
    if "methods" in config:
        methods = config["methods"]
        if not isinstance(methods, list) or not methods:
            raise ConfigError("methods must be a non-empty list")

    if not registered:
        raise ConfigError("configuration must declare method or methods")
    for method in registered:
        if not isinstance(method, str) or method not in SUPPORTED_METHODS:
            raise ConfigError(f"unknown method {method!r}")


def _validate_task_registration(config: Mapping[str, Any]) -> None:
    experiment = _mapping_at(config, "experiment")
    declared = {
        "benchmark": "benchmark" in experiment,
        "task": "task" in config,
        "adapter": "adapter" in config,
    }
    if not any(declared.values()):
        return

    missing = [field for field, is_declared in declared.items() if not is_declared]
    if missing:
        raise ConfigError(
            "task-bound configurations must declare benchmark, task, and adapter; "
            f"missing {', '.join(missing)}"
        )

    benchmark = experiment["benchmark"]
    task = config["task"]
    adapter = _mapping_at(config, "adapter").get("name")
    if not isinstance(benchmark, str) or benchmark not in _BENCHMARK_TASKS:
        raise ConfigError(f"unknown benchmark {benchmark!r}")
    if not isinstance(task, str) or task not in _TASK_METHODS:
        raise ConfigError(f"unknown task {task!r}")
    if not isinstance(adapter, str) or adapter not in _REGISTERED_ADAPTERS:
        raise ConfigError(f"unknown adapter {adapter!r}")

    expected_task = _BENCHMARK_TASKS[benchmark]
    if task != expected_task:
        raise ConfigError(f"benchmark {benchmark!r} is incompatible with task {task!r}")

    expected_adapter = _TASK_ADAPTERS[task]
    if adapter != expected_adapter:
        raise ConfigError(f"adapter {adapter!r} is incompatible with task {task!r}")

    allowed_methods = _TASK_METHODS[task]
    for method in _registered_methods(config):
        if method not in allowed_methods:
            raise ConfigError(f"method {method!r} is incompatible with task {task!r}")


def _validate_seeds(config: Mapping[str, Any]) -> None:
    seeds = config.get("seeds")
    if seeds is None:
        experiment = _mapping_at(config, "experiment")
        seeds = experiment.get("seeds")
    valid_items = isinstance(seeds, list) and all(
        isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds
    )
    if not valid_items or len(seeds) != 3 or len(set(seeds)) != 3:
        raise ConfigError("seeds must contain exactly three distinct seeds")


def _negative_count(config: Mapping[str, Any]) -> Any:
    negative_sampling = _mapping_at(config, "negative_sampling")
    for key in ("k", "K"):
        if key in negative_sampling:
            return negative_sampling[key]
    for key in ("k", "K"):
        if key in config:
            return config[key]
    return None


def _validate_negative_sampling_keys(config: Mapping[str, Any]) -> None:
    negative_sampling = _mapping_at(config, "negative_sampling")
    if "reliability_tolerance" in negative_sampling:
        raise ConfigError(
            "negative_sampling.reliability_tolerance is unsupported; "
            "reliability matching requires the same reliability_stratum"
        )


def _validate_negative_count(config: Mapping[str, Any]) -> None:
    value = _negative_count(config)
    if value is not None and (
        not isinstance(value, int) or isinstance(value, bool) or value < 1
    ):
        raise ConfigError("negative_sampling.k must be a positive integer")

    mode = _mode(config)
    if mode == "confirmatory" and (value is None or value < 4):
        raise ConfigError("confirmatory negative_sampling.k must be at least 4")


def _validate_confirmatory_shortfalls(config: Mapping[str, Any]) -> None:
    if _mode(config) != "confirmatory":
        return
    negative_sampling = _mapping_at(config, "negative_sampling")
    if negative_sampling.get("shortfall_policy") != "skip_and_log":
        raise ConfigError(
            "confirmatory negative_sampling.shortfall_policy must be skip_and_log"
        )
    diagnostics = _mapping_at(config, "diagnostics")
    for field in ("effective_k", "negative_validity", "shortfall_reason"):
        if diagnostics.get(field) is not True:
            raise ConfigError(f"confirmatory diagnostics.{field} must be true")


def _uses_nra(config: Mapping[str, Any]) -> bool:
    return any(method in NRA_METHODS for method in _registered_methods(config))


def _uses_any(config: Mapping[str, Any], methods: frozenset[str]) -> bool:
    return any(method in methods for method in _registered_methods(config))


def _validate_grounding_reward(config: Mapping[str, Any]) -> None:
    """Validate the dense_box_reward scalar contract without loading model code."""
    if config.get("task") != "visual_grounding" or not _uses_any(
        config, _GROUNDING_REWARD_METHODS
    ):
        return

    reward = _mapping_at(config, "reward")
    fields = ("iou_threshold", "threshold_bonus", "invalid_penalty")
    missing = set(fields) - set(reward)
    if missing:
        raise ConfigError(
            f"reward is missing required fields: {', '.join(sorted(missing))}"
        )
    unknown = set(reward) - set(fields)
    if unknown:
        names = ", ".join(sorted(str(field) for field in unknown))
        raise ConfigError(f"reward has unknown fields: {names}")

    for field in fields:
        value = reward[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"reward.{field} must be a finite number")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ConfigError(f"reward.{field} must be a finite number")
        if field == "iou_threshold":
            if not 0 <= value <= 1:
                raise ConfigError("reward.iou_threshold must lie in [0, 1]")
        elif value < 0:
            raise ConfigError(f"reward.{field} must be non-negative")


def _validate_reliability_contract(config: Mapping[str, Any]) -> None:
    if not _uses_any(config, _RELIABILITY_METHODS):
        return
    reliability = _mapping_at(config, "reliability")
    if reliability.get("provenance") != "train_only_cross_fitted":
        raise ConfigError("reliability.provenance must be train_only_cross_fitted")
    if reliability.get("cross_fitted") is not True:
        raise ConfigError("reliability.cross_fitted must be true")


def _validate_brain_support_contract(config: Mapping[str, Any]) -> None:
    if not _uses_any(config, _BRAIN_SUPPORT_METHODS):
        return
    controls = _mapping_at(config, "controls")
    if controls.get("brain_only_source") != "frozen_crossfit_ce":
        raise ConfigError("controls.brain_only_source must be frozen_crossfit_ce")
    if controls.get("brain_only_provenance") != "train_only_cross_fitted":
        raise ConfigError(
            "controls.brain_only_provenance must be train_only_cross_fitted"
        )
    if controls.get("identical_prefix") is not True:
        raise ConfigError("controls.identical_prefix must be true")
    if controls.get("shared_vocabulary_support") is not True:
        raise ConfigError("controls.shared_vocabulary_support must be true")


def _validate_matching_tolerance(value: Any, *, field: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ConfigError(
            f"negative_sampling.{field} must be a non-negative finite number"
        )


def _validate_teacher_matching_contract(config: Mapping[str, Any]) -> None:
    if not _uses_any(config, _TEACHER_MATCH_METHODS):
        return

    negative_sampling = _mapping_at(config, "negative_sampling")
    for field in (
        "same_subject",
        "same_session",
        "duration_matching",
        "reliability_matching",
        "deterministic_sampler",
        "distinct_pronunciations",
    ):
        if negative_sampling.get(field) is not True:
            raise ConfigError(f"negative_sampling.{field} must be true")
    _validate_matching_tolerance(
        negative_sampling.get("duration_tolerance"), field="duration_tolerance"
    )


def _validate_control_brain_matching_contract(config: Mapping[str, Any]) -> None:
    if not _uses_any(config, _CONTROL_BRAIN_MATCH_METHODS):
        return

    controls = _mapping_at(config, "controls")
    control_brain = _mapping_at(controls, "control_brain")
    if control_brain.get("scope") != "anchor_subject_session":
        raise ConfigError("controls.control_brain.scope must be anchor_subject_session")
    for field in ("same_subject", "same_session", "use_own_pronunciation"):
        if control_brain.get(field) is not True:
            raise ConfigError(f"controls.control_brain.{field} must be true")
    distinct_from = control_brain.get("pronunciation_distinct_from")
    if (
        not isinstance(distinct_from, list)
        or len(distinct_from) != 2
        or any(not isinstance(item, str) for item in distinct_from)
        or set(distinct_from) != {"anchor", "negative"}
    ):
        raise ConfigError(
            "controls.control_brain.pronunciation_distinct_from must contain "
            "anchor and negative"
        )


def _validate_nra_controls(config: Mapping[str, Any]) -> None:
    if not _uses_nra(config):
        return

    negative_sampling = _mapping_at(config, "negative_sampling")
    if negative_sampling.get("log_effective_k") is not True:
        raise ConfigError("NRA negative_sampling.log_effective_k must be true")
    shortfall_policy = negative_sampling.get("shortfall_policy")
    if shortfall_policy not in {"error", "report_and_skip", "skip_and_log"}:
        raise ConfigError(
            "NRA negative_sampling.shortfall_policy must be error, report_and_skip, "
            "or skip_and_log"
        )
    stability = _mapping_at(config, "stability")
    if stability.get("provenance") != "train_only_cross_fitted":
        raise ConfigError("NRA stability.provenance must be train_only_cross_fitted")
    if stability.get("cross_fitted") is not True:
        raise ConfigError("NRA stability.cross_fitted must be true")

    fallback = stability.get("missing_stability_fallback")
    if isinstance(fallback, bool) or fallback != 0.0:
        raise ConfigError("NRA stability.missing_stability_fallback must equal 0.0")


def _validate_null_controls(config: Mapping[str, Any]) -> None:
    if not _uses_nra(config):
        return

    mode = _mode(config)
    controls = _mapping_at(config, "controls")
    permutations = controls.get("null_permutations")
    minimum = {"ablation": 99, "confirmatory": 999, "pilot": 99, "smoke": 1}[mode]
    if (
        not isinstance(permutations, int)
        or isinstance(permutations, bool)
        or permutations < minimum
    ):
        raise ConfigError(
            f"{mode} controls.null_permutations must be at least {minimum}"
        )

    alpha = controls.get("null_alpha")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isclose(float(alpha), 0.05, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ConfigError(f"{mode} controls.null_alpha must equal 0.05")
    if controls.get("null_target") != "final_signed_normalized_alignment":
        raise ConfigError(
            f"{mode} null calibration must target final_signed_normalized_alignment"
        )


def _first_value(*mappings_and_keys: tuple[Mapping[str, Any], tuple[str, ...]]) -> Any:
    for mapping, keys in mappings_and_keys:
        for key in keys:
            if key in mapping:
                return mapping[key]
    return None


def _as_positive_number(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ConfigError(f"{name} must be a positive finite number")
    return number


def _validate_hardware_overlay(hardware: Mapping[str, Any]) -> None:
    unexpected = set(hardware) - _HARDWARE_OVERLAY_KEYS
    if unexpected:
        names = ", ".join(sorted(str(key) for key in unexpected))
        raise ConfigError(
            "hardware profile may only contain hardware, resources, and whitelisted "
            f"training settings; found {names}"
        )

    resources = _mapping_at(hardware, "resources")
    unexpected_resources = set(resources) - _HARDWARE_RESOURCE_KEYS
    if unexpected_resources:
        names = ", ".join(sorted(str(key) for key in unexpected_resources))
        raise ConfigError(
            f"hardware resources may only declare allocation limit keys; found {names}"
        )

    training = _mapping_at(hardware, "training")
    unexpected_training = set(training) - _HARDWARE_TRAINING_KEYS
    if unexpected_training:
        names = ", ".join(sorted(str(key) for key in unexpected_training))
        raise ConfigError(
            f"hardware training may only contain execution keys; found {names}"
        )


def _validate_memory(config: Mapping[str, Any]) -> None:
    resources = _mapping_at(config, "resources")
    hardware = _mapping_at(config, "hardware")
    memory = _mapping_at(config, "memory")

    estimate = _first_value(
        (
            resources,
            (
                "estimated_peak_allocation_gib",
                "estimated_peak_gib",
                "peak_allocation_gib",
            ),
        ),
        (memory, ("estimated_peak_allocation_gib", "estimated_peak_gib")),
        (
            config,
            ("estimated_peak_allocation_gib", "estimated_peak_gib"),
        ),
    )
    limit = _first_value(
        (
            resources,
            ("max_peak_allocation_gib", "max_peak_gib", "peak_limit_gib"),
        ),
        (
            hardware,
            ("max_peak_allocation_gib", "max_peak_gib", "peak_limit_gib"),
        ),
        (memory, ("max_peak_allocation_gib", "max_peak_gib")),
    )
    total = _first_value(
        (hardware, ("total_memory_gib", "memory_gib")),
        (resources, ("total_memory_gib",)),
    )

    if estimate is None:
        raise ConfigError("estimated peak allocation is required")
    if limit is None:
        raise ConfigError("peak allocation limit is required")
    estimate_number = _as_positive_number(estimate, name="estimated peak allocation")
    limit_number = _as_positive_number(limit, name="peak allocation limit")
    total_number = _as_positive_number(total, name="total hardware memory")

    normalized_profile = None
    if "profile" in hardware:
        profile = hardware["profile"]
        if not isinstance(profile, str) or not profile.strip():
            raise ConfigError("hardware.profile must be a nonempty string")
        if profile != profile.strip():
            raise ConfigError(
                "hardware.profile must not contain leading or trailing whitespace"
            )
        # Registered profile names are case-insensitive; preserve the config spelling.
        normalized_profile = profile.lower()
    registered_cap = _PROFILE_ALLOCATION_CAPS_GIB.get(normalized_profile)
    if (
        registered_cap is not None
        and limit_number is not None
        and limit_number > registered_cap
    ):
        raise ConfigError(
            f"{normalized_profile} peak allocation limit cannot exceed registered "
            f"cap {registered_cap:.1f} GiB"
        )

    if (
        limit_number is not None
        and total_number is not None
        and limit_number > total_number
    ):
        raise ConfigError(
            f"peak allocation limit {limit_number:g} GiB exceeds total hardware "
            f"memory {total_number:g} GiB"
        )
    if (
        estimate_number is not None
        and limit_number is not None
        and estimate_number > limit_number
    ):
        raise ConfigError(
            f"estimated peak allocation {estimate_number:g} GiB exceeds "
            f"hardware limit {limit_number:g} GiB"
        )


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate registered methods, confirmatory controls, seeds, and memory."""

    if not isinstance(config, Mapping):
        raise ConfigError("configuration must be a mapping")
    _validate_mode(config)
    _validate_methods(config)
    _validate_task_registration(config)
    _validate_grounding_reward(config)
    _validate_seeds(config)
    _validate_negative_sampling_keys(config)
    _validate_negative_count(config)
    _validate_confirmatory_shortfalls(config)
    _validate_teacher_matching_contract(config)
    _validate_control_brain_matching_contract(config)
    _validate_reliability_contract(config)
    _validate_brain_support_contract(config)
    _validate_nra_controls(config)
    _validate_null_controls(config)
    _validate_memory(config)


def resolve_config(
    config: ConfigSource,
    hardware_profile: ConfigSource | None = None,
) -> dict[str, Any]:
    """Resolve config inheritance, overlay hardware, and enforce run gates.

    Merge precedence is parent config, child config, then hardware profile.  A
    caller-supplied hardware profile overrides one declared in the experiment.
    """

    resolved, config_dir = _load_source(config, base_dir=Path.cwd())
    declared_profile = resolved.pop("hardware_profile", None)
    selected_profile = (
        hardware_profile if hardware_profile is not None else declared_profile
    )
    if selected_profile is not None:
        if isinstance(selected_profile, Mapping):
            hardware = deepcopy(dict(selected_profile))
        elif isinstance(selected_profile, (str, Path)):
            profile_path = _profile_source(selected_profile, base_dir=config_dir)
            hardware = _load_layered_path(profile_path)
        else:
            raise ConfigError("hardware_profile must be a mapping or local YAML path")
        _validate_hardware_overlay(hardware)
        resolved = deep_merge(resolved, hardware)

    validate_config(resolved)
    return resolved


load_config = resolve_config
load_experiment_config = resolve_config


__all__ = [
    "ALLOWED_METHODS",
    "NRA_METHODS",
    "SUPPORTED_METHODS",
    "SUPPORTED_MODES",
    "ConfigError",
    "deep_merge",
    "load_config",
    "load_experiment_config",
    "resolve_config",
    "validate_config",
]
