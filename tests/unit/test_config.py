from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from brain_evidence.experiments.config import ConfigError, resolve_config


def _write_yaml(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _valid_experiment() -> dict:
    return {
        "mode": "confirmatory",
        "method": "interaction_projection",
        "seeds": [41, 42, 43],
        "negative_sampling": {
            "k": 4,
            "same_subject": True,
            "same_session": True,
            "duration_tolerance": 0.15,
            "duration_matching": True,
            "reliability_matching": True,
            "deterministic_sampler": True,
            "distinct_pronunciations": True,
            "log_effective_k": True,
            "shortfall_policy": "skip_and_log",
        },
        "controls": {
            "control_brain": {
                "scope": "anchor_subject_session",
                "same_subject": True,
                "same_session": True,
                "use_own_pronunciation": True,
                "pronunciation_distinct_from": ["anchor", "negative"],
            },
            "null_permutations": 999,
            "null_alpha": 0.05,
            "null_target": "final_signed_normalized_alignment",
            "brain_only_source": "frozen_crossfit_ce",
            "brain_only_provenance": "train_only_cross_fitted",
            "identical_prefix": True,
            "shared_vocabulary_support": True,
        },
        "reliability": {
            "provenance": "train_only_cross_fitted",
            "cross_fitted": True,
        },
        "stability": {
            "provenance": "train_only_cross_fitted",
            "cross_fitted": True,
            "missing_stability_fallback": 0.0,
        },
        "diagnostics": {
            "effective_k": True,
            "negative_validity": True,
            "shortfall_reason": True,
        },
        "training": {"micro_batch_size": 2, "learning_rate": 0.0001},
        "resources": {
            "estimated_peak_allocation_gib": 14.0,
            "max_peak_allocation_gib": 14.5,
        },
    }


def _task_bound_experiment(
    *,
    benchmark: str = "brain_to_text_2024",
    task: str = "brain_to_text",
    adapter: str = "bit",
    method: str = "interaction_projection",
) -> dict:
    config = _valid_experiment()
    config["experiment"] = {"benchmark": benchmark}
    config["task"] = task
    config["adapter"] = {"name": adapter}
    config["method"] = method
    if task == "visual_grounding":
        config["reward"] = {
            "iou_threshold": 0.5,
            "threshold_bonus": 0.25,
            "invalid_penalty": 1.0,
        }
    return config


@pytest.fixture(params=["iou_rlvr", "evidence_iou_rlvr"])
def grounding_reward_config(request: pytest.FixtureRequest) -> dict:
    return _task_bound_experiment(
        benchmark="brainhub",
        task="visual_grounding",
        adapter="brainhub",
        method=request.param,
    )


def test_grounding_reward_methods_require_reward_registration(
    grounding_reward_config: dict,
) -> None:
    grounding_reward_config.pop("reward")

    with pytest.raises(ConfigError, match="reward"):
        resolve_config(grounding_reward_config)


@pytest.mark.parametrize("reward", [None, {}, [], True, 1.0, "iou"])
def test_grounding_reward_registration_requires_a_nonempty_mapping(
    grounding_reward_config: dict, reward: object
) -> None:
    grounding_reward_config["reward"] = reward

    with pytest.raises(ConfigError, match="reward"):
        resolve_config(grounding_reward_config)


@pytest.mark.parametrize(
    "field", ["iou_threshold", "threshold_bonus", "invalid_penalty"]
)
def test_grounding_reward_registration_requires_every_scalar(
    grounding_reward_config: dict, field: str
) -> None:
    grounding_reward_config["reward"].pop(field)

    with pytest.raises(ConfigError, match=f"reward.*{field}"):
        resolve_config(grounding_reward_config)


@pytest.mark.parametrize("field", ["threshold_bouns", "weight", 1])
def test_grounding_reward_registration_rejects_unknown_fields(
    grounding_reward_config: dict, field: object
) -> None:
    grounding_reward_config["reward"][field] = 0.5

    with pytest.raises(ConfigError, match=f"reward.*{field}"):
        resolve_config(grounding_reward_config)


@pytest.mark.parametrize(
    "field", ["iou_threshold", "threshold_bonus", "invalid_penalty"]
)
@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        "0.5",
        float("nan"),
        float("inf"),
        -float("inf"),
        -0.01,
        pytest.param(10**400, id="float-overflow"),
    ],
)
def test_grounding_reward_scalars_reject_invalid_numeric_values(
    grounding_reward_config: dict, field: str, value: object
) -> None:
    grounding_reward_config["reward"][field] = value

    with pytest.raises(ConfigError, match=f"reward.{field}"):
        resolve_config(grounding_reward_config)


@pytest.mark.parametrize("value", [1.01, 2])
def test_grounding_reward_iou_threshold_cannot_exceed_one(
    grounding_reward_config: dict, value: float
) -> None:
    grounding_reward_config["reward"]["iou_threshold"] = value

    with pytest.raises(ConfigError, match="reward.iou_threshold"):
        resolve_config(grounding_reward_config)


@pytest.mark.parametrize(
    "reward",
    [
        {"iou_threshold": 0, "threshold_bonus": 0, "invalid_penalty": 0},
        {"iou_threshold": 1, "threshold_bonus": 2, "invalid_penalty": 3},
        {"iou_threshold": 0.5, "threshold_bonus": 0.25, "invalid_penalty": 1.0},
    ],
)
def test_grounding_reward_registration_accepts_numeric_boundaries(
    grounding_reward_config: dict, reward: dict
) -> None:
    grounding_reward_config["reward"] = reward

    resolved = resolve_config(grounding_reward_config)

    assert resolved["reward"] == reward


def test_grounding_ablation_validates_reward_methods_in_the_method_list(
    grounding_reward_config: dict,
) -> None:
    method = grounding_reward_config.pop("method")
    grounding_reward_config["methods"] = ["coordinate_ce", method]
    grounding_reward_config["reward"]["iou_threshold"] = 2

    with pytest.raises(ConfigError, match="reward.iou_threshold"):
        resolve_config(grounding_reward_config)


@pytest.mark.parametrize(
    "method",
    [
        "coordinate_ce",
        "smooth_l1_giou",
        "image_privileged_opsd",
        "image_privileged_evidence_opsd",
    ],
)
def test_grounding_methods_without_reward_objectives_allow_no_reward(
    method: str,
) -> None:
    config = _task_bound_experiment(
        benchmark="brainhub", task="visual_grounding", adapter="brainhub", method=method
    )
    config.pop("reward")

    resolved = resolve_config(config)

    assert resolved["method"] == method
    assert "reward" not in resolved


@pytest.mark.parametrize("method", ["ce", "rlvr", "nra_opsd_then_rlvr"])
def test_text_methods_do_not_require_grounding_reward_registration(method: str) -> None:
    resolved = resolve_config(_task_bound_experiment(method=method))

    assert resolved["method"] == method
    assert "reward" not in resolved


def test_generic_template_does_not_require_task_specific_reward_registration() -> None:
    config = _valid_experiment()
    config["method"] = "iou_rlvr"

    resolved = resolve_config(config)

    assert resolved["method"] == "iou_rlvr"
    assert "task" not in resolved
    assert "reward" not in resolved


def test_resolve_config_layers_parent_then_child_then_hardware(tmp_path: Path) -> None:
    _write_yaml(tmp_path / "base.yaml", _valid_experiment())
    child = _write_yaml(
        tmp_path / "child.yaml",
        {
            "extends": "base.yaml",
            "experiment": {"name": "layered"},
            "training": {"max_sequence_length": 256},
        },
    )
    hardware = _write_yaml(
        tmp_path / "gpu16gb.yaml",
        {
            "hardware": {"profile": "gpu16gb", "total_memory_gib": 16},
            "training": {"micro_batch_size": 1, "precision": "bf16"},
            "resources": {"max_peak_allocation_gib": 14.5},
        },
    )

    resolved = resolve_config(child, hardware_profile=hardware)

    assert resolved["experiment"] == {"name": "layered"}
    assert resolved["negative_sampling"]["k"] == 4
    assert resolved["training"] == {
        "micro_batch_size": 1,
        "learning_rate": 0.0001,
        "max_sequence_length": 256,
        "precision": "bf16",
    }
    assert resolved["hardware"]["profile"] == "gpu16gb"
    assert "extends" not in resolved


def test_resolve_config_uses_declared_hardware_profile(tmp_path: Path) -> None:
    hardware = _write_yaml(
        tmp_path / "hardware" / "gpu16gb.yaml",
        {"resources": {"max_peak_allocation_gib": 14.5}},
    )
    config = _valid_experiment()
    config["hardware_profile"] = "hardware/gpu16gb.yaml"
    experiment = _write_yaml(tmp_path / "experiment.yaml", config)

    resolved = resolve_config(experiment)

    assert resolved["resources"]["max_peak_allocation_gib"] == 14.5
    assert "hardware_profile" not in resolved
    assert hardware.is_file()


def test_resolve_config_applies_declared_hardware_profile(tmp_path: Path) -> None:
    hardware = _write_yaml(
        tmp_path / "hardware.yaml",
        {
            "hardware": {"profile": "test-gpu"},
            "training": {"micro_batch_size": 1},
            "resources": {"max_peak_allocation_gib": 14.5},
        },
    )
    config = _valid_experiment()
    config["hardware_profile"] = hardware.name
    experiment = _write_yaml(tmp_path / "experiment.yaml", config)

    resolved = resolve_config(experiment)

    assert resolved["hardware"]["profile"] == "test-gpu"
    assert resolved["training"]["micro_batch_size"] == 1
    assert "hardware_profile" not in resolved


@pytest.mark.parametrize(
    ("forbidden_key", "value"),
    [
        ("method", "ce"),
        ("seeds", [1, 2, 3]),
        ("negative_sampling", {"k": 5}),
        ("controls", {"null_permutations": 1000}),
    ],
)
def test_hardware_profile_cannot_override_experiment_registration(
    tmp_path: Path,
    forbidden_key: str,
    value: object,
) -> None:
    config = _write_yaml(tmp_path / "experiment.yaml", _valid_experiment())
    hardware = {
        "hardware": {"profile": "test-gpu"},
        "resources": {"max_peak_allocation_gib": 14.5},
        forbidden_key: value,
    }

    with pytest.raises(
        ConfigError,
        match="hardware profile.*only.*hardware.*resources",
    ):
        resolve_config(config, hardware_profile=hardware)


def test_hardware_profile_rejects_non_execution_training_keys(tmp_path: Path) -> None:
    config = _write_yaml(tmp_path / "experiment.yaml", _valid_experiment())
    hardware = {
        "hardware": {"profile": "test-gpu"},
        "training": {"learning_rate": 1.0},
        "resources": {"max_peak_allocation_gib": 14.5},
    }

    with pytest.raises(ConfigError, match="hardware training.*learning_rate"):
        resolve_config(config, hardware_profile=hardware)


def test_hardware_profile_retains_all_whitelisted_execution_overrides(
    tmp_path: Path,
) -> None:
    config = _write_yaml(tmp_path / "experiment.yaml", _valid_experiment())
    execution_overrides = {
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "precision": "bf16",
        "quantization_bits": 4,
        "activation_checkpointing": True,
        "max_sequence_length": 512,
        "rollout_group_size": 4,
    }
    hardware = {
        "hardware": {"profile": "test-gpu"},
        "training": execution_overrides,
        "resources": {"max_peak_allocation_gib": 14.5},
    }

    resolved = resolve_config(config, hardware_profile=hardware)

    for key, value in execution_overrides.items():
        assert resolved["training"][key] == value
    assert resolved["training"]["learning_rate"] == 0.0001


def test_hardware_profile_cannot_override_experiment_memory_estimate(
    tmp_path: Path,
) -> None:
    config = _write_yaml(tmp_path / "experiment.yaml", _valid_experiment())
    hardware = {
        "hardware": {"profile": "test-gpu"},
        "resources": {
            "estimated_peak_allocation_gib": 1.0,
            "max_peak_allocation_gib": 14.5,
        },
    }

    with pytest.raises(ConfigError, match="hardware resources.*limit"):
        resolve_config(config, hardware_profile=hardware)


def test_resolve_config_rejects_unknown_method(tmp_path: Path) -> None:
    config = _valid_experiment()
    config["method"] = "copy_every_privileged_token"
    path = _write_yaml(tmp_path / "unknown.yaml", config)

    with pytest.raises(ConfigError, match="unknown method"):
        resolve_config(path)


@pytest.mark.parametrize(
    ("missing_field", "config"),
    [
        (
            "benchmark",
            {
                **_task_bound_experiment(),
                "experiment": {},
            },
        ),
        (
            "task",
            {
                key: value
                for key, value in _task_bound_experiment().items()
                if key != "task"
            },
        ),
        (
            "adapter",
            {
                key: value
                for key, value in _task_bound_experiment().items()
                if key != "adapter"
            },
        ),
    ],
)
def test_task_bound_config_requires_complete_benchmark_task_adapter(
    missing_field: str,
    config: dict,
) -> None:
    with pytest.raises(ConfigError, match=missing_field):
        resolve_config(config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("benchmark", "brainhub", "benchmark.*task"),
        ("task", "visual_grounding", "benchmark.*task"),
        ("adapter", "brainhub", "adapter.*task"),
        ("method", "evidence_iou_rlvr", "method.*task"),
    ],
)
def test_brain_to_text_rejects_cross_family_registration(
    field: str,
    value: str,
    message: str,
) -> None:
    config = _task_bound_experiment()
    if field == "benchmark":
        config["experiment"]["benchmark"] = value
    elif field == "adapter":
        config["adapter"]["name"] = value
    else:
        config[field] = value

    with pytest.raises(ConfigError, match=message):
        resolve_config(config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("benchmark", "brain_to_text_2030", "unknown benchmark"),
        ("task", "neural_captioning", "unknown task"),
        ("adapter", "generic_multimodal", "unknown adapter"),
    ],
)
def test_task_binding_rejects_unknown_registered_values(
    field: str,
    value: str,
    message: str,
) -> None:
    config = _task_bound_experiment()
    if field == "benchmark":
        config["experiment"]["benchmark"] = value
    elif field == "adapter":
        config["adapter"]["name"] = value
    else:
        config[field] = value

    with pytest.raises(ConfigError, match=message):
        resolve_config(config)


def test_brain_to_text_ablation_rejects_grounding_method() -> None:
    config = _task_bound_experiment()
    config["mode"] = "ablation"
    config.pop("method")
    config["methods"] = [
        "uniform",
        "interaction_projection",
        "evidence_iou_rlvr",
    ]

    with pytest.raises(ConfigError, match="evidence_iou_rlvr.*brain_to_text"):
        resolve_config(config)


def test_visual_grounding_rejects_brain_to_text_method() -> None:
    config = _task_bound_experiment(
        benchmark="brainhub",
        task="visual_grounding",
        adapter="brainhub",
        method="ce",
    )

    with pytest.raises(ConfigError, match="ce.*visual_grounding"):
        resolve_config(config)


@pytest.mark.parametrize(
    ("benchmark", "task", "adapter", "method"),
    [
        ("brain_to_text_2024", "brain_to_text", "bit", "ce"),
        ("brain_to_text_2025", "brain_to_text", "bit", "nra_opsd_then_rlvr"),
        ("brainhub", "visual_grounding", "brainhub", "evidence_iou_rlvr"),
    ],
)
def test_registered_task_families_accept_compatible_methods(
    benchmark: str,
    task: str,
    adapter: str,
    method: str,
) -> None:
    config = _task_bound_experiment(
        benchmark=benchmark,
        task=task,
        adapter=adapter,
        method=method,
    )

    resolved = resolve_config(config)

    assert resolved["method"] == method


def test_resolve_config_validates_every_ablation_method(tmp_path: Path) -> None:
    config = _valid_experiment()
    config.pop("method")
    config["methods"] = ["uniform", "interaction_projection", "invented_gate"]
    path = _write_yaml(tmp_path / "unknown-ablation.yaml", config)

    with pytest.raises(ConfigError, match="invented_gate"):
        resolve_config(path)


def test_config_rejects_unknown_mode(tmp_path: Path) -> None:
    config = _valid_experiment()
    config["mode"] = "unregistered_exploration"
    path = _write_yaml(tmp_path / "unknown-mode.yaml", config)

    with pytest.raises(ConfigError, match="unknown mode"):
        resolve_config(path)


def test_non_nra_method_does_not_require_nra_null_controls(tmp_path: Path) -> None:
    config = _valid_experiment()
    config["method"] = "ce"
    config.pop("controls")
    config.pop("reliability")
    config.pop("stability")
    path = _write_yaml(tmp_path / "ce.yaml", config)

    resolved = resolve_config(path)

    assert resolved["method"] == "ce"


@pytest.mark.parametrize(
    ("method", "section", "field", "bad_value", "message"),
    [
        (
            "reliability",
            "reliability",
            "provenance",
            "evaluation_fold",
            "reliability.provenance",
        ),
        (
            "brain_contrast",
            "controls",
            "brain_only_source",
            "live_student",
            "frozen_crossfit_ce",
        ),
        (
            "brain_contrast",
            "controls",
            "brain_only_provenance",
            "evaluation_fold",
            "brain_only_provenance",
        ),
        (
            "legacy_dual_cosine",
            "reliability",
            "cross_fitted",
            False,
            "reliability.cross_fitted",
        ),
        (
            "legacy_dual_cosine",
            "controls",
            "brain_only_source",
            "live_student",
            "frozen_crossfit_ce",
        ),
        (
            "interaction_projection",
            "controls",
            "brain_only_provenance",
            "evaluation_fold",
            "brain_only_provenance",
        ),
    ],
)
def test_method_specific_evidence_contracts_fail_closed(
    tmp_path: Path,
    method: str,
    section: str,
    field: str,
    bad_value: object,
    message: str,
) -> None:
    config = _valid_experiment()
    config["method"] = method
    config[section][field] = bad_value
    path = _write_yaml(tmp_path / f"bad-{method}-{field}.yaml", config)

    with pytest.raises(ConfigError, match=message):
        resolve_config(path)


@pytest.mark.parametrize(
    ("method", "field", "bad_value"),
    [
        ("teacher_contrast", "same_subject", False),
        ("teacher_contrast", "same_session", False),
        ("teacher_contrast", "duration_matching", False),
        ("teacher_contrast", "reliability_matching", False),
        ("teacher_contrast", "deterministic_sampler", False),
        ("teacher_contrast", "distinct_pronunciations", False),
        ("teacher_contrast", "duration_tolerance", -0.1),
        ("legacy_dual_cosine", "deterministic_sampler", False),
        ("interaction_projection", "reliability_matching", False),
    ],
)
def test_teacher_matching_methods_require_registered_negative_controls(
    tmp_path: Path,
    method: str,
    field: str,
    bad_value: object,
) -> None:
    config = _valid_experiment()
    config["method"] = method
    config["negative_sampling"][field] = bad_value
    path = _write_yaml(tmp_path / f"bad-{method}-{field}.yaml", config)

    with pytest.raises(ConfigError, match=field):
        resolve_config(path)


@pytest.mark.parametrize(
    "method", ["teacher_contrast", "legacy_dual_cosine", "interaction_projection"]
)
def test_matching_methods_accept_same_stratum_without_numeric_tolerance(
    method: str,
) -> None:
    config = _valid_experiment()
    config["method"] = method

    resolved = resolve_config(config)

    assert resolved["negative_sampling"]["reliability_matching"] is True
    assert "reliability_tolerance" not in resolved["negative_sampling"]


@pytest.mark.parametrize(
    "method", ["ce", "teacher_contrast", "legacy_dual_cosine", "interaction_projection"]
)
@pytest.mark.parametrize("value", [0, 0.1, 1, None, -0.1])
def test_negative_sampling_rejects_unsupported_reliability_tolerance(
    method: str, value: object
) -> None:
    config = _valid_experiment()
    config["method"] = method
    config["negative_sampling"]["reliability_tolerance"] = value

    with pytest.raises(
        ConfigError, match="reliability_tolerance.*unsupported"
    ) as error:
        resolve_config(config)

    assert "reliability_stratum" in str(error.value)


@pytest.mark.parametrize(
    ("method", "field", "bad_value"),
    [
        ("brain_contrast", "scope", "global"),
        ("brain_contrast", "same_subject", False),
        ("brain_contrast", "same_session", False),
        ("brain_contrast", "use_own_pronunciation", False),
        ("brain_contrast", "pronunciation_distinct_from", ["anchor"]),
        ("legacy_dual_cosine", "use_own_pronunciation", False),
        ("interaction_projection", "same_session", False),
    ],
)
def test_brain_matching_methods_require_anchor_local_control_brain(
    tmp_path: Path,
    method: str,
    field: str,
    bad_value: object,
) -> None:
    config = _valid_experiment()
    config["method"] = method
    config["controls"]["control_brain"][field] = bad_value
    path = _write_yaml(tmp_path / f"bad-{method}-control-{field}.yaml", config)

    with pytest.raises(ConfigError, match=field):
        resolve_config(path)


@pytest.mark.parametrize(
    "method", ["reliability", "brain_contrast", "legacy_dual_cosine"]
)
def test_simpler_evidence_methods_do_not_require_nra_only_artifacts(
    tmp_path: Path,
    method: str,
) -> None:
    config = _valid_experiment()
    config["method"] = method
    for field in ("null_permutations", "null_alpha", "null_target"):
        config["controls"].pop(field)
    config.pop("stability")
    if method == "reliability":
        config.pop("controls")
    elif method == "brain_contrast":
        config.pop("reliability")
    path = _write_yaml(tmp_path / f"minimal-{method}.yaml", config)

    resolved = resolve_config(path)

    assert resolved["method"] == method


def test_ablation_with_interaction_projection_requires_pilot_null_count(
    tmp_path: Path,
) -> None:
    config = _valid_experiment()
    config["mode"] = "ablation"
    config.pop("method")
    config["methods"] = ["uniform", "interaction_projection"]
    config["controls"]["null_permutations"] = 98
    config["negative_sampling"]["shortfall_policy"] = "report_and_skip"
    path = _write_yaml(tmp_path / "ablation-too-few-nulls.yaml", config)

    with pytest.raises(ConfigError, match="null_permutations.*at least 99"):
        resolve_config(path)


@pytest.mark.parametrize(
    ("section", "field", "message"),
    [
        ("negative_sampling", "distinct_pronunciations", "distinct_pronunciations"),
        ("negative_sampling", "log_effective_k", "log_effective_k"),
        ("reliability", "cross_fitted", "reliability.cross_fitted"),
        ("stability", "cross_fitted", "stability.cross_fitted"),
    ],
)
def test_nra_config_rejects_false_required_controls(
    tmp_path: Path,
    section: str,
    field: str,
    message: str,
) -> None:
    config = _valid_experiment()
    config[section][field] = False
    path = _write_yaml(tmp_path / f"false-{field}.yaml", config)

    with pytest.raises(ConfigError, match=message):
        resolve_config(path)


@pytest.mark.parametrize("section", ["reliability", "stability"])
def test_nra_config_requires_crossfit_provenance(tmp_path: Path, section: str) -> None:
    config = _valid_experiment()
    config[section].pop("provenance")
    path = _write_yaml(tmp_path / f"missing-{section}-provenance.yaml", config)

    with pytest.raises(ConfigError, match=f"{section}.provenance"):
        resolve_config(path)


@pytest.mark.parametrize("mode", ["confirmatory", "ablation"])
@pytest.mark.parametrize("fallback", [1.0, 0.5, -1.0, True, False, None, "0.0"])
def test_nra_config_requires_fail_closed_missing_stability_fallback(
    tmp_path: Path,
    mode: str,
    fallback: object,
) -> None:
    config = _valid_experiment()
    config["mode"] = mode
    config["stability"]["missing_stability_fallback"] = fallback
    path = _write_yaml(tmp_path / "invalid-stability-fallback.yaml", config)

    with pytest.raises(ConfigError, match="missing_stability_fallback.*0.0"):
        resolve_config(path)


@pytest.mark.parametrize("mode", ["confirmatory", "ablation"])
def test_nra_config_accepts_fail_closed_missing_stability_fallback(mode: str) -> None:
    config = _valid_experiment()
    config["mode"] = mode

    resolved = resolve_config(config)

    assert resolved["stability"]["missing_stability_fallback"] == 0.0


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("scope", "global", "anchor_subject_session"),
        ("same_subject", False, "same_subject"),
        ("same_session", False, "same_session"),
        ("use_own_pronunciation", False, "use_own_pronunciation"),
        ("pronunciation_distinct_from", ["anchor"], "pronunciation_distinct_from"),
    ],
)
def test_nra_config_requires_anchor_local_control_brain(
    tmp_path: Path,
    field: str,
    bad_value: object,
    message: str,
) -> None:
    config = _valid_experiment()
    config["controls"]["control_brain"][field] = bad_value
    path = _write_yaml(tmp_path / f"bad-control-{field}.yaml", config)

    with pytest.raises(ConfigError, match=message):
        resolve_config(path)


def test_confirmatory_config_requires_skip_and_log_shortfall_policy(
    tmp_path: Path,
) -> None:
    config = _valid_experiment()
    config["negative_sampling"]["shortfall_policy"] = "error"
    path = _write_yaml(tmp_path / "confirmatory-shortfall.yaml", config)

    with pytest.raises(ConfigError, match="shortfall_policy.*skip_and_log"):
        resolve_config(path)


@pytest.mark.parametrize(
    "diagnostic",
    ["effective_k", "negative_validity", "shortfall_reason"],
)
def test_confirmatory_config_requires_shortfall_diagnostics(
    tmp_path: Path,
    diagnostic: str,
) -> None:
    config = _valid_experiment()
    config["diagnostics"][diagnostic] = False
    path = _write_yaml(tmp_path / f"missing-{diagnostic}.yaml", config)

    with pytest.raises(ConfigError, match=f"diagnostics.{diagnostic}.*true"):
        resolve_config(path)


def test_confirmatory_config_requires_at_least_four_negatives(tmp_path: Path) -> None:
    config = _valid_experiment()
    config["negative_sampling"]["k"] = 3
    path = _write_yaml(tmp_path / "too-few-negatives.yaml", config)

    with pytest.raises(ConfigError, match="k.*at least 4"):
        resolve_config(path)


def test_confirmatory_config_requires_null_permutations(tmp_path: Path) -> None:
    config = _valid_experiment()
    config["controls"]["null_permutations"] = 998
    path = _write_yaml(tmp_path / "no-null-permutations.yaml", config)

    with pytest.raises(ConfigError, match="null_permutations.*at least 999"):
        resolve_config(path)


def test_pilot_config_requires_at_least_99_null_permutations(tmp_path: Path) -> None:
    config = _valid_experiment()
    config["mode"] = "pilot"
    config["controls"]["null_permutations"] = 98
    path = _write_yaml(tmp_path / "undersized-pilot-null.yaml", config)

    with pytest.raises(ConfigError, match="null_permutations.*at least 99"):
        resolve_config(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("null_alpha", 0.1, "null_alpha.*0.05"),
        ("null_target", "raw_cosine", "final_signed_normalized_alignment"),
        ("brain_only_source", "current_student", "frozen_crossfit_ce"),
        ("identical_prefix", False, "identical_prefix.*true"),
        ("shared_vocabulary_support", False, "shared_vocabulary_support.*true"),
    ],
)
def test_confirmatory_config_enforces_registered_null_and_control_contract(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    config = _valid_experiment()
    config["controls"][field] = value
    path = _write_yaml(tmp_path / f"bad-{field}.yaml", config)

    with pytest.raises(ConfigError, match=message):
        resolve_config(path)


@pytest.mark.parametrize("seeds", ([41, 42], [41, 41, 43]))
def test_config_requires_exactly_three_distinct_seeds(
    tmp_path: Path, seeds: list[int]
) -> None:
    config = _valid_experiment()
    config["seeds"] = seeds
    path = _write_yaml(tmp_path / "bad-seeds.yaml", config)

    with pytest.raises(ConfigError, match="exactly three distinct seeds"):
        resolve_config(path)


def test_config_rejects_estimated_memory_over_hardware_limit(tmp_path: Path) -> None:
    config = _valid_experiment()
    config["resources"]["estimated_peak_allocation_gib"] = 14.6
    experiment = _write_yaml(tmp_path / "experiment.yaml", config)
    hardware = _write_yaml(
        tmp_path / "gpu16gb.yaml",
        {
            "hardware": {"profile": "gpu16gb", "total_memory_gib": 16},
            "resources": {"max_peak_allocation_gib": 14.5},
        },
    )

    with pytest.raises(ConfigError, match="14.6.*exceeds.*14.5"):
        resolve_config(experiment, hardware_profile=hardware)


@pytest.mark.parametrize(
    ("missing_field", "message"),
    [
        ("estimated_peak_allocation_gib", "estimated peak allocation.*required"),
        ("max_peak_allocation_gib", "peak allocation limit.*required"),
    ],
)
def test_config_requires_memory_estimate_and_limit(
    tmp_path: Path,
    missing_field: str,
    message: str,
) -> None:
    config = _valid_experiment()
    config["resources"].pop(missing_field)
    path = _write_yaml(tmp_path / f"missing-{missing_field}.yaml", config)

    with pytest.raises(ConfigError, match=message):
        resolve_config(path)


@pytest.mark.parametrize(
    ("profile", "total_gib", "limit_gib", "registered_cap"),
    [
        ("gpu16gb", 16, 14.6, 14.5),
        ("GPU16GB", 16, 14.6, 14.5),
        ("Gpu16Gb", 16, 14.6, 14.5),
        ("gpu24gb", 24, 22.1, 22.0),
        ("GPU24GB", 24, 22.1, 22.0),
        ("Gpu24Gb", 24, 22.1, 22.0),
    ],
)
def test_named_hardware_profiles_cannot_exceed_registered_allocation_cap(
    tmp_path: Path,
    profile: str,
    total_gib: int,
    limit_gib: float,
    registered_cap: float,
) -> None:
    config = _valid_experiment()
    config["resources"]["estimated_peak_allocation_gib"] = 14.0
    experiment = _write_yaml(tmp_path / "experiment.yaml", config)
    hardware = {
        "hardware": {"profile": profile, "total_memory_gib": total_gib},
        "resources": {"max_peak_allocation_gib": limit_gib},
    }

    with pytest.raises(ConfigError, match=f"{profile.lower()}.*{registered_cap}"):
        resolve_config(experiment, hardware_profile=hardware)


@pytest.mark.parametrize("profile", ["gpu16gb", "gpu24gb"])
@pytest.mark.parametrize("padding", [" {}", "{} ", "\t{}\n", "\u00a0{}\u00a0"])
@pytest.mark.parametrize("source", ["inline", "overlay"])
def test_hardware_profile_whitespace_cannot_bypass_registered_cap(
    profile: str, padding: str, source: str
) -> None:
    config = _valid_experiment()
    config["resources"]["max_peak_allocation_gib"] = 99
    hardware = {"hardware": {"profile": padding.format(profile)}}
    if source == "inline":
        config.update(hardware)
        hardware = None

    with pytest.raises(ConfigError, match="hardware.profile.*leading.*trailing"):
        resolve_config(config, hardware_profile=hardware)


@pytest.mark.parametrize("profile", [None, True, False, 16, 1.5, [], {}, "", " \t\n"])
def test_explicit_hardware_profile_must_be_a_nonempty_string(profile: object) -> None:
    config = _valid_experiment()
    config["hardware"] = {"profile": profile}

    with pytest.raises(ConfigError, match="hardware.profile.*nonempty string"):
        resolve_config(config)


@pytest.mark.parametrize(
    ("profile", "cap"),
    [
        ("gpu16gb", 14.5),
        ("GPU16GB", 14.5),
        ("Gpu16Gb", 14.5),
        ("gpu24gb", 22.0),
        ("GPU24GB", 22.0),
        ("Gpu24Gb", 22.0),
    ],
)
def test_named_hardware_profiles_allow_case_variants_at_registered_cap(
    profile: str, cap: float
) -> None:
    config = _valid_experiment()
    config["hardware"] = {"profile": profile}
    config["resources"] = {
        "estimated_peak_allocation_gib": cap,
        "max_peak_allocation_gib": cap,
    }

    resolved = resolve_config(config)

    assert resolved["hardware"]["profile"] == profile
    assert resolved["resources"]["max_peak_allocation_gib"] == cap


@pytest.mark.parametrize("profile", [None, "custom-gpu", "Custom-GPU"])
def test_unregistered_or_omitted_hardware_profile_keeps_declared_memory_limits(
    profile: str | None,
) -> None:
    config = _valid_experiment()
    config["hardware"] = {"total_memory_gib": 128}
    if profile is not None:
        config["hardware"]["profile"] = profile
    config["resources"]["max_peak_allocation_gib"] = 99

    resolved = resolve_config(config)

    assert resolved["resources"]["max_peak_allocation_gib"] == 99
    config["resources"]["estimated_peak_allocation_gib"] = 100
    with pytest.raises(ConfigError, match="100.*exceeds.*99"):
        resolve_config(config)
    config["resources"]["estimated_peak_allocation_gib"] = 14
    config["resources"]["max_peak_allocation_gib"] = 129
    with pytest.raises(ConfigError, match="129.*exceeds.*128"):
        resolve_config(config)


@pytest.mark.parametrize(
    ("config_path", "hardware_path"),
    [
        ("configs/b2t24/nra_opsd.yaml", "configs/hardware/gpu16gb.yaml"),
        ("configs/b2t25/nra_opsd.yaml", "configs/hardware/gpu16gb.yaml"),
        ("configs/brainhub/umbrae_grounding.yaml", "configs/hardware/gpu24gb.yaml"),
        ("configs/experiments/ablation.yaml", "configs/hardware/gpu16gb.yaml"),
    ],
)
def test_repository_experiment_configs_resolve(
    config_path: str, hardware_path: str
) -> None:
    resolved = resolve_config(config_path, hardware_profile=hardware_path)

    assert len(resolved["seeds"]) == 3
    assert resolved["negative_sampling"]["reliability_matching"] is True
    assert resolved["stability"]["missing_stability_fallback"] == 0.0
    assert "reliability_tolerance" not in resolved["negative_sampling"]
    assert (
        resolved["resources"]["estimated_peak_allocation_gib"]
        <= resolved["resources"]["max_peak_allocation_gib"]
    )


def test_confirmatory_profile_registers_anchor_local_controls_and_provenance() -> None:
    resolved = resolve_config("configs/experiments/confirmatory.yaml")

    assert "reliability_tolerance" not in resolved["negative_sampling"]
    assert resolved["negative_sampling"]["distinct_pronunciations"] is True
    assert resolved["negative_sampling"]["log_effective_k"] is True
    assert resolved["controls"]["control_brain"]["scope"] == "anchor_subject_session"
    assert resolved["controls"]["control_brain"]["use_own_pronunciation"] is True
    assert resolved["controls"]["control_brain"]["pronunciation_distinct_from"] == [
        "anchor",
        "negative",
    ]
    assert resolved["controls"]["null_permutations"] >= 999
    assert resolved["controls"]["null_alpha"] == 0.05
    assert resolved["controls"]["null_target"] == "final_signed_normalized_alignment"
    assert resolved["controls"]["brain_only_source"] == "frozen_crossfit_ce"
    assert resolved["controls"]["identical_prefix"] is True
    assert resolved["controls"]["shared_vocabulary_support"] is True
    assert resolved["reliability"]["provenance"] == "train_only_cross_fitted"
    assert resolved["stability"]["provenance"] == "train_only_cross_fitted"
    assert resolved["stability"]["missing_stability_fallback"] == 0.0
