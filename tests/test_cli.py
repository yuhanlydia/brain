import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
from types import ModuleType

import pytest
import yaml


ROOT = Path(__file__).parents[1]


def run_brain_npp(*arguments: str) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("brain-npp")
    if executable is None:
        user_script = Path(sysconfig.get_path("scripts", "posix_user")) / "brain-npp"
        executable = str(user_script) if user_script.is_file() else None
    assert executable is not None, "brain-npp must be installed as a console command"
    return subprocess.run(
        [executable, *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_installed_smoke_command_runs_real_npp_optimization():
    """Catches a missing console script or a mocked/non-NPP smoke result."""
    result = run_brain_npp("smoke", "--config", "configs/toy_cpu.yaml")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["method"] == "NPP-OPSD"
    assert payload["prefix_consistent"] is True
    assert payload["steps"] == 30
    assert len(payload["npp_losses"]) == 30
    assert all(math.isfinite(loss) for loss in payload["npp_losses"])
    assert payload["final_npp_loss_mean"] < payload["initial_npp_loss_mean"]


def test_validate_manifest_reports_image_identity_split_leakage(tmp_path):
    """Catches accepting the same image identity in train and validation splits."""
    manifest = tmp_path / "leaking.jsonl"
    common = {
        "subject_id": "subj01",
        "session_id": "session01",
        "run_id": "run01",
        "trial_index": 0,
        "image_id": "shared-image",
        "repeat_id": 0,
        "brain_path": "/local/brain.npy",
        "image_path": "/local/image.png",
    }
    records = [
        {**common, "trial_id": "trial-train", "split": "train"},
        {
            **common,
            "trial_id": "trial-validation",
            "trial_index": 1,
            "split": "validation",
        },
    ]
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    result = run_brain_npp("validate-manifest", str(manifest))

    assert result.returncode != 0
    assert "shared-image" in result.stderr
    assert "train" in result.stderr
    assert "validation" in result.stderr


@pytest.mark.parametrize(
    ("backend", "missing_keys"),
    [
        ("vindex", ["paths.vindex_checkout", "paths.model", "paths.data"]),
        (
            "brainjanus",
            ["paths.brainjanus_checkout", "paths.model", "paths.data"],
        ),
    ],
)
def test_external_train_lists_exact_missing_local_path_keys(
    tmp_path, backend, missing_keys
):
    """Catches silently falling back to toy when an external backend is unavailable."""
    config = tmp_path / f"{backend}.yaml"
    config.write_text(f"backend: {backend}\npaths: {{}}\n", encoding="utf-8")

    result = run_brain_npp("train", "--config", str(config))

    assert result.returncode != 0
    assert "missing required local paths" in result.stderr
    for key in missing_keys:
        assert key in result.stderr
    assert '"method"' not in result.stdout


def test_external_train_fails_closed_when_local_factory_is_missing(tmp_path):
    """Catches bypassing the pinned local factory after path validation succeeds."""
    checkout = tmp_path / "vindex"
    model = tmp_path / "model"
    data = tmp_path / "data"
    for path in (checkout, model, data):
        path.mkdir()
    config = tmp_path / "missing-factory.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "backend": "vindex",
                "paths": {
                    "vindex_checkout": str(checkout),
                    "model": str(model),
                    "data": str(data),
                },
            }
        ),
        encoding="utf-8",
    )

    result = run_brain_npp("train", "--config", str(config))

    assert result.returncode != 0
    assert "brain_npp_factory.py" in result.stderr
    assert "no toy fallback" in result.stderr
    assert result.stdout == ""


def test_external_train_calls_runner_returned_by_local_factory(tmp_path):
    """Catches loading a valid factory and then leaving its training runner unreachable."""
    checkout = tmp_path / "vindex"
    model = tmp_path / "model"
    data = tmp_path / "data"
    for path in (checkout, model, data):
        path.mkdir()
    marker = tmp_path / "runner-called.txt"
    (checkout / "brain_npp_factory.py").write_text(
        """from pathlib import Path

class StubRunner:
    def train(self, config):
        Path(config[\"result_path\"]).write_text(\"trained\", encoding=\"utf-8\")
        return {\"backend\": config[\"backend\"], \"trained\": True}

def create_adapter(config):
    return StubRunner()
""",
        encoding="utf-8",
    )
    config = tmp_path / "working-factory.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "backend": "vindex",
                "result_path": str(marker),
                "paths": {
                    "vindex_checkout": str(checkout),
                    "model": str(model),
                    "data": str(data),
                },
            }
        ),
        encoding="utf-8",
    )

    result = run_brain_npp("train", "--config", str(config))

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"backend": "vindex", "trained": True}
    assert marker.read_text(encoding="utf-8") == "trained"


def test_external_annotated_dataclass_factory_supports_the_trainer_contract(tmp_path):
    """Catches executing a local factory before its module is registered for annotations."""
    checkout = tmp_path / "vindex"
    checkout.mkdir()
    (checkout / "brain_npp_factory.py").write_text(
        '''from __future__ import annotations
from dataclasses import dataclass
import torch
from brain_npp.toy import ToyAdapter
from brain_npp.trainer import NPPTrainer

@dataclass
class LocalContractRunner:
    backend: str

    def train(self, config):
        # Synthetic contract fixture only, never an external backend fallback.
        adapter = ToyAdapter(batch_size=1, time_steps=1, candidates=4)
        trainer = NPPTrainer(adapter, torch.optim.SGD(adapter.parameters(), lr=0.01),
                             **config["trainer"])
        batch = adapter.make_batch()
        batch["candidate_log_prior"] = torch.tensor([[-20.0, 0.0, 0.0, 0.0]])
        diagnostics = trainer.step(batch)
        return {"backend": self.backend, "target_shift": diagnostics["target_shift"]}

def create_adapter(config):
    return LocalContractRunner(config["backend"])
''', encoding="utf-8",
    )
    config = {
        "backend": "vindex",
        "paths": {"vindex_checkout": str(checkout), "model": str(checkout), "data": str(checkout)},
        "trainer": {
            "pooling": "arithmetic",
            "ratio_strength": 0.0,
            "strength_mode": "constant",
            "score_semantics": "log_likelihood",
        },
    }
    config_path = tmp_path / "dataclass.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = run_brain_npp("train", "--config", str(config_path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["backend"] == "vindex"
    assert payload["target_shift"] == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("previous_entry", [False, True])
def test_external_failed_module_execution_restores_registry(tmp_path, monkeypatch, previous_entry):
    """Catches missing pre-registration and leaking/replacing modules after exec failure."""
    from brain_npp.adapters.external import create_external_adapter

    name = "brain_npp_external_vindex"
    previous = ModuleType(name)
    if previous_entry:
        monkeypatch.setitem(sys.modules, name, previous)
    else:
        monkeypatch.delitem(sys.modules, name, raising=False)
    marker = tmp_path / "registered.txt"
    (tmp_path / "brain_npp_factory.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "assert sys.modules[__name__].__dict__ is globals()\n"
        f"Path({str(marker)!r}).write_text('registered', encoding='utf-8')\n"
        "raise RuntimeError('intentional factory execution failure')\n",
        encoding="utf-8",
    )
    config = {
        "backend": "vindex",
        "paths": {"vindex_checkout": str(tmp_path), "model": str(tmp_path), "data": str(tmp_path)},
    }

    with pytest.raises(RuntimeError, match="intentional factory execution failure"):
        create_external_adapter(config)

    assert marker.read_text(encoding="utf-8") == "registered"
    if previous_entry:
        assert sys.modules[name] is previous
    else:
        assert name not in sys.modules


def test_cli_rejects_non_finite_runner_results_instead_of_printing_nan_json(tmp_path):
    """Catches emitting non-standard JSON NaN values from training diagnostics."""
    checkout = tmp_path / "brainjanus"
    model = tmp_path / "model"
    data = tmp_path / "data"
    for path in (checkout, model, data):
        path.mkdir()
    (checkout / "brain_npp_factory.py").write_text(
        """class StubRunner:
    def train(self, config):
        return {\"loss\": float(\"nan\")}

def create_adapter(config):
    return StubRunner()
""",
        encoding="utf-8",
    )
    config = tmp_path / "non-finite.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "backend": "brainjanus",
                "paths": {
                    "brainjanus_checkout": str(checkout),
                    "model": str(model),
                    "data": str(data),
                },
            }
        ),
        encoding="utf-8",
    )

    result = run_brain_npp("train", "--config", str(config))

    assert result.returncode != 0
    assert "NaN" not in result.stdout


def test_gpu16gb_profile_has_bounded_projector_lora_only_defaults():
    """Catches a default profile that exceeds the stated 16 GB experiment envelope."""
    profile = yaml.safe_load((ROOT / "configs/hardware/gpu16gb.yaml").read_text())

    assert profile["candidate_count"] == 4
    assert profile["micro_batch_size"] == 1
    assert profile["generation"]["max_new_tokens"] == 64
    assert profile["model"] == {
        "size": "7B",
        "quantization": "4-bit",
        "frozen": True,
    }
    assert profile["trainable"] == {"projector": True, "lora": True}
