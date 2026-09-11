import importlib.util
from pathlib import Path
import pytest

PATH = Path(__file__).parents[1] / "integrations/vindex/brain_npp_factory.py"
SPEC = importlib.util.spec_from_file_location("test_vindex_factory_module", PATH)
factory = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(factory)

def config():
    return {"backend":"vindex", "task":"nsd_vqa", "method":"NPP-OPSD", "candidate_count":4, "micro_batch_size":1,
        "generation":{"max_new_tokens":64}, "trainer":{"gradient_accumulation_steps":1},
        "experiment":{"phase":"P1", "subjects":["subj01"], "methods":["npp-opsd"], "controls":["correct"],
                      "max_optimizer_steps":1, "max_examples":8}}

@pytest.mark.parametrize(("change","message"), [
    (lambda c: c["experiment"].update(max_examples=0), "max_examples"),
    (lambda c: c.update(generation=[]), "generation must be a mapping"),
    (lambda c: c.update(trainer=[]), "trainer must be a mapping"),
    (lambda c: c["trainer"].update(gradient_accumulation_steps=2), "gradient_accumulation_steps"),
])
def test_scope_validation_fails_closed(change, message):
    value=config(); change(value)
    with pytest.raises(ValueError, match=message): factory._validate_scope(value)

def test_scope_validation_preserves_trainer_and_generation_options():
    value=config(); value["trainer"].update(pooling="geometric", ratio_strength=.4, strength_mode="information_gain")
    experiment, trainer, generation = factory._validate_scope(value)
    assert trainer["pooling"] == "geometric" and trainer["ratio_strength"] == .4
    assert generation == {"max_new_tokens":64} and experiment["max_examples"] == 8

def test_training_input_hash_changes_with_every_material_input(tmp_path):
    model = tmp_path / "model"; model.mkdir(); (model / "config.json").write_text("one")
    paths = {}
    for name in ("features", "vqa", "projector"):
        path = tmp_path / name; path.write_text(name); paths[name] = str(path)
    calibration = tmp_path / "calibration"; calibration.mkdir(); (calibration / "fold0.npz").write_text("fold")
    paths["model"] = str(model); paths["calibration"] = str(calibration)
    baseline, manifest = factory._training_input_hash(paths, "config")
    assert set(manifest) == {"config", "features", "vqa", "projector", "model", "calibration"}
    (model / "config.json").write_text("two")
    assert factory._training_input_hash(paths, "config")[0] != baseline

def test_factory_forwards_complete_options_to_core_trainer(monkeypatch):
    captured = {}
    class Trainer:
        def __init__(self, adapter, optimizer, **options): captured.update(options)
    monkeypatch.setattr(factory, "NPPTrainer", Trainer)
    value = config(); value["trainer"].update(pooling="geometric", ratio_strength=.4, strength_mode="information_gain")
    _, trainer, generation = factory._validate_scope(value)
    factory._create_trainer(object(), object(), trainer, generation)
    assert captured == {"generation_config":{"max_new_tokens":64}, "gradient_accumulation_steps":1,
                        "pooling":"geometric", "ratio_strength":.4, "strength_mode":"information_gain"}
