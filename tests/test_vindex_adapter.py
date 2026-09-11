import json
from pathlib import Path

import numpy as np
import pytest
import torch
from types import SimpleNamespace


def test_causal_logits_start_at_last_prompt_position_and_match_rollout():
    from brain_npp.adapters.vindex import causal_rollout_logits

    logits = torch.arange(2 * 7 * 3, dtype=torch.float32).reshape(2, 7, 3)
    actual = causal_rollout_logits(logits, prompt_length=4, rollout_length=3)
    assert torch.equal(actual, logits[:, 3:6])


def test_rollout_mask_retains_eos_and_excludes_later_padding():
    from brain_npp.adapters.vindex import rollout_mask

    ids = torch.tensor([[7, 2, 2, 2], [7, 8, 9, 2]])
    assert torch.equal(
        rollout_mask(ids, eos_token_id=2),
        torch.tensor([[True, True, False, False], [True, True, True, True]]),
    )


def test_frozen_projector_is_an_independent_exact_snapshot():
    from brain_npp.adapters.vindex import frozen_snapshot

    student = torch.nn.Linear(3, 4)
    teacher = frozen_snapshot(student)
    before = {name: value.clone() for name, value in teacher.state_dict().items()}
    with torch.no_grad():
        student.weight.add_(1)
    assert teacher.training is False
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert all(torch.equal(before[name], value) for name, value in teacher.state_dict().items())
    assert teacher.weight.data_ptr() != student.weight.data_ptr()


def test_validated_artifacts_recompute_fold_scores_and_reject_train_image(tmp_path):
    from brain_npp.adapters.vindex import load_validated_p1_artifacts
    from brain_npp.calibration import GaussianEncodingLikelihood

    features = tmp_path / "features.npz"
    np.savez(
        features,
        image_ids=np.array(["nsd00001", "nsd00002"]),
        trial_ids=np.array(["trial-a"]),
        trial_image_ids=np.array(["nsd00001"]),
        brains=np.array([[1.0, 2.0]]),
        image_patches=np.zeros((2, 2, 3), dtype=np.float32),
        image_pooled=np.array([[1.0], [2.0]]),
        brain_features=np.zeros((1, 2, 3), dtype=np.float32),
    )
    import hashlib
    feature_hash = hashlib.sha256(features.read_bytes()).hexdigest()
    (tmp_path / "features.provenance.json").write_text(json.dumps({"artifact_sha256": feature_hash}))
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    model = GaussianEncodingLikelihood(
        coefficients=np.array([[1.0]]), intercept=np.array([0.0]),
        covariance=np.array([[1.0]]), projection=np.array([[1.0], [0.0]]),
        provenance={"train_sample_ids": ["other"], "train_image_ids": ["nsd00002"],
                    "inner_fold_by_image": {"nsd00002": 0}, "projection_seed": 1,
                    "projection_dimension": 1, "feature_provenance": "fixture",
                    "brain_provenance": "fixture",
                    "score_semantics": "normalized_gaussian_log_density_projected_brain",
                    "model_semantics": "p(projected_brain | fixed_image_features, subject)"},
    )
    model.save(calibration / "fold0.npz")
    fold_hash = hashlib.sha256((calibration / "fold0.npz").read_bytes()).hexdigest()
    np.savez(calibration / "scores.npz", trial_ids=np.array(["trial-a"]),
             candidate_ids=np.array(["nsd00001", "nsd00002"]), fold_ids=np.array([0]),
             log_likelihood=np.zeros((1, 2)))
    (calibration / "provenance.json").write_text(json.dumps({
        "input_features_sha256": feature_hash,
        "candidate_ids": ["nsd00001", "nsd00002"],
        "folds": {"0": {"sha256": fold_hash}},
    }))

    loaded = load_validated_p1_artifacts(features, calibration)
    assert loaded.log_likelihood.shape == (1, 2)
    assert not np.allclose(loaded.log_likelihood, 0)

    model.provenance["train_image_ids"] = ["nsd00001"]
    model.save(calibration / "fold0.npz")
    fold_hash = hashlib.sha256((calibration / "fold0.npz").read_bytes()).hexdigest()
    provenance = json.loads((calibration / "provenance.json").read_text())
    provenance["folds"]["0"]["sha256"] = fold_hash
    (calibration / "provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="train_image_ids"):
        load_validated_p1_artifacts(features, calibration)


class _TinyAdapterModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.embedding = torch.nn.Embedding(11, 4); self.adapter = torch.nn.Parameter(torch.ones(4)); self.disabled = False
    def get_input_embeddings(self): return self.embedding
    def disable_adapter(self):
        model = self
        class Context:
            def __enter__(self): model.disabled = True
            def __exit__(self, *args): model.disabled = False
        return Context()
    def forward(self, inputs_embeds, use_cache=False):
        logits = inputs_embeds @ self.embedding.weight.T
        if not self.disabled: logits = logits + (inputs_embeds @ self.adapter)[:, :, None]
        return SimpleNamespace(logits=logits)


def test_small_adapter_disables_live_adapter_only_for_teacher_and_keeps_full_vocab():
    from brain_npp import Rollout
    from brain_npp.adapters.vindex import VindexP1Adapter, frozen_snapshot
    model = _TinyAdapterModel(); projector = torch.nn.Linear(3, 4, bias=False)
    adapter = VindexP1Adapter(model, SimpleNamespace(eos_token_id=2), projector,
        frozen_snapshot(projector), torch.tensor([[1]]), torch.tensor([[2]]), "cpu")
    rollout = Rollout(torch.tensor([[3, 4]]), torch.ones((1, 2), dtype=torch.bool))
    batch = {"brain_features": torch.ones(1, 1, 3), "candidate_image_features": torch.ones(1, 2, 1, 3)}
    student = adapter.student_logits(batch, rollout)
    teachers = list(adapter.teacher_candidate_logits(batch, rollout))
    assert student.shape == teachers[0].shape == (1, 2, 11)
    assert len(teachers) == 2
    assert not torch.equal(student, teachers[0])
    assert model.disabled is False


def test_checkpoint_restore_matches_uninterrupted_second_adam_update(tmp_path):
    from brain_npp.adapters.vindex import load_training_state, save_training_state
    def make():
        model = torch.nn.Linear(2, 1); optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        return model, optimizer
    torch.manual_seed(8); uninterrupted, optimizer_a = make(); initial = {k: v.clone() for k, v in uninterrupted.state_dict().items()}
    split, optimizer_b = make(); split.load_state_dict(initial)
    def step(model, optimizer):
        optimizer.zero_grad(); loss = model(torch.tensor([[1.0, 2.0]])).square().mean(); loss.backward(); optimizer.step()
    step(uninterrupted, optimizer_a); step(split, optimizer_b)
    save_training_state(tmp_path / "state.pt", dict(split.named_parameters()), optimizer_b, optimizer_steps=1, input_hash="abc")
    step(uninterrupted, optimizer_a)
    restored, optimizer_c = make()
    metadata = load_training_state(tmp_path / "state.pt", dict(restored.named_parameters()), optimizer_c, expected_input_hash="abc")
    step(restored, optimizer_c)
    assert metadata["optimizer_steps"] == 1
    for expected, actual in zip(uninterrupted.parameters(), restored.parameters()): assert torch.equal(expected, actual)


def test_checkpoint_restore_rejects_changed_input_hash(tmp_path):
    from brain_npp.adapters.vindex import load_training_state, save_training_state
    model = torch.nn.Linear(1, 1); optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    save_training_state(tmp_path / "state.pt", dict(model.named_parameters()), optimizer, optimizer_steps=0, input_hash="old")
    with pytest.raises(ValueError, match="input hash"):
        load_training_state(tmp_path / "state.pt", dict(model.named_parameters()), optimizer, expected_input_hash="new")
