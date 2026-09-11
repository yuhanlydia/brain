import json
from pathlib import Path

import pytest
import torch

from brain_npp.nsd_runner import (
    AtomicRunState, ComputeAccounting, DAPDSnapshot, Method,
    MatchedSchedule, PredictionJournal, accumulation_windows,
    credit_negative_mapping, degrade_vad_image, method_loss, method_view_plan,
    MatrixExperiment, run_update_window, validate_brain_array, validate_control_artifact,
    select_question_ids,
    select_image_group_fraction,
)


def test_method_plan_preserves_required_views_and_names():
    assert {m.value for m in Method} == {
        "ce", "exact-image-opsd", "map-image-opsd", "uniform-mixture",
        "plain-posterior-mixture", "credit-style-image-contrastive",
        "dapd-brain-text-reference", "vad-style-full-image", "npp-opsd",
        "npp-geometric", "npp-information-gain", "npp-sqrt-information-gain",
    }
    dapd = method_view_plan(Method.DAPD)
    assert dapd.sequence_views == 10 and dapd.rollouts == 1
    assert dapd.live_views == ("rollout_none", "reference_reference", "rollout_rollout", "reference_none")
    assert len(dapd.anchor_views) == 6
    assert method_view_plan(Method.VAD).teacher_views == ("clear", "degraded")


def test_accumulation_includes_partial_final_window_with_matched_updates():
    assert list(accumulation_windows(35, 16, max_updates=3)) == [(0, 16), (16, 32), (32, 35)]
    assert list(accumulation_windows(100, 16, max_updates=2)) == [(0, 16), (16, 32)]


def test_partial_accumulation_matches_mean_batch_gradient():
    examples = [torch.tensor([1.]), torch.tensor([2.]), torch.tensor([4.])]
    a = torch.nn.Linear(1,1,bias=False); b = torch.nn.Linear(1,1,bias=False); b.load_state_dict(a.state_dict())
    oa=torch.optim.SGD(a.parameters(),lr=.1); ob=torch.optim.SGD(b.parameters(),lr=.1)
    got = run_update_window(a, oa, examples, lambda model,x:model(x).square().mean(), gradient_clip=None)
    ob.zero_grad(); torch.stack([b(x).square().mean() for x in examples]).mean().backward(); ob.step()
    assert got["micro_examples"] == 3 and torch.allclose(a.weight,b.weight)


def test_matrix_experiment_split_resume_matches_uninterrupted_next_update(tmp_path):
    class Backend:
        def __init__(self):
            self.model=torch.nn.Linear(1,1,bias=False); self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=.01)
        def loss(self, trial_id, method):
            x=torch.tensor([float(trial_id)]); return self.model(x).square().mean()
    schedule=MatchedSchedule.create(["1","2","3","4"],seed=1,max_examples=4,accumulation_steps=2,max_updates=2)
    torch.manual_seed(5); full=Backend(); initial={k:v.clone() for k,v in full.model.state_dict().items()}; MatrixExperiment(full,schedule,Method.CE,tmp_path/"full",identity={"x":1}).train()
    split=Backend(); split.model.load_state_dict(initial); first=MatchedSchedule(schedule.schema_version,schedule.seed,schedule.trial_ids,
        schedule.accumulation_steps,1,schedule.schedule_hash)
    MatrixExperiment(split,first,Method.CE,tmp_path/"split",identity={"x":1}).train()
    MatrixExperiment(split,schedule,Method.CE,tmp_path/"split",identity={"x":1}).train()
    assert torch.allclose(full.model.weight,split.model.weight)


def test_schedule_is_deterministic_and_hash_bound(tmp_path):
    schedule = MatchedSchedule.create(["t3", "t1", "t2", "t4"], seed=41,
                                      max_examples=3, accumulation_steps=2, max_updates=2)
    again = MatchedSchedule.create(["t3", "t1", "t2", "t4"], seed=41,
                                   max_examples=3, accumulation_steps=2, max_updates=2)
    assert schedule == again and len(schedule.trial_ids) == 3
    path = tmp_path / "schedule.json"; schedule.save(path)
    assert MatchedSchedule.load(path) == schedule
    payload = json.loads(path.read_text()); payload["trial_ids"][0] = "changed"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash"):
        MatchedSchedule.load(path)


def test_question_selection_is_method_independent_and_not_sidecar_first():
    available={"t1":[10,11,12],"t2":[20,21]}
    assert select_question_ids(available,seed=41)==select_question_ids(available,seed=41)
    assert select_question_ids(available,seed=41)=={"t1":11,"t2":21}


def test_p3_fraction_selects_complete_image_groups():
    selected=select_image_group_fraction({"a1":"a","a2":"a","b1":"b","c1":"c","d1":"d"},fraction=.5,seed=41)
    assert len({{"a1":"a","a2":"a","b1":"b","c1":"c","d1":"d"}[x] for x in selected})==2
    assert ("a1" in selected)==("a2" in selected)


def test_credit_negative_mapping_rejects_duplicate_image_identity():
    mapping = credit_negative_mapping(["a", "a", "b", "c"], seed=7)
    assert set(mapping) == {0, 1, 2, 3}
    assert all(["a", "a", "b", "c"][i] != ["a", "a", "b", "c"][j] for i, j in mapping.items())
    with pytest.raises(ValueError, match="identities"):
        credit_negative_mapping(["a", "a"], seed=7)


def test_vad_degradation_is_exact_rgb_bilinear_then_nearest():
    image = torch.arange(3 * 20 * 30, dtype=torch.float32).reshape(3, 20, 30) / 1800
    got = degrade_vad_image(image)
    expected = torch.nn.functional.interpolate(
        torch.nn.functional.interpolate(image[None], (2, 3), mode="bilinear", align_corners=False),
        (20, 30), mode="nearest")[0]
    assert got.shape == image.shape and torch.equal(got, expected)
    with pytest.raises(ValueError, match="RGB"):
        degrade_vad_image(image[:1])


def test_atomic_resume_restores_model_optimizer_rng_snapshot_and_cursor(tmp_path):
    torch.manual_seed(9)
    model = torch.nn.Linear(2, 1); optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    snapshot = DAPDSnapshot.capture(dict(model.named_parameters()), update_index=0)
    state = AtomicRunState(tmp_path / "checkpoint.pt", identity={"schedule": "abc", "generation": {"temperature": 1}})
    state.save(model=model, optimizer=optimizer, cursor=4, optimizer_steps=2,
               snapshot=snapshot, accounting=ComputeAccounting(student_tokens=8))
    expected_random = torch.rand(3)
    with torch.no_grad(): model.weight.add_(10)
    torch.manual_seed(123)
    restored = state.restore(model=model, optimizer=optimizer)
    assert restored.cursor == 4 and restored.optimizer_steps == 2
    assert restored.snapshot.update_index == 0
    assert restored.accounting.student_tokens == 8
    assert torch.equal(torch.rand(3), expected_random)


def test_prediction_journal_resumes_without_duplicates_and_rejects_conflicts(tmp_path):
    journal = PredictionJournal(tmp_path / "predictions.jsonl", identity={"checkpoint": "h", "control": "zero"})
    assert journal.append({"example_id": "q1", "token_ids": [1, 2], "text": "yes", "source_image_id": "i1"})
    assert not journal.append({"example_id": "q1", "token_ids": [1, 2], "text": "yes", "source_image_id": "i1"})
    with pytest.raises(ValueError, match="conflicting"):
        journal.append({"example_id": "q1", "token_ids": [3], "text": "no", "source_image_id": "i1"})
    assert PredictionJournal(tmp_path / "predictions.jsonl", identity={"checkpoint": "h", "control": "zero"}).completed_ids == {"q1"}

def test_prediction_journal_recovers_only_incomplete_terminal_record(tmp_path):
    path=tmp_path/"p.jsonl";identity={"x":1};row={"example_id":"1","token_ids":[],"text":"a","source_image_id":"i"}
    journal=PredictionJournal(path,identity=identity);journal.append(row)
    with path.open("ab") as stream:stream.write(b'{"example_id":"2"')
    assert PredictionJournal(path,identity=identity).completed_ids=={"1"}
    path.write_text(json.dumps(row))
    PredictionJournal(path,identity=identity).append({"example_id":"2","token_ids":[],"text":"b","source_image_id":"i"})
    assert len(path.read_text().splitlines())==2
    path.write_text("bad\n"+json.dumps(row)+"\n")
    with pytest.raises(ValueError,match="malformed"):PredictionJournal(path,identity=identity)


def test_brain_validation_hashes_loaded_array_bytes_not_npy_container(tmp_path):
    import hashlib, numpy as np
    path = tmp_path / "b.npy"; value = np.arange(6, dtype=np.float32); np.save(path, value)
    expected = hashlib.sha256(value.tobytes()).hexdigest()
    assert torch.equal(validate_brain_array(path, expected_sha256=expected), torch.from_numpy(value))
    with pytest.raises(ValueError, match="sha256"):
        validate_brain_array(path, expected_sha256="0" * 64)


def test_control_artifact_is_manifest_bound_and_validated_before_subset(tmp_path):
    payload = {"schema_version":1, "control":"shuffled", "manifest_sha256":"m",
               "mapping":{"a":"b"}, "unmatched_ids":["c"]}
    path = tmp_path / "map.json"; path.write_text(json.dumps(payload))
    records = {"a":("subj01","i1",0), "b":("subj01","i2",0), "c":("subj01","i3",0)}
    got = validate_control_artifact(path, expected_control="shuffled", manifest_sha256="m",
                                    records=records, selected_source_ids=["a","c"])
    assert got.mapping == {"a":"b"} and got.selected_unmatched == ("c",)
    records["b"] = ("subj02","i2",0)
    with pytest.raises(ValueError, match="subject"):
        validate_control_artifact(path, expected_control="shuffled", manifest_sha256="m",
                                  records=records, selected_source_ids=["a"])


@pytest.mark.parametrize("method", [Method.CE, Method.EXACT, Method.MAP, Method.UNIFORM,
                                     Method.POSTERIOR, Method.CREDIT, Method.DAPD, Method.VAD,
                                     Method.NPP, Method.NPP_GEOMETRIC, Method.NPP_IG, Method.NPP_SQRT_IG])
def test_every_declared_method_produces_finite_live_gradient(method):
    torch.manual_seed(4); live = torch.randn(1, 2, 5, requires_grad=True)
    mask = torch.tensor([[True, True]]); teachers = torch.softmax(torch.randn(1, 2, 4, 5), -1)
    kwargs = {"student_logits":live, "mask":mask, "gold_tokens":torch.tensor([[1, 2]]),
              "teacher_probabilities":teachers, "posterior_weights":torch.tensor([[.1,.2,.3,.4]]),
              "prior_weights":torch.full((1,4),.25), "exact_index":2,
              "positive_teacher_probabilities":teachers[:,:,0], "negative_teacher_probabilities":teachers[:,:,1],
              "clear_teacher_probabilities":teachers[:,:,0], "degraded_teacher_probabilities":teachers[:,:,1],
              "sampled_log_probs":torch.zeros(1,2), "old_log_probs":torch.zeros(1,2),
              "live_logits":{"rollout_none":live,"reference_reference":live,"rollout_rollout":live,"reference_none":live},
              "anchor_logits":{name:torch.randn(1,2,5) for name in method_view_plan(Method.DAPD).anchor_views},
              "completion_masks":{"rollout":mask,"reference":mask},
              "candidate_scores":torch.tensor([[0.,1.,2.,3.]]), "log_prior":torch.full((1,4),-1.38629436)}
    loss = method_loss(method, **kwargs)
    loss.backward()
    assert torch.isfinite(loss) and live.grad is not None and torch.isfinite(live.grad).all()
