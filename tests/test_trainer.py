from pathlib import Path

import pytest
import torch


def test_public_contract_uses_rollout_and_has_no_reference_forward():
    from brain_npp import PosteriorTeacherStudent, Rollout

    required = {
        "generate_student", "student_logits", "teacher_candidate_logits",
        "brain_candidate_scores", "student_supervised_logits",
    }
    assert required <= set(PosteriorTeacherStudent.__dict__)
    assert "student_reference_logits" not in PosteriorTeacherStudent.__dict__
    rollout = Rollout(torch.tensor([[1]]), torch.tensor([[True]]))
    clone = rollout.detached_clone()
    assert clone.token_ids.data_ptr() != rollout.token_ids.data_ptr()
    assert clone.token_mask.data_ptr() != rollout.token_mask.data_ptr()


class Adapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from brain_npp import Rollout

        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.rollout = Rollout(torch.tensor([[0, 1]]), torch.tensor([[True, False]]))
        self.events = []

    def generate_student(self, batch, generation_config):
        self.events.append("generate")
        return self.rollout

    def brain_candidate_scores(self, batch):
        self.events.append("brain")
        return torch.tensor([[2.0, -2.0]])

    def teacher_candidate_logits(self, batch, rollout):
        self.events.append("teacher")

        def stream():
            for probabilities in ((0.8, 0.2), (0.2, 0.8)):
                self.events.append("yield")
                yield torch.log(torch.tensor(probabilities)).reshape(1, 1, 2).expand(1, 2, 2)

        return stream()

    def student_logits(self, batch, rollout):
        self.events.append("student")
        return torch.stack((self.weight, -self.weight)).reshape(1, 1, 2).expand(1, 2, 2)

    def student_supervised_logits(self, batch, target_ids, target_mask):
        self.events.append("supervised")
        return batch["supervised_logits"] + self.weight * 0.0


def test_teacher_target_is_finished_before_single_live_student_forward():
    from brain_npp.trainer import NPPTrainer

    adapter = Adapter()
    trainer = NPPTrainer(adapter, torch.optim.SGD(adapter.parameters(), lr=0.1))
    result = trainer.step({"candidate_log_prior": torch.zeros((1, 2))})
    assert adapter.events == ["generate", "brain", "teacher", "yield", "yield", "student"]
    assert result["token_count"] == 1
    assert torch.equal(trainer.last_rollout_mask, torch.tensor([[True, False]]))


def test_ce_uses_gold_targets_only_when_enabled():
    from brain_npp.trainer import NPPTrainer

    adapter = Adapter()
    trainer = NPPTrainer(adapter, torch.optim.SGD(adapter.parameters(), lr=0.1), ce_weight=1.0)
    with pytest.raises(ValueError, match="target_ids"):
        trainer.step({})
    result = trainer.step({
        "target_ids": torch.tensor([[0, -100]]),
        "target_mask": torch.tensor([[True, False]]),
        "supervised_logits": torch.tensor([[[2.0, 0.0], [torch.nan, torch.nan]]]),
    })
    assert result["ce_token_count"] == 1
    assert "supervised" in adapter.events


def test_gradient_accumulation_and_partial_flush_are_explicit():
    from brain_npp.trainer import NPPTrainer

    adapter = Adapter()
    trainer = NPPTrainer(
        adapter,
        torch.optim.SGD(adapter.parameters(), lr=0.1),
        gradient_accumulation_steps=2,
    )
    before = adapter.weight.detach().clone()
    first = trainer.step({})
    assert first["optimizer_step"] == 0
    assert torch.equal(adapter.weight, before)
    second = trainer.step({})
    assert second["optimizer_step"] == 1
    assert not torch.equal(adapter.weight, before)
    assert trainer.flush()["optimizer_step"] == 0


def test_toy_protocol_and_arithmetic_profiles_are_migrated():
    from brain_npp import Rollout
    from brain_npp.cli import load_config
    from brain_npp.toy import ToyAdapter

    adapter = ToyAdapter(seed=7, batch_size=2)
    batch = adapter.make_batch()
    rollout = adapter.generate_student(batch, {"max_new_tokens": 2})
    assert isinstance(rollout, Rollout)
    adapter.teacher_candidate_logits(batch, rollout)
    assert adapter.prefix_consistent

    root = Path(__file__).parents[1]
    paths = [root / "configs/toy_cpu.yaml", *sorted((root / "configs/nsd").glob("*npp*.yaml"))]
    for path in paths:
        trainer = load_config(path)["trainer"]
        assert trainer["pooling"] == "arithmetic"
        assert trainer["ratio_strength"] == 1.0
        assert trainer["strength_mode"] == "constant"
        assert "alpha_scale" not in trainer and "alpha_max" not in trainer


def test_obsolete_alpha_kwargs_are_rejected():
    from brain_npp.trainer import NPPTrainer

    adapter = Adapter()
    with pytest.raises(TypeError, match="alpha_scale"):
        NPPTrainer(adapter, torch.optim.SGD(adapter.parameters()), alpha_scale=1.0)
