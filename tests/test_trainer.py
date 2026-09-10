import math
from pathlib import Path

import pytest
import torch


def test_adapter_protocol_is_exported_from_the_public_package():
    """Catches making the promised adapter protocol an internal-only symbol."""
    from brain_npp import PosteriorTeacherStudent

    required_methods = {
        "generate_student",
        "student_logits",
        "student_reference_logits",
        "teacher_candidate_log_probs",
        "brain_candidate_log_likelihoods",
    }

    assert required_methods <= set(PosteriorTeacherStudent.__dict__)


class InstrumentedAdapter:
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.current_rollout = torch.tensor([[0, 1]], dtype=torch.long)
        self.generated_storage_pointer = None
        self.teacher_prefixes = []
        self.teacher_input_storage_pointer = None

    def generate_student(self, batch, generation_config):
        self.generated_storage_pointer = self.current_rollout.data_ptr()
        return self.current_rollout

    def student_logits(self, batch, rollout_ids):
        logits = torch.stack((self.weight, -self.weight)).view(1, 1, 2)
        return logits.expand(1, rollout_ids.shape[1], 2)

    def student_reference_logits(self, batch, rollout_ids):
        return torch.zeros((1, rollout_ids.shape[1], 2))

    def teacher_candidate_log_probs(self, batch, rollout_ids):
        self.teacher_input_storage_pointer = rollout_ids.data_ptr()
        for _ in range(2):
            self.teacher_prefixes.append(rollout_ids.clone())
        probabilities = torch.tensor(
            [[[[0.8, 0.2], [0.2, 0.8]], [[0.8, 0.2], [0.2, 0.8]]]]
        )
        return probabilities.log()

    def brain_candidate_log_likelihoods(self, batch):
        return torch.tensor([[2.0, -2.0]])


class SparsePosteriorAdapter:
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def generate_student(self, batch, generation_config):
        return torch.tensor([[0]], dtype=torch.long)

    def student_logits(self, batch, rollout_ids):
        return torch.stack((self.weight, -self.weight)).reshape(1, 1, 2)

    def student_reference_logits(self, batch, rollout_ids):
        return torch.zeros((1, 1, 2))

    def teacher_candidate_log_probs(self, batch, rollout_ids):
        probabilities = torch.tensor(
            [[[[0.8, 0.2], [0.2, 0.8], [0.6, 0.4], [0.4, 0.6]]]]
        )
        return probabilities.log()

    def brain_candidate_log_likelihoods(self, batch):
        return torch.tensor([[0.0, -torch.inf, 0.0, 0.0]])


class ControlledEvidenceAdapter(InstrumentedAdapter):
    def brain_candidate_log_likelihoods(self, batch):
        return batch["log_likelihood"]


def _configured_target_shift(trainer_config, *, strong_evidence):
    """Exercise config forwarding into the real trainer and target builder."""
    from brain_npp.trainer import NPPTrainer

    adapter = ControlledEvidenceAdapter()
    trainer = NPPTrainer(
        adapter, torch.optim.SGD([adapter.weight], lr=0.1), **trainer_config
    )
    batch = {
        "log_likelihood": torch.tensor([[0.0, -torch.inf]]) if strong_evidence
        else torch.tensor([[0.8, 0.2]]).log(),
        "candidate_log_prior": torch.tensor([[0.01, 0.99]]).log() if strong_evidence
        else torch.tensor([[0.5, 0.5]]).log(),
    }
    return trainer.step(batch)["target_shift"]


_ROOT = Path(__file__).parents[1]
_PRIMARY_PROFILES = ["configs/toy_cpu.yaml"] + [
    str(path.relative_to(_ROOT)) for path in sorted((_ROOT / "configs/nsd").glob("*.yaml"))
    if path.name not in {"p2_vindex_baselines.yaml", "p3_vindex_bypass_seed41.yaml"}
    and not path.name.startswith("local_")
]


@pytest.mark.parametrize("profile", [None, *_PRIMARY_PROFILES])
@pytest.mark.parametrize("strong_evidence", [False, True])
def test_primary_trainer_profiles_reach_the_scaled_bounded_target(profile, strong_evidence):
    """Catches primary profiles or the trainer silently selecting raw information gain."""
    from brain_npp.cli import load_config

    config = {} if profile is None else load_config(_ROOT / profile)
    actual_shift = _configured_target_shift(config.get("trainer", {}), strong_evidence=strong_evidence)
    # For the literal two-candidate fixture, U0-U1 = 2*(posterior0-prior0)*log(4).
    if strong_evidence:
        log_odds = 2 * 0.99 * math.log(4)  # KL=log(100)>1, bounded alpha=1.
    else:
        information_gain = 0.8 * math.log(1.6) + 0.2 * math.log(0.4)
        log_odds = 2 * 0.3 * math.log(4) * information_gain
    expected_shift = 1 / (1 + math.exp(-log_odds)) - 0.5
    assert actual_shift == pytest.approx(expected_shift, abs=1e-7)


def test_raw_strength_ablation_config_is_explicit_and_behaviorally_distinct():
    """Catches an absent raw ablation or silently clamping its requested raw strength."""
    from brain_npp.cli import load_config

    raw_path = _ROOT / "configs/ablations/toy_cpu_raw_strength.yaml"
    assert raw_path.is_file(), "the raw-strength ablation must have its own runnable profile"
    raw = load_config(raw_path)
    raw_shift = _configured_target_shift(raw["trainer"], strong_evidence=True)
    primary = load_config(_ROOT / "configs/toy_cpu.yaml")
    primary_shift = _configured_target_shift(primary["trainer"], strong_evidence=True)
    assert raw["trainer"]["alpha_max"] is None
    assert raw_shift > 0.499
    assert raw_shift > primary_shift + 0.05


def test_toy_rollout_is_bounded_and_generated_from_current_student_distribution():
    """Catches returning prebuilt batch IDs instead of an on-policy student rollout."""
    from brain_npp.toy import ToyAdapter

    adapter = ToyAdapter(seed=7, batch_size=2, time_steps=3, vocab_size=5)
    batch = adapter.make_batch()
    with torch.no_grad():
        adapter.student.weight.zero_()
        adapter.student.bias.fill_(-10.0)
        adapter.student.bias.reshape(3, 5)[:, 1] = 10.0
    first = adapter.generate_student(batch, {"max_new_tokens": 2})

    with torch.no_grad():
        adapter.student.bias.fill_(-10.0)
        adapter.student.bias.reshape(3, 5)[:, 4] = 10.0
    second = adapter.generate_student(batch, {"max_new_tokens": 2})

    assert first.shape == (2, 2)
    assert second.shape == (2, 2)
    assert torch.equal(first, torch.full((2, 2), 1))
    assert torch.equal(second, torch.full((2, 2), 4))
    assert first.grad_fn is None
    assert second.grad_fn is None


def test_toy_prefix_consistency_is_scoped_to_each_rollout_step():
    """Catches comparing a new on-policy rollout with teacher snapshots from old steps."""
    from brain_npp.toy import ToyAdapter

    adapter = ToyAdapter(seed=11, batch_size=2, time_steps=2, vocab_size=5)
    batch = adapter.make_batch()
    for token in (1, 4):
        with torch.no_grad():
            adapter.student.weight.zero_()
            adapter.student.bias.fill_(-10.0)
            adapter.student.bias.reshape(2, 5)[:, token] = 10.0
        rollout = adapter.generate_student(batch, {"max_new_tokens": 2})
        adapter.teacher_candidate_log_probs(batch, rollout.clone())

        assert adapter.prefix_consistent is True


def test_toy_teacher_cannot_see_current_or_future_tokens():
    """Catches using the predicted token, or a future token, as teacher context."""
    from brain_npp.toy import ToyAdapter

    adapter = ToyAdapter(batch_size=1, time_steps=3)
    rollout = torch.tensor([[0, 1, 2]])
    expected = adapter.teacher_candidate_log_probs({}, rollout)
    for position in range(3):
        changed = rollout.clone()
        changed[:, position:] = (changed[:, position:] + 2) % adapter.vocab_size
        actual = adapter.teacher_candidate_log_probs({}, changed)
        torch.testing.assert_close(actual[:, position], expected[:, position])


def test_toy_teacher_distribution_responds_to_earlier_prefix_tokens():
    """Catches a teacher that ignores all causal student prefix information."""
    from brain_npp.toy import ToyAdapter

    adapter = ToyAdapter(batch_size=1, time_steps=3)
    rollout = torch.tensor([[0, 1, 2]])
    expected = adapter.teacher_candidate_log_probs({}, rollout)
    changed = rollout.clone()
    changed[:, 0] = 4
    actual = adapter.teacher_candidate_log_probs({}, changed)

    torch.testing.assert_close(actual[:, 0], expected[:, 0])
    assert not torch.equal(actual[:, 1], expected[:, 1])


def test_toy_reference_matches_current_student_and_is_detached():
    """Catches using a fixed reference unrelated to the current student snapshot."""
    from brain_npp.toy import ToyAdapter

    adapter = ToyAdapter(seed=17, batch_size=2, time_steps=3, vocab_size=5)
    batch = adapter.make_batch()
    with torch.no_grad():
        adapter.student.weight.fill_(0.125)
        adapter.student.bias.copy_(torch.linspace(-1.5, 1.5, 15))
    rollout = adapter.generate_student(batch, {"max_new_tokens": 2})

    current_logits = adapter.student_logits(batch, rollout)
    reference_logits = adapter.student_reference_logits(batch, rollout)

    torch.testing.assert_close(
        reference_logits.softmax(dim=-1), current_logits.detach().softmax(dim=-1)
    )
    assert reference_logits.requires_grad is False
    assert reference_logits.grad_fn is None


def test_step_clones_one_fixed_student_prefix_for_every_candidate_teacher():
    """Catches target calls observing mutable or independently generated prefixes."""
    from brain_npp.trainer import NPPTrainer

    adapter = InstrumentedAdapter()
    trainer = NPPTrainer(
        adapter,
        torch.optim.SGD([adapter.weight], lr=0.1),
        ce_weight=0.0,
    )

    trainer.step({})
    fixed_prefix = torch.tensor([[0, 1]], dtype=torch.long)

    assert len(adapter.teacher_prefixes) == 2
    assert all(torch.equal(prefix, fixed_prefix) for prefix in adapter.teacher_prefixes)
    assert adapter.teacher_input_storage_pointer != adapter.generated_storage_pointer
    assert torch.equal(trainer.last_rollout_ids, fixed_prefix)

    adapter.current_rollout.fill_(1)
    with torch.no_grad():
        adapter.weight.add_(5.0)

    assert all(torch.equal(prefix, fixed_prefix) for prefix in adapter.teacher_prefixes)
    assert torch.equal(trainer.last_rollout_ids, fixed_prefix)


def test_sparse_neural_posterior_produces_only_finite_trainer_diagnostics():
    """Catches entropy evaluating zero-probability support as zero times negative infinity."""
    from brain_npp.trainer import NPPTrainer

    adapter = SparsePosteriorAdapter()
    trainer = NPPTrainer(
        adapter,
        torch.optim.SGD([adapter.weight], lr=0.1),
        ce_weight=0.0,
    )

    diagnostics = trainer.step({})

    assert all(math.isfinite(value) for value in diagnostics.values())


def test_toy_adapter_runs_thirty_real_optimizer_steps_and_reduces_npp_loss():
    """Catches a mocked smoke or a trainer whose NPP loss cannot optimize the student."""
    from brain_npp.toy import ToyAdapter
    from brain_npp.trainer import NPPTrainer

    torch.manual_seed(314159)
    adapter = ToyAdapter(seed=314159)
    batch = adapter.make_batch()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=0.08)
    trainer = NPPTrainer(adapter, optimizer, ce_weight=0.0, npp_weight=1.0)
    initial_parameters = [parameter.detach().clone() for parameter in adapter.parameters()]

    diagnostics = [trainer.step(batch) for _ in range(30)]
    npp_losses = [step["npp_loss"] for step in diagnostics]

    assert all(
        math.isfinite(step[key])
        for step in diagnostics
        for key in ("loss", "npp_loss", "gradient_norm")
    )
    assert sum(npp_losses[-5:]) / 5 < sum(npp_losses[:5]) / 5
    assert any(
        not torch.equal(before, after)
        for before, after in zip(initial_parameters, adapter.parameters())
    )
