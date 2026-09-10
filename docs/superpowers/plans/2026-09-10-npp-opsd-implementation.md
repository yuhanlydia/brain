# NPP-OPSD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` to implement this plan task by
> task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the documentation-only repository into a tested research
package implementing Neural Posterior-Predictive On-Policy
Self-Distillation, NSD leakage controls, a real torch optimization smoke test,
and explicit integration contracts for VINDEX/LLaVA-7B and BrainJanus-7B.

**Architecture:** A torch-native numerical core converts image-to-brain log
likelihoods into candidate posteriors, forms posterior-versus-prior teacher
corrections at a fixed student prefix, and constructs a detached
student-anchored target. Separate modules own NSD metadata controls,
calibration/brain-dependence metrics, and training orchestration. A tiny
adapter executes the same interfaces on CPU; external 7B adapters require
local paths and fail with actionable errors rather than downloading models or
silently using toy data.

**Tech Stack:** Python 3.10+, PyTorch 2.2+, NumPy, PyYAML, pytest, setuptools.

**Spec:** `docs/superpowers/specs/2026-09-10-npp-opsd-design.md`

## Global Constraints

- Method name is exactly `NPP-OPSD` / Neural Posterior-Predictive On-Policy
  Self-Distillation.
- No KV-cache intervention, hidden-state subspace, SVD/PCA, trainable evidence
  gate, or four-cell cosine gate is part of the primary method.
- Teacher candidates and the reference student are target-side detached; only
  current student logits receive distillation gradients.
- Every candidate teacher sees the same stored student-generated prefix.
- Posterior and prior are normalized over the same valid candidate set.
- Expected teacher log probabilities are used; they must not be replaced by
  the log of an arithmetic probability mixture.
- Single-trial NSD metadata and image-identity grouped splits are mandatory.
- Real backends must use local model/data paths and never silently download
  private data or substitute the toy backend.
- The default 16 GB profile uses four candidates, micro-batch one, bounded
  generation, 4-bit frozen 7B weights, and projector/LoRA-only training.

---

### Task 1: Package bootstrap and posterior mathematics

**Files:**
- Create: `pyproject.toml`
- Create: `src/brain_npp/__init__.py`
- Create: `src/brain_npp/posterior.py`
- Test: `tests/test_posterior.py`

**Interfaces:**
- Produces `NeuralPosterior`, `normalize_candidate_prior`, and
  `build_neural_posterior` for Task 2.
- All tensor inputs use batch-first shape `[batch, candidates]`.

- [ ] **Step 1: Add a failing hand-fixture test for the candidate posterior**

```python
def test_build_neural_posterior_matches_hand_computed_fixture():
    log_likelihood = torch.log(torch.tensor([[0.8, 0.2]], dtype=torch.float64))
    log_prior = torch.log(torch.tensor([[0.5, 0.5]], dtype=torch.float64))
    result = build_neural_posterior(log_likelihood, log_prior)
    torch.testing.assert_close(result.posterior.exp(),
                               torch.tensor([[0.8, 0.2]], dtype=torch.float64))
    assert result.information_gain.item() == pytest.approx(0.1927447570)
```

- [ ] **Step 2: Run the test and verify it fails because the package/function
  does not exist**

Run: `python -m pytest tests/test_posterior.py -q`  
Expected: failure importing `brain_npp.posterior`.

- [ ] **Step 3: Implement stable masked posterior/prior normalization**

```python
@dataclass(frozen=True)
class NeuralPosterior:
    posterior: torch.Tensor      # normalized log posterior
    prior: torch.Tensor          # normalized log prior
    information_gain: torch.Tensor
    candidate_mask: torch.Tensor

def build_neural_posterior(
    log_likelihood: torch.Tensor,
    log_prior: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
) -> NeuralPosterior: ...
```

Validate rank/shape, require at least one valid candidate per row, apply
`-inf` only to padded entries, use `torch.logsumexp`, and raise `ValueError`
when prior support is absent for a candidate receiving posterior support.

- [ ] **Step 4: Add failing tests for unnormalized priors, masks, all-masked
  rows, shape mismatch, logit-shift invariance, and scores near `-10000`**

- [ ] **Step 5: Run the new tests, observe the intended validation/numerical
  failures, then implement only the behavior required for them to pass**

Run: `python -m pytest tests/test_posterior.py -q`  
Expected final result: all posterior tests pass with finite normalized values.

- [ ] **Step 6: Add package metadata and install the editable test environment**

`pyproject.toml` exposes project `brain-npp-opsd`, requires Python `>=3.10`,
depends on `torch>=2.2`, `numpy>=1.24`, and `PyYAML>=6.0`, and defines a
`test` extra containing `pytest>=8`. Use the `src/` setuptools layout.

- [ ] **Step 7: Run the posterior test from the installed package and commit**

Run: `python -m pytest tests/test_posterior.py -q`  
Commit: `feat: add neural candidate posterior core`

---

### Task 2: Posterior-contrastive target and differentiable NPP loss

**Files:**
- Create: `src/brain_npp/objective.py`
- Test: `tests/test_objective.py`

**Interfaces:**
- Consumes `NeuralPosterior` from Task 1.
- Produces `NPPTarget`, `NPPLossStats`, `build_npp_target`,
  `forward_kl_loss`, and `npp_opsd_loss` for Task 4.
- Teacher log probabilities have shape `[batch, time, candidates, vocab]`;
  student logits and token masks have shapes `[batch, time, vocab]` and
  `[batch, time]`.

- [ ] **Step 1: Write the failing exact target test**

Use the literal fixture:

```python
teacher_probs = torch.tensor([[[[0.9, 0.1], [0.2, 0.8]]]], dtype=torch.float64)
student_reference_logits = torch.log(
    torch.tensor([[[0.25, 0.75]]], dtype=torch.float64)
)
```

With posterior `(0.8, 0.2)`, prior `(0.5, 0.5)`, raw information-gain
strength, assert utility `(0.451223..., -0.623832...)`, target probabilities
approximately `(0.2908, 0.7092)`, and normalization to one.

- [ ] **Step 2: Run the test and verify the missing objective fails**

Run: `python -m pytest tests/test_objective.py -q`  
Expected: failure importing `brain_npp.objective`.

- [ ] **Step 3: Implement detached target construction**

```python
@dataclass(frozen=True)
class NPPTarget:
    log_probabilities: torch.Tensor
    utility: torch.Tensor
    strength: torch.Tensor

def build_npp_target(
    teacher_log_probs: torch.Tensor,
    student_reference_logits: torch.Tensor,
    neural_posterior: NeuralPosterior,
    *,
    alpha_scale: float = 1.0,
    alpha_max: float | None = None,
) -> NPPTarget: ...
```

Normalize teacher inputs with `log_softmax`, compute posterior and prior
expectations of log probabilities, use raw KL strength when `alpha_max=None`,
otherwise use `clamp(KL / alpha_scale, max=alpha_max)`, and run the entire
target construction under `torch.no_grad()`.

- [ ] **Step 4: Add failing tests for expected-log rather than log-mixture
  semantics, `posterior==prior` fixed point, candidate masks, batch
  consistency, bounded strength, and shape validation**

- [ ] **Step 5: Implement full-vocabulary forward KL and masked NPP loss**

```python
def forward_kl_loss(
    target_log_probs: torch.Tensor,
    student_logits: torch.Tensor,
    token_mask: torch.Tensor | None = None,
    *,
    clip_log_ratio: float | None = None,
) -> NPPLossStats: ...
```

The token mask controls numerator and denominator. Pointwise clipping clamps
the per-vocabulary log ratio before taking its target expectation. Empty masks
raise `ValueError`.

- [ ] **Step 6: Verify stop-gradient and KL direction with a failing gradient
  test before completing the loss**

For `KL(Q.detach() || softmax(z))`, assert `z.grad == softmax(z)-Q` for one
unmasked token. Also assert that teacher, posterior, and reference tensors have
no gradient while current student logits do.

- [ ] **Step 7: Run objective and posterior tests, then commit**

Run: `python -m pytest tests/test_posterior.py tests/test_objective.py -q`  
Commit: `feat: implement posterior-contrastive OPSD objective`

---

### Task 3: NSD manifest controls and evaluation metrics

**Files:**
- Create: `src/brain_npp/data.py`
- Create: `src/brain_npp/controls.py`
- Create: `src/brain_npp/metrics.py`
- Test: `tests/test_data_controls.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces `TrialRecord`, `load_jsonl_manifest`, `validate_manifest`,
  `assign_grouped_folds`, `build_block_derangement`, `apply_brain_control`,
  `neural_dependence_gap`, `brier_score`, `expected_calibration_error`,
  `topk_coverage`, and `cluster_bootstrap_difference`.
- Task 4 consumes validated records and the brain-control API.

- [ ] **Step 1: Write failing manifest tests with literal trial records**

Test that a valid pair of repeats in one fold passes and that the same
`image_id` in train and test raises a message naming that image. Test missing
single-trial fields, duplicate `trial_id`, invalid repeat IDs, and nonexistent
matched-wrong references.

- [ ] **Step 2: Run and observe the missing data module failure**

Run: `python -m pytest tests/test_data_controls.py -q`

- [ ] **Step 3: Implement immutable records, JSONL loading, and grouped folds**

```python
@dataclass(frozen=True)
class TrialRecord:
    trial_id: str
    subject_id: str
    session_id: str
    run_id: str
    trial_index: int
    image_id: str
    repeat_id: int
    split: str
    brain_path: str
    image_path: str
```

`assign_grouped_folds(records, folds, seed)` hashes/shuffles unique
`image_id`s deterministically and never separates repeats.

- [ ] **Step 4: Write failing derangement/control tests**

The matcher must be deterministic for a seed; preserve subject/session and,
when configured, run; reject self/image matches; and respect an adjacency
radius. Impossible blocks return explicit unmatched IDs instead of silently
crossing strata. Zero and Gaussian controls act on normalized brain tensors;
shuffle/matched-wrong use the precomputed mapping.

- [ ] **Step 5: Implement the matcher and brain controls, then run tests**

Run: `python -m pytest tests/test_data_controls.py -q`

- [ ] **Step 6: Write failing literal metric tests**

Cover NDG sign, multiclass Brier score, top-k coverage ties, ECE bins, and a
seeded image-cluster bootstrap whose point estimate equals the direct paired
difference. Questions/repeats sharing an `image_id` must be resampled as one
cluster.

- [ ] **Step 7: Implement metrics, run both test files, and commit**

Run: `python -m pytest tests/test_data_controls.py tests/test_metrics.py -q`  
Commit: `feat: add NSD leakage controls and neural metrics`

---

### Task 4: Adapter protocol, real torch smoke training, CLI, and configs

**Files:**
- Create: `src/brain_npp/protocols.py`
- Create: `src/brain_npp/toy.py`
- Create: `src/brain_npp/trainer.py`
- Create: `src/brain_npp/cli.py`
- Create: `src/brain_npp/adapters/__init__.py`
- Create: `src/brain_npp/adapters/external.py`
- Create: `configs/toy_cpu.yaml`
- Create: `configs/nsd/vindex_llava7b_npp.yaml`
- Create: `configs/nsd/brainjanus7b_npp.yaml`
- Create: `configs/hardware/gpu16gb.yaml`
- Create: `configs/hardware/gpu24gb.yaml`
- Test: `tests/test_trainer.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes Tasks 1-3.
- Produces the `brain-npp` command and the public adapter protocol.

- [ ] **Step 1: Write a failing trainer test proving prefix identity**

Create an instrumented toy adapter. Call one NPP step and assert every
candidate-teacher call received a value-equal clone of the student rollout
prefix and that mutating later model state does not change the stored prefix.

- [ ] **Step 2: Run and verify the trainer import/behavior failure**

Run: `python -m pytest tests/test_trainer.py -q`

- [ ] **Step 3: Implement the protocol and trainer orchestration**

```python
class PosteriorTeacherStudent(Protocol):
    def generate_student(self, batch, generation_config): ...
    def student_logits(self, batch, rollout_ids): ...
    def student_reference_logits(self, batch, rollout_ids): ...
    def teacher_candidate_log_probs(self, batch, rollout_ids): ...
    def brain_candidate_log_likelihoods(self, batch): ...
```

`NPPTrainer.step(batch)` stores/clones the on-policy rollout before any target
call, builds the posterior/target, combines CE and NPP losses, performs one
optimizer step, and returns scalar diagnostics including loss, information
gain, posterior entropy, target shift, and gradient norm.

- [ ] **Step 4: Write a failing CPU optimization smoke test**

With a fixed seed and a tiny trainable linear brain student, require 30 NPP
steps to produce finite losses and final mean NPP loss below the initial
five-step mean. This is a real optimizer test, not a mocked call-count test.

- [ ] **Step 5: Implement the deterministic toy adapter until the smoke passes**

Run: `python -m pytest tests/test_trainer.py -q`

- [ ] **Step 6: Write failing CLI tests**

`brain-npp smoke --config configs/toy_cpu.yaml` must exit zero and print JSON
with `method="NPP-OPSD"`, finite losses, and `prefix_consistent=true`.
`brain-npp validate-manifest PATH` must return nonzero with an actionable
message on split leakage. `brain-npp train --config` with backend `vindex` or
`brainjanus` and missing local paths must return nonzero and list the exact
missing keys; it must never run the toy adapter.

- [ ] **Step 7: Implement CLI/config loading and explicit external adapters**

The external adapter module validates local VINDEX/BrainJanus/model/data paths
and exposes integration commands. It raises `ExternalBackendUnavailable`
until the pinned sibling checkout supplies its factory. No model or NSD data
is downloaded by the package.

- [ ] **Step 8: Run trainer and CLI tests, then commit**

Run: `python -m pytest tests/test_trainer.py tests/test_cli.py -q`  
Commit: `feat: add executable NPP trainer and experiment profiles`

---

### Task 5: Reproduction documentation and full-repository verification

**Files:**
- Create: `README.md`
- Create: `docs/benchmark_matrix.md`
- Create: `docs/reproduction.md`
- Create: `scripts/prepare_nsd_manifest.py`
- Create: `.gitignore`
- Modify: `docs/superpowers/specs/2026-09-07-brain-evidence-posttraining-design.md`
- Test: `tests/test_prepare_manifest.py`

**Interfaces:**
- Documents and exercises all public commands created in Task 4.

- [ ] **Step 1: Write a failing end-to-end manifest preparation test**

Run the script on a temporary CSV containing two image groups and assert the
produced JSONL validates, preserves single-trial identifiers, and groups
repeats. A CSV that assigns repeats to conflicting splits must exit nonzero.

- [ ] **Step 2: Implement the script and verify the test passes**

Run: `python -m pytest tests/test_prepare_manifest.py -q`

- [ ] **Step 3: Write exact setup and execution documentation**

README and reproduction docs include commands for editable installation,
full tests, CPU smoke, manifest preparation/validation, P1 eight-example
real-shape smoke, P2 baseline reproduction, P3 ten-percent seed-41 bypass
diagnostic, and P4 seeds 41/42/43. They distinguish runnable package code from
external adapter requirements and state that no real benchmark result exists
until the external checkpoints/data have been run.

- [ ] **Step 4: Document the benchmark and ablation matrix**

The matrix includes CE, exact-image OPSD, MAP-image OPSD, uniform mixture,
plain posterior mixture, CREDIT-style, DAPD, VAD-style, and NPP-OPSD under
correct/shuffled/zero/covariance-noise/wrong-subject controls. Primary success
requires both task quality and NDG improvement.

- [ ] **Step 5: Run fresh full verification**

```bash
python -m pytest -q
brain-npp smoke --config configs/toy_cpu.yaml
python -m compileall -q src scripts
git diff --check
```

All commands must exit zero. Read the full output and record test counts and
smoke metrics before any completion claim.

- [ ] **Step 6: Commit documentation and scripts**

Commit: `docs: add NPP-OPSD reproduction workflow`

