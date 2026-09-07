# Neural-Recoverability-Aware OPSD Implementation Plan

> **For implementers:** Follow `superpowers:test-driven-development`. Add each behavioral test, run it to observe the expected missing-feature failure, add the smallest implementation, and rerun the focused test before the full suite.

**Goal:** Deliver a tested, CPU-runnable research package and reproducible experiment matrix for vanilla OPSD, legacy evidence gating, and Neural-Recoverability-Aware OPSD (NRA-OPSD), with explicit hooks for Brain-to-Text and UMBRAE.

**Architecture:** Pure tensor functions implement Fisher-centered vocabulary geometry, paired 2x2 brain--phoneme interactions, multi-negative aggregation, reliability/stability, and one-sided target projection. Dataset controls, objectives, metrics, and experiment orchestration are separate modules. A deterministic synthetic experiment exercises the entire path without private data; real adapters validate manifests and expose typed batch contracts without vendoring external repositories.

**Tech Stack:** Python 3.10+, PyTorch, PyYAML, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-07-brain-evidence-posttraining-design.md`

**Current boundary (2026-09-07):** this plan has delivered a research skeleton with pure-tensor functions, executable contracts, task-aware config dry-runs, manifest validation, and deterministic CPU synthetic method smoke. Toy overfit and checkpoint-resume equivalence remain unimplemented, so P0 is only partial. Real dataset/model runners, BIT/UMBRAE wiring, 7B/other decoder integration, trainers, Hungarian evaluation, and bootstrap inference remain planned and unimplemented. `SUPPORTED_METHODS` is a config registry, not a trainer registry; all three real benchmarks remain `NOT RUN`.

## Global constraints

- At the implemented tensor boundary, supplied teacher distributions, interactions, reliability, stability, and constructed targets are detached; real orchestration must still prove upstream artifact isolation.
- The student never consumes privileged phonemes at inference.
- Main comparisons use identical vocabulary support, forward/token budgets, negative sampler, and seeds.
- Real loaders first build pools inside official splits; matched-negative, negative-pronunciation source, and control trials stay within split/fold/subject/session/`reliability_stratum`, differ as required in pronunciation, and satisfy registered duration tolerance.
- Synthetic runs demonstrate code execution only; documentation must not present them as benchmark results.
- B2T24, B2T25, and BrainHub/UMBRAE remain `NOT RUN` until real-data gates execute; synthetic checks support no benchmark or SOTA claim.
- External BIT/UMBRAE checkpoints and datasets remain user-supplied, pinned inputs.
- Default experiments must resolve on CPU and enforce the declared 14.5 GiB allocation gate for the 16 GB profile.

## Task 1: Package foundation and recoverability geometry

**Files:**

- Create: `pyproject.toml`
- Create: `src/brain_evidence/__init__.py`
- Create: `src/brain_evidence/recoverability/__init__.py`
- Create: `src/brain_evidence/recoverability/geometry.py`
- Create: `src/brain_evidence/recoverability/interaction.py`
- Create: `src/brain_evidence/recoverability/projection.py`
- Test: `tests/unit/test_geometry.py`
- Test: `tests/unit/test_interaction.py`
- Test: `tests/unit/test_projection.py`

**Behavior:**

- `fisher_center(values, probabilities)` returns zero probability-weighted mean and validates shapes/probability mass.
- `fisher_inner` and `fisher_norm_sq` operate on the last vocabulary axis.
- `anchor_local_interaction(xi_pi, xi_pj, xc_pi, xc_pj, student_probs)` computes the centered difference-in-differences where only `xi_pi` is matched; raw-null subtraction remains a named lower-level ablation only and is rejected by the NRA path.
- `aggregate_interactions` supports deterministic mean and median aggregation plus optional normalized non-negative weights, then always re-centers under the current student distribution.
- `recoverable_target` requires a typed `NRAEvidence` plus `NRAInteractionBatch` in NRA mode, checks their complete current context/statistic identity and verifiable uniform weights, and requires runtime `epsilon` to exactly match the artifact's calibrated `statistic_epsilon`, then projects interaction onto a detached brain-only contrast supplied by a frozen cross-fitted CE checkpoint. NRA callers must explicitly supply reliability and session stability; these remain upstream-validated raw detached values rather than typed gate artifacts. The constructor validates student log-probability mass via vocabulary-aware `logsumexp`, explicitly normalizes accepted values before Fisher arithmetic (including large-vocabulary BF16), calibrates signed Fisher-cosine alignment against the exact-statistic permutation threshold, projects the full privileged correction onto the retained direction, applies reliability/stability/null-excess and span-preserving scalar clipping, and returns a normalized detached target plus detached diagnostics. `target_kl` is clamped non-negative against numeric roundoff. `geometric_coverage` records interaction/projection/`kappa` support; `transfer_coverage` additionally requires positive reliability/stability, and legacy `coverage` aliases the latter. Raw tensors, separate weights, and raw-null subtraction are explicit interaction-only ablations. This local detachment contract is not an end-to-end leakage-proof claim for the unimplemented real orchestration.

**TDD sequence:**

1. Add centering/inner-product tests; run `pytest tests/unit/test_geometry.py -q` and observe import failure.
2. Implement geometry; rerun focused tests.
3. Add additive pronunciation-copy cancellation, `x`-modulated-copy null subtraction, anchor-only matching, control-only matching rejection, additive-logit invariance, and matched-specific-effect tests; run `pytest tests/unit/test_interaction.py -q` and observe import failure.
4. Implement interaction and aggregation; rerun focused tests.
5. Add aligned, brain-only-orthogonal, opposite, zero-interaction, clipping, normalization, nonuniform-reference, and full target-side gradient-detachment tests; run `pytest tests/unit/test_projection.py -q` and observe import failure.
6. Implement projection target and diagnostics; rerun the three files together.

## Task 2: Matched controls, reliability, and session stability

**Files:**

- Create: `src/brain_evidence/controls/__init__.py`
- Create: `src/brain_evidence/controls/matched_negative.py`
- Create: `src/brain_evidence/gates/__init__.py`
- Create: `src/brain_evidence/gates/reliability.py`
- Create: `src/brain_evidence/gates/counterfactual.py`
- Create: `src/brain_evidence/recoverability/stability.py`
- Test: `tests/unit/test_matched_negative.py`
- Test: `tests/unit/test_reliability.py`
- Test: `tests/unit/test_stability.py`
- Test: `tests/unit/test_legacy_gate.py`

**Behavior:**

- Seeded anchor-local control sampling returns `K` pairs of distinct negative pronunciations and control brains from the same split/fold/subject/session/`reliability_stratum`; each control's true pronunciation differs from both supplied pronunciations, duration tolerance is honored, and shortfall is reported rather than silently crossing constraints. Real loaders must construct candidate pools within official splits before sampling.
- Reliability maps neural posterior confidence and boolean/finite-binary validity masks into detached `[0,1]` weights with an explicit calibration temperature; calibration provenance binds the current pronunciation identity exactly.
- Cross-session stability compares a current interaction with raw same-pronunciation contrasts whose entries carry split/fold/session/trial/checkpoint/prefix/alignment/support provenance, re-centers both under one reference distribution, enforces training-only leave-current-session-out lookup, maps positive cosine to `[0,1]`, and in primary/confirmatory use returns `0` plus missing-support flags when no compatible support exists. A `1.0` fallback is permitted only as an explicit ablation.
- The legacy dual-cosine gate uses centered Fisher geometry and remains a detached scalar ablation.

**TDD sequence:** add one focused test file at a time, observe the missing import/function failure, implement only its module, and rerun it. Finish with `pytest tests/unit/test_matched_negative.py tests/unit/test_reliability.py tests/unit/test_stability.py tests/unit/test_legacy_gate.py -q`.

## Task 3: OPSD objectives, rewards, and official metrics

**Files:**

- Create: `src/brain_evidence/objectives/__init__.py`
- Create: `src/brain_evidence/objectives/opsd.py`
- Create: `src/brain_evidence/objectives/rewards.py`
- Create: `src/brain_evidence/metrics/__init__.py`
- Create: `src/brain_evidence/metrics/text.py`
- Create: `src/brain_evidence/metrics/grounding.py`
- Create: `src/brain_evidence/metrics/brain_dependence.py`
- Test: `tests/unit/test_opsd.py`
- Test: `tests/unit/test_rewards.py`
- Test: `tests/unit/test_text_metrics.py`
- Test: `tests/unit/test_grounding_metrics.py`
- Test: `tests/unit/test_brain_dependence.py`

**Behavior:**

- Forward KL accepts token masks and optional pointwise clipping, detaches its target, and validates active student/target distributions before arithmetic.
- Uniform weighted OPSD is numerically identical to vanilla OPSD.
- WER exposes substitutions/deletions/insertions and normalized score for empty/non-empty references.
- Box parsing and IoU reject malformed/degenerate coordinates; dense reward combines IoU, threshold bonus, and invalid penalty.
- Group-relative reward standardization marks equal-reward groups as skipped and reports the skip rate.
- Brain dependence reports real-minus-wrong gap and difference-in-gap against CE.

**TDD sequence:** create and run each focused test before its production module, then run `pytest tests/unit/test_opsd.py tests/unit/test_rewards.py tests/unit/test_text_metrics.py tests/unit/test_grounding_metrics.py tests/unit/test_brain_dependence.py -q`.

## Task 4: Typed adapters and experiment configuration

**Files:**

- Create: `src/brain_evidence/protocols.py`
- Create: `src/brain_evidence/adapters/__init__.py`
- Create: `src/brain_evidence/adapters/bit.py`
- Create: `src/brain_evidence/adapters/brainhub.py`
- Create: `src/brain_evidence/experiments/__init__.py`
- Create: `src/brain_evidence/experiments/config.py`
- Create: `configs/hardware/gpu16gb.yaml`
- Create: `configs/hardware/gpu24gb.yaml`
- Create: `configs/experiments/ablation.yaml`
- Create: `configs/experiments/confirmatory.yaml`
- Create: `configs/b2t24/nra_opsd.yaml`
- Create: `configs/b2t25/nra_opsd.yaml`
- Create: `configs/brainhub/umbrae_grounding.yaml`
- Test: `tests/unit/test_config.py`
- Test: `tests/unit/test_adapters.py`

**Behavior:**

- Protocol dataclasses document the brain-only student and brain-plus-privilege teacher contract.
- BIT and BrainHub adapters validate tensor shapes and external data/checkpoint manifests; missing user data fails with an actionable message.
- Layered YAML loading applies hardware overrides, validates compatible benchmark/task/adapter/method families and the dense grounding-reward scalar schema, rejects unknown methods, enforces `K>=4`, distinct pronunciations, anchor-local controls, frozen-cross-fitted CE brain support, `alpha=0.05`, at least 99 exact-statistic null permutations in pilot mode and 999 in confirmatory mode, validates three distinct seeds, and enforces peak-allocation limits. Registered methods remain planned config names unless an execution path is separately implemented.

**TDD sequence:** write adapter/config tests, observe missing-module failures, implement typed validation and config resolution, then run both focused files.

## Task 5: Deterministic end-to-end smoke experiment and CLIs

**Files:**

- Create: `src/brain_evidence/experiments/synthetic.py`
- Create: `src/brain_evidence/cli/__init__.py`
- Create: `src/brain_evidence/cli/train.py`
- Create: `src/brain_evidence/cli/evaluate.py`
- Create: `scripts/smoke_test.py`
- Create: `scripts/prepare_b2t.py`
- Create: `scripts/prepare_brainhub.py`
- Test: `tests/integration/test_synthetic_experiment.py`
- Test: `tests/integration/test_cli.py`

**Behavior:**

- A tiny frozen teacher and trainable student run CE, vanilla OPSD, legacy dual cosine, NRA-OPSD, and group-relative reward steps on CPU with finite losses. All synthetic OPSD arms use the current detached brain-only greedy rollout; legacy weights the same unchanged privileged-teacher KL used by vanilla OPSD.
- The deliberately favorable NRA null fixture reuses one frozen synthetic teacher with 99 pronunciation-label donors and three held-out calibration neural records. Current and calibration are both train but have disjoint fold/session/trial identities and neural storage; records within each cohort share subject/session/`reliability_stratum`. A seed orders all 99 cyclic assignments (not 99 independent random permutations); the exact signed-Fisher statistic binds `statistic_epsilon=1e-8` and its threshold is the `0.95` higher quantile, with 98 broken anchor pairings, three any-cell anchor-label assignments, and one anchor match. This is smoke plumbing, not real cross-fitted neural or benchmark evidence.
- `brain-train --config ... --dry-run` resolves and prints the experiment without private data.
- `brain-train --synthetic --method {ce,vanilla_opsd,legacy_dual_cosine,nra_opsd,rlvr}` emits deterministic JSON metrics.
- `brain-evaluate` accepts config dry-run only; it has neither synthetic nor real evaluation execution.
- Preparation scripts require a user-provided dataset directory, checkpoint file, actual pinned upstream `--revision`, and matching `--checkpoint-sha256`; they create validated manifests, reject output aliases of the checkpoint, and never download private datasets implicitly. `/local/` is ignored for local manifests and machine-specific paths.

**TDD sequence:** add integration assertions for all modes and CLI JSON, observe missing entry-point failures, implement the synthetic runner and CLIs, rerun focused tests, then run the full suite.

## Task 6: Reproduction documentation and experiment matrix

**Files:**

- Modify: `.gitignore`
- Create: `README.md`
- Create: `docs/algorithm.md`
- Create: `docs/benchmark_matrix.md`
- Create: `docs/reproduction.md`
- Create: `docs/experiment_status.md`
- Test: `tests/test_documentation.py`

**Behavior:**

- README gives install, unit-test, smoke, dry-run, BIT preparation, mechanism pilot, and confirmatory commands; preparation examples include checkpoint file paths, pinned revisions, and SHA-256 digests.
- Algorithm documentation translates `x_i/x_j` and `p_i/p_j` into Chinese examples and explains why `K` may exceed one.
- Benchmark matrix names planned datasets, baselines, primary/secondary metrics, brain-dependence controls, seeds, and stop/go gates without implying their trainers/evaluators exist.
- Experiment status clearly separates locally verified synthetic execution from real-data experiments requiring datasets/checkpoints.
- Local manifests and machine-specific paths under repository-root `/local/` stay ignored.
- Documentation tests execute referenced config/script command contracts and verify the local-manifest ignore behavior without grepping prose.

**TDD sequence:** add the path/command consistency test and observe missing-file failures, add documentation, rerun it, and finish with `pytest -q`, `ruff check .`, `python scripts/smoke_test.py`, and all documented `--dry-run` commands.

## Commit and push gate

Before committing, run from repository root:

```bash
python -m pip install -e '.[dev]'
pytest -q
ruff check .
python scripts/smoke_test.py
brain-train --config configs/b2t24/nra_opsd.yaml --dry-run
brain-train --config configs/b2t25/nra_opsd.yaml --dry-run
brain-train --config configs/brainhub/umbrae_grounding.yaml --dry-run
brain-evaluate --config configs/b2t24/nra_opsd.yaml --dry-run
brain-evaluate --config configs/b2t25/nra_opsd.yaml --dry-run
brain-evaluate --config configs/brainhub/umbrae_grounding.yaml --dry-run
git diff --check
```

Review the full diff against the design spec, commit on `codex/nra-opsd`, push `codex/nra-opsd` to `origin`, and verify that local `HEAD` equals `refs/remotes/origin/codex/nra-opsd`.
