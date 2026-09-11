# Real NSD calibration and P1 runner

Spec: `docs/arithmetic_npp_method.md`, `docs/benchmark_matrix.md`, and
`configs/nsd/p1_vindex_real_shape_smoke.yaml`.

Goal: execute the declared real P1 through `brain-npp train`, with an explicit
fitted candidate likelihood, frozen real teachers, trainable projector/LoRA,
and auditable outputs. P1 completion does not complete P2–P4, captioning, or
the BrainJanus extension.

Acceptance update (user, 2026-09-11): experiment phases are diagnostics rather
than strict performance gates. Continue the complete matrix without requiring
fixed accuracy/NDG thresholds; honestly report exploratory positive and negative
changes. Reusable configuration, data preparation and resumable execution are
priorities. Numerical correctness and data isolation remain mandatory.

Architecture: NumPy image-group cross-fitted Gaussian encoding model supplies
`log_likelihood` scores. A VINDEX/LLaVA adapter implements the existing
`PosteriorTeacherStudent` protocol and reuses `NPPTrainer`. A versioned thin
factory is installed into the pinned external checkout. Large assets remain
outside git; revision pins, split and artifact hashes are saved.

Global constraints:
- Preserve arithmetic mixture ratio, lambda=1, detached Q, exact full-vocab KL,
  one live student rollout forward, and fixed rollout IDs for every teacher.
- Use actual single-trial brain tensors and NSD-VQA labels. No synthetic fallback.
- Group calibration by image identity; all repeated trials share a fold. Test
  images never enter fitting. No raw cosine relabeling as log likelihood.
- A fixed brain-independent candidate gallery is shared across controls; never
  force-insert the true image. Report its limited coverage in P1.
- Fail explicitly for unimplemented methods, subjects, or control modes.
- P1 is a shape/optimizer diagnostic, never an accuracy or calibration claim.
- Preserve upstream checkpoint provenance and record the new likelihood model
  as an experimental addition, not a released VINDEX calibration artifact.

## Task 1: Gaussian encoding likelihood

Create `src/brain_npp/calibration.py` and `tests/test_calibration.py`.
Implement fixed seeded raw-brain random projection, image-balanced ridge
encoding, inner image-group OOF residual covariance with diagonal shrinkage,
and outer image-group cross-fitting. The output is a normalized Gaussian
log density over the projected brain statistic, not all voxels. Fixed P1
hyperparameters avoid selection on smoke outputs; full evaluation must freeze
choices before hidden-test use. Artifact serialization must include train IDs,
folds, projection seed/dimension, coefficients, covariance and model semantics.

Tests first: Gaussian scoring against an independent analytic density;
repeated-image fold exclusion; held-out brain perturbation cannot change that
fold's fitted parameters; serialization preserves scores; too few groups,
nonfinite inputs, and feature/brain mismatches fail. Validate covariance
positive definiteness and explicit no-evidence scores equal zero.

## Task 2: Real P1 integration

Create `src/brain_npp/adapters/vindex.py`,
`integrations/vindex/brain_npp_factory.py`, a reproducible preparation script,
and focused adapter tests. Keep heavyweight dependencies optional/lazy.
Load strict BrainXS state and CLIP224 layer -2 image features. Load LLaVA 7B
NF4; frozen base with projector and LoRA trainable. Frozen image teacher uses
the initial projector and base without student adapters. Cache frozen visual
features; stream candidate teacher logits on the same student rollout.
Slice causal logits starting at prompt length minus one; masks exclude padding
and retain EOS. Save optimizer state, trainable parameters, RNG state, rollout
diagnostics, input hashes and actual optimizer steps.

Validate causal alignment and teacher/student isolation with small test modules
before a real GPU run. Run all existing tests, then configured real P1 on up to
8 records, 64 generation tokens, one optimizer step; report the actual consumed
example count. Verify nonzero finite projector/LoRA gradients and frozen teacher
state. Review the complete implementation before publishing and merging.

## Remaining experiment work after P1

Implement the complete CE/OPSD/CREDIT/DAPD/VAD comparison contracts with declared
matched budgets, validated controls and metrics; run P2, the 10% P3 gate, all
P4 subjects/seeds, pooling/strength ablations and captioning. Validate UMBRAE's
released cross-subject architecture for subjects 02/05/07. Continue searching
for complete BrainJanus sources/checkpoints; report a verified release blocker
if none exists, without substituting a different backend for that extension.
