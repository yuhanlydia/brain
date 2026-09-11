# Reusable exploratory NSD experiment matrix

Spec: `docs/arithmetic_npp_method.md`, `docs/benchmark_matrix.md`, the nine NSD
configs, and `results/2026-09-11/real-preparation/baseline_sources.md`.

Goal: implement and execute every available-backend comparison, seed, subject,
control and ablation with honest exploratory results and resumable state.
Performance gates are diagnostics under the user's latest instruction.

Architecture: shared real-data preparation and model loading; separate tested
baseline objectives; explicit teacher-view construction; one resumable schedule
and evaluation interface. Existing P1 behavior remains available. Full runs
start from original weights and full training-only calibration, never P1 state.

Global constraints:
- Keep exact arithmetic NPP unchanged. No synthetic fallback or fabricated
  baseline names/results. Describe all NSD adaptations explicitly.
- Preserve image-group splits, held-out calibration, fixed rollout prefixes,
  detached teacher targets, brain-only student inference and control provenance.
- Run correct-brain training and evaluate all declared brain controls on the
  same trained checkpoint. Separate intervention training is an explicit ablation.
- Use the same seed-specific data order, trainable parameter set and optimizer
  update budget across methods; log actual teacher/student sequence-view tokens.
  DAPD's extra views are reported, not removed to force artificial equality.
- Save model/optimizer/RNG/data cursor state and restore it before continuing.
- Keep exploratory data/step/evaluation budgets explicit and configurable; do
  not claim full benchmark coverage for a budgeted subset.
- Complete BrainJanus remains release-dependent; continue available VINDEX/
  UMBRAE work while independently auditing that source gap.

## Task 1: faithful baseline numerical objectives

Create `src/brain_npp/baselines.py` and `tests/test_baselines.py`.
Read the pinned baseline source audit before implementation. Expose reusable
tensor-only functions, detached targets and authoritative token masks.

Implement full-vocabulary forward KL targets for exact-image, MAP-image,
uniform arithmetic mixture and posterior arithmetic mixture. CE uses separate
gold targets. Preserve arithmetic NPP in its existing module.

Implement clearly named CREDIT-style image-contrastive frozen-teacher reverse
KL: normalized target proportional to T_positive/T_negative**0.1, full vocabulary.
This is a disclosed paper-based adaptation, not the unreleased official code.

Implement VAD-style full-image intervention objective faithful to released
budgeted_asvc: student-top100 plus aggregated tail, common support, signed
projection budget, positive cap0.8, element clip20, alpha0.5 JSD, eta0.1 residual
rho-gated weak teacher, detached token importance multiplier capped2. Input
clear/degraded teacher distributions share prefix and weights. Degradation
belongs in the runner, with scale0.10 bilinear downsample then nearest upsample.

Implement DAPD's six directed, detached-anchor losses from the source table,
temperature1.1, per-vocabulary contribution upper cap0.05, weights summing2;
accept mappings of four live and six anchor views with potentially differing
rollout/reference token masks. Reference privilege is dataset gold answer text
for the NSD adaptation; snapshot lifecycle belongs in the runner.

Tests first: hand-computable mixture/gradient direction; permutation and mask
invariance; no teacher gradients; CREDIT reverse-vs-forward distinction;
VAD support tail mass and identical-teacher identity, independent signed-budget
reference formula; DAPD all six weighted terms against independent manual
calculation, differing completion lengths, component clipping and detach.
Reject inconsistent shapes, nonfinite active inputs, invalid probabilities or
empty masks with useful errors. Run focused tests and existing regression.

## Task 2: reusable real dataset and controls

Integrate the validated full manifest, VQA/caption sidecars, subject-specific
BrainXS/BrainX routing and full train-only Gaussian artifacts. Version the
successful full feature/calibration preparation scripts. Predeclare gallery
selection and report support/coverage; never force-insert true images.
Build same-session/run derangements excluding repeated image/self/HRF neighbors,
covariance noise fitted on training brains, actual zero-input and metadata-
validated same-image wrong-subject mappings. Persist mappings and unmatched IDs.

## Task 3: multi-method resumable runner

Extend real adapter/factory for P2/P3/P4 and captioning using Task1 losses and
Task2 data. Build DAPD six anchors/four live views and100-step snapshots; construct
VAD degradation and CREDIT mismatched-image negatives honestly. Reuse one frozen
base and explicit projector/LoRA contexts. Respect all declared generation and
accumulation settings, including partial windows. Add train/evaluate/checkpoint
artifacts and measured model-pass accounting. Small CPU protocol tests precede
GPU integration checks. Publish the exact matched exploratory schedule before
starting comparative runs.

## Task 4: execute and report the complete matrix

Run all methods, subjects01/02/05/07, seeds41/42/43, five inference controls,
arithmetic/geometric/IG/sqrtIG ablations; P3 uses the declared10% data subset.
Run captioning and report its metrics; attempt the BrainJanus extension only
with a complete verified backend/checkpoint. Report VQA/category accuracy,
caption metrics, posterior/retrieval/repeat diagnostics, NDG and image-cluster
uncertainty with subset coverage, compute budgets and negative findings. Continue
past performance diagnostics; debug actual failures. Review and push artifacts.
