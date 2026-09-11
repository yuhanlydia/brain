# NPP-OPSD Design Specification (legacy expected-log draft)

> Superseded by `docs/arithmetic_npp_method.md`. This file is retained as
> design history; its expected-log aggregation and alpha defaults are not the
> current executable contract.

**Status:** Approved for implementation  
**Date:** 2026-09-10  
**Repository:** `yuhanlydia/brain`  
**Method:** Neural Posterior-Predictive On-Policy Self-Distillation

## 1. Research question

Can privileged image supervision improve a brain-conditioned 7B language model
without teaching stimulus details that are not identifiable from the observed
neural recording?

Exact-image OPSD conditions a frozen teacher on the ground-truth stimulus while
the deployable student observes only a noisy, subject- and session-dependent
brain measurement. The paired image is a complete description of the external
stimulus; a single fMRI trial is a stochastic and partial observation. Copying
the exact-image teacher distribution can therefore improve caption or VQA
metrics while reducing reliance on the correct neural trial.

NPP-OPSD replaces the exact-image teacher target with a neural-posterior
predictive target. It transfers only the change in teacher belief induced by
the candidate-stimulus posterior supported by the observed brain trial.

## 2. Novelty boundary

The method is not a KV-cache, hidden-state, SVD, PCA, or learned-subspace
method. It is also not a four-cell cosine gate. Correct/shuffled brain
comparisons remain evaluation controls, not the main training contribution.

The contribution is the combination of:

1. a probabilistic image-to-brain likelihood conditioned on subject/session;
2. a Bayes posterior over candidate stimuli for each neural trial;
3. a posterior-versus-prior teacher correction evaluated on a fixed student
   on-policy prefix;
4. a student-anchored token target with strength calibrated by neural
   information gain;
5. repeat-aware calibration and neural-dependence evaluation.

This scope must not be described merely as generic evidence gating,
counterfactual token selection, or Bayesian model averaging.

## 3. Mathematical contract

### 3.1 Candidate-bank neural posterior

For trial `i`, candidate stimulus `k`, subject `s_i`, and session `c_i`, the
encoding model supplies a log likelihood

```math
\ell_{ik}=\log p_\phi(b_i\mid I_k,s_i,c_i).
```

Given candidate-bank log prior `log p0_ik`, normalize posterior and prior over
the same valid candidate set:

```math
\log w_{ik}
=
\ell_{ik}+\log p^0_{ik}
-\operatorname{LSE}_{j\in\mathcal C_i}
(\ell_{ij}+\log p^0_{ij}),
```

```math
\log r_{ik}
=
\log p^0_{ik}
-\operatorname{LSE}_{j\in\mathcal C_i}\log p^0_{ij}.
```

Padded candidates have zero posterior and prior mass. A posterior with positive
mass outside the support of the prior is invalid.

### 3.2 Fixed on-policy intervention

The student first samples a response using only the brain input:

```math
\hat y_i\sim
\pi_\theta(\cdot\mid b_i,q_i).
```

Every teacher candidate is evaluated on the exact same stored prefix
`hat y_i,<t`:

```math
T_{ikt}
=
\pi_{\theta_0}
(\cdot\mid I_k,q_i,\operatorname{do}(\hat y_{i,<t})).
```

The prefix and candidate posterior are fixed target-side quantities. Teacher
outputs or later student changes must not update the posterior.

### 3.3 Posterior-contrastive correction

Let `T_iktv` be the teacher probability for vocabulary item `v`. Use expected
log probabilities, not the log of an arithmetic probability mixture:

```math
u_{itv}
=
\sum_k w_{ik}\log T_{iktv}
-
\sum_k r_{ik}\log T_{iktv}.
```

The trial-level neural information gain is

```math
a_i
=
D_{\mathrm{KL}}(w_i\Vert r_i).
```

The default strength transform is bounded:

```math
\alpha_i
=
\min(\alpha_{\max}, a_i / a_{\mathrm{scale}}).
```

`alpha_max` and `alpha_scale` are registered configuration values. The raw
information-gain variant `alpha_i=a_i` is a required ablation.

### 3.4 Student-anchored target and loss

Let `S_ref` be a detached copy of the current student distribution at the same
prefix. Construct

```math
\log Q_{itv}
=
\log S^{\mathrm{ref}}_{itv}
+\alpha_i u_{itv}
-\operatorname{LSE}_{z}
(\log S^{\mathrm{ref}}_{itz}+\alpha_i u_{itz}).
```

The training objective is

```math
\mathcal L
=
\mathcal L_{\mathrm{CE}}
+\lambda
\frac{\sum_{i,t}m_{it}
D_{\mathrm{KL}}(\operatorname{sg}[Q_{it}]\Vert S_{it})}
{\sum_{i,t}m_{it}+\epsilon}.
```

Only the current student logits receive gradient from the distillation term.
The encoding likelihood, candidate weights, teacher probabilities,
information gain, reference student distribution, and target are detached.

The initial implementation uses full-vocabulary forward KL. Candidate count,
not vocabulary support, is reduced for low-memory profiles.

## 4. Model boundary

The core package does not vendor VINDEX, BrainJanus, LLaVA, or SynBrain.
Instead, an adapter must provide:

```python
class PosteriorTeacherStudent(Protocol):
    def generate_student(self, batch, generation_config): ...
    def student_logits(self, batch, rollout_ids): ...
    def teacher_candidate_log_probs(self, batch, rollout_ids): ...
    def brain_candidate_log_likelihoods(self, batch): ...
```

The default real experiment is VINDEX-compatible brain soft tokens with a
LLaVA-1.5/1.6-7B language backbone. BrainJanus-7B is the unified-model
extension. A CPU toy adapter must exercise the exact objective end to end.

At inference, the model receives only brain tokens and the question/prompt.
Images, the candidate bank, the encoding model, and teacher calls are
training-only.

## 5. Data contract

The NSD manifest has one row per single trial and includes:

- `trial_id`
- `subject_id`
- `session_id`
- `run_id`
- `trial_index`
- `image_id`
- `repeat_id`
- `split`
- paths or immutable keys for brain data and image data
- optional question/answer/caption identifiers
- optional memory-status, response, motion, and beta-norm nuisance fields

Rules:

1. All repeats of one `image_id` remain in the same train/calibration/test
   fold.
2. Posterior/temperature calibration is cross-fitted by `image_id`.
3. Candidate priors and posterior calibration never use hidden-test labels.
4. Formal shuffle controls use predeclared same-subject, same-session/run
   derangements. Hard-mined negatives are training ablations, not permutation
   nulls.
5. Adjacent trials are excluded from matched-wrong controls by a configurable
   radius because of HRF overlap.
6. Metrics and confidence intervals cluster questions, captions, repeats, and
   subjects by unique image identity.

## 6. Evaluation

### 6.1 Primary benchmark

- Dataset: NSD single-trial GLMsingle betas.
- Tasks: NSD-VQA primary; brain captioning secondary.
- First subjects: `subj01`, `subj02`, `subj05`, `subj07`.
- Backbone: VINDEX/LLaVA-7B primary; BrainJanus-7B extension.

### 6.2 Baselines

1. CE/alignment-only brain decoder.
2. Exact-image vanilla OPSD.
3. MAP-image OPSD.
4. Uniform candidate mixture.
5. Posterior predictive mixture without prior contrast.
6. CREDIT-style input-specific control.
7. DAPD.
8. VAD-style visual attribution.
9. Full NPP-OPSD.

All methods use matched data, student rollouts, trainable parameters, update
steps, and teacher-forward budgets wherever their definitions permit.

### 6.3 Required controls and metrics

Every main result reports correct, within-block shuffled, zero, covariance-
preserving noise, and wrong-subject brain inputs.

```math
\mathrm{NDG}
=
M(b_{\mathrm{correct}})
-\mathbb E_\pi M(b_{\pi(i)}).
```

Report:

- VQA accuracy and question-category macro accuracy;
- CIDEr, SPICE, METEOR, ROUGE-L, and a factual caption metric;
- NDG and change in NDG over CE;
- candidate posterior NLL, Brier score, ECE, and top-k coverage;
- performance and posterior entropy for one, two, and three repeats;
- correct-trial versus shuffled-trial retrieval R@1/R@5;
- posterior entropy by VQA question category.

### 6.4 Exploratory continuation policy (updated 2026-09-11)

The user superseded the original performance go/no-go rule. Measure whether
exact-image OPSD improves task quality while leaving NDG unchanged or worse,
but continue the full matrix regardless. Report task-quality and NDG changes
separately; a positive change in either is an exploratory finding, with no
fixed threshold or joint-improvement requirement. Preserve negative outcomes
and distinguish task-quality gains from evidence of neural dependence.
Numerical correctness and data isolation remain requirements.

## 7. Resource profiles

The repository ships CPU synthetic, 16 GB, and 24 GB profiles.

- CPU: exact math and one tiny end-to-end optimization smoke test.
- 16 GB: frozen 4-bit 7B, LoRA/projector training, micro-batch 1, gradient
  accumulation, 4 candidate stimuli, bounded response length.
- 24 GB: the same method with up to 8 candidates or longer responses.

The 24 GB profile may change capacity but not the objective. Teacher candidate
forwards are sequential or chunked; the student and frozen teacher must never
be materialized as two full 7B copies.

## 8. Initial repository deliverable

The first push must contain:

1. a typed Python package with stable posterior/target/loss math;
2. a torch implementation with target-side stop-gradient guarantees;
3. NSD manifest validation, grouped split checks, matched-shuffle controls,
   and cross-fit assignment;
4. neural-dependence and posterior-calibration metrics;
5. a deterministic CPU toy model and smoke command;
6. configs for toy, NSD-VQA, captioning, 16 GB, and 24 GB;
7. exact commands for setup, tests, smoke, data validation, and the first
   pilot experiment;
8. an adapter contract plus explicit markers for external VINDEX/BrainJanus
   integration that fail loudly rather than silently substituting toy data.

## 9. Acceptance criteria

The implementation is accepted only when:

1. hand-computed fixtures verify posterior, information gain, correction,
   target normalization, and forward-KL direction;
2. extreme scores and candidate masks remain finite and normalized;
3. the target has no gradient and student logits do;
4. all teacher candidates receive an identical stored student prefix;
5. `image_id` groups never cross folds;
6. matched shuffles preserve requested subject/session/run blocks, reject
   self-pairs, and exclude adjacent trials;
7. CPU smoke performs a real optimization step and lowers NPP loss;
8. a missing real backend or dataset raises an actionable error;
9. the complete test suite and CLI smoke exit successfully from a clean clone.
