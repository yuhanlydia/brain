# Neural-Recoverability-Aware OPSD Design

**Status:** Approved research design; partial skeleton implementation
**Date:** 2026-09-07
**Repository:** `yuhanlydia/brain`

**Implementation snapshot (2026-09-07):** the repository currently provides pure-tensor methods, typed contracts, task-aware config validation/dry-runs, local manifest validation, and a deterministic CPU synthetic smoke. It does not yet provide a real dataset runner, BIT/UMBRAE model wiring, a 7B or other production decoder integration, real trainers, multi-instance Hungarian evaluation, or bootstrap inference. B2T24, B2T25, and BrainHub/UMBRAE are all **NOT RUN**; this document specifies planned research and makes no SOTA claim.

## 1. Goal

Build a reproducible research codebase for testing whether on-policy self-distillation (OPSD) and reinforcement learning with verifiable rewards (RLVR) improve brain-conditioned language generation for the mechanism they are designed to address, rather than merely amplifying the language-model prior or copying privileged labels that are not recoverable from neural recordings.

The planned first publishable track is Brain-to-Text '24/'25. A later adapter is intended to target BrainHub grounding with an external Shikra/Vicuna-7B stack; that integration is not implemented in the current repository.

## 2. Research claims

The codebase must test four separate claims.

1. **Prefix-distribution claim:** teacher-forced cross-entropy trains on gold prefixes, while deployment uses model-generated prefixes. Brain-conditioned long-form decoding therefore benefits from token-dense feedback on student rollouts.
2. **Objective-alignment claim:** token CE and phoneme error do not directly optimize final WER; coordinate-token CE does not directly optimize geometric IoU.
3. **Brain-evidence claim:** any gain must increase dependence on the correct brain trial, not only improve linguistic or dataset priors.
4. **Neural-recoverability claim:** a privileged phoneme correction should be transferred only along vocabulary directions whose paired brain--phoneme interaction is specific to the matching neural trial, reliable within that trial, and stable across sessions.

The third claim is mandatory. A result that improves WER/IoU without increasing the real-brain versus matched-wrong-brain gap is not treated as improved brain decoding.

## 3. Scope and phased delivery

### Planned Phase 1: Brain-to-Text flagship

Implement a BIT-compatible end-to-end adapter:

```
neural recording -> neural encoder -> projector -> autoregressive decoder -> text
```

Training stages:

1. CE plus the base alignment loss for warm start.
2. Neural-Recoverability-Aware OPSD (NRA-OPSD).
3. Optional WER-RLVR refinement.

The repository will not call CTC or a CTC-to-n-gram cascade “OPSD.” OPSD applies only to an autoregressive decoder evaluated on its own text prefixes.

### Planned Phase 2: BrainHub 7B adapter

Implement an adapter for VINDEX/UMBRAE brain tokens and Shikra/Vicuna-7B grounding. Optimize a dense IoU reward and evaluate matched-wrong-brain counterfactuals. Caption generation remains an OPSD auxiliary experiment, not a strict RLVR task.

### Out of scope for the first release

- Training a brain encoder or 7B LLM from scratch.
- Claiming that 7B is intrinsically better than 1–2B.
- Applying RLVR to continuous NLB/FALCON regression.
- Treating CIDEr, BLEU, BERTScore, or CLIPScore as strict verifiers.
- Reproducing private leaderboard systems with unavailable code.

## 4. Common architecture

The package will have four independent layers.

### 4.1 Model adapter

A small protocol exposes brain-conditioned autoregressive models without depending on one model family:

- `student_logits(brain, prefix)`
- `teacher_logits(brain, privilege, prefix)`
- `generate(brain, generation_config)`
- `trainable_parameters()`

Planned real adapters translate BIT or Shikra-specific tensors into this protocol. The current implementation exposes typed batch and manifest contracts only, not those external model runners.

### 4.2 Objectives

- Standard next-token CE.
- Full-vocabulary forward-KL OPSD on student-generated prefixes, with fail-closed validation of active student/target distributions.
- Pointwise KL clipping.
- WER sequence reward.
- Bounding-box IoU, threshold, and invalid-format rewards.
- Optional staged hybrid: CE checkpoint -> OPSD checkpoint -> RLVR checkpoint. OPSD and RLVR are not silently mixed into one scalar objective.

### 4.3 Neural-recoverability target construction

The primary method is a target constructor, not a scalar gate over an unchanged teacher KL:

```python
target, diagnostics = recoverable_target(
    mode="nra",
    student_log_probs=student_log_probs,
    teacher_log_probs=teacher_log_probs,
    anchor_interactions=nra_interaction_batch,
    nra_evidence=nra_evidence,
    neural_reliability=neural_reliability,
    session_stability=session_stability,
)
```

The NRA path requires two typed, mutually checked artifacts. `NRAEvidence` binds the frozen cross-fitted CE direction and exact signed-Fisher null threshold to training-only producer membership, current subject/pronunciation/reliability and fold/session/trial identity, checkpoint, `K`, sampler, aggregation/weight policy, seed, prefix/alignment/support, phase/permutation count, and the anchor-local three-mismatch control policy. `NRAInteractionBatch` binds the interaction values and any weights to the same current context. The constructor rejects any field mismatch, a raw interaction tensor, separate weights, or any `null_interaction` in NRA mode. Raw tensors, separately supplied weights, and raw-null subtraction are available only through the explicit `interaction_only_ablation` mode.

Every anchor trial `i` has neural recording `x_i` and aligned privileged pronunciation/phoneme sequence `p_i`. A real loader must first build candidate pools strictly inside the official data split. For each contrast, sample the trial supplying negative pronunciation `p_j` and a separate control brain `x_c` from the same split, fold, subject, session, and reliability stratum as the anchor, with matched duration. The control's true pronunciation `p_c` must differ from both `p_i` and `p_j`. This makes exactly one of the four cells matched:

| | `p_i` | `p_j` |
|---|---|---|
| `x_i` | matched `(x_i,p_i)` | mismatched `(x_i,p_j)` |
| `x_c` with `p_c` not in `{p_i,p_j}` | mismatched `(x_c,p_i)` | mismatched `(x_c,p_j)` |

The teacher receives brain soft tokens plus privileged pronunciation. The student, including at inference, receives brain soft tokens only. Let `z_t(x,p)` be detached teacher log-probabilities over the vocabulary. Geometry is measured around the current student distribution `s_t`:

```math
C_{s_t}[a]
=
a-\left(\sum_v s_t(v)a(v)\right)\mathbf 1,
```

```math
\langle a,b\rangle_{s_t}
=
\sum_v s_t(v)a(v)b(v).
```

All four teacher calls use the identical generated prefix and detached student reference distribution. The anchor-local factorial interaction is

```math
\Gamma_{icj}^t
=
C_{s_t}\!\left[
\{z_t(x_i,p_i)-z_t(x_i,p_j)\}
-
\{z_t(x_c,p_i)-z_t(x_c,p_j)\}
\right].
```

The control row subtracts pronunciation effects that occur for a brain whose true pronunciation is neither supplied option. Additive pronunciation copying cancels exactly. Brain-dependent gain, attention saturation, SNR, or prefix compatibility can still create nuisance interactions, so the raw interaction is not sufficient evidence. Such nuisance effects are controlled by calibrating the complete signed alignment statistic below, rather than averaging vocabulary vectors whose directions are not comparable. This remains an association test under explicit matching and cross-fitting assumptions, not an unconditional causal claim.

Aggregate multiple negatives before projection:

```math
\bar\Gamma_i^t
=
C_{s_t}\!\left[\sum_{(c,j)=1}^{K}w_{icj}\Gamma_{icj}^t\right],
\qquad
w_{icj}\ge0,
\quad
\sum_{c,j}w_{icj}=1.
```

Uniform mean is the default, with componentwise median as a registered robustness ablation; the result is always re-centered under the current `s_t`. Use `K=2` for smoke runs and at least four distinct negative pronunciations for confirmatory runs. Confirmatory examples with a sampler shortfall are skipped and logged rather than weakening constraints.

The method also requires brain-only evidence from a frozen, cross-fitted CE checkpoint that is never updated by NRA-OPSD and never consumes privileged input. Both brain conditions use the identical anchor prefix, token alignment, and vocabulary support:

```math
\bar B_i^t
=
C_{s_t}\!\left[
\sum_c \omega_{ic}
\{\log p_S^t(\cdot\mid x_i)-\log p_S^t(\cdot\mid x_c)\}
\right].
```

Retain only the interaction component supported by this brain-only direction:

```math
u_i^t
=
\frac{[\langle\bar\Gamma_i^t,\bar B_i^t\rangle_{s_t}]_+}
{\lVert\bar B_i^t\rVert_{s_t}^2+\epsilon}\bar B_i^t.
```

Define the signed normalized alignment

```math
A_i^t
=
\frac{\langle\bar\Gamma_i^t,\bar B_i^t\rangle_{s_t}}
{\lVert\bar\Gamma_i^t\rVert_{s_t}\lVert\bar B_i^t\rVert_{s_t}+\epsilon}.
```

For each held-out training fold, permute pronunciation assignments while holding prefix, stratum, `K`, sampler, aggregation, weights, and vocabulary support fixed. Let `tau` be the training-only `(1-alpha)` quantile of that exact statistic, with `alpha=0.05`, at least 99 permutations for a pilot, and at least 999 for confirmatory runs. The transfer strength is

```math
\kappa_i^t
=
\operatorname{clip}
\left(
\frac{A_i^t-\tau_i^t}{1-\tau_i^t+\epsilon},0,1
\right).
```

The calibration artifact records the frozen source, training split, producer fold/session/trial membership, current subject/session/trial/pronunciation identity, producing checkpoint, reliability stratum, `K`, sampler, aggregation/weight policy, prefix/alignment policy, vocabulary support, seed, statistic identifier, `statistic_epsilon`, alpha, phase, and permutation count. Producer membership must exclude the current fold/session/trial. These fields are matched against `NRAInteractionBatch` before the threshold can be used, and `recoverable_target` requires its runtime `epsilon` to exactly equal the artifact's `statistic_epsilon`; raw-null subtraction is not a substitute for exact-statistic calibration.

The CPU synthetic path deliberately uses a favorable controlled fixture rather than claiming real calibration. One frozen synthetic teacher is shared by current and calibration computations. Calibration contains 99 pronunciation-label donor records and three held-out neural records (one anchor plus two controls). Both cohorts are labeled train, but their fold/session/trial identities and neural tensor storages are disjoint; within each cohort, records share subject, session, and reliability stratum. A seed orders all 99 cyclic shifts, so these are 99 cyclic assignments, not 99 independent random permutations. Each assignment is scored with the exact `signed_fisher_alignment` and registered `statistic_epsilon=1e-8`, and the threshold is its `0.95` higher quantile: 98 assignments break the anchor pairing, three place the anchor label in any scored cell, and one preserves the anchor match. This fixture tests statistic and provenance wiring only; it is not cross-fitted real neural evidence and cannot satisfy P3 or support a benchmark claim.

Let the complete privileged correction be

```math
r_i^t
=
C_{s_t}\!\left[z_t(x_i,p_i)-\log s_t\right].
```

Only its positive projection onto neural-supported interaction is reconstructed:

```math
\beta_i^t
=
\frac{[\langle r_i^t,u_i^t\rangle_{s_t}]_+}
{\lVert u_i^t\rVert_{s_t}^2+\epsilon},
```

```math
r_{NR,i}^t
=
\rho_i^t\,\sigma_i^t\,\kappa_i^t\,\beta_i^t u_i^t,
```

```math
\lambda_{clip,i}^t
=
\begin{cases}
1,&\lVert r_{NR,i}^t\rVert_\infty=0,\\
\min\left(1,\dfrac{c}{\lVert r_{NR,i}^t\rVert_\infty}\right),&\text{otherwise},
\end{cases}
\qquad
q_{NR,i}^t
=
\operatorname{softmax}
\left(
\operatorname{stopgrad}(\log s_t)
+
\alpha_{step}\lambda_{clip,i}^t r_{NR,i}^t
\right).
```

The clip is one scalar for the whole correction (reported as `clip_scale`), so it preserves the brain-supported span; `alpha_step` is distinct from the null-test `alpha=0.05`. Here `rho` is calibrated neural/CTC reliability computed in cross-fitted training folds, `sigma` is a detached cross-session stability score, and `kappa` is permutation-null excess. NRA mode requires callers to supply both `rho` and `sigma` explicitly. Reliability/calibration artifacts and stability-bank entries carry split, fold, subject, session, trial, pronunciation, checkpoint, prefix policy, token-alignment policy, and vocabulary-support provenance. Calibration pronunciation identity must match the current pronunciation exactly, and reliability validity masks are boolean or finite binary `{0,1}` tensors. Stability stores raw contrasts and re-centers both current and bank directions under the same detached reference distribution before comparison. Lookup is training-only, checkpoint-matched, and leaves the current trial/session out. In primary and confirmatory modes, missing cross-session support fails closed to stability `0`, blocks transfer, and is logged; a `1.0` fallback may exist only as an explicitly named ablation. All supplied teacher states, frozen CE support states, interactions, weights, reliability, stability, null statistics, and targets are detached at the tensor constructor boundary; gradients update only the live student path there.

This does not establish end-to-end leakage safety for the unimplemented real pipeline. The projection API still receives reliability and stability as upstream-validated raw detached scalars/tensors rather than provenance-bound typed gate artifacts. A real runner must separately enforce their artifact lineage and data-flow isolation.

Student log-probabilities are checked by `logsumexp` in the internal work dtype with a normalization tolerance that accounts for source dtype and vocabulary size. Accepted values are then explicitly log-normalized before Fisher arithmetic. This supports large-vocabulary BF16 inputs without accepting materially unnormalized logs. The reported `target_kl` diagnostic is clamped at zero to remove negative roundoff artifacts; the target and training objective are otherwise unchanged.

For a 16 GB GPU, the interaction and projection may use a shared union of top-vocabulary entries plus a residual bucket, but all methods in a comparison must use the identical support. Raw uncentered Euclidean cosine is not the primary geometry.

Required ablations under matched compute are:

- `uniform`: vanilla full-target OPSD;
- `reliability`: reliability-weighted full-target OPSD;
- `teacher_contrast`: privileged teacher contrast only;
- `brain_contrast`: brain contrast only;
- `legacy_dual_cosine`: the earlier scalar cosine gate;
- `interaction_projection`: NRA-OPSD, the primary method.

The legacy cosine is retained only as an ablation because it weights the entire privileged KL; it does not remove teacher correction dimensions unsupported by neural evidence. Diagnostics distinguish `geometric_coverage` (interaction/projection/`kappa` support) from `transfer_coverage` (also requires positive reliability and stability); legacy `coverage` aliases transfer coverage. They also include projection magnitude, reliability, stability, negative validity, and target KL.

### 4.4 Evaluation

All evaluators return both task quality and brain dependence:

```
task_score = metric(real_brain)
brain_gap = metric(real_brain) - metric(matched_wrong_brain)
did_brain_gap = brain_gap(method) - brain_gap(CE_baseline)
```

Diagnostics include KL, entropy, generation length, invalid-output rate, reward-group zero-standard-deviation rate, and per-session performance.

## 5. Brain-to-Text method

For a neural trial (x), reference transcript (y^*), and student rollout (hat y), the primary NRA-OPSD loss is:

```math
L_{NRA}
=
\frac{1}{|\hat y|}
\sum_t
\operatorname{KL}_{clip}
\left[
q_{NR}(\cdot\mid x,p^*,\hat y_{<t})
\parallel
p_S(\cdot\mid x,\hat y_{<t})
\right].
```

Teacher parameters are a frozen copy of the CE checkpoint. If LoRA is enabled, the frozen teacher adapter and trainable student adapter share one loaded base model and run sequentially; the changing student adapter must never serve as teacher. The student sees no privileged transcript or phoneme sequence in its inference context. Transcript and G2P-derived pronunciation privilege are separate experiment arms, with phoneme privilege as the main condition and transcript privilege as an oracle upper bound. Vanilla OPSD still distills the full privileged teacher distribution and is reported as a distinct baseline.

The final sequence reward is:

```math
r_{WER}(\hat y,y^*)=-\frac{S+D+I}{N}.
```

CER, exact match, and truncation penalties may be used only as registered reward-shaping ablations. Official aggregate WER remains the primary metric.

### Brain-to-Text baselines

Under identical data, trainable parameters, decoding settings, and token/forward budget:

1. BIT CE plus contrastive alignment.
2. Scheduled sampling.
3. Off-policy privileged KD on gold prefixes.
4. Minimum-WER/MBR training.
5. Vanilla OPSD.
6. Vanilla WER-RLVR.
7. Teacher-contrast-only and brain-contrast-only gates.
8. Reliability-weighted and legacy dual-cosine OPSD.
9. NRA-OPSD interaction projection.
10. NRA-OPSD followed by WER-RLVR.
11. Published CTC/n-gram cascade as a benchmark reference, not a matched end-to-end method.

### Data protocol

- Brain-to-Text '24 and '25 are both required.
- Use official train/validation/hidden-test boundaries.
- Report per-session and corpus-type results.
- Build candidate pools inside official splits, then pre-register the same-split/fold/subject/session/reliability matching, negative sampler, `K`, duration/length tolerance, aggregation, reliability calibration, and fail-closed missing-session policy.
- Report interaction magnitude against shuffled-phoneme and shuffled-brain nulls, plus cross-session stability and coverage.
- Never tune repeatedly against a public leaderboard.
- Compare a compact audio-LLM with a 7B model; 7B is a scale control, not the assumed winner.

## 6. BrainHub grounding method

For predicted box (hat b) and target box (b^*):

```math
r_{box}
=
\operatorname{IoU}(\hat b,b^*)
+
\lambda \mathbb{1}[\operatorname{IoU}\ge 0.5]
-
\eta \mathbb{1}[\text{invalid}].
```

The dense IoU term is mandatory because threshold-only reward is too sparse at current brain-grounding accuracy.

A matched-negative trial (x_j) comes from the same official split, fold, subject, session, and reliability stratum and, when possible, the same queried class. In the planned real release it is an independent diagnostic, not part of the reward:

```math
U_{ground}
=
\operatorname{IoU}(\hat b(x_i),b_i)
-
\operatorname{IoU}(\hat b(x_j),b_i).
```

This prevents a reward from looking better merely by making the wrong-brain branch worse. A counterfactual reward is allowed only as a separately named ablation after the diagnostic result is established. Multiple instances use Hungarian matching with false-positive and false-negative penalties. The evaluator reports the original BrainHub protocol and an all-instance/absent-object corrected protocol.

### BrainHub baselines

1. Published image-feature MSE/denoising alignment.
2. Coordinate-token CE.
3. Smooth-L1 plus GIoU box head.
4. Vanilla IoU policy optimization.
5. Evidence-aware IoU policy optimization.
6. Image-privileged vanilla OPSD.
7. Image-privileged evidence-gated OPSD.

## 7. Falsification controls

Every main table contains:

- correct brain;
- zero brain;
- channel/time shuffled brain;
- matched-wrong trial;
- correct versus shuffled privilege;
- true versus random reward;
- at least one non-Qwen decoder where technically possible;
- at least three seeds.

A method claim fails if any of the following occurs:

- true reward does not outperform random reward at matched compute;
- real-versus-wrong brain gap does not improve over CE;
- WER training does not beat MWER/MBR;
- grounding does not beat GIoU or a matched IoU-RL baseline;
- only the training verifier improves while the official held-out metric does not;
- the effect exists only on one model family without qualification.

## 8. Resource constraints

The default configuration must fit a 16 GB GPU:

- frozen base LLM;
- 4-bit loading where supported;
- LoRA on selected decoder/projector modules;
- micro-batch size 1 with gradient accumulation;
- one student rollout per example for OPSD;
- bounded generation length;
- cached frozen teacher logits only when mathematically equivalent;
- sequential teacher/student/wrong-control forwards rather than simultaneous model copies;
- activation checkpointing and bf16/fp16 selected by hardware.

A planned 16 GB run has a 14.5 GiB peak-allocation gate; a planned 24 GB run has a 22 GiB gate. The 24 GB profile may increase sequence length or rollout group size, but must not silently change the method. Full 7B RLVR is optional; projector/adapter/LoRA training is the intended path, not a currently implemented runner. The planned default Brain-to-Text model is the stronger compact audio-LLM route; 7B is a scale control.

## 9. Repository layout

```
brain/
  pyproject.toml
  README.md
  src/brain_evidence/
    protocols.py
    objectives/
      opsd.py
      rewards.py
    recoverability/
      geometry.py
      interaction.py
      projection.py
      stability.py
    gates/
      reliability.py
      counterfactual.py
    controls/
      matched_negative.py
    metrics/
      text.py
      grounding.py
      brain_dependence.py
    adapters/
      bit.py
      brainhub.py
    cli/
      train.py
      evaluate.py
  configs/
    experiments/
      ablation.yaml
      confirmatory.yaml
    b2t24/
    b2t25/
    brainhub/
    hardware/
      gpu16gb.yaml
      gpu24gb.yaml
  tests/
    unit/
    integration/
  scripts/
    prepare_b2t.py
    prepare_brainhub.py
    smoke_test.py
  docs/
    benchmark_matrix.md
    reproduction.md
```

External BIT, VINDEX, and UMBRAE repositories remain optional pinned dependencies or documented sibling checkouts. Their source code will not be copied without need.

## 10. Acceptance criteria

The initial implementation is accepted when:

1. Unit tests verify OPSD uses student prefixes and every target-side tensor is detached.
2. Uniform gating exactly reproduces vanilla OPSD.
3. Matched negatives, negative-pronunciation source trials, and anchor-local controls never cross official split/fold/subject/session/reliability stratum; controls use a brain whose true pronunciation differs from both supplied pronunciations, respect duration tolerance, enforce distinct negative pronunciations, and expose effective-`K` shortfall.
4. A synthetic teacher that additively copies pronunciation yields zero anchor-local interaction.
5. The controlled favorable synthetic fixture suppresses copier-only/unsupported targets and retains a supported matched effect under the exact statistic; it is explicitly not accepted as evidence that a real cross-fitted neural interaction exists.
6. Fisher projection is invariant to additive logit constants, re-centers robust aggregates, removes orthogonal teacher correction, and becomes zero for unsupported or oppositely aligned correction.
7. Reliability and cross-session stability are bounded/detached and enforce split/session/trial/checkpoint provenance plus the exact missing-support fallback.
8. WER matches a trusted reference implementation on substitutions, deletions, and insertions.
9. IoU rewards reject malformed boxes and handle degenerate geometry.
10. A tiny synthetic model completes CE, vanilla OPSD, legacy-cosine OPSD, NRA-OPSD, and reward-optimization smoke runs on CPU; OPSD uses the current detached brain-only greedy rollout, legacy weights the unchanged privileged-teacher KL, and backpropagation reaches only the student path.
11. Equal-reward RLVR groups are skipped and their rate is logged.
12. The 16 GB dry-run configuration resolves without loading private data and enforces the 14.5 GiB gate.
13. README gives exact setup, data preparation, smoke-test, ablation-pilot, and first confirmatory experiment commands.

## 11. Execution gates

Implementation proceeds in this order:

1. **P0 CPU synthetic CI:** unit tests, toy overfit, checkpoint-resume equivalence, and all three OPSD modes.
2. **P1 real-shape smoke:** eight examples and three training steps per adapter; verify gradients, finite losses, target detachment, and peak memory.
3. **P2 baseline fidelity:** reproduce BIT validation behavior and one pinned UMBRAE/BrainHub checkpoint before testing new objectives.
4. **P3 interaction-existence pilot:** compare matched interactions with shuffled-brain and shuffled-phoneme nulls. Stop if interaction magnitude is not above the registered null interval or is unstable across sessions.
5. **P4 mechanism pilot:** compare vanilla, teacher-only, brain-only, reliability, legacy cosine, and NRA projection at matched compute. Stop if the NRA target is not more predictive of held-out real-versus-wrong brain benefit than the simpler gates.
6. **P5 full evaluation:** seeds 41/42/43, `K>=4`, paired bootstrap for Brain-to-Text, and unique-image cluster bootstrap for BrainHub.

The planned first real BrainHub implementation selects UMBRAE as the grounding backend, with VINDEX reserved as a later stronger baseline rather than a second simultaneous backend. The current repository validates a BrainHub/UMBRAE manifest and config only; Shikra token replacement, adapter/projector/LoRA training, and real grounding evaluation are not implemented here.

## 12. Prior-art positioning

The broad idea of confidence- or evidence-gated privileged distillation is not claimed as new. Comparisons and related-work discussion must include visual/audio privileged-distillation families such as VAD, FP-OPD, EDGE, CREDIT, CROP, and RLCSD when their public formulations are applicable.

The narrow proposed contribution is the combination of (1) a paired, within-subject/session brain--pronunciation factorial interaction, (2) a Fisher-centered vocabulary geometry, (3) reliability and cross-session stability, and (4) reconstruction of only the teacher correction projected onto that neural-supported direction. This is brain-specific because neural recordings have trial mismatch, acquisition reliability, and session-drift structure that generic image-to-text OPSD does not model. Claims remain empirical unless stronger identification assumptions are introduced.
