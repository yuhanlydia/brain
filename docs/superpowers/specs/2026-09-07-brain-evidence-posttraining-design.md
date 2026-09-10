# Brain-Evidence Post-Training Design

> **Superseded on 2026-09-10.** This document records the earlier
> four-cell/counterfactual-gate design. It is retained for provenance, but it
> is not the implementation target. See
> [`2026-09-10-npp-opsd-design.md`](2026-09-10-npp-opsd-design.md) for the
> approved Neural Posterior-Predictive OPSD design. Do not restore or use the
> four-cell/counterfactual-gate design below as an implementation or benchmark
> requirement; only the 2026-09-10 specification is binding.

**Status:** Superseded; provenance only
**Date:** 2026-09-07  
**Repository:** `yuhanlydia/brain`

## 1. Goal

Build a reproducible research codebase for testing whether on-policy self-distillation (OPSD) and reinforcement learning with verifiable rewards (RLVR) improve brain-conditioned language generation for the mechanism they are designed to address, rather than merely amplifying the language-model prior.

The first publishable track is Brain-to-Text '24/'25. A second adapter targets BrainHub grounding with an existing Shikra/Vicuna-7B stack.

## 2. Research claims

The codebase must test three separate claims.

1. **Prefix-distribution claim:** teacher-forced cross-entropy trains on gold prefixes, while deployment uses model-generated prefixes. Brain-conditioned long-form decoding therefore benefits from token-dense feedback on student rollouts.
2. **Objective-alignment claim:** token CE and phoneme error do not directly optimize final WER; coordinate-token CE does not directly optimize geometric IoU.
3. **Brain-evidence claim:** any gain must increase dependence on the correct brain trial, not only improve linguistic or dataset priors.

The third claim is mandatory. A result that improves WER/IoU without increasing the real-brain versus matched-wrong-brain gap is not treated as improved brain decoding.

## 3. Scope and phased delivery

### Phase 1: Brain-to-Text flagship

Implement a BIT-compatible end-to-end adapter:

```
neural recording -> neural encoder -> projector -> autoregressive decoder -> text
```

Training stages:

1. CE plus the base alignment loss for warm start.
2. Brain-evidence-gated OPSD.
3. Optional WER-RLVR refinement.

The repository will not call CTC or a CTC-to-n-gram cascade “OPSD.” OPSD applies only to an autoregressive decoder evaluated on its own text prefixes.

### Phase 2: BrainHub 7B adapter

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

Adapters translate BIT or Shikra-specific tensors into this protocol.

### 4.2 Objectives

- Standard next-token CE.
- Full-vocabulary forward-KL OPSD on student-generated prefixes.
- Pointwise KL clipping.
- WER sequence reward.
- Bounding-box IoU, threshold, and invalid-format rewards.
- Optional staged hybrid: CE checkpoint -> OPSD checkpoint -> RLVR checkpoint. OPSD and RLVR are not silently mixed into one scalar objective.

### 4.3 Brain-evidence gates and controls

A gate has the interface:

```python
gate = gate_fn(
    student_logits=student_logits,
    teacher_logits=teacher_logits,
    neural_reliability=neural_reliability,
    matched_wrong_logits=matched_wrong_logits,
)
```

Initial gates:

- `uniform`: vanilla OPSD.
- `reliability`: weights tokens using normalized CTC/neural posterior confidence.
- `counterfactual`: keeps supervision only where the correct brain trial changes the prediction more than a matched wrong trial.
- `reliability_counterfactual`: product of the two signals.

For the main evidence gate:

```math
\Delta_T^t
=
\log p_T^t(\cdot\mid privilege_i)
-
\log p_T^t(\cdot\mid privilege_j),
```

```math
\Delta_B^t
=
\log p_S^t(\cdot\mid brain_i)
-
\log p_S^t(\cdot\mid brain_j),
```

```math
g_t
=
\operatorname{stopgrad}
\left[
\rho_t
\max(0,\cos(\Delta_T^t,\Delta_B^t))
\right].
```

The cosine may be estimated on the union of the teacher top-K vocabulary entries to save memory, but the final distillation loss remains full-vocabulary KL. Gate tensors are stop-gradient weights. Every proposed gate must be compared with the uniform gate under matched compute.

### 4.4 Evaluation

All evaluators return both task quality and brain dependence:

```
task_score = metric(real_brain)
brain_gap = metric(real_brain) - metric(matched_wrong_brain)
did_brain_gap = brain_gap(method) - brain_gap(CE_baseline)
```

Diagnostics include KL, entropy, generation length, invalid-output rate, reward-group zero-standard-deviation rate, and per-session performance.

## 5. Brain-to-Text method

For a neural trial (x), reference transcript (y^*), and student rollout (hat y):

```math
L_{OPSD}
=
\frac{1}{|\hat y|}
\sum_t g_t
\operatorname{KL}_{clip}
\left[
p_T(\cdot\mid x,y^*,\hat y_{<t})
\parallel
p_S(\cdot\mid x,\hat y_{<t})
\right].
```

Teacher parameters are a frozen copy of the CE checkpoint. If LoRA is enabled, the frozen teacher adapter and trainable student adapter share one loaded base model and run sequentially; the changing student adapter must never serve as teacher. The student sees no privileged transcript in its inference context. Transcript and G2P-derived phoneme privilege are separate experiment arms, with phoneme privilege as the main condition and transcript privilege as an oracle upper bound.

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
7. Brain-evidence-gated OPSD.
8. Gated OPSD followed by WER-RLVR.
9. Published CTC/n-gram cascade as a benchmark reference, not a matched end-to-end method.

### Data protocol

- Brain-to-Text '24 and '25 are both required.
- Use official train/validation/hidden-test boundaries.
- Report per-session and corpus-type results.
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

A matched-negative trial (x_j) has the same subject/session and, when possible, the same queried class. In the first release it is an independent diagnostic, not part of the reward:

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

A 16 GB run has a 14.5 GiB peak-allocation gate; a 24 GB run has a 22 GiB gate. The 24 GB profile may increase sequence length or rollout group size, but must not silently change the method. Full 7B RLVR is optional; projector/adapter/LoRA training is the supported path. The default Brain-to-Text model is the stronger compact audio-LLM route; 7B is a scale control.

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

1. Unit tests verify OPSD uses student prefixes and teacher gradients are stopped.
2. Uniform gating exactly reproduces vanilla OPSD.
3. Matched-wrong construction never crosses subject and respects session/class constraints when requested.
4. WER matches a trusted reference implementation on substitutions, deletions, and insertions.
5. IoU rewards reject malformed boxes and handle degenerate geometry.
6. A tiny synthetic model completes CE, OPSD, and reward-optimization smoke runs on CPU.
7. Equal-reward RLVR groups are skipped and their rate is logged.
8. The 16 GB dry-run configuration resolves without loading private data and enforces the 14.5 GiB gate.
9. README gives exact setup, data preparation, smoke-test, and first real experiment commands.

## 11. Execution gates

Implementation proceeds in this order:

1. **P0 CPU synthetic CI:** unit tests, toy overfit, checkpoint-resume equivalence.
2. **P1 real-shape smoke:** eight examples and three training steps per adapter; verify gradients, finite losses, and peak memory.
3. **P2 baseline fidelity:** reproduce BIT validation behavior and one pinned VINDEX/BrainHub checkpoint before testing new objectives.
4. **P3 ten-percent pilot:** vanilla OPSD before evidence gating, and evidence gating before RLVR. Stop if the privileged teacher advantage is below 0.05 nat/token, gate coverage is outside 10%-90%, or more than 50% of RLVR groups have zero reward variance.
5. **P4 full evaluation:** seeds 41/42/43, paired bootstrap for Brain-to-Text, and unique-image cluster bootstrap for BrainHub.

The first implementation selects UMBRAE as the BrainHub grounding backend because its Shikra token-replacement path is already implemented and reproducible. VINDEX remains a later stronger baseline adapter rather than a second simultaneous backend.
