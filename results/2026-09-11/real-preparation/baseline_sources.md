# Verified baseline sources and NSD adaptation boundaries

Verification date: 2026-09-11 UTC. This is source research, not an executed benchmark. Scope follows `docs/arithmetic_npp_method.md` and `docs/benchmark_matrix.md`: arithmetic NPP, NSD-VQA primary, brain-only inference, true-image privilege during training. No packages installed, GPU jobs launched, external writes, or production-code edits were performed for this audit.

## Sources, availability, and immutable code pins

| Method | Primary paper | Official code and checked commit | Local checkout |
|---|---|---|---|
| CREDIT | [arXiv 2605.11613v1](https://arxiv.org/abs/2605.11613v1), May 12, 2026 | No author-linked implementation found; no commit can responsibly be supplied | Paper/metadata only |
| DAPD | [arXiv 2608.01735v2](https://arxiv.org/abs/2608.01735v2), August 13, 2026 | [uanu2002/DAPD, fd902b4e805d66be1de6b97df4d08041c14441ee](https://github.com/uanu2002/DAPD/tree/fd902b4e805d66be1de6b97df4d08041c14441ee) | `/root/baseline-sources/DAPD` |
| VAD | [arXiv 2607.28590v1](https://arxiv.org/abs/2607.28590v1), July 30, 2026 | [DeepExperience/VAD_Multimodal_OPD, 4f2dd54103904a07dc39c3a60b2ad7bc29d8cb99](https://github.com/DeepExperience/VAD_Multimodal_OPD/tree/4f2dd54103904a07dc39c3a60b2ad7bc29d8cb99) | `/root/baseline-sources/VAD_Multimodal_OPD` |

The VAD arXiv abstract links its repository directly. DAPD's author-associated [HF metadata](https://huggingface.co/api/papers/2608.01735) links its repository; the repository identifies the matching paper. Both git clones succeeded, without recursive submodules. Paper markdown and metadata were saved as `/root/baseline-sources/<arxiv-id>.md` and `.metadata.json`; these markdown files identify their original arXiv HTML source. Repository HEADs were verified with `git rev-parse HEAD`.

CREDIT availability was checked against its arXiv abstract and full paper, [HF metadata](https://huggingface.co/api/papers/2605.11613), web title/ID searches, and GitHub repository search. Metadata has no `githubRepo`; the full paper contains no GitHub URL. GitHub API repository queries for the exact ID and title returned zero results. A broader query returned `TU2021/UCOB`; inspection established that this is a different paper, so it is not CREDIT code. This is a failure to locate released author code, not proof that none exists anywhere. Do not call a locally reconstructed CREDIT implementation “official.”

## CREDIT: exact paper prescription

Let `x` be the input, `y` the student rollout, `z=Env(x,y)` its feedback, `S_t(v)=pi_theta(v|x,y_<t)`, and `T_t(v)=pi_ref(v|x,y_<t,z)`. Sample `C` unrelated batch inputs `x'_k`, retaining the original rollout prefix and feedback. Equation 8 and Algorithm 1 prescribe

```math
G_t(v)=\frac1C\sum_{k=1}^C\log\pi_{ref}(v\mid x'_k,y_{<t},z),
\qquad R_t(v)=\log T_t(v)-\log S_t(v)-\lambda G_t(v).
```

The negative baseline is an average of log probabilities. It is not the log of an arithmetic mixture, not a teacher-minus-student correction under each negative, and not an input permutation that also replaces feedback. Algorithm 1 uses these vocabulary advantages in a reverse-KL gradient. An equivalent normalized target for the ideal fixed-teacher full-vocabulary reverse-KL gradient is `Q(v) proportional to T(v) exp(-lambda G(v))`; using forward KL to that target changes the optimization and must be labeled an adaptation. This equivalence does not settle how the unreleased implementation integrates the paper's JSD setting or truncation.

Paper Appendix A/Table 3 defaults:

| Setting | LiveCodeBench | SciKnowEval / ToolUse |
|---|---|---|
| C, lambda | 1, 0.1 | 1, 0.1 |
| Teacher | EMA student, update rate 0.01 | Same |
| Divergence | reverse KL, alpha=1 | JSD, alpha=0.5 |
| Vocabulary | teacher top 20 | teacher top 100 |
| Optimizer / learning rate | Adam / 1e-6 | Same |
| Warmup / epochs | 0 / 30 | 10 / 3 |
| Problem batch / rollouts | 32 / 8 | 32 / 8 |
| PPO mini-batch | 8 | 32 |
| Training sampling | temperature 1, top-p 1, top-k -1 | Same |
| Evaluation sampling | temperature .6, top-p .95 | Same |
| Max sequence length | 18,944 | Same |

At least one student autoregressive rollout, one trainable student scoring pass, and `1+C` detached teacher scoring views are required. These can be batched but remain distinct view/token costs. Feedback acquisition is additional. EMA updates are required for the reported self-distillation configuration. The paper's low wall-clock overhead is specific to its workloads, not a guarantee for multimodal NSD.

**NSD adaptation recommendation (explicitly adapted):** set inference input to `(brain, question)` and privilege to the paired true image. To preserve CREDIT's input-axis intervention literally, hold the true-image privilege, question/prefix policy, and feedback fixed while replacing the declared nonprivileged input component with an unrelated example. This requires a teacher that actually consumes the brain input. A teacher taking only image + question is invariant to swapping brain; calling that a brain-contrastive CREDIT control would be misleading. A feasible image-contrastive variant uses `T(I,q,prefix)` and `T(I',q,prefix)`, with mismatched images sampled from a saved same-batch derangement, then applies the reward above. It changes the intervention from input to privilege and must be named **CREDIT-style image-contrastive**, not an official CREDIT reproduction. Keep `C=1, lambda=.1`, the same prefix and same teacher weights across views. If using a frozen true-image teacher instead of EMA, state that separately. Reject self-pairs and duplicate image identities; do not mine negatives using test outcomes.

Unresolved reproduction detail: actual top-k mass handling, JSD reward integration, and prompt implementation cannot be verified from released code. A paper-based implementation can be useful but cannot fill these gaps by claiming official parity.

## DAPD: six losses, not one target

Actual objective: [dapd/objective.py](https://github.com/uanu2002/DAPD/blob/fd902b4e805d66be1de6b97df4d08041c14441ee/dapd/objective.py). Actual view construction, anchors, clipping and reduction: [dapd/trainer.py](https://github.com/uanu2002/DAPD/blob/fd902b4e805d66be1de6b97df4d08041c14441ee/dapd/trainer.py). Defaults: [train.py](https://github.com/uanu2002/DAPD/blob/fd902b4e805d66be1de6b97df4d08041c14441ee/train.py).

Write `P_a(c|h)` for model state `a`, evaluated along completion `c` with conditioning `h`. Completion is reference `r` or rollout `y`; condition is none `0`, reference solution `r`, or own rollout `y`. Every row below means `weight * D(stopgrad(left) || live-right)`, averaged over valid completion positions:

| Official pair | Completion | Detached left condition / model | Live right condition | Weight |
|---|---|---|---|---:|
| entangled_rollout | y | r / snapshot | 0 | 2/15 |
| inference_reference | r | 0 / base | r | 2/15 |
| privileged_rollout | y | r / base | y | 2/15 |
| entangled_reference | r | y / snapshot | 0 | 2/5 |
| inference_rollout | y | 0 / base | y | 2/5 |
| privileged_reference | r | y / base | r | 4/5 |

Weights sum to 2, not 1. With logit temperature `T=1.1` and vocabulary cap `c=.05`, the exact code loss is

```math
\ell_c(q,p)=\frac1{|M|}\sum_{t\in M}\sum_v
\min\{q_t(v)[\log q_t(v)-\log p_t(v)],0.05\}.
```

It upper-clips each vocabulary contribution, including leaving negative contributions unmodified; it is not clipping the final KL or clamping a likelihood ratio. There is no T-squared factor and no importance ratio. “Full-vocabulary forward KL” alone omits the material component clipping.

Base anchors use the same model with LoRA disabled. The two entangled teachers use a shared hard-copy snapshot of trainable LoRA weights, refreshed every 100 optimizer steps, initially copied before training. All anchor tensors are detached. Four distinct live scoring views suffice after deduplication: `(y,0), (r,r), (y,y), (r,0)`. There are six anchor views (two snapshot, four base). The release combines these into three physical forward calls, one per model-state group, so count **10 sequence-view equivalents**, not “3 ordinary passes,” plus one autoregressive rollout. Reference completions come from dataset gold solutions; generating substitute references adds cost and changes data provenance.

Defaults: effective batch 32; learning rate 5e-6; linear 500-step schedule without warmup, training stop at step 200; gradient clipping .1; bfloat16; LoRA rank 64, alpha 128 over q/k/v/o and gate/up/down projections; requested LoRA dropout .05 but trainer disables model dropout; generation temperature 1.1, top-p .95, top-k 20, one rollout/problem, max 1,024 new tokens; max context 20,000; seed 42; ZeRO-2 with CPU optimizer offload. Do not silently change to a 200-step LR schedule.

**Faithful NSD text-reference adaptation:** preserve all six rows, conditions, anchors and clipped divergence, replacing the ordinary text question input with `(brain, question)` throughout. Dataset reference answers are text privilege; richer reference solutions cannot be assumed present in NSD-VQA. A true-image teacher may generate a frozen, training-only reference answer/solution corpus, but record its checkpoint, prompts, token IDs and generation cost. This is **DAPD on brain-conditioned inputs with image-derived text references**, not the paper's original task reproduction. The base anchor must be an already usable brain-conditioned base, retaining the trained brain encoder/projector; blindly disabling every adapter may destroy its brain input interface. Explicitly freeze which brain parameters belong to the base and which are the DAPD-updated adapters.

**Direct image privilege is not a drop-in replacement:** the reference and rollout sources in DAPD are completions that can each act as both prefix condition and scored continuation. A true image cannot replace the reference completion in reverse-source losses. A one-step `T(image)||S(brain)` loss, even combined with a bridge KL, is not the released six-loss DAPD. For the requested matrix, implement the text-reference adaptation above or explicitly define and label a new image-source variant; do not fabricate a nonexistent official brain/image objective.

## VAD: released reconstruction, support, loss and counterfactual

Source: [core_algos.py](https://github.com/DeepExperience/VAD_Multimodal_OPD/blob/4f2dd54103904a07dc39c3a60b2ad7bc29d8cb99/verl/trainer/ppo/core_algos.py), target reconstruction around lines 1287–1395 and divergence around 1472–1530. Launch overrides matter: [scripts/train_vad.sh](https://github.com/DeepExperience/VAD_Multimodal_OPD/blob/4f2dd54103904a07dc39c3a60b2ad7bc29d8cb99/scripts/train_vad.sh) selects `budgeted_asvc`, not the generic `single` default.

At each student prefix, score fixed teacher with clear evidence (`T+`) and degraded evidence (`T-`), and live student `S`. The release selects **student top 100 vocabulary tokens plus one aggregated tail bin**, retaining actual probability mass. Its `full_logit_distillation=True` flag does not mean all vocabulary entries are retained individually. If tail is disabled, the implementation instead renormalizes support; that is a different setting.

On this common support, let `C(l)=l-mean(l)` (unweighted mean over valid bins), `phi_s=C(log stopgrad(S))`, `phi_+=C(log T+)`, `phi_-=C(log T-)`:

```math
r=\phi_+-\phi_s,\quad u=\phi_+-\phi_-,\quad
\beta=\frac{\max(\langle r,u\rangle,0)}{\|u\|^2+10^{-3}},\quad
r_{single}=\beta u,\quad B=\|r_{single}\|.
```

Released `budgeted_asvc` splits `u_+=max(u,0)` and `u_-=min(u,0)`, computes `s_+=max(<r,u_+>,0)` and `s_-=max(<r,u_->,0)`, then

```math
\pi_+=s_+/(s_++s_-+10^{-8}),\quad \pi_-=s_-/(s_++s_-+10^{-8}),
\quad a_+=\min(\pi_+,\tau_+),\quad a_-=\pi_-,
\quad r_{vis}=B\left(a_+\frac{u_+}{\|u_+\|+10^{-8}}+
 a_-\frac{u_-}{\|u_-\|+10^{-8}}\right),
\quad Q=\operatorname{softmax}(\phi_s+\operatorname{clip}(r_{vis},-20,20)).
```

Main step multiplier is 1, rho gate is off, and total shift-norm cap is unset. `rho=||r_vis||/(||r||+1e-8)` is computed before elementwise clipping. Targets and rho are detached. With `M_alpha=(1-alpha)S+alpha Q`, the implemented divergence for alpha=.5 is JSD:

```math
D_\alpha(S,Q)=(1-\alpha)KL(S\Vert M_\alpha)+\alpha KL(Q\Vert M_\alpha),
\quad L_t=D_{.5}(S,Q)+\eta\,\operatorname{clip}(1-\rho,0,1)D_{.5}(S,T^+).
```

The weak teacher coefficient is residual-gated, not a constant mixture of teacher and reconstructed target. Code also implements alpha=0 as FKL and alpha=1 as RKL, but the main launcher sets .5. Distillation losses are multiplied by a detached sampled-token policy/old-policy ratio capped at 2 (log ratio stabilized in [-20,20]); optional rollout correction weights are applied separately. The launcher selects token importance correction threshold 2. Report these choices when reimplementing.

[4B profile](https://github.com/DeepExperience/VAD_Multimodal_OPD/blob/4f2dd54103904a07dc39c3a60b2ad7bc29d8cb99/configs/vad_4b.env): Qwen3.5-4B, fixed same-initial-checkpoint teacher, positive cap .8, eta .1, learning rate 2e-6, warmup 10, batch/mini-batch 96, eight rollouts/problem, two epochs with 130-step budget, seed 42, max prompt 8192 and response 1024. [9B profile](https://github.com/DeepExperience/VAD_Multimodal_OPD/blob/4f2dd54103904a07dc39c3a60b2ad7bc29d8cb99/configs/vad_9b.env) uses Qwen3.5-9B and positive cap .7. The launcher explicitly fixes the teacher even though legacy EMA-related config fields remain; do not infer an EMA teacher from those inactive fields.

[Data preparation](https://github.com/DeepExperience/VAD_Multimodal_OPD/blob/4f2dd54103904a07dc39c3a60b2ad7bc29d8cb99/scripts/prepare_data.py) uses full image for student, clear relevant crop for teacher, and a degraded version of that same crop for the counterfactual. Degradation converts RGB, downsamples each dimension to `max(1,round(size*.10))` with bilinear interpolation, then upsamples to the original dimensions with nearest neighbor. It is neither a Gaussian blur nor an unrelated-image swap. Teacher and counterfactual keep question, prefix, framing and crop identity fixed.

Minimal objective compute: one student rollout, one differentiable student scoring view, and two detached fixed-teacher scoring views. Old-policy/logprob scoring or a separate top-k-support pass can add execution cost; count actual calls and image token lengths in the backend rather than asserting the minimum is the runtime total.

**NSD adaptation:** substitute brain+question for the student's full-image input. Set `T+` to the paired true image+question, `T-` to its exact declared degradation, both using the same frozen teacher and identical prefix. If genuine task-relevant crop annotations exist, preserve the clear/degraded crop construction. Otherwise use clear/degraded full image and name the run **VAD-style full-image intervention for brain student**; do not invent relevant crops from test answers. Preserve signed budgeting, support/tail, JSD, residual regularizer and importance behavior for the closest code-based adaptation. A full-vocabulary FKL variant can be a matched-loss ablation but must not replace the closest adaptation silently. Visual attribution here establishes evidence dependence in the teacher; it does not establish that the student uses its brain input, so all five brain controls remain necessary.

## Integration and reporting rules

1. None of these original papers reports NSD brain decoding. All NSD runs are adaptations even when the objective implementation is copied faithfully.
2. Keep arithmetic NPP unchanged: its posterior/prior ratio uses arithmetic teacher mixtures and detached live-student anchoring. CREDIT's expected negative log and VAD's centered-log projection are different baselines, not justification to reinstate legacy expected-log NPP.
3. Log objective name/version, code SHA, support size and tail policy, KL/JSD direction, teacher lifecycle, six DAPD weights where applicable, privilege mapping, negative/degradation manifest and loss masks. Preserve shared vocabulary/tokenizer or explicitly implement token alignment; a common vocabulary cannot be assumed across different model families.
4. Compare matched subject/image splits, seeds 41/42/43, trainable parameter set and optimizer updates. Report teacher/student **sequence-view tokens** and wall-clock separately. DAPD inherently costs more scoring views; provide both matched-update and budget accounting rather than deleting terms to force equality.
5. Evaluate correct, shuffled, zero, covariance-noise and wrong-subject brain on inference-only student outputs. All privileges/reference generation stay training-only. Image identity governs splits, negative exclusions and bootstrap units. These controls support interpretation; the user subsequently replaced strict performance gates with exploratory reporting (see docs/benchmark_matrix.md).
6. This audit verifies source availability and objective definitions only. Official runtime reproduction, dependency compatibility, model/data acquisition, NSD adapter implementation and real experimental results remain separate work. CREDIT additionally lacks a located official implementation; DAPD and VAD code availability is no longer a blocker.
