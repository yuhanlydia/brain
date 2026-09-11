# Arithmetic Neural Posterior-Predictive OPSD

## Executable mathematical contract

For candidate privileged stimuli/images \(I_i\), the brain encoder supplies
candidate scores and the adapter declares their semantics. With a reference
prior \(r_i\), log-likelihood scores give

```math
w_i=p(I_i\mid b)=\operatorname{softmax}_i(\log r_i+\ell_i(b)).
```

Teacher logits are normalized candidate by candidate into \(T_i(v)\). The
primary correction is an arithmetic mixture ratio, not an expected log:

```math
m_w(v)=\sum_iw_iT_i(v),\qquad m_r(v)=\sum_ir_iT_i(v),
```

```math
\log Q_t(v)=\log S_{t,\mathrm{anchor}}(v)
+\lambda[\log m_{w,t}(v)-\log m_{r,t}(v)]-\log Z_t.
```

The primary setting is \(\lambda=1\). `information_gain` and
`sqrt_information_gain` strength are named ablations. The optimization loss is
the full-vocabulary forward KL over valid rollout tokens:

```math
\mathcal L=\beta_{CE}\mathcal L_{CE}
+\beta_{NPP}\frac1{|M|}\sum_{t\in M}D_{KL}(Q_t\Vert S_t).
```

`Q` is detached. The same live `student_logits` tensor both anchors `Q` through
`detach()` and receives the KL gradient; there is no stochastic second
reference forward. When CE is enabled it uses a separate teacher-forced call
with dataset gold labels. Gradient accumulation averages microbatch scalar
objectives and partial windows divide by their actual count.

## Required identities and diagnostics

- If \(w=r\), then \(Q=S_{anchor}\) exactly and the update is zero.
- If \(S_{anchor}=m_r\) and \(\lambda=1\), then \(Q=m_w\).
- Joint candidate permutation leaves the result unchanged.
- A masked candidate cannot affect the posterior or teacher mixtures.
- `target_shift`, posterior entropy, information gain, KL, active token counts,
  gradient norm, and optimizer-step status are emitted by the trainer.
- fp16/bfloat16 inputs are promoted to at least fp32 for posterior, mixture,
  normalization, KL, and norm arithmetic.

## NSD experiment matrix (not yet executed here)

Primary backend: VINDEX/LLaVA-7B on NSD-VQA; secondary tasks: NSD captioning and
BrainJanus-7B extension. The external checkout must implement the protocol and
load real checkpoints/data; the core repository fails closed if it is absent.

| Comparison | Purpose |
|---|---|
| CE | no privileged teacher |
| exact-image OPSD | privileged upper bound |
| MAP-image OPSD | hard brain retrieval |
| uniform mixture | no neural evidence |
| posterior mixture | no prior-contrast ratio |
| arithmetic NPP-OPSD | proposed target |
| geometric NPP | aggregation ablation |
| IG / sqrt-IG strength | strength ablations |

Run seeds 41/42/43 across subjects 01/02/05/07. Report VQA accuracy (primary),
caption metrics where applicable, mean ± standard deviation, paired bootstrap
confidence intervals, and subject/session-stratified results. Required brain
specific controls are correct, within-session shuffled, zero, covariance-
matched noise, and wrong-subject inputs. Report gains in each measured outcome
as exploratory findings, without a fixed performance gate for continuing runs.
Claims about neural dependence require support from the control and session
comparisons; task accuracy alone is insufficient evidence for that interpretation.

No NSD or 7B result is claimed by the repository until these external runs
produce artifacts.
