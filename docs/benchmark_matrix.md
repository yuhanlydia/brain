# NPP-OPSD benchmark and ablation matrix

Every method is evaluated on matched NSD image groups, student rollouts,
trainable parameters, optimizer steps, and teacher-forward budgets wherever
the method definition permits. NSD-VQA is primary; captioning is secondary.

## Required matrix

| Method | Correct brain | Shuffled | Zero | Covariance-noise | Wrong-subject |
|---|---:|---:|---:|---:|---:|
| CE/alignment-only | Required | Required | Required | Required | Required |
| Exact-image OPSD | Required | Required | Required | Required | Required |
| MAP-image OPSD | Required | Required | Required | Required | Required |
| Uniform candidate mixture | Required | Required | Required | Required | Required |
| Plain posterior mixture | Required | Required | Required | Required | Required |
| CREDIT-style | Required | Required | Required | Required | Required |
| DAPD | Required | Required | Required | Required | Required |
| VAD-style visual attribution | Required | Required | Required | Required | Required |
| NPP-OPSD | Required | Required | Required | Required | Required |

“Plain posterior mixture” means posterior predictive mixing without the prior
contrast. “Shuffled” is the predeclared within-subject, same-session/run
derangement with self-pairs, repeated-image pairs, and trials within the HRF
adjacency radius excluded. Hard-mined negatives are training ablations, not
permutation nulls.

## Metrics

Report at minimum:

- NSD-VQA accuracy and question-category macro accuracy;
- CIDEr, SPICE, METEOR, ROUGE-L, and a factual caption metric;
- neural-dependence gap (NDG) and change in NDG over CE;
- candidate posterior NLL, Brier score, ECE, and top-k coverage;
- task performance and posterior entropy for one, two, and three repeats;
- correct-trial versus shuffled-trial retrieval R@1/R@5;
- posterior entropy by VQA question category.

Use image-identity grouping for splits, calibration, metrics, and confidence
intervals so questions, captions, and repeated trials from one image never act
as independent samples. Posterior/temperature calibration is cross-fitted by
`image_id` and cannot use hidden-test labels.

For a task metric `M`, report:

```text
NDG = M(correct brain) - mean_pi M(predeclared shuffled brain)
```

The current user-approved evaluation is exploratory: positive changes in task
quality, neural dependence, or other measured performance are useful findings.
There is no fixed accuracy threshold or requirement that task quality and NDG
improve together before continuing the experiment matrix. Report each change
against its matched baseline, with uncertainty and negative results alongside
positive ones. A task-quality gain alone does not establish neural dependence.

## Wrong-subject control boundary

Wrong-subject mappings must be precomputed from manifest subject metadata and
validated before loading or reindexing brain tensors. For every persisted
`source_trial_id -> target_trial_id` pair, the evaluation code must verify:

1. both IDs exist in the validated manifest;
2. source and target have different `subject_id` values;
3. they satisfy the declared cross-subject image/stimulus matching policy;
4. every evaluated source has exactly one target and no source maps to itself;
5. the mapping, seed, exclusions, and unmatched trial IDs are saved with the
   run artifacts.

`brain_npp.controls.apply_brain_control(..., control="wrong-subject")` is a
generic application API. It checks that mapping references exist in the
provided tensor batch, but it does **not** inspect manifest metadata or verify
that a target came from a different subject. Passing an arbitrary mapping to
that API is therefore not evidence of a valid wrong-subject control; provenance
validation belongs to the external dataset/evaluation factory.

## Interpretation and result status

P3 diagnoses the neural-bypass premise on 10% of the data with seed 41:
measure whether exact-image OPSD improves task quality while leaving NDG
unchanged or worse. Its outcome is a diagnostic, not a stop condition. Continue
P4 and the remaining comparisons regardless of that pattern. Preserve the
measured outcomes rather than changing metrics or selecting favorable test
subsets after seeing results.

On 2026-09-11 the user replaced strict performance gates with this exploratory
policy and requested reusable experiment code. Prioritize explicit configuration,
resumable runs, shared data preparation, and complete diagnostic artifacts.
Numerical failures, invalid data provenance, or unavailable implementations
still require a fix or an explicit unresolved status; they are not valid results.

No real benchmark values are included in this repository. This document is a
preregistered comparison contract, not a result table or SOTA claim.
