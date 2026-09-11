# Real NSD exploratory matrix: incremental results

**1 of 164 cases is complete.** The remaining comparisons are running; this is
not a complete benchmark table. Accuracy and neural-dependence measurements
are diagnostics, without performance stopping thresholds.

The first case is subject01, seed41, CE: 256 training examples,16 AdamW updates,
with the frozen BrainX/NF4 LLaVA base and trainable projector plus q/v LoRA.
It includes all five main controls and correct-brain repeat-prefix2/3 checks,
for both trained and original students:1,786 unique predictions in total.

| Student | Stratified-subset accuracy (128 images) | Category macro (32 categories) | Matched correct-minus-shuffled NDG (125 images) |
|---|---:|---:|---:|
| CE,16 updates | 42.96875% | 40.64236% | 7.2 percentage points |
| Original pretrained | 44.53125% | 44.39236% | 9.6 percentage points |

These are descriptive single-subject/single-seed measurements. This case does
not show a task-accuracy improvement over the original student; it does not
establish a general comparative conclusion. Paired uncertainty, exposure
sensitivity and cross-method summaries will be added with the reporting bundle.
NDG uses the same125 matched original examples on both sides, not128-minus125
marginal accuracy. Same-image wrong-subject is a transfer diagnostic.

The fixed128-image cohort covers32 categories; microaccuracy is not the natural
full-test prevalence-weighted accuracy. Two images,nsd33245 andnsd43156, were
exposed in an earlier engineering pilot and remain declared for exclusion
sensitivity. NSD-VQA references are auto-generated alternatives, not human
consensus. The TRAIN-only K=4 candidate gallery has zero held-out target
coverage by construction, so conditional NLL/Brier/ECE remain undefined.

Each compressed case archive contains the original prediction journals,
per-example scores, journal identities, summaries, selected examples, schedule,
resolved config and full checkpoint identity. Internal SHA256SUMS verifies every
archived data file. The adjacent JSON summary is readable without unpacking.
Model/optimizer tensors remain local; their checkpoint hash is recorded.
The deterministic packaging/export scripts accompany these artifacts.

The controlled continuation test interrupted the first case after update1
(cursor16), then used a fresh normal CLI process. It completed update16/cursor256,
preserved the saved history prefix and produced no duplicate predictions.
The evidence is in resume-verification/. The queue continues with the remaining
163 fixed configs; only actual execution failures require debugging.
