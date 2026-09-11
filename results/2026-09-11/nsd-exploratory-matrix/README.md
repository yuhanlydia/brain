# Real NSD exploratory matrix: incremental results

**13 of 164 cases are complete.** The remaining comparisons are running; this is
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

## Second completed case: exact-image OPSD

Subject01/seed41 exact-image OPSD consumed the same 256 examples and 16 updates
and completed 893 predictions across all required trained-student controls and
repeat diagnostics. Stratified-subset accuracy is 51.5625% (66/128).

Relative to the matched CE case, the paired accuracy difference is +8.59375
percentage points (image-bootstrap 95% interval: -3.125 to +19.53125). The joint
125-example delta-NDG is +7.2 points (interval: -2.4 to +16.8). These exploratory
intervals include zero; this is not a significance or generalization claim.
The full matrix continues without selecting runs based on these outcomes.

`report_snapshot.json.gz` contains the reviewed reporter's current partial
matrix snapshot: paired comparisons, category/session and posterior summaries,
exposure-exclusion sensitivity, coverage/status and input provenance. Its
unfinished cases remain explicitly incomplete or pending. This snapshot is
updated incrementally; the case archives preserve completed raw outputs.

## Third completed case: MAP-image OPSD

Subject01/seed41 MAP-image OPSD completed the same 256-example, 16-update
budget and 893 required predictions. Accuracy is 48.4375% (62/128), with
category macro accuracy 50.20833%. Its matched 125-example NDG is 12.8 points
(image-bootstrap 95% interval: 4.8 to 20.8).

Relative to matched CE, the paired accuracy difference is +5.46875 points
(interval: -7.03125 to +17.1875), and delta-NDG is +5.6 points
(interval: -4.0 to +15.2). These are exploratory comparisons from one subject
and seed; the full set of methods, subjects and seeds is still running.

`CURRENT_RESULTS.md` provides the readable completed-case metric tables and
full partial-matrix status, using the same reviewed values as the snapshot.

## Fourth completed case: uniform mixture

Subject01/seed41 uniform mixture completed the same 256 examples and 16 updates,
with 893 predictions across the required controls and repeat diagnostics.
Accuracy is 49.21875% (63/128), and category macro accuracy is 50.98958%.
The paired comparisons and uncertainty appear in `CURRENT_RESULTS.md`;
these remain exploratory single-subject, single-seed results.

## Fifth completed case: plain posterior mixture

Subject01/seed41 plain posterior mixture completed the same 256-example,
16-update budget and 893 required predictions. Accuracy is 49.21875%,
with category macro accuracy 50.98958%.
Paired differences and uncertainty are recorded in `CURRENT_RESULTS.md`.
The remaining fixed matrix continues without performance stopping thresholds.

## Sixth completed case: CREDIT-style image contrastive

Subject01/seed41 CREDIT-style image contrastive completed the matched
256-example, 16-update budget and 893 required predictions. Accuracy is 50.78125%,
with category macro accuracy 52.55208%.
This is the documented CREDIT-style adaptation; it is not an official-code
reproduction. Paired differences and uncertainty appear in `CURRENT_RESULTS.md`.

## Seventh completed case: DAPD brain/text reference

Subject01/seed41 DAPD completed the matched 256-example, 16-update budget
and 893 required predictions. Accuracy is 20.3125%, with
category macro accuracy 20.31250%. The reference snapshot remains
at its initial state because the fixed budget ends before its 100-update
refresh interval. Paired comparisons and uncertainty are in `CURRENT_RESULTS.md`.
All observed results are retained; performance does not stop the queue.

## Eighth completed case: VAD-style full image

Subject01/seed41 VAD-style full image completed the matched 256-example,
16-update budget and 893 required predictions. Accuracy is 51.5625%, with
category macro accuracy 53.76736%. Paired comparisons and uncertainty
are recorded in `CURRENT_RESULTS.md`. This is the documented VAD-style adaptation;
it is not an official-code reproduction.

## Ninth completed case: NPP-OPSD

Subject01/seed41 NPP-OPSD completed the matched 256-example, 16-update
budget and 893 required predictions. Accuracy is 45.3125%, with
category macro accuracy 45.32986%. Paired comparisons and uncertainty are in `CURRENT_RESULTS.md`.

## Tenth completed case: NPP geometric

Subject01/seed41 NPP geometric completed the matched 256-example, 16-update
budget and 893 required predictions. Stratified-subset accuracy is 43.7500%
(category macro 42.13542%). Matched NDG is 8.00
percentage points; paired comparisons and uncertainty are in `CURRENT_RESULTS.md`.
The remaining fixed matrix continues without performance stopping thresholds.

## Eleventh completed case: NPP information gain

Subject01/seed41 NPP information gain completed the matched 256-example, 16-update
budget and 893 required predictions. Stratified-subset accuracy is 46.8750%
(category macro 46.73611%). Matched NDG is 9.60
percentage points; paired comparisons and uncertainty are in `CURRENT_RESULTS.md`.
The remaining fixed matrix continues without performance stopping thresholds.

## Twelfth completed case: NPP square-root information gain

Subject01/seed41 NPP square-root information gain completed the matched 256-example,
16-update budget and 893 required predictions. Stratified-subset accuracy is 45.3125%
(category macro 45.17361%). Matched NDG is 11.20
percentage points; paired comparisons and uncertainty are in `CURRENT_RESULTS.md`.
The remaining fixed matrix continues without performance stopping thresholds.

## Thirteenth completed case: subject02 CE

Subject02/seed41 CE completed the matched 256-example, 16-update budget and
1,786 required predictions across trained and original controls. Stratified-subset
accuracy is 39.0625% (category macro 40.10417%),
with matched NDG 2.40 percentage points. The remaining fixed
matrix continues without performance stopping thresholds.
