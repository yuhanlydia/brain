# Full matrix preparation and predeclared evaluation cohort

No comparative VQA or caption model results are reported here. This bundle fixes
selection and verifies reusable dependencies before those runs begin.

`evaluation_cohort.json` selects 128 distinct shared-test images with one VQA
question each, covering all 32 observed categories. Selection uses seed 41001:
visit categories from rarest to most common, choose up to four distinct images
per category by SHA-256 rank, then fill remaining slots from globally ranked
unused images. It uses image/question IDs and category metadata, not answers or
model outcomes. The rare toy/toilet categories each have only one test question.
The same cohort is used for all subjects, methods and seeds; raw trial IDs for
all three presentations are fixed for each subject. Main evaluation uses repeat 0;
repeat diagnostics use fixed prefixes 0, 0+1, 0+1+2 and are not repeat-choice averages.
Caption evaluation uses these same 128 images and their full held-out references.

Report micro accuracy as **stratified-subset accuracy**, alongside category
macro accuracy and image-cluster uncertainty. It is not full-test prevalence-
weighted accuracy. Preserve unmatched controls and their denominators. Training
update budgets will be declared after the real resource pilot and before any
comparative runs; neither this subset nor observed accuracy is a stopping gate.

`prepare_matrix_eval_cohort.py` reproduces the exact selection from the complete
prepared manifest/sidecars with configurable paths, image count and seed. Source
file hashes are embedded in the cohort. No raw brain data or generated answers
are redistributed here.

`row_mapping_binding.v1.json` explicitly migrates the already-audited legacy
feature/calibration provenance: it binds image-ID row order, feature bytes,
manifest, original provenance, independent migration evidence and all 24 model
hashes. The original artifact files were not rewritten. New preparation code
binds row identities natively. The evidence verified that all 37,000 image IDs are
exactly the original script's sorted unique manifest IDs.

`caption_metric_preflight.json` records pycocoevalcap 1.2, Java 11 and Stanford
CoreNLP 3.6 model hashes plus successful CIDEr/METEOR/ROUGE-L/SPICE plumbing checks
on handcrafted sentences. Those values are not model benchmark results. SPICE
object precision/recall is the declared reference-grounded factual-object proxy;
it is not an independent CHAIR evaluation or a claim of complete factuality.
Undefined categories are retained as undefined with their denominators.

## Subsequent engineering pilot and control cache

`engineering-pilot/` contains eight real frozen brain-only generations on the
first two cohort images across four subjects. Cached greedy decoding used at
most 64 new tokens, averaging 0.735 seconds/example with peak 5,164,656,640 CUDA bytes.
This is a resource/format pilot, not a trained-method comparison or accuracy
estimate. Its original prompt often elicited complete sentences; comparative
VQA uses one fixed short-answer instruction, recorded in the resolved config.
The two images `nsd33245` and `nsd43156` were thus exposed during engineering.
Keep the predeclared 128-image cohort, disclose that exposure, and additionally
report paired sensitivity summaries excluding those 2 images. Do not replace
images after seeing outcomes or silently use substring answer matching.

The first attempt failed on a logging field name and performed no optimizer
update; the corrected attempt completed all 8 cases. Supplemental provenance
binds actual model, checkpoint, source, manifest and predictions; it also
explains a null encoder-provenance logging field in the unchanged original report.

`control_preparation.json` records fixed shuffle/wrong-subject maps and cached
noise identities. Shuffle matched 11,780/12,000 shared-test trials and 125/128 main
cohort trials per subject for each seed. Wrong-subject matched all 12,000.
Noise has 128 native-voxel vectors per subject/seed, shared across methods; every
training reference array was checked against canonical trial provenance before
fitting. Large mappings, noise arrays and full reference-hash inventories remain
local with hashes reported here. The two preparation scripts reproduce/cache
these controls using reviewed data/control APIs.

## Predeclared training budget

`exploratory_budget.json` fixes the first complete available-backend comparison:
12 VQA methods/ablations × 4 subjects × 3 seeds, each with 256 training examples
and 16 AdamW updates (accumulation 16, learning rate 5e-6). Caption NPP uses the
same training budget across all four subjects and three seeds. Every trained
checkpoint receives the five inference controls; controls are not separate
training runs. Correct-condition repeat diagnostics are also required.

P3 CE/exact-image runs with seed41 instead consume all trials from the declared
10% training image-group subset, one selected QA per trial, including a correctly
scaled partial final accumulation window. The two P3 methods share that complete
schedule. This preserves the intended subset traversal while the primary matrix
has an explicitly limited training budget. No convergence or full-data training
claim follows from 16 updates. DAPD keeps its 100-update snapshot interval; the
16-update primary run ends before its first scheduled refresh.

`update-pilot/` records real CE, NPP and ten-view DAPD updates on a training trial,
with finite nonzero gradients/parameter changes. They measured resource use to
choose this budget. These engineering commands exercise model primitives; the
production configuration-to-run binding must be completed and checked before
comparative execution. They are not comparative accuracy results.
