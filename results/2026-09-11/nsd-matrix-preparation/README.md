# Full matrix preparation and predeclared evaluation cohort

No comparative VQA or caption model results are reported here. This bundle fixes
selection and verifies reusable dependencies before those runs begin.

`evaluation_cohort.json` selects 128 distinct shared-test images with one VQA
question each, covering all 32 observed categories. Selection uses seed41001:
visit categories from rarest to most common, choose up to four distinct images
per category by SHA-256 rank, then fill remaining slots from globally ranked
unused images. It uses image/question IDs and category metadata, not answers or
model outcomes. The rare toy/toilet categories each have only one test question.
The same cohort is used for all subjects, methods and seeds; raw trial IDs for
all three presentations are fixed for each subject. Main evaluation uses repeat0;
repeat diagnostics use fixed prefixes0,0+1,0+1+2 and are not repeat-choice averages.
Caption evaluation uses these same128images and their full held-out references.

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
manifest, original provenance, independent migration evidence and all24model
hashes. The original artifact files were not rewritten. New preparation code
binds row identities natively. The evidence verified that all37000image IDs are
exactly the original script's sorted unique manifest IDs.

`caption_metric_preflight.json` records pycocoevalcap1.2, Java11 and Stanford
CoreNLP3.6 model hashes plus successful CIDEr/METEOR/ROUGE-L/SPICE plumbing checks
on handcrafted sentences. Those values are not model benchmark results. SPICE
object precision/recall is the declared reference-grounded factual-object proxy;
it is not an independent CHAIR evaluation or a claim of complete factuality.
Undefined categories are retained as undefined with their denominators.
