# Full calibration and validation retrieval audit

## Verdict

Approved for the stated scope: **validation-only, closed-gallery Gaussian retrieval diagnostics using subject-specific models fit only on training trials**. The saved results are internally consistent with the current manifest, feature array, calibration artifacts, shuffle mappings, and evaluation implementation. I found no score/label misalignment, train/validation or train/test image leakage, shuffle-constraint violation, or discrepancy between reported scalar metrics and persisted arrays.

These results do not measure hidden-test performance or VQA/task accuracy. They should not be described as either.

One interpretation limitation needs to remain explicit: the repeat-ranking curve uses a fixed prefix of presentations (repeat 0; repeats 0+1; repeats 0+1+2). It is not averaged over the three possible single presentations or the three possible pairs, so the one- and two-repeat points can contain presentation-order effects. The three-repeat point is the average of all available repeats. This does not invalidate the saved numbers, but labels such as “expected performance with N repeats” would overstate what was estimated.

## Evidence checked

### Fit population and fold exclusion

- `fit_full_calibration.py` filters the validated manifest to `split == "train"` before loading brain rows or fitting any model.
- For each subject, the persisted final model's `train_sample_ids` exactly equal the manifest's ordered training trial IDs.
- The final models contain zero validation-gallery image IDs and zero hidden-test image IDs in `train_image_ids`.
- Recomputed outer folds from the full training manifest with seed 1731 exactly equal every saved `oof_density.npz/fold_ids` array. For every subject and every fold 0--4, the held-out image set has empty intersection with that fold model's `train_image_ids`.
- Inner covariance folds in `GaussianEncodingLikelihood.fit` are grouped by image identity, so all repetitions of an image remain together. The final ridge coefficients are fit on all training trials; the residual covariance is estimated from image-grouped out-of-fold predictions.
- OOF trial IDs and image IDs exactly match the ordered subject training records, all OOF densities are finite, and each OOF mean exactly matches its provenance report.
- All calibration artifact SHA-256 values match the per-subject provenance. The current manifest, feature array, and `calibration.py` hashes match the hashes recorded by every calibration run.

Subject training/validation/test trial counts were:

| Subject | Train | Validation | Hidden test |
|---|---:|---:|---:|
| subj01 | 24,459 | 2,541 | 3,000 |
| subj02 | 24,273 | 2,727 | 3,000 |
| subj05 | 24,375 | 2,625 | 3,000 |
| subj07 | 24,405 | 2,595 | 3,000 |

### Validation selection and score alignment

- `evaluate_validation_retrieval.py` selects only records with the current subject and `split == "val"`.
- For every subject, persisted score rows exactly match the ordered validation trial IDs; columns exactly match the lexicographically sorted unique validation gallery; and every target equals the column index of that row's manifest image ID.
- Each validation image has exactly three trials with repeat IDs `[0, 1, 2]`. Gallery sizes are 847, 909, 875, and 865 for subj01, subj02, subj05, and subj07 respectively.
- Every probability row is normalized (maximum absolute row-sum error `8.88e-16`). The optimized whitened-distance calculation is mathematically aligned with `GaussianEncodingLikelihood.score`; the script also checks representative rows at runtime.

### Matched shuffle and HRF-locality exclusion

- Each persisted mapping is a bijection on its retained trials. Mapping keys plus reported unmatched trials form the complete validation trial set and are disjoint.
- Every retained source/replacement pair is from the same subject, session, and run; uses different image IDs; and has absolute within-run trial-index distance greater than 2. Thus the specified radius-2 local/HRF-neighbor exclusion is enforced.
- The shuffled score uses the replacement brain row's complete gallery score vector while retaining the source row's target label. That is the intended matched-wrong-brain comparison.
- Because the matcher discards an entire run block if it cannot construct a complete bijection, control metrics and their intervals apply to the reported matched subset, not necessarily all validation trials. Coverage is high but not complete: 2,463/2,541, 2,670/2,727, 2,579/2,625, and 2,534/2,595 trials respectively.

### Paired image-cluster bootstrap

- The bootstrap input is the paired per-source difference `correct - shuffled`, with identical retained rows on both sides.
- Resampling occurs over validation image IDs and concatenates all retained trials for each sampled image, preserving within-image repeat dependence.
- Re-running the seeded 1,000-draw bootstrap reproduces every point estimate and percentile interval exactly. The intervals are percentile Monte Carlo intervals with only 1,000 draws; report them at their saved precision cautiously. Since unmatched blocks can leave unequal retained cluster sizes, the implementation targets a retained-trial-weighted mean while resampling image clusters.

### Metrics reproduced from persisted arrays

All per-subject JSON files exactly equal their entries in `summary.json`. R@1, R@5 (including the implementation's cutoff-tie convention), candidate NLL, multiclass Brier score, 10-bin ECE, entropy, and uniform-gallery NLL reproduce exactly from each `*.scores.npz` file. The shuffle point estimates and confidence intervals also reproduce exactly.

| Subject | R@1 | R@5 | Candidate NLL | Uniform NLL | Matched R@1 | Shuffled R@1 | Difference | Image-cluster 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| subj01 | 0.02519 | 0.08579 | 5.77636 | 6.74170 | 0.02558 | 0.00162 | 0.02395 | [0.01784, 0.03131] |
| subj02 | 0.02237 | 0.08801 | 5.78171 | 6.81235 | 0.02247 | 0.00112 | 0.02135 | [0.01541, 0.02726] |
| subj05 | 0.04343 | 0.14590 | 5.39953 | 6.77422 | 0.04265 | 0.00039 | 0.04226 | [0.03411, 0.05088] |
| subj07 | 0.01927 | 0.06590 | 6.06888 | 6.76273 | 0.01894 | 0.00079 | 0.01815 | [0.01302, 0.02374] |

The repeat-ranking values also reproduce exactly by loading the saved final models and source beta arrays and reapplying the script's whitening, averaging, and ranking calculation.

## Reporting and provenance limitations

1. Treat repeat rankings for one and two repeats as fixed-prefix results. If the intended claim is performance as a function of repeat count independent of presentation choice, average metrics across all singleton/pair subsets or predeclare a repeat-selection rule and state it.
2. Retrieval JSON files record the final model hash but do not bind the manifest, feature file, beta file, evaluation script, score NPZ, or shuffle JSON by hash. The present audit matched them against the current files, but the retrieval bundle alone is not fully self-authenticating. Add those hashes to retrieval provenance when packaging the run.
3. Brier score and ECE concern the softmax distribution over the subject's fixed validation gallery under an implicit uniform candidate prior. They are closed-gallery retrieval calibration diagnostics, not unconditional calibration of the Gaussian brain density and not calibration for another gallery size or candidate distribution.
4. The shuffle confidence interval covers the matched subset and reflects image-cluster resampling conditional on the fixed mapping. It does not incorporate randomness across alternative derangements.

## Approved claims

- The subject-specific Gaussian encoding models used for retrieval were fit from training trials only, with image-grouped cross-fitting used for residual covariance estimation.
- On each subject's fixed validation gallery, the reported retrieval metrics and matched wrong-brain differences are correct for the persisted arrays and mappings.
- Hidden-test responses and images were not used in these fits or retrieval evaluations.
- Averaging all three validation repeats improves the reported ranking accuracy relative to the script's fixed first-repeat result for all four subjects.

Do not extend approval to hidden-test generalization, VQA accuracy, causal interpretation of the brain signal, repeat-choice-averaged performance, or calibration outside the fixed validation gallery.
