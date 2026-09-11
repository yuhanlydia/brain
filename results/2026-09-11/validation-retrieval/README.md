# Validation-only Gaussian retrieval diagnostics

All four subjects show positive matched correct-minus-shuffled retrieval differences. These are closed-gallery validation diagnostics, not VQA accuracy or hidden-test results. Performance thresholds do not gate subsequent experiments.

| Subject | Gallery | Matched trials | Correct R@1 | Shuffled R@1 | Difference (percentage points) | Image-cluster 95% CI (pp) |
|---|---:|---:|---:|---:|---:|---|
| subj01 | 847 | 2463/2541 | 2.56% | 0.16% | 2.40 | [1.78, 3.13] |
| subj02 | 909 | 2670/2727 | 2.25% | 0.11% | 2.13 | [1.54, 2.73] |
| subj05 | 875 | 2579/2625 | 4.27% | 0.04% | 4.23 | [3.41, 5.09] |
| subj07 | 865 | 2534/2595 | 1.89% | 0.08% | 1.82 | [1.30, 2.37] |

The comparison uses the same retained trials on both sides. Shuffle maps are within subject/session/run, exclude self and repeated images, and exclude trial distances of two or fewer. Intervals use 1,000 paired image-cluster bootstrap draws, conditional on the saved shuffle mapping; they do not capture variation over alternative mappings.

All 37,000 corpus images have fixed CLIP pooler features. Each subject has five outer-fold models and a final Gaussian model fit from its training trials only. Image-grouped inner folds estimate residual covariance. Validation and hidden-test image identities are excluded from fitting. The public beta preprocessing was supplied upstream and was not re-estimated here.

`summary.json` also reports all-trial R@1/R@5, candidate NLL, Brier, ECE and entropy. Those probabilities are conditional on the fixed validation gallery and a uniform prior; they do not establish calibration for another candidate policy. One- and two-repeat rankings use fixed first-presentation prefixes, not averages over all repeat choices. Three-repeat rankings use all presentations. No probability-calibration claim is made for repeat means.

`audit.md` records independent reproduction and exclusions. `provenance.json` binds the actual manifest, features, beta arrays, models, evaluation code, score arrays and mappings by SHA-256. Raw brain arrays, weights and per-trial score arrays remain in local artifact storage; their hashes are included without committing them. The three Python files are exact snapshots of the scripts used for this run and contain local paths; reusable configurable preparation entrypoints are being integrated separately.

The full matrix starts from original released weights, not the earlier P1 smoke checkpoint. P1 used three trials overlapping the subsequently declared full validation split and is retained only as an engineering probe.
