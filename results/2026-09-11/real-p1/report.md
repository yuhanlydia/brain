# Task 2 report: real P1 VINDEX integration

Status: complete for the bounded real P1 scope only. This is not a full experiment or benchmark result.

## Implemented

- `src/brain_npp/adapters/vindex.py`: causal alignment, EOS-preserving masks, independent frozen projector snapshot, strict feature/calibration hash and ID validation, fold-exclusion enforcement, recomputed Gaussian likelihoods, and streamed frozen-image teacher logits.
- `integrations/vindex/brain_npp_factory.py`: lazy PEFT/Transformers/bitsandbytes imports; LLaVA-7B NF4 plus q/v LoRA; strict projector loading; subject-01 correct-brain NPP-OPSD P1-only validation; AdamW update; gradient/freeze/delta diagnostics; optimizer/trainable/RNG checkpoint state; and content-validated completed-run reuse.
- Versioned feature and calibration preparation entrypoints under `scripts/`.
- Optional `vindex` dependencies and concise setup documentation.
- Focused tests for causal slicing, EOS masking, frozen snapshot independence, artifact score recomputation, hash validation, and fold leakage rejection.

## Verification commands and results

Focused TDD red:

```text
PYTHONPATH=src python3 -m pytest -q tests/test_vindex_adapter.py
4 failed: ModuleNotFoundError: brain_npp.adapters.vindex
```

Focused green:

```text
PYTHONPATH=src python3 -m pytest -q tests/test_vindex_adapter.py
4 passed in 1.19s
```

CPU regression before and after the GPU execution:

```text
PYTHONPATH=src python3 -m pytest -q
105 passed in 26.59s

PYTHONPATH=src python3 -m pytest -q
105 passed in 26.79s
```

Actual P1 command through the unchanged external CLI boundary:

```text
PYTHONPATH=src python3 -m brain_npp.cli train --config configs/nsd/local_p1_vindex_real_shape_smoke.yaml
```

Key first-run results:

- status: `completed_real_p1`
- selected examples: 8; actually consumed: 1
- optimizer steps: 1; LR: 1e-4; weight decay: 0.0
- rollout active tokens: 19 (generation bound was 64)
- NPP loss: 0.08948653067570392
- projector gradient norm: 2.5184500488754837, finite
- LoRA gradient norm: 0.30915340230551175, finite
- combined optimizer delta norm: 0.4775803670822295
- frozen base has gradient: false
- teacher projector unchanged: true
- peak CUDA allocation: 5,428,183,040 bytes
- feature and all five fold hashes matched declared provenance; likelihoods were recomputed from each assigned fold model.

That round-one second invocation was only an idempotent completed-result lookup, despite the earlier `resumed` label. It was preserved under `outputs/p1_vindex_real_shape_smoke_seed41/history-round1/` and is not continuation evidence.

## Review round two

The reusable `save_training_state` / `load_training_state` API now restores named trainable tensors, complete optimizer state, and Python, NumPy, Torch, and CUDA RNG. A tiny AdamW split-run test restored into fresh objects and matched an uninterrupted second update exactly. Terminal P1 reruns are accurately labeled `completed_run_reused: true`; they are not described as resumed training.

Preparation was rerun after verifying UMBRAE revision `8c1795c...`, VINDEX revision `ad0dab49...`, BrainXS checkpoint hash `5e300e73...`, CLIP checkpoint-tree hash `676e9e6a...`, and every source brain/image file. The regenerated feature artifact retained hash `c2729734...`; calibration was refit and validated. The runner now hashes the full model tree, VINDEX checkout, projector, VQA, manifest, features/provenance, calibration tree, and config before accepting a completed-result cache.

Review-focused TDD initially failed 8 tests for mixed ID coercion, absent continuation restore, and missing scope validation. The implemented set passed:

```text
PYTHONPATH=src python3 -m pytest -q tests/test_calibration.py tests/test_vindex_adapter.py tests/test_vindex_factory.py
34 passed in 2.41s

PYTHONPATH=src python3 -m pytest -q
116 passed in 26.73s
```

The final fresh post-review GPU P1 reproduced the same one-step numerical diagnostics. It recorded Python 3.10.12, Torch 2.13.0+cu126, CUDA 12.6, cuDNN 91002, Transformers 4.45.2, PEFT 0.13.2, Accelerate 1.0.1, and bitsandbytes 0.50.2. Its final VINDEX checkout content hash was `141636ce...`; a subsequent invocation recomputed all material input hashes before returning `completed_run_reused: true` with no duplicate update.

Ignored local artifacts:

- config: `configs/nsd/local_p1_vindex_real_shape_smoke.yaml`
- results/checkpoint: `outputs/p1_vindex_real_shape_smoke_seed41/`

## Concerns and limits

- This run uses the deliberately small preparation subset (33 images, 46 trials), not the full NSD benchmark.
- It validates real tensors, calibration isolation, causal alignment, trainable gradients, one update, reusable continuation-state mechanics, and terminal completed-run reuse. It makes no task-quality or accuracy claim.
- Three shape-only P1 preparation trials fall in the newly declared full-development validation split. P1 weights and calibration must never initialize or otherwise influence full benchmark evaluation; full runs start fresh and refit on full-train data only.
- P2/P3/P4, additional subjects, controls, baselines, captioning, pooling/strength sweeps, and BrainJanus remain outside this implementation and fail closed.
- The frozen teacher shares the immutable quantized base module with the student while adapters are disabled; all base parameters are frozen and had no gradients. Its projector is an independent exact initial snapshot and was unchanged after training.
