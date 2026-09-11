# Brain NPP-OPSD

This repository implements the model-independent mathematics and validation
surface for Neural Posterior-Predictive On-Policy Self-Distillation
(NPP-OPSD). It includes posterior/target/loss primitives, leakage-safe NSD
manifest validation, brain-input controls and metrics, a deterministic CPU toy
optimizer, and a fail-closed adapter boundary for local VINDEX/LLaVA-7B or
BrainJanus-7B integrations.

The primary target is the arithmetic posterior-predictive density ratio:

```math
m_w(v)=\sum_i w_iT_i(v),\quad m_r(v)=\sum_i r_iT_i(v),\quad
Q(v)\propto S_{\mathrm{anchor}}(v)\left[\frac{m_w(v)}{m_r(v)}\right]^\lambda.
```

The default is `pooling=arithmetic`, `ratio_strength` \(\lambda=1\), and a
true unclipped forward KL. Expected-log (geometric) aggregation and the
official OPSD pointwise-clipped surrogate are available only through
separately named ablations. The rollout mask is carried by an immutable
`Rollout`; teacher construction is detached and completed before the single
live student rollout forward. Optional CE uses gold `target_ids/target_mask`,
never the student's own sampled tokens as labels.

The repository does **not** bundle NSD, VINDEX, LLaVA-7B, BrainJanus-7B, or
their checkpoints. No real NSD benchmark has been run from this repository,
and this repository makes no numerical or state-of-the-art claim.

## Install and verify

From the repository root:

```bash
python -m pip install -e '.[test]'
if ! command -v brain-npp >/dev/null 2>&1; then
  export PATH="$PATH:$(python -m site --user-base)/bin"
fi
python -m pytest -q
brain-npp smoke --config configs/toy_cpu.yaml
```

The conditional `PATH` fallback is needed only when `brain-npp` is not found
after installation. It appends the user-site scripts directory, preserving
the precedence of commands from an active virtual environment.

The smoke command runs real PyTorch optimizer steps on deterministic synthetic
data. It does not stand in for an NSD or 7B-model experiment.

## Prepare and validate an NSD manifest

The input CSV must contain exactly one row per single trial and these columns:

```text
trial_id,subject_id,session_id,run_id,trial_index,image_id,repeat_id,split,brain_path,image_path
```

Convert and validate it with:

```bash
python scripts/prepare_nsd_manifest.py \
  --input-csv /absolute/path/to/nsd_trials.csv \
  --output-jsonl /absolute/path/to/nsd_manifest.jsonl
brain-npp validate-manifest /absolute/path/to/nsd_manifest.jsonl
```

The converter preserves every `trial_id` and repeat row. It calls the package's
canonical `TrialRecord` and `validate_manifest` implementation, so repeated
presentations of an `image_id` cannot cross splits. A conflict exits nonzero
and names the image and both splits.

## What is runnable here

- `python -m pytest -q`: package and CLI tests.
- `brain-npp smoke --config configs/toy_cpu.yaml`: synthetic CPU optimization.
- `scripts/prepare_nsd_manifest.py` and `brain-npp validate-manifest`: local
  manifest conversion and validation.
- `brain-npp train --config ...`: only the integration entrypoint. For an
  external backend it requires local data, checkpoints, and a pinned checkout
  containing `brain_npp_factory.py`; it fails loudly if they are unavailable.

The checked-in P1-P4 YAML files are explicit contracts for that external
factory, not claims that core `brain_npp` implements VINDEX/BrainJanus data
loading or the named baseline algorithms. See [the reproduction guide](docs/reproduction.md)
for setup and commands, and [the benchmark matrix](docs/benchmark_matrix.md)
for required comparisons and controls.

The exact arithmetic objective, invariants, and evaluation matrix are recorded
in [the method contract](docs/arithmetic_npp_method.md).

## Real experiment profiles

- `configs/nsd/vindex_llava7b_npp.yaml`: primary NSD-VQA profile.
- `configs/nsd/vindex_llava7b_captioning.yaml`: secondary captioning profile.
- `configs/nsd/brainjanus7b_npp.yaml`: BrainJanus-7B extension profile.
- `configs/nsd/p1_*.yaml` through `p4_*.yaml`: staged reproduction contracts.
- `configs/hardware/gpu16gb.yaml` and `gpu24gb.yaml`: bounded local resource
  envelopes consumed by the external runner.

The full design contract is
[`docs/superpowers/specs/2026-09-10-npp-opsd-design.md`](docs/superpowers/specs/2026-09-10-npp-opsd-design.md).
