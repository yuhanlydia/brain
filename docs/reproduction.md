# NPP-OPSD reproduction workflow

This guide separates commands implemented by this repository from contracts
that require locally supplied research assets. Run commands from the repository
root.

## 1. Local package verification

Install the package in editable mode with its test dependency, then run the
full suite and deterministic CPU smoke:

```bash
python -m pip install -e '.[test]'
if ! command -v brain-npp >/dev/null 2>&1; then
  export PATH="$PATH:$(python -m site --user-base)/bin"
fi
python -m pytest -q
brain-npp smoke --config configs/toy_cpu.yaml
```

The conditional fallback applies only when `brain-npp` is not found after
installation. It appends the user-site scripts directory so an active virtual
environment's commands keep precedence. If needed, keep that appended `PATH`
in the shell used for the remaining commands.

These commands exercise the package's real posterior, loss, trainer, controls,
metrics, CLI, and synthetic optimizer. They do not load NSD or a 7B model.

## 2. Prepare the single-trial manifest

Create a CSV with exactly this header and one row per NSD presentation:

```csv
trial_id,subject_id,session_id,run_id,trial_index,image_id,repeat_id,split,brain_path,image_path
subj01_s01_r01_t000,subj01,session01,run01,0,nsd00001,0,train,/data/betas/t000.npy,/data/images/nsd00001.png
subj01_s02_r01_t017,subj01,session02,run01,17,nsd00001,1,train,/data/betas/t017.npy,/data/images/nsd00001.png
```

`trial_id` identifies a single presentation; it is never replaced by
`image_id`. `trial_index` and `repeat_id` are integers, and every repeat of an
`image_id` must use the same split.

```bash
python scripts/prepare_nsd_manifest.py \
  --input-csv /absolute/path/to/nsd_trials.csv \
  --output-jsonl /absolute/path/to/nsd_manifest.jsonl
brain-npp validate-manifest /absolute/path/to/nsd_manifest.jsonl
```

Preparation constructs the package's `TrialRecord` values and calls
`brain_npp.data.validate_manifest`; there is no second implementation of the
image-leakage rule. Invalid input exits nonzero before the output is written.

Question, answer, caption, nuisance, and cross-fit metadata may be kept by the
external dataset implementation in side tables keyed by `trial_id` and
`image_id`. The core JSONL loader currently accepts the ten fields above.

## 3. External backend integration contract

Real-shape and benchmark stages require all of the following local assets:

1. NSD single-trial GLMsingle betas and images;
2. a validated manifest;
3. a local VINDEX checkout plus LLaVA-7B checkpoint, or a local BrainJanus-7B
   checkout/checkpoint;
4. a `brain_npp_factory.py` file at the root of that pinned checkout.

For VINDEX, the checked-in YAML templates leave these paths blank:

```yaml
paths:
  vindex_checkout: /absolute/path/to/pinned-vindex
  model: /absolute/path/to/llava-7b-checkpoint
  data: /absolute/path/to/nsd
  manifest: /absolute/path/to/nsd_manifest.jsonl
```

Core `brain-npp train` validates the checkout, model, and data paths, loads
`<vindex_checkout>/brain_npp_factory.py`, calls `create_adapter(config)`, and
then calls the returned runner's `train(config)`. It passes the complete YAML
mapping through unchanged. The external factory must validate `paths.manifest`
and return an object with this contract:

```python
from collections.abc import Mapping
from typing import Any

def create_adapter(config: Mapping[str, Any]):
    """Return a runner exposing train(config) -> Mapping[str, Any]."""
```

The runner—not the core package—must load `hardware_profile`, implement the
named `task`, `method`/`experiment.methods`, enforce `experiment` limits,
construct leakage-safe datasets and controls, run training/evaluation, and
return finite JSON-serializable diagnostics. It must fail rather than use toy
data when any requested real asset or method is missing.

For NPP-OPSD the runner must forward the complete `trainer` mapping into
`NPPTrainer(adapter, optimizer, generation_config=config.get("generation", {}),
**config.get("trainer", {}))`, or implement precisely the same field semantics.
It must preserve the explicit objective fields. The primary profiles use
`pooling: arithmetic`, `ratio_strength: 1.0`, `strength_mode: constant`, and
`score_semantics: log_likelihood`. Thus the teacher correction is

```math
\log m_w(v)-\log m_r(v),\qquad
m_w(v)=\sum_i w_iT_i(v),\quad m_r(v)=\sum_i r_iT_i(v),
```

and the detached on-policy target is

```math
\log Q(v)=\log S_{\mathrm{anchor}}(v)
+\lambda[\log m_w(v)-\log m_r(v)]-\log Z,
\qquad \lambda=1.
```

The dedicated `configs/ablations/toy_cpu_raw_strength.yaml` changes only the
strength rule to `information_gain`; it can be run with:

```bash
brain-npp smoke --config configs/ablations/toy_cpu_raw_strength.yaml
```

Label this as an information-gain strength ablation, not the primary method.
The historical expected-log aggregation is separately isolated in
`configs/ablations/toy_cpu_geometric_kl.yaml`; never report it as arithmetic
posterior-predictive mixing.

Create ignored, machine-local configs from the tracked templates and fill the
four paths in each file:

```bash
cp configs/nsd/p1_vindex_real_shape_smoke.yaml configs/nsd/local_p1_vindex_real_shape_smoke.yaml
cp configs/nsd/p2_vindex_baselines.yaml configs/nsd/local_p2_vindex_baselines.yaml
cp configs/nsd/p3_vindex_bypass_seed41.yaml configs/nsd/local_p3_vindex_bypass_seed41.yaml
cp configs/nsd/p4_vindex_npp_seed41.yaml configs/nsd/local_p4_vindex_npp_seed41.yaml
cp configs/nsd/p4_vindex_npp_seed42.yaml configs/nsd/local_p4_vindex_npp_seed42.yaml
cp configs/nsd/p4_vindex_npp_seed43.yaml configs/nsd/local_p4_vindex_npp_seed43.yaml
```

The `configs/nsd/local_*.yaml` files are ignored so checkpoint locations are
not committed.

## 4. Staged run commands

The commands below use only the existing `brain-npp train --config` interface.
Their phase-specific fields are consumed by the external factory contract
above; the core package does not implement those phases itself.

### P1: eight-example real-shape smoke

The P1 template fixes seed 41, `max_examples: 8`, and one optimizer step. Once
its local paths and factory are present, run:

```bash
brain-npp train --config configs/nsd/local_p1_vindex_real_shape_smoke.yaml
```

The external runner must verify real NSD tensor/model shapes, one bounded
forward/backward/update, fixed student prefixes across all teacher candidates,
and finite diagnostics. This is not the synthetic CPU smoke.

### P2: baseline reproduction

The P2 template requests CE, exact-image OPSD, MAP-image OPSD, uniform mixture,
plain posterior mixture, CREDIT-style, DAPD, and VAD-style under matched data
and compute budgets:

```bash
brain-npp train --config configs/nsd/local_p2_vindex_baselines.yaml
```

The external runner must emit task-quality and NDG metrics for every supported
cell in the benchmark matrix, and fail explicitly for an unavailable baseline.

### P3: ten-percent seed-41 neural-bypass diagnostic

The P3 template fixes `data_fraction: 0.10`, seed 41, and compares CE with
exact-image OPSD under correct and shuffled inputs:

```bash
brain-npp train --config configs/nsd/local_p3_vindex_bypass_seed41.yaml
```

Record whether exact-image OPSD improves task quality while NDG is unchanged
or worse as a bypass diagnostic. Continue P4 and every remaining experiment
regardless of this pattern; the user removed performance stop conditions on
2026-09-11.

### P4: NPP-OPSD seeds 41, 42, and 43

Each P4 file contains one explicit seed and all five required brain controls:

```bash
for seed in 41 42 43; do
  brain-npp train --config "configs/nsd/local_p4_vindex_npp_seed${seed}.yaml"
done
```

Aggregate across the three completed runs only after verifying each run used
the same split, trainable parameters, update budget, and teacher-forward budget.
Report task quality and NDG changes separately, with uncertainty and negative
results. An improvement in either can be a positive exploratory finding; there
is no fixed threshold or requirement that both improve. Task-quality gains
alone do not establish neural dependence.

## 5. Captioning and BrainJanus profiles

`configs/nsd/vindex_llava7b_captioning.yaml` supplies the required secondary
captioning profile, and `configs/nsd/brainjanus7b_npp.yaml` supplies the
BrainJanus extension profile. Copy either to an ignored `local_*.yaml`, fill
its paths, and invoke the same supported command:

```bash
cp configs/nsd/vindex_llava7b_captioning.yaml configs/nsd/local_vindex_llava7b_captioning.yaml
brain-npp train --config configs/nsd/local_vindex_llava7b_captioning.yaml
```

These profiles are integration contracts. Without the relevant local checkout,
checkpoint, NSD data, manifest, and implemented factory runner, the command is
expected to fail with an actionable error.

## 6. Result status

The repository currently contains no real NSD benchmark measurements. The toy
loss decrease validates executable mechanics only; it is not scientific
evidence and must not be reported as an NSD result or SOTA claim.
