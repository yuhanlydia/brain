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
In particular, it must not ignore or override `trainer.alpha_scale` or
`trainer.alpha_max`. The primary trainer and every primary toy/NSD NPP-OPSD
profile use `alpha_scale: 1.0`, `alpha_max: 1.0`:
`alpha = min(information_gain / alpha_scale, alpha_max)`.
These are fixed configuration values, not a trainable gate.

The low-level `build_npp_target(..., alpha_max=None)` default remains the raw
mathematical option `alpha = information_gain`; in raw mode `alpha_scale`
does not rescale the value. The dedicated
`configs/ablations/toy_cpu_raw_strength.yaml` opts into it with
`alpha_max: null` and can be run with:

```bash
brain-npp smoke --config configs/ablations/toy_cpu_raw_strength.yaml
```

Label this as the raw-strength ablation, not the primary method. The toy
fixture's information gain is below the cap, so its raw and bounded smoke
curves can coincide; the regression suite also exercises information gain
above one to verify that the two configurations produce distinct targets.

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

Proceed to the full NPP claim only after checking whether exact-image OPSD
improves task quality while NDG is unchanged or worse, as specified by the
go/no-go rule.

### P4: NPP-OPSD seeds 41, 42, and 43

Each P4 file contains one explicit seed and all five required brain controls:

```bash
for seed in 41 42 43; do
  brain-npp train --config "configs/nsd/local_p4_vindex_npp_seed${seed}.yaml"
done
```

Aggregate across the three completed runs only after verifying each run used
the same split, trainable parameters, update budget, and teacher-forward budget.
Success requires NPP-OPSD to improve both task quality and NDG over the stated
comparators; improvement in only one is not a positive method result.

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
