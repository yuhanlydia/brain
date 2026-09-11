# Verification and synthetic experiment results — 2026-09-11

Evaluated code: `e9197ef53ed74df2822ed7544b71e97d4b64bfe7` (main after [PR #1](https://github.com/yuhanlydia/brain/pull/1)).
Includes requested commit `302e64c` and numerical-stability fix `00be1e0`.

**Real NSD / 7B experiments completed: 0.** These measurements use deterministic synthetic data and demonstrate executable optimization only. They are not NSD accuracy, evidence of neural dependence, or a scientific performance claim.

## Verification

- Full suite on merged main: **81 passed in 25.44s**; see `pytest_main.log`.
- Python compilation and `git diff --check`: passed.
- CPU: **12/12** runs lowered mean NPP loss; prefixes remained consistent.
- CUDA: **3/3** arithmetic runs on RTX 3090 lowered mean NPP loss; diagnostics were finite.
- External preflight: all **9** tracked NSD profiles exited with code 2 because required paths were unset. See `external_preflight.json` for exact commands' configuration paths and errors.

The original `302e64c` tree failed nine existing posterior tests. Stable log-softmax normalization and evidence centering repaired loss of prior/posterior mass and invariance under common score shifts. Two added regression tests cover zero-prior extreme candidates and compatibility-temperature scaling; an independent review and re-review checked these cases.

## Environment

Python 3.10.12; PyTorch 2.13.0+cu126; NVIDIA GeForce RTX 3090, 24 GB. CPU runs used two OpenMP/MKL threads. Each run used 30 real optimizer updates; reported initial/final values average the first/last five step losses. CUDA runs used the primary default arithmetic trainer, Adam learning rate 0.08, and seeds 41/42/43.

## CPU results

| Configuration and seed | Initial NPP loss | Final NPP loss |
|---|---:|---:|
| toy_cpu_seed314159 | 0.291079432 | 0.012887472 |
| toy_cpu_seed41 | 0.286682034 | 0.011022796 |
| toy_cpu_seed42 | 0.282228661 | 0.005578107 |
| toy_cpu_seed43 | 0.280573717 | 0.007381931 |
| toy_cpu_geometric_kl_seed314159 | 0.102484323 | 0.005263935 |
| toy_cpu_geometric_kl_seed41 | 0.101709057 | 0.005200922 |
| toy_cpu_geometric_kl_seed42 | 0.106465469 | 0.002680317 |
| toy_cpu_geometric_kl_seed43 | 0.102425909 | 0.003254560 |
| toy_cpu_raw_strength_seed314159 | 0.091116819 | 0.010747529 |
| toy_cpu_raw_strength_seed41 | 0.091279581 | 0.008316394 |
| toy_cpu_raw_strength_seed42 | 0.089919610 | 0.002731237 |
| toy_cpu_raw_strength_seed43 | 0.089292800 | 0.003087421 |

`toy_cpu_geometric_kl` is the historical geometric + bounded information-gain configuration; it changes more than pooling and is not a pooling-only comparison. `toy_cpu_raw_strength` is the arithmetic information-gain strength ablation. The primary method is `toy_cpu` (arithmetic, constant ratio strength 1).

Every CPU JSON contains the per-step loss sequence, final diagnostics, seed, configuration path and evaluated commit. `cuda_synthetic.json` contains all 30 diagnostic records for each CUDA run. `checksums.sha256` inventories the result files.

## Reproduce

Run from the repository root with the package installed:

```bash
python3 -m pip install -e '.[test]'
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 -m pytest -q
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 results/2026-09-11/reproduce.py
```

The reproduction script writes to ignored `artifacts/reproduced_2026-09-11/`. CUDA is required to reproduce the three GPU records. Different PyTorch/device versions may introduce numerical differences. The script has been executed on the recorded environment and compared with the archived results.

## Real-experiment readiness

Pinned source checkouts inspected:

- [VINDEX](https://github.com/weihaox/VINDEX/tree/ad0dab49e99d18a179097e9b79246f7ca3559f1b): `ad0dab49e99d18a179097e9b79246f7ca3559f1b`.
- [BrainJanus](https://github.com/HaitaoWuTJU/BrainJanus/tree/6d797688c26c0f2ac2ff2be21f4598fe2b1ed946): `6d797688c26c0f2ac2ff2be21f4598fe2b1ed946`.

Neither checkout provides the repository-specific `brain_npp_factory.py` integration. VINDEX's default inference averages repeats; BrainJanus documents averaged test data. These paths cannot directly supply the single-trial/session/run controls required by this project's method contract.

Before P1–P4, captioning and BrainJanus runs can execute, supply single-trial NSD assets and a validated image-group split manifest, task labels, model/brain-encoder checkpoints, and a conforming backend runner. Local ignored configurations for all nine profiles have been prepared with the source checkout locations; missing data/model/manifest values remain unset. `readiness.json` records unresolved dependencies. No missing resource was replaced by synthetic data for real-experiment profiles.
