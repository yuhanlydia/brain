"""Reproduce the synthetic runs recorded with this result bundle."""

import json
import math
from pathlib import Path
import subprocess

import torch

from brain_npp.cli import load_config, run_smoke
from brain_npp.toy import ToyAdapter
from brain_npp.trainer import NPPTrainer


def main():
    torch.set_num_threads(2)
    output = Path("artifacts/reproduced_2026-09-11")
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    summary = []
    for path in [Path("configs/toy_cpu.yaml"), *sorted(Path("configs/ablations").glob("*.yaml"))]:
        config = load_config(path)
        for seed in [config["seed"], 41, 42, 43]:
            result = run_smoke({**config, "seed": seed})
            result.update(commit=commit, synthetic=True, seed=seed, config=str(path))
            name = f"{path.stem}_seed{seed}"
            (output / f"{name}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            summary.append({"run": name, "commit": commit,
                            "initial": result["initial_npp_loss_mean"],
                            "final": result["final_npp_loss_mean"],
                            "prefix_consistent": result["prefix_consistent"]})
    (output / "toy_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce the recorded GPU runs; CPU runs were saved")
    gpu = []
    for seed in [41, 42, 43]:
        torch.manual_seed(seed)
        adapter = ToyAdapter(seed=seed).cuda()
        batch = {key: value.cuda() for key, value in adapter.make_batch().items()}
        trainer = NPPTrainer(adapter, torch.optim.Adam(adapter.parameters(), lr=0.08))
        diagnostics = [trainer.step(batch) for _ in range(30)]
        trainer.flush()
        losses = [row["npp_loss"] for row in diagnostics]
        initial, final = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
        assert all(math.isfinite(value) for row in diagnostics for value in row.values())
        assert final < initial and adapter.prefix_consistent
        gpu.append({"commit": commit, "seed": seed,
                    "device": torch.cuda.get_device_name(), "synthetic": True,
                    "initial": initial, "final": final, "diagnostics": diagnostics})
    (output / "cuda_synthetic.json").write_text(json.dumps(gpu, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"cpu_smokes_passed": len(summary), "cuda_smokes_passed": len(gpu)}))


if __name__ == "__main__":
    main()
