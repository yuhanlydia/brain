"""Pinned VINDEX factory for the bounded subject-01 real P1 execution."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from brain_npp.adapters.vindex import VindexP1Adapter, frozen_snapshot, load_validated_p1_artifacts, save_training_state
from brain_npp.trainer import NPPTrainer
from brain_npp.provenance import manifest_hash, realized_environment, sha256_file, sha256_tree
from brain_npp.nsd_experiment import run_matrix_config


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _training_input_hash(paths, config_hash):
    manifest = {"config": config_hash,
        "features": sha256_file(paths["features"]), "vqa": sha256_file(paths["vqa"]),
        "projector": sha256_file(paths["projector"]), "model": sha256_tree(paths["model"]),
        "calibration": sha256_tree(paths["calibration"])}
    if paths.get("manifest"): manifest["manifest"] = sha256_file(paths["manifest"])
    if paths.get("vindex_checkout"): manifest["vindex_checkout"] = sha256_tree(paths["vindex_checkout"])
    provenance = Path(paths["features"]).with_name("features.provenance.json")
    if provenance.exists(): manifest["features_provenance"] = sha256_file(provenance)
    return manifest_hash(manifest), manifest


def _validate_scope(config):
    experiment = _mapping(config.get("experiment"), "experiment")
    expected = {"phase": "P1", "subjects": ["subj01"], "methods": ["npp-opsd"], "controls": ["correct"]}
    for key, value in expected.items():
        if experiment.get(key) != value:
            raise ValueError(f"real adapter currently supports only {key}={value!r}")
    if config.get("backend") != "vindex" or config.get("task") != "nsd_vqa" or config.get("method") != "NPP-OPSD":
        raise ValueError("real adapter supports only VINDEX P1 NSD-VQA NPP-OPSD")
    if int(experiment.get("max_optimizer_steps", -1)) != 1:
        raise ValueError("P1 requires max_optimizer_steps: 1")
    if int(config.get("micro_batch_size", -1)) != 1:
        raise ValueError("P1 requires micro_batch_size: 1")
    if int(config.get("candidate_count", -1)) != 4:
        raise ValueError("P1 requires candidate_count: 4")
    max_examples = experiment.get("max_examples")
    if isinstance(max_examples, bool) or not isinstance(max_examples, int) or not 1 <= max_examples <= 8:
        raise ValueError("P1 max_examples must be an integer from 1 through 8")
    trainer = _mapping(config.get("trainer"), "trainer")
    generation = _mapping(config.get("generation"), "generation")
    if trainer.get("gradient_accumulation_steps") != 1:
        raise ValueError("P1 requires trainer.gradient_accumulation_steps: 1")
    return experiment, trainer, generation


def _gradient_tracker(named_parameters):
    totals = {"projector": 0.0, "lora": 0.0}
    finite = {"projector": True, "lora": True}
    handles = []
    for name, parameter in named_parameters:
        group = "lora" if "lora_" in name else "projector"
        def hook(gradient, group=group):
            finite[group] = finite[group] and bool(torch.isfinite(gradient).all())
            totals[group] += float(gradient.float().square().sum())
        handles.append(parameter.register_hook(hook))
    return totals, finite, handles


def _create_trainer(adapter, optimizer, trainer_config, generation_config):
    return NPPTrainer(adapter, optimizer, generation_config=generation_config, **dict(trainer_config))


class VindexP1Runner:
    def train(self, config: Mapping[str, Any]):
        experiment, trainer_config, generation_config = _validate_scope(config)
        paths = _mapping(config.get("paths"), "paths")
        required = ("model", "features", "vqa", "calibration", "projector")
        missing = [name for name in required if not paths.get(name)]
        if missing:
            raise ValueError("P1 missing local paths: " + ", ".join(f"paths.{x}" for x in missing))
        output = Path(experiment["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = output / "checkpoint.pt"
        config_hash = _hash_json(config)
        input_hash, complete_input_hashes = _training_input_hash(paths, config_hash)
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if saved.get("input_hash") != input_hash or saved.get("optimizer_steps") != 1:
                raise ValueError("existing checkpoint is incompatible or incomplete")
            result = dict(saved["extra"]["result"])
            result["completed_run_reused"] = True
            result["duplicate_optimizer_update"] = False
            return result

        seed = int(experiment.get("seed", 41))
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        artifacts = load_validated_p1_artifacts(paths["features"], paths["calibration"])
        max_examples = min(int(experiment.get("max_examples", 8)), len(artifacts.arrays["trial_ids"]))
        selected = np.random.default_rng(seed).permutation(len(artifacts.arrays["trial_ids"]))[:max_examples]
        row = int(selected[0])
        image_lookup = {str(value): i for i, value in enumerate(artifacts.arrays["image_ids"])}
        candidate_rows = [image_lookup[value] for value in artifacts.candidate_ids]
        vqa = [json.loads(line) for line in Path(paths["vqa"]).read_text().splitlines()]
        image_number = int(str(artifacts.arrays["trial_image_ids"][row])[3:])
        qa = next((entry for entry in vqa if entry.get("nsd_image_id") == image_number), None)
        if qa is None or not isinstance(qa.get("question"), str):
            raise ValueError("selected image has no valid VQA question")

        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from transformers import BitsAndBytesConfig, LlamaConfig, LlamaForCausalLM, LlamaTokenizer
        except ImportError as error:
            raise RuntimeError("real VINDEX P1 requires the optional 'vindex' dependencies") from error
        model_path = Path(paths["model"])
        model = LlamaForCausalLM.from_pretrained(model_path,
            config=LlamaConfig(**json.loads((model_path / "config.json").read_text())), local_files_only=True,
            device_map={"": 0}, quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        model.config.use_cache = False
        tokenizer = LlamaTokenizer.from_pretrained(model_path, local_files_only=True)
        projector = torch.nn.Sequential(torch.nn.Linear(1024, 4096), torch.nn.GELU(), torch.nn.Linear(4096, 4096)).cuda()
        state = torch.load(paths["projector"], map_location="cpu", weights_only=True)
        state = {key.split("mm_projector.", 1)[1]: value for key, value in state.items() if "mm_projector." in key}
        projector.load_state_dict(state, strict=True)
        teacher_projector = frozen_snapshot(projector)
        teacher_before = {name: value.detach().cpu().clone() for name, value in teacher_projector.state_dict().items()}
        prefix = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: "
        pre = tokenizer(prefix, return_tensors="pt").input_ids
        post = tokenizer("\n" + qa["question"] + " ASSISTANT:", add_special_tokens=False, return_tensors="pt").input_ids
        adapter = VindexP1Adapter(model, tokenizer, projector, teacher_projector, pre, post, "cuda")
        batch = {
            "brain_features": torch.from_numpy(artifacts.arrays["brain_features"][row:row+1]).cuda(),
            "candidate_image_features": torch.from_numpy(artifacts.arrays["image_patches"][candidate_rows][None]).cuda(),
            "candidate_scores": torch.from_numpy(artifacts.log_likelihood[row:row+1]).cuda(),
            "candidate_mask": torch.ones((1, len(candidate_rows)), dtype=torch.bool, device="cuda"),
        }
        trainable = [("projector." + name, parameter) for name, parameter in projector.named_parameters()]
        trainable += [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        optimizer_config = _mapping(config.get("optimizer", {}), "optimizer")
        lr, weight_decay = float(optimizer_config.get("lr", 1e-4)), float(optimizer_config.get("weight_decay", 0.0))
        optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=lr, weight_decay=weight_decay)
        before = {name: parameter.detach().cpu().clone() for name, parameter in trainable}
        totals, finite, handles = _gradient_tracker(trainable)
        trainer = _create_trainer(adapter, optimizer, trainer_config, generation_config)
        diagnostics = trainer.step(batch)
        for handle in handles: handle.remove()
        deltas = {name: float((parameter.detach().cpu() - before[name]).float().norm()) for name, parameter in trainable}
        teacher_unchanged = all(torch.equal(value, teacher_projector.state_dict()[name].detach().cpu()) for name, value in teacher_before.items())
        frozen_base_grad = any(parameter.grad is not None for name, parameter in model.named_parameters() if "lora_" not in name)
        gradient = {group: {"norm": totals[group] ** 0.5, "finite": finite[group]} for group in totals}
        if diagnostics["optimizer_step"] != 1 or any(value["norm"] <= 0 or not value["finite"] for value in gradient.values()):
            raise ValueError("P1 did not produce one finite nonzero projector and LoRA update")
        if not teacher_unchanged or frozen_base_grad or not any(value > 0 for value in deltas.values()):
            raise ValueError("P1 freeze or optimizer-delta invariant failed")
        result = {"status": "completed_real_p1", "phase": "P1", "selected_example_count": max_examples,
            "consumed_example_count": 1, "optimizer_steps": 1, "resumed": False,
            "duplicate_optimizer_update": False, "trial_id": str(artifacts.arrays["trial_ids"][row]),
            "question": qa["question"], "rollout_token_count": int(trainer.last_rollout_mask.sum()),
            "gradient": gradient, "optimizer_delta_norm": sum(v*v for v in deltas.values()) ** 0.5,
            "teacher_projector_unchanged": teacher_unchanged, "frozen_base_has_gradient": frozen_base_grad,
            "learning_rate": lr, "weight_decay": weight_decay, "input_hashes": complete_input_hashes,
            "environment": realized_environment(("torch", "numpy", "PyYAML", "transformers", "peft", "accelerate", "bitsandbytes", "Pillow", "sentencepiece", "protobuf")),
            "diagnostics": diagnostics, "max_memory_allocated_bytes": torch.cuda.max_memory_allocated()}
        save_training_state(checkpoint, dict(trainable), optimizer, optimizer_steps=1,
            input_hash=input_hash, extra={"config_hash": config_hash, "result": result})
        (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result


class VindexMatrixRunner:
    def train(self, config: Mapping[str, Any]):
        return run_matrix_config(config)


def create_adapter(config):
    experiment = _mapping(config.get("experiment"), "experiment")
    if experiment.get("phase") == "P1":
        _validate_scope(config)
        return VindexP1Runner()
    if experiment.get("phase") in {"P2", "P3", "P4", "matrix", "captioning"}:
        return VindexMatrixRunner()
    raise ValueError("experiment.phase must select P1 or a full NSD matrix phase")
