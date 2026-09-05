"""Shared trainer for an atomically published recommendation Original."""
from __future__ import annotations

import gc
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml
from transformers import T5Tokenizer

from src.bota_if import p1_trajectory_transport_audit as p1
from src.diagnostics.t5_lora_influence_feasibility_audit_a2 import FixedABLinear
from src.diagnostics.t5_reconstructed_official import JsonPromptDataset, load_legacy_model, move_batch
from src.if_a2_optimization.group_a_gradient_audit import GIB, masked_batch
from src.paper_baselines.common import capture_rng, restore_rng, tensor_tree_hash
from src.paper_if_a2.artifacts import atomic_torch_save
from src.paper_if_a2.common import atomic_json, canonical_hash, directory_hash, git_snapshot, safe_run_name, seed_everything, sha256_file

SCHEMA = "bota-recommendation-original-v1"
MARKER = "BOTA_RECOMMENDATION_ORIGINAL_V1_COMPLETED"


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != SCHEMA or value.get("test_access_policy") != "forbidden":
        raise ValueError("invalid recommendation Original config")
    if value.get("coordinate") != {"target_modules": ["q", "v"], "module_count": 72, "rank": 16, "alpha": 32, "trainable": "B_only", "initial_B": "zero", "fixed_a_seed": 42}:
        raise ValueError("recommendation Original coordinate changed")
    expected_training = {
        "seed": 42, "optimizer": "AdamW", "learning_rate": .001, "betas": [.9, .999], "eps": 1e-8,
        "weight_decay": .01, "effective_batch_size": 16, "physical_microbatch": 4, "gradient_accumulation": 4,
        "maximum_epochs": 100, "early_stopping_metric": "development_sample_mean_answer_loss", "patience": 5,
        "min_delta": 0., "development_batch_size": 4, "checkpoint_every_epochs": 1,
    }
    if value.get("training") != expected_training:
        raise ValueError("recommendation Original training protocol changed")
    if value.get("publication", {}).get("merge_adapter_into_t5") is not True or value.get("scientific_scope", {}).get("final_test_access") is not False:
        raise ValueError("recommendation Original publication/split policy changed")
    return value


def early_stopping_transition(best: float, count: int, value: float, patience: int, min_delta: float) -> dict[str, Any]:
    if patience != 5 or min_delta != 0. or not math.isfinite(value):
        raise ValueError("invalid recommendation P5 state")
    improved = value < best - min_delta
    next_count = 0 if improved else count + 1
    return {"improved": improved, "best": value if improved else best, "count": next_count, "stop": bool(not improved and next_count >= patience)}


def _verify_sources(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    source = config["source"]
    model = root / source["pretrained_model"]
    prepared = root / source["prepared_root"]
    train = root / source["train_json"]
    development = root / source["development_json"]
    if not model.is_dir() or not prepared.is_dir() or not train.is_file() or not development.is_file():
        raise FileNotFoundError("recommendation Original source is incomplete")
    actual = {
        "pretrained_model_sha256": directory_hash(model),
        "prepared_manifest_sha256": sha256_file(prepared / "manifest.json"),
        "train_sha256": sha256_file(train),
        "development_sha256": sha256_file(development),
    }
    for key, digest in actual.items():
        if digest != source[key]:
            raise ValueError(f"recommendation Original source SHA mismatch: {key}")
    train_rows = json.loads(train.read_text(encoding="utf-8")); development_rows = json.loads(development.read_text(encoding="utf-8"))
    if len(train_rows) != source["train_samples"] or len(development_rows) != source["development_samples"]:
        raise ValueError("recommendation Original source count mismatch")
    return actual


def _parameter_values(names: Sequence[str], parameters: Sequence[torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().float().cpu().clone() for name, parameter in zip(names, parameters)}


def _load_values(names: Sequence[str], parameters: Sequence[torch.Tensor], values: dict[str, torch.Tensor]) -> None:
    if list(names) != list(values):
        raise ValueError("recommendation Original B tensor order mismatch")
    with torch.no_grad():
        for name, parameter in zip(names, parameters):
            value = values[name]
            if value.shape != parameter.shape or not torch.isfinite(value).all():
                raise ValueError(f"invalid recommendation Original B tensor: {name}")
            parameter.copy_(value.to(parameter))


def _sample_mean_loss(model: torch.nn.Module, dataset: JsonPromptDataset, indices: Sequence[int], device: torch.device, pad: int) -> torch.Tensor:
    batch = move_batch(masked_batch(dataset, indices, pad), device)
    losses = p1._sample_losses(model, batch)
    result = losses.mean()
    del batch, losses
    return result


def _backward_effective_batch(model: torch.nn.Module, dataset: JsonPromptDataset, indices: Sequence[int], device: torch.device, pad: int, microbatch: int) -> float:
    total = 0.
    for start in range(0, len(indices), microbatch):
        selected = list(indices[start:start + microbatch]); loss = _sample_mean_loss(model, dataset, selected, device, pad); weight = len(selected) / len(indices)
        (loss * weight).backward(); total += float(loss.detach().cpu()) * weight; del loss
    return total


def _validation_loss(model: torch.nn.Module, dataset: JsonPromptDataset, device: torch.device, pad: int, batch_size: int) -> float:
    model.eval(); total = 0.; samples = 0
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            selected = list(range(start, min(start + batch_size, len(dataset)))); loss = _sample_mean_loss(model, dataset, selected, device, pad)
            total += float(loss.detach().cpu()) * len(selected); samples += len(selected); del loss
    model.train()
    value = total / samples
    if samples != len(dataset) or not math.isfinite(value):
        raise RuntimeError("invalid recommendation Development loss")
    return value


def _epoch_order(size: int, seed: int, epoch: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed + 1_000_003 * epoch)
    return torch.randperm(size, generator=generator).tolist()


def merge_fixed_ab_modules(model: torch.nn.Module) -> dict[str, Any]:
    rows = [(name, module) for name, module in model.named_modules() if isinstance(module, FixedABLinear)]
    if len(rows) != 72:
        raise ValueError(f"expected 72 fixed-A/B modules, got {len(rows)}")
    modules = []; total_delta_norm_sq = 0.
    with torch.no_grad():
        for name, module in rows:
            delta = module.scaling * (module.B.float() @ module.fixed_A.float())
            if delta.shape != module.base.weight.shape or not torch.isfinite(delta).all():
                raise RuntimeError(f"invalid merged delta: {name}")
            module.base.weight.add_(delta.to(module.base.weight)); total_delta_norm_sq += float(torch.sum(delta.double().square()).cpu())
            if "." in name:
                parent_name, child = name.rsplit(".", 1); parent = model.get_submodule(parent_name)
            else:
                child = name; parent = model
            setattr(parent, child, module.base)
            modules.append({"name": name, "delta_norm": float(torch.linalg.vector_norm(delta).cpu())})
    if any(isinstance(module, FixedABLinear) for module in model.modules()):
        raise RuntimeError("fixed-A/B wrapper remained after merge")
    return {"module_count": len(modules), "total_delta_norm": math.sqrt(total_delta_norm_sq), "modules": modules}


def _smoke_logits(model: torch.nn.Module, dataset: JsonPromptDataset, device: torch.device, pad: int, samples: int) -> torch.Tensor:
    model.eval(); rows = []
    with torch.inference_mode():
        for start in range(0, samples, 4):
            selected = list(range(start, min(start + 4, samples))); batch = move_batch(masked_batch(dataset, selected, pad), device)
            rows.append(model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["target_ids"]).logits.detach().float().cpu()); del batch
    return torch.cat(rows)


def _checkpoint(path: Path, payload: dict[str, Any]) -> None:
    atomic_torch_save(path, payload)


def _paths(root: Path, config: dict[str, Any], run_name: str) -> tuple[Path, Path]:
    destination = root / config["output_root"] / safe_run_name(run_name)
    work = destination.parent / ".work" / safe_run_name(run_name)
    return destination, work


def preflight(root: Path, config_path: Path, run_name: str) -> dict[str, Any]:
    config = load_config(config_path); sources = _verify_sources(root, config)
    return {"schema": SCHEMA, "run_name": run_name, "sources": sources, "train_samples": config["source"]["train_samples"], "development_samples": config["source"]["development_samples"], "model_loaded": False, "optimizer_constructed": False, "final_test_accessed": False, "test_accessed": False}


def execute(root: Path, config_path: Path, run_name: str, resume: bool) -> dict[str, Any]:
    config = load_config(config_path); sources = _verify_sources(root, config); run_name = safe_run_name(run_name); destination, work = _paths(root, config, run_name)
    if destination.exists(): raise FileExistsError(destination)
    git = git_snapshot(root)
    if not git["clean"]: raise RuntimeError("formal recommendation Original training requires clean Git")
    if not torch.cuda.is_available() or torch.cuda.device_count() != config["runtime"]["required_cuda_devices"]: raise RuntimeError("exactly one CUDA GPU required")
    torch.cuda.set_device(0); device = torch.device("cuda:0"); free, _ = torch.cuda.mem_get_info(device)
    if free / GIB < config["runtime"]["minimum_free_gib"]: raise RuntimeError("insufficient dedicated GPU memory")
    torch.cuda.set_per_process_memory_fraction(config["runtime"]["allocator_fraction"], device); torch.cuda.reset_peak_memory_stats()
    contract = {"schema": SCHEMA, "config_sha256": sha256_file(config_path), "sources": sources, "git": git, "run_name": run_name, "test_accessed": False}
    if resume:
        if not work.is_dir() or not (work / "checkpoint.pt").is_file() or not (work / "contract.json").is_file(): raise FileNotFoundError("resumable recommendation Original work state not found")
        if json.loads((work / "contract.json").read_text(encoding="utf-8")) != contract: raise ValueError("recommendation Original Resume contract mismatch")
    else:
        if work.exists(): raise FileExistsError(f"unfinished work exists; use Resume: {work}")
        work.mkdir(parents=True); atomic_json(work / "contract.json", contract)
    training = config["training"]; publication = config["publication"]; model = optimizer = train = development = tokenizer = None; started = time.perf_counter(); elapsed_before = 0.
    try:
        seed_everything(training["seed"]); tokenizer = T5Tokenizer.from_pretrained(root / config["source"]["pretrained_model"])
        train = JsonPromptDataset(root / config["source"]["train_json"], tokenizer); development = JsonPromptDataset(root / config["source"]["development_json"], tokenizer)
        runtime = {"original_checkpoint": config["source"]["pretrained_model"], "coordinate": {"lora_rank": 16, "lora_alpha": 32, "fixed_a_seed": 42}}
        model, names, parameters, bases, basis_report = p1._fresh_runtime(root, runtime, device)
        optimizer = torch.optim.AdamW(parameters, lr=training["learning_rate"], betas=tuple(training["betas"]), eps=training["eps"], weight_decay=training["weight_decay"])
        epoch = 0; total_steps = 0; best = float("inf"); best_epoch = 0; non_improving = 0; history = []; best_values = None
        if resume:
            saved = torch.load(work / "checkpoint.pt", map_location="cpu", weights_only=False)
            if saved.get("contract_sha256") != canonical_hash(contract): raise ValueError("recommendation Original checkpoint binding mismatch")
            _load_values(names, parameters, saved["values"]); optimizer.load_state_dict(saved["optimizer"]); restore_rng(saved["rng"])
            epoch = int(saved["epoch"]); total_steps = int(saved["total_steps"]); best = float(saved["best"]); best_epoch = int(saved["best_epoch"]); non_improving = int(saved["non_improving"]); history = saved["history"]; best_values = saved["best_values"]; elapsed_before = float(saved["elapsed_seconds"])
            (work / "INTERRUPTED").unlink(missing_ok=True)
        else:
            _checkpoint(work / "checkpoint.pt", {"schema": SCHEMA, "contract_sha256": canonical_hash(contract), "epoch": 0, "total_steps": 0, "best": best, "best_epoch": best_epoch, "non_improving": non_improving, "history": history, "values": _parameter_values(names, parameters), "best_values": best_values, "optimizer": optimizer.state_dict(), "rng": capture_rng(), "elapsed_seconds": 0., "test_accessed": False})
        for current_epoch in range(epoch + 1, training["maximum_epochs"] + 1):
            model.train(); order = _epoch_order(len(train), training["seed"], current_epoch); total_loss = 0.; seen = 0; epoch_started = time.perf_counter()
            for start in range(0, len(order), training["effective_batch_size"]):
                selected = order[start:start + training["effective_batch_size"]]; optimizer.zero_grad(set_to_none=True)
                loss = _backward_effective_batch(model, train, selected, device, tokenizer.pad_token_id, training["physical_microbatch"])
                optimizer.step(); total_steps += 1; total_loss += loss * len(selected); seen += len(selected)
                if torch.cuda.max_memory_reserved() / GIB > config["runtime"]["hard_peak_reserved_gib"]: raise RuntimeError("GPU hard cap exceeded")
            validation = _validation_loss(model, development, device, tokenizer.pad_token_id, training["development_batch_size"])
            transition = early_stopping_transition(best, non_improving, validation, training["patience"], training["min_delta"]); best = transition["best"]; non_improving = transition["count"]
            if transition["improved"]: best_epoch = current_epoch; best_values = _parameter_values(names, parameters)
            history.append({"epoch": current_epoch, "train_loss": total_loss / seen, "development_loss": validation, "improved": transition["improved"], "consecutive_non_improving_epochs": non_improving, "optimizer_steps": total_steps, "epoch_wall_seconds": time.perf_counter() - epoch_started, "sample_order_sha256": canonical_hash(order), "test_accessed": False})
            elapsed = elapsed_before + time.perf_counter() - started
            _checkpoint(work / "checkpoint.pt", {"schema": SCHEMA, "contract_sha256": canonical_hash(contract), "epoch": current_epoch, "total_steps": total_steps, "best": best, "best_epoch": best_epoch, "non_improving": non_improving, "history": history, "values": _parameter_values(names, parameters), "best_values": best_values, "optimizer": optimizer.state_dict(), "rng": capture_rng(), "elapsed_seconds": elapsed, "test_accessed": False})
            atomic_json(work / "run_state.json", {"schema": SCHEMA, "status": "RUNNING", "epoch": current_epoch, "optimizer_steps": total_steps, "best_epoch": best_epoch, "best_development_loss": best, "consecutive_non_improving_epochs": non_improving, "test_accessed": False})
            if transition["stop"]: break
        pilot_fixed_epoch = config.get("scientific_scope", {}).get("pilot_fixed_epoch_endpoint") is True
        if best_values is None or (not pilot_fixed_epoch and non_improving < training["patience"]): raise RuntimeError("recommendation Original did not reach the frozen endpoint")
        if pilot_fixed_epoch and history[-1]["epoch"] != training["maximum_epochs"]: raise RuntimeError("pilot Original did not reach the fixed epoch endpoint")
        _load_values(names, parameters, best_values); smoke_before = _smoke_logits(model, development, device, tokenizer.pad_token_id, publication["smoke_samples"]); adapter_state = {"schema": SCHEMA, "A": {name: value.detach().cpu() for name, value in bases.items()}, "B": best_values, "rank": 16, "alpha": 32, "base_model_sha256": sources["pretrained_model_sha256"], "test_accessed": False}
        adapter_dir = work / "adapter"; adapter_dir.mkdir(); atomic_torch_save(adapter_dir / "adapter_model.pt", adapter_state); atomic_json(adapter_dir / "adapter_config.json", {"schema": SCHEMA, "format": "fixed-A-B-LoRA-merged-source", "target_modules": ["q", "v"], "rank": 16, "alpha": 32, "trainable": "B_only", "test_accessed": False})
        merge = merge_fixed_ab_modules(model); smoke_merged = _smoke_logits(model, development, device, tokenizer.pad_token_id, publication["smoke_samples"])
        if not torch.allclose(smoke_before, smoke_merged, atol=publication["smoke_atol"], rtol=publication["smoke_rtol"]): raise RuntimeError("fixed-A/B merge smoke mismatch")
        model.cpu(); gc.collect(); torch.cuda.empty_cache(); model_dir = work / publication["model_subdirectory"]; model.save_pretrained(model_dir, safe_serialization=True, max_shard_size="1GB"); tokenizer.save_pretrained(model_dir)
        merged_sha = tensor_tree_hash(model.state_dict()); del model; model = None; reload_model = load_legacy_model(model_dir).to(device).eval(); reload_sha = tensor_tree_hash(reload_model.state_dict()); smoke_reload = _smoke_logits(reload_model, development, device, tokenizer.pad_token_id, publication["smoke_samples"])
        if merged_sha != reload_sha or not torch.allclose(smoke_merged, smoke_reload, atol=publication["smoke_atol"], rtol=publication["smoke_rtol"]): raise RuntimeError("published recommendation Original reload mismatch")
        peak = torch.cuda.max_memory_reserved() / GIB; wall = elapsed_before + time.perf_counter() - started
        result = {"schema": SCHEMA, "status": "COMPLETED", "run_name": run_name, "model_path": str((destination / publication["model_subdirectory"]).resolve()), "model_directory_sha256": directory_hash(model_dir), "merged_parameter_sha256": merged_sha, "source_pretrained_sha256": sources["pretrained_model_sha256"], "train_samples": len(train), "development_samples": len(development), "best_epoch": best_epoch, "stopping_epoch": history[-1]["epoch"], "best_development_loss": best, "optimizer_steps": total_steps, "endpoint_rule": "fixed_epoch_pilot" if pilot_fixed_epoch else "development_patience_five", "coordinate_training": "fixed-A/B Q/V LoRA B-only", "deployment_representation": "LoRA delta merged into full T5 weights", "test_accessed": False}
        atomic_json(work / "training_history.json", history); atomic_json(work / "merge_report.json", {**merge, "premerge_smoke_sha256": tensor_tree_hash(smoke_before), "merged_smoke_sha256": tensor_tree_hash(smoke_merged), "reload_smoke_sha256": tensor_tree_hash(smoke_reload), "maximum_merge_absolute_error": float(torch.max(torch.abs(smoke_before - smoke_merged))), "maximum_reload_absolute_error": float(torch.max(torch.abs(smoke_merged - smoke_reload))), "within_tolerance": True, "test_accessed": False}); atomic_json(work / "timing.json", {"end_to_end_wall_seconds": wall, "training_and_development_seconds": sum(row["epoch_wall_seconds"] for row in history), "epochs": len(history), "optimizer_steps": total_steps, "test_accessed": False}); atomic_json(work / "original_manifest.json", result); atomic_json(work / "run_state.json", {"schema": SCHEMA, "status": "COMPLETED", "best_epoch": best_epoch, "stopping_epoch": history[-1]["epoch"], "optimizer_steps": total_steps, "peak_gpu_reserved_gib": peak, "test_accessed": False}); (work / "checkpoint.pt").unlink(); (work / "COMPLETED").write_text(MARKER + "\n", encoding="utf-8", newline="\n")
        files = ["adapter", "model", "contract.json", "training_history.json", "merge_report.json", "timing.json", "original_manifest.json", "run_state.json", "COMPLETED"]
        atomic_json(work / "manifest.json", {"schema": SCHEMA, "artifacts": {name: directory_hash(work / name) if (work / name).is_dir() else sha256_file(work / name) for name in files}, "published_atomically": True, "test_accessed": False}); destination.parent.mkdir(parents=True, exist_ok=True); os.replace(work, destination)
        return result
    except BaseException as error:
        if work.exists():
            atomic_json(work / "run_state.json", {"schema": SCHEMA, "status": "INTERRUPTED", "reason": type(error).__name__, "message": str(error), "resume_allowed": (work / "checkpoint.pt").is_file(), "test_accessed": False}); (work / "INTERRUPTED").write_text("BOTA_RECOMMENDATION_ORIGINAL_INTERRUPTED\n", encoding="utf-8", newline="\n")
        raise RuntimeError(f"recommendation Original interrupted; resumable evidence at {work}") from error
    finally:
        if model is not None: del model
        if optimizer is not None: del optimizer
        gc.collect(); torch.cuda.empty_cache()


def analyze(root: Path, config_path: Path, run_name: str) -> dict[str, Any]:
    config = load_config(config_path); destination, _ = _paths(root, config, run_name)
    required = {"adapter", "model", "contract.json", "training_history.json", "merge_report.json", "timing.json", "original_manifest.json", "run_state.json", "COMPLETED", "manifest.json"}
    if not destination.is_dir() or {path.name for path in destination.iterdir()} != required or (destination / "COMPLETED").read_text(encoding="utf-8") != MARKER + "\n": raise ValueError("invalid recommendation Original run")
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8")); state = json.loads((destination / "run_state.json").read_text(encoding="utf-8")); result = json.loads((destination / "original_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest.get("artifacts", {}).items():
        path = destination / name; actual = directory_hash(path) if path.is_dir() else sha256_file(path)
        if actual != expected: raise ValueError(f"recommendation Original artifact mismatch: {name}")
    if state.get("status") != "COMPLETED" or state.get("test_accessed") is not False or result.get("test_accessed") is not False: raise ValueError("recommendation Original completion invariant failed")
    if directory_hash(destination / "model") != result["model_directory_sha256"]: raise ValueError("recommendation Original model SHA mismatch")
    return {"status": "COMPLETED", "run_dir": str(destination), "model_path": result["model_path"], "model_directory_sha256": result["model_directory_sha256"], "best_epoch": result["best_epoch"], "stopping_epoch": result["stopping_epoch"], "optimizer_steps": result["optimizer_steps"], "best_development_loss": result["best_development_loss"], "test_accessed": False}

