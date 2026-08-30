"""Matched short-window paper baselines: P5 controls, NegGrad and PCGrad."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml
from transformers import T5Tokenizer

from src.bota_if.p1_trajectory_transport_audit import StepBudget
from src.diagnostics.t5_reconstructed_official import JsonPromptDataset, move_batch
from src.if_a2_optimization.group_a_gradient_audit import GIB, masked_batch
from src.paper_if_a2.artifacts import atomic_torch_save
from src.paper_if_a2.common import atomic_json, canonical_hash, directory_hash, git_snapshot, safe_run_name, seed_everything, sha256_file

from . import runner as v1_runner
from .protocol import validate_prepared
from .timing import SCENARIO_CHOICES, select_scenarios, timing_record

SCHEMA = "bota-short-paper-v2"
MARKER = "BOTA_SHORT_PAPER_V2_METHOD_COMPLETED"
METHODS = {
    "FullControlP5": "FullControl-P5-Short",
    "RetainP5": "Retain-Retrain-P5-Short",
    "NegGrad": "NegGrad-Mixed-Short-BOnly",
    "PCGrad": "PCGrad-Short-BOnly",
}


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != SCHEMA or value.get("test_access_policy") != "forbidden":
        raise ValueError("invalid BOTA short paper v2 config")
    p = value.get("protocol", {})
    frozen = {"window_samples": 3200, "batch_size": 16, "physical_microbatch": 4, "gradient_accumulation": 4, "learning_rate": .001, "max_epochs": 100, "patience": 5, "min_delta": 0., "validation_batch_size": 4, "neggrad_steps": 200, "neggrad_forget_weight": .2, "pcgrad_steps": 200, "pcgrad_projection": "deterministic_symmetric_two_task"}
    if any(p.get(key) != expected for key, expected in frozen.items()):
        raise ValueError("frozen short paper protocol changed")
    if p.get("seed") not in {41, 42, 43}:
        raise ValueError("paper seed must be one of 41/42/43")
    if p.get("deleted_samples") not in {2, 8}:
        raise ValueError("unsupported frozen deletion cardinality")
    if value.get("coordinate") != {"target_modules": ["q", "v"], "rank": 16, "alpha": 32, "trainable": "B_only", "initial_B": "zero"}:
        raise ValueError("fixed-A/B coordinate changed")
    if value.get("evaluation") != {"split": "Development", "final_test": False, "inference_batch_size": 4, "bootstrap_resamples": 1000}:
        raise ValueError("Development-only evaluation changed")
    return value


def early_stopping_transition(best: float, count: int, value: float, patience: int = 5) -> dict[str, Any]:
    if patience != 5 or not math.isfinite(value):
        raise ValueError("invalid P5 early-stopping state")
    improved = value < best
    next_count = 0 if improved else count + 1
    return {"improved": improved, "best": value if improved else best, "count": next_count, "stop": bool(not improved and next_count >= 5)}


def _v1(root: Path, config: dict[str, Any]) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    path = root / config["source"]["v1_config"]
    value = v1_runner.load_config(path)
    protocol, _, registry = validate_prepared(root, value, config["source"]["v1_benchmark_name"])
    if int(value["protocol"].get("run_seed", value["protocol"]["seed"])) != int(config["protocol"]["seed"]):
        raise ValueError("paper/v1 seed binding mismatch")
    return value, protocol, registry


def _context(root: Path, v1_config: dict[str, Any], registry: dict[str, Any], device: torch.device, fixed_a=None):
    return v1_runner._model_context(root, v1_config, registry, device, fixed_a)


def _loss(model, dataset, indices: Sequence[int], device: torch.device, pad: int) -> torch.Tensor:
    batch = move_batch(masked_batch(dataset, list(indices), pad), device)
    losses = v1_runner.p1._sample_losses(model, batch)
    result = losses.mean()
    del batch, losses
    return result


def _validation_loss(model, dataset, device: torch.device, pad: int, batch_size: int) -> float:
    model.eval(); total = 0.; samples = 0
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            selected = list(range(start, min(start + batch_size, len(dataset))))
            batch = move_batch(masked_batch(dataset, selected, pad), device)
            losses = v1_runner.p1._sample_losses(model, batch)
            total += float(losses.double().sum().cpu()); samples += len(selected)
            del batch, losses
            if ((start // batch_size) + 1) % 64 == 0: torch.cuda.empty_cache()
    model.train()
    if samples != len(dataset) or not math.isfinite(total):
        raise RuntimeError("invalid Development validation loss")
    return total / samples


def _memory_gate(config: dict[str, Any]) -> None:
    if torch.cuda.max_memory_reserved() / GIB > config["runtime"]["hard_peak_reserved_gib"]:
        raise RuntimeError("GPU hard cap exceeded during bounded execution")


def _backward_mean(model, dataset, indices: Sequence[int], device: torch.device, pad: int, coefficient: float, microbatch: int = 4) -> float:
    if not indices: raise ValueError("empty gradient component")
    total = 0.
    for start in range(0, len(indices), microbatch):
        selected = list(indices[start:start + microbatch]); loss = _loss(model, dataset, selected, device, pad); weight = len(selected) / len(indices); (loss * coefficient * weight).backward(); total += float(loss.detach().cpu()) * weight; del loss
    return total


def _mean_grads(model, dataset, indices: Sequence[int], parameters: Sequence[torch.Tensor], device: torch.device, pad: int, coefficient: float, microbatch: int = 4) -> tuple[list[torch.Tensor], float]:
    if not indices: raise ValueError("empty gradient component")
    result = [torch.zeros_like(parameter) for parameter in parameters]; total = 0.
    for start in range(0, len(indices), microbatch):
        selected = list(indices[start:start + microbatch]); loss = _loss(model, dataset, selected, device, pad); weight = len(selected) / len(indices); gradients = torch.autograd.grad(loss * coefficient * weight, parameters); total += float(loss.detach().cpu()) * weight
        for target, gradient in zip(result, gradients): target.add_(gradient)
        del loss, gradients
    return result, total


def _epoch_order(indices: Sequence[int], seed: int, epoch: int) -> list[int]:
    permutation = torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed + 1_000_003 * epoch)).tolist()
    return [int(indices[index]) for index in permutation]


def _save_adapter(path: Path, names, values, bases, method_id: str) -> dict[str, Any]:
    return v1_runner._save_fixed_ab(path, names, values, bases, method_id)


def _scenario_manifest(path: Path, method_id: str, scenario: dict[str, Any], artifact: dict[str, Any], extra: dict[str, Any]) -> None:
    atomic_json(path / "scenario_manifest.json", {"schema": SCHEMA, "method_id": method_id, "scenario_id": scenario["id"], "request_hash": scenario["request_hash"], "model_type": "if_a2_fixed_ab", "artifact": artifact, "deleted_interactions": int(scenario["deleted_interactions"]), "test_accessed": False, **extra})


def _p5_one(root: Path, config: dict[str, Any], v1_config: dict[str, Any], registry: dict[str, Any], train_indices: Sequence[int], device: torch.device, fixed_a=None):
    run_seed = int(config["protocol"]["seed"]); seed_everything(run_seed)
    model, names, parameters, bases, _, tokenizer, train, _ = _context(root, v1_config, registry, device, fixed_a)
    development = JsonPromptDataset(root / v1_config["source"]["development_json"], tokenizer)
    optimizer = torch.optim.AdamW(parameters, lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=.01)
    best = float("inf"); best_values = None; best_epoch = 0; count = 0; history = []; steps = 0
    for epoch in range(1, 101):
        model.train(); total = 0.; seen = 0
        order = _epoch_order(train_indices, run_seed, epoch)
        for start in range(0, len(order), 16):
            selected = order[start:start + 16]; optimizer.zero_grad(set_to_none=True); batch_loss = _backward_mean(model, train, selected, device, tokenizer.pad_token_id, 1., 4); _memory_gate(config); optimizer.step(); total += batch_loss * len(selected); seen += len(selected); steps += 1
        validation = _validation_loss(model, development, device, tokenizer.pad_token_id, 4); _memory_gate(config)
        transition = early_stopping_transition(best, count, validation); best = transition["best"]; count = transition["count"]
        if transition["improved"]:
            best_epoch = epoch; best_values = v1_runner._copy_parameters(names, parameters)
        history.append({"epoch": epoch, "train_loss": total / seen, "validation_loss": validation, "improved": transition["improved"], "consecutive_non_improving_epochs": count, "optimizer_steps": steps})
        if transition["stop"]: break
    if best_values is None or count < 5:
        raise RuntimeError("P5 did not produce a valid patience-five endpoint")
    result = (names, best_values, {name: value.detach().cpu().clone() for name, value in bases.items()}, history, {"best_epoch": best_epoch, "stopping_epoch": history[-1]["epoch"], "best_validation_loss": best, "optimizer_steps": steps, "configured_patience": 5, "early_stopping_triggered": True})
    del model, optimizer, development, train; gc.collect(); torch.cuda.empty_cache()
    return result


def _load_original_endpoint(root: Path, v1_config: dict[str, Any], registry: dict[str, Any], source_run_name: str, scenario_id: str, device: torch.device):
    run = root / v1_config["output_root"] / "models" / v1_runner.METHODS["Original"] / safe_run_name(source_run_name)
    v1_runner.analyze(root, Path(v1_config["_config_path"]), source_run_name, "Original")
    run_state = json.loads((run / "run_state.json").read_text(encoding="utf-8"))
    if scenario_id not in run_state.get("scenarios", []): raise ValueError(f"source Original does not contain {scenario_id}")
    state = torch.load(run / "scenarios" / scenario_id / "adapter" / "adapter_model.pt", map_location="cpu", weights_only=True)
    model, names, parameters, bases, _, tokenizer, dataset, users = _context(root, v1_config, registry, device, state["A"])
    with torch.no_grad():
        for name, parameter in zip(names, parameters): parameter.copy_(state["B"][name].to(parameter))
    return model, names, parameters, bases, tokenizer, dataset, users, sha256_file(run / "manifest.json")


def _pair_indices(retain: Sequence[int], forget: Sequence[int], step: int, seed: int = 42) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed + 1_000_003 * (step // max(1, math.ceil(len(retain) / 16))))
    order = torch.randperm(len(retain), generator=generator).tolist(); start = (step * 16) % len(retain)
    if start + 16 <= len(retain): r = [retain[index] for index in order[start:start + 16]]
    else: r = [retain[index] for index in (order[start:] + order[:16 - (len(retain) - start)])]
    return r, list(forget)


def _pcgrad(grads_a: Sequence[torch.Tensor], grads_b: Sequence[torch.Tensor]) -> tuple[list[torch.Tensor], bool]:
    dot = sum(torch.sum(a.double() * b.double()) for a, b in zip(grads_a, grads_b)); conflict = bool(dot < 0)
    if not conflict: return [((a + b) * .5).to(a.dtype) for a, b in zip(grads_a, grads_b)], False
    norm_a = sum(torch.sum(a.double().square()) for a in grads_a).clamp_min(1e-30); norm_b = sum(torch.sum(b.double().square()) for b in grads_b).clamp_min(1e-30)
    left = [a - (dot / norm_b).to(a.dtype) * b for a, b in zip(grads_a, grads_b)]; right = [b - (dot / norm_a).to(b.dtype) * a for a, b in zip(grads_a, grads_b)]
    return [((a + b) * .5).to(a.dtype) for a, b in zip(left, right)], True


def _posthoc_one(root: Path, v1_config: dict[str, Any], registry: dict[str, Any], scenario: dict[str, Any], source_run_name: str, method: str, device: torch.device, run_seed: int = 42):
    seed_everything(run_seed)
    model, names, parameters, bases, tokenizer, dataset, _, source_hash = _load_original_endpoint(root, v1_config, registry, source_run_name, scenario["id"], device)
    optimizer = torch.optim.AdamW(parameters, lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=.01); forget = list(scenario["forget_train_indices"]); forget_set = set(forget); retain = [index for index in registry["order"] if index not in forget_set]; conflicts = 0; history = []
    for step in range(200):
        ridx, fidx = _pair_indices(retain, forget, step, run_seed); optimizer.zero_grad(set_to_none=True)
        if method == "NegGrad":
            retain_value = _backward_mean(model, dataset, ridx, device, tokenizer.pad_token_id, 1., 4); forget_value = _backward_mean(model, dataset, fidx, device, tokenizer.pad_token_id, -.2, 4); objective_value = retain_value - .2 * forget_value
        else:
            retain_grads, retain_value = _mean_grads(model, dataset, ridx, parameters, device, tokenizer.pad_token_id, 1., 4); forget_grads, forget_value = _mean_grads(model, dataset, fidx, parameters, device, tokenizer.pad_token_id, -1., 4); merged, conflict = _pcgrad(retain_grads, forget_grads); conflicts += int(conflict)
            for parameter, gradient in zip(parameters, merged): parameter.grad = gradient
            objective_value = retain_value - forget_value
        optimizer.step(); history.append({"step": step + 1, "retain_loss": retain_value, "forget_loss": forget_value, "objective": objective_value, "retain_batch_hash": canonical_hash(ridx), "forget_batch_hash": canonical_hash(fidx)}); del retain_value, forget_value, objective_value
    values = v1_runner._copy_parameters(names, parameters); result = (names, values, {name: value.detach().cpu().clone() for name, value in bases.items()}, history, {"optimizer_steps": 200, "source_original_run_manifest_sha256": source_hash, "pcgrad_conflict_rate": conflicts / 200 if method == "PCGrad" else None})
    del model, optimizer, dataset; gc.collect(); torch.cuda.empty_cache(); return result


def execute(root: Path, config_path: Path, benchmark_name: str, run_name: str, method: str, source_original_run_name: str, scenario_selection: str = "All") -> dict[str, Any]:
    config = load_config(config_path); run_seed = int(config["protocol"]["seed"]); v1_config, _, registry = _v1(root, config)
    if benchmark_name != config["source"]["v1_benchmark_name"]: raise ValueError("benchmark lineage mismatch")
    git = git_snapshot(root)
    if not git["clean"]: raise RuntimeError("formal short paper run requires clean Git")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1: raise RuntimeError("exactly one CUDA GPU required")
    method_id = METHODS[method]; destination = root / config["output_root"] / "models" / method_id / safe_run_name(run_name)
    if destination.exists(): raise FileExistsError(destination)
    torch.cuda.set_device(0); device = torch.device("cuda:0"); torch.cuda.reset_peak_memory_stats(); stage = destination.parent / ".work" / f"{safe_run_name(run_name)}.{uuid.uuid4().hex}.stage"; stage.mkdir(parents=True); started = time.perf_counter(); total_steps = 0; fixed_a = None
    try:
        selected = select_scenarios(registry["scenarios"], scenario_selection); scenarios = selected if method != "FullControlP5" else [selected[0]]
        cached = None
        for scenario in scenarios:
            scenario_started = time.perf_counter(); phase_started = time.perf_counter()
            if method in {"FullControlP5", "RetainP5"}:
                forget = set(scenario["forget_train_indices"]); indices = list(registry["order"]) if method == "FullControlP5" else [index for index in registry["order"] if index not in forget]
                result = _p5_one(root, config, v1_config, registry, indices, device, fixed_a); fixed_a = result[2]; cached = result
            else:
                if not source_original_run_name: raise ValueError("NegGrad/PCGrad requires SourceOriginalRunName")
                result = _posthoc_one(root, v1_config, registry, scenario, source_original_run_name, method, device, run_seed)
            online_compute_seconds = time.perf_counter() - phase_started; names, values, bases, history, summary = result; scenario_dir = stage / "scenarios" / scenario["id"]; scenario_dir.mkdir(parents=True); phase_started = time.perf_counter(); artifact = _save_adapter(scenario_dir / "adapter", names, values, bases, method_id); publication_seconds = time.perf_counter() - phase_started; artifact["path"] = str((destination / "scenarios" / scenario["id"] / "adapter").resolve()); atomic_json(scenario_dir / "training_history.json", history); phase_timing = timing_record(scenario=scenario["id"], initialization_seconds=0.0, offline_construction_seconds=0.0, online_compute_seconds=online_compute_seconds, adapter_publication_seconds=publication_seconds, end_to_end_seconds=time.perf_counter() - scenario_started, details={"training_or_update_seconds": online_compute_seconds}); atomic_json(scenario_dir / "phase_timing.json", phase_timing); _scenario_manifest(scenario_dir, method_id, scenario, artifact, {**summary, "phase_timing": phase_timing}); total_steps += int(summary["optimizer_steps"])
        if method == "FullControlP5":
            assert cached is not None
            for scenario in selected[1:]:
                names, values, bases, history, summary = cached; scenario_dir = stage / "scenarios" / scenario["id"]; scenario_dir.mkdir(parents=True); phase_started = time.perf_counter(); artifact = _save_adapter(scenario_dir / "adapter", names, values, bases, method_id); publication_seconds = time.perf_counter() - phase_started; artifact["path"] = str((destination / "scenarios" / scenario["id"] / "adapter").resolve()); atomic_json(scenario_dir / "training_history.json", history); phase_timing = timing_record(scenario=scenario["id"], initialization_seconds=0.0, offline_construction_seconds=0.0, online_compute_seconds=0.0, adapter_publication_seconds=publication_seconds, end_to_end_seconds=publication_seconds); atomic_json(scenario_dir / "phase_timing.json", phase_timing); _scenario_manifest(scenario_dir, method_id, scenario, artifact, {**summary, "shared_full_control_endpoint": True, "phase_timing": phase_timing})
        peak = torch.cuda.max_memory_reserved()
        if peak / GIB > config["runtime"]["hard_peak_reserved_gib"]: raise RuntimeError("GPU hard cap exceeded")
        state = {"schema": SCHEMA, "status": "COMPLETED", "method_id": method_id, "run_name": run_name, "benchmark_name": benchmark_name, "run_seed": run_seed, "scenarios": [row["id"] for row in selected], "scenario_selection": scenario_selection, "physical_optimizer_steps": total_steps, "wall_time_seconds": time.perf_counter() - started, "peak_gpu_reserved": peak, "source_original_run_name": source_original_run_name or None, "test_accessed": False}; atomic_json(stage / "run_state.json", state); atomic_json(stage / "contract.json", {"schema": SCHEMA, "config_sha256": sha256_file(config_path), "v1_registry_sha256": registry["registry_sha256"], "method_id": method_id, "run_seed": run_seed, "scenario_selection": scenario_selection, "git": git, "test_accessed": False}); (stage / "COMPLETED").write_text(MARKER + "\n", encoding="utf-8", newline="\n"); atomic_json(stage / "manifest.json", {"schema": SCHEMA, "run_state_sha256": sha256_file(stage / "run_state.json"), "contract_sha256": sha256_file(stage / "contract.json"), "scenario_manifests": {row["id"]: sha256_file(stage / "scenarios" / row["id"] / "scenario_manifest.json") for row in selected}, "phase_timings": {row["id"]: sha256_file(stage / "scenarios" / row["id"] / "phase_timing.json") for row in selected}, "published_atomically": True, "test_accessed": False}); destination.parent.mkdir(parents=True, exist_ok=True); os.replace(stage, destination); return {"status": "COMPLETED", "method_id": method_id, "run_dir": str(destination), "scenarios": state["scenarios"], "physical_optimizer_steps": total_steps, "test_accessed": False}
    finally:
        if stage.exists(): shutil.rmtree(stage)
        gc.collect(); torch.cuda.empty_cache()


def analyze(root: Path, config_path: Path, run_name: str, method: str) -> dict[str, Any]:
    config = load_config(config_path); run = root / config["output_root"] / "models" / METHODS[method] / safe_run_name(run_name); required = {"COMPLETED", "contract.json", "manifest.json", "run_state.json", "scenarios"}
    if not run.is_dir() or {path.name for path in run.iterdir()} != required or (run / "COMPLETED").read_text(encoding="utf-8") != MARKER + "\n": raise ValueError("invalid short paper run")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8")); state = json.loads((run / "run_state.json").read_text(encoding="utf-8"))
    if sha256_file(run / "run_state.json") != manifest.get("run_state_sha256") or state.get("status") != "COMPLETED" or state.get("test_accessed") is not False: raise ValueError("short paper integrity mismatch")
    return {"status": "COMPLETED", "method_id": METHODS[method], "run_dir": str(run), "physical_optimizer_steps": state["physical_optimizer_steps"], "test_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path.cwd()); parser.add_argument("--config", type=Path, default=Path("configs/bota_short_paper_v2.yaml")); parser.add_argument("--mode", choices=["Preflight", "SyntheticDryRun", "Full", "Analyze"], default="Preflight"); parser.add_argument("--method", choices=sorted(METHODS), required=True); parser.add_argument("--benchmark-name", default=""); parser.add_argument("--run-name", default=""); parser.add_argument("--source-original-run-name", default=""); parser.add_argument("--scenario", choices=SCENARIO_CHOICES, default="All"); args = parser.parse_args(); root = args.root.resolve(); cp = (root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve(); config = load_config(cp)
    if args.mode == "Preflight": result = {"schema": SCHEMA, "method_id": METHODS[args.method], "scenario_selection": args.scenario, "patience": 5 if args.method.endswith("P5") else None, "model_loaded": False, "optimizer_constructed": False, "test_accessed": False}
    elif args.mode == "SyntheticDryRun": result = {"schema": SCHEMA, "method_id": METHODS[args.method], "scenario_selection": args.scenario, "real_model_loaded": False, "test_accessed": False}
    elif not args.run_name: parser.error(f"{args.mode} requires RunName")
    elif args.mode == "Analyze": result = analyze(root, cp, args.run_name, args.method)
    else: result = execute(root, cp, args.benchmark_name, args.run_name, args.method, args.source_original_run_name, args.scenario)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
