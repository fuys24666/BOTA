"""Method runners for the frozen 200-step BOTA short benchmark."""
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

from src.bota_if import p1_trajectory_transport_audit as p1
from src.bota_if import p2b_full_module_adamw_transport_audit as p2b
from src.bota_if.p1_trajectory_transport_audit import StepBudget
from src.diagnostics.t5_lora_influence_feasibility_audit import conjugate_gradient, flatten_tensors
from src.diagnostics.t5_lora_influence_feasibility_audit_a2 import estimate_lambda_max
from src.diagnostics.t5_reconstructed_official import JsonPromptDataset, load_config as load_t5_config, move_batch
from src.if_a2_optimization.group_a_gradient_audit import GIB, masked_batch
from src.if_a2_optimization.group_b_scale_audit import quantized_delta
from src.if_a2_optimization.group_c_gpu_resident_hvp import materialize_resident_panel
from src.paper_baselines import partitioned
from src.paper_if_a2.artifacts import atomic_torch_save
from src.paper_if_a2.common import atomic_json, canonical_hash, directory_hash, git_snapshot, safe_run_name, seed_everything, sha256_file
from src.paper_ratio_suite.ifru_t5 import deterministic_panel, make_sample_mean_ggn_operator, sample_mean_gradients

from .protocol import SCHEMA, freeze_registry, load_config, prepare, source_sha256, validate_prepared
from .timing import SCENARIO_CHOICES, select_scenarios, timing_record

METHODS = {
    "Original": "Original-Short",
    "Retrain": "Retrain-Short",
    "IFRU": "IFRU-Short-LoRA",
    "SISA": "SISA-Short-T5",
    "RecEraser": "RecEraser-Adapter-Short",
    "BOTA": "BOTA-T2-Short",
}
MARKER = "BOTA_SHORT_METHOD_V1_COMPLETED"


def allocator_fraction_for(total_gib: float, requested_fraction: float, hard_peak_gib: float, safety_gib: float = .1) -> float:
    if total_gib <= 0 or not 0 < requested_fraction <= 1 or hard_peak_gib <= safety_gib:
        raise ValueError("invalid GPU memory guard")
    return min(requested_fraction, (hard_peak_gib - safety_gib) / total_gib)


def _engine(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    del root
    return {"schedule": {"batch_size": 16}, "optimizer": config["optimizer"]}


def _base_t5_config(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Resolve the strict T5 config through the BOTA P1 runtime config."""
    p1_config = yaml.safe_load((root / config["source"]["base_config"]).read_text(encoding="utf-8"))
    base_path = p1_config.get("base_config")
    if not isinstance(base_path, str) or not base_path:
        raise ValueError("BOTA P1 config does not name its strict T5 base_config")
    return load_t5_config(root / base_path, root)


def _runtime_config(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    value = yaml.safe_load((root / config["source"]["base_config"]).read_text(encoding="utf-8"))
    # The benchmark protocol may deliberately bind a newly trained, immutable
    # recommendation Original while retaining the same data/request registry.
    value["original_checkpoint"] = config["source"]["original_checkpoint"]
    run_seed = int(config["protocol"].get("run_seed", config["protocol"]["seed"]))
    value["schedule"] = {"seed": run_seed, "batch_size": 16}
    value["coordinate"]["fixed_a_seed"] = int(config["coordinate"]["fixed_a_seed"])
    value["optimizer"] = config["optimizer"]
    return value


def _copy_parameters(names: Sequence[str], parameters: Sequence[torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().float().cpu().clone() for name, parameter in zip(names, parameters)}


def _save_fixed_ab(path: Path, names: Sequence[str], values: dict[str, torch.Tensor], bases: dict[str, torch.Tensor], method_id: str) -> dict[str, Any]:
    path.mkdir(parents=True); state = {"A": {name: value.detach().cpu() for name, value in bases.items()}, "B": {name: values[name].detach().cpu() for name in names}, "schema": "bota-short-fixed-ab-v1", "rank": 16, "alpha": 32, "method_id": method_id}
    atomic_torch_save(path / "adapter_model.pt", state); atomic_json(path / "adapter_config.json", {"format": "paper-fixed-A-B-LoRA-v1", "target_modules": ["q", "v"], "rank": 16, "alpha": 32, "method_id": method_id})
    loaded = torch.load(path / "adapter_model.pt", map_location="cpu", weights_only=True)
    if canonical_hash(sorted(loaded["B"])) != canonical_hash(sorted(state["B"])) or any(not torch.equal(loaded["B"][name], state["B"][name]) for name in names): raise RuntimeError("adapter reload mismatch")
    return {"path": str(path.resolve()), "sha256": directory_hash(path), "type": "directory", "reload_exact": True}


def _train(model, dataset, order, user_ids, target_users, parameters, device, pad, engine, budget, arm):
    optimizer = p1._optimizer(parameters, engine); targets = set(map(int, target_users)); trace = []
    for step, start in enumerate(range(0, len(order), 16), 1):
        chosen = list(order[start:start + 16]); batch_users = [int(user_ids[index]) for index in chosen]; batch = move_batch(masked_batch(dataset, chosen, pad), device); losses = p1._sample_losses(model, batch)
        weights = torch.tensor([0. if user in targets else 1. for user in batch_users], dtype=losses.dtype, device=device); loss = torch.sum(losses * weights) / len(chosen); optimizer.zero_grad(set_to_none=True); loss.backward(); budget.step(optimizer, arm); trace.append({"step": step, "batch_hash": canonical_hash(chosen), "masked_slots": int(sum(user in targets for user in batch_users))}); del batch, losses, weights, loss
    return _copy_parameters([name for name, _ in zip(parameters, parameters)], parameters), optimizer, trace


def _train_named(model, dataset, order, user_ids, target_users, names, parameters, device, pad, engine, budget, arm):
    optimizer = p1._optimizer(parameters, engine); targets = set(map(int, target_users)); trace = []
    for step, start in enumerate(range(0, len(order), 16), 1):
        chosen = list(order[start:start + 16]); batch_users = [int(user_ids[index]) for index in chosen]; batch = move_batch(masked_batch(dataset, chosen, pad), device); losses = p1._sample_losses(model, batch); weights = torch.tensor([0. if user in targets else 1. for user in batch_users], dtype=losses.dtype, device=device); loss = torch.sum(losses * weights) / len(chosen); optimizer.zero_grad(set_to_none=True); loss.backward(); budget.step(optimizer, arm); trace.append({"step": step, "batch_hash": canonical_hash(chosen), "masked_slots": int(sum(user in targets for user in batch_users))}); del batch, losses, weights, loss
    return _copy_parameters(names, parameters), optimizer, trace


def _prepare_influence_endpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, parameters: Sequence[torch.Tensor]) -> None:
    """Freeze the trained endpoint's runtime state before autograd.grad-based IF."""
    optimizer.zero_grad(set_to_none=True); model.eval()
    if any(parameter.grad is not None for parameter in parameters):
        raise RuntimeError("IFRU endpoint retained parameter gradients after explicit clearing")


def _model_context(root: Path, config: dict[str, Any], registry: dict[str, Any], device: torch.device, fixed_a=None):
    runtime = _runtime_config(root, config); model, names, parameters, bases, basis_report = p1._fresh_runtime(root, runtime, device, fixed_a); tokenizer = T5Tokenizer.from_pretrained(_base_t5_config(root, config)["paths"]["model_dir"]); dataset = JsonPromptDataset(root / config["source"]["train_json"], tokenizer)
    if "train_user_ids" in config["source"]:
        users = json.loads((root / config["source"]["train_user_ids"]).read_text(encoding="utf-8"))
    else:
        users, _ = p1._train_user_ids_only(root / config["source"]["raw_data"])
    if len(dataset) != registry["global_train_samples"] or len(users) != registry["global_train_samples"]: raise RuntimeError("short model source count mismatch")
    return model, names, parameters, bases, basis_report, tokenizer, dataset, users


def _scenario_manifest(path: Path, *, method_id: str, scenario: dict[str, Any], artifact: dict[str, Any], steps: int, timing: float, peak: int, extra: dict[str, Any], phase_timing: dict[str, Any] | None = None) -> None:
    if phase_timing is not None:
        atomic_json(path / "phase_timing.json", phase_timing)
    atomic_json(path / "scenario_manifest.json", {"schema": SCHEMA, "method_id": method_id, "scenario_id": scenario["id"], "request_hash": scenario["request_hash"], "model_type": "if_a2_fixed_ab", "artifact": artifact, "optimizer_steps": steps, "window_optimizer_steps": 200, "deleted_interactions": scenario["deleted_interactions"], "wall_time_seconds": timing, "peak_gpu_reserved": peak, "phase_timing": phase_timing, "test_accessed": False, **extra})


def _fixed_ab_methods(root: Path, config: dict[str, Any], registry: dict[str, Any], method: str, stage: Path) -> dict[str, Any]:
    device = torch.device("cuda:0"); engine = _engine(root, config); run_seed = int(config["protocol"].get("run_seed", config["protocol"]["seed"])); order = registry["order"]; scenario_rows = registry["scenarios"]; selected_users = sorted(set(user for row in scenario_rows for user in row["users"])); model = None; started = time.perf_counter(); torch.cuda.reset_peak_memory_stats(device)
    try:
        if method == "BOTA":
            seed_everything(run_seed)
            phase_started = time.perf_counter(); model, names, parameters, bases, basis_report, tokenizer, dataset, users = _model_context(root, config, registry, device); initialization_seconds = time.perf_counter() - phase_started
            phase_started = time.perf_counter(); budget = StepBudget(200); canonical, states, trace, optimizer_hash, rng_hash = p2b.run_canonical_full(model, dataset, order, users, selected_users, parameters, names, device, tokenizer.pad_token_id, engine, budget); offline_seconds = time.perf_counter() - phase_started; index = {user: slot for slot, user in enumerate(selected_users)}
            for scenario in scenario_rows:
                scenario_started = time.perf_counter(); phase_started = time.perf_counter()
                candidate = {name: canonical[name].clone() for name in names}
                for module, name in enumerate(names): candidate[name].add_(states["T2_AdamW_full_state"][module]["theta"][[index[user] for user in scenario["users"]]].sum(0).float())
                online_compute_seconds = time.perf_counter() - phase_started; scenario_dir = stage / "scenarios" / scenario["id"]; scenario_dir.mkdir(parents=True); phase_started = time.perf_counter(); artifact = _save_fixed_ab(scenario_dir / "adapter", names, candidate, bases, METHODS[method]); publication_seconds = time.perf_counter() - phase_started; elapsed = time.perf_counter() - scenario_started
                phase_timing = timing_record(scenario=scenario["id"], initialization_seconds=initialization_seconds, offline_construction_seconds=offline_seconds, online_compute_seconds=online_compute_seconds, adapter_publication_seconds=publication_seconds, end_to_end_seconds=initialization_seconds + offline_seconds + elapsed, details={"trajectory_transport_seconds": offline_seconds, "online_vector_composition_seconds": online_compute_seconds})
                _scenario_manifest(scenario_dir, method_id=METHODS[method], scenario=scenario, artifact=artifact, steps=200, timing=time.perf_counter()-started, peak=torch.cuda.max_memory_reserved(device), phase_timing=phase_timing, extra={"online_optimizer_steps": 0, "offline_trajectory_steps": 200, "transport": "T2_AdamW_full_state", "canonical_trace_sha256": canonical_hash(trace), "optimizer_state_sha256": optimizer_hash, "rng_sha256": rng_hash})
        else:
            fixed_a = None
            cached_original = None
            for number, scenario in enumerate(scenario_rows):
                if method == "Original" and cached_original is not None:
                    scenario_dir = stage / "scenarios" / scenario["id"]; scenario_dir.mkdir(parents=True); phase_started = time.perf_counter(); artifact = _save_fixed_ab(scenario_dir / "adapter", cached_original[0], cached_original[1], cached_original[2], METHODS[method]); publication_seconds = time.perf_counter() - phase_started; phase_timing = timing_record(scenario=scenario["id"], initialization_seconds=0.0, offline_construction_seconds=0.0, online_compute_seconds=0.0, adapter_publication_seconds=publication_seconds, end_to_end_seconds=publication_seconds); _scenario_manifest(scenario_dir, method_id=METHODS[method], scenario=scenario, artifact=artifact, steps=200, timing=time.perf_counter()-started, peak=torch.cuda.max_memory_reserved(device), phase_timing=phase_timing, extra={"masked_slots_total": 0, "batch_denominator_preserved": True, "trace_sha256": cached_original[3], "shared_canonical_endpoint": True}); continue
                seed_everything(run_seed)
                scenario_started = time.perf_counter(); phase_started = time.perf_counter(); model, names, parameters, bases, basis_report, tokenizer, dataset, users = _model_context(root, config, registry, device, fixed_a); initialization_seconds = time.perf_counter() - phase_started; fixed_a = {name: value.detach().cpu().clone() for name, value in bases.items()}; budget = StepBudget(200); targets = [] if method == "Original" else scenario["users"]; phase_started = time.perf_counter(); values, optimizer, trace = _train_named(model, dataset, order, users, targets, names, parameters, device, tokenizer.pad_token_id, engine, budget, f"{method}_{scenario['id']}"); training_seconds = time.perf_counter() - phase_started; scenario_dir = stage / "scenarios" / scenario["id"]; scenario_dir.mkdir(parents=True); phase_started = time.perf_counter(); artifact = _save_fixed_ab(scenario_dir / "adapter", names, values, bases, METHODS[method]); publication_seconds = time.perf_counter() - phase_started; phase_timing = timing_record(scenario=scenario["id"], initialization_seconds=initialization_seconds, offline_construction_seconds=training_seconds if method == "Original" else 0.0, online_compute_seconds=0.0 if method == "Original" else training_seconds, adapter_publication_seconds=publication_seconds, end_to_end_seconds=time.perf_counter() - scenario_started, details={"optimizer_training_seconds": training_seconds}); _scenario_manifest(scenario_dir, method_id=METHODS[method], scenario=scenario, artifact=artifact, steps=200, timing=time.perf_counter()-started, peak=torch.cuda.max_memory_reserved(device), phase_timing=phase_timing, extra={"run_seed": run_seed, "masked_slots_total": 0 if method == "Original" else scenario["deleted_interactions"], "batch_denominator_preserved": True, "trace_sha256": canonical_hash(trace)})
                if method == "Original": cached_original = (list(names), values, {name: value.detach().cpu().clone() for name, value in bases.items()}, canonical_hash(trace))
                del model, optimizer; model = None; gc.collect(); torch.cuda.empty_cache()
        return {"physical_optimizer_steps": 200 if method in {"BOTA", "Original"} else 200 * len(scenario_rows), "online_optimizer_steps": 0 if method == "BOTA" else None, "wall_time_seconds": time.perf_counter()-started, "peak_gpu_reserved": torch.cuda.max_memory_reserved(device)}
    finally:
        if model is not None: del model
        gc.collect(); torch.cuda.empty_cache()


def _ifru(root: Path, config: dict[str, Any], registry: dict[str, Any], stage: Path) -> dict[str, Any]:
    device = torch.device("cuda:0"); engine = _engine(root, config); run_seed = int(config["protocol"].get("run_seed", config["protocol"]["seed"])); seed_everything(run_seed); started = time.perf_counter(); operator_calls = 0; torch.cuda.reset_peak_memory_stats(device); phase_started = time.perf_counter(); model, names, parameters, bases, _, tokenizer, dataset, users = _model_context(root, config, registry, device); initialization_seconds = time.perf_counter() - phase_started; phase_started = time.perf_counter(); budget = StepBudget(200); _, optimizer, trace = _train_named(model, dataset, registry["order"], users, [], names, parameters, device, tokenizer.pad_token_id, engine, budget, "IFRU_shared_base"); _prepare_influence_endpoint(model, optimizer, parameters); panel_indices = [registry["order"][index] for index in deterministic_panel(3200, 512, run_seed)]; resident, panel_meta = materialize_resident_panel(dataset, panel_indices, 8, tokenizer.pad_token_id, device); offline_seconds = time.perf_counter() - phase_started
    for scenario in registry["scenarios"]:
        scenario_started = time.perf_counter(); phase_started = time.perf_counter(); forget_indices = scenario["forget_train_indices"]; gradient_parts, gradient_meta = sample_mean_gradients(model, dataset, forget_indices, parameters, device, 8, tokenizer.pad_token_id); gradient = flatten_tensors(gradient_parts); gradient_seconds = time.perf_counter() - phase_started; counter: dict[str, Any] = {}; operator = make_sample_mean_ggn_operator(model, resident, parameters, counter); phase_started = time.perf_counter(); estimate = estimate_lambda_max(operator, gradient.numel(), seed=run_seed, iterations=12, convergence_tolerance=.0001, numerical_lower_bound=1e-14); lambda_seconds = time.perf_counter() - phase_started; damping = estimate["lambda_max_hat"] * .01; phase_started = time.perf_counter(); cg = conjugate_gradient(operator, gradient, damping=damping, relative_tolerance=.0001, absolute_tolerance=1e-10, max_iterations=40, residual_explosion_factor=1000., pap_tolerance=1e-14, allow_truncated_solution=True); cg_seconds = time.perf_counter() - phase_started; phase_started = time.perf_counter(); direction = cg.pop("solution"); scale = scenario["deleted_interactions"] / scenario["window_interactions"]; actual = quantized_delta(flatten_tensors([parameter.detach() for parameter in parameters]), direction, scale); offset = 0; candidate = {}
        for name, parameter in zip(names, parameters):
            size = parameter.numel(); candidate[name] = (parameter.detach().reshape(-1).cpu() + actual[offset:offset+size].cpu()).reshape_as(parameter); offset += size
        candidate_seconds = time.perf_counter() - phase_started; online_compute_seconds = gradient_seconds + lambda_seconds + cg_seconds + candidate_seconds; scenario_dir = stage / "scenarios" / scenario["id"]; scenario_dir.mkdir(parents=True); phase_started = time.perf_counter(); artifact = _save_fixed_ab(scenario_dir / "adapter", names, candidate, bases, METHODS["IFRU"]); publication_seconds = time.perf_counter() - phase_started; elapsed = time.perf_counter() - scenario_started; phase_timing = timing_record(scenario=scenario["id"], initialization_seconds=initialization_seconds, offline_construction_seconds=offline_seconds, online_compute_seconds=online_compute_seconds, adapter_publication_seconds=publication_seconds, end_to_end_seconds=initialization_seconds + offline_seconds + elapsed, details={"forget_gradient_seconds": gradient_seconds, "lambda_estimation_seconds": lambda_seconds, "cg_seconds": cg_seconds, "candidate_reconstruction_seconds": candidate_seconds})
        _scenario_manifest(scenario_dir, method_id=METHODS["IFRU"], scenario=scenario, artifact=artifact, steps=200, timing=time.perf_counter()-started, peak=torch.cuda.max_memory_reserved(device), phase_timing=phase_timing, extra={"online_optimizer_steps": 0, "canonical_prefix_steps": 200, "formula": "delta=(k/3200)*(H_window_GGN+damping*I)^-1*g_F_sample_mean", "deletion_scale": scale, "coordinate_adaptation": "request-independent fixed-A B-only", "curvature_panel_samples": 512, "hvp_calls": counter.get("operator_calls", 0), "cg": cg, "gradient": gradient_meta, "panel": panel_meta, "trace_sha256": canonical_hash(trace)}); operator_calls += counter.get("operator_calls", 0); del gradient, direction, actual; gc.collect(); torch.cuda.empty_cache()
    del model, optimizer, resident
    return {"physical_optimizer_steps": 200, "online_optimizer_steps": 0, "hvp_calls": operator_calls, "wall_time_seconds": time.perf_counter()-started, "peak_gpu_reserved": torch.cuda.max_memory_reserved(device)}


def _partitioned(root: Path, config: dict[str, Any], protocol: Path, registry: dict[str, Any], method: str, run_name: str, stage: Path) -> dict[str, Any]:
    kind = "sisa" if method == "SISA" else "receraser"; component_rows = []; started = time.perf_counter()
    for scenario in registry["scenarios"]:
        config_path = protocol / "data" / scenario["id"] / f"{kind}.yaml"; component_name = f"{safe_run_name(run_name)}-{scenario['id']}"; component_config = partitioned.load_config(config_path, partitioned.SCHEMAS[kind]); component = root / component_config["output_root"] / component_name
        if component.exists():
            verified = partitioned.analyze_run(component, partitioned.MARKERS[kind]); result = {"status": verified["status"], "run_dir": str(component), "resumed_from_completed_component": True, "test_accessed": False}
        else: result = partitioned.execute(root, config_path, component_name, kind, False, partition_seed_override=int(config["protocol"]["seed"]))
        component = Path(result["run_dir"]); manifest = json.loads((component / "paper_model_manifest.json").read_text(encoding="utf-8")); scenario_dir = stage / "scenarios" / scenario["id"]; scenario_dir.mkdir(parents=True); phase_timing = timing_record(scenario=scenario["id"], initialization_seconds=0.0, offline_construction_seconds=0.0, online_compute_seconds=float(manifest["wall_time_seconds"]), adapter_publication_seconds=None, end_to_end_seconds=float(manifest["wall_time_seconds"]), publication_included_in_online_compute=True); atomic_json(scenario_dir / "phase_timing.json", phase_timing); atomic_json(scenario_dir / "scenario_manifest.json", {"schema": SCHEMA, "method_id": METHODS[method], "scenario_id": scenario["id"], "request_hash": scenario["request_hash"], "model_type": manifest["model_type"], "component_run": str(component.resolve()), "component_manifest_sha256": sha256_file(component / "paper_model_manifest.json"), "optimizer_steps": manifest["optimizer_steps"], "window_optimizer_steps": 200, "deleted_interactions": scenario["deleted_interactions"], "wall_time_seconds": manifest["wall_time_seconds"], "phase_timing": phase_timing, "test_accessed": False}); component_rows.append(result)
    return {"physical_optimizer_steps": sum(json.loads((Path(row["run_dir"]) / "paper_model_manifest.json").read_text(encoding="utf-8"))["optimizer_steps"] for row in component_rows), "wall_time_seconds": time.perf_counter()-started, "component_runs": [row["run_dir"] for row in component_rows]}


def preflight(root: Path, config_path: Path, benchmark_name: str, method: str, scenario: str = "All") -> dict[str, Any]:
    config = load_config(config_path); registry = freeze_registry(root, config); selected = select_scenarios(registry["scenarios"], scenario)
    return {"schema": SCHEMA, "mode": "Preflight", "method_id": METHODS[method], "benchmark_name": benchmark_name, "scenario_selection": scenario, "scenarios": [{"id": row["id"], "composition": row["composition"], "deleted_interactions": row["deleted_interactions"], "window_ratio": row["actual_window_ratio"]} for row in selected], "optimizer_steps": 200, "model_loaded": False, "optimizer_constructed": False, "development_loaded": False, "test_accessed": False}


def execute(root: Path, config_path: Path, benchmark_name: str, run_name: str, method: str, scenario: str = "All") -> dict[str, Any]:
    config = load_config(config_path); protocol, contract, registry = validate_prepared(root, config, benchmark_name); git = git_snapshot(root); selected = select_scenarios(registry["scenarios"], scenario); active_registry = {**registry, "scenarios": selected}
    if not git["clean"]: raise RuntimeError("formal short benchmark requires clean Git")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1: raise RuntimeError("exactly one CUDA GPU required")
    device = torch.device("cuda:0"); free, _ = torch.cuda.mem_get_info(device)
    if free / GIB < config["runtime"]["minimum_free_gib"]: raise RuntimeError("insufficient dedicated GPU memory")
    total_gib = torch.cuda.get_device_properties(device).total_memory / GIB; effective_fraction = allocator_fraction_for(total_gib, config["runtime"]["allocator_fraction"], config["runtime"]["hard_peak_reserved_gib"]); torch.cuda.set_per_process_memory_fraction(effective_fraction, device); run_name = safe_run_name(run_name); destination = root / config["output_root"] / "models" / METHODS[method] / run_name
    if destination.exists(): raise FileExistsError(destination)
    stage = destination.parent / ".work" / f"{run_name}.{uuid.uuid4().hex}.stage"; stage.mkdir(parents=True); original_before = source_sha256(root / config["source"]["original_checkpoint"])
    try:
        if method in {"Original", "Retrain", "BOTA"}: result = _fixed_ab_methods(root, config, active_registry, method, stage)
        elif method == "IFRU": result = _ifru(root, config, active_registry, stage)
        else: result = _partitioned(root, config, protocol, active_registry, method, run_name, stage)
        if source_sha256(root / config["source"]["original_checkpoint"]) != original_before: raise RuntimeError("Original checkpoint changed")
        peak = int(torch.cuda.max_memory_reserved(device));
        if peak / GIB > config["runtime"]["hard_peak_reserved_gib"]: raise RuntimeError("GPU reserved-memory hard cap exceeded")
        run_seed = int(config["protocol"].get("run_seed", config["protocol"]["seed"])); aggregate = {"schema": SCHEMA, "status": "COMPLETED", "method_id": METHODS[method], "run_name": run_name, "benchmark_name": benchmark_name, "run_seed": run_seed, "registry_sha256": registry["registry_sha256"], "scenarios": [row["id"] for row in selected], "scenario_selection": scenario, "physical_optimizer_steps": result["physical_optimizer_steps"], "authoritative_optimizer_steps_committed": 0, "git_head": git["head"], "source_original_sha256": original_before, "memory_guard": {"requested_allocator_fraction": config["runtime"]["allocator_fraction"], "effective_allocator_fraction": effective_fraction, "hard_peak_reserved_gib": config["runtime"]["hard_peak_reserved_gib"], "safety_gib": .1}, "test_accessed": False, **result}; atomic_json(stage / "run_state.json", aggregate); atomic_json(stage / "contract.json", {**contract, "method_id": METHODS[method], "run_name": run_name, "run_seed": run_seed, "scenario_selection": scenario, "git": git, "test_accessed": False}); (stage / "COMPLETED").write_text(MARKER + "\n", encoding="utf-8", newline="\n"); atomic_json(stage / "manifest.json", {"schema": SCHEMA, "method_id": METHODS[method], "run_state_sha256": sha256_file(stage / "run_state.json"), "contract_sha256": sha256_file(stage / "contract.json"), "scenario_manifests": {row["id"]: sha256_file(stage / "scenarios" / row["id"] / "scenario_manifest.json") for row in selected}, "phase_timings": {row["id"]: sha256_file(stage / "scenarios" / row["id"] / "phase_timing.json") for row in selected}, "published_atomically": True, "test_accessed": False}); destination.parent.mkdir(parents=True, exist_ok=True); os.replace(stage, destination)
        return {"status": "COMPLETED", "method_id": METHODS[method], "run_dir": str(destination), "scenarios": aggregate["scenarios"], "physical_optimizer_steps": result["physical_optimizer_steps"], "test_accessed": False}
    finally:
        if stage.exists(): shutil.rmtree(stage)
        gc.collect(); torch.cuda.empty_cache()


def analyze(root: Path, config_path: Path, run_name: str, method: str) -> dict[str, Any]:
    config = load_config(config_path); run = root / config["output_root"] / "models" / METHODS[method] / safe_run_name(run_name); required = {"COMPLETED", "contract.json", "manifest.json", "run_state.json", "scenarios"}
    if not run.is_dir() or {path.name for path in run.iterdir()} != required or (run / "COMPLETED").read_text(encoding="utf-8") != MARKER + "\n": raise ValueError("invalid short method run")
    state = json.loads((run / "run_state.json").read_text(encoding="utf-8")); manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(run / "run_state.json") != manifest["run_state_sha256"] or state["status"] != "COMPLETED" or state["test_accessed"] is not False: raise ValueError("short method run integrity mismatch")
    return {"status": "COMPLETED", "method_id": METHODS[method], "run_dir": str(run), "scenarios": state["scenarios"], "physical_optimizer_steps": state["physical_optimizer_steps"], "test_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path.cwd()); parser.add_argument("--config", type=Path, default=Path("configs/bota_short_benchmark_v1.yaml")); parser.add_argument("--mode", choices=["Prepare", "Preflight", "SyntheticDryRun", "Full", "Analyze"], default="Preflight"); parser.add_argument("--method", choices=list(METHODS), default="Original"); parser.add_argument("--benchmark-name", default=""); parser.add_argument("--run-name", default=""); parser.add_argument("--scenario", choices=SCENARIO_CHOICES, default="All"); args = parser.parse_args(); root = args.root.resolve(); config_path = (root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    if not args.benchmark_name: parser.error("BenchmarkName is required")
    if args.mode == "Prepare": result = prepare(root, config_path, args.benchmark_name)
    elif args.mode == "Preflight": result = preflight(root, config_path, args.benchmark_name, args.method, args.scenario)
    elif args.mode == "SyntheticDryRun": result = {"schema": SCHEMA, "method_id": METHODS[args.method], "scenario_selection": args.scenario, "scenarios": [row["id"] for row in select_scenarios([{"id": value} for value in SCENARIO_CHOICES[1:]], args.scenario)], "real_model_loaded": False, "optimizer_constructed": False, "test_accessed": False}
    elif not args.run_name: parser.error(f"{args.mode} requires RunName")
    elif args.mode == "Analyze": result = analyze(root, config_path, args.run_name, args.method)
    else: result = execute(root, config_path, args.benchmark_name, args.run_name, args.method, args.scenario)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
