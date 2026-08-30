"""Development-free 2% IFRU direct-influence adaptation for T5 Q/V LoRA."""
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
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F
import yaml
from transformers import T5Tokenizer

from src.diagnostics.t5_lora_influence_feasibility_audit import conjugate_gradient, flatten_tensors, split_vector
from src.diagnostics.t5_lora_influence_feasibility_audit_a2 import (
    build_fixed_a_basis,
    collect_qv_modules,
    estimate_lambda_max,
    install_fixed_ab_coordinate,
)
from src.diagnostics.t5_reconstructed_official import JsonPromptDataset, load_config, load_legacy_model, move_batch
from src.diagnostics.t5_step817_forget_conflict_audit import tensor_tree_hash
from src.if_a2_optimization.group_a_gradient_audit import GIB, masked_batch, masked_forward, token_and_sample_numerators
from src.if_a2_optimization.group_b_scale_audit import quantized_delta
from src.if_a2_optimization.group_c_gpu_resident_hvp import materialize_resident_panel, resident_hessian_vector_product
from src.paper_baselines.common import paper_model_manifest, verify_paper_model_manifest
from src.paper_if_a2.artifacts import atomic_torch_save, complete, publish_manifest, write_contract
from src.paper_if_a2.common import atomic_json, canonical_hash, directory_hash, require_formal_preflight, safe_run_name, sha256_file
from src.paper_if_a2.if_a2_method import load_method_config

SCHEMA = "paper-ifru-t5-direct-lora-2pct-v1"
METHOD_ID = "IFRU-T5-Direct-LoRA"
MARKER = "PAPER_IFRU_T5_DIRECT_LORA_2PCT_V1_COMPLETED"


def load_frozen(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"schema", "development_only", "test_access_policy", "experiment_root", "experiment_contract_sha256", "method_config", "output_root", "method", "coordinate", "curvature", "solver", "runtime", "forbidden"}
    if set(value) != required or value["schema"] != SCHEMA or value["development_only"] is not True or value["test_access_policy"] != "forbidden":
        raise ValueError("invalid IFRU-T5 config schema/policy")
    if value["experiment_root"] != "outputs/ru1/i02s42v1" or value["output_root"] != "outputs/ru1/i02s42v1/models/IFRU-T5/runs":
        raise ValueError("IFRU-T5 is frozen to the 2% experiment")
    expected_method = {"method_id": METHOD_ID, "source_algorithm": "IFRU", "direct_influence_only": True, "spillover_term": "zero", "hessian_definition": "full_train_empirical_risk", "curvature_approximation": "softmax_cross_entropy_ggn", "forget_gradient_reduction": "sample_mean", "update_scale": "k_over_n"}
    if value["method"] != expected_method:
        raise ValueError("IFRU mathematical contract changed")
    if value["coordinate"] != {"target_modules": ["q", "v"], "rank": 16, "alpha": 32, "trainable": "B_only", "initial_B": "zero", "basis_source": "forget_sample_mean_gradient"}:
        raise ValueError("IFRU coordinate changed")
    if value["curvature"] != {"dataset": "full_train", "panel_samples": 4096, "panel_seed": 42, "batch_size": 8, "reference_strategy": "detached_same_forward", "power_iterations": 12, "convergence_tolerance": .0001, "numerical_lower_bound": 1e-14, "relative_damping_ratio": .01}:
        raise ValueError("IFRU curvature changed")
    if value["solver"] != {"implementation": "standard_conjugate_gradient", "relative_residual_tolerance": .0001, "absolute_residual_tolerance": 1e-10, "max_iterations": 40, "residual_explosion_factor": 1000., "pap_absolute_tolerance": 1e-14}:
        raise ValueError("IFRU solver changed")
    if set(value["forbidden"]) != {"retrain_reference", "development_selection", "retain_projection", "trust_scale", "optimizer_step", "final_test"}:
        raise ValueError("IFRU forbidden operations changed")
    return value


def ifru_scale(k: int, n: int) -> float:
    if not 0 < k < n:
        raise ValueError("IFRU requires 0 < k < n")
    return k / n


def deterministic_panel(n: int, count: int, seed: int) -> list[int]:
    if count <= 0 or count > n:
        raise ValueError("invalid curvature panel size")
    generator = torch.Generator(device="cpu"); generator.manual_seed(seed)
    return torch.randperm(n, generator=generator)[:count].tolist()


def sample_mean_gradients(model: torch.nn.Module, dataset: JsonPromptDataset, indices: Sequence[int], parameters: Sequence[torch.Tensor], device: torch.device, batch_size: int, pad_token_id: int) -> tuple[list[torch.Tensor], dict[str, Any]]:
    accumulators = [torch.zeros_like(parameter, dtype=torch.float64, device="cpu") for parameter in parameters]
    loss_sum = 0.; valid_tokens = 0; token_counts: list[int] = []; calls = 0
    for start in range(0, len(indices), batch_size):
        chosen = list(indices[start:start + batch_size]); batch = move_batch(masked_batch(dataset, chosen, pad_token_id), device); output = masked_forward(model, batch)
        _, sample_sum, tokens = token_and_sample_numerators(output.logits, batch["target_ids"]); gradients = torch.autograd.grad(sample_sum, parameters)
        for accumulator, gradient in zip(accumulators, gradients): accumulator.add_(gradient.detach().to(dtype=torch.float64, device="cpu"))
        counts = batch["target_ids"].ne(-100).sum(dim=1).detach().cpu().tolist(); token_counts.extend(int(item) for item in counts); valid_tokens += tokens; loss_sum += float(sample_sum.detach().cpu()); calls += 1
        del chosen, batch, output, sample_sum, gradients
    if not indices or any(parameter.grad is not None for parameter in parameters):
        raise RuntimeError("invalid sample-mean gradient state")
    return [value / len(indices) for value in accumulators], {"reduction": "sample_mean", "samples": len(indices), "valid_tokens": valid_tokens, "minimum_tokens_per_sample": min(token_counts), "maximum_tokens_per_sample": max(token_counts), "loss": loss_sum / len(indices), "forward_batches": calls, "autograd_grad_calls": calls}


def make_sample_mean_ggn_operator(model: torch.nn.Module, batches: Sequence[dict[str, torch.Tensor]], parameters: Sequence[torch.Tensor], counter: dict[str, Any]) -> Callable[[torch.Tensor], torch.Tensor]:
    if not batches:
        raise ValueError("empty full-data curvature panel")
    device = parameters[0].device
    def operator(vector: torch.Tensor) -> torch.Tensor:
        pieces = split_vector(vector, parameters); total = torch.zeros(vector.numel(), dtype=torch.float64, device=device); samples = 0; started = time.perf_counter()
        for batch in batches:
            logits = masked_forward(model, batch).logits; reference = logits.detach(); probability = F.softmax(reference, dim=-1); token_kl = (probability * (F.log_softmax(reference, dim=-1) - F.log_softmax(logits, dim=-1))).sum(-1); mask = batch["target_ids"].ne(-100); counts = mask.sum(dim=1)
            if bool((counts <= 0).any()): raise RuntimeError("curvature sample without target token")
            loss = ((token_kl * mask).sum(dim=1) / counts).mean(); weighted = resident_hessian_vector_product(loss, parameters, pieces) * int(mask.shape[0]); total.add_(weighted); samples += int(mask.shape[0]); counter["hvp_batches"] = counter.get("hvp_batches", 0) + 1
            del logits, reference, probability, token_kl, mask, counts, loss, weighted
        result = (total / samples).cpu(); quadratic = float(torch.dot(vector, result)); tolerance = 1e-10 + 1e-8 * float(torch.linalg.vector_norm(vector)) * float(torch.linalg.vector_norm(result))
        if quadratic < -tolerance or not torch.isfinite(result).all(): raise RuntimeError("invalid GGN curvature")
        counter["operator_calls"] = counter.get("operator_calls", 0) + 1; counter.setdefault("operator_seconds", []).append(time.perf_counter() - started)
        print(f"[ifru:hvp] operator_call={counter['operator_calls']} batches={len(batches)} seconds={counter['operator_seconds'][-1]:.3f}", flush=True)
        return result
    return operator


def _lineage(root: Path, config: dict[str, Any], method: dict[str, Any]) -> tuple[dict[str, Any], Path, Path, Path]:
    experiment = root / config["experiment_root"]; manifest_path = experiment / "experiment_manifest.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_contract_sha256") != config["experiment_contract_sha256"] or manifest.get("final_test_accessed") is not False:
        raise ValueError("2% experiment contract mismatch")
    train = experiment / manifest["paths"]["train"]; forget = root / method["forget"]; original = root / method["original"]
    checks = {"train": manifest["sha256"]["train"], "forget": method["forget_sha256"], "original": method["original_sha256"]}
    for name, path in (("train", train), ("forget", forget), ("original", original)):
        if sha256_file(path) != checks[name]: raise ValueError(f"{name} SHA mismatch")
    counts = manifest["counts"]
    if counts["train"] != 60000 or counts["forget"] != 1258 or not math.isclose(manifest["actual_interaction_ratio"], 1258 / 60000, rel_tol=0., abs_tol=1e-16):
        raise ValueError("2% counts/ratio changed")
    return manifest, train, forget, original


def preflight(root: Path, config_path: Path) -> dict[str, Any]:
    config = load_frozen(config_path); method = load_method_config(root / config["method_config"]); manifest, train, forget, original = _lineage(root, config, method); panel = deterministic_panel(manifest["counts"]["train"], config["curvature"]["panel_samples"], config["curvature"]["panel_seed"])
    return {"schema": SCHEMA, "status": "READY", "method_id": METHOD_ID, "formula": "delta=(k/n)*(H_D_GGN+damping*I)^-1*g_F_sample_mean", "k": manifest["counts"]["forget"], "n": manifest["counts"]["train"], "scale": ifru_scale(manifest["counts"]["forget"], manifest["counts"]["train"]), "curvature_panel_sha256": canonical_hash(panel), "paths": {"train": str(train), "forget": str(forget), "original": str(original)}, "model_loaded": False, "optimizer_constructed": False, "development_loaded": False, "test_accessed": False}


def _save_adapter(path: Path, names: list[str], parameters: list[torch.Tensor], bases: dict[str, torch.Tensor]) -> None:
    path.mkdir(); state = {"B": {name: parameter.detach().cpu() for name, parameter in zip(names, parameters)}, "A": {name: value.detach().cpu() for name, value in bases.items()}, "schema": "ifru-t5-fixed-ab-v1", "alpha": 32, "rank": 16}; atomic_torch_save(path / "adapter_model.pt", state); loaded = torch.load(path / "adapter_model.pt", map_location="cpu", weights_only=True)
    if tensor_tree_hash(state["A"]) != tensor_tree_hash(loaded["A"]) or tensor_tree_hash(state["B"]) != tensor_tree_hash(loaded["B"]): raise RuntimeError("IFRU adapter reload mismatch")


def execute(root: Path, config_path: Path, run_name: str) -> dict[str, Any]:
    config = load_frozen(config_path); method = load_method_config(root / config["method_config"]); experiment, train_path, forget_path, checkpoint = _lineage(root, config, method); formal = require_formal_preflight(root); run_name = safe_run_name(run_name); destination = root / config["output_root"] / run_name
    if destination.exists(): raise FileExistsError(destination)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1: raise RuntimeError("IFRU Full requires exactly one CUDA GPU")
    device = torch.device("cuda:0"); free, _ = torch.cuda.mem_get_info(device)
    if free / GIB < config["runtime"]["minimum_free_gib"]: raise RuntimeError("insufficient free dedicated GPU memory")
    work = destination.parent / ".work"; work.mkdir(parents=True, exist_ok=True); stage = work / f"{run_name}.{uuid.uuid4().hex}.stage"; stage.mkdir(); model = None; resident = None; started = time.perf_counter(); before = sha256_file(checkpoint)
    try:
        torch.manual_seed(config["runtime"]["seed"]); torch.cuda.manual_seed_all(config["runtime"]["seed"]); torch.cuda.set_per_process_memory_fraction(config["runtime"]["allocator_fraction"], device); torch.cuda.reset_peak_memory_stats(device)
        base = load_config(root / method["base_config"], root); tokenizer = T5Tokenizer.from_pretrained(base["paths"]["model_dir"]); model = load_legacy_model(checkpoint).to(device).eval(); forget = JsonPromptDataset(forget_path, tokenizer); full_train = JsonPromptDataset(train_path, tokenizer); pad = tokenizer.pad_token_id
        if len(forget) != experiment["counts"]["forget"] or len(full_train) != experiment["counts"]["train"]: raise RuntimeError("dataset counts changed")
        modules = collect_qv_modules(model); module_names = [name for name, _ in modules]; weights = [module.weight for _, module in modules]
        for parameter in model.parameters(): parameter.requires_grad_(False)
        for parameter in weights: parameter.requires_grad_(True)
        weight_gradients, weight_meta = sample_mean_gradients(model, forget, range(len(forget)), weights, device, config["runtime"]["forget_batch_size"], pad); bases = {}; basis_rows = []
        for name, matrix in zip(module_names, weight_gradients): bases[name], row = build_fixed_a_basis(matrix, rank=16, name=name, seed=42); basis_rows.append(row)
        for parameter in weights: parameter.requires_grad_(False)
        names, parameters = install_fixed_ab_coordinate(model, bases, 32); gradient_parts, gradient_meta = sample_mean_gradients(model, forget, range(len(forget)), parameters, device, config["runtime"]["forget_batch_size"], pad); gradient = flatten_tensors(gradient_parts)
        panel_indices = deterministic_panel(len(full_train), config["curvature"]["panel_samples"], config["curvature"]["panel_seed"]); resident, panel_meta = materialize_resident_panel(full_train, panel_indices, config["curvature"]["batch_size"], pad, device); counter: dict[str, Any] = {}; operator = make_sample_mean_ggn_operator(model, resident, parameters, counter); curvature_started = time.perf_counter(); estimate = estimate_lambda_max(operator, gradient.numel(), seed=42, iterations=config["curvature"]["power_iterations"], convergence_tolerance=config["curvature"]["convergence_tolerance"], numerical_lower_bound=config["curvature"]["numerical_lower_bound"]); damping = estimate["lambda_max_hat"] * config["curvature"]["relative_damping_ratio"]
        solver = config["solver"]; cg = conjugate_gradient(operator, gradient, damping=damping, relative_tolerance=solver["relative_residual_tolerance"], absolute_tolerance=solver["absolute_residual_tolerance"], max_iterations=solver["max_iterations"], residual_explosion_factor=solver["residual_explosion_factor"], pap_tolerance=solver["pap_absolute_tolerance"]); direction = cg.pop("solution"); curvature_seconds = time.perf_counter() - curvature_started; scale = ifru_scale(len(forget), len(full_train)); base_flat = flatten_tensors([parameter.detach() for parameter in parameters]); actual = quantized_delta(base_flat, direction, scale)
        if float(torch.dot(gradient, actual)) <= 0: raise RuntimeError("IFRU deletion sign gate failed")
        with torch.no_grad():
            offset = 0
            for parameter in parameters:
                size = parameter.numel(); parameter.copy_((base_flat[offset:offset + size] + actual[offset:offset + size]).reshape_as(parameter).to(parameter)); offset += size
        verification_batch = move_batch(masked_batch(forget, [0], pad), device)
        with torch.no_grad(): candidate_logits = masked_forward(model, verification_batch).logits.detach().cpu()
        _save_adapter(stage / "adapter", names, parameters, bases); candidate_sha = tensor_tree_hash({"delta": actual}); trainable = sum(parameter.numel() for parameter in parameters)
        reload_model = load_legacy_model(checkpoint).to(device).eval(); saved = torch.load(stage / "adapter/adapter_model.pt", map_location=device, weights_only=True); reload_names, reload_parameters = install_fixed_ab_coordinate(reload_model, saved["A"], 32)
        with torch.no_grad():
            for name, parameter in zip(reload_names, reload_parameters): parameter.copy_(saved["B"][name].to(parameter))
            reload_logits = masked_forward(reload_model, verification_batch).logits.detach().cpu()
        adapter_reload_exact = torch.equal(candidate_logits, reload_logits); del reload_model, reload_names, reload_parameters, saved, reload_logits
        if not adapter_reload_exact: raise RuntimeError("IFRU adapter reload logits mismatch")
        peak_allocated = torch.cuda.max_memory_allocated(device); peak_reserved = torch.cuda.max_memory_reserved(device)
        if peak_reserved / GIB > config["runtime"]["hard_cap_reserved_gib"]: raise RuntimeError("IFRU reserved-memory hard cap exceeded")
        implementation_paths = [Path(__file__), config_path, root / "scripts/paper/run_ifru_t5_2pct_v1.ps1"]; implementation = {str(path.resolve().relative_to(root.resolve())).replace("\\", "/"): sha256_file(path) for path in implementation_paths}; timing = {"training_seconds": 0., "curvature_and_solve_seconds": curvature_seconds, "end_to_end_wall_seconds": time.perf_counter() - started, "augmentation_seconds": 0.}
        contract = {"schema": SCHEMA, "run_name": run_name, "git": formal["git"], "implementation": implementation, "deletion_experiment": method["deletion_experiment"], "sources": {"original": {"path": method["original"], "sha256": method["original_sha256"]}, "train": {"path": str(train_path.relative_to(root)).replace("\\", "/"), "sha256": sha256_file(train_path)}, "forget": {"path": method["forget"], "sha256": method["forget_sha256"]}}, "formula": "delta=(k/n)*(H_D_GGN+damping*I)^-1*g_F_sample_mean", "development_loaded": False, "test_accessed": False}; write_contract(stage, contract)
        result = {"schema": SCHEMA, "method_id": METHOD_ID, "candidate_sha256": candidate_sha, "adapter_reload_logits_exact": adapter_reload_exact, "k": len(forget), "n": len(full_train), "scale": scale, "forget_gradient": {**gradient_meta, "norm": float(torch.linalg.vector_norm(gradient)), "sha256": tensor_tree_hash({"g_F_sample_mean": gradient})}, "weight_gradient": weight_meta, "basis": {"sha256": canonical_hash([(row["name"], row["basis_sha256"]) for row in basis_rows]), "modules": len(basis_rows), "mean_captured_energy": sum(row["captured_frobenius_energy_ratio"] for row in basis_rows) / len(basis_rows)}, "curvature": {"definition": "full_train_sample_mean_softmax_cross_entropy_GGN", "approximation": True, "panel": panel_meta, "lambda_max": estimate, "damping": damping}, "cg": cg, "hvp_calls": counter.get("operator_calls", 0), "hvp_batch_calls": counter.get("hvp_batches", 0), "direction_norm": float(torch.linalg.vector_norm(direction)), "actual_delta_norm": float(torch.linalg.vector_norm(actual)), "g_dot_actual_delta": float(torch.dot(gradient, actual)), "spillover_term": 0., "retain_projection_used": False, "trust_scale_used": False, "retrain_loaded": False, "development_loaded": False, "optimizer_constructed": False, "optimizer_steps": 0, "trainable_parameters": trainable, "peak_gpu_allocated": peak_allocated, "peak_gpu_reserved": peak_reserved, "test_accessed": False}; atomic_json(stage / "method_result.json", result); atomic_json(stage / "timing.json", timing)
        manifest = paper_model_manifest(method_id=METHOD_ID, display_name="IFRU-T5 Direct Influence (2%)", run_name=run_name, model_type="if_a2_fixed_ab", artifacts=[stage / "adapter"], root=root, contract=contract, config_sha256=sha256_file(config_path), trainable_parameters=trainable, total_parameters=220000000, optimizer_steps=0, timing=timing, resources={"peak_gpu_allocated": peak_allocated, "peak_gpu_reserved": peak_reserved, "shared_gpu_memory_used": False}, completion_marker=MARKER, extra={"ifru_direct_scale": scale, "spillover_term": 0., "hessian_approximation": "full_train_panel_softmax_cross_entropy_ggn", "t5_adaptation": True}); manifest["artifacts"][0]["path"] = str((destination / "adapter").resolve()); atomic_json(stage / "paper_model_manifest.json", manifest); publish_manifest(stage, ["adapter", "contract.json", "method_result.json", "timing.json", "paper_model_manifest.json"], {"schema": SCHEMA, "optimizer_steps": 0, "test_accessed": False}); complete(stage, MARKER)
        if sha256_file(checkpoint) != before: raise RuntimeError("Original checkpoint changed")
        destination.parent.mkdir(parents=True, exist_ok=True); os.replace(stage, destination); verify_paper_model_manifest(json.loads((destination / "paper_model_manifest.json").read_text(encoding="utf-8")), verify_artifacts=True)
        return {"status": "COMPLETED", "run_dir": str(destination), "method_id": METHOD_ID, "scale": scale, "hvp_calls": result["hvp_calls"], "optimizer_steps": 0, "test_accessed": False}
    finally:
        if resident is not None: del resident
        if model is not None: del model
        if stage.exists(): shutil.rmtree(stage)
        gc.collect(); torch.cuda.empty_cache()


def analyze(root: Path, config_path: Path, run_name: str) -> dict[str, Any]:
    config = load_frozen(config_path); run = root / config["output_root"] / safe_run_name(run_name); required = {"COMPLETED", "adapter", "contract.json", "manifest.json", "method_result.json", "paper_model_manifest.json", "timing.json"}
    if not run.is_dir() or {path.name for path in run.iterdir()} != required or (run / "COMPLETED").read_text(encoding="utf-8") != MARKER + "\n": raise ValueError("invalid IFRU run")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8")); result = json.loads((run / "method_result.json").read_text(encoding="utf-8")); model = json.loads((run / "paper_model_manifest.json").read_text(encoding="utf-8")); verify_paper_model_manifest(model, verify_artifacts=True)
    for name, expected in manifest["files"].items():
        path = run / name; actual = directory_hash(path) if path.is_dir() else sha256_file(path)
        if actual != expected: raise ValueError(f"IFRU artifact mismatch: {name}")
    if result["optimizer_steps"] != 0 or result["test_accessed"] is not False or result["retain_projection_used"] is not False or result["adapter_reload_logits_exact"] is not True: raise ValueError("IFRU scientific contract mismatch")
    return {"status": "COMPLETED", "run_dir": str(run), "method_id": METHOD_ID, "scale": result["scale"], "candidate_sha256": result["candidate_sha256"], "hvp_calls": result["hvp_calls"], "wall_time_seconds": model["wall_time_seconds"], "peak_reserved_gib": result["peak_gpu_reserved"] / GIB, "optimizer_steps": 0, "test_accessed": False}


def synthetic() -> dict[str, Any]:
    return {"schema": SCHEMA, "method_id": METHOD_ID, "formula": "delta=(k/n)*(H_D_GGN+damping*I)^-1*g_F_sample_mean", "scale": 1258 / 60000, "spillover_term": 0., "retain_projection_used": False, "trust_scale_used": False, "model_loaded": False, "optimizer_constructed": False, "test_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path.cwd()); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--mode", choices=["Preflight", "BudgetAudit", "SyntheticDryRun", "Full", "Analyze"], default="Preflight"); parser.add_argument("--run-name", default=""); args = parser.parse_args(); root = args.root.resolve(); config = args.config.resolve()
    if args.mode == "Preflight": result = preflight(root, config)
    elif args.mode == "BudgetAudit": result = {**preflight(root, config), "estimated_hvp_operator_calls": 53, "estimated_hvp_batch_calls": 53 * 512, "estimated_peak_reserved_gib_upper_bound": 14., "formal_model_loaded": False}
    elif args.mode == "SyntheticDryRun": result = synthetic()
    elif not args.run_name: parser.error(f"{args.mode} requires RunName")
    elif args.mode == "Analyze": result = analyze(root, config, args.run_name)
    else: result = execute(root, config, args.run_name)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
