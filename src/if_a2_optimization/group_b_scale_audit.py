"""Group B: corrected-masking, zero-commit IF-A2 scale audit for the 2% split.

The audit reconstructs one retain-safe influence direction, evaluates frozen
scale arms transactionally, and restores the authoritative Original model.
It never constructs an optimizer, loads Retrain, or publishes a candidate.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import yaml
from transformers import T5Tokenizer

from src.diagnostics.t5_lora_influence_feasibility_audit import (
    _temporary_delta,
    conjugate_gradient,
    flatten_tensors,
    hessian_vector_product,
    project_update_space,
    self_kl_loss,
    split_vector,
)
from src.diagnostics.t5_lora_influence_feasibility_audit_a2 import (
    analytic_b_gradient,
    build_fixed_a_basis,
    collect_qv_modules,
    estimate_lambda_max,
    install_fixed_ab_coordinate,
    select_retain_panels,
)
from src.diagnostics.t5_reconstructed_official import JsonPromptDataset, load_config, load_legacy_model, move_batch
from src.diagnostics.t5_step813_update_space_stage_b import _binary_summary
from src.diagnostics.t5_step817_forget_conflict_audit import tensor_tree_hash
from src.if_a2_optimization.group_a_gradient_audit import (
    masked_batch,
    masked_forward,
    stream_weight_gradients,
    validate_ratio_authority,
)
from src.paper_if_a2.common import atomic_json, canonical_hash, git_snapshot, hardware_snapshot, safe_run_name, sha256_file
from src.paper_if_a2.if_a2_method import _method_lineage, load_method_config

SCHEMA = "if-a2-group-b-scale-audit-v1"
MARKER = "IF_A2_GROUP_B_SCALE_AUDIT_V1_COMPLETED"
GIB = 1024 ** 3


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict): return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)): return all(_all_finite(item) for item in value)
    if isinstance(value, (float, np.floating)): return math.isfinite(float(value))
    return True


def _validate_config(config: dict[str, Any]) -> None:
    expected_top = {"schema", "development_only", "test_access_policy", "audit_target", "group_a_authority", "ratio_experiment", "method_config", "output_root", "input_masking", "runtime", "curvature", "cg", "projection", "arms", "trust_region", "scientific_scope"}
    if set(config) != expected_top or config["schema"] != SCHEMA or config["development_only"] is not True or config["test_access_policy"] != "forbidden": raise ValueError("invalid Group-B schema/policy")
    if config["audit_target"] != "ratio_i02s42v1_interaction_2pct_corrected_masking": raise ValueError("Group-B target changed")
    if config["group_a_authority"] != {"run_dir": "outputs/if_a2_optimization/group_a_gradient_audit_v2/if_a2_group_a_i02s42v1_masked_seed42_v1", "result_sha256": "a5780eee50529f3dfc068360ce232cc6e21555a70bca93621fcbc43c1591638a", "manifest_sha256": "55b27ac14ff09e6f5f7d8830713289fd8fb528b406d02a21f71b1f9cb497aa00", "required_scale": 0.02141568213543972}: raise ValueError("Group-A authority changed")
    if config["ratio_experiment"] != {"experiment_name": "i02s42v1", "requested_forget_ratio": .02, "actual_interaction_ratio": .020966666666666668, "forget_samples": 1258, "retain_samples": 58742, "forget_valid_tokens": 3774, "retain_valid_tokens": 176226}: raise ValueError("2% ratio authority changed")
    expected_mask = {"protocol": "explicit_encoder_attention_mask", "mask_definition": "input_ids_ne_pad_token_id", "applies_to": ["basis", "forget_gradient", "curvature", "projection_constraints", "candidate_response", "development_utility"], "legacy_unmasked_equivalence_claimed": False}
    if config["input_masking"] != expected_mask: raise ValueError("corrected masking contract changed")
    if config["runtime"] != {"forget_batch_size": 16, "curvature_batch_size": 8, "safety_batch_size": 8, "development_batch_size": 32, "allocator_fraction": .88, "minimum_free_gib_before_load": 8.0, "hard_cap_reserved_gib": 14.0}: raise ValueError("runtime contract changed")
    if config["curvature"] != {"panel_samples": 4096, "panel_seed": 42, "reference_strategy": "detached_same_forward", "power_iterations": 12, "convergence_tolerance": 1e-4, "numerical_lower_bound": 1e-14, "relative_damping_ratio": .01}: raise ValueError("curvature contract changed")
    if config["cg"] != {"relative_residual_tolerance": 1e-4, "absolute_residual_tolerance": 1e-10, "max_iterations": 40, "residual_explosion_factor": 1000.0, "pap_absolute_tolerance": 1e-14}: raise ValueError("CG contract changed")
    if config["projection"] != {"constraints": ["retain_supervised", "retain_self_kl"], "safety_panel_samples": 2048, "safety_panel_seed": 44, "relative_singular_tolerance": 1e-10, "normalized_constraint_tolerance": 1e-8, "safe_raw_norm_ratio_min": .10}: raise ValueError("projection contract changed")
    if config["arms"] != {"fixed_scale": .5, "ratio_scale": .02141568213543972, "trust_selection": "maximum_passing_preregistered_grid_scale", "diagnostic_scales": [.01, .02, .05, .1, .25, .5, 1.0]}: raise ValueError("scale registry changed")
    trust = config["trust_region"]
    if trust != {"retain_self_kl_max": .01, "effective_global_relative_norm_max": .01, "effective_module_relative_norm_max": .05, "forget_directional_derivative_must_be_positive": True, "utility": {"baseline": "corrected_masked_original", "overall_auc_damage_max": .005, "retain_user_auc_damage_max": .005, "overall_log_loss_damage_max": .01, "retain_user_log_loss_damage_max": .01, "prediction_collapse_forbidden": True}}: raise ValueError("trust-region thresholds changed")


def validate_group_a_authority(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    authority = config["group_a_authority"]; run = root / authority["run_dir"]
    required = {"COMPLETED", "group_a_gradient_audit.json", "manifest.json", "run_state.json"}
    if not run.is_dir() or {item.name for item in run.iterdir()} != required: raise ValueError("invalid Group-A authority inventory")
    if (run / "COMPLETED").read_text(encoding="utf-8") != "IF_A2_GROUP_A_GRADIENT_AUDIT_V2_COMPLETED\n": raise ValueError("invalid Group-A completion marker")
    if sha256_file(run / "manifest.json") != authority["manifest_sha256"] or sha256_file(run / "group_a_gradient_audit.json") != authority["result_sha256"]: raise ValueError("Group-A SHA mismatch")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8")); result = json.loads((run / "group_a_gradient_audit.json").read_text(encoding="utf-8")); state = json.loads((run / "run_state.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETED" or manifest.get("result_sha256") != authority["result_sha256"] or manifest.get("test_accessed") is not False: raise ValueError("invalid Group-A manifest")
    if state.get("status") != "COMPLETED" or state.get("failed_gates") != [] or result.get("passed") is not True or not all(result.get("gates", {}).values()): raise ValueError("Group-A gates did not pass")
    if result.get("runtime_protocol") != "corrected_masking_not_legacy_exact" or result.get("masking_evidence", {}).get("definition") != "input_ids_ne_pad_token_id": raise ValueError("Group-A corrected-mask evidence missing")
    scale = result.get("unit_ledger", {}).get("group_B_required_derived_scale")
    if type(scale) is not float or scale != authority["required_scale"]: raise ValueError("Group-A derived scale mismatch")
    return {"run_dir": str(run), "result_sha256": authority["result_sha256"], "manifest_sha256": authority["manifest_sha256"], "derived_scale": scale, "passed": True, "test_accessed": False}


def make_masked_curvature_operator(model: torch.nn.Module, dataset: JsonPromptDataset, indices: Sequence[int], parameters: Sequence[torch.Tensor], device: torch.device, batch_size: int, pad_token_id: int, counter: dict[str, int]) -> Callable[[torch.Tensor], torch.Tensor]:
    def operator(vector: torch.Tensor) -> torch.Tensor:
        total = torch.zeros_like(vector, dtype=torch.float64); tokens_total = 0
        for start in range(0, len(indices), batch_size):
            batch = move_batch(masked_batch(dataset, indices[start:start + batch_size], pad_token_id), device); mask = batch["target_ids"].ne(-100); tokens = int(mask.sum())
            current = masked_forward(model, batch).logits; reference = current.detach(); loss = self_kl_loss(reference, current, mask)
            total += tokens * hessian_vector_product(loss, parameters, vector); tokens_total += tokens; counter["hvp_batches"] += 1
            del batch, mask, current, reference, loss
        if tokens_total <= 0: raise RuntimeError("empty curvature panel")
        result = total / tokens_total; curvature = float(torch.dot(vector, result)); tolerance = 1e-10 + 1e-8 * float(torch.linalg.vector_norm(vector)) * float(torch.linalg.vector_norm(result))
        if curvature < -tolerance: raise RuntimeError("significant_negative_curvature")
        counter["operator_calls"] = counter.get("operator_calls", 0) + 1
        print(f"[group-b:hvp] operator_call={counter['operator_calls']} cumulative_batches={counter['hvp_batches']}", flush=True)
        return result
    return operator


def masked_self_kl_gradient(model: torch.nn.Module, dataset: JsonPromptDataset, indices: Sequence[int], parameters: Sequence[torch.Tensor], device: torch.device, batch_size: int, pad_token_id: int) -> tuple[torch.Tensor, dict[str, Any]]:
    total = torch.zeros(sum(parameter.numel() for parameter in parameters), dtype=torch.float64); tokens_total = 0; calls = 0
    for start in range(0, len(indices), batch_size):
        batch = move_batch(masked_batch(dataset, indices[start:start + batch_size], pad_token_id), device); mask = batch["target_ids"].ne(-100); tokens = int(mask.sum()); current = masked_forward(model, batch).logits; loss = self_kl_loss(current.detach(), current, mask); gradients = torch.autograd.grad(loss, parameters)
        total += tokens * flatten_tensors(gradients); tokens_total += tokens; calls += 1; del batch, mask, current, loss, gradients
    value = total / tokens_total
    return value, {"valid_tokens": tokens_total, "forward_batches": calls, "autograd_grad_calls": calls, "norm": float(torch.linalg.vector_norm(value)), "sha256": tensor_tree_hash({"retain_self_kl_gradient": value}), "encoder_attention_mask": "input_ids_ne_pad_token_id"}


def masked_mean_loss(model: torch.nn.Module, dataset: JsonPromptDataset, indices: Sequence[int], device: torch.device, batch_size: int, pad_token_id: int) -> dict[str, Any]:
    numerator = 0.; tokens_total = 0; calls = 0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = move_batch(masked_batch(dataset, indices[start:start + batch_size], pad_token_id), device); output = masked_forward(model, batch); tokens = int(batch["target_ids"].ne(-100).sum()); numerator += float(output.loss.detach().cpu()) * tokens; tokens_total += tokens; calls += 1; del batch, output
    return {"loss": numerator / tokens_total, "valid_tokens": tokens_total, "forward_batches": calls, "encoder_attention_mask": "input_ids_ne_pad_token_id"}


def masked_predictions(model: torch.nn.Module, dataset: JsonPromptDataset, device: torch.device, batch_size: int, pad_token_id: int) -> dict[str, Any]:
    probabilities: list[float] = []; gold: list[int] = []; order: list[int] = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            indices = list(range(start, min(start + batch_size, len(dataset)))); batch = move_batch(masked_batch(dataset, indices, pad_token_id), device); output = masked_forward(model, batch); pair = torch.softmax(output.logits[:, 0, [465, 2163]], dim=-1)[:, 1]
            probabilities.extend(pair.detach().cpu().tolist()); gold.extend(batch["target_ids"][:, 0].eq(2163).long().cpu().tolist()); order.extend(indices); del batch, output, pair
    return {"probabilities": probabilities, "gold": gold, "sample_order_sha256": canonical_hash(order), "samples": len(order), "encoder_attention_mask": "input_ids_ne_pad_token_id", "test_accessed": False}


def utility_summary(predictions: dict[str, Any], retain_indices: Sequence[int]) -> dict[str, Any]:
    all_indices = list(range(predictions["samples"])); return {"overall_validation": _binary_summary(predictions["probabilities"], predictions["gold"], all_indices), "retain_user_validation": _binary_summary(predictions["probabilities"], predictions["gold"], list(retain_indices)), "sample_order_sha256": predictions["sample_order_sha256"], "encoder_attention_mask": predictions["encoder_attention_mask"], "test_accessed": False}


def utility_evidence(baseline: dict[str, Any], candidate: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    thresholds = config["trust_region"]["utility"]
    damage = {"overall_auc": baseline["overall_validation"]["auc"] - candidate["overall_validation"]["auc"], "retain_user_auc": baseline["retain_user_validation"]["auc"] - candidate["retain_user_validation"]["auc"], "overall_log_loss": candidate["overall_validation"]["log_loss"] - baseline["overall_validation"]["log_loss"], "retain_user_log_loss": candidate["retain_user_validation"]["log_loss"] - baseline["retain_user_validation"]["log_loss"]}
    checks = {"finite": _all_finite(candidate) and _all_finite(damage), "overall_auc": damage["overall_auc"] <= thresholds["overall_auc_damage_max"], "retain_user_auc": damage["retain_user_auc"] <= thresholds["retain_user_auc_damage_max"], "overall_log_loss": damage["overall_log_loss"] <= thresholds["overall_log_loss_damage_max"], "retain_user_log_loss": damage["retain_user_log_loss"] <= thresholds["retain_user_log_loss_damage_max"], "probability_not_collapsed": candidate["overall_validation"]["probability_std"] > 1e-6 and candidate["retain_user_validation"]["probability_std"] > 1e-6}
    return {"baseline": "corrected_masked_original", "damage": damage, "checks": checks, "passed": bool(all(checks.values()))}


def quantized_delta(base: torch.Tensor, direction: torch.Tensor, scale: float) -> torch.Tensor:
    return (base + float(scale) * direction).to(torch.float32).to(torch.float64) - base.to(torch.float32).to(torch.float64)


def effective_update_metrics(actual: torch.Tensor, parameters: Sequence[torch.Tensor], names: Sequence[str], bases: dict[str, torch.Tensor], modules: Sequence[tuple[str, torch.nn.Linear]], alpha: float, rank: int) -> dict[str, Any]:
    pieces = split_vector(actual, parameters); rows = []; delta_sq = 0.; base_sq = 0.
    for name, piece, (module_name, module) in zip(names, pieces, modules):
        if name != module_name: raise RuntimeError("effective-update module order mismatch")
        delta = (float(alpha) / rank) * piece.detach().to(torch.float64).cpu() @ bases[name].to(torch.float64); base = module.weight.detach().to(torch.float64).cpu(); delta_norm = float(torch.linalg.vector_norm(delta)); base_norm = float(torch.linalg.vector_norm(base)); ratio = delta_norm / max(base_norm, 1e-300); rows.append({"module": name, "delta_norm": delta_norm, "base_weight_norm": base_norm, "relative_norm": ratio}); delta_sq += delta_norm ** 2; base_sq += base_norm ** 2
    return {"global_relative_norm": math.sqrt(delta_sq) / max(math.sqrt(base_sq), 1e-300), "maximum_module_relative_norm": max(row["relative_norm"] for row in rows), "modules": rows}


def candidate_gate(row: dict[str, Any], config: dict[str, Any], projection_passed: bool) -> dict[str, bool]:
    trust = config["trust_region"]
    checks = {"finite": _all_finite(row), "projection": bool(projection_passed), "positive_forget_direction": row["predicted_forget_loss_change"] > 0, "retain_self_kl": row["retain_safety"]["self_kl"] <= trust["retain_self_kl_max"], "effective_global_norm": row["effective_update"]["global_relative_norm"] <= trust["effective_global_relative_norm_max"], "effective_module_norm": row["effective_update"]["maximum_module_relative_norm"] <= trust["effective_module_relative_norm_max"], "utility": row["utility"]["passed"] is True}
    return checks


def select_trust_scale(rows: Sequence[dict[str, Any]], diagnostic_scales: Sequence[float]) -> dict[str, Any]:
    allowed = {float(value) for value in diagnostic_scales}; eligible = [row for row in rows if float(row["scale"]) in allowed and row["trust_region_passed"] is True]; selected = max(eligible, key=lambda row: row["scale"]) if eligible else None
    return {"selection_rule": "maximum_passing_preregistered_grid_scale", "selected_scale": None if selected is None else selected["scale"], "selected_candidate_sha256": None if selected is None else selected["candidate_sha256"], "eligible_scales": [row["scale"] for row in eligible], "continuous_maximum_claimed": False, "posthoc_forget_selection_used": False}


def _candidate_scales(config: dict[str, Any]) -> list[float]:
    values = list(config["arms"]["diagnostic_scales"]) + [config["arms"]["ratio_scale"]]
    return sorted(set(float(value) for value in values))


def _publish(stage: Path, destination: Path, report: dict[str, Any], implementation_sha: str) -> None:
    if not _all_finite(report): raise ValueError("Group-B report contains NaN/Inf")
    atomic_json(stage / "group_b_scale_audit.json", report); atomic_json(stage / "run_state.json", {"schema": SCHEMA, "status": "COMPLETED", "optimizer_constructed": False, "optimizer_steps_committed": 0, "candidate_model_published": False, "test_accessed": False}); atomic_json(stage / "manifest.json", {"schema": SCHEMA, "status": "COMPLETED", "result_sha256": sha256_file(stage / "group_b_scale_audit.json"), "run_state_sha256": sha256_file(stage / "run_state.json"), "implementation_sha256": implementation_sha, "published_atomically": True, "optimizer_constructed": False, "optimizer_steps_committed": 0, "candidate_model_published": False, "test_accessed": False}); (stage / "COMPLETED").write_text(MARKER + "\n", encoding="utf-8", newline="\n"); destination.parent.mkdir(parents=True, exist_ok=True); os.replace(stage, destination)


def analyze(root: Path, config_path: Path, run_name: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")); _validate_config(config); run = root / config["output_root"] / safe_run_name(run_name); required = {"COMPLETED", "group_b_scale_audit.json", "manifest.json", "run_state.json"}
    if not run.is_dir() or {item.name for item in run.iterdir()} != required or (run / "COMPLETED").read_text(encoding="utf-8") != MARKER + "\n": raise ValueError("invalid Group-B run")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8")); report = json.loads((run / "group_b_scale_audit.json").read_text(encoding="utf-8"))
    if manifest.get("result_sha256") != sha256_file(run / "group_b_scale_audit.json") or manifest.get("test_accessed") is not False or report.get("test_accessed") is not False: raise ValueError("Group-B manifest mismatch")
    return {"status": "COMPLETED", "run_dir": str(run), "arms": report["arms"], "trust_selection": report["trust_selection"], "test_accessed": False}


def preflight(root: Path, config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")); _validate_config(config); group_a = validate_group_a_authority(root, config); method_path = root / config["method_config"]; method = load_method_config(method_path); base = load_config(root / method["base_config"], root); ratio_config = {"ratio_experiment": {"experiment_name": "i02s42v1", "manifest": "outputs/ru1/i02s42v1/experiment_manifest.json", "completion_marker": "outputs/ru1/i02s42v1/COMPLETED", "contract_sha256": method["deletion_experiment"]["contract_sha256"], "requested_forget_ratio": .02, "actual_interaction_ratio": .020966666666666668, "forget_samples": 1258, "retain_samples": 58742, "forget_users": 8}}
    ratio = validate_ratio_authority(root, ratio_config, method_path, method)
    return {"schema": SCHEMA, "mode": "Preflight", "group_a": group_a, "ratio_authority": ratio, "original_sha256": sha256_file(root / method["original"]), "scales": _candidate_scales(config), "trust_thresholds": config["trust_region"], "model_loaded": False, "optimizer_constructed": False, "test_accessed": False, "base_protocol": base["protocol_name"]}


def run(root: Path, config_path: Path, run_name: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")); _validate_config(config); run_name = safe_run_name(run_name); destination = (root / config["output_root"] / run_name).resolve()
    if destination.exists(): raise FileExistsError(destination)
    group_a = validate_group_a_authority(root, config); git = git_snapshot(root)
    if not git["clean"]: raise RuntimeError("formal Group-B audit requires clean Git")
    method_path = root / config["method_config"]; method = load_method_config(method_path); base = load_config(root / method["base_config"], root); checkpoint = root / method["original"]
    for key in ("original", "forget", "retain", "development"):
        path = checkpoint if key == "original" else root / method[key]
        if sha256_file(path) != method[f"{key}_sha256"]: raise ValueError(f"{key} authority SHA mismatch")
    ratio_config = {"ratio_experiment": {"experiment_name": "i02s42v1", "manifest": "outputs/ru1/i02s42v1/experiment_manifest.json", "completion_marker": "outputs/ru1/i02s42v1/COMPLETED", "contract_sha256": method["deletion_experiment"]["contract_sha256"], "requested_forget_ratio": .02, "actual_interaction_ratio": .020966666666666668, "forget_samples": 1258, "retain_samples": 58742, "forget_users": 8}}
    ratio = validate_ratio_authority(root, ratio_config, method_path, method)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1: raise RuntimeError("Group B requires exactly one CUDA GPU")
    device = torch.device("cuda:0"); free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes / GIB < config["runtime"]["minimum_free_gib_before_load"]: raise RuntimeError("insufficient clean-GPU free VRAM")
    work = destination.parent / ".work"; work.mkdir(parents=True, exist_ok=True); stage = work / f"{run_name}.{uuid.uuid4().hex}.stage"; stage.mkdir(); model = None; started = time.perf_counter(); timing: dict[str, float] = {}; checkpoint_before = sha256_file(checkpoint)
    try:
        print("[group-b:start] loading corrected-masking Original", flush=True); torch.cuda.set_per_process_memory_fraction(config["runtime"]["allocator_fraction"], device); torch.cuda.reset_peak_memory_stats(); model = load_legacy_model(checkpoint).to(device).eval(); tokenizer = T5Tokenizer.from_pretrained(base["paths"]["model_dir"]); pad = tokenizer.pad_token_id
        if pad is None: raise RuntimeError("corrected masking requires pad_token_id")
        forget = JsonPromptDataset(root / method["forget"], tokenizer); retain = JsonPromptDataset(root / method["retain"], tokenizer); development = JsonPromptDataset(root / method["development"], tokenizer); lineage, indices, users = _method_lineage(root, base, method)
        if len(forget) != 1258 or len(retain) != 58742 or len(development) != 20000 or len(indices["retain_user_validation"]) != 19827: raise RuntimeError("2% data lineage/count mismatch")
        a2 = yaml.safe_load((root / method["if_a2_config"]).read_text(encoding="utf-8")); panels = select_retain_panels(users["retain_train"], a2["panels"]); primary = panels["primary"]["indices"]; safety = panels["safety"]["indices"]
        modules = collect_qv_modules(model); module_names = [name for name, _ in modules]; weights = [module.weight for _, module in modules]; official = list(model.parameters()); buffers = list(model.buffers()); official_before = tensor_tree_hash({str(i): value.detach() for i, value in enumerate(official)}); buffers_before = tensor_tree_hash({str(i): value.detach() for i, value in enumerate(buffers)}); rng_before = tensor_tree_hash({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()})
        for parameter in model.parameters(): parameter.requires_grad_(False)
        for weight in weights: weight.requires_grad_(True)
        print("[group-b:basis] full Forget gradient and fixed-A basis", flush=True); tick = time.perf_counter(); matrices, basis_gradient = stream_weight_gradients(model, forget, indices["forget_train"], weights, device, config["runtime"]["forget_batch_size"], pad); timing["basis_gradient_seconds"] = time.perf_counter() - tick
        bases = {}; basis_reports = []
        for name, matrix in zip(module_names, matrices):
            bases[name], basis_report = build_fixed_a_basis(matrix, rank=16, name=name, seed=42); basis_reports.append(basis_report)
        basis_sha = canonical_hash([(row["name"], row["basis_sha256"]) for row in basis_reports]); sample = move_batch(masked_batch(forget, [0], pad), device)
        with torch.no_grad(): before_coordinate = masked_forward(model, sample).logits.detach().cpu()
        for weight in weights: weight.requires_grad_(False)
        names, parameters = install_fixed_ab_coordinate(model, bases, 32); names = [name[:-2] if name.endswith(".B") else name for name in names]
        with torch.no_grad(): after_coordinate = masked_forward(model, sample).logits.detach().cpu()
        function_equivalence_exact = torch.equal(before_coordinate, after_coordinate)
        if names != module_names or not function_equivalence_exact or any(torch.count_nonzero(parameter).item() for parameter in parameters): raise RuntimeError("fixed-A/B zero-coordinate equivalence failed")
        del sample, before_coordinate, after_coordinate
        tick = time.perf_counter(); direct, forget_gradient = stream_weight_gradients(model, forget, indices["forget_train"], parameters, device, 16, pad); g_f = flatten_tensors(direct); timing["forget_B_gradient_seconds"] = time.perf_counter() - tick
        analytic = flatten_tensors([analytic_b_gradient(matrix, bases[name], 32, 16) for name, matrix in zip(module_names, matrices)]); analytic_error = float(torch.linalg.vector_norm(g_f - analytic) / torch.linalg.vector_norm(analytic))
        if analytic_error > 2e-5: raise RuntimeError("analytic B-gradient mismatch")
        print("[group-b:curvature] 4096-sample Retain panel, 12 power iterations", flush=True); counter = {"hvp_batches": 0}; operator = make_masked_curvature_operator(model, retain, primary, parameters, device, 8, pad, counter)
        tick = time.perf_counter(); estimate = estimate_lambda_max(operator, g_f.numel(), seed=42, iterations=12, convergence_tolerance=1e-4, numerical_lower_bound=1e-14); timing["lambda_max_seconds"] = time.perf_counter() - tick; damping = .01 * estimate["lambda_max_hat"]
        print(f"[group-b:cg] lambda_max={estimate['lambda_max_hat']:.8g} damping={damping:.8g}", flush=True); tick = time.perf_counter(); cg = conjugate_gradient(operator, g_f, damping=damping, relative_tolerance=1e-4, absolute_tolerance=1e-10, max_iterations=40, residual_explosion_factor=1000., pap_tolerance=1e-14); raw = cg.pop("solution"); timing["cg_seconds"] = time.perf_counter() - tick
        print("[group-b:projection] 2048-sample Retain safety constraints", flush=True); tick = time.perf_counter(); sup_parts, sup_meta = stream_weight_gradients(model, retain, safety, parameters, device, 8, pad); g_sup = flatten_tensors(sup_parts); g_kl, kl_meta = masked_self_kl_gradient(model, retain, safety, parameters, device, 8, pad); base_flat = flatten_tensors([parameter.detach() for parameter in parameters]); projection = project_update_space(raw, [g_sup, g_kl], relative_tolerance=1e-10, normalized_tolerance=1e-8, formal_dtype=torch.float32, base=base_flat); direction = projection.pop("actual"); timing["projection_seconds"] = time.perf_counter() - tick
        projection["safe_raw_norm_ratio"] = float(torch.linalg.vector_norm(direction) / torch.linalg.vector_norm(raw)); projection["forget_first_order_retained_ratio"] = float(torch.dot(g_f, direction) / max(float(torch.dot(g_f, raw)), 1e-300)); projection["absolute_constraint_gate"] = all(abs(row["normalized_dot"]) <= 1e-8 for row in projection["constraint_dots"]); projection_pass = projection["passed"] and projection["absolute_constraint_gate"] and projection["safe_raw_norm_ratio"] >= .10 and float(torch.dot(g_f, direction)) > 0
        tick = time.perf_counter(); original_forget = masked_mean_loss(model, forget, indices["forget_train"], device, 16, pad); original_safety = masked_mean_loss(model, retain, safety, device, 8, pad); original_predictions = masked_predictions(model, development, device, 32, pad); original_utility = utility_summary(original_predictions, indices["retain_user_validation"]); timing["original_evaluation_seconds"] = time.perf_counter() - tick
        rows = []
        for scale in _candidate_scales(config):
            print(f"[group-b:candidate] scale={scale:.17g}", flush=True)
            actual = quantized_delta(base_flat, direction, scale); predicted = float(torch.dot(g_f, actual)); candidate_sha = tensor_tree_hash({"candidate": actual})
            tick = time.perf_counter()
            with _temporary_delta(list(parameters), actual):
                forget_metrics = masked_mean_loss(model, forget, indices["forget_train"], device, 16, pad); safety_supervised = masked_mean_loss(model, retain, safety, device, 8, pad); candidate_predictions = masked_predictions(model, development, device, 32, pad); candidate_utility = utility_summary(candidate_predictions, indices["retain_user_validation"])
                kl_numerator = 0.; kl_tokens = 0
                with torch.no_grad():
                    for start in range(0, len(safety), 8):
                        selected = safety[start:start + 8]; batch = move_batch(masked_batch(retain, selected, pad), device); current = masked_forward(model, batch).logits
                        with _temporary_delta(list(parameters), -actual): reference = masked_forward(model, batch).logits.detach()
                        mask = batch["target_ids"].ne(-100); tokens = int(mask.sum()); kl_numerator += float(self_kl_loss(reference, current, mask).detach().cpu()) * tokens; kl_tokens += tokens; del batch, current, reference, mask
            actual_change = forget_metrics["loss"] - original_forget["loss"]; update = effective_update_metrics(actual, parameters, names, bases, modules, 32, 16); utility = utility_evidence(original_utility, candidate_utility, config)
            row = {"scale": scale, "candidate_sha256": candidate_sha, "arms": [name for name, value in (("B-Fixed", .5), ("B-Ratio", config["arms"]["ratio_scale"])) if scale == value], "predicted_forget_loss_change": predicted, "actual_forget_loss_change": actual_change, "predicted_actual_ratio": actual_change / predicted if predicted != 0 else None, "forget": forget_metrics, "retain_safety": {"self_kl": kl_numerator / kl_tokens, "supervised_loss": safety_supervised["loss"], "supervised_loss_change": safety_supervised["loss"] - original_safety["loss"], "valid_tokens": kl_tokens}, "effective_update": update, "utility_metrics": candidate_utility, "utility": utility, "evaluation_seconds": time.perf_counter() - tick}
            row["trust_checks"] = candidate_gate(row, config, projection_pass); row["trust_region_passed"] = bool(all(row["trust_checks"].values())); rows.append(row); del actual, candidate_predictions
        trust = select_trust_scale(rows, config["arms"]["diagnostic_scales"]); by_scale = {row["scale"]: row for row in rows}; arms = {"B-Fixed": {"scale": .5, "candidate_sha256": by_scale[.5]["candidate_sha256"], "trust_region_passed": by_scale[.5]["trust_region_passed"]}, "B-Ratio": {"scale": config["arms"]["ratio_scale"], "candidate_sha256": by_scale[config["arms"]["ratio_scale"]]["candidate_sha256"], "trust_region_passed": by_scale[config["arms"]["ratio_scale"]]["trust_region_passed"]}, "B-TrustRegion": {"scale": trust["selected_scale"], "candidate_sha256": trust["selected_candidate_sha256"], "trust_region_passed": trust["selected_scale"] is not None}}
        peak_allocated = torch.cuda.max_memory_allocated() / GIB; peak_reserved = torch.cuda.max_memory_reserved() / GIB; checkpoint_after = sha256_file(checkpoint); integrity = {"checkpoint_unchanged": checkpoint_after == checkpoint_before, "official_parameters_unchanged": tensor_tree_hash({str(i): value.detach() for i, value in enumerate(official)}) == official_before, "buffers_unchanged": tensor_tree_hash({str(i): value.detach() for i, value in enumerate(buffers)}) == buffers_before, "B_restored_zero": all(torch.count_nonzero(parameter).item() == 0 for parameter in parameters), "parameter_grad_absent": all(parameter.grad is None for parameter in official + list(parameters)), "rng_unchanged": tensor_tree_hash({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}) == rng_before, "model_eval": model.training is False}
        if not all(integrity.values()): raise RuntimeError("Group-B transactional restoration failed")
        implementation_files = [Path(__file__), Path(__file__).with_name("group_a_gradient_audit.py"), config_path, method_path]; implementation = {str(path.resolve().relative_to(root.resolve())).replace("\\", "/"): sha256_file(path) for path in implementation_files}; implementation_sha = canonical_hash(implementation)
        registry_sha = canonical_hash([{key: value for key, value in row.items() if key != "evaluation_seconds"} for row in rows])
        report = {"schema": SCHEMA, "run_name": run_name, "audit_target": config["audit_target"], "runtime_protocol": "2pct_corrected_masking_zero_commit_scale_audit", "config_sha256": sha256_file(config_path), "method_config_sha256": sha256_file(method_path), "group_a_authority": group_a, "ratio_authority": ratio, "scale_ledger": {"fixed": .5, "ratio": config["arms"]["ratio_scale"], "fixed_over_ratio": .5 / config["arms"]["ratio_scale"], "ratio_source": "Group-A valid-token empirical-risk ledger"}, "lineage": lineage, "input_masking": config["input_masking"], "coordinate": {"kind": "function_preserving_data_adaptive_fixed_A_B_space", "module_order": names, "basis_reports": basis_reports, "basis_sha256": basis_sha, "function_equivalence_exact": function_equivalence_exact, "A_frozen": True, "B_only": True, "alpha": 32, "rank": 16}, "curvature": {"panel": {key: value for key, value in panels["primary"].items() if key != "indices"}, "lambda_max": estimate, "relative_damping_ratio": .01, "damping": damping, "operator_calls": counter.get("operator_calls", 0), "hvp_batches": counter["hvp_batches"], "encoder_attention_mask": "input_ids_ne_pad_token_id"}, "cg": cg, "forget_gradient": {**forget_gradient, "basis_gradient": basis_gradient, "norm": float(torch.linalg.vector_norm(g_f)), "sha256": tensor_tree_hash({"g_F": g_f}), "analytic_relative_error": analytic_error}, "projection_constraints": {"retain_supervised": {**sup_meta, "norm": float(torch.linalg.vector_norm(g_sup)), "sha256": tensor_tree_hash({"g_sup": g_sup})}, "retain_self_kl": kl_meta}, "projection": projection, "projection_passed": projection_pass, "original": {"forget": original_forget, "retain_safety": original_safety, "utility": original_utility}, "candidate_registry": rows, "candidate_registry_sha256": registry_sha, "candidate_registry_hash_excludes": ["evaluation_seconds"], "arms": arms, "trust_selection": trust, "trust_thresholds": config["trust_region"], "scientific_scope": config["scientific_scope"], "timing": {**timing, "candidate_evaluation_seconds": sum(row["evaluation_seconds"] for row in rows), "wall_time_seconds": time.perf_counter() - started}, "memory": {"device_total_gib": total_bytes / GIB, "free_before_load_gib": free_bytes / GIB, "peak_allocated_gib": peak_allocated, "peak_reserved_gib": peak_reserved, "hard_cap_reserved_gib": 14.0}, "integrity": integrity, "checkpoint_sha256_before": checkpoint_before, "checkpoint_sha256_after": checkpoint_after, "git": git, "hardware": hardware_snapshot(), "implementation_files": implementation, "implementation_sha256": implementation_sha, "optimizer_constructed": False, "optimizer_steps_committed": 0, "candidate_model_published": False, "retrain_loaded": False, "test_loader_built": False, "test_accessed": False}
        if peak_reserved > 14.0 or not _all_finite(report): raise RuntimeError("Group-B memory/nonfinite gate failed")
        _publish(stage, destination, report, implementation_sha); return {"status": "COMPLETED", "run_dir": str(destination), "arms": arms, "trust_selection": trust, "optimizer_steps_committed": 0, "test_accessed": False}
    finally:
        if model is not None: del model
        if stage.exists():
            import shutil
            shutil.rmtree(stage)
        gc.collect(); torch.cuda.empty_cache()


def synthetic() -> dict[str, Any]:
    rows = [{"scale": .01, "candidate_sha256": "a", "trust_region_passed": True}, {"scale": .5, "candidate_sha256": "b", "trust_region_passed": False}, {"scale": .25, "candidate_sha256": "c", "trust_region_passed": True}]
    return {"schema": SCHEMA, "trust_selection": select_trust_scale(rows, [.01, .25, .5]), "quantized": quantized_delta(torch.zeros(2), torch.tensor([1., -1.], dtype=torch.float64), .5).tolist(), "optimizer_constructed": False, "test_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path.cwd()); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--run-name"); parser.add_argument("--mode", choices=["Preflight", "SyntheticDryRun", "Full", "Analyze"], default="Preflight"); args = parser.parse_args(); root = args.root.resolve(); config = args.config.resolve()
    if args.mode == "Preflight": value = preflight(root, config)
    elif args.mode == "SyntheticDryRun": value = synthetic()
    elif not args.run_name: raise ValueError(f"{args.mode} requires --run-name")
    elif args.mode == "Analyze": value = analyze(root, config, args.run_name)
    else: value = run(root, config, args.run_name)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__": main()
