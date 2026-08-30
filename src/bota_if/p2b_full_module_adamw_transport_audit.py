"""P2-B full-module AdamW trajectory-transport audit for BOTA-IF."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from transformers import T5Tokenizer

from src.bota_if import p1_trajectory_transport_audit as p1
from src.diagnostics.t5_reconstructed_official import JsonPromptDataset, load_config, move_batch
from src.if_a2_optimization.group_a_gradient_audit import GIB, masked_batch
from src.paper_baselines.common import capture_rng, restore_rng, tensor_tree_hash
from src.paper_if_a2.common import atomic_json, canonical_hash, git_snapshot, hardware_snapshot, safe_run_name, sha256_file

SCHEMA = "bota-if-p2b-full-module-adamw-transport-v1"
MARKER = "BOTA_IF_P2B_FULL_MODULE_ADAMW_TRANSPORT_V1_COMPLETED"
REPORT = "p2b_full_module_adamw_transport.json"
VARIANTS = p1.VARIANTS


def load_frozen_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"schema", "test_access_policy", "base_config", "train_data", "raw_data", "original_checkpoint", "output_root", "authority", "coordinate", "schedule", "optimizer", "transport", "quantization", "gates", "runtime", "privacy", "scientific_scope"}
    if not isinstance(value, dict) or set(value) != required or value["schema"] != SCHEMA or value["test_access_policy"] != "forbidden":
        raise ValueError("invalid P2-B configuration")
    if value["coordinate"] != {"target_modules": ["q", "v"], "module_count": 72, "lora_rank": 16, "lora_alpha": 32, "trainable_coordinate": "B_only", "initial_B": "zero", "fixed_a_seed": 42, "transport_coordinate": "full_qv_lora_b", "transport_dimension": 884736, "shared_low_rank_coordinate_used": False, "module_truncation_used": False}:
        raise ValueError("P2-B coordinate registry changed")
    if value["schedule"] != {"seed": 42, "steps": 50, "batch_size": 16, "users": ["high_frequency", "median_frequency", "low_frequency"], "exact_predecessor_train_order_required": True, "exact_predecessor_user_roles_required": True}:
        raise ValueError("P2-B schedule changed")
    if value["optimizer"] != {"name": "AdamW", "learning_rate": .001, "betas": [.9, .999], "eps": 1e-8, "weight_decay": .01, "scheduler": "none", "gradient_clipping": "none"}:
        raise ValueError("P2-B optimizer changed")
    if value["transport"] != {"curvature": "per_sample_block_diagonal_empirical_fisher", "variants": list(VARIANTS), "primary": "T2_AdamW_full_state", "state_dtype": "float32_formal_parameter_dtype", "full_v_linearization": True}:
        raise ValueError("P2-B transport changed")
    if value["quantization"] != {"formats": ["float16", "bfloat16"], "source": "full_precision_authority_vector", "dequantize_to": "float32", "fp16_relative_l2_maximum": .001, "bf16_relative_l2_maximum": .01, "cosine_minimum": .9999, "quantized_vectors_persisted": False}:
        raise ValueError("P2-B quantization registry changed")
    if value["gates"] != {"cosine_minimum": .8, "norm_ratio_minimum": .5, "norm_ratio_maximum": 2., "relative_l2_maximum": .75, "positive_module_fraction_minimum": .9, "t2_must_beat_t0": True, "t2_must_not_be_worse_than_t1": True}:
        raise ValueError("P2-B gates changed")
    if value["runtime"]["physical_optimizer_step_limit"] != 200:
        raise ValueError("P2-B step budget changed")
    if value["scientific_scope"] != {"full_module_transport_only": True, "masked_reference": True, "masked_slots_preserved": True, "masked_batch_denominator_preserved": True, "zero_authoritative_update": True, "compacted_retrain_reference": False, "certified_unlearning_claimed": False, "historical_equivalence_claimed": False, "development_access": False, "retrain_access": False, "test_access": False}:
        raise ValueError("P2-B scientific scope changed")
    return value


def validate_predecessor(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    authority = config["authority"]; run = root / authority["predecessor_run"]
    required = {"COMPLETED", "manifest.json", "p2a_user_sparse_module_oracle.json", "run_state.json"}
    if not run.is_dir() or {path.name for path in run.iterdir()} != required:
        raise ValueError("P2-A predecessor layout mismatch")
    checks = {"report": sha256_file(run / "p2a_user_sparse_module_oracle.json"), "manifest": sha256_file(run / "manifest.json"), "run_state": sha256_file(run / "run_state.json")}
    expected = {"report": authority["predecessor_report_sha256"], "manifest": authority["predecessor_manifest_sha256"], "run_state": authority["predecessor_run_state_sha256"]}
    if checks != expected:
        raise ValueError("P2-A predecessor SHA mismatch")
    report = json.loads((run / "p2a_user_sparse_module_oracle.json").read_text(encoding="utf-8")); state = json.loads((run / "run_state.json").read_text(encoding="utf-8"))
    if report.get("classification") != "user_specific_module_sparsity_insufficient" or report.get("execution", {}).get("physical_optimizer_step_calls") != 200 or state.get("status") != "COMPLETED" or report.get("test_accessed") is not False:
        raise ValueError("P2-A predecessor semantics mismatch")
    if sha256_file(root / config["train_data"]) != authority["train_sha256"] or sha256_file(root / config["original_checkpoint"]) != authority["original_sha256"]:
        raise ValueError("P2-B source authority changed")
    return {"run": str(run), "sha256": checks, "classification": report["classification"], "schedule": report["schedule"], "test_accessed": False}


def new_full_states(parameters: Sequence[torch.Tensor], users: int) -> dict[str, list[dict[str, torch.Tensor]]]:
    states = {}
    for variant in VARIANTS:
        states[variant] = [{key: torch.zeros((users, *parameter.shape), device=parameter.device, dtype=parameter.dtype) for key in ("theta", "m", "v")} for parameter in parameters]
    return states


def full_fisher_product(coefficients: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    samples = coefficients.shape[0]; flat = vectors.reshape(vectors.shape[0], -1); rows = coefficients.reshape(samples, -1)
    return ((rows @ flat.T).T @ rows / samples).reshape_as(vectors)


def stable_adamw_tangent_step(theta, gradient, m, v, dtheta, dm, dv, dgradient, *, step, lr, beta1, beta2, eps, weight_decay, full_v):
    """AdamW tangent with an exact zero-moment branch in formal parameter dtype."""
    del theta  # The decoupled-weight-decay tangent depends on dtheta, not theta.
    m_new = beta1 * m + (1 - beta1) * gradient; v_new = beta2 * v + (1 - beta2) * gradient.square(); dm_new = beta1 * dm + (1 - beta1) * dgradient
    dv_new = beta2 * dv + 2 * (1 - beta2) * gradient * dgradient if full_v else torch.zeros_like(dv)
    bc1, bc2 = 1 - beta1**step, 1 - beta2**step; mhat, vhat = m_new / bc1, v_new / bc2; root = torch.sqrt(vhat); denominator = root + eps; d_mhat = dm_new / bc1
    d_update = d_mhat / denominator
    if full_v:
        d_vhat = dv_new / bc2; numerator = mhat * d_vhat; root_full = root.expand_as(numerator); denominator_full = denominator.expand_as(numerator); zero_root = root_full == 0; singular = zero_root & (numerator != 0)
        if bool(singular.any()):
            raise RuntimeError("singular_adamw_tangent_at_zero_second_moment")
        active = (~zero_root) & (numerator != 0); correction = torch.zeros_like(d_update)
        correction[active] = numerator[active] / (2 * root_full[active] * denominator_full[active].square())
        if not bool(torch.isfinite(correction).all()):
            raise RuntimeError("nonfinite_adamw_second_moment_correction")
        d_update = d_update - correction
    next_theta = (1 - lr * weight_decay) * dtheta - lr * d_update
    if not all(bool(torch.isfinite(value).all()) for value in (next_theta, dm_new, dv_new)):
        raise RuntimeError("nonfinite_stable_adamw_tangent")
    return next_theta, dm_new, dv_new


def advance_full_transports(states, coefficient_rows, sources, parameters, optimizer, step, config):
    opt = config["optimizer"]; beta1, beta2 = map(float, opt["betas"]); lr = float(opt["learning_rate"]); eps = float(opt["eps"]); wd = float(opt["weight_decay"])
    for module_index, (parameter, coefficients, source) in enumerate(zip(parameters, coefficient_rows, sources)):
        gradient = parameter.grad.detach(); optimizer_state = optimizer.state.get(parameter, {}); m = optimizer_state.get("exp_avg", torch.zeros_like(parameter)).detach(); v = optimizer_state.get("exp_avg_sq", torch.zeros_like(parameter)).detach()
        for variant in VARIANTS:
            current = states[variant][module_index]; dgradient = full_fisher_product(coefficients, current["theta"]) + source
            if variant == "T0_SGD":
                current["theta"] = (1 - lr * wd) * current["theta"] - lr * dgradient
                continue
            theta, dm, dv = stable_adamw_tangent_step(parameter.detach(), gradient, m, v, current["theta"], current["m"], current["v"], dgradient, step=step, lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=wd, full_v=variant == "T2_AdamW_full_state")
            current["theta"], current["m"], current["v"] = theta, dm, dv


def run_canonical_full(model, dataset, indices, user_ids, selected_users, parameters, names, device, pad, config, budget):
    optimizer = p1._optimizer(parameters, config); states = new_full_states(parameters, len(selected_users)); traces = []
    for step, start in enumerate(range(0, len(indices), config["schedule"]["batch_size"]), 1):
        chosen = indices[start:start + config["schedule"]["batch_size"]]; batch_users = [int(user_ids[index]) for index in chosen]; batch = move_batch(masked_batch(dataset, chosen, pad), device); losses = p1._sample_losses(model, batch)
        per_module_samples = [[] for _ in parameters]; summed = [torch.zeros_like(parameter) for parameter in parameters]
        for sample in range(len(chosen)):
            gradients = torch.autograd.grad(losses[sample], parameters, retain_graph=sample + 1 < len(chosen))
            for module_index, gradient in enumerate(gradients):
                detached = gradient.detach(); summed[module_index].add_(detached / len(chosen)); per_module_samples[module_index].append(detached)
        optimizer.zero_grad(set_to_none=True)
        for parameter, gradient in zip(parameters, summed): parameter.grad = gradient
        coefficient_rows = [torch.stack(rows) for rows in per_module_samples]; sources = []
        for coefficients, parameter in zip(coefficient_rows, parameters):
            rows = []
            for target in selected_users:
                slots = [index for index, user in enumerate(batch_users) if user == target]
                rows.append(-coefficients[slots].sum(0) / len(chosen) if slots else torch.zeros_like(parameter))
            sources.append(torch.stack(rows))
        advance_full_transports(states, coefficient_rows, sources, parameters, optimizer, step, config); budget.step(optimizer, "canonical_reference")
        traces.append({"step": step, "loss": float(losses.mean().detach()), "batch_hash": canonical_hash(chosen), "selected_user_slot_counts": [batch_users.count(user) for user in selected_users]})
        del batch, losses, per_module_samples, summed, coefficient_rows, sources
    cpu_states = {variant: [{key: value.detach().cpu() for key, value in module.items()} for module in modules] for variant, modules in states.items()}
    return {name: parameter.detach().cpu().clone() for name, parameter in zip(names, parameters)}, cpu_states, traces, tensor_tree_hash(optimizer.state_dict()), tensor_tree_hash(capture_rng())


def vector_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    left = reference.float(); right = candidate.float(); left_finite = bool(torch.isfinite(left).all()); right_finite = bool(torch.isfinite(right).all())
    result = {"reference_finite": left_finite, "candidate_finite": right_finite, "reference_nonfinite_count": int((~torch.isfinite(left)).sum()), "candidate_nonfinite_count": int((~torch.isfinite(right)).sum()), "candidate_sha256": tensor_tree_hash({"vector": right})}
    if not left_finite or not right_finite:
        result.update({"reference_norm": float(torch.linalg.vector_norm(left.double())) if left_finite else None, "candidate_norm": float(torch.linalg.vector_norm(right.double())) if right_finite else None, "cosine": None, "norm_ratio": None, "relative_l2_error": None, "maximum_absolute_error": None, "finite": False})
        return result
    ln = float(torch.linalg.vector_norm(left.double())); rn = float(torch.linalg.vector_norm(right.double())); exact_zero = ln == 0 and rn == 0
    cosine = 1. if exact_zero else float(torch.dot(left.double(), right.double()) / (ln * rn)) if ln and rn else None
    relative = 0. if exact_zero else float(torch.linalg.vector_norm((right - left).double()) / ln) if ln else None
    result.update({"reference_norm": ln, "candidate_norm": rn, "cosine": cosine, "norm_ratio": 1. if exact_zero else rn / ln if ln else None, "relative_l2_error": relative, "maximum_absolute_error": float(torch.max(torch.abs(right - left))), "finite": True})
    return result


def quantization_report(vector: torch.Tensor, config: dict[str, Any]) -> dict[str, Any]:
    source = vector.float(); result = {}
    for name, dtype, limit in (("float16", torch.float16, config["quantization"]["fp16_relative_l2_maximum"]), ("bfloat16", torch.bfloat16, config["quantization"]["bf16_relative_l2_maximum"])):
        restored = source.to(dtype).float(); metrics = vector_metrics(source, restored); metrics["quantized_nonfinite_count"] = int((~torch.isfinite(restored)).sum()); metrics["status"] = "source_nonfinite" if not metrics["reference_finite"] else "quantization_overflow" if not metrics["candidate_finite"] else "finite"; metrics["passed"] = metrics["finite"] and metrics["cosine"] is not None and metrics["cosine"] >= config["quantization"]["cosine_minimum"] and metrics["relative_l2_error"] is not None and metrics["relative_l2_error"] <= limit; result[name] = metrics
    return result


def compare_full(actual_by_name, canonical_by_name, states, user_index, names, config):
    actual_parts = [(actual_by_name[name] - canonical_by_name[name]).reshape(-1).float() for name in names]; actual = torch.cat(actual_parts); variants = {}
    for variant in VARIANTS:
        predicted_parts = [states[variant][index]["theta"][user_index].reshape(-1).float() for index in range(len(names))]; predicted = torch.cat(predicted_parts); overall = vector_metrics(actual, predicted); modules = []
        for name, left, right in zip(names, actual_parts, predicted_parts):
            left_finite = bool(torch.isfinite(left).all()); right_finite = bool(torch.isfinite(right).all()); ln = float(torch.linalg.vector_norm(left.double())) if left_finite else None; rn = float(torch.linalg.vector_norm(right.double())) if right_finite else None; cosine = float(torch.dot(left.double(), right.double()) / (ln * rn)) if left_finite and right_finite and ln and rn else None
            modules.append({"module_hash": hashlib.sha256(name.encode()).hexdigest(), "reference_finite": left_finite, "candidate_finite": right_finite, "cosine": cosine, "actual_norm": ln, "predicted_norm": rn, "positive": cosine is not None and cosine > 0})
        positive = sum(row["positive"] for row in modules) / len(modules); high = sorted(modules, key=lambda row: row["actual_norm"] if row["actual_norm"] is not None else -1., reverse=True)[:18]; high_positive = sum(row["positive"] for row in high) / len(high)
        overall.update({"positive_module_fraction": positive, "high_energy_positive_module_fraction": high_positive, "modules": modules, "prediction_quantization": quantization_report(predicted, config)})
        overall["base_gates_passed"] = overall["finite"] and overall["cosine"] is not None and overall["cosine"] >= config["gates"]["cosine_minimum"] and overall["norm_ratio"] is not None and config["gates"]["norm_ratio_minimum"] <= overall["norm_ratio"] <= config["gates"]["norm_ratio_maximum"] and overall["relative_l2_error"] is not None and overall["relative_l2_error"] <= config["gates"]["relative_l2_maximum"] and positive >= config["gates"]["positive_module_fraction_minimum"] and high_positive >= config["gates"]["positive_module_fraction_minimum"]
        variants[variant] = overall
    actual_finite = bool(torch.isfinite(actual).all())
    return {"actual_delta_finite": actual_finite, "actual_delta_nonfinite_count": int((~torch.isfinite(actual)).sum()), "actual_delta_norm": float(torch.linalg.vector_norm(actual.double())) if actual_finite else None, "actual_delta_sha256": tensor_tree_hash({"actual": actual}), "authority_quantization": quantization_report(actual, config), "variants": variants, "raw_vectors_persisted": False}


def classify(users: Sequence[dict[str, Any]]) -> tuple[str, dict[str, bool]]:
    numerical = all(row["transport"]["authority_quantization"]["bfloat16"]["reference_finite"] and all(row["transport"]["variants"][variant]["finite"] for variant in VARIANTS) for row in users)
    transport = all(row["transport"]["variants"]["T2_AdamW_full_state"]["base_gates_passed"] and row["optimizer_aware_gate_passed"] for row in users)
    fp16 = all(row["transport"]["authority_quantization"]["float16"]["passed"] and row["transport"]["variants"]["T2_AdamW_full_state"]["prediction_quantization"]["float16"]["passed"] for row in users)
    bf16 = all(row["transport"]["authority_quantization"]["bfloat16"]["passed"] and row["transport"]["variants"]["T2_AdamW_full_state"]["prediction_quantization"]["bfloat16"]["passed"] for row in users)
    classification = "full_module_transport_numerically_unstable" if not numerical else "full_module_transport_supported" if transport else "full_module_transport_insufficient"
    return classification, {"transport_numerically_finite_all_users": numerical, "full_module_t2_all_users": transport, "fp16_bank_all_users": fp16, "bf16_bank_all_users": bf16, "all": numerical and transport and fp16}


def _finite(value: Any) -> bool:
    if isinstance(value, float): return math.isfinite(value)
    if isinstance(value, dict): return all(_finite(item) for item in value.values())
    if isinstance(value, list): return all(_finite(item) for item in value)
    return True


def preflight(root: Path, config_path: Path) -> dict[str, Any]:
    config = load_frozen_config(config_path); predecessor = validate_predecessor(root, config)
    return {"schema": SCHEMA, "mode": "Preflight", "predecessor": {key: value for key, value in predecessor.items() if key != "schedule"}, "schedule": config["schedule"], "transport": config["transport"], "quantization": config["quantization"], "physical_optimizer_step_budget": 200, "expected_breakdown": {"canonical_reference": 50, "masked_high_frequency": 50, "masked_median_frequency": 50, "masked_low_frequency": 50}, "shared_low_rank_coordinate_used": False, "module_truncation_used": False, "model_loaded": False, "development_loaded": False, "retrain_loaded": False, "test_accessed": False}


def budget(config):
    total = config["schedule"]["steps"] * 4
    if total != config["runtime"]["physical_optimizer_step_limit"]: raise RuntimeError("P2-B budget mismatch")
    return {"schema": SCHEMA, "mode": "BudgetAudit", "step_positions": 50, "canonical_optimizer_steps": 50, "masked_optimizer_steps": 150, "physical_optimizer_step_calls": total, "transport_extra_optimizer_steps": 0, "quantization_extra_optimizer_steps": 0, "authoritative_optimizer_steps_committed": 0, "test_accessed": False}


def synthetic():
    coefficients = torch.tensor([[1., 2.], [3., 4.]]); vectors = torch.tensor([[1., 0.], [0., 1.]]); got = full_fisher_product(coefficients, vectors); expected = torch.stack([coefficients.T @ coefficients[:, 0] / 2, coefficients.T @ coefficients[:, 1] / 2]); q = quantization_report(torch.tensor([.1, -.2, .3]), {"quantization": {"fp16_relative_l2_maximum": .001, "bf16_relative_l2_maximum": .01, "cosine_minimum": .9999}})
    return {"schema": SCHEMA, "full_fisher_exact": torch.allclose(got, expected), "fp16_passed": q["float16"]["passed"], "bf16_passed": q["bfloat16"]["passed"], "optimizer_steps": 0, "real_model_loaded": False, "test_accessed": False}


def publish(stage, destination, report, implementation_sha):
    if not _finite(report): raise ValueError("nonfinite P2-B report")
    atomic_json(stage / REPORT, report); atomic_json(stage / "run_state.json", {"schema": SCHEMA, "status": "COMPLETED", "classification": report["classification"], "gates": report["gates"], "physical_optimizer_step_calls": report["execution"]["physical_optimizer_step_calls"], "authoritative_parameters_modified": False, "test_accessed": False}); (stage / "COMPLETED").write_text(MARKER + "\n", encoding="utf-8", newline="\n"); atomic_json(stage / "manifest.json", {"schema": SCHEMA, "status": "COMPLETED", "report_sha256": sha256_file(stage / REPORT), "run_state_sha256": sha256_file(stage / "run_state.json"), "implementation_sha256": implementation_sha, "model_artifact_published": False, "bank_artifact_published": False, "published_atomically": True, "test_accessed": False}); destination.parent.mkdir(parents=True, exist_ok=True); os.replace(stage, destination)


def analyze(root, config_path, run_name):
    config = load_frozen_config(config_path); run = root / config["output_root"] / safe_run_name(run_name); required = {"COMPLETED", "manifest.json", REPORT, "run_state.json"}
    if not run.is_dir() or {path.name for path in run.iterdir()} != required or (run / "COMPLETED").read_text(encoding="utf-8") != MARKER + "\n": raise ValueError("invalid P2-B run")
    report = json.loads((run / REPORT).read_text(encoding="utf-8")); manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8")); state = json.loads((run / "run_state.json").read_text(encoding="utf-8"))
    if manifest.get("report_sha256") != sha256_file(run / REPORT) or manifest.get("run_state_sha256") != sha256_file(run / "run_state.json") or state.get("physical_optimizer_step_calls") != 200 or report.get("test_accessed") is not False: raise ValueError("P2-B evidence mismatch")
    summary = {row["role"]: {variant: {key: value for key, value in row["transport"]["variants"][variant].items() if key in {"cosine", "norm_ratio", "relative_l2_error", "positive_module_fraction", "base_gates_passed"}} for variant in VARIANTS} for row in report["users"]}
    return {"status": "COMPLETED", "run_dir": str(run), "classification": report["classification"], "gates": report["gates"], "user_summary": summary, "quantization_summary": report["quantization_summary"], "execution": report["execution"], "test_accessed": False}


def execute(root: Path, config_path: Path, run_name: str):
    config = load_frozen_config(config_path); run_name = safe_run_name(run_name); destination = (root / config["output_root"] / run_name).resolve()
    if destination.exists(): raise FileExistsError(destination)
    predecessor = validate_predecessor(root, config); git = git_snapshot(root)
    if not git["clean"]: raise RuntimeError("formal P2-B requires clean Git")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1: raise RuntimeError("one CUDA GPU required")
    device = torch.device("cuda:0"); free, total = torch.cuda.mem_get_info(device)
    if free / GIB < config["runtime"]["minimum_free_gib"]: raise RuntimeError("insufficient clean dedicated GPU memory")
    train_users, replay = p1._train_user_ids_only(root / config["raw_data"]); schedule = p1.freeze_schedule(train_users, seed=42, steps=50, batch_size=16, calibration_batches=64); public, old = schedule["public"], predecessor["schedule"]
    for key in ("train_order_sha256", "batch_order_sha256", "selected_user_roles", "selected_user_frequency", "selected_user_window_visits", "selected_user_hashes"):
        if public[key] != old[key]: raise RuntimeError(f"P2-A schedule mismatch: {key}")
    work = destination.parent / ".work"; work.mkdir(parents=True, exist_ok=True); stage = work / f"{run_name}.{uuid.uuid4().hex}.stage"; stage.mkdir(); started = time.perf_counter(); checkpoint_before = sha256_file(root / config["original_checkpoint"]); step_budget = p1.StepBudget(200); model = None; initial_rng = None
    try:
        torch.cuda.set_per_process_memory_fraction(config["runtime"]["allocator_fraction"], device); torch.cuda.reset_peak_memory_stats(); base = load_config(root / config["base_config"], root); tokenizer = T5Tokenizer.from_pretrained(base["paths"]["model_dir"]); pad = tokenizer.pad_token_id; dataset = JsonPromptDataset(root / config["train_data"], tokenizer)
        if pad is None or len(dataset) != len(train_users): raise RuntimeError("train/tokenizer mismatch")
        initial_rng = capture_rng(); random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42); arm_rng = capture_rng(); runtime_config = {"original_checkpoint": config["original_checkpoint"], "coordinate": {"lora_rank": 16, "fixed_a_seed": 42, "lora_alpha": 32}}
        model, names, parameters, fixed_a, fixed_report = p1._fresh_runtime(root, runtime_config, device)
        if len(names) != 72 or sum(parameter.numel() for parameter in parameters) != config["coordinate"]["transport_dimension"]: raise RuntimeError("full Q/V coordinate schema mismatch")
        del model; model = None; gc.collect(); torch.cuda.empty_cache(); restore_rng(arm_rng); model, names, parameters, _, _ = p1._fresh_runtime(root, runtime_config, device, fixed_a); canonical, states, canonical_trace, canonical_optimizer_hash, canonical_rng_hash = run_canonical_full(model, dataset, schedule["train_indices"], train_users, schedule["selected_users"], parameters, names, device, pad, config, step_budget); del model; model = None; gc.collect(); torch.cuda.empty_cache(); users = []
        for user_index, (role, target) in enumerate(zip(config["schedule"]["users"], schedule["selected_users"])):
            restore_rng(arm_rng); model, names, parameters, _, _ = p1._fresh_runtime(root, runtime_config, device, fixed_a); actual, trace, optimizer_hash, rng_hash = p1.run_masked(model, dataset, schedule["train_indices"], train_users, target, parameters, names, device, pad, config, step_budget, f"masked_{role}"); transport = compare_full(actual, canonical, states, user_index, names, config); variants = transport["variants"]; errors = {variant: variants[variant]["relative_l2_error"] for variant in VARIANTS}; optimizer_aware = all(value is not None for value in errors.values()) and errors["T2_AdamW_full_state"] < errors["T0_SGD"] and errors["T2_AdamW_full_state"] <= errors["T1_AdamW_frozen_v"] + 1e-12
            users.append({"role": role, "user_hash": public["selected_user_hashes"][user_index], "global_frequency": public["selected_user_frequency"][user_index], "window_visits": public["selected_user_window_visits"][user_index], "masked_reference": {"steps": 50, "slots_preserved": True, "batch_denominator_preserved": True, "trace_sha256": canonical_hash(trace), "optimizer_state_sha256": optimizer_hash, "canonical_rng_exact": rng_hash == canonical_rng_hash}, "transport": transport, "optimizer_aware_gate_passed": optimizer_aware}); del model, actual; model = None; gc.collect(); torch.cuda.empty_cache()
        expected_steps = {"canonical_reference": 50, "masked_high_frequency": 50, "masked_median_frequency": 50, "masked_low_frequency": 50}
        if step_budget.calls != 200 or step_budget.by_arm != expected_steps: raise RuntimeError("P2-B physical step accounting mismatch")
        classification, gates = classify(users); peak = torch.cuda.max_memory_reserved() / GIB; source_unchanged = sha256_file(root / config["original_checkpoint"]) == checkpoint_before
        if peak > config["runtime"]["hard_peak_reserved_gib"] or not source_unchanged or not all(row["masked_reference"]["canonical_rng_exact"] for row in users): raise RuntimeError("P2-B integrity failure")
        implementation_paths = (Path(__file__), config_path, root / "scripts/bota_if/run_p2b_full_module_adamw_transport_v1.ps1"); implementation = {str(path.relative_to(root)).replace("\\", "/"): sha256_file(path) for path in implementation_paths}; implementation_sha = canonical_hash(implementation)
        def finite_max(values):
            rows = [value for value in values if value is not None and math.isfinite(value)]
            return max(rows) if rows else None
        quantization_summary = {format_name: {"authority_all_users": all(row["transport"]["authority_quantization"][format_name]["passed"] for row in users), "t2_prediction_all_users": all(row["transport"]["variants"]["T2_AdamW_full_state"]["prediction_quantization"][format_name]["passed"] for row in users), "authority_nonfinite_or_overflow_users": sum(not row["transport"]["authority_quantization"][format_name]["passed"] for row in users), "t2_nonfinite_or_overflow_users": sum(not row["transport"]["variants"]["T2_AdamW_full_state"]["prediction_quantization"][format_name]["passed"] for row in users), "maximum_authority_relative_l2": finite_max(row["transport"]["authority_quantization"][format_name]["relative_l2_error"] for row in users), "maximum_t2_relative_l2": finite_max(row["transport"]["variants"]["T2_AdamW_full_state"]["prediction_quantization"][format_name]["relative_l2_error"] for row in users)} for format_name in ("float16", "bfloat16")}
        report = {"schema": SCHEMA, "run_name": run_name, "status": "COMPLETED", "classification": classification, "gates": gates, "predecessor": {key: value for key, value in predecessor.items() if key != "schedule"}, "coordinate": {"fixed_a": fixed_report, "module_count": 72, "transport_dimension": 884736, "transport_coordinate": "full_qv_lora_b", "shared_low_rank_coordinate_used": False, "module_truncation_used": False}, "transport": config["transport"], "quantization": config["quantization"], "quantization_summary": quantization_summary, "schedule": public, "users": users, "execution": {"step_positions": 50, "canonical_optimizer_steps": 50, "masked_optimizer_steps": 150, "physical_optimizer_step_calls": 200, "physical_optimizer_step_limit": 200, "transport_extra_optimizer_steps": 0, "quantization_extra_optimizer_steps": 0, "authoritative_optimizer_steps_committed": 0, "by_arm": step_budget.by_arm, "canonical_trace_sha256": canonical_hash(canonical_trace), "canonical_optimizer_state_sha256": canonical_optimizer_hash, "canonical_final_rng_sha256": canonical_rng_hash}, "lineage": replay, "integrity": {"source_checkpoint_unchanged": source_unchanged, "source_checkpoint_sha256_before": checkpoint_before, "source_checkpoint_sha256_after": sha256_file(root / config["original_checkpoint"]), "authoritative_parameters_modified": False, "model_artifact_published": False, "bank_artifact_published": False, "development_loaded": False, "retrain_loaded": False, "test_loader_built": False}, "privacy": config["privacy"], "scientific_scope": config["scientific_scope"], "memory": {"peak_reserved_gib": peak, "hard_peak_reserved_gib": config["runtime"]["hard_peak_reserved_gib"], "device_total_gib": total / GIB}, "hardware": hardware_snapshot(), "git": git, "implementation": implementation, "implementation_sha256": implementation_sha, "wall_time_seconds": time.perf_counter() - started, "test_accessed": False}; publish(stage, destination, report, implementation_sha); restore_rng(initial_rng); return {"status": "COMPLETED", "run_dir": str(destination), "classification": classification, "gates": gates, "physical_optimizer_step_calls": 200, "test_accessed": False}
    except Exception as error:
        if stage.exists() and not destination.exists():
            failure = {"schema": SCHEMA, "status": "INTERRUPTED", "reason": type(error).__name__, "message": str(error), "physical_optimizer_step_calls": step_budget.calls, "physical_optimizer_step_limit": 200, "source_checkpoint_sha256_before": checkpoint_before, "source_checkpoint_sha256_after": sha256_file(root / config["original_checkpoint"]), "authoritative_parameters_modified": False, "test_accessed": False}; atomic_json(stage / "run_state.json", failure); (stage / "INTERRUPTED").write_text("BOTA_IF_P2B_INTERRUPTED\n", encoding="utf-8", newline="\n"); atomic_json(stage / "manifest.json", {"schema": SCHEMA, "status": "INTERRUPTED", "run_state_sha256": sha256_file(stage / "run_state.json"), "physical_optimizer_step_calls": step_budget.calls, "published_atomically": True, "test_accessed": False}); destination.parent.mkdir(parents=True, exist_ok=True); os.replace(stage, destination)
        raise RuntimeError(f"P2-B interrupted; immutable evidence published at {destination}") from error
    finally:
        if model is not None: del model
        if initial_rng is not None: restore_rng(initial_rng)
        if stage.exists(): shutil.rmtree(stage)
        gc.collect(); torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path.cwd()); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--mode", choices=["Preflight", "BudgetAudit", "SyntheticDryRun", "Full", "Analyze"], default="Preflight"); parser.add_argument("--run-name"); args = parser.parse_args(); root, config_path = args.root.resolve(), args.config.resolve(); config = load_frozen_config(config_path)
    if args.mode == "Preflight": result = preflight(root, config_path)
    elif args.mode == "BudgetAudit": result = budget(config)
    elif args.mode == "SyntheticDryRun": result = synthetic()
    elif not args.run_name: raise ValueError(f"{args.mode} requires --run-name")
    elif args.mode == "Analyze": result = analyze(root, config_path, args.run_name)
    else: result = execute(root, config_path, args.run_name)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
