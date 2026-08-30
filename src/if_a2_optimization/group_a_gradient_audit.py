"""Group A: zero-update audit of IF-A2 Forget-gradient units and sign.

This module deliberately does not construct an optimizer, HVP, CG solve, or
update-space projection.  It publishes scalar summaries and tensor hashes only.
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
from typing import Any, Sequence

import torch
import torch.nn.functional as F
import yaml
from transformers import T5Tokenizer

from src.diagnostics.t5_full_runner import _batch
from src.diagnostics.t5_lora_influence_feasibility_audit import flatten_tensors
from src.diagnostics.t5_lora_influence_feasibility_audit_a2 import (
    analytic_b_gradient,
    build_fixed_a_basis,
    collect_qv_modules,
    install_fixed_ab_coordinate,
    select_retain_panels,
)
from src.diagnostics.t5_reconstructed_official import JsonPromptDataset, load_config, load_legacy_model, move_batch
from src.diagnostics.t5_step817_forget_conflict_audit import tensor_tree_hash
from src.paper_if_a2.common import atomic_json, canonical_hash, git_snapshot, hardware_snapshot, safe_run_name, sha256_file
from src.paper_if_a2.if_a2_method import _method_lineage, load_method_config

SCHEMA = "if-a2-group-a-gradient-audit-v2"
MARKER = "IF_A2_GROUP_A_GRADIENT_AUDIT_V2_COMPLETED"
GIB = 1024 ** 3


def cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denominator = float(torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right))
    return float(torch.dot(left, right)) / denominator if denominator else None


def scalar_equivalence(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    """Fit left ~= coefficient * right and report the non-scalar residual."""
    denominator = float(torch.dot(right, right))
    if denominator == 0:
        return {"coefficient": None, "relative_residual": None, "scalar_equivalent": False}
    coefficient = float(torch.dot(left, right)) / denominator
    residual = left - coefficient * right
    relative = float(torch.linalg.vector_norm(residual)) / max(float(torch.linalg.vector_norm(left)), 1e-300)
    return {"coefficient": coefficient, "relative_residual": relative, "scalar_equivalent": None}


def reduction_identity_report(vectors: dict[str, torch.Tensor], samples: int, tokens: int, tolerance: float) -> dict[str, Any]:
    comparisons = {
        "sample_sum_vs_k_sample_mean": (vectors["sample_sum"], samples * vectors["sample_mean"]),
        "token_sum_vs_N_token_mean": (vectors["token_sum"], tokens * vectors["token_mean"]),
        "current_vs_token_mean": (vectors["current"], vectors["token_mean"]),
    }
    rows: dict[str, Any] = {}
    for name, (left, right) in comparisons.items():
        residual = left - right; reference = max(float(torch.linalg.vector_norm(left)), float(torch.linalg.vector_norm(right)), 1e-300)
        relative = float(torch.linalg.vector_norm(residual)) / reference
        rows[name] = {"absolute_error": float(torch.linalg.vector_norm(residual)), "relative_error": relative, "cosine": cosine(left, right), "passed": relative <= tolerance}
    rows["all_passed"] = all(row["passed"] for row in rows.values())
    return rows


def _projection_kind(name: str) -> str:
    tail = name.rsplit(".", 1)[-1].lower()
    return {"q": "Q", "k": "K", "v": "V", "o": "O"}.get(tail, "OTHER")


def vector_report(vectors: dict[str, torch.Tensor], module_names: Sequence[str], shapes: Sequence[torch.Size]) -> dict[str, Any]:
    offsets = [0]
    for shape in shapes: offsets.append(offsets[-1] + math.prod(shape))
    summary: dict[str, Any] = {}
    for name, value in vectors.items():
        modules = {}; layers: dict[str, float] = {}; projections = {key: {"registered": key in {"Q", "V"}, "squared_norm": 0.0} for key in ("Q", "K", "V", "O")}
        for module, start, end in zip(module_names, offsets[:-1], offsets[1:]):
            norm = float(torch.linalg.vector_norm(value[start:end])); modules[module] = norm
            layer = module.rsplit(".", 1)[0]; layers[layer] = layers.get(layer, 0.0) + norm ** 2
            kind = _projection_kind(module)
            if kind in projections: projections[kind]["squared_norm"] += norm ** 2
        for row in projections.values(): row["norm"] = math.sqrt(row.pop("squared_norm"))
        summary[name] = {"norm": float(torch.linalg.vector_norm(value)), "sha256": tensor_tree_hash({name: value}), "module_norms": modules, "layer_norms": {key: math.sqrt(item) for key, item in layers.items()}, "projection_norms": projections}
    pairwise = {}
    names = list(vectors)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            pairwise[f"{left}__{right}"] = {"cosine": cosine(vectors[left], vectors[right]), "norm_ratio_left_over_right": summary[left]["norm"] / max(summary[right]["norm"], 1e-300)}
    return {"vectors": summary, "pairwise": pairwise}


def gradient_comparison(reference: torch.Tensor, candidate: torch.Tensor, module_names: Sequence[str], shapes: Sequence[torch.Size], tolerance: float) -> dict[str, Any]:
    if reference.shape != candidate.shape: raise ValueError("gradient comparison shape mismatch")
    offsets = [0]
    for shape in shapes: offsets.append(offsets[-1] + math.prod(shape))
    modules = []
    for name, start, end in zip(module_names, offsets[:-1], offsets[1:]):
        left = reference[start:end]; right = candidate[start:end]; absolute = float(torch.linalg.vector_norm(right - left)); denominator = max(float(torch.linalg.vector_norm(left)), float(torch.linalg.vector_norm(right)), 1e-300)
        modules.append({"module": name, "absolute_error": absolute, "relative_error": absolute / denominator, "cosine": cosine(left, right)})
    absolute = float(torch.linalg.vector_norm(candidate - reference)); denominator = max(float(torch.linalg.vector_norm(reference)), float(torch.linalg.vector_norm(candidate)), 1e-300); relative = absolute / denominator
    return {"absolute_error": absolute, "relative_error_vs_production": relative, "cosine_vs_production": cosine(candidate, reference), "reference_sha256": tensor_tree_hash({"gradient": reference}), "candidate_sha256": tensor_tree_hash({"gradient": candidate}), "module_errors": modules, "maximum_module_relative_error": max(row["relative_error"] for row in modules), "tolerance": tolerance, "passed": relative <= tolerance}


def token_and_sample_numerators(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction="none").reshape_as(labels)
    mask = labels.ne(-100); counts = mask.sum(dim=1)
    if bool((counts <= 0).any()): raise RuntimeError("sample without valid target token")
    token_sum = (losses * mask).sum(); sample_sum = ((losses * mask).sum(dim=1) / counts).sum()
    return token_sum, sample_sum, int(mask.sum())


def masked_batch(dataset: JsonPromptDataset, indices: Sequence[int], pad_token_id: int) -> dict[str, torch.Tensor]:
    batch = _batch(dataset, list(indices)); mask = batch["input_ids"].ne(int(pad_token_id))
    if mask.shape != batch["input_ids"].shape or mask.dtype is not torch.bool: raise RuntimeError("invalid encoder attention mask")
    if not bool(mask.any(dim=1).all()): raise RuntimeError("empty encoder input in corrected masking protocol")
    return {**batch, "attention_mask": mask}


def masked_forward(model: torch.nn.Module, batch: dict[str, torch.Tensor]):
    if "attention_mask" not in batch: raise RuntimeError("corrected masking protocol requires attention_mask")
    return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["target_ids"])


def stream_weight_gradients(model: torch.nn.Module, dataset: JsonPromptDataset, indices: Sequence[int], parameters: Sequence[torch.Tensor], device: torch.device, batch_size: int, pad_token_id: int) -> tuple[list[torch.Tensor], dict[str, Any]]:
    accumulators = [torch.zeros_like(parameter, dtype=torch.float64, device="cpu") for parameter in parameters]; total_tokens = 0; loss_numerator = 0.; calls = 0; masked_padding_tokens = 0
    for start in range(0, len(indices), batch_size):
        batch = move_batch(masked_batch(dataset, indices[start:start + batch_size], pad_token_id), device); output = masked_forward(model, batch); tokens = int((batch["target_ids"] != -100).sum()); weighted = output.loss * tokens; gradients = torch.autograd.grad(weighted, parameters, create_graph=False, retain_graph=False)
        for accumulator, gradient in zip(accumulators, gradients): accumulator.add_(gradient.detach().to(torch.float64).cpu())
        total_tokens += tokens; loss_numerator += float(weighted.detach().cpu()); calls += 1; masked_padding_tokens += int((~batch["attention_mask"]).sum()); del batch, output, weighted, gradients
    if total_tokens <= 0 or any(parameter.grad is not None for parameter in parameters): raise RuntimeError("invalid corrected streaming gradient state")
    return [value / total_tokens for value in accumulators], {"loss": loss_numerator / total_tokens, "valid_tokens": total_tokens, "forward_batches": calls, "autograd_grad_calls": calls, "encoder_attention_mask": "input_ids_ne_pad_token_id", "masked_padding_tokens": masked_padding_tokens}


def stream_reductions(model: torch.nn.Module, dataset: JsonPromptDataset, indices: Sequence[int], parameters: Sequence[torch.Tensor], device: torch.device, batch_size: int, pad_token_id: int) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    token_acc = [torch.zeros_like(p, dtype=torch.float64, device="cpu") for p in parameters]; sample_acc = [torch.zeros_like(p, dtype=torch.float64, device="cpu") for p in parameters]
    total_tokens = 0; batches = 0
    for start in range(0, len(indices), batch_size):
        chosen = list(indices[start:start + batch_size]); batch = move_batch(masked_batch(dataset, chosen, pad_token_id), device); output = masked_forward(model, batch)
        token_sum, sample_sum, tokens = token_and_sample_numerators(output.logits, batch["target_ids"])
        token_grad = torch.autograd.grad(token_sum, parameters, retain_graph=True); sample_grad = torch.autograd.grad(sample_sum, parameters)
        for target, values in ((token_acc, token_grad), (sample_acc, sample_grad)):
            for accumulator, gradient in zip(target, values): accumulator.add_(gradient.detach().to(dtype=torch.float64, device="cpu"))
        total_tokens += tokens; batches += 1; del chosen, batch, output, token_sum, sample_sum, token_grad, sample_grad
    token_sum_flat = flatten_tensors(token_acc); sample_sum_flat = flatten_tensors(sample_acc)
    return {"sample_sum": sample_sum_flat, "sample_mean": sample_sum_flat / len(indices), "token_sum": token_sum_flat, "token_mean": token_sum_flat / total_tokens}, {"samples": len(indices), "valid_tokens": total_tokens, "batches": batches, "batch_size": batch_size, "encoder_attention_mask": "input_ids_ne_pad_token_id"}


def count_valid_tokens(dataset: JsonPromptDataset, indices: Sequence[int], batch_size: int) -> int:
    total = 0
    for start in range(0, len(indices), batch_size): total += int((_batch(dataset, list(indices[start:start + batch_size]))["target_ids"] != -100).sum())
    return total


def coordinate_mapping_report(matrices: Sequence[torch.Tensor], module_names: Sequence[str], bases: dict[str, torch.Tensor], current: torch.Tensor, alpha: int, rank: int, tolerance: float) -> dict[str, Any]:
    analytic = []; rows = []
    for name, matrix in zip(module_names, matrices):
        basis = bases[name]; projected = matrix @ basis.T @ basis; matrix_norm = float(torch.linalg.vector_norm(matrix)); projected_norm = float(torch.linalg.vector_norm(projected))
        rows.append({"module": name, "weight_gradient_norm": matrix_norm, "fixed_A_projected_norm": projected_norm, "retained_norm_ratio": projected_norm / max(matrix_norm, 1e-300), "projection_cosine": cosine(matrix.reshape(-1), projected.reshape(-1))})
        analytic.append(analytic_b_gradient(matrix, basis, alpha, rank))
    analytic_flat = flatten_tensors(analytic); residual = float(torch.linalg.vector_norm(current - analytic_flat)) / max(float(torch.linalg.vector_norm(analytic_flat)), 1e-300)
    return {"modules": rows, "analytic_B_gradient_sha256": tensor_tree_hash({"analytic_B": analytic_flat}), "current_vs_analytic_relative_error": residual, "current_vs_analytic_cosine": cosine(current, analytic_flat), "alpha": alpha, "rank": rank, "alpha_over_rank": alpha / rank, "passed": residual <= tolerance}


def select_cross_user_indices(indices: Sequence[int], user_ids: Sequence[int], count: int) -> tuple[list[int], list[int]]:
    if len(indices) != len(user_ids): raise ValueError("Forget indices/user IDs are not aligned")
    chosen: list[int] = []; chosen_users: list[int] = []; seen: set[int] = set()
    for index, user_id in zip(indices, user_ids):
        user_id = int(user_id)
        if user_id in seen: continue
        chosen.append(int(index)); chosen_users.append(user_id); seen.add(user_id)
        if len(chosen) == count: break
    if len(chosen) != count: raise RuntimeError("not enough distinct Forget users for sign audit")
    return chosen, chosen_users


def finite_difference_sign(model: torch.nn.Module, dataset: JsonPromptDataset, indices: Sequence[int], user_ids: Sequence[int], parameters: Sequence[torch.Tensor], module_names: Sequence[str], bases: dict[str, torch.Tensor], base_modules: Sequence[tuple[str, torch.nn.Module]], device: torch.device, epsilons: Sequence[float], tolerance: float, alpha: int, rank: int, pad_token_id: int) -> dict[str, Any]:
    chosen, chosen_users = select_cross_user_indices(indices, user_ids, 2)
    if len(chosen) != 2: raise RuntimeError("sign audit requires exactly two fixed Forget samples")
    batch = move_batch(masked_batch(dataset, chosen, pad_token_id), device); output = masked_forward(model, batch); token_sum, _, tokens = token_and_sample_numerators(output.logits, batch["target_ids"]); objective = token_sum / tokens
    local_parts = torch.autograd.grad(objective, parameters); local = flatten_tensors(local_parts); local_norm = float(torch.linalg.vector_norm(local))
    if not math.isfinite(local_norm) or local_norm == 0: raise RuntimeError("invalid local sign-audit gradient")
    direction = local / local_norm; pieces = []; offset = 0
    for parameter in parameters:
        count = parameter.numel(); pieces.append(direction[offset:offset + count].reshape_as(parameter).to(parameter)); offset += count
    base_norm = math.sqrt(sum(float(torch.sum(module.weight.detach().float() ** 2)) for _, module in base_modules)); effective_direction_norm = 0.0
    for name, piece in zip(module_names, pieces): effective_direction_norm += float(torch.sum(((alpha / rank) * piece @ bases[name].to(piece)).float() ** 2))
    effective_direction_norm = math.sqrt(effective_direction_norm)
    if effective_direction_norm == 0: raise RuntimeError("zero effective-weight finite-difference direction")
    snapshots = [parameter.detach().clone() for parameter in parameters]; before = tensor_tree_hash({str(i): value for i, value in enumerate(snapshots)})
    def panel_loss() -> float:
        item = move_batch(masked_batch(dataset, chosen, pad_token_id), device)
        with torch.no_grad(): result = masked_forward(model, item); numerator, _, count = token_and_sample_numerators(result.logits, item["target_ids"]); value = float((numerator / count).detach().cpu())
        del item, result, numerator; return value
    rows = []
    try:
        for epsilon in epsilons:
            multiplier = float(epsilon) * base_norm / effective_direction_norm
            with torch.no_grad():
                for parameter, snapshot, piece in zip(parameters, snapshots, pieces): parameter.copy_(snapshot + multiplier * piece)
            plus = panel_loss()
            with torch.no_grad():
                for parameter, snapshot, piece in zip(parameters, snapshots, pieces): parameter.copy_(snapshot - multiplier * piece)
            minus = panel_loss(); central = (plus - minus) / (2 * multiplier); relative_error = abs(central - local_norm) / max(abs(local_norm), 1e-300)
            rows.append({"relative_effective_weight_epsilon": float(epsilon), "B_space_multiplier": multiplier, "loss_plus": plus, "loss_minus": minus, "central_difference": central, "analytic_directional_derivative": local_norm, "sign_matches": central > 0, "relative_error": relative_error})
    finally:
        with torch.no_grad():
            for parameter, snapshot in zip(parameters, snapshots): parameter.copy_(snapshot)
    after = tensor_tree_hash({str(i): parameter.detach() for i, parameter in enumerate(parameters)}); best_row = min(rows, key=lambda row: row["relative_error"]); best = best_row["relative_error"]
    del batch, output, token_sum, objective, local_parts
    return {"sample_indices": chosen, "authoritative_user_ids": chosen_users, "distinct_users": len(set(chosen_users)) == 2, "sample_count": len(chosen), "objective": "valid_token_mean_CE_on_same_two_cross_user_samples_with_explicit_encoder_mask", "direction": "normalized_analytic_gradient_of_same_objective", "encoder_attention_mask": "input_ids_ne_pad_token_id", "increase_forget_loss_requires_positive_derivative": True, "rows": rows, "best_relative_error": best, "best_scale": best_row["relative_effective_weight_epsilon"], "at_least_one_scale_within_tolerance": best <= tolerance, "best_scale_sign_matches": best_row["sign_matches"], "negative_scale_count": sum(not row["sign_matches"] for row in rows), "B_restored_exact": before == after, "passed": best <= tolerance and best_row["sign_matches"] and before == after}


def derive_unit_ledger(*, samples_full: int, samples_forget: int, samples_retain: int, tokens_forget: int, tokens_retain: int, h_panel_samples: int, h_panel_tokens: int, current_vs_sample: dict[str, Any]) -> dict[str, Any]:
    if samples_forget + samples_retain != samples_full or tokens_forget <= 0 or tokens_retain <= 0: raise ValueError("invalid empirical-risk counts")
    rho = samples_forget / samples_full; token_rho = tokens_forget / (tokens_forget + tokens_retain)
    return {
        "current_gradient_exact_form": f"(1/{tokens_forget}) * sum_over_Forget_valid_tokens grad_B CE_token with encoder attention_mask=(input_ids!=pad_token_id)",
        "current_vs_sample_mean_scalar_fit": current_vs_sample,
        "contains_sample_k_over_n": False,
        "contains_token_NF_over_N": False,
        "sample_counts": {"n": samples_full, "k": samples_forget, "retain": samples_retain, "rho": rho},
        "valid_token_counts": {"full": tokens_forget + tokens_retain, "forget": tokens_forget, "retain": tokens_retain, "token_rho": token_rho},
        "classical_sample_risk_rhs_multiplier_for_current": None if not current_vs_sample.get("scalar_equivalent") else (rho / (1 - rho)) / current_vs_sample["coefficient"],
        "token_risk_rhs_multiplier_for_current": tokens_forget / tokens_retain,
        "curvature_exact_form": f"(1/{h_panel_tokens}) * sum_over_{h_panel_samples}_Retain_panel_valid_tokens Hessian_B self_KL_token",
        "linear_system": f"[((1/{h_panel_tokens}) * sum_Rpanel_token H_KL_token) + (0.01 * lambda_max) I] x = (1/{tokens_forget}) * sum_F_token grad_B CE_token, under explicit encoder padding masks",
        "risk_unit_consistency": "g_F and H_R are both valid-token means, but the production right-hand side omits the token-risk deletion mass N_F/N_R",
        "sign_convention": "the production system solves a positive Forget-CE gradient direction and adds it to B; local positive directional derivative therefore means increased Forget loss",
        "group_B_required_derived_scale": tokens_forget / tokens_retain,
        "group_B_scale_is_preregistered_candidate": False,
    }


def _validate_config(config: dict[str, Any]) -> None:
    expected_top = {"schema", "development_only", "test_access_policy", "audit_target", "input_masking", "ratio_experiment", "method_config", "output_root", "runtime", "sign_audit", "tolerances"}
    if set(config) != expected_top or config["schema"] != SCHEMA or config["development_only"] is not True or config["test_access_policy"] != "forbidden": raise ValueError("invalid Group-A config schema/policy")
    expected_ratio = {"experiment_name": "i02s42v1", "manifest": "outputs/ru1/i02s42v1/experiment_manifest.json", "completion_marker": "outputs/ru1/i02s42v1/COMPLETED", "contract_sha256": "21057df5fe217c36d5d459355eedc062e4c9fd0e83bb9cd50a357b18ce7d7527", "requested_forget_ratio": .02, "actual_interaction_ratio": .020966666666666668, "forget_samples": 1258, "retain_samples": 58742, "forget_users": 8}
    expected_masking = {"protocol": "explicit_encoder_attention_mask", "mask_definition": "input_ids_ne_pad_token_id", "applies_to": ["basis", "forget_gradient", "reduction_matrix", "finite_difference"], "legacy_unmasked_equivalence_claimed": False}
    if config["audit_target"] != "ratio_i02s42v1_interaction_2pct_corrected_masking" or config["input_masking"] != expected_masking or config["ratio_experiment"] != expected_ratio or config["runtime"] != {"production_batch_size": 16, "comparison_batch_sizes": [8, 16], "allocator_fraction": .88, "minimum_free_gib_before_load": 8.0, "hard_cap_reserved_gib": 14.0}: raise ValueError("frozen Group-A 2% corrected-masking target/runtime changed")
    if config["sign_audit"]["sample_count"] != 2 or config["sign_audit"]["relative_effective_weight_epsilons"] != [1e-5, 1e-4, 1e-3]: raise ValueError("frozen sign audit changed")


def validate_ratio_authority(root: Path, config: dict[str, Any], method_config_path: Path, method_config: dict[str, Any]) -> dict[str, Any]:
    ratio = config["ratio_experiment"]; manifest_path = root / ratio["manifest"]; marker_path = root / ratio["completion_marker"]
    if marker_path.read_text(encoding="utf-8") != "PAPER_RATIO_UNLEARNING_EXPERIMENT_PREPARED\n": raise ValueError("2% experiment completion marker mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")); deletion = method_config.get("deletion_experiment")
    required = {"schema": "paper-ratio-unlearning-suite-v1", "experiment_name": ratio["experiment_name"], "experiment_contract_sha256": ratio["contract_sha256"], "requested_forget_ratio": ratio["requested_forget_ratio"], "actual_interaction_ratio": ratio["actual_interaction_ratio"], "ratio_basis": "Interaction", "test_accessed": False, "final_test_accessed": False, "processed_test_split_read": False}
    for key, value in required.items():
        if manifest.get(key) != value: raise ValueError(f"2% experiment manifest mismatch: {key}")
    expected_counts = {"train": 60000, "forget": ratio["forget_samples"], "retain": ratio["retain_samples"], "forget_users": ratio["forget_users"], "unique_train_users": 1025, "development": 20000}
    for key, value in expected_counts.items():
        if manifest.get("counts", {}).get(key) != value: raise ValueError(f"2% experiment count mismatch: {key}")
    relative_method = str(method_config_path.resolve().relative_to(root.resolve())).replace("\\", "/")
    if manifest.get("generated_configs", {}).get("IF-A2") != relative_method or manifest.get("config_sha256", {}).get("IF-A2") != sha256_file(method_config_path): raise ValueError("2% IF-A2 generated-config authority mismatch")
    if not isinstance(deletion, dict) or deletion.get("contract_sha256") != ratio["contract_sha256"] or deletion.get("experiment_name") != ratio["experiment_name"] or deletion.get("requested_forget_ratio") != .02 or deletion.get("actual_interaction_ratio") != ratio["actual_interaction_ratio"] or deletion.get("ratio_basis") != "Interaction": raise ValueError("2% IF-A2 deletion contract mismatch")
    protocol = root / method_config["protocol_root"]
    for key in ("train", "forget", "retain", "development"):
        data_path = protocol / manifest.get("paths", {}).get(key, "")
        if not data_path.is_file() or sha256_file(data_path) != manifest.get("sha256", {}).get(key): raise ValueError(f"2% manifest data SHA mismatch: {key}")
        if key != "train" and manifest["sha256"][key] != method_config.get(f"{key}_sha256"): raise ValueError(f"2% manifest/method data SHA mismatch: {key}")
    return {"manifest_sha256": sha256_file(manifest_path), "completion_marker_sha256": sha256_file(marker_path), "experiment_contract_sha256": ratio["contract_sha256"], "selected_user_order_sha256": manifest["selected_user_order_sha256"], "selected_user_ids": manifest["selected_user_ids"], "counts": manifest["counts"], "requested_forget_ratio": manifest["requested_forget_ratio"], "actual_interaction_ratio": manifest["actual_interaction_ratio"], "actual_user_ratio": manifest["actual_user_ratio"]}


def publish_audit(stage: Path, destination: Path, report: dict[str, Any], implementation_sha256: str) -> None:
    status = "COMPLETED" if report["passed"] else "FAILED"
    atomic_json(stage / "group_a_gradient_audit.json", report)
    atomic_json(stage / "run_state.json", {"schema": SCHEMA, "status": status, "failed_gates": [key for key, value in report["gates"].items() if not value], "zero_update": True, "optimizer_constructed": False, "test_accessed": False})
    atomic_json(stage / "manifest.json", {"schema": SCHEMA, "status": status, "result_sha256": sha256_file(stage / "group_a_gradient_audit.json"), "run_state_sha256": sha256_file(stage / "run_state.json"), "implementation_sha256": implementation_sha256, "published_atomically": True, "zero_update": True, "optimizer_constructed": False, "test_accessed": False})
    marker = "COMPLETED" if report["passed"] else "FAILED"
    marker_text = MARKER if report["passed"] else "IF_A2_GROUP_A_GRADIENT_AUDIT_FAILED"
    (stage / marker).write_text(marker_text + "\n", encoding="utf-8", newline="\n")
    destination.parent.mkdir(parents=True, exist_ok=True); os.replace(stage, destination)


def run(root: Path, config_path: Path, run_name: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")); _validate_config(config); run_name = safe_run_name(run_name); destination = (root / config["output_root"] / run_name).resolve()
    if destination.exists(): raise FileExistsError(destination)
    git = git_snapshot(root)
    if not git["clean"]: raise RuntimeError("formal Group-A audit requires clean Git")
    method_config_path = root / config["method_config"]; method_config = load_method_config(method_config_path); ratio_authority = validate_ratio_authority(root, config, method_config_path, method_config); base = load_config(root / method_config["base_config"], root); checkpoint = root / method_config["original"]
    for key in ("original", "forget", "retain", "development"):
        path = checkpoint if key == "original" else root / method_config[key]
        if sha256_file(path) != method_config[f"{key}_sha256"]: raise ValueError(f"{key} authority SHA mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1: raise RuntimeError("Group A requires exactly one CUDA GPU")
    device = torch.device("cuda:0"); free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes / GIB < config["runtime"]["minimum_free_gib_before_load"]: raise RuntimeError(f"insufficient clean-GPU free VRAM: {free_bytes/GIB:.2f} GiB")
    work_parent = destination.parent / ".work"; work_parent.mkdir(parents=True, exist_ok=True); stage = work_parent / f"{run_name}.{uuid.uuid4().hex}.stage"; stage.mkdir()
    model = None; checkpoint_before = sha256_file(checkpoint); started = time.perf_counter()
    try:
        torch.cuda.set_per_process_memory_fraction(config["runtime"]["allocator_fraction"], device); torch.cuda.reset_peak_memory_stats(); model = load_legacy_model(checkpoint).to(device).eval(); tokenizer = T5Tokenizer.from_pretrained(base["paths"]["model_dir"]); pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None: raise RuntimeError("corrected masking protocol requires tokenizer pad_token_id")
        forget = JsonPromptDataset(root / method_config["forget"], tokenizer); retain = JsonPromptDataset(root / method_config["retain"], tokenizer)
        lineage, split_indices, users = _method_lineage(root, base, method_config); forget_indices = split_indices["forget_train"]; retain_indices = split_indices["retain_train"]
        if len(forget) != ratio_authority["counts"]["forget"] or len(retain) != ratio_authority["counts"]["retain"] or len(set(users["forget_train"])) != ratio_authority["counts"]["forget_users"]: raise RuntimeError("2% runtime lineage/count mismatch")
        a2 = yaml.safe_load((root / method_config["if_a2_config"]).read_text(encoding="utf-8")); panels = select_retain_panels(users["retain_train"], a2["panels"]); primary = panels["primary"]["indices"]
        modules = collect_qv_modules(model); module_names = [name for name, _ in modules]; weights = [module.weight for _, module in modules]; official = list(model.parameters()); buffers = list(model.buffers()); official_before = tensor_tree_hash({str(i): p.detach() for i, p in enumerate(official)}); buffers_before = tensor_tree_hash({str(i): p.detach() for i, p in enumerate(buffers)}); rng_before = tensor_tree_hash({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()})
        for parameter in model.parameters(): parameter.requires_grad_(False)
        for weight in weights: weight.requires_grad_(True)
        matrices, basis_source = stream_weight_gradients(model, forget, forget_indices, weights, device, config["runtime"]["production_batch_size"], pad_token_id); bases = {name: build_fixed_a_basis(matrix, rank=method_config["coordinate"]["rank"], name=name, seed=42)[0] for name, matrix in zip(module_names, matrices)}
        for weight in weights: weight.requires_grad_(False)
        parameter_names, parameters = install_fixed_ab_coordinate(model, bases, method_config["coordinate"]["alpha"]); coordinate_names = [name[:-2] if name.endswith(".B") else name for name in parameter_names]
        if coordinate_names != module_names or any(torch.count_nonzero(parameter).item() for parameter in parameters): raise RuntimeError("unexpected fixed-A/B coordinate")
        vectors, aggregation = stream_reductions(model, forget, forget_indices, parameters, device, config["runtime"]["production_batch_size"], pad_token_id); current_parts, current_meta = stream_weight_gradients(model, forget, forget_indices, parameters, device, config["runtime"]["production_batch_size"], pad_token_id); vectors["current"] = flatten_tensors(current_parts)
        comparison = {}; shapes = [parameter.shape for parameter in parameters]
        for batch_size in config["runtime"]["comparison_batch_sizes"]:
            parts, meta = stream_weight_gradients(model, forget, forget_indices, parameters, device, batch_size, pad_token_id); value = flatten_tensors(parts); comparison[str(batch_size)] = {**gradient_comparison(vectors["current"], value, coordinate_names, shapes, config["tolerances"]["gradient_relative_error"]), "metadata": meta}
        repeat_parts, repeat_meta = stream_weight_gradients(model, forget, forget_indices, parameters, device, config["runtime"]["production_batch_size"], pad_token_id); repeat_value = flatten_tensors(repeat_parts); repeat = {**gradient_comparison(vectors["current"], repeat_value, coordinate_names, shapes, 0.0), "metadata": repeat_meta, "purpose": "same_partition_repeat_determinism"}; repeat["passed"] = repeat["relative_error_vs_production"] == 0.0
        identities = reduction_identity_report(vectors, len(forget_indices), aggregation["valid_tokens"], config["tolerances"]["gradient_relative_error"]); current_vs_sample = scalar_equivalence(vectors["current"], vectors["sample_mean"]); current_vs_sample["scalar_equivalent"] = current_vs_sample["relative_residual"] is not None and current_vs_sample["relative_residual"] <= config["tolerances"]["scalar_equivalence_relative_error"]
        mapping = coordinate_mapping_report(matrices, module_names, bases, vectors["current"], method_config["coordinate"]["alpha"], method_config["coordinate"]["rank"], config["tolerances"]["gradient_relative_error"])
        sign = finite_difference_sign(model, forget, forget_indices, users["forget_train"], parameters, coordinate_names, bases, modules, device, config["sign_audit"]["relative_effective_weight_epsilons"], config["tolerances"]["finite_difference_relative_error"], method_config["coordinate"]["alpha"], method_config["coordinate"]["rank"], pad_token_id)
        retain_tokens = count_valid_tokens(retain, retain_indices, 256); primary_tokens = count_valid_tokens(retain, primary, 256); ledger = derive_unit_ledger(samples_full=len(forget_indices)+len(retain_indices), samples_forget=len(forget_indices), samples_retain=len(retain_indices), tokens_forget=aggregation["valid_tokens"], tokens_retain=retain_tokens, h_panel_samples=len(primary), h_panel_tokens=primary_tokens, current_vs_sample=current_vs_sample)
        peak_allocated = torch.cuda.max_memory_allocated()/GIB; peak_reserved = torch.cuda.max_memory_reserved()/GIB
        integrity = {"checkpoint_sha256_before": checkpoint_before, "checkpoint_sha256_after": sha256_file(checkpoint), "checkpoint_unchanged": checkpoint_before == sha256_file(checkpoint), "official_parameters_unchanged": official_before == tensor_tree_hash({str(i): p.detach() for i, p in enumerate(official)}), "buffers_unchanged": buffers_before == tensor_tree_hash({str(i): p.detach() for i, p in enumerate(buffers)}), "B_zero_after_sign_audit": all(torch.count_nonzero(parameter).item() == 0 for parameter in parameters), "autograd_grad_did_not_populate_parameter_grad": all(parameter.grad is None for parameter in official + list(parameters)), "rng_unchanged": rng_before == tensor_tree_hash({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}), "model_remained_eval": model.training is False}
        masking_evidence = {"pad_token_id": int(pad_token_id), "definition": "input_ids_ne_pad_token_id", "basis_forward_masked": basis_source.get("encoder_attention_mask") == "input_ids_ne_pad_token_id", "reduction_forward_masked": aggregation.get("encoder_attention_mask") == "input_ids_ne_pad_token_id", "current_forward_masked": current_meta.get("encoder_attention_mask") == "input_ids_ne_pad_token_id", "comparison_forwards_masked": all(row["metadata"].get("encoder_attention_mask") == "input_ids_ne_pad_token_id" for row in comparison.values()), "finite_difference_forward_masked": sign.get("encoder_attention_mask") == "input_ids_ne_pad_token_id", "observed_masked_padding_tokens": int(current_meta.get("masked_padding_tokens", 0)), "legacy_unmasked_equivalence_claimed": False}
        masking_passed = all(masking_evidence[key] for key in ("basis_forward_masked", "reduction_forward_masked", "current_forward_masked", "comparison_forwards_masked", "finite_difference_forward_masked")) and masking_evidence["observed_masked_padding_tokens"] > 0
        gates = {"ratio_authority": lineage.get("experiment_contract_sha256") == ratio_authority["experiment_contract_sha256"], "corrected_masking": masking_passed, "reduction_identities": identities["all_passed"], "same_partition_determinism": repeat["passed"], "microbatch_invariance": all(row["passed"] for row in comparison.values()), "analytic_coordinate_mapping": mapping["passed"], "finite_difference_sign": sign["passed"] and sign["distinct_users"], "integrity": all(integrity[key] for key in ("checkpoint_unchanged", "official_parameters_unchanged", "buffers_unchanged", "B_zero_after_sign_audit", "autograd_grad_did_not_populate_parameter_grad", "rng_unchanged", "model_remained_eval")), "memory": peak_reserved <= config["runtime"]["hard_cap_reserved_gib"]}
        implementation_files = [Path(__file__), config_path, method_config_path]; implementation = {str(path.resolve().relative_to(root.resolve())).replace("\\", "/"): sha256_file(path) for path in implementation_files}; provenance = {"git": git, "implementation_files": implementation, "implementation_sha256": canonical_hash(implementation), "hardware": hardware_snapshot(), "source_checkpoint_sha256": checkpoint_before, "forget_sha256": method_config["forget_sha256"], "retain_sha256": method_config["retain_sha256"], "ratio_authority": ratio_authority, "lineage": lineage, "input_masking": config["input_masking"], "runtime_protocol": "corrected_masking_not_legacy_exact", "legacy_unmasked_equivalence_claimed": False, "development_loaded": False, "final_test_loaded": False, "test_accessed": False}
        report = {"schema": SCHEMA, "run_name": run_name, "audit_target": config["audit_target"], "runtime_protocol": "corrected_masking_not_legacy_exact", "legacy_unmasked_equivalence_claimed": False, "input_masking": config["input_masking"], "masking_evidence": masking_evidence, "zero_update": True, "optimizer_constructed": False, "hvp_constructed": False, "cg_constructed": False, "update_space_projection_constructed": False, "test_loader_built": False, "test_accessed": False, "config_sha256": sha256_file(config_path), "method_config_sha256": sha256_file(root / config["method_config"]), "aggregation": aggregation, "corrected_production_aggregation": current_meta, "identities": identities, "same_partition_repeat": repeat, "microbatch_invariance": comparison, **vector_report(vectors, coordinate_names, shapes), "coordinate_mapping": mapping, "sign_audit": sign, "unit_ledger": ledger, "update_space_projection": {"status": "not_applicable_in_Group_A", "reason": "Group A forbids HVP, CG and candidate projection; fixed-A coordinate projection is reported separately"}, "integrity": integrity, "memory": {"device_total_gib": total_bytes/GIB, "free_before_load_gib": free_bytes/GIB, "allocator_fraction": config["runtime"]["allocator_fraction"], "hard_cap_reserved_gib": config["runtime"]["hard_cap_reserved_gib"], "peak_allocated_gib": peak_allocated, "peak_reserved_gib": peak_reserved}, "basis_source": basis_source, "provenance": provenance, "gates": gates, "passed": all(gates.values()), "wall_time_seconds": time.perf_counter() - started}
        publish_audit(stage, destination, report, provenance["implementation_sha256"])
        if not report["passed"]: raise RuntimeError(f"Group-A audit gates failed; FAILED evidence published at {destination}: {[key for key,value in gates.items() if not value]}")
        return report
    finally:
        if model is not None: del model
        gc.collect(); torch.cuda.empty_cache()


def synthetic() -> dict[str, Any]:
    logits = torch.tensor([[[2., 0.], [0., 2.]], [[1., 0.], [3., 0.]]], requires_grad=True); labels = torch.tensor([[0, -100], [1, 0]]); token_sum, sample_sum, tokens = token_and_sample_numerators(logits, labels)
    sample_sum_value = float(sample_sum.detach()); token_sum_value = float(token_sum.detach())
    ledger = derive_unit_ledger(samples_full=10, samples_forget=2, samples_retain=8, tokens_forget=3, tokens_retain=12, h_panel_samples=4, h_panel_tokens=6, current_vs_sample={"coefficient": 1.0, "relative_residual": 0.0, "scalar_equivalent": True})
    return {"tokens": tokens, "token_sum": token_sum_value, "sample_sum": sample_sum_value, "sample_and_token_weighting_differ": token_sum_value != sample_sum_value, "token_risk_scale": ledger["group_B_required_derived_scale"], "zero_update": True, "optimizer_constructed": False, "test_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path.cwd()); parser.add_argument("--config", type=Path); parser.add_argument("--run-name"); parser.add_argument("--synthetic", action="store_true"); args = parser.parse_args()
    if args.synthetic: value = synthetic()
    elif not args.config or not args.run_name: raise ValueError("formal Group-A audit requires --config and --run-name")
    else: value = run(args.root.resolve(), args.config.resolve(), args.run_name)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__": main()
