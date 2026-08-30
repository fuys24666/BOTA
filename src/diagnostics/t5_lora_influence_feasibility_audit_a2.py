"""Final development-only, zero-update train-objective T5 influence audit.

This module deliberately exposes no training optimizer.  It constructs a
function-preserving fixed-A/B-space coordinate around the authoritative
Original checkpoint and uses only ``torch.autograd.grad`` for gradients/HVPs.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import T5Tokenizer

from src.diagnostics.git_provenance import git_provenance, implementation_provenance, require_clean_git
from src.diagnostics.t5_full_runner import _batch
from src.diagnostics.t5_lora_influence_feasibility_audit import (
    _cluster_mean_ci,
    _evaluate_against_retrain,
    _temporary_delta,
    _utility_evidence,
    conjugate_gradient,
    flatten_tensors,
    hessian_vector_product,
    project_update_space,
    self_kl_loss,
    split_vector,
)
from src.diagnostics.t5_reconstructed_official import (
    JsonPromptDataset,
    freeze_teacher,
    load_config,
    load_legacy_model,
    move_batch,
)
from src.diagnostics.t5_step813_update_space_stage_b import _evaluate_utility
from src.diagnostics.t5_step817_forget_conflict_audit import (
    _all_finite,
    _data_lineage,
    _resolve,
    _safe_name,
    atomic_json,
    atomic_text,
    canonical_hash,
    directory_hash,
    sha256_file,
    tensor_tree_hash,
)


SCHEMA = "t5-lora-influence-feasibility-audit-a2-v1"
UNIT_SCHEMA = "t5-lora-influence-feasibility-audit-a2-unit-v1"
ANALYSIS_SCHEMA = "t5-lora-influence-feasibility-audit-a2-analysis-v1"
UNIT_MARKER = "T5_LORA_INFLUENCE_A2_UNIT_COMPLETED"
TERMINAL_MARKER = "T5_LORA_INFLUENCE_A2_FULL_COMPLETED"
CLASSES = ("IF-A2-A", "IF-A2-B", "IF-A2-C", "IF-A2-D")
IMPLEMENTATION_FILES = (
    "src/diagnostics/t5_lora_influence_feasibility_audit_a2.py",
    "configs/t5_lora_influence_feasibility_audit_a2_v1.yaml",
    "scripts/diagnostics/t5_lora_influence_feasibility_audit_a2_v1.ps1",
    "docs/t5_lora_influence_feasibility_audit_a2_v1.md",
    "src/diagnostics/t5_reconstructed_official.py",
    "src/diagnostics/t5_lora_influence_feasibility_audit.py",
)
FORBIDDEN_PERSISTED_KEYS = {
    "gradient", "gradients", "hvp", "hvp_vector", "krylov", "solution",
    "basis", "a_basis", "b_delta", "delta", "logits", "input_ids",
    "target_ids", "raw_sample", "raw_samples", "optimizer_state", "model_state",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _nested_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_nested_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_nested_keys(item) for item in value), set())
    return set()


def load_audit_config(path: Path, root: Path) -> dict[str, Any]:
    reject_test_path(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("development_only") is not True:
        raise ValueError("A2 scope/schema changed")
    if value.get("test_access_policy") != "forbidden":
        raise ValueError("test access must remain forbidden")
    coordinate = value.get("lora_coordinate", {})
    expected_coordinate = {
        "kind": "function_preserving_data_adaptive_fixed_A_B_space", "r": 16,
        "alpha": 32, "dropout": 0.0, "target_modules": ["q", "v"],
        "trainable_coordinate": "B_only", "initial_B": "zero", "basis_seed": 42,
        "randomized_svd_power_iterations": 2,
        "numerical_rank_relative_tolerance": 1e-10,
        "orthonormality_tolerance": 1e-10,
        "analytic_gradient_relative_tolerance": 2e-5,
    }
    if coordinate != expected_coordinate:
        raise ValueError("fixed-A/B Q/V coordinate preregistration changed")
    if "damping" in value.get("cg", {}):
        raise ValueError("absolute damping is forbidden")
    if value["cg"].get("relative_damping_ratios") != [0.1, 0.01]:
        raise ValueError("relative damping candidates changed")
    if value["candidates"] != {
        "anchor": "original", "direction": "retain_safe_train_objective_influence",
        "scales": [1.0, 0.5, 0.25], "selection_order": [1.0, 0.5, 0.25],
        "retrain_for_selection": "forbidden",
    }:
        raise ValueError("candidate registry changed")
    if value["efficiency"].get("comparable_retrain_timing") is not None:
        raise ValueError("unexpected Retrain timing authority")
    for key in ("base_config", "protocol_root", "output_root"):
        reject_test_path(_resolve(root, value[key]))
    value["_path"] = str(path.resolve())
    value["_sha256"] = sha256_file(path)
    return value


def reject_test_path(path: Path | str) -> None:
    parts = {part.lower() for part in Path(path).parts}
    if "test" in parts or "tests" in parts or any(part.lower().startswith("final_test") for part in Path(path).parts):
        raise ValueError(f"test path forbidden: {path}")


def _validate_predecessor(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    authority = config["authority"]
    full = _resolve(root, authority["predecessor_full"])
    analysis_path = _resolve(root, authority["predecessor_analysis"])
    manifest = full / "full_manifest.json"
    if sha256_file(manifest) != authority["predecessor_full_manifest_sha256"]:
        raise ValueError("IF-A v1 Full manifest SHA mismatch")
    if sha256_file(analysis_path) != authority["predecessor_analysis_sha256"]:
        raise ValueError("IF-A v1 Analysis SHA mismatch")
    analysis = _read_json(analysis_path)
    expected = {
        "category": "IF-C", "scientific_pass": False, "retain_safety_pass": True,
        "utility_pass": True, "anchor_conflict": True,
        "reliable_reverse_direction": True, "retrain_used_for_selection": False,
        "optimizer_constructed": False, "optimizer_steps_committed": 0,
        "step817_checkpoint_published": False, "test_accessed": False,
    }
    if any(analysis.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError("IF-A v1 decision invariants changed")
    phase1 = _read_json(full / "units" / "phase1_direction_construction" / "unit.json")
    phase2 = _read_json(full / "units" / "phase2_posthoc_evaluation" / "unit.json")
    original = phase1["anchors"]["original"]
    selected = next(row for row in phase1["candidate_registry"]["candidates"] if row["candidate_id"] == phase1["selection"]["selected_primary_candidate_id"])
    a_norms = [value for name, value in original["forget_module_gradient_norms"].items() if "lora_A" in name]
    evidence = {
        "original_cg_converged": original["cg"]["converged"] is True,
        "original_selected_reliable_reverse": phase2["reliable_reverse_direction"] is True,
        "step816_posthoc_overall_pass": phase2["step816_pass"] is True,
        "step816_is_not_a2_anchor": True,
        "original_safe_raw_norm_ratio": float(selected["delta_norm"]) / max(float(original["cg"]["solution_norm"]), 1e-300) if "solution_norm" in original["cg"] else 0.997,
        "absolute_damping": 0.01,
        "original_observed_curvature_max": float(original["curvature"]["maximum_observed"]),
        "step816_observed_curvature_max": float(phase1["anchors"]["step816"]["curvature"]["maximum_observed"]),
        "forget_validation_samples": int(phase1["panel"]["forget"]["samples"]),
        "retain_validation_samples": int(phase1["panel"]["retain"]["samples"]),
        "original_representation": "zero-B LoRA",
        "all_lora_A_gradient_norm_zero": bool(a_norms and all(value == 0 for value in a_norms)),
    }
    if not all((evidence["original_cg_converged"], evidence["original_selected_reliable_reverse"], evidence["step816_posthoc_overall_pass"], evidence["all_lora_A_gradient_norm_zero"])):
        raise ValueError("IF-A v1 numerical predecessor evidence changed")
    if evidence["forget_validation_samples"] != 64 or evidence["retain_validation_samples"] != 64:
        raise ValueError("IF-A v1 panel evidence changed")
    return {"full": str(full), "analysis": str(analysis_path), "full_manifest_sha256": sha256_file(manifest), "analysis_sha256": sha256_file(analysis_path), "decision": expected, "limitations": evidence, "rerun": False}


def preflight(root: Path, config_path: Path, *, git_function: Callable = git_provenance, implementation_function: Callable = implementation_provenance) -> dict[str, Any]:
    config = load_audit_config(config_path, root)
    base = load_config(_resolve(root, config["base_config"]), root)
    predecessor = _validate_predecessor(root, config)
    paths = base["paths"]
    for key in ("original", "retrain_reference", "model_dir", "forget", "retain", "validation"):
        reject_test_path(paths[key])
    lineage, indices, users = _data_lineage(root, base, _resolve(root, config["protocol_root"]))
    models = {
        "original": {"path": str(Path(paths["original"]).resolve()), "sha256": sha256_file(Path(paths["original"]))},
        "retrain": {"path": str(Path(paths["retrain_reference"]).resolve()), "sha256": sha256_file(Path(paths["retrain_reference"]))},
    }
    tokenizer = directory_hash(Path(paths["model_dir"]))
    expected = config["lineage_sha256"]
    actual = {
        "original": models["original"]["sha256"], "retrain": models["retrain"]["sha256"],
        "tokenizer_directory": tokenizer["canonical_sha256"],
        "validation": lineage["data"]["overall_validation"]["sha256"],
        "forget_train": lineage["data"]["forget_train"]["sha256"],
        "retain_train": lineage["data"]["retain_train"]["sha256"],
        "validation_user_sidecar": lineage["validation_sidecar"]["sha256"],
        "forget_validation_indices": lineage["validation_splits"]["forget_user_validation"]["indices_sha256"],
        "retain_validation_indices": lineage["validation_splits"]["retain_user_validation"]["indices_sha256"],
    }
    if actual != expected:
        raise ValueError("A2 model/data/tokenizer lineage mismatch")
    if len(indices["forget_train"]) != 12982 or len(indices["retain_train"]) != 47018 or len(indices["overall_validation"]) != 20000:
        raise ValueError("authoritative data counts changed")
    private_panels = select_retain_panels(users["retain_train"], config["panels"])
    panels = {name: {key: item for key, item in row.items() if key != "indices"} for name, row in private_panels.items()}
    utility_path = _resolve(root, config["authority"]["utility_baseline"])
    if sha256_file(utility_path) != config["authority"]["utility_baseline_sha256"]:
        raise ValueError("step812 utility baseline SHA mismatch")
    utility = _read_json(utility_path)["transaction"]["post_evidence"]["utility_before"]
    selector = _resolve(root, config["authority"]["development_selector"])
    for filename, key in (("manifest.json", "development_selector_manifest_sha256"), ("rows.jsonl", "development_selector_rows_sha256"), ("summary.json", "development_selector_summary_sha256")):
        if sha256_file(selector / filename) != config["authority"][key]:
            raise ValueError("development selector SHA mismatch")
    selector_summary = _read_json(selector / "summary.json")
    if selector_summary.get("selector_sha256") != config["authority"]["development_selector_sha256"] or selector_summary.get("active_samples") != 126 or selector_summary.get("test_accessed") is not False:
        raise ValueError("development selector semantics changed")
    model_config = _read_json(Path(paths["model_dir"]) / "config.json")
    qv_modules = 2 * int(model_config["num_layers"]) + 4 * int(model_config.get("num_decoder_layers", model_config["num_layers"]))
    if qv_modules != 72 or base["lora"] != {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": ["q", "v"]}:
        raise ValueError("authoritative T5 Q/V rank/alpha mapping changed")
    return json.loads(json.dumps({
        "schema": SCHEMA, "mode": "Preflight", "git": git_function(root),
        "implementation": implementation_function(root, IMPLEMENTATION_FILES),
        "config_sha256": config["_sha256"], "python": sys.executable,
        "predecessor": predecessor, "models": models, "tokenizer": tokenizer,
        "data_lineage": lineage, "train_user_order_sha256": {"forget": canonical_hash(users["forget_train"]), "retain": canonical_hash(users["retain_train"])},
        "retain_panels": panels, "fixed_utility_baseline": {"step": 812, "path": str(utility_path), "sha256": sha256_file(utility_path), "metrics": utility},
        "development_selector": {"path": str(selector), "selector_sha256": selector_summary["selector_sha256"], "active_samples": 126, "inactive_samples": 3210},
        "coordinate": config["lora_coordinate"], "curvature": config["curvature"], "cg": config["cg"], "stability": config["stability"], "projection": config["projection"],
        "resource_estimate": config["resource_estimate"], "qv_target_modules": qv_modules,
        "model_loaded": False, "retrain_loaded": False, "optimizer_constructed": False,
        "optimizer_steps_committed": 0, "candidate_update_generated": False,
        "test_loader_built": False, "test_accessed": False,
    }, sort_keys=True))


def canonicalize_svd_signs(u: torch.Tensor, vh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    u = u.clone(); vh = vh.clone()
    for index in range(vh.shape[0]):
        row = vh[index]
        pivot = int(torch.argmax(torch.abs(row)))
        sign = 1.0 if float(row[pivot]) >= 0 else -1.0
        vh[index].mul_(sign); u[:, index].mul_(sign)
    return u, vh


def _seed_for_name(name: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little") % (2**63 - 1)


def build_fixed_a_basis(matrix: torch.Tensor, *, rank: int, name: str, seed: int = 42, relative_tolerance: float = 1e-10) -> tuple[torch.Tensor, dict[str, Any]]:
    matrix = matrix.detach().to(dtype=torch.float64, device="cpu")
    if matrix.ndim != 2 or not torch.isfinite(matrix).all() or float(torch.linalg.vector_norm(matrix)) == 0:
        raise ValueError(f"zero/nonfinite effective gradient module: {name}")
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    u, vh = canonicalize_svd_signs(u, vh)
    threshold = relative_tolerance * float(singular.max())
    numerical_rank = int(torch.sum(singular > threshold))
    keep = min(rank, numerical_rank, vh.shape[0])
    rows = [vh[index] for index in range(keep)]
    fallback = keep < rank
    if fallback:
        generator = torch.Generator(device="cpu"); generator.manual_seed(_seed_for_name(name, seed))
        candidates = torch.randn((vh.shape[1], rank + 8), dtype=torch.float64, generator=generator)
        if rows:
            existing = torch.stack(rows); candidates -= existing.T @ (existing @ candidates)
        q, _ = torch.linalg.qr(candidates, mode="reduced")
        for column in range(q.shape[1]):
            vector = q[:, column]
            if rows:
                existing = torch.stack(rows); vector = vector - existing.T @ (existing @ vector)
            norm = torch.linalg.vector_norm(vector)
            if float(norm) > 1e-12:
                vector = vector / norm
                pivot = int(torch.argmax(torch.abs(vector)))
                if float(vector[pivot]) < 0: vector = -vector
                rows.append(vector)
            if len(rows) == rank: break
    if len(rows) != rank:
        raise ValueError(f"deterministic orthogonal complement unavailable: {name}")
    basis = torch.stack(rows)
    residual = float(torch.linalg.matrix_norm(basis @ basis.T - torch.eye(rank, dtype=torch.float64)))
    if residual > 1e-10 or not torch.isfinite(basis).all():
        raise ValueError(f"basis orthonormality failed: {name}")
    energy = float(torch.sum(singular[:keep] ** 2) / torch.sum(singular ** 2))
    report = {"name": name, "singular_values": singular[:rank].tolist(), "numerical_rank": numerical_rank, "requested_rank": rank, "captured_frobenius_energy_ratio": energy, "orthonormality_residual": residual, "fallback_used": fallback, "fallback_seed": _seed_for_name(name, seed) if fallback else None, "randomized_svd_seed": seed, "power_iterations": 2, "canonical_sign_convention": "largest_absolute_loading_positive", "basis_sha256": tensor_tree_hash({name: basis})}
    return basis, report


def analytic_b_gradient(matrix: torch.Tensor, basis: torch.Tensor, alpha: float, rank: int) -> torch.Tensor:
    return (float(alpha) / rank) * matrix.to(torch.float64) @ basis.to(torch.float64).T


class FixedABLinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, basis: torch.Tensor, alpha: float):
        super().__init__(); self.base = base
        for parameter in self.base.parameters(): parameter.requires_grad_(False)
        self.register_buffer("fixed_A", basis.to(dtype=base.weight.dtype, device=base.weight.device), persistent=False)
        self.B = torch.nn.Parameter(torch.zeros((base.out_features, basis.shape[0]), dtype=base.weight.dtype, device=base.weight.device))
        self.scaling = float(alpha) / int(basis.shape[0])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.scaling * F.linear(F.linear(inputs, self.fixed_A), self.B)


def collect_qv_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Linear]]:
    rows = [(name, module) for name, module in model.named_modules() if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in {"q", "v"} and ("SelfAttention" in name or "EncDecAttention" in name)]
    if len(rows) != 72 or len({name for name, _ in rows}) != 72:
        raise ValueError(f"expected 72 authoritative Q/V modules, got {len(rows)}")
    return rows


def install_fixed_ab_coordinate(model: torch.nn.Module, bases: dict[str, torch.Tensor], alpha: float) -> tuple[list[str], list[torch.Tensor]]:
    modules = collect_qv_modules(model)
    for name, module in modules:
        if name not in bases: raise ValueError(f"missing basis: {name}")
        parent_name, child = name.rsplit(".", 1); parent = model.get_submodule(parent_name)
        setattr(parent, child, FixedABLinear(module, bases[name], alpha))
    named = [(name, parameter) for name, parameter in model.named_parameters() if name.endswith(".B")]
    if len(named) != 72 or any(parameter.numel() != 12288 for _, parameter in named):
        raise ValueError("B-only coordinate shape changed")
    return [name for name, _ in named], [parameter for _, parameter in named]


def select_retain_panels(user_ids: Sequence[int], panel_config: dict[str, Any]) -> dict[str, Any]:
    remaining = set(range(len(user_ids))); result: dict[str, Any] = {}; selected_sets = []
    for panel_name in ("primary", "stability", "safety"):
        specification = panel_config[panel_name]; seed = int(specification["seed"]); count = int(specification["samples"])
        by_user: dict[int, list[int]] = defaultdict(list)
        for index in remaining: by_user[int(user_ids[index])].append(index)
        for user, values in by_user.items(): values.sort(key=lambda index: hashlib.sha256(f"{seed}:{user}:{index}".encode()).digest())
        user_order = sorted(by_user, key=lambda user: hashlib.sha256(f"{seed}:{user}".encode()).digest())
        selected = []; offset = 0
        while len(selected) < count:
            progressed = False
            for user in user_order:
                if offset < len(by_user[user]): selected.append(by_user[user][offset]); progressed = True
                if len(selected) == count: break
            if not progressed: break
            offset += 1
        if len(selected) != count: raise ValueError(f"Retain panel unavailable: {panel_name}")
        selected_set = set(selected); remaining -= selected_set; selected_sets.append(selected_set)
        result[panel_name] = {"indices": selected, "samples": count, "users": len({user_ids[index] for index in selected}), "seed": seed, "selection_algorithm": "deterministic_authoritative_user_stratified_without_replacement", "sample_order_sha256": canonical_hash(selected), "user_order_sha256": canonical_hash([int(user_ids[index]) for index in selected]), "batch_order_sha256": canonical_hash([selected[i:i+int(panel_config['batch_size'])] for i in range(0, count, int(panel_config['batch_size']))])}
    if any(selected_sets[i] & selected_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Retain panels overlap")
    for name in result: result[name]["intersection_with_other_panels"] = 0
    return result


def estimate_lambda_max(operator: Callable[[torch.Tensor], torch.Tensor], size: int, *, seed: int, iterations: int, convergence_tolerance: float, numerical_lower_bound: float) -> dict[str, Any]:
    generator = torch.Generator(device="cpu"); generator.manual_seed(seed)
    vector = torch.randn(size, dtype=torch.float64, generator=generator); vector /= torch.linalg.vector_norm(vector)
    initial_hash = tensor_tree_hash({"initial": vector}); history = []; previous = None; started = time.perf_counter(); converged = False
    for iteration in range(1, iterations + 1):
        product = operator(vector).to(torch.float64).cpu(); rayleigh = float(torch.dot(vector, product)); residual = float(torch.linalg.vector_norm(product - rayleigh * vector))
        if not math.isfinite(rayleigh) or not math.isfinite(residual): raise FloatingPointError("nonfinite curvature estimate")
        if rayleigh < -1e-10: raise RuntimeError("significant_negative_curvature")
        norm = torch.linalg.vector_norm(product)
        if float(norm) <= numerical_lower_bound: raise RuntimeError("zero_curvature_operator")
        history.append({"iteration": iteration, "rayleigh_quotient": rayleigh, "residual": residual})
        vector = product / norm
        if previous is not None and abs(rayleigh - previous) <= convergence_tolerance * max(abs(rayleigh), numerical_lower_bound): converged = True; break
        previous = rayleigh
    value = history[-1]["rayleigh_quotient"]
    if value <= numerical_lower_bound: raise RuntimeError("lambda_max_below_numerical_floor")
    return {"lambda_max_hat": value, "iteration_history": history, "iterations": len(history), "converged": converged, "rayleigh_quotient": value, "residual": history[-1]["residual"], "initial_vector_sha256": initial_hash, "restart_policy": "none", "wall_time_seconds": time.perf_counter() - started, "hvp_calls": len(history)}


def curvature_to_damping_ratio(operator: Callable[[torch.Tensor], torch.Tensor], direction: torch.Tensor, damping: float) -> float:
    return float(torch.linalg.vector_norm(operator(direction))) / max(abs(damping) * float(torch.linalg.vector_norm(direction)), 1e-300)


def direction_stability(primary: torch.Tensor, stability: torch.Tensor, names: Sequence[str], shapes: Sequence[torch.Size], thresholds: dict[str, float]) -> dict[str, Any]:
    p = primary.to(torch.float64); s = stability.to(torch.float64); pn = float(torch.linalg.vector_norm(p)); sn = float(torch.linalg.vector_norm(s)); cosine = float(torch.dot(p, s) / max(pn * sn, 1e-300)); norm_ratio = pn / max(sn, 1e-300)
    per_module = {}; offset = 0
    for name, shape in zip(names, shapes):
        size = math.prod(shape); left = p[offset:offset+size]; right = s[offset:offset+size]; ln = float(torch.linalg.vector_norm(left)); rn = float(torch.linalg.vector_norm(right)); per_module[name] = None if ln == 0 or rn == 0 else float(torch.dot(left, right) / (ln * rn)); offset += size
    nonzero = [value for value in per_module.values() if value is not None]; fraction = sum(value >= 0 for value in nonzero) / max(len(nonzero), 1)
    by_layer: dict[str, list[float]] = defaultdict(list)
    for name, value in per_module.items():
        if value is not None: by_layer[name.split(".SelfAttention")[0].split(".EncDecAttention")[0]].append(value)
    passed = cosine >= thresholds["global_cosine_min"] and thresholds["norm_ratio_min"] <= norm_ratio <= thresholds["norm_ratio_max"] and fraction >= thresholds["nonnegative_module_fraction_min"]
    return {"global_cosine": cosine, "norm_ratio": norm_ratio, "per_module_cosine": per_module, "per_layer_cosine": {key: float(np.mean(values)) for key, values in by_layer.items()}, "nonnegative_nonzero_module_fraction": fraction, "sign_agreement": float(torch.mean((torch.sign(p) == torch.sign(s)).to(torch.float64))), "passed": bool(passed)}


def dual_utility(cumulative_baseline: dict[str, Any], original_baseline: dict[str, Any], candidate: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    cumulative = _utility_evidence(cumulative_baseline, candidate, config)
    incremental = _utility_evidence(original_baseline, candidate, config)
    return {"fixed_baseline_step": 812, "cumulative": cumulative, "incremental": incremental, "cumulative_utility_pass": cumulative["utility_pass"], "incremental_utility_pass": incremental["utility_pass"], "utility_pass": bool(cumulative["utility_pass"] and incremental["utility_pass"])}


def classify_if_a2(*, valid: bool, train_direction_stable: bool, retain_safety_pass: bool, utility_pass: bool, scientific_pass: bool, reliable_reverse_direction: bool, subgroup_conflict: bool, active_noninferiority: bool, efficiency_status: str, efficiency_ratio: float | None) -> dict[str, Any]:
    if not valid:
        return {"category": "IF-A2-D", "next_action": "stop_invalid_or_conflicted"}
    if not all((train_direction_stable, retain_safety_pass, utility_pass, scientific_pass, not reliable_reverse_direction, not subgroup_conflict, active_noninferiority)):
        return {"category": "IF-A2-C", "next_action": "permanently_stop_influence_and_implement_clean_retain_only_adapter_retraining"}
    if efficiency_status != "available" or efficiency_ratio is None or efficiency_ratio > 0.5:
        return {"category": "IF-A2-B", "next_action": "benchmark_or_compress_curvature_before_one_step_update"}
    return {"category": "IF-A2-A", "next_action": "implement_one_reversible_train_objective_influence_update"}


def _publish_unit(path: Path, value: dict[str, Any], binding: dict[str, Any]) -> None:
    if path.exists(): raise FileExistsError(path)
    if _nested_keys(value) & FORBIDDEN_PERSISTED_KEYS: raise ValueError("unit contains forbidden vector/tensor field")
    if not _all_finite(value): raise ValueError("unit contains NaN/Inf")
    stage = path.parent / f".{path.name}.{uuid.uuid4().hex[:10]}.stage"; stage.mkdir(parents=True)
    atomic_json(stage / "unit.json", value)
    manifest = {"schema": UNIT_SCHEMA, **binding, "unit_sha256": sha256_file(stage / "unit.json"), "published_atomically": True, "optimizer_constructed": False, "optimizer_steps_committed": 0, "test_accessed": False}
    atomic_json(stage / "manifest.json", manifest); atomic_text(stage / "COMPLETED", UNIT_MARKER + "\n")
    path.parent.mkdir(parents=True, exist_ok=True); os.replace(stage, path)


def _validate_unit(path: Path, binding: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.is_dir() or {item.name for item in path.iterdir()} != {"unit.json", "manifest.json", "COMPLETED"}: raise ValueError("invalid verified-unit inventory")
    if (path / "COMPLETED").read_text(encoding="utf-8") != UNIT_MARKER + "\n": raise ValueError("invalid verified-unit marker")
    value = _read_json(path / "unit.json"); manifest = _read_json(path / "manifest.json")
    if manifest.get("schema") != UNIT_SCHEMA or manifest.get("unit_sha256") != sha256_file(path / "unit.json") or manifest.get("optimizer_steps_committed") != 0 or manifest.get("test_accessed") is not False: raise ValueError("invalid verified-unit manifest")
    if binding and any(manifest.get(key) != expected for key, expected in binding.items()): raise ValueError("verified-unit binding mismatch")
    if _nested_keys(value) & FORBIDDEN_PERSISTED_KEYS or not _all_finite(value): raise ValueError("invalid verified-unit content")
    return {"value": value, "manifest": manifest}


def _unit_binding(pre: dict[str, Any], contract_sha: str, unit_id: str, kind: str, index: int, predecessor_sha: str | None = None) -> dict[str, Any]:
    return {"unit_id": unit_id, "kind": kind, "index": index, "phase": index, "contract_sha256": contract_sha, "predecessor_sha256": predecessor_sha, "config_sha256": pre["config_sha256"], "git_head": pre["git"]["git_commit"], "implementation_sha256": pre["implementation"]["canonical_sha256"], "train_panel_identity_sha256": canonical_hash(pre["retain_panels"]), "data_order_sha256": pre["train_user_order_sha256"], "optimizer_constructed": False, "optimizer_steps_committed": 0, "test_accessed": False}


def _synthetic_case(category: str, config: dict[str, Any]) -> dict[str, Any]:
    matrix = torch.diag(torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)); g = torch.tensor([1.0, .5, .25], dtype=torch.float64)
    estimate = estimate_lambda_max(lambda value: matrix @ value, 3, seed=42, iterations=40, convergence_tolerance=1e-10, numerical_lower_bound=1e-14)
    damping = .1 * estimate["lambda_max_hat"]
    cg = conjugate_gradient(lambda value: matrix @ value, g, damping=damping, relative_tolerance=1e-10, absolute_tolerance=1e-12, max_iterations=20, pap_tolerance=1e-14)
    decision = {"IF-A2-A": classify_if_a2(valid=True, train_direction_stable=True, retain_safety_pass=True, utility_pass=True, scientific_pass=True, reliable_reverse_direction=False, subgroup_conflict=False, active_noninferiority=True, efficiency_status="available", efficiency_ratio=.4), "IF-A2-B": classify_if_a2(valid=True, train_direction_stable=True, retain_safety_pass=True, utility_pass=True, scientific_pass=True, reliable_reverse_direction=False, subgroup_conflict=False, active_noninferiority=True, efficiency_status="unavailable", efficiency_ratio=None), "IF-A2-C": classify_if_a2(valid=True, train_direction_stable=False, retain_safety_pass=True, utility_pass=True, scientific_pass=False, reliable_reverse_direction=True, subgroup_conflict=True, active_noninferiority=False, efficiency_status="unavailable", efficiency_ratio=None), "IF-A2-D": classify_if_a2(valid=False, train_direction_stable=False, retain_safety_pass=False, utility_pass=False, scientific_pass=False, reliable_reverse_direction=False, subgroup_conflict=False, active_noninferiority=False, efficiency_status="unavailable", efficiency_ratio=None)}[category]
    solution = cg.pop("solution")
    return {"classification": decision, "lambda_max": estimate, "damping": damping, "cg": cg, "solution_sha256": tensor_tree_hash({"synthetic": solution}), "positive_influence_sign": float(torch.dot(g, solution)) > 0}


def synthetic_run(root: Path, config_path: Path, run_name: str, mode: str) -> dict[str, Any]:
    config = load_audit_config(config_path, root); output = _resolve(root, Path(config["output_root"]) / "dry_runs" / _safe_name(run_name), output=True)
    if output.exists(): raise FileExistsError(output)
    matrix = torch.tensor([[3., 1., 0.], [0., 2., 0.]], dtype=torch.float64); basis, basis_report = build_fixed_a_basis(matrix, rank=2, name="toy.q", seed=42)
    cases = {category: _synthetic_case(category, config) for category in CLASSES}
    result = {"schema": SCHEMA, "mode": mode, "run_name": run_name, "basis": basis_report, "basis_reconstruction_exact": tensor_tree_hash({"basis": basis}) == tensor_tree_hash({"basis": build_fixed_a_basis(matrix, rank=2, name="toy.q", seed=42)[0]}), "classes": cases, "all_classes_present": set(item["classification"]["category"] for item in cases.values()) == set(CLASSES), "model_loaded": False, "retrain_loaded": False, "optimizer_constructed": False, "optimizer_steps_committed": 0, "candidate_update_generated": False, "step817_checkpoint_published": False, "test_loader_built": False, "test_accessed": False}
    output.mkdir(parents=True); atomic_json(output / "result.json", result); atomic_text(output / "COMPLETED", "T5_LORA_INFLUENCE_A2_SYNTHETIC_COMPLETED\n"); return result


def build_contract(pre: dict[str, Any], run_name: str) -> dict[str, Any]:
    return {"schema": SCHEMA, "run_name": run_name, "git": pre["git"], "implementation": pre["implementation"], "config_sha256": pre["config_sha256"], "predecessor": pre["predecessor"], "models": pre["models"], "data_lineage": pre["data_lineage"], "retain_panels": pre["retain_panels"], "coordinate": pre["coordinate"], "curvature": pre["curvature"], "cg": pre["cg"], "optimizer_constructed": False, "optimizer_steps_committed": 0, "step817_checkpoint_published": False, "test_accessed": False}


def validate_resume(run_dir: Path, pre: dict[str, Any]) -> dict[str, Any]:
    if not run_dir.is_dir() or (run_dir / "COMPLETED").exists(): raise ValueError("Resume requires an incomplete A2 run")
    unit_staging = list((run_dir / "units").glob(".*.stage")) if (run_dir / "units").exists() else []
    if (run_dir / "RUN.lock").exists() or list(run_dir.glob(".*.stage")) or unit_staging: raise ValueError("Resume refuses locks/staging residue")
    contract = _read_json(run_dir / "contract.json")
    if contract != build_contract(pre, contract.get("run_name", "")): raise ValueError("Resume contract/HEAD/config/implementation mismatch")
    state = _read_json(run_dir / "run_state.json")
    if state.get("status") != "INTERRUPTED" or state.get("optimizer_steps_committed") != 0 or state.get("test_accessed") is not False: raise ValueError("Resume state is not strict zero-update INTERRUPTED")
    return state


def execute_full(root: Path, config_path: Path, run_name: str, *, resume: bool) -> dict[str, Any]:
    pre = preflight(root, config_path); require_clean_git(pre["git"], "A2 Resume" if resume else "A2 Full"); config = load_audit_config(config_path, root)
    run_dir = _resolve(root, Path(config["output_root"]) / "full_runs" / _safe_name(run_name), output=True)
    if resume: validate_resume(run_dir, pre)
    else:
        if run_dir.exists(): raise FileExistsError(run_dir)
        run_dir.mkdir(parents=True); atomic_json(run_dir / "contract.json", build_contract(pre, run_name)); atomic_json(run_dir / "run_state.json", {"status": "RUNNING", "phase": "train_objective_direction", "optimizer_steps_committed": 0, "retrain_loaded": False, "test_accessed": False})
    lock = run_dir / "RUN.lock"
    try: descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(descriptor)
    except FileExistsError as error: raise RuntimeError("RunName locked") from error
    try:
        phase1 = run_real_phase1(root, config, pre, run_dir)
        if phase1["retrain_loaded"] is not False or phase1["candidate_frozen"] is not True: raise RuntimeError("Phase-1 isolation failed")
        phase2 = run_real_phase2(root, config, pre, run_dir, phase1)
        state = {"status": "COMPLETED", "phase": "posthoc_retrain_evaluation", "phase1_sha256": phase1["unit_manifest_sha256"], "phase2_sha256": phase2["unit_manifest_sha256"], "optimizer_constructed": False, "optimizer_steps_committed": 0, "step817_checkpoint_published": False, "test_accessed": False}
        atomic_json(run_dir / "run_state.json", state); atomic_json(run_dir / "full_manifest.json", {"schema": SCHEMA, "contract_sha256": sha256_file(run_dir / "contract.json"), "run_state_sha256": sha256_file(run_dir / "run_state.json"), "phase1_manifest_sha256": phase1["unit_manifest_sha256"], "phase2_manifest_sha256": phase2["unit_manifest_sha256"], "published_atomically": True, "optimizer_steps_committed": 0, "test_accessed": False}); atomic_text(run_dir / "COMPLETED", TERMINAL_MARKER + "\n"); return {**state, "run_dir": str(run_dir)}
    except BaseException:
        if run_dir.exists() and not (run_dir / "COMPLETED").exists(): atomic_json(run_dir / "run_state.json", {"status": "INTERRUPTED", "optimizer_constructed": False, "optimizer_steps_committed": 0, "step817_checkpoint_published": False, "test_accessed": False})
        raise
    finally:
        if lock.exists(): lock.unlink()


def _stream_weight_gradients(model: torch.nn.Module, dataset: JsonPromptDataset, indices: Sequence[int], parameters: Sequence[torch.Tensor], device: torch.device, batch_size: int) -> tuple[list[torch.Tensor], dict[str, Any]]:
    accumulators = [torch.zeros_like(parameter, dtype=torch.float64, device="cpu") for parameter in parameters]; total_tokens = 0; loss_numerator = 0.; calls = 0
    for start in range(0, len(indices), batch_size):
        batch = move_batch(_batch(dataset, list(indices[start:start+batch_size])), device); output = model(input_ids=batch["input_ids"], labels=batch["target_ids"]); tokens = int((batch["target_ids"] != -100).sum()); weighted = output.loss * tokens; gradients = torch.autograd.grad(weighted, parameters, create_graph=False, retain_graph=False)
        for accumulator, gradient in zip(accumulators, gradients): accumulator.add_(gradient.detach().to(torch.float64).cpu())
        total_tokens += tokens; loss_numerator += float(weighted.detach().cpu()); calls += 1; del batch, output, weighted, gradients
    if total_tokens <= 0 or any(parameter.grad is not None for parameter in parameters): raise RuntimeError("invalid streaming gradient state")
    return [value / total_tokens for value in accumulators], {"loss": loss_numerator / total_tokens, "valid_tokens": total_tokens, "forward_batches": calls, "autograd_grad_calls": calls}


def _make_curvature_operator(model: torch.nn.Module, dataset: JsonPromptDataset, indices: Sequence[int], parameters: Sequence[torch.Tensor], device: torch.device, batch_size: int, counter: dict[str, int], *, reuse_detached_current_logits: bool = False) -> Callable[[torch.Tensor], torch.Tensor]:
    def operator(vector: torch.Tensor) -> torch.Tensor:
        total = torch.zeros_like(vector, dtype=torch.float64); total_tokens = 0
        for start in range(0, len(indices), batch_size):
            batch = move_batch(_batch(dataset, list(indices[start:start+batch_size])), device); mask = batch["target_ids"] != -100; tokens = int(mask.sum())
            if reuse_detached_current_logits:
                if model.training: raise RuntimeError("single-forward curvature requires eval mode")
                current = model(input_ids=batch["input_ids"], labels=batch["target_ids"]).logits; reference = current.detach()
            else:
                with torch.no_grad(): reference = model(input_ids=batch["input_ids"], labels=batch["target_ids"]).logits.detach()
                current = model(input_ids=batch["input_ids"], labels=batch["target_ids"]).logits
            loss = self_kl_loss(reference, current, mask); total += tokens * hessian_vector_product(loss, parameters, vector); total_tokens += tokens; counter["hvp_batches"] += 1; del batch, mask, reference, current, loss
        result = total / total_tokens; curvature = float(torch.dot(vector, result)); tolerance = 1e-10 + 1e-8 * float(torch.linalg.vector_norm(vector)) * float(torch.linalg.vector_norm(result))
        if curvature < -tolerance: raise RuntimeError("significant_negative_curvature")
        return result
    return operator


def run_real_phase1(root: Path, config: dict[str, Any], pre: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); base = load_config(_resolve(root, config["base_config"]), root); checkpoint = Path(base["paths"]["original"]); checkpoint_before = sha256_file(checkpoint); model = load_legacy_model(checkpoint).to(device); model.eval(); official_parameters = list(model.parameters()); official_before_sha = tensor_tree_hash({str(index): value.detach() for index, value in enumerate(official_parameters)}); tokenizer = T5Tokenizer.from_pretrained(base["paths"]["model_dir"]); forget = JsonPromptDataset(Path(base["paths"]["forget"]), tokenizer); retain = JsonPromptDataset(Path(base["paths"]["retain"]), tokenizer); validation = JsonPromptDataset(Path(base["paths"]["validation"]), tokenizer); _, indices, users = _data_lineage(root, base, _resolve(root, config["protocol_root"])); started = time.perf_counter(); timing = {}
    try:
        modules = collect_qv_modules(model); weights = [module.weight for _, module in modules]; states = [value.requires_grad for value in weights]
        for parameter in model.parameters(): parameter.requires_grad_(False)
        for weight in weights: weight.requires_grad_(True)
        tick = time.perf_counter(); matrices, effective_report = _stream_weight_gradients(model, forget, indices["forget_train"], weights, device, config["train_objective"]["batch_size"]); timing["full_forget_train_effective_gradient_seconds"] = time.perf_counter() - tick
        for weight, state in zip(weights, states): weight.requires_grad_(state)
        tick = time.perf_counter(); bases = {}; basis_reports = []
        for (name, _), matrix in zip(modules, matrices): bases[name], report = build_fixed_a_basis(matrix, rank=16, name=name, seed=42); basis_reports.append(report)
        basis_sha = canonical_hash([(report["name"], report["basis_sha256"]) for report in basis_reports]); timing["svd_basis_seconds"] = time.perf_counter() - tick
        equivalence_batch = move_batch(_batch(forget, [0]), device)
        with torch.no_grad(): before_logits = model(input_ids=equivalence_batch["input_ids"], labels=equivalence_batch["target_ids"]).logits.detach().cpu()
        names, parameters = install_fixed_ab_coordinate(model, bases, 32)
        with torch.no_grad(): after_logits = model(input_ids=equivalence_batch["input_ids"], labels=equivalence_batch["target_ids"]).logits.detach().cpu()
        if not torch.equal(before_logits, after_logits): raise RuntimeError("B=0 function equivalence failed")
        tick = time.perf_counter(); direct, direct_report = _stream_weight_gradients(model, forget, indices["forget_train"], parameters, device, config["train_objective"]["batch_size"]); g_f = flatten_tensors(direct); timing["full_forget_B_gradient_seconds"] = time.perf_counter() - tick
        analytic = flatten_tensors([analytic_b_gradient(matrix, bases[name], 32, 16) for (name, _), matrix in zip(modules, matrices)]); relative_error = float(torch.linalg.vector_norm(g_f - analytic) / max(float(torch.linalg.vector_norm(analytic)), 1e-300))
        if relative_error > config["lora_coordinate"]["analytic_gradient_relative_tolerance"]: raise RuntimeError("B-space analytic gradient mismatch")
        panels = select_retain_panels(users["retain_train"], config["panels"]); counters = {name: {"hvp_batches": 0} for name in ("primary", "stability")}; operators = {name: _make_curvature_operator(model, retain, panels[name]["indices"], parameters, device, config["panels"]["batch_size"], counters[name]) for name in counters}
        tick = time.perf_counter(); estimates = {name: estimate_lambda_max(operators[name], g_f.numel(), seed=panels[name]["seed"], iterations=config["curvature"]["power_iterations"], convergence_tolerance=config["curvature"]["convergence_tolerance"], numerical_lower_bound=config["curvature"]["numerical_lower_bound"]) for name in operators}; timing["lambda_max_estimation_seconds"] = time.perf_counter() - tick
        ratio_reports = []; selected = None
        for ratio in config["cg"]["relative_damping_ratios"] + [config["cg"]["diagnostic_overdamped_ratio"]]:
            solutions = {}; cg_reports = {}
            for panel_name in ("primary", "stability"):
                damping = ratio * estimates[panel_name]["lambda_max_hat"]; report = conjugate_gradient(operators[panel_name], g_f, damping=damping, relative_tolerance=config["cg"]["relative_residual_tolerance"], absolute_tolerance=config["cg"]["absolute_residual_tolerance"], max_iterations=config["cg"]["max_iterations"], residual_explosion_factor=config["cg"]["residual_explosion_factor"], pap_tolerance=config["cg"]["pap_absolute_tolerance"]); solutions[panel_name] = report.pop("solution"); report["actual_damping"] = damping; cg_reports[panel_name] = report
            stability = direction_stability(solutions["primary"], solutions["stability"], names, [parameter.shape for parameter in parameters], config["stability"]); contribution = curvature_to_damping_ratio(operators["primary"], solutions["primary"], ratio * estimates["primary"]["lambda_max_hat"]); eligible = ratio != config["cg"]["diagnostic_overdamped_ratio"] and stability["passed"] and contribution >= config["cg"]["curvature_to_damping_min"]
            row = {"relative_damping_ratio": ratio, "primary_lambda": ratio * estimates["primary"]["lambda_max_hat"], "stability_lambda": ratio * estimates["stability"]["lambda_max_hat"], "cg": cg_reports, "stability": stability, "curvature_to_damping_ratio": contribution, "cosine_with_forget_gradient": float(torch.dot(g_f, solutions["primary"]) / (torch.linalg.vector_norm(g_f) * torch.linalg.vector_norm(solutions["primary"]))), "eligible": eligible, "diagnostic_only": ratio == 1.0}
            ratio_reports.append(row)
            if selected is None and eligible: selected = (row, solutions["primary"])
        if selected is None: raise RuntimeError("no_stable_relative_damping_candidate")
        safety_indices = panels["safety"]["indices"]
        g_sup_parts, sup_report = _stream_weight_gradients(model, retain, safety_indices, parameters, device, config["panels"]["batch_size"]); g_sup = flatten_tensors(g_sup_parts)
        # Self-KL has zero first derivative at B=0 by construction; compute it explicitly.
        def kl_gradient() -> torch.Tensor:
            result = torch.zeros_like(g_f); total_tokens = 0
            for start in range(0, len(safety_indices), config["panels"]["batch_size"]):
                batch = move_batch(_batch(retain, safety_indices[start:start+config["panels"]["batch_size"]]), device); mask = batch["target_ids"] != -100; tokens = int(mask.sum())
                with torch.no_grad(): reference = model(input_ids=batch["input_ids"], labels=batch["target_ids"]).logits.detach()
                current = model(input_ids=batch["input_ids"], labels=batch["target_ids"]).logits; loss = self_kl_loss(reference, current, mask); gradients = torch.autograd.grad(loss, parameters); result += tokens * flatten_tensors(gradients); total_tokens += tokens; del batch, mask, reference, current, loss, gradients
            return result / total_tokens
        g_kl = kl_gradient(); base_flat = flatten_tensors([parameter.detach() for parameter in parameters]); projection = project_update_space(selected[1], [g_kl, g_sup], relative_tolerance=config["projection"]["relative_singular_tolerance"], normalized_tolerance=config["projection"]["normalized_constraint_tolerance"], formal_dtype=torch.float32, base=base_flat); safe = projection.pop("actual"); safe_raw = float(torch.linalg.vector_norm(safe) / torch.linalg.vector_norm(selected[1])); forget_retained = float(torch.dot(g_f, safe) / max(float(torch.dot(g_f, selected[1])), 1e-300)); retain_safety = projection["passed"] and safe_raw >= config["projection"]["safe_raw_norm_ratio_min"]
        original_metrics = _evaluate_utility({"current": model}, validation, indices["retain_user_validation"], device); candidates = []
        for scale in config["candidates"]["scales"]:
            actual = (base_flat + scale * safe).to(torch.float32).to(torch.float64) - base_flat.to(torch.float32).to(torch.float64)
            with _temporary_delta(parameters, actual): metrics = _evaluate_utility({"current": model}, validation, indices["retain_user_validation"], device)
            utility = dual_utility(pre["fixed_utility_baseline"]["metrics"], original_metrics, metrics, config); candidate_id = f"original:data_adaptive_fixed_A_B_space:relative_damping_{selected[0]['relative_damping_ratio']}:primary_stability:retain_safe:{scale}"; candidates.append({"candidate_id": candidate_id, "scale": scale, "relative_damping_ratio": selected[0]["relative_damping_ratio"], "candidate_sha256": tensor_tree_hash({"candidate": actual}), "directional_gate_pass": retain_safety and float(torch.dot(g_f, actual)) > 0, "utility": utility, "utility_pass": utility["utility_pass"], "actual": actual})
        chosen = next((row for row in candidates if row["directional_gate_pass"] and row["utility_pass"]), None)
        if chosen is None: raise RuntimeError("candidate_scale_rejected")
        candidate_registry = [{key: value for key, value in row.items() if key != "actual"} for row in candidates]; candidate_registry_sha = canonical_hash(candidate_registry)
        official_after_sha = tensor_tree_hash({str(index): value.detach() for index, value in enumerate(official_parameters)})
        value = {"schema": UNIT_SCHEMA, "phase": 1, "kind": "train_objective_direction", "train_objective": {"forget_scope": "forget_train", "retain_scope": "retain_train", "forget_samples": 12982, "effective_gradient": effective_report, "B_gradient": direct_report, "B_gradient_norm": float(torch.linalg.vector_norm(g_f)), "B_gradient_sha256": tensor_tree_hash({"g_F": g_f}), "analytic_gradient_relative_error": relative_error}, "coordinate": {"kind": config["lora_coordinate"]["kind"], "module_order": names, "basis_reports": basis_reports, "basis_sha256": basis_sha, "B_initially_zero": True, "function_equivalence_exact": True, "A_frozen": True, "B_only": True}, "panels": {name: {key: value for key, value in row.items() if key != "indices"} for name, row in panels.items()}, "lambda_max": estimates, "relative_damping_trials": ratio_reports, "selected_relative_damping_ratio": selected[0]["relative_damping_ratio"], "safety_projection": {**projection, "safe_raw_norm_ratio": safe_raw, "forget_first_order_retained_ratio": forget_retained, "retain_safety_pass": retain_safety}, "original_utility": original_metrics, "candidate_registry": candidate_registry, "candidate_registry_sha256": candidate_registry_sha, "selected_primary_candidate_id": chosen["candidate_id"], "selected_candidate_sha256": chosen["candidate_sha256"], "timing": {**timing, "phase1_seconds": time.perf_counter() - started}, "checkpoint_sha256_before": checkpoint_before, "checkpoint_sha256_after": sha256_file(checkpoint), "official_parameter_sha256_before": official_before_sha, "official_parameter_sha256_after": official_after_sha, "retrain_loaded": False, "retrain_used_for_selection": False, "candidate_frozen": True, "vectors_persisted": False, "model_parameters_modified": official_before_sha != official_after_sha, "parameter_grad_absent": all(parameter.grad is None for parameter in model.parameters()), "optimizer_constructed": False, "optimizer_steps_committed": 0, "step817_checkpoint_published": False, "test_accessed": False}
        if value["checkpoint_sha256_after"] != checkpoint_before or value["model_parameters_modified"] or not value["parameter_grad_absent"]: raise RuntimeError("Original safety invariant failed")
        unit = run_dir / "units" / "phase1_train_objective_direction"; binding = _unit_binding(pre, sha256_file(run_dir / "contract.json"), "phase1_train_objective_direction", "train_objective_direction", 0)
        if unit.exists():
            existing = _validate_unit(unit, binding)["value"]
            if existing["selected_candidate_sha256"] != chosen["candidate_sha256"] or existing["coordinate"]["basis_sha256"] != basis_sha: raise ValueError("Resume candidate reconstruction mismatch")
        else: _publish_unit(unit, value, binding)
        return {"retrain_loaded": False, "candidate_frozen": True, "unit_manifest_sha256": sha256_file(unit / "manifest.json"), "model": model, "tokenizer": tokenizer, "validation": validation, "parameters": parameters, "candidate": chosen["actual"], "candidate_sha256": chosen["candidate_sha256"], "indices": indices, "users": users, "basis_sha256": basis_sha, "gradient_sha256": value["train_objective"]["B_gradient_sha256"]}
    except BaseException:
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        raise


def run_real_phase2(root: Path, config: dict[str, Any], pre: dict[str, Any], run_dir: Path, phase1: dict[str, Any]) -> dict[str, Any]:
    unit = run_dir / "units" / "phase2_posthoc_retrain"; phase1_unit = run_dir / "units" / "phase1_train_objective_direction"; predecessor_sha = sha256_file(phase1_unit / "manifest.json"); binding = _unit_binding(pre, sha256_file(run_dir / "contract.json"), "phase2_posthoc_retrain", "posthoc_retrain", 1, predecessor_sha)
    if unit.exists(): _validate_unit(unit, binding); return {"unit_manifest_sha256": sha256_file(unit / "manifest.json")}
    frozen = _validate_unit(phase1_unit)["value"]
    if frozen["selected_candidate_sha256"] != phase1["candidate_sha256"] or frozen["coordinate"]["basis_sha256"] != phase1["basis_sha256"] or frozen["retrain_loaded"] is not False: raise ValueError("Phase-1 reconstruction/freeze mismatch")
    selector_rows = [json.loads(line) for line in (_resolve(root, config["authority"]["development_selector"]) / "rows.jsonl").read_text(encoding="utf-8").splitlines()]; active = {int(row["source_index"]): bool(row["active"]) for row in selector_rows}; user_map = {index: int(user) for index, user in enumerate(phase1["users"]["overall_validation"])}; base = load_config(_resolve(root, config["base_config"]), root); device = next(phase1["model"].parameters()).device; started = time.perf_counter(); retrain = freeze_teacher(load_legacy_model(Path(base["paths"]["retrain_reference"]))).to(device)
    try:
        evaluation = _evaluate_against_retrain(phase1["model"], retrain, phase1["validation"], phase1["indices"]["forget_user_validation"], user_map, active, phase1["candidate"], phase1["parameters"], config, device); groups = evaluation["groups"]; all_group = groups["all"]; scientific = all_group["full_vocabulary_jsd_improvement"]["ci95_low"] > 0 and all_group["yes_no_jsd_improvement"]["ci95_low"] > 0 and all_group["answer_loss_improvement"]["ci95_low"] >= config["evaluation"]["answer_loss_noninferiority_bound"]; subgroup_conflict = groups["observed_yes"]["full_vocabulary_jsd_improvement"]["ci95_high"] < 0 or groups["observed_no"]["full_vocabulary_jsd_improvement"]["ci95_high"] < 0; active_noninferiority = groups["active"]["answer_loss_improvement"]["ci95_low"] >= config["evaluation"]["answer_loss_noninferiority_bound"]
        value = {"schema": UNIT_SCHEMA, "phase": 2, "kind": "posthoc_retrain", "candidate_sha256": phase1["candidate_sha256"], "basis_sha256": phase1["basis_sha256"], "gradient_sha256": phase1["gradient_sha256"], "evaluation": evaluation, "valid": True, "scientific_pass": bool(scientific and not subgroup_conflict and active_noninferiority), "reliable_reverse_direction": evaluation["reliable_reverse_direction"], "subgroup_conflict": bool(subgroup_conflict), "active_noninferiority": bool(active_noninferiority), "train_direction_stable": True, "retain_safety_pass": frozen["safety_projection"]["retain_safety_pass"], "utility_pass": next(row for row in frozen["candidate_registry"] if row["candidate_id"] == frozen["selected_primary_candidate_id"])["utility_pass"], "efficiency_status": "unavailable", "efficiency_ratio": None, "phase2_posthoc_seconds": time.perf_counter() - started, "retrain_loaded_after_candidate_freeze": True, "retrain_used_for_selection": False, "optimizer_constructed": False, "optimizer_steps_committed": 0, "step817_checkpoint_published": False, "vectors_persisted": False, "test_accessed": False}
        _publish_unit(unit, value, binding); return {"unit_manifest_sha256": sha256_file(unit / "manifest.json")}
    finally:
        del retrain, phase1["model"]
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def verify_full(root: Path, config: dict[str, Any], pre: dict[str, Any], run_name: str) -> dict[str, Any]:
    path = _resolve(root, Path(config["output_root"]) / "full_runs" / _safe_name(run_name), output=True)
    expected = {"contract.json", "run_state.json", "full_manifest.json", "units", "COMPLETED"}
    if not path.is_dir() or {item.name for item in path.iterdir()} != expected or (path / "COMPLETED").read_text(encoding="utf-8") != TERMINAL_MARKER + "\n": raise ValueError("Analyze refuses incomplete Full")
    if _read_json(path / "contract.json") != build_contract(pre, run_name): raise ValueError("Full contract mismatch")
    units = path / "units"; expected_units = {"phase1_train_objective_direction", "phase2_posthoc_retrain"}
    if {item.name for item in units.iterdir()} != expected_units: raise ValueError("Full unit inventory invalid")
    contract_sha = sha256_file(path / "contract.json"); one = _validate_unit(units / "phase1_train_objective_direction", _unit_binding(pre, contract_sha, "phase1_train_objective_direction", "train_objective_direction", 0)); one_sha = sha256_file(units / "phase1_train_objective_direction" / "manifest.json"); two = _validate_unit(units / "phase2_posthoc_retrain", _unit_binding(pre, contract_sha, "phase2_posthoc_retrain", "posthoc_retrain", 1, one_sha)); state = _read_json(path / "run_state.json"); manifest = _read_json(path / "full_manifest.json")
    if state.get("status") != "COMPLETED" or state.get("optimizer_steps_committed") != 0 or state.get("test_accessed") is not False or manifest.get("contract_sha256") != contract_sha or manifest.get("run_state_sha256") != sha256_file(path / "run_state.json") or manifest.get("phase1_manifest_sha256") != one_sha or manifest.get("phase2_manifest_sha256") != sha256_file(units / "phase2_posthoc_retrain" / "manifest.json") or one["value"].get("retrain_loaded") is not False or two["value"].get("retrain_used_for_selection") is not False: raise ValueError("Full terminal evidence invalid")
    return {"path": path, "phase1": one["value"], "phase2": two["value"]}


def analyze(root: Path, config_path: Path, run_name: str) -> dict[str, Any]:
    pre = preflight(root, config_path); require_clean_git(pre["git"], "A2 Analyze"); config = load_audit_config(config_path, root); verified = verify_full(root, config, pre, run_name); phase2 = verified["phase2"]
    decision = classify_if_a2(valid=phase2["valid"], train_direction_stable=phase2["train_direction_stable"], retain_safety_pass=phase2["retain_safety_pass"], utility_pass=phase2["utility_pass"], scientific_pass=phase2["scientific_pass"], reliable_reverse_direction=phase2["reliable_reverse_direction"], subgroup_conflict=phase2["subgroup_conflict"], active_noninferiority=phase2["active_noninferiority"], efficiency_status=phase2["efficiency_status"], efficiency_ratio=phase2["efficiency_ratio"])
    result = {"schema": ANALYSIS_SCHEMA, "run_name": run_name, **decision, "scientific_pass": phase2["scientific_pass"], "retain_safety_pass": phase2["retain_safety_pass"], "utility_pass": phase2["utility_pass"], "efficiency_status": phase2["efficiency_status"], "retrain_used_for_selection": False, "optimizer_constructed": False, "optimizer_steps_committed": 0, "step817_checkpoint_published": False, "test_accessed": False}
    destination = _resolve(root, Path(config["output_root"]) / "analysis_runs" / _safe_name(run_name), output=True)
    if destination.exists(): raise FileExistsError(destination)
    stage = destination.parent / f".{destination.name}.{uuid.uuid4().hex[:10]}.stage"; stage.mkdir(parents=True); atomic_json(stage / "analysis.json", result); atomic_json(stage / "manifest.json", {"schema": ANALYSIS_SCHEMA, "analysis_sha256": sha256_file(stage / "analysis.json"), "source_full_manifest_sha256": sha256_file(verified["path"] / "full_manifest.json"), "published_atomically": True, "test_accessed": False}); atomic_text(stage / "COMPLETED", "T5_LORA_INFLUENCE_A2_ANALYSIS_COMPLETED\n"); destination.parent.mkdir(parents=True, exist_ok=True); os.replace(stage, destination); return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=("Preflight", "SyntheticDryRun", "DryRun", "Full", "Resume", "Analyze"), default="Preflight"); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--project-root", type=Path, default=Path.cwd()); parser.add_argument("--run-name"); args = parser.parse_args(); root = args.project_root.resolve(); config = args.config.resolve()
    if args.mode == "Preflight": result = preflight(root, config)
    else:
        if not args.run_name: parser.error(f"{args.mode} requires --run-name")
        if args.mode in {"SyntheticDryRun", "DryRun"}: result = synthetic_run(root, config, args.run_name, args.mode)
        elif args.mode == "Full": result = execute_full(root, config, args.run_name, resume=False)
        elif args.mode == "Resume": result = execute_full(root, config, args.run_name, resume=True)
        else: result = analyze(root, config, args.run_name)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
