from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from peft import get_peft_model_state_dict
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from src.diagnostics.git_provenance import (
    git_provenance,
    implementation_provenance,
    require_clean_git,
)
from src.diagnostics.t5_full_runner import (
    _batch,
    _rng_payload,
    evaluate_overall_validation,
)
from src.diagnostics.t5_reconstructed_official import (
    JsonPromptDataset,
    compute_components,
    load_config,
    move_batch,
    sha256_file,
)
from src.diagnostics.t5_step812_gradient_geometry import (
    _load_runtime,
    _tensor_state_hash,
    canonical_hash,
    catalog_pair,
)
from src.diagnostics.t5_step813_optimizer_aware_audit import (
    EXPECTED_BATCH_HASH,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_STATE_SHA256,
    _flatten_gradients,
    _runtime_binding,
    _split_flat,
    _tensor_hash,
    direct_svd_projection,
    directional_metrics,
    json_native,
    preflight as optimizer_preflight,
    validate_optimizer_mapping,
)
from src.diagnostics.t5_trajectory_diagnostics import (
    RngState,
    capture_rng,
    restore_rng,
    rng_hashes,
)
from src.diagnostics.t5_zero_training_audit import _data_lineage
from src.diagnostics.t5_zero_training_decision_v2 import (
    COLLAPSE_STD_EPSILON,
    FROZEN_UTILITY_THRESHOLDS,
    utility_evidence,
)

SCHEMA = "t5-step813-update-space-stage-b-v1"
ANALYSIS_SCHEMA = "t5-step813-update-space-stage-b-analysis-v1"
CHECKPOINT_SCHEMA = "t5-step813-update-space-stage-b-checkpoint-v1"
SOURCE_FULL_MANIFEST_SHA = "19de758fbfb12f5dbe0971e0976dd6aa1b4d6ca227cdd4be06a482c59cf4f5b3"
SOURCE_ANALYSIS_SHA = "891006aa7d7226e96dab04ed74d46ac597080482e2f373ebfde62d00c4068c9f"
EXPECTED_A3_HASH = "8bfa35edf2dacd1e45d4760a9eaf1cd715a915bda07b802d6cbd7dfa06d6d239"
EXPECTED_A3_NORM = 0.32584806756766416
EXPECTED_A3_FORGET_DOT = -0.0003372693687872579
EXPECTED_A3_KL_NORMALIZED = 1.6500204996479158e-16
EXPECTED_A3_SUP_NORMALIZED = 4.636444069132894e-17
EXPECTED_A1_EFFECTIVENESS = 0.9850037911548284
LOSSES = ("L_forget", "L_retain_KL", "L_sup")
IMPLEMENTATION_FILES = (
    "src/diagnostics/t5_step813_update_space_stage_b.py",
    "configs/t5_step813_update_space_stage_b_v1.yaml",
    "scripts/diagnostics/t5_step813_update_space_stage_b_v1.ps1",
    "docs/t5_step813_update_space_stage_b_v1.md",
    "src/diagnostics/t5_step813_optimizer_aware_audit.py",
    "src/diagnostics/t5_step812_gradient_geometry.py",
    "src/diagnostics/t5_full_runner.py",
    "src/diagnostics/t5_reconstructed_official.py",
    "src/diagnostics/t5_zero_training_audit.py",
    "src/diagnostics/t5_zero_training_decision_v2.py",
    "src/diagnostics/git_provenance.py",
)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:12]}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(json_native(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _safe_name(value: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError("RunName must be one path component")
    return value


def _resolve(root: Path, value: str | Path, *, output: bool = False) -> Path:
    path = (root / value).resolve()
    if path != root.resolve() and root.resolve() not in path.parents:
        raise ValueError("path escapes project root")
    if not output and any("test" in part.lower() for part in path.parts):
        raise ValueError(f"test path forbidden: {path}")
    return path


def load_stage_config(path: Path, root: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("development_only") is not True:
        raise ValueError("Stage-B config schema/scope mismatch")
    if value.get("test_access_policy") != "forbidden":
        raise ValueError("test access policy must be forbidden")
    expected = value.get("expected", {})
    fixed = {
        "checkpoint_state_sha256": EXPECTED_STATE_SHA256,
        "checkpoint_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "optimizer_full_manifest_sha256": SOURCE_FULL_MANIFEST_SHA,
        "optimizer_analysis_sha256": SOURCE_ANALYSIS_SHA,
        "source_category": "OA-B",
        "source_next_action": "implement_update_space_projection_before_stage_B",
        "a3_delta_sha256": EXPECTED_A3_HASH,
        "a3_delta_norm": EXPECTED_A3_NORM,
        "a3_forget_dot": EXPECTED_A3_FORGET_DOT,
        "a3_kl_normalized": EXPECTED_A3_KL_NORMALIZED,
        "a3_sup_normalized": EXPECTED_A3_SUP_NORMALIZED,
        "a1_forget_effectiveness": EXPECTED_A1_EFFECTIVENESS,
        "step813_batch_hash": EXPECTED_BATCH_HASH,
    }
    if expected != fixed:
        raise ValueError("Stage-B source authority changed")
    if value.get("gates") != {
        "normalized_retain_tolerance": 1e-8,
        "forget_descent_zero_tolerance": 1e-12,
        "forget_effectiveness_min": 0.10,
        "a3_norm_absolute_tolerance": 1e-12,
    }:
        raise ValueError("Stage-B directional gates changed")
    utility = value.get("utility", {})
    for key, expected_value in FROZEN_UTILITY_THRESHOLDS.items():
        if utility.get(key) != expected_value:
            raise ValueError(f"frozen utility threshold changed: {key}")
    if (
        utility.get("authority") != "t5_zero_training_decision_v2"
        or utility.get("collapse_std_epsilon") != COLLAPSE_STD_EPSILON
        or utility.get("prediction_collapse_forbidden") is not True
        or utility.get("nan_inf_forbidden") is not True
    ):
        raise ValueError("Stage-B utility authority changed")
    value["_path"] = str(path.resolve())
    value["_sha256"] = sha256_file(path)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_optimizer_aware_source(full: Path, analysis: Path) -> dict[str, Any]:
    full_required = {"audit.json", "contract.json", "manifest.json", "provenance.json", "COMPLETED"}
    analysis_required = {"analysis.json", "manifest.json", "COMPLETED"}
    if not full.is_dir() or {item.name for item in full.iterdir()} != full_required:
        raise ValueError("optimizer-aware Full inventory mismatch")
    if not analysis.is_dir() or {item.name for item in analysis.iterdir()} != analysis_required:
        raise ValueError("optimizer-aware Analyze inventory mismatch")
    if (full / "COMPLETED").read_text(encoding="utf-8") != "FULL_AUDIT_COMPLETED\n":
        raise ValueError("optimizer-aware Full completion mismatch")
    if (analysis / "COMPLETED").read_text(encoding="utf-8") != "ANALYSIS_COMPLETED\n":
        raise ValueError("optimizer-aware Analyze completion mismatch")
    full_manifest = _read_json(full / "manifest.json")
    analysis_manifest = _read_json(analysis / "manifest.json")
    full_manifest_sha = sha256_file(full / "manifest.json")
    analysis_sha = sha256_file(analysis / "analysis.json")
    if full_manifest_sha != SOURCE_FULL_MANIFEST_SHA or analysis_sha != SOURCE_ANALYSIS_SHA:
        raise ValueError("optimizer-aware source SHA mismatch")
    for key, name in (("contract_sha256", "contract.json"), ("audit_sha256", "audit.json"), ("provenance_sha256", "provenance.json")):
        if full_manifest.get(key) != sha256_file(full / name):
            raise ValueError(f"optimizer-aware Full {name} SHA mismatch")
    if (
        full_manifest.get("published_atomically") is not True
        or full_manifest.get("test_accessed") is not False
        or analysis_manifest.get("analysis_sha256") != analysis_sha
        or analysis_manifest.get("source_manifest_sha256") != full_manifest_sha
        or analysis_manifest.get("published_atomically") is not True
        or analysis_manifest.get("test_accessed") is not False
    ):
        raise ValueError("optimizer-aware source manifest binding mismatch")
    contract, provenance = _read_json(full / "contract.json"), _read_json(full / "provenance.json")
    audit, decision = _read_json(full / "audit.json"), _read_json(analysis / "analysis.json")
    if contract.get("git") != provenance.get("git") or contract.get("implementation") != provenance.get("implementation"):
        raise ValueError("optimizer-aware Full provenance mismatch")
    if decision.get("git") != contract.get("git") or decision.get("implementation") != contract.get("implementation"):
        raise ValueError("optimizer-aware Analyze provenance mismatch")
    if decision.get("category") != "OA-B" or decision.get("next_action") != "implement_update_space_projection_before_stage_B":
        raise ValueError("optimizer-aware source is not the required OA-B decision")
    if decision.get("source_manifest_sha256") != full_manifest_sha:
        raise ValueError("optimizer-aware analysis source binding mismatch")
    a3 = audit.get("counterfactuals", {}).get("A3", {})
    directional = a3.get("directional", {})
    observed = {
        "delta_hash": a3.get("delta_hash"),
        "delta_norm": a3.get("delta_norm"),
        "forget_dot": directional.get("L_forget", {}).get("dot"),
        "kl_normalized": directional.get("L_retain_KL", {}).get("normalized"),
        "sup_normalized": directional.get("L_sup", {}).get("normalized"),
        "forget_effectiveness": audit.get("forget_effectiveness"),
        "a0_forget_dot": audit.get("counterfactuals", {}).get("A0", {}).get("directional", {}).get("L_forget", {}).get("dot"),
    }
    expected = {
        "delta_hash": EXPECTED_A3_HASH,
        "delta_norm": EXPECTED_A3_NORM,
        "forget_dot": EXPECTED_A3_FORGET_DOT,
        "kl_normalized": EXPECTED_A3_KL_NORMALIZED,
        "sup_normalized": EXPECTED_A3_SUP_NORMALIZED,
        "forget_effectiveness": EXPECTED_A1_EFFECTIVENESS,
        "a0_forget_dot": -0.0003376303641663901,
    }
    if observed != expected:
        raise ValueError("optimizer-aware A3 authority mismatch")
    return json_native({
        "full_path": str(full),
        "analysis_path": str(analysis),
        "full_manifest_sha256": full_manifest_sha,
        "analysis_sha256": analysis_sha,
        "source_git": contract["git"],
        "source_implementation": contract["implementation"],
        "category": decision["category"],
        "next_action": decision["next_action"],
        "a3": observed,
        "test_accessed": False,
    })


def preflight(
    root: Path,
    config_path: Path,
    *,
    git_function=git_provenance,
    implementation_function=implementation_provenance,
) -> dict[str, Any]:
    config = load_stage_config(config_path, root)
    source = validate_optimizer_aware_source(
        _resolve(root, config["source_optimizer_full"]),
        _resolve(root, config["source_optimizer_analysis"]),
    )
    optimizer = optimizer_preflight(
        root,
        _resolve(root, config["optimizer_audit_config"]),
        git_function=git_function,
        implementation_function=lambda _root, _paths: {"deferred_to_stage_b": True},
    )
    base = load_config(_resolve(root, config["base_config"]), root)
    lineage, indices, _ = _data_lineage(
        root, base, _resolve(root, config["protocol_root"])
    )
    pair813 = catalog_pair(0, 12982, 47018, 16, 42)
    pair814 = catalog_pair(1, 12982, 47018, 16, 42)
    if pair813["batch_hash"] != EXPECTED_BATCH_HASH:
        raise ValueError("step813 batch authority mismatch")
    result = {
        "schema": SCHEMA,
        "mode": "Preflight",
        "development_only": True,
        "config_sha256": config["_sha256"],
        "git": git_function(root),
        "implementation": implementation_function(root, IMPLEMENTATION_FILES),
        "optimizer_aware_source": source,
        "source_checkpoint": optimizer["checkpoint"],
        "optimizer_mapping": optimizer["optimizer_mapping"],
        "teachers": optimizer["teachers"],
        "step813": pair813,
        "step814": pair814,
        "lineage": lineage,
        "validation_indices": {
            "overall_samples": len(indices["overall_validation"]),
            "retain_user_samples": len(indices["retain_user_validation"]),
            "retain_user_indices_sha256": canonical_hash(indices["retain_user_validation"]),
        },
        "utility": {
            **FROZEN_UTILITY_THRESHOLDS,
            "collapse_std_epsilon": COLLAPSE_STD_EPSILON,
            "authority": "t5_zero_training_decision_v2",
        },
        "model_loaded": False,
        "retrain_loaded": False,
        "authoritative_optimizer_step_calls": 0,
        "logical_optimizer_steps_committed": 0,
        "test_loader_built": False,
        "test_accessed": False,
    }
    return json_native(result)


def shadow_adamw_proposal(
    official_parameters: list[torch.Tensor],
    optimizer_state: dict[str, Any],
    flat_gradient: torch.Tensor,
) -> dict[str, Any]:
    if any(getattr(value, "grad", None) is not None for value in official_parameters):
        raise ValueError("source parameter grad must be None")
    source_hashes = [_tensor_hash(value) for value in official_parameters]
    shadow = [torch.nn.Parameter(value.detach().clone(), requires_grad=True) for value in official_parameters]
    group = optimizer_state["param_groups"][0]
    optimizer = torch.optim.AdamW(
        shadow,
        lr=group["lr"],
        betas=tuple(group["betas"]),
        eps=group["eps"],
        weight_decay=group["weight_decay"],
        amsgrad=group.get("amsgrad", False),
        maximize=group.get("maximize", False),
        foreach=group.get("foreach"),
        capturable=group.get("capturable", False),
        differentiable=group.get("differentiable", False),
        fused=group.get("fused"),
    )
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    before = [value.detach().clone() for value in shadow]
    for parameter, gradient in zip(shadow, _split_flat(flat_gradient, shadow)):
        parameter.grad = gradient
    optimizer.step()
    delta = torch.cat([
        (value.detach() - prior).double().cpu().reshape(-1)
        for value, prior in zip(shadow, before)
    ])
    if any(_tensor_hash(value) != digest for value, digest in zip(official_parameters, source_hashes)):
        raise RuntimeError("shadow proposal changed source parameter")
    if any(getattr(value, "grad", None) is not None for value in official_parameters):
        raise RuntimeError("shadow proposal populated source grad")
    state = optimizer.state_dict()
    steps = [float(item["step"]) for item in state["state"].values()]
    if not steps or any(step != 813.0 for step in steps):
        raise RuntimeError("shadow AdamW state did not advance exactly 812 to 813")
    return {
        "delta": delta,
        "optimizer_state": state,
        "delta_hash": _tensor_hash(delta),
        "delta_norm": float(torch.linalg.vector_norm(delta)),
        "shadow_optimizer_steps_executed": 1,
        "authoritative_optimizer_step_calls": 0,
    }


def update_space_projection(delta: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    value = delta.detach().double().reshape(-1)
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("non-finite candidate delta")
    return value - basis @ (basis.T @ value) if basis.numel() else value.clone()


def materialize_dtype_delta(
    parameters: list[torch.Tensor], projected: torch.Tensor
) -> tuple[list[torch.Tensor], torch.Tensor]:
    updates, actual, offset = [], [], 0
    for parameter in parameters:
        count = parameter.numel()
        before = parameter.detach().cpu()
        requested = projected[offset : offset + count].reshape(parameter.shape).to(parameter.dtype)
        after = before + requested
        realized = after - before
        updates.append(after)
        actual.append(realized.double().reshape(-1))
        offset += count
    if offset != projected.numel():
        raise ValueError("projected delta does not match parameter order")
    return updates, torch.cat(actual)


def precommit_gate(
    f: torch.Tensor,
    k: torch.Tensor,
    s: torch.Tensor,
    raw_delta: torch.Tensor,
    projected_float64: torch.Tensor,
    actual_delta: torch.Tensor,
    config: dict[str, Any],
    *,
    expected_a3: bool,
    effectiveness_reference_dot: float | None = None,
) -> dict[str, Any]:
    raw = directional_metrics(f, k, s, raw_delta)
    projected = directional_metrics(f, k, s, projected_float64)
    actual = directional_metrics(f, k, s, actual_delta)
    reference_dot = raw["L_forget"]["dot"] if effectiveness_reference_dot is None else effectiveness_reference_dot
    denominator = -reference_dot
    effectiveness = None if denominator <= 0 else -actual["L_forget"]["dot"] / denominator
    gates = config["gates"]
    finite = all(math.isfinite(value) for value in (
        projected_float64.norm().item(), actual_delta.norm().item(),
        actual["L_forget"]["dot"], actual["L_retain_KL"]["dot"], actual["L_sup"]["dot"],
    ))
    checks = {
        "finite": finite,
        "nonzero_delta": actual["delta_norm"] > 0,
        "forget_descent": actual["L_forget"]["dot"] < -gates["forget_descent_zero_tolerance"],
        "forget_effectiveness": effectiveness is not None and effectiveness >= gates["forget_effectiveness_min"],
        "retain_kl": actual["L_retain_KL"]["normalized"] is not None and actual["L_retain_KL"]["normalized"] <= gates["normalized_retain_tolerance"],
        "retain_sup": actual["L_sup"]["normalized"] is not None and actual["L_sup"]["normalized"] <= gates["normalized_retain_tolerance"],
        "a3_hash": (not expected_a3) or _tensor_hash(projected_float64) == EXPECTED_A3_HASH,
        "a3_norm": (not expected_a3) or abs(projected["delta_norm"] - EXPECTED_A3_NORM) <= gates["a3_norm_absolute_tolerance"],
        "a3_direction": (not expected_a3) or (
            projected["L_forget"]["dot"] == EXPECTED_A3_FORGET_DOT
            and projected["L_retain_KL"]["normalized"] == EXPECTED_A3_KL_NORMALIZED
            and projected["L_sup"]["normalized"] == EXPECTED_A3_SUP_NORMALIZED
        ),
    }
    return json_native({
        "passed": all(checks.values()),
        "checks": checks,
        "raw_adamw": raw,
        "projected_float64": projected,
        "actual_dtype_delta": actual,
        "forget_effectiveness": effectiveness,
        "forget_effectiveness_reference_dot": reference_dot,
        "projected_float64_hash": _tensor_hash(projected_float64),
        "actual_dtype_delta_hash": _tensor_hash(actual_delta),
    })


def commit_or_rollback(
    parameters: list[torch.nn.Parameter],
    after_values: list[torch.Tensor],
    actual_delta_hash: str,
    *,
    post_gate: Callable[[], tuple[bool, dict[str, Any]]],
) -> dict[str, Any]:
    before = [value.detach().cpu().clone() for value in parameters]
    evidence: dict[str, Any] | None = None
    try:
        with torch.no_grad():
            for parameter, value in zip(parameters, after_values):
                parameter.copy_(value.to(parameter.device))
        realized = torch.cat([
            (parameter.detach().cpu() - old).double().reshape(-1)
            for parameter, old in zip(parameters, before)
        ])
        if _tensor_hash(realized) != actual_delta_hash:
            raise RuntimeError("committed parameter delta hash mismatch")
        passed, evidence = post_gate()
        if not passed:
            raise RuntimeError("post-update gate rejected derived update")
        return {"committed": True, "rolled_back": False, "post_evidence": evidence}
    except BaseException as error:
        with torch.no_grad():
            for parameter, value in zip(parameters, before):
                parameter.copy_(value.to(parameter.device))
        return {
            "committed": False,
            "rolled_back": True,
            "reason": type(error).__name__,
            "message": str(error),
            "post_evidence": evidence,
        }


def classify_stage_b(precommit: dict[str, Any], transaction: dict[str, Any], utility_known: bool) -> dict[str, str]:
    if not precommit.get("passed"):
        return {"category": "SB-C", "next_action": "revise_constrained_optimizer_protocol"}
    if not utility_known:
        return {"category": "SB-D", "next_action": "stop_invalid_or_inconclusive"}
    if transaction.get("committed"):
        return {"category": "SB-A", "next_action": "design_10_step_projected_pilot"}
    return {"category": "SB-B", "next_action": "reduce_trust_radius_or_add_transactional_line_search"}


def build_stage_contract(
    preflight_value: dict[str, Any], config_sha256: str, run_name: str
) -> dict[str, Any]:
    return json_native({
        "schema": SCHEMA,
        "run_name": run_name,
        "config_sha256": config_sha256,
        "git": preflight_value["git"],
        "implementation": preflight_value["implementation"],
        "source_checkpoint": preflight_value["source_checkpoint"],
        "optimizer_aware_source": preflight_value["optimizer_aware_source"],
        "step813": preflight_value["step813"],
        "step814": preflight_value["step814"],
        "utility": preflight_value["utility"],
        "test_accessed": False,
    })


def validate_stage_contract(
    contract: dict[str, Any], preflight_value: dict[str, Any], config_sha256: str, run_name: str
) -> None:
    expected = build_stage_contract(preflight_value, config_sha256, run_name)
    checks = (
        ("schema", "Stage-B schema contract mismatch"),
        ("run_name", "Stage-B RunName contract mismatch"),
        ("config_sha256", "Stage-B config SHA mismatch"),
        ("git", "Stage-B HEAD contract mismatch"),
        ("implementation", "Stage-B implementation contract mismatch"),
        ("source_checkpoint", "Stage-B source checkpoint contract mismatch"),
        ("optimizer_aware_source", "Stage-B OA-B/A3 source contract mismatch"),
        ("step813", "Stage-B step813 contract mismatch"),
        ("step814", "Stage-B step814 contract mismatch"),
        ("utility", "Stage-B utility contract mismatch"),
        ("test_accessed", "Stage-B test-access contract mismatch"),
    )
    for field, message in checks:
        if contract.get(field) != expected[field]:
            raise ValueError(message)


def _binary_summary(probabilities: list[float], gold: list[int], indices: list[int]) -> dict[str, Any]:
    p = np.asarray([probabilities[index] for index in indices], dtype=np.float64)
    y = np.asarray([gold[index] for index in indices], dtype=np.int64)
    if p.size == 0:
        raise ValueError("empty utility partition")
    values = {
        "samples": int(p.size),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "confidence_mean": float(np.maximum(p, 1 - p).mean()),
        "positive_rate": float((p >= 0.5).mean()),
        "probability_mean": float(p.mean()),
        "probability_std": float(p.std()),
    }
    if values["auc"] is None or any(not math.isfinite(float(value)) for value in values.values() if isinstance(value, (int, float))):
        raise FloatingPointError("utility metric unavailable or non-finite")
    return values


def utility_gate(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    result = utility_evidence(after, before, dict(FROZEN_UTILITY_THRESHOLDS))
    result["authority"] = "t5_zero_training_decision_v2"
    result["reference_semantics"] = "post-step813 damage relative to derived step812 pre-update"
    return json_native(result)


def _rng_state_from_payload(value: dict[str, Any]) -> RngState:
    return RngState(value["python"], value["numpy"], value["torch_cpu"], value["torch_cuda"])


def _frozen_hash(model: torch.nn.Module) -> str:
    rows = []
    for name, value in model.named_parameters():
        if not value.requires_grad:
            rows.append([name, _tensor_hash(value)])
    return canonical_hash(rows)


def _component_values(runtime: dict[str, Any], forget: dict[str, torch.Tensor], retain: dict[str, torch.Tensor]) -> dict[str, float]:
    with torch.no_grad():
        values = compute_components(runtime["current"], runtime["original"], runtime["augmented"], forget, retain, 2.0)
    return {name: float(values[name].detach().cpu()) for name in LOSSES}


def isolated_rng_evaluation(
    function: Callable[[], Any], evaluation_rng: RngState, continuation_rng: RngState
) -> dict[str, Any]:
    restore_rng(evaluation_rng)
    value = function()
    consumed = rng_hashes(capture_rng())
    restore_rng(continuation_rng)
    if rng_hashes(capture_rng()) != rng_hashes(continuation_rng):
        raise RuntimeError("isolated evaluation failed to restore continuation RNG")
    return {"value": value, "evaluation_post_rng_hash": consumed, "continuation_rng_restored": True}


def _evaluate_utility(runtime: dict[str, Any], validation: JsonPromptDataset, retain_indices: list[int], device: torch.device) -> dict[str, dict[str, Any]]:
    raw = evaluate_overall_validation(runtime["current"], validation, device, 16)
    all_indices = list(range(len(raw["gold"])))
    summary = {
        "overall_validation": _binary_summary(raw["probabilities"], raw["gold"], all_indices),
        "retain_user_validation": _binary_summary(raw["probabilities"], raw["gold"], retain_indices),
    }
    del raw
    return summary


def _checkpoint_payload(
    current: torch.nn.Module,
    optimizer_state: dict[str, Any],
    continuation_rng: RngState,
    preflight_value: dict[str, Any],
    run_name: str,
) -> dict[str, Any]:
    pair814 = preflight_value["step814"]
    return {
        "schema": CHECKPOINT_SCHEMA,
        "adapter_state": {key: value.detach().cpu() for key, value in get_peft_model_state_dict(current).items()},
        "optimizer_state": optimizer_state,
        "state": {
            "step": 813,
            "next_optimizer_step": 814,
            "branch_optimizer_steps": 1,
            "executed_projected_updates": 1,
            "next_batch_hash": pair814["batch_hash"],
        },
        "rng": {
            "python": continuation_rng.python,
            "numpy": continuation_rng.numpy,
            "torch_cpu": continuation_rng.torch_cpu,
            "torch_cuda": continuation_rng.torch_cuda,
        },
        "rng_hash": rng_hashes(continuation_rng),
        "contract": {
            "schema": SCHEMA,
            "run_name": run_name,
            "source_checkpoint": preflight_value["source_checkpoint"],
            "optimizer_aware_source": preflight_value["optimizer_aware_source"],
            "git": preflight_value["git"],
            "implementation": preflight_value["implementation"],
        },
        "provenance": {
            "optimizer_moments": "advanced once with gradient-projected Forget gradient",
            "parameter_values": "updated with update-space projected delta from the same AdamW proposal",
            "ordinary_adamw_step_claimed": False,
            "test_accessed": False,
        },
        "test_accessed": False,
    }


def validate_derived_checkpoint_payload(
    payload: dict[str, Any], preflight_value: dict[str, Any]
) -> dict[str, Any]:
    adapter = payload.get("adapter_state")
    optimizer = payload.get("optimizer_state", {})
    groups, states = optimizer.get("param_groups"), optimizer.get("state")
    if payload.get("schema") != CHECKPOINT_SCHEMA or not isinstance(adapter, dict) or not adapter:
        raise ValueError("derived checkpoint schema/adapter mismatch")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(states, dict):
        raise ValueError("derived optimizer schema mismatch")
    identifiers = groups[0].get("params")
    if (
        not isinstance(identifiers, list)
        or len(identifiers) != len(adapter)
        or len(set(identifiers)) != len(identifiers)
        or set(identifiers) != set(states)
    ):
        raise ValueError("derived optimizer mapping is ambiguous")
    bindings = []
    for identifier, (name, parameter) in zip(identifiers, adapter.items()):
        state = states[identifier]
        if not torch.is_tensor(parameter) or not {"step", "exp_avg", "exp_avg_sq"} <= set(state):
            raise ValueError("derived optimizer state is incomplete")
        if float(state["step"]) != 813.0:
            raise ValueError("derived optimizer step counter is not 813")
        for key in ("exp_avg", "exp_avg_sq"):
            if state[key].shape != parameter.shape or state[key].dtype != parameter.dtype:
                raise ValueError("derived optimizer state shape/dtype mismatch")
        bindings.append([identifier, name, list(parameter.shape), str(parameter.dtype)])
    state = payload.get("state", {})
    if (
        state.get("step") != 813
        or state.get("next_optimizer_step") != 814
        or state.get("branch_optimizer_steps") != 1
        or state.get("executed_projected_updates") != 1
        or state.get("next_batch_hash") != preflight_value["step814"]["batch_hash"]
    ):
        raise ValueError("derived checkpoint continuation state mismatch")
    rng = payload.get("rng", {})
    if set(rng) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("derived checkpoint RNG fields mismatch")
    rng_state = _rng_state_from_payload(rng)
    if payload.get("rng_hash") != rng_hashes(rng_state):
        raise ValueError("derived checkpoint RNG hash mismatch")
    report = {
        "tensor_count": len(adapter),
        "parameter_count": sum(value.numel() for value in adapter.values()),
        "adapter_order_sha256": canonical_hash(list(adapter)),
        "optimizer_mapping_sha256": canonical_hash(bindings),
        "all_step_counters": 813,
        "next_batch_hash": state["next_batch_hash"],
        "hyperparameters": json_native({
            key: value for key, value in groups[0].items() if key != "params"
        }),
    }
    source = preflight_value["optimizer_mapping"]
    for key in ("tensor_count", "parameter_count", "adapter_order_sha256", "optimizer_mapping_sha256"):
        if report[key] != source[key]:
            raise ValueError(f"derived checkpoint {key} differs from step812 mapping")
    if report["hyperparameters"] != source["hyperparameters"]:
        raise ValueError("derived checkpoint optimizer hyperparameters changed")
    if payload.get("test_accessed") is not False:
        raise ValueError("derived checkpoint test safety mismatch")
    return report


def _publish_checkpoint(stage: Path, payload: dict[str, Any]) -> dict[str, Any]:
    final = stage / "checkpoints" / "step_00813"
    temporary = final.parent / f".step_00813.{uuid.uuid4().hex[:10]}.stage"
    temporary.mkdir(parents=True)
    torch.save(payload, temporary / "state.pt")
    state_sha = sha256_file(temporary / "state.pt")
    atomic_json(temporary / "manifest.json", {
        "schema": CHECKPOINT_SCHEMA,
        "step": 813,
        "next_optimizer_step": 814,
        "state_sha256": state_sha,
        "published_atomically": True,
        "test_accessed": False,
    })
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, final)
    return {"state_sha256": state_sha, "manifest_sha256": sha256_file(final / "manifest.json")}


def synthetic_dry_run(root: Path, config_path: Path, run_name: str) -> dict[str, Any]:
    config = load_stage_config(config_path, root)
    final = _resolve(root, Path(config["output_root"]) / "synthetic_runs" / _safe_name(run_name), output=True)
    if final.exists():
        raise FileExistsError("refusing to overwrite SyntheticDryRun")
    stage = final.parent / f".{run_name}.{uuid.uuid4().hex[:10]}.stage"
    stage.mkdir(parents=True)
    parameter = torch.nn.Parameter(torch.tensor([0.2, -0.4, 0.7]))
    f = torch.tensor([1.0, 2.0, 3.0])
    k = torch.tensor([1.0, 0.0, 0.0])
    s = torch.tensor([0.0, 1.0, 0.0])
    projection = direct_svd_projection(f, k, s)
    optimizer_state = {
        "state": {0: {"step": torch.tensor(812.0), "exp_avg": torch.zeros(3), "exp_avg_sq": torch.ones(3)}},
        "param_groups": [{"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01, "amsgrad": False, "maximize": False, "foreach": None, "capturable": False, "differentiable": False, "fused": None, "decoupled_weight_decay": True, "params": [0]}],
    }
    proposal = shadow_adamw_proposal([parameter], optimizer_state, projection["safe"])
    projected = update_space_projection(proposal["delta"], projection["basis"])
    after, actual = materialize_dtype_delta([parameter], projected)
    gate = precommit_gate(f, k, s, proposal["delta"], projected, actual, config, expected_a3=False)
    transaction = commit_or_rollback([parameter], after, gate["actual_dtype_delta_hash"], post_gate=lambda: (True, {"toy": "passed"})) if gate["passed"] else {"committed": False, "rolled_back": False}
    rollback_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    rollback = commit_or_rollback([rollback_parameter], [torch.tensor([2.0])], _tensor_hash(torch.tensor([1.0], dtype=torch.float64)), post_gate=lambda: (False, {}))
    if rollback_parameter.item() != 1.0 or not rollback["rolled_back"]:
        raise RuntimeError("toy rollback failed")
    checkpoint_payload = {
        "schema": CHECKPOINT_SCHEMA,
        "adapter_state": {"toy": parameter.detach().clone()},
        "optimizer_state": proposal["optimizer_state"],
        "state": {"step": 813, "next_optimizer_step": 814},
        "test_accessed": False,
    }
    checkpoint = _publish_checkpoint(stage, checkpoint_payload)
    result = {
        "schema": SCHEMA,
        "mode": "SyntheticDryRun",
        "precommit": gate,
        "transaction": transaction,
        "rollback_verified": True,
        "checkpoint": checkpoint,
        "shadow_optimizer_steps_executed": 1,
        "authoritative_optimizer_step_calls": 0,
        "logical_optimizer_steps_proposed": 1,
        "logical_optimizer_steps_committed": 1 if transaction.get("committed") else 0,
        "gradient_vectors_persisted": False,
        "delta_vectors_persisted": False,
        "logits_persisted": False,
        "tokens_persisted": False,
        "raw_samples_persisted": False,
        "model_loaded": False,
        "test_accessed": False,
    }
    atomic_json(stage / "synthetic_result.json", result)
    atomic_json(stage / "manifest.json", {"schema": SCHEMA, "result_sha256": sha256_file(stage / "synthetic_result.json"), "published_atomically": True, "test_accessed": False})
    atomic_text(stage / "COMPLETED", "SYNTHETIC_STAGE_B_COMPLETED\n")
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage, final)
    return {**result, "run_dir": str(final)}


def _publish_rejected(stage: Path, final: Path, contract: dict[str, Any], before: dict[str, Any], result: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    result = {**result, "logical_optimizer_steps_committed": 0, "resumable_checkpoint_published": False, "test_accessed": False}
    atomic_json(stage / "contract.json", contract)
    atomic_json(stage / "before.json", before)
    atomic_json(stage / "step_result.json", result)
    atomic_json(stage / "provenance.json", provenance)
    atomic_json(stage / "manifest.json", {
        "schema": SCHEMA,
        "status": "REJECTED",
        "contract_sha256": sha256_file(stage / "contract.json"),
        "before_sha256": sha256_file(stage / "before.json"),
        "step_result_sha256": sha256_file(stage / "step_result.json"),
        "provenance_sha256": sha256_file(stage / "provenance.json"),
        "published_atomically": True,
        "test_accessed": False,
    })
    atomic_text(stage / "REJECTED", "STAGE_B_UPDATE_REJECTED\n")
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage, final)
    return {**result, "run_dir": str(final)}


def publish_full(root: Path, config_path: Path, run_name: str) -> dict[str, Any]:
    pre = preflight(root, config_path)
    require_clean_git(pre["git"], "Stage-B Full")
    config = load_stage_config(config_path, root)
    final = _resolve(root, Path(config["output_root"]) / "full_runs" / _safe_name(run_name), output=True)
    if final.exists():
        raise FileExistsError("refusing to overwrite Stage-B Full")
    stage = final.parent / f".{run_name}.{uuid.uuid4().hex[:10]}.stage"
    stage.mkdir(parents=True)
    outer_rng = capture_rng()
    runtime = None
    checkpoint_path = _resolve(root, config["source_checkpoint"])
    source_hash_before = {name: sha256_file(checkpoint_path / name) for name in ("state.pt", "manifest.json")}
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        runtime = _load_runtime(root, root / "configs" / "t5_step812_gradient_geometry_audit_v1.yaml", device)
        names, parameters = _runtime_binding(runtime)
        validate_optimizer_mapping(runtime["payload"]["adapter_state"], runtime["payload"]["optimizer_state"])
        source_trainable = [_tensor_hash(value) for value in parameters]
        frozen_before = _frozen_hash(runtime["current"])
        teachers_before = {name: _tensor_state_hash(runtime[name]) for name in ("original", "augmented")}
        teacher_files_before = {
            name: sha256_file(Path(pre["teachers"][name]["path"]))
            for name in ("original", "augmented")
        }
        restore_rng(_rng_state_from_payload(runtime["payload"]["rng"]))
        runtime["current"].train()
        pair = catalog_pair(0, len(runtime["forget"]), len(runtime["retain"]), 16, 42)
        forget = move_batch(_batch(runtime["forget"], pair["forget_indices"]), device)
        retain = move_batch(_batch(runtime["retain"], pair["retain_indices"]), device)
        components = compute_components(runtime["current"], runtime["original"], runtime["augmented"], forget, retain, 2.0)
        gradients = _flatten_gradients(components, parameters)
        continuation_rng = capture_rng()
        f, k, s = (gradients[name] for name in LOSSES)
        projection = direct_svd_projection(f, k, s)
        proposal = shadow_adamw_proposal([value.detach().cpu() for value in parameters], runtime["payload"]["optimizer_state"], projection["safe"])
        projected = update_space_projection(proposal["delta"], projection["basis"])
        after_values, actual_delta = materialize_dtype_delta([value.detach().cpu() for value in parameters], projected)
        gate = precommit_gate(
            f,
            k,
            s,
            proposal["delta"],
            projected,
            actual_delta,
            config,
            expected_a3=True,
            effectiveness_reference_dot=pre["optimizer_aware_source"]["a3"]["a0_forget_dot"],
        )
        contract = build_stage_contract(pre, config["_sha256"], run_name)
        before = {"trainable_tensor_hashes": source_trainable, "frozen_hash": frozen_before, "teacher_hashes": teachers_before, "continuation_rng_hash": rng_hashes(continuation_rng), "source_checkpoint_sha": source_hash_before, "test_accessed": False}
        provenance = {"protocol": "AdamW moments advanced with gradient-projected Forget; parameters updated with update-space projected proposal", "ordinary_adamw_step_claimed": False, "retrain_used_for_parameter_selection": False, "utility_authority": "t5_zero_training_decision_v2", "test_accessed": False}
        if not gate["passed"]:
            classification = classify_stage_b(gate, {"committed": False}, True)
            return _publish_rejected(stage, final, contract, before, {"classification": classification, "precommit": gate}, provenance)
        replay_rng = continuation_rng
        pre_replay = isolated_rng_evaluation(
            lambda: _component_values(runtime, forget, retain), replay_rng, continuation_rng
        )
        pre_losses = pre_replay["value"]
        base = load_config(_resolve(root, config["base_config"]), root)
        validation = JsonPromptDataset(Path(base["paths"]["validation"]), runtime["tokenizer"])
        _, indices, _ = _data_lineage(root, base, _resolve(root, config["protocol_root"]))
        utility_before = isolated_rng_evaluation(
            lambda: _evaluate_utility(runtime, validation, indices["retain_user_validation"], device),
            continuation_rng,
            continuation_rng,
        )["value"]

        def post_gate() -> tuple[bool, dict[str, Any]]:
            post_replay = isolated_rng_evaluation(
                lambda: _component_values(runtime, forget, retain), replay_rng, continuation_rng
            )
            post_losses = post_replay["value"]
            utility_after = isolated_rng_evaluation(
                lambda: _evaluate_utility(runtime, validation, indices["retain_user_validation"], device),
                continuation_rng,
                continuation_rng,
            )["value"]
            utility = utility_gate(utility_before, utility_after)
            loss_delta = {name: post_losses[name] - pre_losses[name] for name in LOSSES}
            predicted = {name: gate["actual_dtype_delta"][name]["dot"] for name in LOSSES}
            paired = {"pre": pre_losses, "post": post_losses, "actual_delta": loss_delta, "first_order_prediction": predicted, "prediction_error": {name: loss_delta[name] - predicted[name] for name in LOSSES}, "actual_predicted_ratio": {name: (None if predicted[name] == 0 else loss_delta[name] / predicted[name]) for name in LOSSES}, "same_dropout_rng": pre_replay["evaluation_post_rng_hash"] == post_replay["evaluation_post_rng_hash"], "continuation_rng_preserved": rng_hashes(capture_rng()) == rng_hashes(continuation_rng)}
            invariants = {
                "only_lora_changed": any(
                    _tensor_hash(value) != old
                    for value, old in zip(parameters, source_trainable)
                ),
                "frozen_base_unchanged": _frozen_hash(runtime["current"]) == frozen_before,
                "teachers_unchanged": all(
                    _tensor_state_hash(runtime[name]) == teachers_before[name]
                    for name in ("original", "augmented")
                ),
                "teacher_files_unchanged": all(
                    sha256_file(Path(pre["teachers"][name]["path"])) == teacher_files_before[name]
                    for name in ("original", "augmented")
                ),
                "source_checkpoint_unchanged": all(
                    sha256_file(checkpoint_path / name) == source_hash_before[name]
                    for name in source_hash_before
                ),
                "parameter_grads_none": all(value.grad is None for value in parameters),
                "continuation_rng_preserved": paired["continuation_rng_preserved"],
            }
            passed = loss_delta["L_forget"] < 0 and utility["utility_pass"] and all(invariants.values())
            return passed, {
                "paired_replay": paired,
                "utility_before": utility_before,
                "utility_after": utility_after,
                "utility_gate": utility,
                "invariants": invariants,
            }

        transaction = commit_or_rollback(parameters, after_values, gate["actual_dtype_delta_hash"], post_gate=post_gate)
        classification = classify_stage_b(gate, transaction, True)
        common_result = {"classification": classification, "precommit": gate, "transaction": transaction, "shadow_optimizer_steps_executed": 1, "authoritative_optimizer_step_calls": 0, "logical_optimizer_steps_proposed": 1, "logical_optimizer_steps_committed": 1 if transaction.get("committed") else 0, "gradient_vectors_persisted": False, "delta_vectors_persisted": False, "logits_persisted": False, "tokens_persisted": False, "raw_samples_persisted": False, "test_accessed": False}
        if not transaction.get("committed"):
            return _publish_rejected(stage, final, contract, before, common_result, provenance)
        invariants = transaction["post_evidence"]["invariants"]
        checkpoint_payload = _checkpoint_payload(runtime["current"], proposal["optimizer_state"], continuation_rng, pre, run_name)
        derived_mapping = validate_derived_checkpoint_payload(checkpoint_payload, pre)
        checkpoint = _publish_checkpoint(stage, checkpoint_payload)
        result = {**common_result, "invariants": invariants, "checkpoint": checkpoint, "derived_optimizer_mapping": derived_mapping, "resumable_checkpoint_published": True, "new_state": {"step": 813, "next_optimizer_step": 814, "branch_optimizer_steps": 1, "executed_projected_updates": 1, "next_batch_hash": pre["step814"]["batch_hash"]}}
        atomic_json(stage / "contract.json", contract)
        atomic_json(stage / "before.json", before)
        atomic_json(stage / "step_result.json", result)
        atomic_json(stage / "provenance.json", provenance)
        atomic_json(stage / "manifest.json", {"schema": SCHEMA, "status": "COMPLETED", "contract_sha256": sha256_file(stage / "contract.json"), "before_sha256": sha256_file(stage / "before.json"), "step_result_sha256": sha256_file(stage / "step_result.json"), "provenance_sha256": sha256_file(stage / "provenance.json"), "checkpoint_manifest_sha256": checkpoint["manifest_sha256"], "published_atomically": True, "test_accessed": False})
        atomic_text(stage / "COMPLETED", "STAGE_B_STEP813_COMPLETED\n")
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, final)
        return {**result, "run_dir": str(final)}
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        restore_rng(outer_rng)
        if runtime is not None:
            del runtime
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def verify_publication(root: Path, config_path: Path, run_name: str) -> dict[str, Any]:
    config = load_stage_config(config_path, root)
    pre = preflight(root, config_path)
    require_clean_git(pre["git"], "Stage-B Analyze")
    source = _resolve(root, Path(config["output_root"]) / "full_runs" / _safe_name(run_name), output=True)
    completed, rejected = source / "COMPLETED", source / "REJECTED"
    if completed.exists() == rejected.exists():
        raise ValueError("Stage-B publication status marker mismatch")
    expected_marker = "STAGE_B_STEP813_COMPLETED\n" if completed.exists() else "STAGE_B_UPDATE_REJECTED\n"
    marker = completed if completed.exists() else rejected
    if marker.read_text(encoding="utf-8") != expected_marker:
        raise ValueError("Stage-B publication status marker content mismatch")
    required = {"contract.json", "before.json", "step_result.json", "provenance.json", "manifest.json", "COMPLETED" if completed.exists() else "REJECTED"}
    if completed.exists():
        required.add("checkpoints")
    if {item.name for item in source.iterdir()} != required:
        raise ValueError("Stage-B publication inventory mismatch")
    manifest = _read_json(source / "manifest.json")
    for key, name in (("contract_sha256", "contract.json"), ("before_sha256", "before.json"), ("step_result_sha256", "step_result.json"), ("provenance_sha256", "provenance.json")):
        if manifest.get(key) != sha256_file(source / name):
            raise ValueError(f"Stage-B {name} SHA mismatch")
    result = _read_json(source / "step_result.json")
    contract = _read_json(source / "contract.json")
    validate_stage_contract(contract, pre, config["_sha256"], run_name)
    if manifest.get("published_atomically") is not True or manifest.get("test_accessed") is not False or result.get("test_accessed") is not False:
        raise ValueError("Stage-B publication safety mismatch")
    if completed.exists():
        checkpoint = source / "checkpoints" / "step_00813"
        cp_manifest = _read_json(checkpoint / "manifest.json")
        if cp_manifest.get("state_sha256") != sha256_file(checkpoint / "state.pt") or manifest.get("checkpoint_manifest_sha256") != sha256_file(checkpoint / "manifest.json"):
            raise ValueError("Stage-B checkpoint SHA mismatch")
        payload = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=False)
        validate_derived_checkpoint_payload(payload, pre)
    elif result.get("resumable_checkpoint_published") is not False:
        raise ValueError("REJECTED publication claims resumable checkpoint")
    return {"source": source, "result": result, "manifest": manifest, "preflight": pre}


def analyze(root: Path, config_path: Path, run_name: str) -> dict[str, Any]:
    verified = verify_publication(root, config_path, run_name)
    final = _resolve(root, Path(load_stage_config(config_path, root)["output_root"]) / "analysis_runs" / _safe_name(run_name), output=True)
    if final.exists():
        raise FileExistsError("refusing to overwrite Stage-B Analyze")
    stage = final.parent / f".{run_name}.{uuid.uuid4().hex[:10]}.stage"
    stage.mkdir(parents=True)
    result = {"schema": ANALYSIS_SCHEMA, "run_name": run_name, **verified["result"]["classification"], "source_manifest_sha256": sha256_file(verified["source"] / "manifest.json"), "test_accessed": False}
    atomic_json(stage / "analysis.json", result)
    atomic_json(stage / "manifest.json", {"schema": ANALYSIS_SCHEMA, "analysis_sha256": sha256_file(stage / "analysis.json"), "source_manifest_sha256": result["source_manifest_sha256"], "published_atomically": True, "test_accessed": False})
    atomic_text(stage / "COMPLETED", "STAGE_B_ANALYSIS_COMPLETED\n")
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage, final)
    return {**result, "analysis_dir": str(final)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reversible T5 step813 update-space projection")
    parser.add_argument("--mode", choices=("Preflight", "SyntheticDryRun", "Full", "Analyze"), default="Preflight")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-name")
    args = parser.parse_args()
    root, config = args.project_root.resolve(), args.config.resolve()
    if args.mode == "Preflight":
        result = preflight(root, config)
    else:
        if not args.run_name:
            parser.error(f"{args.mode} requires --run-name")
        if args.mode == "SyntheticDryRun":
            result = synthetic_dry_run(root, config, args.run_name)
        elif args.mode == "Full":
            result = publish_full(root, config, args.run_name)
        else:
            result = analyze(root, config, args.run_name)
    print(json.dumps(json_native(result), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
