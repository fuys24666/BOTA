from __future__ import annotations

import argparse
import copy
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from src.diagnostics.git_provenance import git_provenance, implementation_provenance, require_clean_git
from src.diagnostics.t5_full_runner import _batch, evaluate_overall_validation
from src.diagnostics.t5_reconstructed_official import JsonPromptDataset, compute_components, forced_logits, load_config, move_batch, sha256_file, teacher_cross_entropy
from src.diagnostics.t5_step812_gradient_geometry import _load_runtime, _tensor_state_hash, catalog_pair
from src.diagnostics.t5_step813_optimizer_aware_audit import _flatten_gradients, _runtime_binding, _tensor_hash, direct_svd_projection, directional_metrics, json_native
from src.diagnostics.t5_step813_update_space_stage_b import (
    _binary_summary,
    _frozen_hash,
    _rng_state_from_payload,
    isolated_rng_evaluation,
    materialize_dtype_delta,
    update_space_projection,
)
from src.diagnostics.t5_trajectory_diagnostics import capture_rng, restore_rng, rng_hashes
from src.diagnostics.t5_zero_training_audit import _data_lineage
from src.diagnostics.t5_zero_training_decision_v2 import COLLAPSE_STD_EPSILON, FROZEN_UTILITY_THRESHOLDS, utility_evidence

SCHEMA = "t5-transactional-projected-pilot-10step-v1"
CHECKPOINT_SCHEMA = "t5-transactional-projected-pilot-checkpoint-v1"
ANALYSIS_SCHEMA = "t5-transactional-projected-pilot-analysis-v1"
FIRST_STEP, LAST_STEP = 814, 822
TRUST_SCALES = (1.0, 0.5, 0.25)
SOURCE_FULL_MANIFEST_SHA = "84f71f935405afa721a4d70d4d626e700bf1a3c58f2118e287e82ed0c50e3a8d"
SOURCE_ANALYSIS_SHA = "433f740f94e46c9c35896e694d3184778b3f19c2e5af89fe71fb3185051ed68e"
SOURCE_STATE_SHA = "e925b6293bbd62d47bd648695178812cfcefb762d5fe516e5e8a65683163b812"
SOURCE_CHECKPOINT_MANIFEST_SHA = "2abcbca5ac07163f167aa75fcd27b1538cde27a9cbcb3f09b350315a1ebe720a"
FIXED_BASELINE = {
    "overall_validation": {"auc": 0.7479758724718022, "log_loss": 0.5946125198210375},
    "retain_user_validation": {"auc": 0.74908276115748, "log_loss": 0.5929660234335364},
}
LOSSES = ("L_forget", "L_retain_KL", "L_sup")
IMPLEMENTATION_FILES = (
    "src/diagnostics/t5_projected_pilot_10step.py",
    "configs/t5_projected_pilot_10step_v1.yaml",
    "scripts/diagnostics/t5_projected_pilot_10step_v1.ps1",
    "docs/t5_projected_pilot_10step_v1.md",
    "configs/t5_e2urec_diagnostics_v1.yaml",
    "configs/t5_step812_gradient_geometry_audit_v1.yaml",
    "src/diagnostics/t5_step813_update_space_stage_b.py",
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
        handle.write(value); handle.flush(); os.fsync(handle.fileno())
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


def batch_plan() -> dict[str, dict[str, Any]]:
    return {str(step): catalog_pair(step - 813, 12982, 47018, 16, 42) for step in range(FIRST_STEP, LAST_STEP + 1)}


def next_batch_hash(step: int) -> str:
    """Return the authoritative next-batch hash without executing that batch."""
    if step < FIRST_STEP or step > LAST_STEP:
        raise ValueError("checkpoint step outside pilot range")
    return catalog_pair(step - 812, 12982, 47018, 16, 42)["batch_hash"]


def shadow_adamw_proposal_for_step(
    official_parameters: list[torch.Tensor],
    optimizer_state: dict[str, Any],
    flat_gradient: torch.Tensor,
    expected_source_step: int,
) -> dict[str, Any]:
    """Advance exactly one isolated AdamW state without touching official tensors."""
    if any(getattr(value, "grad", None) is not None for value in official_parameters):
        raise ValueError("authoritative parameters must have grad=None")
    source_hashes = [_tensor_hash(value) for value in official_parameters]
    source_steps = [float(item["step"]) for item in optimizer_state.get("state", {}).values()]
    if not source_steps or any(value != float(expected_source_step) for value in source_steps):
        raise ValueError("source optimizer counters do not match current checkpoint")
    shadows = [torch.nn.Parameter(value.detach().cpu().clone()) for value in official_parameters]
    group = optimizer_state["param_groups"][0]
    optimizer = torch.optim.AdamW(
        shadows,
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
    offset = 0
    before = [value.detach().clone() for value in shadows]
    for parameter in shadows:
        count = parameter.numel()
        parameter.grad = flat_gradient[offset : offset + count].reshape(parameter.shape).to(parameter.dtype)
        offset += count
    if offset != flat_gradient.numel():
        raise ValueError("flat gradient does not match optimizer parameter order")
    optimizer.step()
    advanced = optimizer.state_dict()
    advanced_steps = [float(item["step"]) for item in advanced["state"].values()]
    if any(value != float(expected_source_step + 1) for value in advanced_steps):
        raise RuntimeError("shadow AdamW did not advance exactly once")
    delta = torch.cat(
        [(value.detach() - prior).double().reshape(-1) for value, prior in zip(shadows, before)]
    )
    if any(_tensor_hash(value) != digest for value, digest in zip(official_parameters, source_hashes)):
        raise RuntimeError("shadow AdamW changed authoritative parameter")
    return {
        "delta": delta,
        "optimizer_state": advanced,
        "delta_hash": _tensor_hash(delta),
        "delta_norm": float(torch.linalg.vector_norm(delta)),
        "shadow_optimizer_steps_executed": 1,
        "source_optimizer_step": expected_source_step,
        "next_optimizer_step": expected_source_step + 1,
    }


def load_pilot_config(path: Path, root: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("development_only") is not True or value.get("test_access_policy") != "forbidden":
        raise ValueError("pilot config schema/scope mismatch")
    if value.get("expected") != {
        "stage_b_full_manifest_sha256": SOURCE_FULL_MANIFEST_SHA,
        "stage_b_analysis_sha256": SOURCE_ANALYSIS_SHA,
        "step813_state_sha256": SOURCE_STATE_SHA,
        "step813_manifest_sha256": SOURCE_CHECKPOINT_MANIFEST_SHA,
        "source_category": "SB-A",
        "source_next_action": "design_10_step_projected_pilot",
    }:
        raise ValueError("pilot source authority changed")
    if value.get("steps") != {"first": 814, "last": 822, "maximum_additional_updates": 9, "total_projected_updates_at_completion": 10}:
        raise ValueError("pilot step budget changed")
    if value.get("trust_scales") != [1.0, 0.5, 0.25]:
        raise ValueError("pilot trust-scale order changed")
    if value.get("projection") != {"algorithm": "direct_float64_retain_matrix_svd", "relative_singular_tolerance": 1e-10, "eta_f_min": 0.10}:
        raise ValueError("pilot projection protocol changed")
    if value.get("gates") != {"normalized_retain_tolerance": 1e-8, "forget_descent_zero_tolerance": 1e-12, "forget_effectiveness_min": 0.10}:
        raise ValueError("pilot directional gates changed")
    if value.get("utility_baseline_step812") != FIXED_BASELINE:
        raise ValueError("fixed step812 utility baseline changed")
    thresholds = value.get("utility_thresholds", {})
    if any(thresholds.get(key) != expected for key, expected in FROZEN_UTILITY_THRESHOLDS.items()) or thresholds.get("collapse_std_epsilon") != COLLAPSE_STD_EPSILON:
        raise ValueError("decision-v2 utility authority changed")
    value["_path"], value["_sha256"] = str(path.resolve()), sha256_file(path)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_stage_b_source(full: Path, analysis: Path) -> dict[str, Any]:
    full_required = {"contract.json", "before.json", "step_result.json", "provenance.json", "manifest.json", "COMPLETED", "checkpoints"}
    analysis_required = {"analysis.json", "manifest.json", "COMPLETED"}
    if not full.is_dir() or {item.name for item in full.iterdir()} != full_required:
        raise ValueError("Stage-B Full inventory mismatch")
    if not analysis.is_dir() or {item.name for item in analysis.iterdir()} != analysis_required:
        raise ValueError("Stage-B Analyze inventory mismatch")
    if (full / "COMPLETED").read_text(encoding="utf-8") != "STAGE_B_STEP813_COMPLETED\n" or (analysis / "COMPLETED").read_text(encoding="utf-8") != "STAGE_B_ANALYSIS_COMPLETED\n":
        raise ValueError("Stage-B completion marker mismatch")
    if sha256_file(full / "manifest.json") != SOURCE_FULL_MANIFEST_SHA or sha256_file(analysis / "analysis.json") != SOURCE_ANALYSIS_SHA:
        raise ValueError("Stage-B source SHA mismatch")
    manifest, result, decision = _read_json(full / "manifest.json"), _read_json(full / "step_result.json"), _read_json(analysis / "analysis.json")
    analysis_manifest = _read_json(analysis / "manifest.json")
    if analysis_manifest != {
        "analysis_sha256": SOURCE_ANALYSIS_SHA,
        "published_atomically": True,
        "schema": "t5-step813-update-space-stage-b-analysis-v1",
        "source_manifest_sha256": SOURCE_FULL_MANIFEST_SHA,
        "test_accessed": False,
    }:
        raise ValueError("Stage-B Analyze manifest mismatch")
    for key, name in (("contract_sha256", "contract.json"), ("before_sha256", "before.json"), ("step_result_sha256", "step_result.json"), ("provenance_sha256", "provenance.json")):
        if manifest.get(key) != sha256_file(full / name):
            raise ValueError(f"Stage-B {name} SHA mismatch")
    if decision.get("source_manifest_sha256") != sha256_file(full / "manifest.json") or decision.get("category") != "SB-A" or decision.get("next_action") != "design_10_step_projected_pilot":
        raise ValueError("Stage-B decision/source binding mismatch")
    if result.get("logical_optimizer_steps_committed") != 1 or result.get("resumable_checkpoint_published") is not True or result.get("test_accessed") is not False:
        raise ValueError("Stage-B committed-step safety mismatch")
    checkpoint = full / "checkpoints" / "step_00813"
    if sha256_file(checkpoint / "state.pt") != SOURCE_STATE_SHA or sha256_file(checkpoint / "manifest.json") != SOURCE_CHECKPOINT_MANIFEST_SHA or manifest.get("checkpoint_manifest_sha256") != SOURCE_CHECKPOINT_MANIFEST_SHA:
        raise ValueError("Stage-B step813 checkpoint SHA mismatch")
    payload = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=False)
    state, adapter, optimizer = payload.get("state", {}), payload.get("adapter_state", {}), payload.get("optimizer_state", {})
    if state.get("step") != 813 or state.get("next_optimizer_step") != 814 or state.get("next_batch_hash") != batch_plan()["814"]["batch_hash"]:
        raise ValueError("Stage-B step813 continuation mismatch")
    if len(adapter) != 144 or sum(value.numel() for value in adapter.values()) != 1_769_472:
        raise ValueError("Stage-B LoRA inventory mismatch")
    if not optimizer.get("state") or any(float(item["step"]) != 813 for item in optimizer["state"].values()):
        raise ValueError("Stage-B optimizer counters mismatch")
    utility_before = result.get("transaction", {}).get("post_evidence", {}).get("utility_before")
    for split, expected in FIXED_BASELINE.items():
        if utility_before is None or any(utility_before[split].get(key) != number for key, number in expected.items()):
            raise ValueError("Stage-B fixed utility baseline mismatch")
    if payload.get("test_accessed") is not False:
        raise ValueError("Stage-B checkpoint test safety mismatch")
    before, contract = _read_json(full / "before.json"), _read_json(full / "contract.json")
    step812 = Path(contract["source_checkpoint"]["directory"])
    expected_step812 = before["source_checkpoint_sha"]
    if (
        contract["source_checkpoint"].get("state_sha256") != expected_step812["state.pt"]
        or contract["source_checkpoint"].get("manifest_sha256") != expected_step812["manifest.json"]
        or sha256_file(step812 / "state.pt") != expected_step812["state.pt"]
        or sha256_file(step812 / "manifest.json") != expected_step812["manifest.json"]
        or result.get("invariants", {}).get("source_checkpoint_unchanged") is not True
    ):
        raise ValueError("Stage-B source step812 checkpoint changed")
    return json_native({
        "full_path": str(full), "analysis_path": str(analysis),
        "full_manifest_sha256": SOURCE_FULL_MANIFEST_SHA,
        "analysis_sha256": SOURCE_ANALYSIS_SHA,
        "checkpoint_path": str(checkpoint),
        "checkpoint_state_sha256": SOURCE_STATE_SHA,
        "checkpoint_manifest_sha256": SOURCE_CHECKPOINT_MANIFEST_SHA,
        "source_step812": {"path": str(step812), **expected_step812, "unchanged": True},
        "category": "SB-A", "next_action": "design_10_step_projected_pilot",
        "step": 813, "next_optimizer_step": 814,
        "adapter_tensors": 144, "adapter_parameters": 1_769_472,
        "utility_baseline_step812": FIXED_BASELINE,
        "test_accessed": False,
    })


def preflight(root: Path, config_path: Path, *, git_function=git_provenance, implementation_function=implementation_provenance) -> dict[str, Any]:
    config = load_pilot_config(config_path, root)
    source = validate_stage_b_source(_resolve(root, config["stage_b_full"]), _resolve(root, config["stage_b_analysis"]))
    base = load_config(_resolve(root, config["base_config"]), root)
    lineage, indices, _ = _data_lineage(root, base, _resolve(root, config["protocol_root"]))
    plan = batch_plan()
    if list(map(int, plan)) != list(range(814, 823)):
        raise ValueError("pilot batch plan range mismatch")
    return json_native({
        "schema": SCHEMA, "mode": "Preflight", "development_only": True,
        "config_sha256": config["_sha256"], "git": git_function(root),
        "implementation": implementation_function(root, IMPLEMENTATION_FILES),
        "stage_b_source": source, "fixed_utility_baseline": FIXED_BASELINE,
        "batch_plan": {step: {"batch_hash": item["batch_hash"], "catalog_index": item["catalog_index"], "forget_epoch": item["forget_epoch"], "forget_position": item["forget_position"], "retain_epoch": item["retain_epoch"], "retain_position": item["retain_position"]} for step, item in plan.items()},
        "step_budget": {"first": 814, "last": 822, "attempted_steps_max": 9, "step823_executed": False},
        "optimizer_mapping": {"tensors": 144, "parameters": 1_769_472, "source_counter": 813},
        "lineage": lineage,
        "validation_partitions": {"overall": len(indices["overall_validation"]), "forget_user": len(indices["forget_user_validation"]), "retain_user": len(indices["retain_user_validation"])},
        "trust_scales": list(TRUST_SCALES), "utility_thresholds": {**FROZEN_UTILITY_THRESHOLDS, "collapse_std_epsilon": COLLAPSE_STD_EPSILON},
        "runtime_protocol": {
            "protocol": base["protocol_name"],
            "optimizer": base["training"]["optimizer"],
            "learning_rate": base["training"]["learning_rate"],
            "batch_size": base["training"]["effective_batch_size"],
            "alpha": base["training"]["alpha"],
            "seed": base["training"]["seed"],
            "sampling": base["training"]["sampler"],
            "lora": base["lora"],
            "attention_implementation": "eager",
            "retrain_used_for_selection": False,
        },
        "model_loaded": False, "retrain_loaded": False, "test_loader_built": False, "test_accessed": False,
    })


def cumulative_utility_gate(candidate: dict[str, dict[str, Any]], previous: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline = {
        split: {
            **candidate[split],
            "auc": FIXED_BASELINE[split]["auc"],
            "log_loss": FIXED_BASELINE[split]["log_loss"],
        }
        for split in ("overall_validation", "retain_user_validation")
    }
    cumulative = utility_evidence(candidate, baseline, dict(FROZEN_UTILITY_THRESHOLDS))
    incremental = {
        "overall_auc_damage": previous["overall_validation"]["auc"] - candidate["overall_validation"]["auc"],
        "overall_log_loss_damage": candidate["overall_validation"]["log_loss"] - previous["overall_validation"]["log_loss"],
        "retain_user_auc_damage": previous["retain_user_validation"]["auc"] - candidate["retain_user_validation"]["auc"],
        "retain_user_log_loss_damage": candidate["retain_user_validation"]["log_loss"] - previous["retain_user_validation"]["log_loss"],
    }
    budget = {
        key: {
            "used": cumulative[key],
            "limit": FROZEN_UTILITY_THRESHOLDS[f"{key}_max"],
            "used_fraction": cumulative[key] / FROZEN_UTILITY_THRESHOLDS[f"{key}_max"],
            "remaining": FROZEN_UTILITY_THRESHOLDS[f"{key}_max"] - cumulative[key],
        }
        for key in ("overall_auc_damage", "retain_user_auc_damage", "overall_log_loss_damage", "retain_user_log_loss_damage")
    }
    return json_native({**cumulative, "fixed_baseline_step": 812, "incremental_damage_vs_previous": incremental, "budget": budget})


def forget_evidence_gate(previous: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Apply the preregistered Forget safety gate and expose metric disagreement."""
    forced_improved = candidate["forced_teacher_ce"] <= previous["forced_teacher_ce"]
    moved_from_original = (
        candidate["student_original_jsd"] >= previous["student_original_jsd"]
        or candidate["original_prediction_agreement"] <= previous["original_prediction_agreement"]
    )
    direction_conflict = forced_improved and (
        candidate["student_original_jsd"] < previous["student_original_jsd"]
        and candidate["original_prediction_agreement"] > previous["original_prediction_agreement"]
    )
    checks = {
        "forced_teacher_ce_not_worse": forced_improved,
        "interpretable_progress": forced_improved and moved_from_original,
        "finite": candidate["finite"] is True,
        "probability_not_collapsed": candidate["probability_collapse"] is False,
    }
    return {
        "passed": all(checks.values()) and not direction_conflict,
        "checks": checks,
        "forget_evidence_conflict": direction_conflict,
        "selection_reference": "previous_accepted_checkpoint",
        "retrain_used_for_selection": False,
    }


def directional_gate(f: torch.Tensor, k: torch.Tensor, s: torch.Tensor, proposal_delta: torch.Tensor, actual: torch.Tensor, eta_f: float | None, config: dict[str, Any]) -> dict[str, Any]:
    proposal, metrics = directional_metrics(f, k, s, proposal_delta), directional_metrics(f, k, s, actual)
    denominator = -proposal["L_forget"]["dot"]
    effectiveness = None if denominator <= 0 else -metrics["L_forget"]["dot"] / denominator
    gates = config["gates"]
    checks = {
        "finite": all(math.isfinite(float(value)) for value in (actual.norm(), metrics["L_forget"]["dot"], metrics["L_retain_KL"]["dot"], metrics["L_sup"]["dot"])),
        "nonzero": metrics["delta_norm"] > 0,
        "forget_descent": metrics["L_forget"]["dot"] < -gates["forget_descent_zero_tolerance"],
        "retain_kl": metrics["L_retain_KL"]["normalized"] is not None and metrics["L_retain_KL"]["normalized"] <= gates["normalized_retain_tolerance"],
        "retain_sup": metrics["L_sup"]["normalized"] is not None and metrics["L_sup"]["normalized"] <= gates["normalized_retain_tolerance"],
        "effectiveness": effectiveness is not None and effectiveness >= gates["forget_effectiveness_min"],
        "eta_f": eta_f is not None and eta_f >= config["projection"]["eta_f_min"],
    }
    return json_native({"passed": all(checks.values()), "checks": checks, "directional": metrics, "effectiveness": effectiveness, "actual_delta_hash": _tensor_hash(actual)})


def select_first_passing_scale(
    scales: tuple[float, ...], trial: Callable[[float], tuple[bool, dict[str, Any]]]
) -> dict[str, Any]:
    if scales != TRUST_SCALES:
        raise ValueError("trust scales must be tried in preregistered order")
    records = []
    for scale in scales:
        passed, evidence = trial(scale)
        records.append({"scale": scale, "passed": bool(passed), "evidence": evidence})
        if passed:
            return {"accepted": True, "accepted_scale": scale, "trials": records}
    return {"accepted": False, "accepted_scale": None, "trials": records}


def transactional_scale_search(
    scales: tuple[float, ...],
    restore_pre_step: Callable[[], None],
    trial: Callable[[float], tuple[bool, dict[str, Any]]],
) -> dict[str, Any]:
    """Run ordered trials from one snapshot; leave only an accepted trial applied."""
    if scales != TRUST_SCALES:
        raise ValueError("trust scales must be tried in preregistered order")
    records = []
    for scale in scales:
        restore_pre_step()
        try:
            passed, evidence = trial(scale)
        except BaseException:
            restore_pre_step()
            raise
        records.append({"scale": scale, "passed": bool(passed), "evidence": evidence})
        if passed:
            return {"accepted": True, "accepted_scale": scale, "trials": records}
        restore_pre_step()
    return {"accepted": False, "accepted_scale": None, "trials": records}


def classify_pilot(status: str, final_step: int, scales: list[float], evidence_conflict: bool = False) -> dict[str, str]:
    if evidence_conflict or status not in {"COMPLETED", "STOPPED_SAFELY"}:
        return {"category": "P10-D", "next_action": "stop_invalid_or_inconclusive"}
    if status == "STOPPED_SAFELY":
        return {"category": "P10-C", "next_action": "tighten_trust_region_or_localize_forget_target_or_stop_projected_training"}
    if final_step != 822 or len(scales) != 9:
        return {"category": "P10-D", "next_action": "stop_invalid_or_inconclusive"}
    if all(scale == 1.0 for scale in scales):
        return {"category": "P10-A", "next_action": "design_25_step_projected_pilot"}
    return {"category": "P10-B", "next_action": "retain_transactional_line_search_and_review_before_extension"}


def _checkpoint_manifest(path: Path) -> dict[str, Any]:
    return _read_json(path / "manifest.json")


def publish_checkpoint(run_dir: Path, payload: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    step = int(metadata["step"]); final = run_dir / "checkpoints" / f"step_{step:05d}"
    if final.exists():
        raise FileExistsError(f"checkpoint step {step} already exists")
    temporary = final.parent / f".step_{step:05d}.{uuid.uuid4().hex[:10]}.stage"; temporary.mkdir(parents=True)
    torch.save(payload, temporary / "state.pt"); state_sha = sha256_file(temporary / "state.pt")
    contract = run_dir / "contract.json"
    if not contract.is_file():
        raise FileNotFoundError("checkpoint publication requires run contract")
    manifest = {"schema": CHECKPOINT_SCHEMA, **metadata, "contract_sha256": sha256_file(contract), "state_sha256": state_sha, "published_atomically": True, "test_accessed": False}
    atomic_json(temporary / "manifest.json", manifest); final.parent.mkdir(parents=True, exist_ok=True); os.replace(temporary, final)
    return {"path": str(final), "state_sha256": state_sha, "manifest_sha256": sha256_file(final / "manifest.json")}


def validate_checkpoint_chain(run_dir: Path, source_state_sha: str, expected_last: int | None = None) -> dict[str, Any]:
    checkpoint_root = run_dir / "checkpoints"
    contract = _read_json(run_dir / "contract.json")
    directories = sorted(item for item in checkpoint_root.iterdir() if item.is_dir()) if checkpoint_root.is_dir() else []
    steps = [];
    parent = source_state_sha
    for directory in directories:
        if not directory.name.startswith("step_"):
            raise ValueError("unexpected checkpoint directory")
        step = int(directory.name.split("_")[1]); expected = FIRST_STEP + len(steps)
        if step != expected or {item.name for item in directory.iterdir()} != {"state.pt", "manifest.json"}:
            raise ValueError("checkpoint chain has missing, extra, or noncontiguous step")
        manifest = _checkpoint_manifest(directory)
        if manifest.get("step") != step or manifest.get("parent_state_sha256") != parent or manifest.get("state_sha256") != sha256_file(directory / "state.pt") or manifest.get("contract_sha256") != sha256_file(run_dir / "contract.json") or manifest.get("published_atomically") is not True or manifest.get("test_accessed") is not False:
            raise ValueError("checkpoint chain SHA/parent contract mismatch")
        if (
            manifest.get("schema") != CHECKPOINT_SCHEMA
            or manifest.get("batch_hash") != batch_plan()[str(step)]["batch_hash"]
            or manifest.get("next_batch_hash") != next_batch_hash(step)
            or manifest.get("next_optimizer_step") != step + 1
            or manifest.get("accepted_scale") not in TRUST_SCALES
            or manifest.get("optimizer_counter") != step
            or manifest.get("cumulative_projected_updates") != step - 812
        ):
            raise ValueError("checkpoint step/batch/counter contract mismatch")
        payload = torch.load(directory / "state.pt", map_location="cpu", weights_only=False)
        state = payload.get("state", {})
        adapter = payload.get("adapter_state", {})
        synthetic = payload.get("synthetic") is True
        adapter_valid = bool(adapter) and (
            synthetic
            or (len(adapter) == 144 and sum(value.numel() for value in adapter.values()) == 1_769_472)
        )
        rng_valid = synthetic or (
            set(payload.get("rng", {})) == {"python", "numpy", "torch_cpu", "torch_cuda"}
            and payload.get("rng_hash") == rng_hashes(_rng_state_from_payload(payload["rng"]))
        )
        provenance = payload.get("provenance", {})
        provenance_valid = provenance.get("source_step813_state_sha256") == source_state_sha
        if not synthetic:
            provenance_valid = provenance_valid and (
                provenance.get("schema") == SCHEMA
                and provenance.get("run_name") == contract.get("run_name")
                and provenance.get("config_sha256") == contract.get("config_sha256")
                and provenance.get("implementation_canonical_sha256") == contract.get("implementation", {}).get("canonical_sha256")
                and provenance.get("runtime_attention_implementation") == "eager"
                and provenance.get("retrain_used_for_selection") is False
                and provenance.get("test_accessed") is False
            )
        if (
            payload.get("schema") != CHECKPOINT_SCHEMA
            or payload.get("test_accessed") is not False
            or state.get("step") != step
            or state.get("next_optimizer_step") != step + 1
            or state.get("next_batch_hash") != next_batch_hash(step)
            or state.get("cumulative_projected_updates") != step - 812
            or len(state.get("accepted_scales", [])) != step - 813
            or state.get("accepted_scales", [])[-1:] != [manifest["accepted_scale"]]
            or not adapter_valid
            or not rng_valid
            or not provenance_valid
            or any(float(item["step"]) != step for item in payload.get("optimizer_state", {}).get("state", {}).values())
        ):
            raise ValueError("checkpoint serialized continuation mismatch")
        parent = manifest["state_sha256"]; steps.append(step)
    if expected_last is not None and (not steps or steps[-1] != expected_last):
        raise ValueError("checkpoint chain does not reach expected last step")
    return {"steps": steps, "last_step": steps[-1] if steps else 813, "last_state_sha256": parent, "next_step": (steps[-1] + 1) if steps else 814}


def _run_state(run_dir: Path, value: dict[str, Any]) -> None:
    atomic_json(run_dir / "run_state.json", value)


def validate_resume(run_dir: Path, pre: dict[str, Any]) -> dict[str, Any]:
    if not run_dir.is_dir():
        raise FileNotFoundError("Resume run directory does not exist")
    allowed = {"contract.json", "run_state.json", "checkpoints"}
    if {item.name for item in run_dir.iterdir()} != allowed:
        raise ValueError("Resume run inventory contains missing or extra artifacts")
    if (run_dir / "COMPLETED").exists() or (run_dir / "STOPPED_SAFELY").exists():
        raise ValueError("Resume forbidden after terminal pilot status")
    state = _read_json(run_dir / "run_state.json")
    if state.get("status") != "INTERRUPTED" or state.get("test_accessed") is not False:
        raise ValueError("Resume requires an abnormal INTERRUPTED state")
    contract = _read_json(run_dir / "contract.json")
    if contract != build_contract(pre, contract.get("run_name", "")):
        raise ValueError("Resume HEAD/implementation/source contract mismatch")
    chain = validate_checkpoint_chain(run_dir, SOURCE_STATE_SHA)
    if (
        chain["next_step"] != state.get("next_step")
        or chain["next_step"] > 822
        or state.get("last_step") != chain["last_step"]
        or state.get("accepted_scales") != [
            _checkpoint_manifest(run_dir / "checkpoints" / f"step_{step:05d}")["accepted_scale"]
            for step in chain["steps"]
        ]
    ):
        raise ValueError("Resume next-step mismatch or exhausted budget")
    return {"state": state, "chain": chain, "contract": contract}


def synthetic_dry_run(root: Path, config_path: Path, run_name: str) -> dict[str, Any]:
    config = load_pilot_config(config_path, root); final = _resolve(root, Path(config["output_root"]) / "synthetic_runs" / _safe_name(run_name), output=True)
    if final.exists(): raise FileExistsError("refusing to overwrite SyntheticDryRun")
    final.mkdir(parents=True); atomic_json(final / "contract.json", {"schema": SCHEMA, "run_name": run_name, "test_accessed": False})
    parameter = torch.tensor([0.0]); optimizer_counter = 813; parent = SOURCE_STATE_SHA; accepted_scales = []
    for step in range(814, 823):
        before = parameter.clone(); proposal_counter = optimizer_counter + 1
        desired = 0.5 if step == 815 else 1.0
        selection = select_first_passing_scale(TRUST_SCALES, lambda scale, desired=desired: (scale == desired, {"rolled_back": scale != desired, "fixed_baseline": True}))
        if not selection["accepted"]: raise RuntimeError("toy acceptance unexpectedly failed")
        scale = selection["accepted_scale"]; parameter = before + scale; optimizer_counter = proposal_counter; accepted_scales.append(scale)
        payload = {"schema": CHECKPOINT_SCHEMA, "synthetic": True, "adapter_state": {"toy": parameter.clone()}, "optimizer_state": {"state": {0: {"step": torch.tensor(float(step))}}, "param_groups": [{"params": [0]}]}, "state": {"step": step, "next_optimizer_step": step + 1, "next_batch_hash": next_batch_hash(step), "accepted_scales": [*accepted_scales], "cumulative_projected_updates": step - 812}, "provenance":{"source_step813_state_sha256":SOURCE_STATE_SHA,"test_accessed":False}, "test_accessed": False}
        metadata = {"parent_state_sha256": parent, "step": step, "next_optimizer_step": step + 1, "batch_hash": batch_plan()[str(step)]["batch_hash"], "next_batch_hash": next_batch_hash(step), "accepted_scale": scale, "optimizer_counter": step, "cumulative_projected_updates": step - 812, "cumulative_utility_damage": {"overall_auc_damage": 0.0001 * (step - 813)}}
        published = publish_checkpoint(final, payload, metadata); parent = published["state_sha256"]
        if step == 817:
            _run_state(final, {"status": "INTERRUPTED", "next_step": 818, "test_accessed": False})
            chain = validate_checkpoint_chain(final, SOURCE_STATE_SHA, 817)
            if chain["next_step"] != 818: raise RuntimeError("toy resume continuity failed")
    _publish_terminal(final, "COMPLETED", {"status": "COMPLETED", "last_step": 822, "next_step": 823, "accepted_scales": accepted_scales, "step823_executed": False, "test_accessed": False})
    chain = validate_checkpoint_chain(final, SOURCE_STATE_SHA, 822)
    stopped = select_first_passing_scale(TRUST_SCALES, lambda scale: (False, {"scale": scale, "restored": True}))
    result = {"schema": SCHEMA, "mode": "SyntheticDryRun", "steps": chain["steps"], "accepted_scales": accepted_scales, "scale_order_verified": True, "resume_verified_at_step": 818, "stopped_safely_verified": stopped["accepted"] is False, "optimizer_proposals": 9, "optimizer_counter_final": optimizer_counter, "step823_executed": False, "gradient_vectors_persisted": False, "delta_vectors_persisted": False, "logits_persisted": False, "tokens_persisted": False, "raw_samples_persisted": False, "model_loaded": False, "test_accessed": False}
    atomic_json(final / "synthetic_result.json", result); return {**result, "run_dir": str(final)}


def _utility_summary(runtime: dict[str, Any], validation: JsonPromptDataset, retain_indices: list[int], device: torch.device) -> dict[str, dict[str, Any]]:
    raw = evaluate_overall_validation(runtime["current"], validation, device, 16); all_indices = list(range(len(raw["gold"])))
    result = {"overall_validation": _binary_summary(raw["probabilities"], raw["gold"], all_indices), "retain_user_validation": _binary_summary(raw["probabilities"], raw["gold"], retain_indices)}; del raw; return result


def _forget_metrics(runtime: dict[str, Any], validation: JsonPromptDataset, indices: list[int], device: torch.device) -> dict[str, Any]:
    probabilities=[]; original_probabilities=[]; gold=[]; forced_loss_sum=0.0; forced_samples=0; jsd=[]
    current_mode = runtime["current"].training
    runtime["current"].eval()
    try:
        with torch.no_grad():
            for start in range(0, len(indices), 16):
                selected=indices[start:start+16]; batch=move_batch(_batch(validation,selected),device)
                student=runtime["current"](input_ids=batch["input_ids"],labels=batch["target_ids"]).logits
                original=runtime["original"](input_ids=batch["input_ids"],labels=batch["target_ids"]).logits
                augmented=runtime["augmented"](input_ids=batch["input_ids"],labels=batch["target_ids"]).logits
                forced=forced_logits(original,augmented,2.0); batch_count=len(selected)
                forced_loss_sum += float(teacher_cross_entropy(forced,student)) * batch_count; forced_samples += batch_count
                sp=torch.softmax(student[:,0,[465,2163]],-1); op=torch.softmax(original[:,0,[465,2163]],-1); midpoint=(sp+op)/2
                probabilities.extend(sp[:,1].cpu().tolist()); original_probabilities.extend(op[:,1].cpu().tolist()); gold.extend((batch["target_ids"][:,0]==2163).long().cpu().tolist())
                jsd.extend((0.5*((sp*(sp.clamp_min(1e-12).log()-midpoint.clamp_min(1e-12).log())).sum(-1)+(op*(op.clamp_min(1e-12).log()-midpoint.clamp_min(1e-12).log())).sum(-1))).cpu().tolist())
    finally:
        runtime["current"].train(current_mode)
    p=np.asarray(probabilities); o=np.asarray(original_probabilities); y=np.asarray(gold)
    forced_ce=forced_loss_sum/forced_samples
    values={"samples":len(y),"accuracy":float(accuracy_score(y,p>=.5)),"auc":float(roc_auc_score(y,p)),"log_loss":float(log_loss(y,p,labels=[0,1])),"confidence_mean":float(np.maximum(p,1-p).mean()),"probability_mean":float(p.mean()),"probability_std":float(p.std()),"positive_rate":float((p>=.5).mean()),"forced_teacher_ce":forced_ce,"student_original_jsd":float(np.mean(jsd)),"original_prediction_agreement":float(np.mean((p>=.5)==(o>=.5))),"student_forced_target_distance":forced_ce,"finite":True,"probability_collapse":bool(p.std()<=COLLAPSE_STD_EPSILON or (p>=.5).mean() in (0,1)),"test_accessed":False}
    if any(not math.isfinite(float(v)) for v in values.values() if isinstance(v,(int,float))): values["finite"]=False
    return values


def build_contract(pre: dict[str, Any], run_name: str) -> dict[str, Any]:
    return json_native({"schema":SCHEMA,"run_name":run_name,"config_sha256":pre["config_sha256"],"git":pre["git"],"implementation":pre["implementation"],"stage_b_source":pre["stage_b_source"],"fixed_utility_baseline":pre["fixed_utility_baseline"],"batch_plan":pre["batch_plan"],"trust_scales":pre["trust_scales"],"runtime_protocol":pre.get("runtime_protocol"),"test_accessed":False})


def _load_pilot_runtime(root: Path, config: dict[str, Any], checkpoint: Path, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime=_load_runtime(root,root/"configs/t5_step812_gradient_geometry_audit_v1.yaml",device); payload=torch.load(checkpoint/"state.pt",map_location="cpu",weights_only=False)
    result=set_peft_model_state_dict(runtime["current"],payload["adapter_state"])
    if getattr(result,"unexpected_keys",[]): raise ValueError("pilot adapter unexpected keys")
    reloaded=get_peft_model_state_dict(runtime["current"])
    if reloaded.keys()!=payload["adapter_state"].keys() or any(not torch.equal(reloaded[k].cpu(),payload["adapter_state"][k]) for k in reloaded): raise ValueError("pilot adapter strict reload mismatch")
    runtime["payload"]=payload
    for name in ("current","original","augmented"):
        model_config=runtime[name].config
        if getattr(model_config,"_attn_implementation_internal",None)!="eager": raise ValueError(f"{name} attention implementation is not eager")
    runtime["current"].to(device); return runtime,payload


def _publish_terminal(run_dir: Path, status: str, value: dict[str, Any]) -> None:
    if status not in {"COMPLETED", "STOPPED_SAFELY"}:
        raise ValueError("invalid terminal status")
    atomic_json(run_dir / "run_state.json", value)
    chain = validate_checkpoint_chain(run_dir, SOURCE_STATE_SHA)
    manifest = {
        "schema": SCHEMA,
        "status": status,
        "run_state_sha256": sha256_file(run_dir / "run_state.json"),
        "contract_sha256": sha256_file(run_dir / "contract.json"),
        "checkpoint_steps": chain["steps"],
        "last_state_sha256": chain["last_state_sha256"],
        "published_atomically": True,
        "test_accessed": False,
    }
    atomic_json(run_dir / "run_manifest.json", manifest)
    atomic_text(run_dir / status, f"PILOT_10STEP_{status}\n")


def _execute(root: Path, config_path: Path, run_name: str, *, resume: bool) -> dict[str, Any]:
    pre=preflight(root,config_path); require_clean_git(pre["git"],"pilot Resume" if resume else "pilot Full"); config=load_pilot_config(config_path,root); run_dir=_resolve(root,Path(config["output_root"])/"full_runs"/_safe_name(run_name),output=True)
    if resume:
        resume_info=validate_resume(run_dir,pre); start=resume_info["chain"]["next_step"]; parent_state=resume_info["chain"]["last_state_sha256"]; checkpoint=run_dir/"checkpoints"/f"step_{start-1:05d}"
    else:
        if run_dir.exists(): raise FileExistsError("refusing to overwrite pilot Full")
        run_dir.mkdir(parents=True); atomic_json(run_dir/"contract.json",build_contract(pre,run_name)); _run_state(run_dir,{"status":"RUNNING","next_step":814,"test_accessed":False}); start=814; parent_state=SOURCE_STATE_SHA; checkpoint=_resolve(root,config["source_checkpoint"])
    outer=capture_rng(); runtime=None
    try:
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); runtime,payload=_load_pilot_runtime(root,config,checkpoint,device); names,parameters=_runtime_binding(runtime); optimizer_state=payload["optimizer_state"]; continuation=_rng_state_from_payload(payload["rng"]); restore_rng(continuation)
        base=load_config(_resolve(root,config["base_config"]),root); validation=JsonPromptDataset(Path(base["paths"]["validation"]),runtime["tokenizer"]); _,indices,_=_data_lineage(root,base,_resolve(root,config["protocol_root"])); previous_utility=_read_json(_resolve(root,config["stage_b_full"])/"step_result.json")["transaction"]["post_evidence"]["utility_after"] if start==814 else payload["accepted_evidence"]["utility"]
        accepted_scales=[] if start==814 else list(payload["state"]["accepted_scales"]); frozen_hash=_frozen_hash(runtime["current"]); teacher_hash={n:_tensor_state_hash(runtime[n]) for n in ("original","augmented")}
        for step in range(start,LAST_STEP+1):
            print(f"[pilot:step:start] step={step}",flush=True); pair=catalog_pair(step-813,len(runtime["forget"]),len(runtime["retain"]),16,42); restore_rng(continuation); runtime["current"].train(); forget=move_batch(_batch(runtime["forget"],pair["forget_indices"]),device); retain=move_batch(_batch(runtime["retain"],pair["retain_indices"]),device); components=compute_components(runtime["current"],runtime["original"],runtime["augmented"],forget,retain,2.0); gradients=_flatten_gradients(components,parameters); next_continuation=capture_rng(); f,k,s=(gradients[n] for n in LOSSES); projection=direct_svd_projection(f,k,s)
            if projection["rank"] <= 0 or projection["eta_F"] is None or projection["eta_F"]<config["projection"]["eta_f_min"]: _publish_terminal(run_dir,"STOPPED_SAFELY",{"status":"STOPPED_SAFELY","last_step":step-1,"failed_step":step,"reason":"gradient_projection_gate","projection":{"rank":projection["rank"],"eta_F":projection["eta_F"]},"accepted_scales":accepted_scales,"test_accessed":False}); return {"status":"STOPPED_SAFELY","last_step":step-1}
            proposal=shadow_adamw_proposal_for_step([p.detach().cpu() for p in parameters],optimizer_state,projection["safe"],step-1); projected=update_space_projection(proposal["delta"],projection["basis"]); replay_rng=next_continuation
            pre_losses=isolated_rng_evaluation(lambda:{n:float(v.detach().cpu()) for n,v in compute_components(runtime["current"],runtime["original"],runtime["augmented"],forget,retain,2.0).items() if n in LOSSES},replay_rng,next_continuation)["value"]; pre_forget=isolated_rng_evaluation(lambda:_forget_metrics(runtime,validation,indices["forget_user_validation"],device),next_continuation,next_continuation)["value"]
            pre_step_parameters=[p.detach().cpu().clone() for p in parameters]
            def restore_pre_step()->None:
                with torch.no_grad():
                    for parameter,value in zip(parameters,pre_step_parameters): parameter.copy_(value.to(parameter.device))
                restore_rng(next_continuation)
            def trial(scale:float)->tuple[bool,dict[str,Any]]:
                after,actual=materialize_dtype_delta([p.detach().cpu() for p in parameters],projected*scale); directional=directional_gate(f,k,s,proposal["delta"],actual,projection["eta_F"],config)
                if not directional["passed"]: return False,{"directional":directional,"rolled_back":True}
                with torch.no_grad():
                    for p,v in zip(parameters,after): p.copy_(v.to(p.device))
                post_replay=isolated_rng_evaluation(lambda:{n:float(v.detach().cpu()) for n,v in compute_components(runtime["current"],runtime["original"],runtime["augmented"],forget,retain,2.0).items() if n in LOSSES},replay_rng,next_continuation)["value"]
                utility=isolated_rng_evaluation(lambda:_utility_summary(runtime,validation,indices["retain_user_validation"],device),next_continuation,next_continuation)["value"]; utility_gate=cumulative_utility_gate(utility,previous_utility)
                forget_metrics=isolated_rng_evaluation(lambda:_forget_metrics(runtime,validation,indices["forget_user_validation"],device),next_continuation,next_continuation)["value"]
                loss_delta={n:post_replay[n]-pre_losses[n] for n in LOSSES}; prediction={n:directional["directional"][n]["dot"] for n in LOSSES}
                replay_diagnostics={n:{"actual":loss_delta[n],"predicted":prediction[n],"actual_to_predicted_ratio":None if prediction[n]==0 else loss_delta[n]/prediction[n],"prediction_error":loss_delta[n]-prediction[n]} for n in LOSSES}
                forget_gate=forget_evidence_gate(pre_forget,forget_metrics)
                passed=(loss_delta["L_forget"]<0 and utility_gate["utility_pass"] and forget_gate["passed"] and _frozen_hash(runtime["current"])==frozen_hash and all(_tensor_state_hash(runtime[n])==teacher_hash[n] for n in teacher_hash) and rng_hashes(capture_rng())==rng_hashes(next_continuation))
                projection_evidence={key:projection[key] for key in ("rank","singular_values","condition_number","rho","eta_F","normalized_residuals","retain_dots_after","algorithm")}
                evidence={"projection":projection_evidence,"shadow_optimizer":{"source_step":proposal["source_optimizer_step"],"next_step":proposal["next_optimizer_step"],"step_calls":proposal["shadow_optimizer_steps_executed"],"delta_hash":proposal["delta_hash"],"delta_norm":proposal["delta_norm"]},"directional":directional,"paired_replay":{"pre":pre_losses,"post":post_replay,"actual_delta":loss_delta,"first_order_prediction":prediction,"diagnostics":replay_diagnostics,"same_dropout_rng":True},"utility":utility,"cumulative_utility_gate":utility_gate,"forget_metrics_before":pre_forget,"forget_metrics":forget_metrics,"forget_gate":forget_gate,"forget_evidence_conflict":forget_gate["forget_evidence_conflict"],"continuation_rng_preserved":True,"retrain_used_for_selection":False}
                return passed,evidence
            selection=transactional_scale_search(TRUST_SCALES,restore_pre_step,trial)
            if not selection["accepted"]: _publish_terminal(run_dir,"STOPPED_SAFELY",{"status":"STOPPED_SAFELY","last_step":step-1,"next_step":step,"failed_step":step,"scale_trials":selection["trials"],"accepted_scales":accepted_scales,"test_accessed":False}); return {"status":"STOPPED_SAFELY","last_step":step-1}
            accepted=selection["trials"][-1]["evidence"]; scale=selection["accepted_scale"]; accepted_scales.append(scale); optimizer_state=proposal["optimizer_state"]; continuation=next_continuation; previous_utility=accepted["utility"]
            cp_payload={"schema":CHECKPOINT_SCHEMA,"adapter_state":{k:v.detach().cpu() for k,v in get_peft_model_state_dict(runtime["current"]).items()},"optimizer_state":optimizer_state,"state":{"step":step,"next_optimizer_step":step+1,"next_batch_hash":next_batch_hash(step),"accepted_scales":accepted_scales,"cumulative_projected_updates":step-812},"rng":{"python":continuation.python,"numpy":continuation.numpy,"torch_cpu":continuation.torch_cpu,"torch_cuda":continuation.torch_cuda},"rng_hash":rng_hashes(continuation),"accepted_evidence":accepted,"scale_trials":selection["trials"],"provenance":{"schema":SCHEMA,"run_name":run_name,"config_sha256":pre["config_sha256"],"implementation_canonical_sha256":pre["implementation"]["canonical_sha256"],"source_step813_state_sha256":SOURCE_STATE_SHA,"runtime_attention_implementation":"eager","retrain_used_for_selection":False,"test_accessed":False},"test_accessed":False}
            meta={"parent_state_sha256":parent_state,"step":step,"next_optimizer_step":step+1,"batch_hash":pair["batch_hash"],"next_batch_hash":next_batch_hash(step),"accepted_scale":scale,"optimizer_counter":step,"cumulative_projected_updates":step-812,"cumulative_utility_damage":accepted["cumulative_utility_gate"]}
            published=publish_checkpoint(run_dir,cp_payload,meta); parent_state=published["state_sha256"]; _run_state(run_dir,{"status":"RUNNING","last_step":step,"next_step":step+1,"accepted_scales":accepted_scales,"test_accessed":False}); print(f"[pilot:commit] step={step} scale={scale}",flush=True)
        classification=classify_pilot("COMPLETED",822,accepted_scales,False); _publish_terminal(run_dir,"COMPLETED",{"status":"COMPLETED","last_step":822,"next_step":823,"accepted_scales":accepted_scales,"classification":classification,"step823_executed":False,"test_accessed":False}); return {"status":"COMPLETED","classification":classification,"last_step":822}
    except BaseException:
        if run_dir.exists() and not (run_dir/"COMPLETED").exists() and not (run_dir/"STOPPED_SAFELY").exists():
            state=_read_json(run_dir/"run_state.json"); state["status"]="INTERRUPTED"; state["test_accessed"]=False; _run_state(run_dir,state)
        raise
    finally:
        restore_rng(outer)
        if runtime is not None: del runtime
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def verify_run(root: Path, config_path: Path, run_name: str) -> dict[str,Any]:
    pre=preflight(root,config_path); require_clean_git(pre["git"],"pilot Analyze"); config=load_pilot_config(config_path,root); run_dir=_resolve(root,Path(config["output_root"])/"full_runs"/_safe_name(run_name),output=True); state=_read_json(run_dir/"run_state.json"); status=state.get("status")
    if status not in {"COMPLETED","STOPPED_SAFELY"} or not (run_dir/status).is_file(): raise ValueError("pilot run is not terminal")
    expected_inventory={"contract.json","run_state.json","run_manifest.json","checkpoints",status}
    if {item.name for item in run_dir.iterdir()}!=expected_inventory: raise ValueError("terminal pilot inventory mismatch")
    contract=_read_json(run_dir/"contract.json")
    if contract!=build_contract(pre,run_name): raise ValueError("pilot contract mismatch")
    chain=validate_checkpoint_chain(run_dir,SOURCE_STATE_SHA,822 if status=="COMPLETED" else None)
    manifest=_read_json(run_dir/"run_manifest.json")
    if manifest!={"schema":SCHEMA,"status":status,"run_state_sha256":sha256_file(run_dir/"run_state.json"),"contract_sha256":sha256_file(run_dir/"contract.json"),"checkpoint_steps":chain["steps"],"last_state_sha256":chain["last_state_sha256"],"published_atomically":True,"test_accessed":False}: raise ValueError("terminal run manifest mismatch")
    if (run_dir/status).read_text(encoding="utf-8")!=f"PILOT_10STEP_{status}\n": raise ValueError("terminal marker mismatch")
    if status=="COMPLETED" and (state.get("next_step")!=823 or state.get("step823_executed") is not False): raise ValueError("completed pilot budget contract mismatch")
    evidence_conflict=any(torch.load(run_dir/"checkpoints"/f"step_{step:05d}"/"state.pt",map_location="cpu",weights_only=False).get("accepted_evidence",{}).get("forget_evidence_conflict") is True for step in chain["steps"])
    classification=classify_pilot(status,chain["last_step"],state.get("accepted_scales",[]),evidence_conflict); return {"run_dir":run_dir,"state":state,"chain":chain,"classification":classification,"preflight":pre}


def analyze(root:Path,config_path:Path,run_name:str)->dict[str,Any]:
    verified=verify_run(root,config_path,run_name); config=load_pilot_config(config_path,root); final=_resolve(root,Path(config["output_root"])/"analysis_runs"/_safe_name(run_name),output=True)
    if final.exists(): raise FileExistsError("refusing to overwrite pilot Analyze")
    stage=final.parent/f".{run_name}.{uuid.uuid4().hex[:10]}.stage";stage.mkdir(parents=True);result={"schema":ANALYSIS_SCHEMA,"run_name":run_name,**verified["classification"],"final_step":verified["chain"]["last_step"],"accepted_scales":verified["state"].get("accepted_scales",[]),"test_accessed":False};atomic_json(stage/"analysis.json",result);atomic_json(stage/"manifest.json",{"schema":ANALYSIS_SCHEMA,"analysis_sha256":sha256_file(stage/"analysis.json"),"published_atomically":True,"test_accessed":False});atomic_text(stage/"COMPLETED","PILOT_10STEP_ANALYSIS_COMPLETED\n");final.parent.mkdir(parents=True,exist_ok=True);os.replace(stage,final);return result


def main()->None:
    parser=argparse.ArgumentParser(description="Transactional T5 10-step projected pilot");parser.add_argument("--mode",choices=("Preflight","SyntheticDryRun","Full","Resume","Analyze"),default="Preflight");parser.add_argument("--config",type=Path,required=True);parser.add_argument("--project-root",type=Path,default=Path.cwd());parser.add_argument("--run-name");args=parser.parse_args();root=args.project_root.resolve();config=args.config.resolve()
    if args.mode=="Preflight": result=preflight(root,config)
    else:
        if not args.run_name: parser.error(f"{args.mode} requires --run-name")
        if args.mode=="SyntheticDryRun": result=synthetic_dry_run(root,config,args.run_name)
        elif args.mode=="Full": result=_execute(root,config,args.run_name,resume=False)
        elif args.mode=="Resume": result=_execute(root,config,args.run_name,resume=True)
        else: result=analyze(root,config,args.run_name)
    print(json.dumps(json_native(result),indent=2,ensure_ascii=False,allow_nan=False))


if __name__=="__main__": main()
