from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from src.diagnostics.git_provenance import (
    git_provenance,
    implementation_provenance,
    require_clean_git,
)
from src.diagnostics.t5_reconstructed_official import sha256_file
from src.diagnostics.t5_zero_training_analysis import ANALYSIS_SCHEMA, _load_cache_unit
from src.diagnostics.t5_zero_training_analyze_runner import (
    FINAL_MANIFEST_SCHEMA,
    _archive_lock,
    _process_record,
    _safe_run_name,
    full_snapshot,
)
from src.diagnostics.t5_zero_training_audit import (
    OUTPUT_NAME,
    SPLITS,
    atomic_json,
    atomic_text,
)
from src.diagnostics.t5_zero_training_decision import (
    PRIMARY_MIA_ATTACK,
    detect_global_dememorization,
    mia_evidence_conflict,
    specificity_bootstrap,
)


DECISION_V2_SCHEMA = "t5-e2urec-zero-training-pareto-decision-v2"
DECISION_V2_RUN_SCHEMA = "t5-e2urec-zero-training-decision-v2-run-v1"
DECISION_V2_LOCK_SCHEMA = "t5-e2urec-zero-training-decision-v2-lock-v1"
CORE_PARETO_AXES = {
    "forget_train_l2_to_retrain": "min",
    "forget_mia_auc_gap_to_retrain": "min",
    "I_specific": "max",
}
EXPECTED_BRANCH_STEPS = {
    branch: (812, 813, 850, 900, 1000, 1200)
    for branch in ("j0", "j2", "j4", "j5")
}
FROZEN_UTILITY_THRESHOLDS = {
    "overall_auc_damage_max": 0.005,
    "retain_user_auc_damage_max": 0.005,
    "overall_log_loss_damage_max": 0.01,
    "retain_user_log_loss_damage_max": 0.01,
}
COLLAPSE_STD_EPSILON = 1e-12
IMPLEMENTATION_FILES = (
    "src/diagnostics/t5_zero_training_decision_v2.py",
    "src/diagnostics/git_provenance.py",
    "configs/t5_e2urec_zero_training_audit_v1.yaml",
    "scripts/diagnostics/t5_zero_training_decision_v2.ps1",
    "docs/t5_zero_training_decision_v2.md",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _primary_auc(analysis: dict[str, Any], model: str, group: str) -> float:
    return float(
        analysis["mia"][model][group]["primary_matched_user"]["attacks"][
            PRIMARY_MIA_ATTACK
        ]["pooled_matched_user"]["roc_auc"]
    )


def _parse_candidate(model: str) -> tuple[str, int] | None:
    match = re.fullmatch(r"(j[0245])_step(\d+)", model)
    return (match.group(1), int(match.group(2))) if match else None


def _finite(values: list[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def utility_evidence(
    candidate_metrics: dict[str, dict[str, Any]],
    original_metrics: dict[str, dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    overall = candidate_metrics["overall_validation"]
    retain = candidate_metrics["retain_user_validation"]
    original_overall = original_metrics["overall_validation"]
    original_retain = original_metrics["retain_user_validation"]
    damages = {
        "overall_auc_damage": float(original_overall["auc"] - overall["auc"]),
        "retain_user_auc_damage": float(
            original_retain["auc"] - retain["auc"]
        ),
        "overall_log_loss_damage": float(
            overall["log_loss"] - original_overall["log_loss"]
        ),
        "retain_user_log_loss_damage": float(
            retain["log_loss"] - original_retain["log_loss"]
        ),
    }
    safety_values = list(damages.values())
    for row in (overall, retain):
        safety_values.extend(
            float(row[key])
            for key in (
                "auc",
                "accuracy",
                "log_loss",
                "confidence_mean",
                "positive_rate",
                "probability_mean",
                "probability_std",
            )
        )
    finite = _finite(safety_values)
    collapse = any(
        float(row["probability_std"]) <= COLLAPSE_STD_EPSILON
        or float(row["positive_rate"]) <= 0.0
        or float(row["positive_rate"]) >= 1.0
        for row in (overall, retain)
    )
    checks = {
        "overall_auc": damages["overall_auc_damage"]
        <= thresholds["overall_auc_damage_max"],
        "retain_user_auc": damages["retain_user_auc_damage"]
        <= thresholds["retain_user_auc_damage_max"],
        "overall_log_loss": damages["overall_log_loss_damage"]
        <= thresholds["overall_log_loss_damage_max"],
        "retain_user_log_loss": damages["retain_user_log_loss_damage"]
        <= thresholds["retain_user_log_loss_damage_max"],
        "finite": finite,
        "probability_not_collapsed": not collapse,
    }
    return {
        **damages,
        "utility_pass": all(checks.values()),
        "utility_checks": checks,
        "probability_collapse": collapse,
        "all_utility_values_finite": finite,
        "utility_fail_reasons": [name for name, passed in checks.items() if not passed],
    }


def candidate_evidence(
    model: str,
    metrics: dict[tuple[str, str], dict[str, Any]],
    analysis: dict[str, Any],
    specificity: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    parsed = _parse_candidate(model)
    if parsed is None:
        raise ValueError(f"invalid decision-v2 candidate name: {model}")
    branch, step = parsed
    candidate_metrics = {split: metrics[model, split] for split in SPLITS}
    original_metrics = {split: metrics["original", split] for split in SPLITS}
    utility = utility_evidence(candidate_metrics, original_metrics, thresholds)
    forget_train_distance = float(
        metrics[model, "forget_train"]["relative_retrain"]["l2_rms"]
    )
    forget_validation_distance = float(
        metrics[model, "forget_user_validation"]["relative_retrain"]["l2_rms"]
    )
    original_train_distance = float(
        metrics["original", "forget_train"]["relative_retrain"]["l2_rms"]
    )
    original_validation_distance = float(
        metrics["original", "forget_user_validation"]["relative_retrain"][
            "l2_rms"
        ]
    )
    candidate_mia = _primary_auc(analysis, model, "forget")
    original_mia = _primary_auc(analysis, "original", "forget")
    retrain_mia = _primary_auc(analysis, "retrain", "forget")
    candidate_mia_gap = abs(candidate_mia - retrain_mia)
    original_mia_gap = abs(original_mia - retrain_mia)
    specific = specificity["bootstrap"]["I_specific"]
    specific_ci = [float(value) for value in specific["percentile_95_ci"]]
    i_specific = float(specificity["point_estimates"]["I_specific"])
    decision_values_finite = _finite(
        [
            forget_train_distance,
            forget_validation_distance,
            original_train_distance,
            original_validation_distance,
            candidate_mia,
            original_mia,
            retrain_mia,
            candidate_mia_gap,
            original_mia_gap,
            i_specific,
            *specific_ci,
        ]
    )
    utility["utility_checks"]["decision_values_finite"] = decision_values_finite
    utility["all_decision_values_finite"] = decision_values_finite
    utility["utility_pass"] = bool(
        utility["utility_pass"] and decision_values_finite
    )
    utility["utility_fail_reasons"] = [
        name for name, passed in utility["utility_checks"].items() if not passed
    ]
    forget_train_closer = forget_train_distance < original_train_distance
    forget_validation_closer = (
        forget_validation_distance < original_validation_distance
    )
    forget_mia_closer = candidate_mia_gap < original_mia_gap
    selectivity_pass = specific_ci[0] > 0.0
    global_shift = detect_global_dememorization(
        original_mia,
        candidate_mia,
        _primary_auc(analysis, "original", "retain"),
        _primary_auc(analysis, model, "retain"),
        float(metrics["original", "overall_validation"]["confidence_mean"]),
        float(metrics[model, "overall_validation"]["confidence_mean"]),
    )
    conflict = mia_evidence_conflict(analysis, model, "forget") or mia_evidence_conflict(
        analysis, model, "retain"
    )
    retrain_directed = forget_train_closer and (
        forget_validation_closer or forget_mia_closer
    )
    return {
        "model": model,
        "branch": branch,
        "step": step,
        **utility,
        "forget_train_l2_to_retrain": forget_train_distance,
        "forget_validation_l2_to_retrain": forget_validation_distance,
        "forget_mia_auc_gap_to_retrain": candidate_mia_gap,
        "original_forget_train_l2_to_retrain": original_train_distance,
        "original_forget_validation_l2_to_retrain": original_validation_distance,
        "original_forget_mia_auc_gap_to_retrain": original_mia_gap,
        "I_specific": i_specific,
        "I_specific_percentile_95_ci": specific_ci,
        "utility_pass": bool(utility["utility_pass"]),
        "forget_train_closer_to_retrain": forget_train_closer,
        "forget_validation_closer_to_retrain": forget_validation_closer,
        "forget_mia_closer_to_retrain": forget_mia_closer,
        "selectivity_ci_pass": selectivity_pass,
        "global_dememorization": global_shift,
        "mia_evidence_conflicted": conflict,
        "retrain_directed": retrain_directed,
        "qualified_selective_deletion": bool(
            forget_train_closer
            and forget_validation_closer
            and forget_mia_closer
            and selectivity_pass
            and not global_shift
            and not conflict
        ),
    }


def _dominates_v2(left: dict[str, Any], right: dict[str, Any]) -> bool:
    no_worse = True
    strictly_better = False
    for axis, direction in CORE_PARETO_AXES.items():
        left_value, right_value = float(left[axis]), float(right[axis])
        if direction == "min":
            no_worse &= left_value <= right_value
            strictly_better |= left_value < right_value
        else:
            no_worse &= left_value >= right_value
            strictly_better |= left_value > right_value
    return bool(no_worse and strictly_better)


def utility_first_pareto(
    candidates: list[dict[str, Any]],
) -> tuple[list[str], dict[str, list[str]]]:
    eligible = [row for row in candidates if row["utility_pass"]]
    front, dominated_by = [], {}
    for row in candidates:
        if not row["utility_pass"]:
            dominated_by[row["model"]] = []
            continue
        dominators = [
            other["model"]
            for other in eligible
            if other is not row and _dominates_v2(other, row)
        ]
        dominated_by[row["model"]] = dominators
        if not dominators:
            front.append(row["model"])
    return front, dominated_by


def branch_trajectory_evidence(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    by_branch: dict[str, dict[int, dict[str, Any]]] = {
        branch: {} for branch in EXPECTED_BRANCH_STEPS
    }
    for row in candidates:
        by_branch[row["branch"]][int(row["step"])] = row
    output = {}
    for branch, expected in EXPECTED_BRANCH_STEPS.items():
        rows = by_branch[branch]
        available = sorted(rows)
        missing = [step for step in expected if step not in rows]
        if len(available) < 2:
            output[branch] = {
                "status": "unavailable",
                "available_steps": available,
                "unavailable_steps": missing,
                "early_step": available[0] if available else None,
                "late_step": None,
                "joint_deletion_regression": None,
                "joint_utility_recovery": None,
                "b_pattern": None,
                "reason": "fewer_than_two_same_branch_checkpoints",
            }
            continue
        early, late = rows[available[0]], rows[available[-1]]
        regression = bool(
            late["forget_train_l2_to_retrain"]
            > early["forget_train_l2_to_retrain"]
            and late["forget_validation_l2_to_retrain"]
            > early["forget_validation_l2_to_retrain"]
            and late["forget_mia_auc_gap_to_retrain"]
            > early["forget_mia_auc_gap_to_retrain"]
        )
        recovery = bool(not early["utility_pass"] and late["utility_pass"])
        output[branch] = {
            "status": "available",
            "available_steps": available,
            "unavailable_steps": missing,
            "early_step": early["step"],
            "late_step": late["step"],
            "early_model": early["model"],
            "late_model": late["model"],
            "early_qualified_selective_deletion": early[
                "qualified_selective_deletion"
            ],
            "joint_deletion_regression": regression,
            "joint_utility_recovery": recovery,
            "b_pattern": bool(
                early["qualified_selective_deletion"] and regression and recovery
            ),
        }
    return output


def step812_evidence(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [row for row in candidates if row["model"] == "j0_step812"]
    if not matches:
        return {
            "status": "unavailable",
            "model": "j0_step812",
            "step812_deletion_direction": None,
            "step812_not_closer_on_forget_train": None,
            "step812_not_closer_on_forget_validation": None,
            "step812_not_closer_on_forget_mia": None,
            "reason": "checkpoint_or_metrics_unavailable; never interpolated",
        }
    row = matches[0]
    return {
        "status": "available",
        "model": row["model"],
        "step812_deletion_direction": bool(
            row["forget_train_closer_to_retrain"]
            and row["forget_validation_closer_to_retrain"]
            and row["forget_mia_closer_to_retrain"]
        ),
        "step812_not_closer_on_forget_train": not row[
            "forget_train_closer_to_retrain"
        ],
        "step812_not_closer_on_forget_validation": not row[
            "forget_validation_closer_to_retrain"
        ],
        "step812_not_closer_on_forget_mia": not row[
            "forget_mia_closer_to_retrain"
        ],
        "selectivity_ci_pass": row["selectivity_ci_pass"],
        "global_dememorization": row["global_dememorization"],
        "mia_evidence_conflicted": row["mia_evidence_conflicted"],
    }


def classify_decision_v2(
    candidates: list[dict[str, Any]],
    branches: dict[str, Any],
    step812: dict[str, Any],
    *,
    retrain_reference_or_evaluation_insufficient: bool = False,
) -> dict[str, Any]:
    a_candidates = [
        row["model"]
        for row in candidates
        if row["utility_pass"]
        and row["forget_train_closer_to_retrain"]
        and row["forget_validation_closer_to_retrain"]
        and row["forget_mia_closer_to_retrain"]
        and row["selectivity_ci_pass"]
        and not row["global_dememorization"]
        and not row["mia_evidence_conflicted"]
    ]
    b_branches = [
        branch
        for branch, value in branches.items()
        if value.get("b_pattern") is True
    ]
    b0_candidates = [
        row["model"]
        for row in candidates
        if row["retrain_directed"]
        and (not row["selectivity_ci_pass"] or row["global_dememorization"])
        and not row["mia_evidence_conflicted"]
    ]
    c_reached = bool(
        step812.get("status") == "available"
        and step812["step812_not_closer_on_forget_train"]
        and step812["step812_not_closer_on_forget_validation"]
        and step812["step812_not_closer_on_forget_mia"]
        and not step812.get("mia_evidence_conflicted", False)
    )
    conflicted = [
        row["model"] for row in candidates if row["mia_evidence_conflicted"]
    ]
    if retrain_reference_or_evaluation_insufficient:
        decision = "D. Retrain reference or evaluation insufficient"
        reason = "retrain_reference_or_evaluation_insufficient"
    elif a_candidates:
        decision = "A. Existing method sufficient"
        reason = "candidate_passes_utility_deletion_mia_selectivity_and_safety"
    elif b_branches:
        decision = "B. Joint objective overwrites deletion"
        reason = "same_branch_selective_early_deletion_regresses_as_utility_recovers"
    elif b0_candidates:
        decision = "B0. Retrain-directed but non-selective shift"
        reason = "retrain_directed_shift_lacks_positive_selectivity_ci_or_is_global"
    elif c_reached:
        decision = "C. Teacher target insufficient"
        reason = "step812_not_closer_on_forget_train_validation_or_mia"
    elif candidates and len(conflicted) == len(candidates):
        decision = "D. MIA evidence conflicted"
        reason = "all_candidate_mia_evidence_conflicted"
    elif step812.get("status") == "unavailable" and not b_branches:
        decision = "D. Required checkpoint evidence unavailable"
        reason = "step812_and_qualifying_same_branch_evidence_unavailable"
    else:
        decision = "D. Evidence insufficient for A/B/B0/C"
        reason = "available_evidence_does_not_satisfy_a_b_b0_or_c"
    return {
        "decision": decision,
        "decision_reason": reason,
        "A_candidates": a_candidates,
        "B_branches": b_branches,
        "B0_candidates": b0_candidates,
        "C_reached": c_reached,
        "conflicted_candidates": conflicted,
        "retrain_reference_or_evaluation_insufficient": (
            retrain_reference_or_evaluation_insufficient
        ),
    }


def assemble_decision_v2(
    candidates: list[dict[str, Any]],
    *,
    retrain_reference_or_evaluation_insufficient: bool = False,
) -> dict[str, Any]:
    front, dominated_by = utility_first_pareto(candidates)
    branches = branch_trajectory_evidence(candidates)
    step812 = step812_evidence(candidates)
    regression_branches = [
        branch
        for branch, value in branches.items()
        if value.get("joint_deletion_regression") is True
    ]
    utility_recovery_branches = [
        branch
        for branch, value in branches.items()
        if value.get("joint_utility_recovery") is True
    ]
    classification = classify_decision_v2(
        candidates,
        branches,
        step812,
        retrain_reference_or_evaluation_insufficient=(
            retrain_reference_or_evaluation_insufficient
        ),
    )
    utility_models = [row["model"] for row in candidates if row["utility_pass"]]
    return {
        "candidates": candidates,
        "utility_eligible_candidates": utility_models,
        "pareto_front": front,
        "dominated_by": dominated_by,
        "branch_trajectory_evidence": branches,
        "step812_evidence": step812,
        "step812_deletion_direction": step812.get(
            "step812_deletion_direction"
        ),
        "step812_not_closer_on_forget_train": step812.get(
            "step812_not_closer_on_forget_train"
        ),
        "step812_not_closer_on_forget_validation": step812.get(
            "step812_not_closer_on_forget_validation"
        ),
        "step812_not_closer_on_forget_mia": step812.get(
            "step812_not_closer_on_forget_mia"
        ),
        "joint_deletion_regression": bool(regression_branches),
        "joint_deletion_regression_branches": regression_branches,
        "joint_utility_recovery": bool(utility_recovery_branches),
        "joint_utility_recovery_branches": utility_recovery_branches,
        **classification,
    }


def reference_evidence(
    analysis: dict[str, Any],
    metrics: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    reasons = []
    values = []
    for model in ("original", "retrain"):
        for split in ("overall_validation", "forget_user_validation", "retain_user_validation"):
            row = metrics.get((model, split))
            if row is None:
                reasons.append(f"missing_metric:{model}:{split}")
                continue
            values.extend(
                float(row[key])
                for key in (
                    "auc",
                    "accuracy",
                    "log_loss",
                    "probability_mean",
                    "probability_std",
                    "positive_rate",
                )
            )
            if (
                float(row["probability_std"]) <= COLLAPSE_STD_EPSILON
                or float(row["positive_rate"]) <= 0.0
                or float(row["positive_rate"]) >= 1.0
            ):
                reasons.append(f"reference_probability_collapse:{model}:{split}")
        for group in ("forget", "retain"):
            try:
                values.append(_primary_auc(analysis, model, group))
            except (KeyError, TypeError, ValueError):
                reasons.append(f"missing_primary_mia:{model}:{group}")
    if not _finite(values):
        reasons.append("reference_nan_or_inf")
    return {
        "sufficient": not reasons,
        "reasons": reasons,
        "criteria": (
            "Original/Retrain required validation metrics and primary MIA exist, "
            "are finite, and reference probabilities are non-collapsed"
        ),
    }


def source_snapshot(project_root: Path, config_path: Path, run_name: str) -> dict[str, Any]:
    v1 = project_root / "outputs" / OUTPUT_NAME / "analysis_runs" / run_name
    required = (
        "metrics_and_mia.json",
        "metrics.csv",
        "manifest.json",
        "analysis_run_manifest.json",
        "ANALYSIS_COMPLETED",
    )
    if any(not (v1 / name).is_file() for name in required):
        raise ValueError("decision-v2 requires completed v1 analysis provenance")
    analysis = _read_json(v1 / "metrics_and_mia.json")
    manifest = _read_json(v1 / "manifest.json")
    run_manifest = _read_json(v1 / "analysis_run_manifest.json")
    completed = _read_json(v1 / "ANALYSIS_COMPLETED")
    if (
        analysis.get("schema") != ANALYSIS_SCHEMA
        or analysis.get("run_name") != run_name
        or analysis.get("test_accessed") is not False
        or analysis.get("optimizer_steps_executed") != 0
        or analysis.get("model_loaded") is not False
        or manifest.get("metrics_and_mia_sha256")
        != sha256_file(v1 / "metrics_and_mia.json")
        or manifest.get("metrics_csv_sha256") != sha256_file(v1 / "metrics.csv")
        or manifest.get("test_accessed") is not False
        or run_manifest.get("schema") != FINAL_MANIFEST_SCHEMA
        or run_manifest.get("status") != "COMPLETED"
        or run_manifest.get("test_accessed") is not False
        or any(
            run_manifest.get("artifacts", {}).get(name) != sha256_file(v1 / name)
            for name in ("metrics_and_mia.json", "metrics.csv", "manifest.json")
        )
        or completed != run_manifest
    ):
        raise ValueError("invalid or unsafe v1 analysis source")
    full = full_snapshot(project_root, config_path, run_name)
    if (
        run_manifest.get("full_cache_inventory_sha256")
        != full["full_cache_inventory_sha256"]
        or run_manifest.get("full_cache_content_sha256")
        != full["full_cache_content_sha256"]
    ):
        raise ValueError("v1 provenance and current Full cache fingerprint differ")
    return {
        "v1_root": str(v1.resolve()),
        "v1_files": {name: sha256_file(v1 / name) for name in required},
        "full_run": full["full_run"],
        "full_contract_sha256": full["full_contract_sha256"],
        "full_run_state_sha256": full["full_run_state_sha256"],
        "full_cache_inventory_sha256": full["full_cache_inventory_sha256"],
        "full_cache_content_sha256": full["full_cache_content_sha256"],
        "bootstrap_seed": full["bootstrap_seed"],
        "bootstrap_resamples": full["bootstrap_resamples"],
        "config_sha256": full["config_sha256"],
        "models": full["models"],
        "test_accessed": False,
    }


def _validated_thresholds(project_root: Path, run_name: str) -> dict[str, float]:
    contract = _read_json(
        project_root
        / "outputs"
        / OUTPUT_NAME
        / "full_runs"
        / run_name
        / "contract.json"
    )
    thresholds = contract.get("success_thresholds", {})
    for key, expected in FROZEN_UTILITY_THRESHOLDS.items():
        if float(thresholds.get(key, math.nan)) != expected:
            raise ValueError(f"frozen utility threshold changed: {key}")
    if thresholds.get("prediction_collapse_forbidden") is not True:
        raise ValueError("prediction collapse safety gate is not frozen true")
    return dict(FROZEN_UTILITY_THRESHOLDS)


def build_decision_v2(
    project_root: Path,
    run_name: str,
    source: dict[str, Any],
    *,
    resamples: int,
    cache_loader: Callable[..., Any] = _load_cache_unit,
    specificity_function: Callable[..., dict[str, Any]] = specificity_bootstrap,
) -> dict[str, Any]:
    analysis = _read_json(Path(source["v1_root"]) / "metrics_and_mia.json")
    metrics = {(row["model"], row["split"]): row for row in analysis["metrics"]}
    references = reference_evidence(analysis, metrics)
    candidates = sorted(
        (model for model in analysis["mia"] if _parse_candidate(model)),
        key=lambda model: (_parse_candidate(model)[0], _parse_candidate(model)[1]),
    )
    thresholds = _validated_thresholds(project_root, run_name)
    full_run = Path(source["full_run"])
    cache: dict[tuple[str, str], list[dict[str, Any]]] = {}

    def cached(model: str, split: str) -> list[dict[str, Any]]:
        key = (model, split)
        if key not in cache:
            cache[key] = cache_loader(full_run, model, split)[0]
        return cache[key]

    rows = []
    specificity_results = {}
    for model in candidates:
        specificity = specificity_function(
            cached("original", "forget_train"),
            cached("original", "retain_train"),
            cached("retrain", "forget_train"),
            cached("retrain", "retain_train"),
            cached(model, "forget_train"),
            cached(model, "retain_train"),
            seed=source["bootstrap_seed"],
            resamples=resamples,
        )
        specificity_results[model] = specificity
        rows.append(candidate_evidence(model, metrics, analysis, specificity, thresholds))
    assembled = assemble_decision_v2(
        rows,
        retrain_reference_or_evaluation_insufficient=not references["sufficient"],
    )
    return {
        "schema": DECISION_V2_SCHEMA,
        "scope": "development_and_train_only",
        "run_name": run_name,
        "utility_hard_gate": {
            **thresholds,
            "probability_collapse_forbidden": True,
            "nan_inf_forbidden": True,
            "collapse_definition": (
                "overall or retain-user validation probability_std <= 1e-12 "
                "or positive_rate in {0,1}"
            ),
        },
        "pareto_definition": {
            "utility_first": True,
            "axes": CORE_PARETO_AXES,
            "automatic_best_checkpoint_selection": False,
        },
        "reference_evidence": references,
        **assembled,
        "specificity_bootstrap": specificity_results,
        "missing_checkpoint_policy": "unavailable; never interpolated or substituted",
        "source_v1": source,
        "model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_steps_executed": 0,
        "runtime_device": "cpu",
        "test_loader_built": False,
        "test_accessed": False,
    }


def _pareto_csv(result: dict[str, Any]) -> str:
    fields = (
        "model",
        "branch",
        "step",
        "utility_pass",
        "forget_train_closer_to_retrain",
        "forget_validation_closer_to_retrain",
        "forget_mia_closer_to_retrain",
        "selectivity_ci_pass",
        "global_dememorization",
        "mia_evidence_conflicted",
        "forget_train_l2_to_retrain",
        "forget_mia_auc_gap_to_retrain",
        "I_specific",
        "overall_auc_damage",
        "retain_user_auc_damage",
        "overall_log_loss_damage",
        "retain_user_log_loss_damage",
        "pareto_front",
    )
    front = set(result["pareto_front"])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for candidate in result["candidates"]:
        writer.writerow(
            {**{key: candidate[key] for key in fields[:-1]}, "pareto_front": candidate["model"] in front}
        )
    return stream.getvalue()


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# T5 zero-training decision-v2",
        "",
        f"Decision: **{result['decision']}**",
        "",
        f"Reason: `{result['decision_reason']}`",
        "",
        "Utility-qualified Pareto front:",
    ]
    lines.extend(f"- {model}" for model in result["pareto_front"])
    if not result["pareto_front"]:
        lines.append("- none")
    lines.extend(
        [
            "",
            "No model was loaded, no optimizer step was executed, and test data was not accessed.",
            "",
        ]
    )
    return "\n".join(lines)


def _verify_stage(stage: Path) -> dict[str, str]:
    required = ("decision_v2.json", "pareto_v2.csv", "report_v2.md")
    for name in required:
        if not (stage / name).is_file():
            raise ValueError(f"missing decision-v2 artifact: {name}")
    value = _read_json(stage / "decision_v2.json")
    if (
        value.get("schema") != DECISION_V2_SCHEMA
        or value.get("model_loaded") is not False
        or value.get("optimizer_steps_executed") != 0
        or value.get("test_accessed") is not False
    ):
        raise ValueError("invalid decision-v2 safety fields")
    return {name: sha256_file(stage / name) for name in required}


def run_decision_v2(
    project_root: Path,
    config_path: Path,
    run_name: str,
    *,
    source_function: Callable[..., dict[str, Any]] = source_snapshot,
    build_function: Callable[..., dict[str, Any]] = build_decision_v2,
    git_function: Callable[[Path], dict[str, Any]] = git_provenance,
    implementation_function: Callable[[Path, tuple[str, ...]], dict[str, Any]] = implementation_provenance,
) -> dict[str, Any]:
    project_root, config_path = project_root.resolve(), config_path.resolve()
    run_name = _safe_run_name(run_name)
    git = git_function(project_root)
    require_clean_git(git, "decision-v2 formal publication")
    implementation = implementation_function(project_root, IMPLEMENTATION_FILES)
    source = source_function(project_root, config_path, run_name)
    output_base = project_root / "outputs" / OUTPUT_NAME
    final = output_base / "decision_v2_runs" / run_name
    if final.exists():
        raise FileExistsError("refusing to overwrite existing decision-v2 output")
    control = output_base / "decision_v2_control"
    control.mkdir(parents=True, exist_ok=True)
    lock = control / f"{run_name}.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError(f"decision-v2 already running or stale for RunName {run_name}") from None
    process = _process_record()
    record = {
        "schema": DECISION_V2_LOCK_SCHEMA,
        "status": "RUNNING",
        "run_name": run_name,
        **process,
        "git_commit": git["git_commit"],
        "git": git,
        "implementation": implementation,
        "source_v1_sha256": source["v1_files"],
        "full_cache_inventory_sha256": source["full_cache_inventory_sha256"],
        "full_cache_content_sha256": source["full_cache_content_sha256"],
        "stage": "locked",
        "model_loaded": False,
        "optimizer_steps_executed": 0,
        "test_accessed": False,
    }
    atomic_json(lock / "record.json", record)
    stage = lock / "stage"
    started = time.monotonic()
    try:
        stage.mkdir()
        record["stage"] = "decision_v2"
        atomic_json(lock / "record.json", record)
        result = build_function(
            project_root,
            run_name,
            source,
            resamples=source["bootstrap_resamples"],
        )
        result["provenance"] = {
            "decision_v2_schema": DECISION_V2_SCHEMA,
            "git_commit": record["git_commit"],
            "git_working_tree_clean": True,
            "git": git,
            "implementation": implementation,
            "v1_source_file_sha256": source["v1_files"],
            "full_cache_inventory_sha256": source["full_cache_inventory_sha256"],
            "full_cache_content_sha256": source["full_cache_content_sha256"],
            "model_loaded": False,
            "optimizer_steps_executed": 0,
            "test_accessed": False,
        }
        atomic_json(stage / "decision_v2.json", result)
        atomic_text(stage / "pareto_v2.csv", _pareto_csv(result))
        atomic_text(stage / "report_v2.md", _report(result))
        artifact_hashes = _verify_stage(stage)
        completed = {
            "schema": DECISION_V2_RUN_SCHEMA,
            "status": "COMPLETED",
            "run_name": run_name,
            "decision_v2_schema": DECISION_V2_SCHEMA,
            "git_commit": record["git_commit"],
            "git_working_tree_clean": True,
            "git": git,
            "implementation": implementation,
            "artifacts": artifact_hashes,
            "v1_source_file_sha256": source["v1_files"],
            "full_cache_inventory_sha256": source["full_cache_inventory_sha256"],
            "full_cache_content_sha256": source["full_cache_content_sha256"],
            "model_loaded": False,
            "optimizer_steps_executed": 0,
            "test_accessed": False,
        }
        atomic_json(stage / "manifest.json", completed)
        atomic_text(stage / "COMPLETED", json.dumps(completed, indent=2))
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            raise FileExistsError("refusing to overwrite existing decision-v2 output")
        os.replace(stage, final)
        record.update({"status": "COMPLETED", "stage": "published"})
        try:
            atomic_json(lock / "record.json", record)
            _archive_lock(control, lock, reason="completed")
        except OSError as error:
            print(
                f"[decision-v2:warning] completed lock archival failed: {error}",
                file=sys.stderr,
                flush=True,
            )
        print(
            f"[decision-v2:completed] run={run_name} output={final} "
            f"elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )
        return completed
    except BaseException as error:
        failure = {
            "schema": DECISION_V2_RUN_SCHEMA,
            "status": "FAILED",
            "run_name": run_name,
            "stage": record.get("stage"),
            "exception_type": type(error).__name__,
            "error": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
            "formal_output_published": False,
            "model_loaded": False,
            "optimizer_steps_executed": 0,
            "test_accessed": False,
        }
        atomic_json(lock / "FAILED.json", failure)
        record.update({"status": "FAILED", "failure_sha256": sha256_file(lock / "FAILED.json")})
        atomic_json(lock / "record.json", record)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="T5 zero-training decision-v2")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    run_decision_v2(args.project_root, args.config, args.run_name)


if __name__ == "__main__":
    main()
