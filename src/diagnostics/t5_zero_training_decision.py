from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.diagnostics.t5_reconstructed_official import sha256_file
from src.diagnostics.t5_zero_training_analysis import (
    _load_cache_unit,
    user_cluster_mia_bootstrap,
)
from src.diagnostics.t5_zero_training_audit import (
    OUTPUT_NAME,
    SCHEMA,
    atomic_json,
    atomic_text,
)

DECISION_SCHEMA = "t5-e2urec-zero-training-pareto-decision-v1"
PRIMARY_MIA_ATTACK = "negative_answer_loss"
PARETO_AXES = {
    "forget_train_l2_to_retrain": "min",
    "forget_train_jsd_to_retrain": "min",
    "forget_mia_auc_gap_to_retrain": "min",
    "forget_user_validation_l2_to_retrain": "min",
    "retain_train_l2_to_original": "min",
    "retain_mia_auc_change_abs_vs_original": "min",
    "retain_user_auc_damage": "min",
    "retain_user_log_loss_damage": "min",
    "overall_auc_damage": "min",
    "overall_accuracy_damage": "min",
    "overall_log_loss_damage": "min",
}


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    comparable = [
        axis
        for axis in PARETO_AXES
        if left.get(axis) is not None and right.get(axis) is not None
    ]
    if len(comparable) != len(PARETO_AXES):
        return False
    no_worse = all(left[axis] <= right[axis] for axis in comparable)
    strictly_better = any(left[axis] < right[axis] for axis in comparable)
    return no_worse and strictly_better


def pareto_front(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, list[str]]]:
    front, dominated_by = [], {}
    for candidate in rows:
        dominators = [
            other["model"]
            for other in rows
            if other is not candidate and dominates(other, candidate)
        ]
        dominated_by[candidate["model"]] = dominators
        if not dominators:
            front.append(candidate["model"])
    return front, dominated_by


def _cluster_l2_improvement(
    reference_error: np.ndarray,
    candidate_error: np.ndarray,
    users: np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> np.ndarray:
    unique = np.unique(users)
    reference_sum, candidate_sum, counts = [], [], []
    for user in unique:
        mask = users == user
        reference_sum.append(float(np.square(reference_error[mask]).sum()))
        candidate_sum.append(float(np.square(candidate_error[mask]).sum()))
        counts.append(int(mask.sum()))
    reference_sum = np.asarray(reference_sum)
    candidate_sum = np.asarray(candidate_sum)
    counts = np.asarray(counts)
    rng = np.random.default_rng(seed)
    result = np.empty(resamples)
    for index in range(resamples):
        sampled = rng.integers(0, len(unique), len(unique))
        denominator = counts[sampled].sum()
        result[index] = np.sqrt(
            reference_sum[sampled].sum() / denominator
        ) - np.sqrt(candidate_sum[sampled].sum() / denominator)
    return result


def specificity_bootstrap(
    original_forget: list[dict[str, Any]],
    original_retain: list[dict[str, Any]],
    retrain_forget: list[dict[str, Any]],
    retrain_retain: list[dict[str, Any]],
    candidate_forget: list[dict[str, Any]],
    candidate_retain: list[dict[str, Any]],
    *,
    seed: int = 42,
    resamples: int = 2000,
) -> dict[str, Any]:
    def arrays(
        original: list[dict[str, Any]],
        retrain: list[dict[str, Any]],
        candidate: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ids = [row["canonical_sample_id"] for row in original]
        if ids != [row["canonical_sample_id"] for row in retrain] or ids != [
            row["canonical_sample_id"] for row in candidate
        ]:
            raise ValueError("specificity cache order mismatch")
        target = np.asarray([row["p_yes"] for row in retrain])
        return (
            np.asarray([row["p_yes"] for row in original]) - target,
            np.asarray([row["p_yes"] for row in candidate]) - target,
            np.asarray([row["user_id"] for row in candidate]),
        )

    ref_f, cand_f, users_f = arrays(
        original_forget, retrain_forget, candidate_forget
    )
    ref_r, cand_r, users_r = arrays(
        original_retain, retrain_retain, candidate_retain
    )
    boot_f = _cluster_l2_improvement(
        ref_f, cand_f, users_f, seed=seed, resamples=resamples
    )
    boot_r = _cluster_l2_improvement(
        ref_r, cand_r, users_r, seed=seed + 1, resamples=resamples
    )
    specific = boot_f - boot_r

    def summarize(value: np.ndarray) -> dict[str, Any]:
        return {
            "mean": float(np.mean(value)),
            "standard_error": float(np.std(value, ddof=1)),
            "percentile_95_ci": [
                float(item) for item in np.percentile(value, [2.5, 97.5])
            ],
        }

    point_f = float(
        np.sqrt(np.mean(np.square(ref_f))) - np.sqrt(np.mean(np.square(cand_f)))
    )
    point_r = float(
        np.sqrt(np.mean(np.square(ref_r))) - np.sqrt(np.mean(np.square(cand_r)))
    )
    return {
        "definition": (
            "I_F/I_R = L2(Original, Retrain) - L2(candidate, Retrain); "
            "I_specific = I_F - I_R"
        ),
        "point_estimates": {
            "I_F": point_f,
            "I_R": point_r,
            "I_specific": point_f - point_r,
        },
        "bootstrap": {
            "I_F": summarize(boot_f),
            "I_R": summarize(boot_r),
            "I_specific": summarize(specific),
        },
        "unit": "user",
        "seed": seed,
        "retain_seed": seed + 1,
        "resamples": resamples,
        "forget_users": int(len(np.unique(users_f))),
        "retain_users": int(len(np.unique(users_r))),
    }


def detect_global_dememorization(
    original_forget_auc: float,
    candidate_forget_auc: float,
    original_retain_auc: float,
    candidate_retain_auc: float,
    original_confidence: float,
    candidate_confidence: float,
    *,
    tolerance: float = 0.02,
) -> bool:
    forget_drop = original_forget_auc - candidate_forget_auc
    retain_drop = original_retain_auc - candidate_retain_auc
    return bool(
        forget_drop > 0
        and retain_drop > 0
        and abs(forget_drop - retain_drop) <= tolerance
        and candidate_confidence < original_confidence
    )


def classify_decision(evidence: dict[str, Any]) -> str:
    if evidence.get("retrain_reference_or_evaluation_insufficient"):
        return "D. Retrain reference or evaluation insufficient"
    if (
        evidence.get("step812_deletion_evidence")
        and evidence.get("joint_deletion_regression")
        and evidence.get("joint_utility_recovery")
    ):
        return "B. Joint objective overwrites deletion"
    if (
        evidence.get("step812_not_closer_on_forget_train")
        and evidence.get("step812_not_closer_on_forget_validation")
        and evidence.get("step812_not_closer_on_forget_mia")
    ):
        return "C. Teacher target insufficient"
    if (
        evidence.get("candidate_meets_utility_thresholds")
        and evidence.get("candidate_forget_evidence")
        and evidence.get("cluster_ci_supports_selectivity")
        and not evidence.get("global_dememorization")
    ):
        return "A. Existing method sufficient"
    return "D. Retrain reference or evaluation insufficient"


def _primary_auc(analysis: dict[str, Any], model: str, group: str) -> float:
    return analysis["mia"][model][group]["primary_matched_user"]["attacks"][
        PRIMARY_MIA_ATTACK
    ]["pooled_matched_user"]["roc_auc"]


def _secondary_auc(analysis: dict[str, Any], model: str, group: str) -> float:
    return analysis["mia"][model][group]["secondary_all_user"]["attacks"][
        PRIMARY_MIA_ATTACK
    ]["roc_auc"]


def mia_evidence_conflict(
    analysis: dict[str, Any], model: str, group: str
) -> bool:
    retrain = "retrain"
    original = "original"
    primary_improvement = abs(
        _primary_auc(analysis, original, group)
        - _primary_auc(analysis, retrain, group)
    ) - abs(
        _primary_auc(analysis, model, group)
        - _primary_auc(analysis, retrain, group)
    )
    secondary_improvement = abs(
        _secondary_auc(analysis, original, group)
        - _secondary_auc(analysis, retrain, group)
    ) - abs(
        _secondary_auc(analysis, model, group)
        - _secondary_auc(analysis, retrain, group)
    )
    return bool(np.sign(primary_improvement) != np.sign(secondary_improvement))


def build_pareto_and_decision(
    project_root: Path,
    run_name: str,
    *,
    resamples: int = 2000,
    analysis_root: Path | None = None,
) -> dict[str, Any]:
    analysis_root = analysis_root or (
        project_root / "outputs" / OUTPUT_NAME / "analysis_runs" / run_name
    )
    analysis_path = analysis_root / "metrics_and_mia.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if (
        analysis.get("test_accessed") is not False
        or analysis.get("schema") != "t5-e2urec-zero-training-analysis-v1"
    ):
        raise ValueError("Pareto requires test-free completed analysis")
    metrics = {(row["model"], row["split"]): row for row in analysis["metrics"]}
    original = "original"
    retrain = "retrain"
    candidates = [
        name
        for name in analysis["mia"]
        if name.startswith(("j0_step", "j2_step", "j4_step", "j5_step"))
    ]
    full_run = project_root / "outputs" / OUTPUT_NAME / "full_runs" / run_name
    cache: dict[tuple[str, str], list[dict[str, Any]]] = {}

    def cached(model: str, split: str) -> list[dict[str, Any]]:
        key = (model, split)
        if key not in cache:
            cache[key] = _load_cache_unit(full_run, model, split)[0]
        return cache[key]

    rows, specificity = [], {}
    original_forget_mia = _primary_auc(analysis, original, "forget")
    retrain_forget_mia = _primary_auc(analysis, retrain, "forget")
    original_retain_mia = _primary_auc(analysis, original, "retain")
    for model in candidates:
        candidate_forget_mia = _primary_auc(analysis, model, "forget")
        candidate_retain_mia = _primary_auc(analysis, model, "retain")
        row = {
            "model": model,
            "forget_train_l2_to_retrain": metrics[
                model, "forget_train"
            ]["relative_retrain"]["l2_rms"],
            "forget_train_jsd_to_retrain": metrics[
                model, "forget_train"
            ]["relative_retrain"]["standard_jsd"],
            "forget_mia_auc_gap_to_retrain": abs(
                candidate_forget_mia - retrain_forget_mia
            ),
            "forget_user_validation_l2_to_retrain": metrics[
                model, "forget_user_validation"
            ]["relative_retrain"]["l2_rms"],
            "retain_train_l2_to_original": metrics[
                model, "retain_train"
            ]["relative_original"]["l2_rms"],
            "retain_mia_auc_change_abs_vs_original": abs(
                candidate_retain_mia - original_retain_mia
            ),
            "retain_user_auc_damage": metrics[
                original, "retain_user_validation"
            ]["auc"]
            - metrics[model, "retain_user_validation"]["auc"],
            "retain_user_log_loss_damage": metrics[
                model, "retain_user_validation"
            ]["log_loss"]
            - metrics[original, "retain_user_validation"]["log_loss"],
            "overall_auc_damage": metrics[original, "overall_validation"]["auc"]
            - metrics[model, "overall_validation"]["auc"],
            "overall_accuracy_damage": metrics[
                original, "overall_validation"
            ]["accuracy"]
            - metrics[model, "overall_validation"]["accuracy"],
            "overall_log_loss_damage": metrics[
                model, "overall_validation"
            ]["log_loss"]
            - metrics[original, "overall_validation"]["log_loss"],
            "forget_mia_improvement_minus_retain_mia_change": (
                abs(original_forget_mia - retrain_forget_mia)
                - abs(candidate_forget_mia - retrain_forget_mia)
                - abs(candidate_retain_mia - original_retain_mia)
            ),
            "global_dememorization": detect_global_dememorization(
                original_forget_mia,
                candidate_forget_mia,
                original_retain_mia,
                candidate_retain_mia,
                metrics[original, "overall_validation"]["confidence_mean"],
                metrics[model, "overall_validation"]["confidence_mean"],
            ),
            "mia_evidence_conflicted": (
                mia_evidence_conflict(analysis, model, "forget")
                or mia_evidence_conflict(analysis, model, "retain")
            ),
        }
        rows.append(row)
        specificity[model] = specificity_bootstrap(
            cached(original, "forget_train"),
            cached(original, "retain_train"),
            cached(retrain, "forget_train"),
            cached(retrain, "retain_train"),
            cached(model, "forget_train"),
            cached(model, "retain_train"),
            resamples=resamples,
        )
    front, dominated_by = pareto_front(rows)
    thresholds = json.loads(
        (full_run / "contract.json").read_text(encoding="utf-8")
    )
    evidence = {
        "retrain_reference_or_evaluation_insufficient": any(
            row["mia_evidence_conflicted"] for row in rows
        ),
        "step812_deletion_evidence": False,
        "joint_deletion_regression": False,
        "joint_utility_recovery": False,
        "step812_not_closer_on_forget_train": False,
        "step812_not_closer_on_forget_validation": False,
        "step812_not_closer_on_forget_mia": False,
        "candidate_meets_utility_thresholds": any(
            row["overall_auc_damage"] <= 0.005
            and row["retain_user_auc_damage"] <= 0.005
            and row["overall_log_loss_damage"] <= 0.01
            and row["retain_user_log_loss_damage"] <= 0.01
            for row in rows
        ),
        "candidate_forget_evidence": any(
            row["forget_train_l2_to_retrain"]
            < metrics["j0_step1200", "forget_train"]["relative_retrain"]["l2_rms"]
            for row in rows
            if ("j0_step1200", "forget_train") in metrics
        ),
        "cluster_ci_supports_selectivity": any(
            value["bootstrap"]["I_specific"]["percentile_95_ci"][0] > 0
            for value in specificity.values()
        ),
        "global_dememorization": any(
            row["global_dememorization"] for row in rows
        ),
    }
    decision = classify_decision(evidence)
    result = {
        "schema": DECISION_SCHEMA,
        "scope": "development_and_train_only",
        "pareto_definition_preregistered": {
            "axes": PARETO_AXES,
            "dominance": (
                "no worse on every fixed axis and strictly better on at least one"
            ),
            "automatic_best_checkpoint_selection": False,
        },
        "rows": rows,
        "pareto_front": front,
        "dominated_by": dominated_by,
        "specificity_bootstrap": specificity,
        "decision_evidence": evidence,
        "decision": decision,
        "mia_evidence_priority": [
            "primary_matched_user_pooled",
            "primary_macro_user_level",
            "secondary_all_user",
        ],
        "mia_primary_score": PRIMARY_MIA_ATTACK,
        "temporal_split": True,
        "membership_and_temporal_shift_not_fully_separable": True,
        "missing_checkpoint_policy": "unavailable; never interpolated or substituted",
        "source_analysis_sha256": sha256_file(analysis_path),
        "optimizer_steps_executed": 0,
        "model_loaded": False,
        "runtime_device": "cpu",
        "test_loader_built": False,
        "test_accessed": False,
    }
    atomic_json(analysis_root / "pareto_and_decision.json", result)
    atomic_text(analysis_root / "pareto.csv", _pareto_csv(rows))
    atomic_text(analysis_root / "report.md", _markdown(result))
    return result


def _pareto_csv(rows: list[dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _markdown(value: dict[str, Any]) -> str:
    lines = [
        "# T5 zero-training development audit",
        "",
        f"Decision: **{value['decision']}**",
        "",
        "No checkpoint is automatically selected. Test data was not accessed.",
        "",
        "Pareto front:",
    ]
    lines.extend(f"- {name}" for name in value["pareto_front"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="T5 checkpoint Pareto and decision")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    result = build_pareto_and_decision(
        args.project_root.resolve(), args.run_name
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
