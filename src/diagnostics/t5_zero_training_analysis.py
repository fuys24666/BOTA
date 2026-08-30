from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, roc_curve

from src.diagnostics.t5_reconstructed_official import sha256_file
from src.diagnostics.t5_zero_training_audit import (
    CACHE_SCHEMA,
    EXPECTED_COUNTS,
    OUTPUT_NAME,
    SCHEMA,
    SPLITS,
    atomic_json,
    atomic_text,
    canonical_hash,
)

ANALYSIS_SCHEMA = "t5-e2urec-zero-training-analysis-v1"
ATTACKS = {
    "negative_answer_loss": lambda row: -row["answer_sequence_loss"],
    "confidence": lambda row: row["confidence"],
    "negative_entropy": lambda row: -row["binary_entropy"],
    "absolute_yes_no_margin": lambda row: abs(row["yes_no_margin"]),
}


def _safe_corr(function: Any, left: np.ndarray, right: np.ndarray) -> float | None:
    if (
        len(left) < 2
        or np.allclose(left, left[0])
        or np.allclose(right, right[0])
    ):
        return None
    value = function(left, right).statistic
    return float(value) if np.isfinite(value) else None


def _relative(probability: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    epsilon = 1e-12
    p, q = np.clip(probability, epsilon, 1 - epsilon), np.clip(
        reference, epsilon, 1 - epsilon
    )
    middle = (p + q) / 2
    jsd = 0.5 * (
        p * np.log(p / middle)
        + (1 - p) * np.log((1 - p) / (1 - middle))
        + q * np.log(q / middle)
        + (1 - q) * np.log((1 - q) / (1 - middle))
    )
    legacy = p * np.log(p / q) + q * np.log(q / p)
    return {
        "l2_rms": float(np.sqrt(np.mean((probability - reference) ** 2))),
        "standard_jsd": float(np.mean(jsd)),
        "legacy_symmetric_kl": float(np.mean(legacy)),
        "prediction_agreement": float(
            np.mean((probability >= 0.5) == (reference >= 0.5))
        ),
    }


def evaluation_metrics(
    rows: list[dict[str, Any]],
    original: list[dict[str, Any]],
    retrain: list[dict[str, Any]],
) -> dict[str, Any]:
    ids = [row["canonical_sample_id"] for row in rows]
    if ids != [row["canonical_sample_id"] for row in original] or ids != [
        row["canonical_sample_id"] for row in retrain
    ]:
        raise ValueError("metric reference sample order mismatch")
    probability = np.asarray([row["p_yes"] for row in rows], dtype=np.float64)
    original_probability = np.asarray(
        [row["p_yes"] for row in original], dtype=np.float64
    )
    retrain_probability = np.asarray(
        [row["p_yes"] for row in retrain], dtype=np.float64
    )
    gold = np.asarray([row["gold_yes"] for row in rows], dtype=np.int64)
    change, target = probability - original_probability, (
        retrain_probability - original_probability
    )
    return {
        "samples": len(rows),
        "users": len({row["user_id"] for row in rows}),
        "auc": float(roc_auc_score(gold, probability))
        if len(np.unique(gold)) == 2
        else None,
        "accuracy": float(accuracy_score(gold, probability >= 0.5)),
        "log_loss": float(log_loss(gold, probability, labels=[0, 1])),
        "answer_loss_mean": float(
            np.mean([row["answer_sequence_loss"] for row in rows])
        ),
        "confidence_mean": float(np.mean([row["confidence"] for row in rows])),
        "entropy_mean": float(np.mean([row["binary_entropy"] for row in rows])),
        "positive_rate": float(np.mean(probability >= 0.5)),
        "probability_mean": float(np.mean(probability)),
        "probability_std": float(np.std(probability)),
        "relative_original": _relative(probability, original_probability),
        "relative_retrain": _relative(probability, retrain_probability),
        "retrain_direction": {
            "sign_agreement": float(np.mean(np.sign(change) == np.sign(target))),
            "pearson": _safe_corr(pearsonr, change, target),
            "spearman": _safe_corr(spearmanr, change, target),
            "rmse": float(np.sqrt(np.mean((change - target) ** 2))),
        },
        "sample_order_hash": canonical_hash(ids),
        "test_accessed": False,
    }


def attack_metrics(member: np.ndarray, nonmember: np.ndarray) -> dict[str, Any]:
    labels = np.concatenate(
        (np.ones(len(member), dtype=np.int8), np.zeros(len(nonmember), dtype=np.int8))
    )
    scores = np.concatenate((member, nonmember))
    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    advantage = tpr - fpr
    index = int(np.argmax(advantage))
    return {
        "roc_auc": auc,
        "membership_advantage": float(advantage[index]),
        "best_development_threshold": float(thresholds[index]),
        "balanced_accuracy": float((tpr[index] + 1 - fpr[index]) / 2),
        "member": _distribution(member),
        "nonmember": _distribution(nonmember),
    }


def _distribution(value: np.ndarray) -> dict[str, Any]:
    return {
        "samples": int(len(value)),
        "mean": float(np.mean(value)),
        "median": float(np.median(value)),
        "q25": float(np.percentile(value, 25)),
        "q75": float(np.percentile(value, 75)),
    }


def _cluster_resample(
    values: np.ndarray, users: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    unique = np.unique(users)
    sampled = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([values[users == user] for user in sampled])


def user_cluster_mia_bootstrap(
    member: np.ndarray,
    member_users: np.ndarray,
    nonmember: np.ndarray,
    nonmember_users: np.ndarray,
    *,
    seed: int = 42,
    resamples: int = 2000,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    auc, advantage = np.empty(resamples), np.empty(resamples)
    for index in range(resamples):
        sampled_member = _cluster_resample(member, member_users, rng)
        sampled_nonmember = _cluster_resample(nonmember, nonmember_users, rng)
        value = attack_metrics(sampled_member, sampled_nonmember)
        auc[index], advantage[index] = (
            value["roc_auc"],
            value["membership_advantage"],
        )
    return {
        "unit": "user",
        "seed": seed,
        "resamples": resamples,
        "member_users": int(len(np.unique(member_users))),
        "nonmember_users": int(len(np.unique(nonmember_users))),
        "same_user_samples_kept_together": True,
        "auc": _bootstrap_summary(auc),
        "membership_advantage": _bootstrap_summary(advantage),
    }


def _bootstrap_summary(value: np.ndarray) -> dict[str, Any]:
    return {
        "mean": float(np.mean(value)),
        "standard_error": float(np.std(value, ddof=1)),
        "percentile_95_ci": [
            float(item) for item in np.percentile(value, [2.5, 97.5])
        ],
    }


def matched_user_protocol(
    member_rows: list[dict[str, Any]],
    nonmember_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    member_users = {int(row["user_id"]) for row in member_rows}
    nonmember_users = {int(row["user_id"]) for row in nonmember_rows}
    matched_users = member_users & nonmember_users
    matched_member = [
        row for row in member_rows if int(row["user_id"]) in matched_users
    ]
    matched_nonmember = [
        row for row in nonmember_rows if int(row["user_id"]) in matched_users
    ]
    member_by_user = {
        user: sum(int(row["user_id"]) == user for row in matched_member)
        for user in matched_users
    }
    nonmember_by_user = {
        user: sum(int(row["user_id"]) == user for row in matched_nonmember)
        for user in matched_users
    }
    if any(
        member_by_user[user] < 1 or nonmember_by_user[user] < 1
        for user in matched_users
    ):
        raise ValueError("matched user lacks member or nonmember observation")
    return matched_member, matched_nonmember, {
        "train_user_count": len(member_users),
        "validation_user_count": len(nonmember_users),
        "intersection_user_count": len(matched_users),
        "excluded_train_only_users": len(member_users - nonmember_users),
        "excluded_validation_only_users": len(nonmember_users - member_users),
        "matched_member_samples": len(matched_member),
        "matched_nonmember_samples": len(matched_nonmember),
        "matching_rule": "strict user ID set intersection",
        "random_cross_user_pairing": False,
        "user_id_used_as_attack_feature": False,
    }


def matched_user_cluster_bootstrap(
    member: np.ndarray,
    member_users: np.ndarray,
    nonmember: np.ndarray,
    nonmember_users: np.ndarray,
    *,
    seed: int = 42,
    resamples: int = 2000,
) -> dict[str, Any]:
    users = np.intersect1d(np.unique(member_users), np.unique(nonmember_users))
    if not len(users):
        raise ValueError("matched-user bootstrap requires shared users")
    rng = np.random.default_rng(seed)
    auc = np.empty(resamples)
    advantage = np.empty(resamples)
    for index in range(resamples):
        sampled = rng.choice(users, size=len(users), replace=True)
        sampled_member = np.concatenate(
            [member[member_users == user] for user in sampled]
        )
        sampled_nonmember = np.concatenate(
            [nonmember[nonmember_users == user] for user in sampled]
        )
        value = attack_metrics(sampled_member, sampled_nonmember)
        auc[index] = value["roc_auc"]
        advantage[index] = value["membership_advantage"]
    point = attack_metrics(member, nonmember)
    return {
        "unit": "matched_user",
        "seed": seed,
        "resamples": resamples,
        "users": int(len(users)),
        "member_and_nonmember_resampled_together": True,
        "pooled_auc": {
            "point_estimate": point["roc_auc"],
            **_bootstrap_summary(auc),
        },
        "membership_advantage": {
            "point_estimate": point["membership_advantage"],
            **_bootstrap_summary(advantage),
        },
    }


def macro_user_auc(
    member: np.ndarray,
    member_users: np.ndarray,
    nonmember: np.ndarray,
    nonmember_users: np.ndarray,
    *,
    seed: int = 42,
    resamples: int = 2000,
) -> dict[str, Any]:
    users = np.intersect1d(np.unique(member_users), np.unique(nonmember_users))
    values: list[float] = []
    invalid: dict[str, int] = {}
    for user in users:
        member_value = member[member_users == user]
        nonmember_value = nonmember[nonmember_users == user]
        if not len(member_value) or not len(nonmember_value):
            reason = "missing_member_or_nonmember"
            invalid[reason] = invalid.get(reason, 0) + 1
            continue
        labels = np.concatenate(
            (np.ones(len(member_value)), np.zeros(len(nonmember_value)))
        )
        scores = np.concatenate((member_value, nonmember_value))
        values.append(float(roc_auc_score(labels, scores)))
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise ValueError("no valid matched user-level AUC")
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [
            float(np.mean(rng.choice(array, size=len(array), replace=True)))
            for _ in range(resamples)
        ]
    )
    return {
        "valid_users": len(values),
        "invalid_users": int(sum(invalid.values())),
        "invalid_reasons": invalid,
        "equal_user_weight": True,
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "standard_deviation": float(np.std(array)),
        "q25": float(np.percentile(array, 25)),
        "q75": float(np.percentile(array, 75)),
        "bootstrap": {
            "unit": "user_auc",
            "seed": seed,
            "resamples": resamples,
            **_bootstrap_summary(bootstrap),
        },
    }


def _attack_arrays(
    member_rows: list[dict[str, Any]],
    nonmember_rows: list[dict[str, Any]],
    score: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([score(row) for row in member_rows]),
        np.asarray([score(row) for row in nonmember_rows]),
        np.asarray([row["user_id"] for row in member_rows]),
        np.asarray([row["user_id"] for row in nonmember_rows]),
    )


def mia_for_model(
    caches: dict[str, list[dict[str, Any]]],
    *,
    seed: int = 42,
    resamples: int = 2000,
) -> dict[str, Any]:
    definitions = {
        "forget": ("forget_train", "forget_user_validation"),
        "retain": ("retain_train", "retain_user_validation"),
    }
    forget_users = {int(row["user_id"]) for row in caches["forget_train"]}
    retain_users = {int(row["user_id"]) for row in caches["retain_train"]}
    forget_validation_users = {
        int(row["user_id"]) for row in caches["forget_user_validation"]
    }
    retain_validation_users = {
        int(row["user_id"]) for row in caches["retain_user_validation"]
    }
    if (
        forget_users & retain_users
        or not forget_validation_users <= forget_users
        or retain_validation_users & forget_users
        or forget_validation_users & retain_validation_users
    ):
        raise ValueError("Forget/Retain MIA authoritative user groups are mixed")
    output: dict[str, Any] = {}
    for name, (member_split, nonmember_split) in definitions.items():
        member_rows, nonmember_rows = caches[member_split], caches[nonmember_split]
        matched_member, matched_nonmember, matched_counts = matched_user_protocol(
            member_rows, nonmember_rows
        )
        primary_attacks: dict[str, Any] = {}
        secondary_attacks: dict[str, Any] = {}
        for attack_name, score in ATTACKS.items():
            member, nonmember, member_users, nonmember_users = _attack_arrays(
                member_rows, nonmember_rows, score
            )
            secondary_attacks[attack_name] = {
                **attack_metrics(member, nonmember),
                "bootstrap": user_cluster_mia_bootstrap(
                    member,
                    member_users,
                    nonmember,
                    nonmember_users,
                    seed=seed,
                    resamples=resamples,
                ),
            }
            matched_values = _attack_arrays(
                matched_member, matched_nonmember, score
            )
            mm, mn, mm_users, mn_users = matched_values
            primary_attacks[attack_name] = {
                "pooled_matched_user": {
                    **attack_metrics(mm, mn),
                    "sample_count_imbalance_can_still_affect_pooled_auc": True,
                },
                "macro_user_level_auc": macro_user_auc(
                    mm,
                    mm_users,
                    mn,
                    mn_users,
                    seed=seed,
                    resamples=resamples,
                ),
                "user_cluster_bootstrap_pooled": matched_user_cluster_bootstrap(
                    mm,
                    mm_users,
                    mn,
                    mn_users,
                    seed=seed,
                    resamples=resamples,
                ),
            }
        output[name] = {
            "member_split": member_split,
            "nonmember_split": nonmember_split,
            "primary_matched_user": {
                "analysis_role": "primary_matched_user",
                "protocol_version": "t5-matched-user-mia-v1",
                "counts": matched_counts,
                "attacks": primary_attacks,
            },
            "secondary_all_user": {
                "analysis_role": "secondary_all_user",
                "user_composition_confounding_possible": True,
                "attacks": secondary_attacks,
            },
            "all_preregistered_attacks_reported": True,
            "primary_score": "negative_answer_loss",
            "temporal_split": True,
            "membership_and_temporal_shift_not_fully_separable": True,
            "matched_user_limitation": (
                "Matched-user MIA reduces user-identity composition confounding "
                "but does not eliminate temporal distribution shift between "
                "train and validation."
            ),
        }
    return output


def _load_cache_unit(
    full_run: Path, model: str, split: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = full_run / "caches" / model / f"{split}.jsonl"
    manifest_path = full_run / "caches" / model / f"{split}.manifest.json"
    if not data.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"missing published cache {model}:{split}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != CACHE_SCHEMA
        or
        manifest.get("published") is not True
        or manifest.get("data_sha256") != sha256_file(data)
        or manifest.get("test_accessed") is not False
        or manifest.get("rows") != EXPECTED_COUNTS[split]
    ):
        raise ValueError(f"invalid/incomplete cache {model}:{split}")
    rows = [
        json.loads(line)
        for line in data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != manifest["rows"]:
        raise ValueError(f"cache row count mismatch {model}:{split}")
    return rows, manifest


def analyze_full_caches(
    project_root: Path,
    run_name: str,
    *,
    resamples: int = 2000,
    output_root: Path | None = None,
    progress: Callable[[str, int, int, str], None] | None = None,
) -> dict[str, Any]:
    full_run = (
        project_root / "outputs" / OUTPUT_NAME / "full_runs" / run_name
    )
    state = json.loads((full_run / "run_state.json").read_text(encoding="utf-8"))
    if (
        state.get("status") != "INFERENCE_COMPLETED"
        or state.get("test_accessed") is not False
    ):
        raise ValueError("Analyze requires a complete test-free Full cache")
    models = state["models"]
    if "original" not in models or "retrain" not in models:
        raise ValueError("Analyze requires Original and Retrain cache references")
    all_caches: dict[str, dict[str, list[dict[str, Any]]]] = {}
    manifests: dict[str, Any] = {}
    for model in models:
        all_caches[model] = {}
        for split in SPLITS:
            rows, manifest = _load_cache_unit(full_run, model, split)
            all_caches[model][split] = rows
            manifests[f"{model}:{split}"] = manifest
    metric_rows = []
    mia: dict[str, Any] = {}
    total_models = len(models)
    for current, model in enumerate(models, start=1):
        if progress is not None:
            progress("model_start", current, total_models, model)
        for split in SPLITS:
            metric_rows.append(
                {
                    "model": model,
                    "split": split,
                    **evaluation_metrics(
                        all_caches[model][split],
                        all_caches["original"][split],
                        all_caches["retrain"][split],
                    ),
                }
            )
        mia[model] = mia_for_model(
            all_caches[model], seed=42, resamples=resamples
        )
        if progress is not None:
            progress("model_end", current, total_models, model)
    result = {
        "schema": ANALYSIS_SCHEMA,
        "scope": "development_and_train_only",
        "run_name": run_name,
        "split_semantics": {
            "forget_train": "seen Forget training examples",
            "retain_train": "seen Retain training examples",
            "forget_user_validation": "unseen Forget-user development validation",
            "retain_user_validation": "unseen Retain-user development validation",
            "overall_validation": "overall development utility",
        },
        "metrics": metric_rows,
        "mia": mia,
        "mia_evidence_priority": [
            "primary_matched_user_pooled",
            "primary_macro_user_level",
            "secondary_all_user",
        ],
        "mia_primary_score": "negative_answer_loss",
        "temporal_split": True,
        "membership_and_temporal_shift_not_fully_separable": True,
        "cache_manifest_hash": canonical_hash(manifests),
        "optimizer_steps_executed": 0,
        "model_loaded": False,
        "runtime_device": "cpu",
        "test_loader_built": False,
        "test_accessed": False,
    }
    output = output_root or (
        project_root / "outputs" / OUTPUT_NAME / "analysis_runs" / run_name
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite existing Analyze output")
    atomic_json(output / "metrics_and_mia.json", result)
    atomic_text(output / "metrics.csv", _metrics_csv(metric_rows))
    atomic_json(
        output / "manifest.json",
        {
            "schema": ANALYSIS_SCHEMA,
            "metrics_and_mia_sha256": sha256_file(output / "metrics_and_mia.json"),
            "metrics_csv_sha256": sha256_file(output / "metrics.csv"),
            "test_accessed": False,
        },
    )
    return result


def _metrics_csv(rows: list[dict[str, Any]]) -> str:
    fields = (
        "model",
        "split",
        "samples",
        "users",
        "auc",
        "accuracy",
        "log_loss",
        "answer_loss_mean",
        "confidence_mean",
        "entropy_mean",
        "positive_rate",
        "probability_mean",
        "probability_std",
    )
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows({key: row[key] for key in fields} for row in rows)
    return stream.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only MIA and subgroup analysis")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    value = analyze_full_caches(args.project_root.resolve(), args.run_name)
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
