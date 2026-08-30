"""Development-only, zero-update T5 Retrain-direction separability audit."""
from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import yaml
from peft import get_peft_model_state_dict
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.diagnostics.git_provenance import git_provenance, implementation_provenance, require_clean_git
from src.diagnostics.t5_full_runner import _batch
from src.diagnostics.t5_projected_pilot_10step import _load_pilot_runtime, load_pilot_config
from src.diagnostics.t5_reconstructed_official import JsonPromptDataset, freeze_teacher, load_config, load_legacy_model, move_batch
from src.diagnostics.t5_step817_forget_conflict_audit import _all_finite, _data_lineage, _load_user_map, _nested_keys, _resolve, _safe_name, atomic_json, atomic_text, canonical_hash, directory_hash, sha256_file, tensor_tree_hash


SCHEMA = "t5-retrain-direction-separability-audit-v1"
CACHE_SCHEMA = "t5-retrain-direction-primary-cache-v1"
ANALYSIS_SCHEMA = "t5-retrain-direction-separability-analysis-v1"
CACHE_MARKER = "RETRAIN_DIRECTION_PRIMARY_CACHE_COMPLETED"
CLASSES = ("DOWN", "EQUIVALENT", "UP")
RULES = (
    "augmented_to_step816", "original_to_step816", "augmented_vs_original",
    "inverse_memory_advantage", "retain_label_calibrated", "always_down",
    "always_up", "always_equivalent", "majority_class",
)
SUBGROUPS = ("all", "active", "inactive", "observed_yes", "observed_no", "active_yes", "active_no")
FORBIDDEN_KEYS = {"logits", "token_tensor", "input_ids", "target_ids", "raw_sample", "raw_samples", "model", "coefficients", "intercept"}
IMPLEMENTATION_FILES = (
    "src/diagnostics/t5_retrain_direction_separability_audit.py",
    "configs/t5_retrain_direction_separability_audit_v1.yaml",
    "scripts/diagnostics/t5_retrain_direction_separability_audit_v1.ps1",
    "docs/t5_retrain_direction_separability_audit_v1.md",
    "src/diagnostics/t5_step817_selector_aware_contrastive_audit.py",
    "src/diagnostics/t5_step817_retrain_free_contrastive_audit.py",
    "src/diagnostics/t5_projected_pilot_10step.py",
)
FULL_PRODUCER_COMPATIBILITY = {
    (
        "2e8244634036a1827be48a4f94f44c715ac17672",
        "adf997307cc94be7fd3ba26647a9dab4674912f6e78539be11887896e64e9650",
    ),
}


def load_audit_config(path: Path, root: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("development_only") is not True or value.get("test_access_policy") != "forbidden":
        raise ValueError("direction audit schema/scope mismatch")
    if value.get("primary_scope") != "forget_user_validation" or value.get("secondary_scope") != "disabled_resource_bounded_primary_only":
        raise ValueError("audit scope changed")
    if value.get("direction") != {"epsilon_prob": .001, "classes": ["DOWN", "EQUIVALENT", "UP"], "primary_truth": "delta_R_prob", "secondary_truths": ["delta_R_logp", "delta_R_margin", "signed_yes_no_jsd"], "answer_position": "first_target_token", "no_token_id": 465, "yes_token_id": 2163}:
        raise ValueError("direction definition changed")
    if value.get("fixed_rules") != {"primary": "augmented_to_step816", "diagnostics": ["original_to_step816", "augmented_vs_original", "inverse_memory_advantage", "retain_label_calibrated"], "baselines": ["always_down", "always_up", "always_equivalent", "majority_class"], "retrain_forbidden": True}:
        raise ValueError("fixed rules changed")
    expected_oracle = {"model": "multinomial_logistic_regression", "binary_model": "logistic_regression_up_down_only", "penalty": "l2", "C": 1.0, "solver": "lbfgs", "max_iter": 1000, "folds": 5, "splitter": "GroupKFold", "group": "authoritative_user_id", "seed": 42, "train_fold_standardization_only": True, "hyperparameter_search": False, "persist_model_or_coefficients": False}
    if value.get("oracle") != expected_oracle:
        raise ValueError("oracle preregistration changed")
    if value.get("bootstrap") != {"cluster": "authoritative_user_id", "seed": 42, "resamples": 2000, "confidence": .95, "paired_baseline_difference": True}:
        raise ValueError("bootstrap preregistration changed")
    if value.get("gates") != {"active_users_min": 30, "binary_coverage_min": .5, "binary_balanced_accuracy_min": .6, "balanced_accuracy_improvement_min": .05, "binary_mcc_min": .1, "spearman_min": .2, "all_binary_balanced_accuracy_ci_upper_min": .5, "all_spearman_ci_upper_min": 0.0}:
        raise ValueError("practical gates changed")
    if value.get("evaluation") != {"samples": 3336, "users": 95, "batch_size": 4, "groups": ["all", "active", "inactive", "observed_yes", "observed_no", "active_yes", "active_no"]}:
        raise ValueError("evaluation plan changed")
    features = value.get("features", {}).get("names", [])
    forbidden_feature_fragments = ("retrain", "user_id", "sample_id", "source_index", "canonical")
    if len(features) != 22 or len(set(features)) != len(features) or any(any(fragment in name.lower() for fragment in forbidden_feature_fragments) for name in features):
        raise ValueError("oracle feature list invalid or Retrain/leakage-bearing")
    value["_path"] = str(path.resolve()); value["_sha256"] = sha256_file(path)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def direction_label(delta: float, epsilon: float = .001) -> str:
    if epsilon != .001 or not math.isfinite(float(delta)):
        raise ValueError("invalid/floating direction threshold")
    if delta > epsilon:
        return "UP"
    if delta < -epsilon:
        return "DOWN"
    return "EQUIVALENT"


def signed_margin(logits_no_yes: torch.Tensor, observed: torch.Tensor, no_id: int = 465, yes_id: int = 2163) -> torch.Tensor:
    if logits_no_yes.ndim != 2 or logits_no_yes.shape[1] != 2 or observed.ndim != 1 or len(observed) != len(logits_no_yes):
        raise ValueError("signed margin requires aligned Yes/No logits")
    valid = (observed == no_id) | (observed == yes_id)
    if not bool(valid.all()):
        raise ValueError("observed answer is not authoritative Yes/No")
    no_minus_yes = logits_no_yes[:, 0] - logits_no_yes[:, 1]
    return torch.where(observed == yes_id, -no_minus_yes, no_minus_yes)


def bernoulli_jsd(left: float, right: float) -> float:
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (left, right)):
        raise ValueError("invalid Bernoulli probability")
    l = np.asarray([1-left, left], dtype=np.float64); r = np.asarray([1-right, right], dtype=np.float64); m = (l+r)/2
    safe = lambda x, y: float(np.sum(np.where(x > 0, x * np.log(x / np.clip(y, 1e-300, None)), 0.0)))
    return .5 * safe(l, m) + .5 * safe(r, m)


def fixed_rule_scores(row: dict[str, Any], retain_medians: dict[str, float]) -> dict[str, float]:
    forbidden = {key for key in row if "retrain" in key.lower() or key.startswith("p_R") or key.startswith("delta_R")}
    # Truth may coexist in an audit row, but it is never read by this function.
    required = ("p_B_observed", "p_O_observed", "p_A_observed", "observed_answer_is_yes", "sab_advantage")
    if any(key not in row for key in required) or set(retain_medians) != {"yes", "no"}:
        raise ValueError("fixed-rule inputs incomplete")
    b, o, a = (float(row[key]) for key in ("p_B_observed", "p_O_observed", "p_A_observed"))
    median = retain_medians["yes" if bool(row["observed_answer_is_yes"]) else "no"]
    scores = {
        "augmented_to_step816": a-b,
        "original_to_step816": o-b,
        "augmented_vs_original": a-o,
        "inverse_memory_advantage": -(o-a),
        "retain_label_calibrated": median-b,
        "always_down": -1.0,
        "always_up": 1.0,
        "always_equivalent": 0.0,
    }
    if forbidden and any(key in scores for key in forbidden):
        raise RuntimeError("Retrain leaked into fixed rule")
    return scores


def feature_vector(row: dict[str, Any], feature_names: list[str]) -> np.ndarray:
    forbidden = {"p_R_observed", "delta_R_prob", "delta_R_logp", "delta_R_margin", "signed_yes_no_jsd", "authoritative_user_id", "sample_order_position", "fold_id", "direction_label"}
    if any(name in forbidden or "retrain" in name.lower() for name in feature_names):
        raise ValueError("Retrain/identifier feature forbidden")
    values = np.asarray([float(row[name]) for name in feature_names], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite feature")
    return values


def _validate_history(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    authority = config["authority"]
    specs = {
        "p10": ("run_manifest.json", "P10-C"), "fc": ("full_manifest.json", "FC-C"),
        "lft": ("full_manifest.json", "LFT-B"), "pb": ("full_manifest.json", "PB-C"),
        "sab": ("full_manifest.json", "SAB-C"),
    }
    output = {}
    for key, (manifest_name, category) in specs.items():
        full = _resolve(root, authority[f"{key}_full"]); analysis = _resolve(root, authority[f"{key}_analysis"])
        actual = {
            "contract_sha256": sha256_file(full/"contract.json"),
            "manifest_sha256": sha256_file(full/manifest_name),
            "analysis_sha256": sha256_file(analysis/"analysis.json"),
        }
        if any(actual[name] != authority[f"{key}_{name}"] for name in actual):
            raise ValueError(f"{key} authority SHA mismatch")
        decision = _read_json(analysis/"analysis.json")
        if decision.get("category") != category or decision.get("test_accessed") is not False:
            raise ValueError(f"{key} authority classification changed")
        if key == "sab" and (decision.get("next_action") != authority["sab_next_action"] or decision.get("optimizer_steps_committed") != 0 or decision.get("step817_checkpoint_published") is not False):
            raise ValueError("SAB-C authority invariant changed")
        output[key] = {"full": str(full), "analysis": str(analysis), "category": category, **actual, "reclassified": False}
    selector = _resolve(root, authority["development_selector_path"])
    if sha256_file(selector/"manifest.json") != authority["development_selector_manifest_sha256"] or sha256_file(selector/"rows.jsonl") != authority["development_selector_rows_sha256"] or sha256_file(selector/"summary.json") != authority["development_selector_summary_sha256"]:
        raise ValueError("development selector cache SHA mismatch")
    summary = _read_json(selector/"summary.json")
    if summary.get("selector_sha256") != authority["development_selector_sha256"] or summary.get("active_samples") != 126 or summary.get("inactive_samples") != 3210 or summary.get("active_users") != 46 or summary.get("test_accessed") is not False:
        raise ValueError("development selector frozen summary changed")
    output["development_selector"] = {"path": str(selector), "manifest_sha256": authority["development_selector_manifest_sha256"], "rows_sha256": authority["development_selector_rows_sha256"], "summary_sha256": authority["development_selector_summary_sha256"], "selector_sha256": authority["development_selector_sha256"], "active_samples": 126, "inactive_samples": 3210, "active_users": 46}
    return output


def preflight(root: Path, config_path: Path, *, git_function: Callable[[Path],dict[str,Any]]=git_provenance, implementation_function: Callable[[Path,Iterable[str]],dict[str,Any]]=implementation_provenance) -> dict[str, Any]:
    config = load_audit_config(config_path, root); history = _validate_history(root, config); checkpoint = _resolve(root, config["source_checkpoint"]); authority = config["authority"]
    if sha256_file(checkpoint/"state.pt") != authority["step816_state_sha256"] or sha256_file(checkpoint/"manifest.json") != authority["step816_manifest_sha256"]:
        raise ValueError("step816 authority changed")
    checkpoint_manifest = _read_json(checkpoint/"manifest.json")
    if checkpoint_manifest.get("step") != 816 or checkpoint_manifest.get("next_optimizer_step") != 817:
        raise ValueError("step816 continuation changed")
    base = load_config(_resolve(root, config["base_config"]), root); lineage, indices, users = _data_lineage(root, base, _resolve(root, config["protocol_root"])); paths = base["paths"]
    models = {"original": {"path": str(Path(paths["original"]).resolve()), "sha256": sha256_file(Path(paths["original"]))}, "augmented": {"path": str(Path(paths["augmented_teacher"]).resolve()), "sha256": sha256_file(Path(paths["augmented_teacher"]))}, "retrain": {"path": str(Path(paths["retrain_reference"]).resolve()), "sha256": sha256_file(Path(paths["retrain_reference"]))}}
    tokenizer = directory_hash(Path(paths["model_dir"])); expected = config["lineage_sha256"]
    actual = {"original": models["original"]["sha256"], "augmented": models["augmented"]["sha256"], "retrain": models["retrain"]["sha256"], "tokenizer_directory": tokenizer["canonical_sha256"], "validation": lineage["data"]["overall_validation"]["sha256"], "forget_train": lineage["data"]["forget_train"]["sha256"], "validation_user_sidecar": lineage["validation_sidecar"]["sha256"], "forget_validation_indices": lineage["validation_splits"]["forget_user_validation"]["indices_sha256"], "retain_validation_indices": lineage["validation_splits"]["retain_user_validation"]["indices_sha256"]}
    if actual != expected or len(indices["forget_user_validation"]) != 3336 or len(set(users["forget_user_validation"])) != 95:
        raise ValueError("model/data/user/sample lineage changed")
    return json.loads(json.dumps({
        "schema": SCHEMA, "mode": "Preflight", "development_only": True, "config_sha256": config["_sha256"],
        "git": git_function(root), "implementation": implementation_function(root, IMPLEMENTATION_FILES), "historical_authorities": history,
        "source_checkpoint": {"path": str(checkpoint), "state_sha256": authority["step816_state_sha256"], "manifest_sha256": authority["step816_manifest_sha256"], "next_optimizer_step": 817},
        "model_lineage": models, "tokenizer_lineage": tokenizer, "data_lineage": lineage,
        "sample_order": {"forget": expected["forget_validation_indices"], "retain": expected["retain_validation_indices"], "user_sidecar": expected["validation_user_sidecar"]},
        "primary": {"scope": "forget_user_validation", "samples": 3336, "users": 95, "groups": config["evaluation"]["groups"]},
        "secondary": {"enabled": False, "reason": "resource_bounded_primary_only", "exploratory_if_ever_enabled": True, "may_not_affect_RD_classification": True},
        "direction": config["direction"], "fixed_rules": config["fixed_rules"], "features": config["features"], "oracle": config["oracle"], "bootstrap": config["bootstrap"], "gates": config["gates"],
        "resource_estimate": config["resource_estimate"], "larger_evaluation_batch_note": "mathematically_safe_in_eval_but_frozen_batch_size_4_must_not_change_for_this_protocol",
        "model_loaded": False, "cuda_used": False, "model_parameters_modified": False, "optimizer_constructed": False, "optimizer_steps_committed": 0, "backward_called": False, "candidate_update_generated": False, "step817_checkpoint_published": False, "retrain_used_for_labels_only": True, "retrain_used_for_fixed_rules": False, "retrain_used_for_feature_construction": False, "test_loader_built": False, "test_accessed": False,
    }, sort_keys=True))


def _balanced_accuracy_binary(truth: np.ndarray, pred: np.ndarray) -> float | None:
    recalls = [float(np.mean(pred[truth == label] == label)) for label in ("DOWN", "UP") if np.any(truth == label)]
    return None if len(recalls) < 2 else float(np.mean(recalls))


def three_class_metrics(truth: Iterable[str], pred: Iterable[str]) -> dict[str, Any]:
    truth, pred = np.asarray(list(truth)), np.asarray(list(pred)); labels = list(CLASSES)
    if len(truth) == 0 or len(truth) != len(pred):
        return {"available": False, "samples": len(truth)}
    precision, recall, f1, support = precision_recall_fscore_support(truth, pred, labels=labels, zero_division=0)
    class_metrics = {label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i,label in enumerate(labels)}
    recalls = [class_metrics[label]["recall"] for label in labels if class_metrics[label]["support"] > 0]
    return {"available": True, "samples": len(truth), "accuracy": float(accuracy_score(truth,pred)), "balanced_accuracy": float(np.mean(recalls)), "macro_f1": float(f1_score(truth,pred,labels=labels,average="macro",zero_division=0)), "multiclass_mcc": float(matthews_corrcoef(truth,pred)), "confusion_matrix": confusion_matrix(truth,pred,labels=labels).tolist(), "classes": class_metrics, "equivalent_coverage": float(np.mean(pred=="EQUIVALENT"))}


def binary_metrics(truth: Iterable[str], pred: Iterable[str], total_samples: int | None=None) -> dict[str, Any]:
    truth, pred = np.asarray(list(truth)), np.asarray(list(pred)); mask = truth != "EQUIVALENT"; t,p = truth[mask],pred[mask]
    total = len(truth) if total_samples is None else total_samples
    if len(t) == 0:
        return {"available": False, "samples": 0, "coverage": 0.0}
    if not set(t.tolist()) <= {"DOWN", "UP"} or not set(p.tolist()) <= set(CLASSES):
        raise ValueError("binary direction labels invalid")
    valid_pred = p != "EQUIVALENT"; sign_accuracy = float(np.mean(p == t)); bal = _balanced_accuracy_binary(t,p)
    # An EQUIVALENT prediction is an abstention on a non-equivalent truth.  It
    # counts as incorrect for accuracy/recall.  For the symmetric correlation
    # form of binary MCC it has zero direction rather than a fabricated sign.
    truth_sign=np.where(t=="UP",1.0,-1.0);pred_sign=np.where(p=="UP",1.0,np.where(p=="DOWN",-1.0,0.0))
    mcc=0.0 if np.std(truth_sign)==0 or np.std(pred_sign)==0 else float(np.corrcoef(truth_sign,pred_sign)[0,1])
    tp=int(np.sum((t=="UP")&(p=="UP")));fp=int(np.sum((t=="DOWN")&(p=="UP")));fn=int(np.sum((t=="UP")&(p!="UP")));denominator=2*tp+fp+fn;up_f1=0.0 if denominator==0 else 2*tp/denominator
    return {"available": bal is not None, "samples": len(t), "coverage": len(t)/max(total,1), "predicted_non_equivalent_coverage": float(np.mean(valid_pred)), "equivalent_prediction_abstentions": int(np.sum(~valid_pred)), "equivalent_predictions_scored_as_incorrect": True, "effective_binary_prediction_policy": "zero_direction_abstention_for_mcc_and_incorrect_for_accuracy_recall", "sign_accuracy": sign_accuracy, "balanced_accuracy": bal, "f1": float(up_f1), "mcc": mcc}


def continuous_metrics(score: Iterable[float], truth: Iterable[float]) -> dict[str, Any]:
    score, truth = np.asarray(list(score),dtype=np.float64),np.asarray(list(truth),dtype=np.float64)
    if len(score)<3 or not np.isfinite(score).all() or not np.isfinite(truth).all():
        return {"available": False, "samples": len(score)}
    pear = None if np.std(score)==0 or np.std(truth)==0 else float(pearsonr(score,truth).statistic)
    spear = None if np.std(score)==0 or np.std(truth)==0 else float(spearmanr(score,truth).statistic)
    slope = None if np.var(score)==0 else float(np.cov(score,truth,ddof=0)[0,1]/np.var(score))
    return {"available": True, "samples": len(score), "pearson": pear, "spearman": spear, "sign_agreement": float(np.mean(np.sign(score)==np.sign(truth))), "calibration_slope": slope, "mean_absolute_directional_error": float(np.mean(np.abs(score-truth)))}


def _subgroup(rows: list[dict[str,Any]], name: str) -> list[dict[str,Any]]:
    predicates = {
        "all": lambda r: True, "active": lambda r: r["sab_active"], "inactive": lambda r: not r["sab_active"],
        "observed_yes": lambda r: r["observed_answer_is_yes"], "observed_no": lambda r: not r["observed_answer_is_yes"],
        "active_yes": lambda r: r["sab_active"] and r["observed_answer_is_yes"], "active_no": lambda r: r["sab_active"] and not r["observed_answer_is_yes"],
    }
    if name not in predicates: raise KeyError(name)
    return [row for row in rows if predicates[name](row)]


def _subgroup_indices(rows: list[dict[str,Any]], name: str) -> np.ndarray:
    selected_ids = {id(row) for row in _subgroup(rows, name)}
    return np.asarray([index for index, row in enumerate(rows) if id(row) in selected_ids], dtype=int)


def _prediction_from_score(score: float) -> str:
    return direction_label(float(score), .001)


def _bootstrap_gate(rows: list[dict[str,Any]], pred: np.ndarray, score: np.ndarray, *, resamples: int, seed: int) -> dict[str,Any]:
    truth = np.asarray([row["direction_label"] for row in rows]); delta = np.asarray([row["delta_R_prob"] for row in rows],dtype=np.float64); users = np.asarray([row["authoritative_user_id"] for row in rows]); unique = sorted(set(users.tolist())); nonneutral = truth != "EQUIVALENT"; majority = "UP" if np.sum(truth[nonneutral]=="UP") >= np.sum(truth[nonneutral]=="DOWN") else "DOWN"; majority_pred=np.full(len(rows),majority)
    point_bal=_balanced_accuracy_binary(truth[nonneutral],pred[nonneutral]); point_major=_balanced_accuracy_binary(truth[nonneutral],majority_pred[nonneutral]); point_mcc=binary_metrics(truth,pred,len(rows)).get("mcc") if np.any(nonneutral) else None; point_spear=continuous_metrics(score,delta).get("spearman")
    rng=np.random.default_rng(seed); draws={"binary_balanced_accuracy":[],"balanced_accuracy_improvement_vs_majority":[],"binary_mcc":[],"spearman":[]}
    by_user={user:np.flatnonzero(users==user) for user in unique}
    for _ in range(resamples):
        sampled=rng.choice(unique,len(unique),replace=True); idx=np.concatenate([by_user[int(user)] for user in sampled]); t,p,m=truth[idx],pred[idx],majority_pred[idx]; mask=t!="EQUIVALENT"; bal=_balanced_accuracy_binary(t[mask],p[mask]); base=_balanced_accuracy_binary(t[mask],m[mask])
        if bal is not None and base is not None: draws["binary_balanced_accuracy"].append(bal);draws["balanced_accuracy_improvement_vs_majority"].append(bal-base)
        if np.any(mask): draws["binary_mcc"].append(binary_metrics(t,p,len(t))["mcc"])
        cont=continuous_metrics(score[idx],delta[idx]).get("spearman")
        if cont is not None: draws["spearman"].append(cont)
    points={"binary_balanced_accuracy":point_bal,"balanced_accuracy_improvement_vs_majority":None if point_bal is None or point_major is None else point_bal-point_major,"binary_mcc":point_mcc,"spearman":point_spear}; result={}
    for key,values in draws.items():
        result[key]={"point":points[key],"ci95_low":None if not values else float(np.percentile(values,2.5)),"ci95_high":None if not values else float(np.percentile(values,97.5)),"valid_draws":len(values)}
    return {"cluster":"authoritative_user_id","seed":seed,"resamples":resamples,"users":len(unique),"samples":len(rows),"paired_majority_difference_directly_bootstrapped":True,"metrics":result}


def evaluate_rule(rows: list[dict[str,Any]], rule: str, *, resamples: int=2000, seed: int=42, override_predictions: np.ndarray|None=None, override_score: np.ndarray|None=None) -> dict[str,Any]:
    if not rows: return {"available":False,"samples":0,"users":0}
    truth=np.asarray([row["direction_label"] for row in rows]); score=np.asarray([row["rule_scores"][rule] for row in rows],dtype=np.float64) if override_score is None else np.asarray(override_score,dtype=np.float64); pred=np.asarray([_prediction_from_score(value) for value in score]) if override_predictions is None else np.asarray(override_predictions)
    return {"available":True,"samples":len(rows),"users":len({row["authoritative_user_id"] for row in rows}),"three_class":three_class_metrics(truth,pred),"binary":binary_metrics(truth,pred,len(rows)),"continuous":{"delta_R_prob":continuous_metrics(score,[r["delta_R_prob"] for r in rows]),"delta_R_logp":continuous_metrics(score,[r["delta_R_logp"] for r in rows]),"delta_R_margin":continuous_metrics(score,[r["delta_R_margin"] for r in rows])},"bootstrap":_bootstrap_gate(rows,pred,score,resamples=resamples,seed=seed)}


def assign_group_folds(rows: list[dict[str,Any]], folds: int=5, seed: int=42) -> dict[str,Any]:
    groups=np.asarray([row["authoritative_user_id"] for row in rows]); labels=np.asarray([row["direction_label"] for row in rows]); splitter=GroupKFold(n_splits=folds,shuffle=True,random_state=seed); assigned=np.full(len(rows),-1); details=[]
    for fold,(train,test) in enumerate(splitter.split(np.zeros((len(rows),1)),labels,groups)):
        train_users=set(groups[train].tolist());test_users=set(groups[test].tolist())
        if train_users&test_users:raise RuntimeError("GroupKFold user leakage")
        assigned[test]=fold;details.append({"fold":fold,"train_users":len(train_users),"test_users":len(test_users),"train_user_hash":canonical_hash(sorted(train_users)),"test_user_hash":canonical_hash(sorted(test_users)),"overlap":0})
    if np.any(assigned<0):raise RuntimeError("incomplete fold assignment")
    for index,row in enumerate(rows):row["fold_id"]=int(assigned[index])
    return {"folds":details,"fold_assignment_sha256":canonical_hash(assigned.tolist()),"group":"authoritative_user_id","seed":seed,"splitter":"GroupKFold"}


def run_oracle(rows: list[dict[str,Any]], feature_names: list[str], config: dict[str,Any], *, resamples: int=2000) -> dict[str,Any]:
    X=np.stack([feature_vector(row,feature_names) for row in rows]); y=np.asarray([row["direction_label"] for row in rows]); fold_ids=np.asarray([row["fold_id"] for row in rows]); groups=np.asarray([row["authoritative_user_id"] for row in rows]); pred=np.empty(len(rows),dtype=object); score=np.zeros(len(rows)); binary_pred=np.full(len(rows),"EQUIVALENT",dtype=object); binary_score=np.zeros(len(rows)); fold_reports=[]
    for fold in range(config["oracle"]["folds"]):
        train=fold_ids!=fold;test=fold_ids==fold
        if set(groups[train])&set(groups[test]):raise RuntimeError("oracle user leakage")
        scaler=StandardScaler().fit(X[train]); xtrain=scaler.transform(X[train]);xtest=scaler.transform(X[test]); model=LogisticRegression(C=1.0,penalty="l2",solver="lbfgs",max_iter=1000,random_state=42).fit(xtrain,y[train]); pred[test]=model.predict(xtest);proba=model.predict_proba(xtest);classes=list(model.classes_);score[test]=proba[:,classes.index("UP")]-proba[:,classes.index("DOWN")]
        train_binary=train&(y!="EQUIVALENT"); test_binary=test&(y!="EQUIVALENT"); bmodel=LogisticRegression(C=1.0,penalty="l2",solver="lbfgs",max_iter=1000,random_state=42).fit(scaler.transform(X[train_binary]),y[train_binary]); binary_pred[test_binary]=bmodel.predict(scaler.transform(X[test_binary]));bproba=bmodel.predict_proba(scaler.transform(X[test_binary]));bclasses=list(bmodel.classes_);binary_score[test_binary]=bproba[:,bclasses.index("UP")]-bproba[:,bclasses.index("DOWN")]
        fold_reports.append({"fold":fold,"train_users":len(set(groups[train])),"test_users":len(set(groups[test])),"train_user_hash":canonical_hash(sorted(set(groups[train].tolist()))),"test_user_hash":canonical_hash(sorted(set(groups[test].tolist()))),"user_overlap":0,"standardizer_fit_scope":"train_fold_only","hyperparameter_search":False,"model_or_coefficients_persisted":False,"three_class":three_class_metrics(y[test],pred[test]),"binary":binary_metrics(y[test],binary_pred[test],int(np.sum(test)))})
    evaluations={}
    for subgroup in SUBGROUPS:
        indices=_subgroup_indices(rows,subgroup)
        subset=[rows[index] for index in indices]
        evaluations[subgroup]=evaluate_rule(subset,"augmented_to_step816",resamples=resamples,seed=42,override_predictions=pred[indices],override_score=score[indices]) if len(indices) else {"available":False,"samples":0}
        mask=np.asarray([row["direction_label"]!="EQUIVALENT" for row in subset])
        if len(indices) and np.any(mask): evaluations[subgroup]["binary_oracle"]={"metrics":binary_metrics(np.asarray([r["direction_label"] for r in subset])[mask],binary_pred[indices][mask],len(subset)),"continuous":continuous_metrics(binary_score[indices][mask],[r["delta_R_prob"] for r in subset if r["direction_label"]!="EQUIVALENT"])}
    return {"model":"multinomial_logistic_regression","binary_model":"logistic_regression_up_down_only","fixed_C":1.0,"penalty":"l2","solver":"lbfgs","hyperparameter_search":False,"feature_names":feature_names,"folds":fold_reports,"evaluations":evaluations,"user_leakage":False,"standardization":"train_fold_only","model_or_coefficients_persisted":False}


def direction_distribution(rows: list[dict[str,Any]]) -> dict[str,Any]:
    def counts(selected): return {label:sum(row["direction_label"]==label for row in selected) for label in CLASSES}
    by_user={};
    for row in rows:by_user.setdefault(row["authoritative_user_id"],[]).append(row["direction_label"])
    mixed=sum(len(set(labels))>1 for labels in by_user.values()); consistency=[max(labels.count(label) for label in set(labels))/len(labels) for labels in by_user.values()]
    active=_subgroup(rows,"active");inactive=_subgroup(rows,"inactive"); truth=[r["direction_label"] for r in rows]
    return {"samples":len(rows),"users":len(by_user),"all":counts(rows),"by_observed_answer":{"yes":counts(_subgroup(rows,"observed_yes")),"no":counts(_subgroup(rows,"observed_no"))},"by_sab_selector":{"active":counts(active),"inactive":counts(inactive)},"delta_distributions":{key:{"mean":float(np.mean([r[key] for r in rows])),"std":float(np.std([r[key] for r in rows])),"min":float(np.min([r[key] for r in rows])),"p05":float(np.percentile([r[key] for r in rows],5)),"median":float(np.median([r[key] for r in rows])),"p95":float(np.percentile([r[key] for r in rows],95)),"max":float(np.max([r[key] for r in rows]))} for key in ("delta_R_prob","delta_R_logp","delta_R_margin","signed_yes_no_jsd")},"users_with_mixed_direction":mixed,"mixed_user_rate":mixed/len(by_user),"mean_within_user_direction_consistency":float(np.mean(consistency)),"uniform_observed_probability_down_max_accuracy":truth.count("DOWN")/len(truth),"augmented_to_step816_accuracy":float(np.mean([_prediction_from_score(r["rule_scores"]["augmented_to_step816"])==r["direction_label"] for r in rows])),"active_down_rate":sum(r["direction_label"]=="DOWN" for r in active)/max(len(active),1),"inactive_down_rate":sum(r["direction_label"]=="DOWN" for r in inactive)/max(len(inactive),1),"active_contains_up_and_down":counts(active)["UP"]>0 and counts(active)["DOWN"]>0}


def _gate_from_evaluation(evaluation: dict[str,Any], gates: dict[str,Any]) -> dict[str,Any]:
    bootstrap=evaluation["bootstrap"]["metrics"];binary=evaluation["binary"];spear=bootstrap["spearman"];improve=bootstrap["balanced_accuracy_improvement_vs_majority"];mcc=bootstrap["binary_mcc"]
    checks={"active_users":evaluation["users"]>=gates["active_users_min"],"coverage":binary["coverage"]>=gates["binary_coverage_min"],"balanced_accuracy":binary["balanced_accuracy"] is not None and binary["balanced_accuracy"]>=gates["binary_balanced_accuracy_min"],"balanced_improvement_point":improve["point"] is not None and improve["point"]>=gates["balanced_accuracy_improvement_min"],"balanced_improvement_ci":improve["ci95_low"] is not None and improve["ci95_low"]>0,"mcc_point":mcc["point"] is not None and mcc["point"]>=gates["binary_mcc_min"],"mcc_ci":mcc["ci95_low"] is not None and mcc["ci95_low"]>0,"spearman_point":spear["point"] is not None and spear["point"]>=gates["spearman_min"],"spearman_ci":spear["ci95_low"] is not None and spear["ci95_low"]>0}
    return {"checks":checks,"passed":all(checks.values())}


def classify_rd(primary: dict[str,Any], oracle: dict[str,Any], *, valid: bool=True, yes_no_conflict: bool=False) -> dict[str,str]:
    if not valid or yes_no_conflict:
        return {"category":"RD-D","next_action":"stop_invalid_or_conflicted"}
    if primary.get("active_gate_pass") and primary.get("all_safety_pass"):
        return {"category":"RD-A","next_action":"design_bidirectional_bounded_forget_objective_from_preregistered_augmented_direction"}
    if oracle.get("active_gate_pass") and oracle.get("all_safety_pass") and oracle.get("user_leakage") is False:
        return {"category":"RD-B","next_action":"do_not_claim_retrain_free_direction_reassess_whether_retrain_supervised_calibration_is_allowed"}
    return {"category":"RD-C","next_action":"stop_teacher_and_probability_proxy_search_consider_influence_based_or_actual_retraining_baseline"}


def _cache_binding(pre:dict[str,Any],contract_sha:str,fold_hash:str) -> dict[str,Any]:
    return {"schema":CACHE_SCHEMA,"contract_sha256":contract_sha,"git_commit":pre["git"]["git_commit"],"implementation_sha256":pre["implementation"]["canonical_sha256"],"checkpoint_state_sha256":pre["source_checkpoint"]["state_sha256"],"model_sha256":{key:value["sha256"] for key,value in pre["model_lineage"].items()},"validation_sha256":pre["data_lineage"]["data"]["overall_validation"]["sha256"],"sample_order_sha256":pre["sample_order"]["forget"],"user_order_sha256":pre["sample_order"]["user_sidecar"],"selector_sha256":pre["historical_authorities"]["development_selector"]["selector_sha256"],"fold_assignment_sha256":fold_hash,"audit_only_retrain_labels":True,"test_accessed":False}


def publish_cache(path:Path,rows:list[dict[str,Any]],summary:dict[str,Any],binding:dict[str,Any]) -> None:
    if path.exists():raise FileExistsError(f"refusing overwrite: {path}")
    if any(_nested_keys(row)&FORBIDDEN_KEYS for row in rows) or not _all_finite(rows) or not _all_finite(summary):raise ValueError("unsafe/nonfinite cache")
    stage=path.parent/f".{path.name}.{uuid.uuid4().hex[:10]}.stage";stage.mkdir(parents=True)
    try:
        atomic_text(stage/"rows.jsonl","".join(json.dumps(row,sort_keys=True,allow_nan=False)+"\n" for row in rows));atomic_json(stage/"summary.json",summary);atomic_json(stage/"manifest.json",{**binding,"rows":len(rows),"rows_sha256":sha256_file(stage/"rows.jsonl"),"summary_sha256":sha256_file(stage/"summary.json"),"row_order_sha256":canonical_hash([row["sample_order_position"] for row in rows]),"published_atomically":True,"full_vocabulary_logits_persisted":False,"token_tensors_persisted":False,"raw_samples_persisted":False,"oracle_model_or_coefficients_persisted":False,"test_accessed":False});atomic_text(stage/"COMPLETED",CACHE_MARKER+"\n");path.parent.mkdir(parents=True,exist_ok=True);os.replace(stage,path)
    except BaseException:
        if stage.exists():
            for child in stage.iterdir():child.unlink()
            stage.rmdir()
        raise


def validate_cache(path:Path,binding:dict[str,Any]|None=None) -> dict[str,Any]:
    if not path.is_dir() or {item.name for item in path.iterdir()}!={"rows.jsonl","summary.json","manifest.json","COMPLETED"}:raise ValueError("cache inventory invalid")
    manifest=_read_json(path/"manifest.json");summary=_read_json(path/"summary.json");rows=[json.loads(line) for line in (path/"rows.jsonl").read_text(encoding="utf-8").splitlines()]
    if (path/"COMPLETED").read_text(encoding="utf-8")!=CACHE_MARKER+"\n" or (binding is not None and any(manifest.get(key)!=value for key,value in binding.items())) or manifest.get("rows")!=len(rows) or manifest.get("rows_sha256")!=sha256_file(path/"rows.jsonl") or manifest.get("summary_sha256")!=sha256_file(path/"summary.json"):raise ValueError("cache binding/SHA invalid")
    positions=[row.get("sample_order_position") for row in rows]
    if positions!=list(range(len(rows))) or manifest.get("row_order_sha256")!=canonical_hash(positions) or any(_nested_keys(row)&FORBIDDEN_KEYS for row in rows) or not _all_finite(rows) or not _all_finite(summary):raise ValueError("cache row/order/safety invalid")
    users={}
    for row in rows:users.setdefault(row["authoritative_user_id"],set()).add(row["fold_id"])
    if any(len(folds)!=1 for folds in users.values()):raise ValueError("user crosses folds")
    return {"rows":rows,"summary":summary,"manifest":manifest}


def _selector_rows(root:Path,config:dict[str,Any],forget_indices:list[int]) -> dict[int,dict[str,Any]]:
    path=_resolve(root,config["authority"]["development_selector_path"]);rows=[json.loads(line) for line in (path/"rows.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows)!=3336 or [row["source_index"] for row in rows]!=forget_indices or canonical_hash(rows)!=config["authority"]["development_selector_sha256"]:raise ValueError("active selector rows/order/SHA mismatch")
    return {int(row["source_index"]):row for row in rows}


def _stream_primary(root:Path,config:dict[str,Any],pre:dict[str,Any],device:torch.device) -> tuple[list[dict[str,Any]],dict[str,Any]]:
    checkpoint=_resolve(root,config["source_checkpoint"]);before=(sha256_file(checkpoint/"state.pt"),sha256_file(checkpoint/"manifest.json"));base=load_config(_resolve(root,config["base_config"]),root);_,indices,_=_data_lineage(root,base,_resolve(root,config["protocol_root"]));forget_indices=sorted(indices["forget_user_validation"]);retain_indices=set(indices["retain_user_validation"]);user_map=_load_user_map(root,pre);selector=_selector_rows(root,config,forget_indices);runtime,payload=_load_pilot_runtime(root,load_pilot_config(_resolve(root,config["pilot_config"]),root),checkpoint,device);validation=JsonPromptDataset(Path(base["paths"]["validation"]),runtime["tokenizer"]);retrain=freeze_teacher(load_legacy_model(Path(base["paths"]["retrain_reference"]))).to(device);models={"B":runtime["current"],"O":runtime["original"],"A":runtime["augmented"],"R":retrain}
    for model in models.values():model.eval();freeze_teacher(model)
    adapter_before=tensor_tree_hash(get_peft_model_state_dict(runtime["current"]));rows=[];retain_values={"yes":[],"no":[]};batch_size=config["evaluation"]["batch_size"];no_id=config["direction"]["no_token_id"];yes_id=config["direction"]["yes_token_id"]
    with torch.inference_mode():
        for start in range(0,len(validation),batch_size):
            batch_indices=list(range(start,min(start+batch_size,len(validation))));batch=move_batch(_batch(validation,batch_indices),device);observed=batch["target_ids"][:,0];values={}
            for name,model in models.items():
                logits=model(input_ids=batch["input_ids"],labels=batch["target_ids"]).logits[:,0,[no_id,yes_id]].float();prob=torch.softmax(logits,-1);local=(observed==yes_id).long();values[name]={"prob_observed":torch.gather(prob,1,local[:,None]).squeeze(1).cpu(),"margin":signed_margin(logits,observed,no_id,yes_id).cpu(),"confidence":prob.max(-1).values.cpu(),"yes_prob":prob[:,1].cpu(),"argmax":prob.argmax(-1).cpu()};del logits,prob
            for offset,index in enumerate(batch_indices):
                is_yes=bool(observed[offset]==yes_id);b=float(values["B"]["prob_observed"][offset])
                if index in retain_indices:retain_values["yes" if is_yes else "no"].append(b)
                if index not in selector:continue
                o=float(values["O"]["prob_observed"][offset]);a=float(values["A"]["prob_observed"][offset]);r=float(values["R"]["prob_observed"][offset]);mb=float(values["B"]["margin"][offset]);mo=float(values["O"]["margin"][offset]);ma=float(values["A"]["margin"][offset]);mr=float(values["R"]["margin"][offset]);delta=r-b;sel=selector[index]
                row={"sample_order_position":len(rows),"authoritative_user_id":int(user_map[index]),"observed_answer_id":int(observed[offset]),"observed_answer_is_yes":is_yes,"p_B_observed":b,"p_O_observed":o,"p_A_observed":a,"margin_B_signed":mb,"margin_O_signed":mo,"margin_A_signed":ma,"prob_B_minus_O":b-o,"prob_B_minus_A":b-a,"prob_O_minus_A":o-a,"margin_B_minus_O":mb-mo,"margin_B_minus_A":mb-ma,"margin_O_minus_A":mo-ma,"jsd_O_A":bernoulli_jsd(float(values["O"]["yes_prob"][offset]),float(values["A"]["yes_prob"][offset])),"argmax_agreement_O_A":float(values["O"]["argmax"][offset]==values["A"]["argmax"][offset]),"confidence_B":float(values["B"]["confidence"][offset]),"confidence_O":float(values["O"]["confidence"][offset]),"confidence_A":float(values["A"]["confidence"][offset]),"sab_advantage":float(sel["advantage"]),"sab_excess":float(sel["excess"]),"sab_active":bool(sel["active"]),"delta_R_prob":delta,"delta_R_logp":math.log(max(r,1e-30))-math.log(max(b,1e-30)),"delta_R_margin":mr-mb,"signed_yes_no_jsd":math.copysign(bernoulli_jsd(float(values["B"]["yes_prob"][offset]),float(values["R"]["yes_prob"][offset])),delta) if delta!=0 else 0.0,"direction_label":direction_label(delta,.001),"audit_only_retrain_label":True}
                rows.append(row)
            del batch,values
    medians={key:float(np.median(value)) for key,value in retain_values.items()}
    for row in rows:
        row["retain_label_calibration_residual"]=medians["yes" if row["observed_answer_is_yes"] else "no"]-row["p_B_observed"];row["rule_scores"]=fixed_rule_scores(row,medians)
    nonneutral=[row["direction_label"] for row in rows if row["direction_label"]!="EQUIVALENT"];majority="UP" if nonneutral.count("UP")>=nonneutral.count("DOWN") else "DOWN"
    for row in rows:row["rule_scores"]["majority_class"]=1.0 if majority=="UP" else -1.0
    folds=assign_group_folds(rows,5,42)
    if len(rows)!=3336 or sum(row["sab_active"] for row in rows)!=126 or len({row["authoritative_user_id"] for row in rows})!=95:raise RuntimeError("Primary count/selector lineage mismatch")
    if adapter_before!=tensor_tree_hash(get_peft_model_state_dict(runtime["current"])) or (sha256_file(checkpoint/"state.pt"),sha256_file(checkpoint/"manifest.json"))!=before:raise RuntimeError("model/checkpoint changed")
    summary={"scope":"forget_user_validation","samples":len(rows),"users":95,"active_samples":126,"inactive_samples":3210,"active_users":len({row["authoritative_user_id"] for row in rows if row["sab_active"]}),"retain_label_calibration_medians":medians,"majority_non_equivalent_class":majority,"folds":folds,"source_checkpoint_sha256_before_after":list(before),"model_parameters_modified":False,"optimizer_constructed":False,"optimizer_steps_committed":0,"backward_called":False,"candidate_update_generated":False,"step817_checkpoint_published":False,"retrain_used_for_labels_only":True,"retrain_used_for_fixed_rules":False,"retrain_used_for_feature_construction":False,"full_vocabulary_logits_persisted":False,"token_tensors_persisted":False,"raw_samples_persisted":False,"test_accessed":False}
    del retrain,runtime
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return rows,summary


def build_contract(pre:dict[str,Any],run_name:str) -> dict[str,Any]:
    return {"schema":SCHEMA,"run_name":run_name,"config_sha256":pre["config_sha256"],"git":pre["git"],"implementation":pre["implementation"],"historical_authorities":pre["historical_authorities"],"source_checkpoint":pre["source_checkpoint"],"model_lineage":pre["model_lineage"],"tokenizer_lineage":pre["tokenizer_lineage"],"data_lineage":pre["data_lineage"],"sample_order":pre["sample_order"],"primary":pre["primary"],"secondary":pre["secondary"],"direction":pre["direction"],"fixed_rules":pre["fixed_rules"],"features":pre["features"],"oracle":pre["oracle"],"bootstrap":pre["bootstrap"],"gates":pre["gates"],"model_parameters_modified":False,"optimizer_constructed":False,"optimizer_steps_committed":0,"backward_called":False,"candidate_update_generated":False,"step817_checkpoint_published":False,"retrain_used_for_labels_only":True,"retrain_used_for_fixed_rules":False,"retrain_used_for_feature_construction":False,"test_accessed":False}


def validate_resume_status(run_dir:Path) -> dict[str,Any]:
    if not run_dir.is_dir() or (run_dir/"COMPLETED").exists():raise ValueError("Resume requires nonterminal Full")
    state=_read_json(run_dir/"run_state.json")
    if state.get("status")!="INTERRUPTED" or state.get("optimizer_steps_committed")!=0 or state.get("test_accessed") is not False:raise ValueError("Resume requires strict zero-update INTERRUPTED state")
    return state


def _full_contract_compatibility(contract:dict[str,Any],current_contract:dict[str,Any],manifest:dict[str,Any]) -> str:
    if contract==current_contract:
        return "current_head_exact"
    producer=(contract.get("git",{}).get("git_commit"),contract.get("implementation",{}).get("canonical_sha256"))
    comparable=dict(current_contract);comparable["git"]=contract.get("git");comparable["implementation"]=contract.get("implementation")
    if producer not in FULL_PRODUCER_COMPATIBILITY or contract!=comparable or manifest.get("git_commit")!=producer[0] or manifest.get("implementation_sha256")!=producer[1]:
        raise ValueError("Full contract mismatch")
    return "allowlisted_analysis_only_predecessor"


def execute_full(root:Path,config_path:Path,run_name:str,*,resume:bool) -> dict[str,Any]:
    pre=preflight(root,config_path);require_clean_git(pre["git"],"direction Resume" if resume else "direction Full");config=load_audit_config(config_path,root);run_dir=_resolve(root,Path(config["output_root"])/"full_runs"/_safe_name(run_name),output=True)
    if resume:
        validate_resume_status(run_dir)
        if _read_json(run_dir/"contract.json")!=build_contract(pre,run_name):raise ValueError("Resume contract mismatch")
    else:
        if run_dir.exists():raise FileExistsError("refusing overwrite Full")
        run_dir.mkdir(parents=True);atomic_json(run_dir/"contract.json",build_contract(pre,run_name));atomic_json(run_dir/"run_state.json",{"status":"RUNNING","completed_units":[],"optimizer_steps_committed":0,"test_accessed":False})
    lock=run_dir/"RUN.lock"
    try:descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(descriptor)
    except FileExistsError as error:raise RuntimeError("RunName locked") from error
    try:
        contract_sha=sha256_file(run_dir/"contract.json");cache=run_dir/"units"/"primary"
        if cache.exists():
            provisional=validate_cache(cache);binding=_cache_binding(pre,contract_sha,provisional["summary"]["folds"]["fold_assignment_sha256"]);validate_cache(cache,binding);summary=provisional["summary"]
        else:
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu");rows,summary=_stream_primary(root,config,pre,device);binding=_cache_binding(pre,contract_sha,summary["folds"]["fold_assignment_sha256"]);publish_cache(cache,rows,summary,binding)
        state={"status":"COMPLETED","completed_units":["primary"],"model_parameters_modified":False,"optimizer_constructed":False,"optimizer_steps_committed":0,"backward_called":False,"candidate_update_generated":False,"step817_checkpoint_published":False,"retrain_used_for_labels_only":True,"retrain_used_for_fixed_rules":False,"retrain_used_for_feature_construction":False,"test_accessed":False};atomic_json(run_dir/"run_state.json",state);atomic_json(run_dir/"full_manifest.json",{"schema":SCHEMA,"status":"COMPLETED","contract_sha256":contract_sha,"primary_manifest_sha256":sha256_file(cache/"manifest.json"),"run_state_sha256":sha256_file(run_dir/"run_state.json"),"git_commit":pre["git"]["git_commit"],"implementation_sha256":pre["implementation"]["canonical_sha256"],"checkpoint_state_sha256":pre["source_checkpoint"]["state_sha256"],"model_sha256":{key:value["sha256"] for key,value in pre["model_lineage"].items()},"sample_order_sha256":pre["sample_order"],"fold_assignment_sha256":summary["folds"]["fold_assignment_sha256"],"published_atomically":True,"optimizer_steps_committed":0,"test_accessed":False});atomic_text(run_dir/"COMPLETED","RETRAIN_DIRECTION_FULL_COMPLETED\n");return {**state,"run_dir":str(run_dir)}
    except BaseException:
        if run_dir.exists() and not (run_dir/"COMPLETED").exists():
            prior=_read_json(run_dir/"run_state.json");prior.update({"status":"INTERRUPTED","optimizer_steps_committed":0,"test_accessed":False});atomic_json(run_dir/"run_state.json",prior)
        raise
    finally:
        if lock.exists():lock.unlink()


def _verify_full(root:Path,config:dict[str,Any],pre:dict[str,Any],run_name:str) -> dict[str,Any]:
    full=_resolve(root,Path(config["output_root"])/"full_runs"/_safe_name(run_name),output=True);expected={"contract.json","units","run_state.json","full_manifest.json","COMPLETED"}
    if not full.is_dir() or {item.name for item in full.iterdir()}!=expected or (full/"COMPLETED").read_text(encoding="utf-8")!="RETRAIN_DIRECTION_FULL_COMPLETED\n":raise ValueError("Analyze refuses incomplete Full")
    contract=_read_json(full/"contract.json");manifest=_read_json(full/"full_manifest.json");current_contract=build_contract(pre,run_name)
    producer_compatibility=_full_contract_compatibility(contract,current_contract,manifest)
    state=_read_json(full/"run_state.json")
    if state.get("status")!="COMPLETED" or state.get("optimizer_steps_committed")!=0 or state.get("test_accessed") is not False or manifest.get("contract_sha256")!=sha256_file(full/"contract.json") or manifest.get("run_state_sha256")!=sha256_file(full/"run_state.json") or manifest.get("primary_manifest_sha256")!=sha256_file(full/"units/primary/manifest.json") or manifest.get("test_accessed") is not False:raise ValueError("Full manifest/state invalid")
    cache=validate_cache(full/"units/primary");binding=_cache_binding(pre,sha256_file(full/"contract.json"),cache["summary"]["folds"]["fold_assignment_sha256"]);binding["git_commit"]=contract["git"]["git_commit"];binding["implementation_sha256"]=contract["implementation"]["canonical_sha256"];validate_cache(full/"units/primary",binding)
    return {"path":full,"cache":cache,"manifest":manifest,"producer_compatibility":producer_compatibility,"producer_git_commit":contract["git"]["git_commit"],"producer_implementation_sha256":contract["implementation"]["canonical_sha256"]}


def _safety(evaluation:dict[str,Any],gates:dict[str,Any]) -> bool:
    bal=evaluation["bootstrap"]["metrics"]["binary_balanced_accuracy"]["ci95_high"];spear=evaluation["bootstrap"]["metrics"]["spearman"]["ci95_high"]
    return bal is not None and bal>=gates["all_binary_balanced_accuracy_ci_upper_min"] and spear is not None and spear>=gates["all_spearman_ci_upper_min"]


def _reliably_reversed(evaluation:dict[str,Any],gates:dict[str,Any]) -> bool:
    """Reject only when the subgroup CI excludes a preregistered safety floor."""
    bal=evaluation["bootstrap"]["metrics"]["binary_balanced_accuracy"]["ci95_high"]
    spear=evaluation["bootstrap"]["metrics"]["spearman"]["ci95_high"]
    return ((bal is not None and bal<gates["all_binary_balanced_accuracy_ci_upper_min"])
            or (spear is not None and spear<gates["all_spearman_ci_upper_min"]))


def analyze(root:Path,config_path:Path,run_name:str) -> dict[str,Any]:
    pre=preflight(root,config_path);require_clean_git(pre["git"],"direction Analyze");config=load_audit_config(config_path,root);verified=_verify_full(root,config,pre,run_name);final=_resolve(root,Path(config["output_root"])/"analysis_runs"/_safe_name(run_name),output=True)
    if final.exists():raise FileExistsError("refusing overwrite Analyze")
    rows=verified["cache"]["rows"];rule_results={rule:{group:evaluate_rule(_subgroup(rows,group),rule,resamples=config["bootstrap"]["resamples"],seed=42) for group in SUBGROUPS} for rule in RULES};oracle=run_oracle(rows,config["features"]["names"],config,resamples=config["bootstrap"]["resamples"]);primary_active_gate=_gate_from_evaluation(rule_results["augmented_to_step816"]["active"],config["gates"]);primary_all_safe=_safety(rule_results["augmented_to_step816"]["all"],config["gates"]);oracle_active_gate=_gate_from_evaluation(oracle["evaluations"]["active"],config["gates"]);oracle_all_safe=_safety(oracle["evaluations"]["all"],config["gates"])
    yes_no_conflict=_reliably_reversed(rule_results["augmented_to_step816"]["observed_yes"],config["gates"]) or _reliably_reversed(rule_results["augmented_to_step816"]["observed_no"],config["gates"]);primary={"active_gate":primary_active_gate,"active_gate_pass":primary_active_gate["passed"],"all_safety_pass":primary_all_safe};oracle_gate={"active_gate":oracle_active_gate,"active_gate_pass":oracle_active_gate["passed"],"all_safety_pass":oracle_all_safe,"user_leakage":oracle["user_leakage"]};decision=classify_rd(primary,oracle_gate,valid=True,yes_no_conflict=yes_no_conflict)
    result={"schema":ANALYSIS_SCHEMA,"run_name":run_name,**decision,"truth_definition":config["direction"],"primary_fixed_rule":"augmented_to_step816","primary":primary,"diagnostic_rules_cannot_authorize_RD_A":True,"rules":rule_results,"oracle":oracle,"oracle_is_audit_only_information_upper_bound":True,"direction_distribution":direction_distribution(rows),"yes_no_reliable_conflict":yes_no_conflict,"secondary_scope":"disabled_resource_bounded_primary_only","source_full_producer_compatibility":verified["producer_compatibility"],"source_full_git_commit":verified["producer_git_commit"],"source_full_implementation_sha256":verified["producer_implementation_sha256"],"model_parameters_modified":False,"optimizer_constructed":False,"optimizer_steps_committed":0,"backward_called":False,"candidate_update_generated":False,"step817_checkpoint_published":False,"retrain_used_for_labels_only":True,"retrain_used_for_fixed_rules":False,"retrain_used_for_feature_construction":False,"test_accessed":False}
    stage=final.parent/f".{run_name}.{uuid.uuid4().hex[:10]}.stage";stage.mkdir(parents=True);atomic_json(stage/"analysis.json",result);atomic_json(stage/"manifest.json",{"schema":ANALYSIS_SCHEMA,"analysis_sha256":sha256_file(stage/"analysis.json"),"source_full_manifest_sha256":sha256_file(verified["path"]/"full_manifest.json"),"git_commit":pre["git"]["git_commit"],"implementation_sha256":pre["implementation"]["canonical_sha256"],"published_atomically":True,"test_accessed":False});atomic_text(stage/"COMPLETED","RETRAIN_DIRECTION_ANALYSIS_COMPLETED\n");final.parent.mkdir(parents=True,exist_ok=True);os.replace(stage,final);return result


def synthetic_dry_run(root:Path,config_path:Path,run_name:str) -> dict[str,Any]:
    config=load_audit_config(config_path,root);destination=_resolve(root,Path(config["output_root"])/"synthetic_runs"/_safe_name(run_name),output=True)
    if destination.exists():raise FileExistsError("refusing overwrite SyntheticDryRun")
    rows=[]
    for user in range(40):
        for offset in range(6):
            direction=CLASSES[(user+offset)%3];delta={"DOWN":-.08,"EQUIVALENT":0.0,"UP":.08}[direction];b=.45+.02*(offset%3);a=b+delta;o=b+.01*((user%3)-1);is_yes=bool((user+offset)%2);row={"sample_order_position":len(rows),"authoritative_user_id":user,"observed_answer_id":2163 if is_yes else 465,"observed_answer_is_yes":is_yes,"p_B_observed":b,"p_O_observed":o,"p_A_observed":a,"margin_B_signed":0.1,"margin_O_signed":.1+(o-b),"margin_A_signed":.1+(a-b),"prob_B_minus_O":b-o,"prob_B_minus_A":b-a,"prob_O_minus_A":o-a,"margin_B_minus_O":b-o,"margin_B_minus_A":b-a,"margin_O_minus_A":o-a,"jsd_O_A":bernoulli_jsd(o,a),"argmax_agreement_O_A":1.0,"confidence_B":max(b,1-b),"confidence_O":max(o,1-o),"confidence_A":max(a,1-a),"sab_advantage":max(0,o-a),"sab_excess":max(0,o-a-.02),"sab_active":offset<3,"retain_label_calibration_residual":.5-b,"delta_R_prob":delta,"delta_R_logp":math.log(b+delta)-math.log(b),"delta_R_margin":2*delta,"signed_yes_no_jsd":math.copysign(bernoulli_jsd(b,b+delta),delta) if delta else 0.0,"direction_label":direction,"audit_only_retrain_label":True};row["rule_scores"]=fixed_rule_scores(row,{"yes":.5,"no":.5});row["rule_scores"]["majority_class"]=-1.0;rows.append(row)
    folds=assign_group_folds(rows,5,42);summary={"scope":"synthetic","samples":len(rows),"users":40,"folds":folds,"model_parameters_modified":False,"optimizer_constructed":False,"optimizer_steps_committed":0,"backward_called":False,"candidate_update_generated":False,"step817_checkpoint_published":False,"test_accessed":False};destination.mkdir(parents=True);atomic_json(destination/"contract.json",{"schema":SCHEMA,"mode":"SyntheticDryRun","run_name":run_name,"test_accessed":False});binding={"schema":CACHE_SCHEMA,"contract_sha256":sha256_file(destination/"contract.json"),"fold_assignment_sha256":folds["fold_assignment_sha256"],"synthetic":True,"test_accessed":False};publish_cache(destination/"primary",rows,summary,binding);validate_cache(destination/"primary",binding);primary=evaluate_rule(rows,"augmented_to_step816",resamples=100,seed=42);oracle=run_oracle(rows,config["features"]["names"],config,resamples=100);examples={"RD-A":classify_rd({"active_gate_pass":True,"all_safety_pass":True},{},valid=True),"RD-B":classify_rd({"active_gate_pass":False,"all_safety_pass":True},{"active_gate_pass":True,"all_safety_pass":True,"user_leakage":False},valid=True),"RD-C":classify_rd({"active_gate_pass":False},{"active_gate_pass":False},valid=True),"RD-D":classify_rd({}, {},valid=False)};result={"schema":SCHEMA,"mode":"SyntheticDryRun","primary":primary,"oracle":{"folds":oracle["folds"],"user_leakage":oracle["user_leakage"],"model_or_coefficients_persisted":False},"classification_examples":examples,"model_parameters_modified":False,"optimizer_constructed":False,"optimizer_steps_committed":0,"backward_called":False,"candidate_update_generated":False,"step817_checkpoint_published":False,"full_vocabulary_logits_persisted":False,"token_tensors_persisted":False,"raw_samples_persisted":False,"test_accessed":False};atomic_json(destination/"synthetic_result.json",result);atomic_text(destination/"COMPLETED","RETRAIN_DIRECTION_SYNTHETIC_COMPLETED\n");return {**result,"run_dir":str(destination)}


def main() -> None:
    parser=argparse.ArgumentParser(description="T5 Retrain-direction separability audit");parser.add_argument("--mode",choices=("Preflight","SyntheticDryRun","Full","Resume","Analyze"),default="Preflight");parser.add_argument("--config",type=Path,required=True);parser.add_argument("--project-root",type=Path,default=Path.cwd());parser.add_argument("--run-name");args=parser.parse_args();root=args.project_root.resolve();config=args.config.resolve()
    if args.mode=="Preflight":result=preflight(root,config)
    else:
        if not args.run_name:parser.error(f"{args.mode} requires --run-name")
        if args.mode=="SyntheticDryRun":result=synthetic_dry_run(root,config,args.run_name)
        elif args.mode=="Full":result=execute_full(root,config,args.run_name,resume=False)
        elif args.mode=="Resume":result=execute_full(root,config,args.run_name,resume=True)
        else:result=analyze(root,config,args.run_name)
    print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))


if __name__=="__main__":main()
