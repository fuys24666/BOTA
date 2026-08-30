from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import yaml
from peft import get_peft_model_state_dict
from transformers import T5Tokenizer

from src.diagnostics.git_provenance import (
    git_provenance,
    implementation_provenance,
    require_clean_git,
)
from src.diagnostics.t5_full_runner import _batch
from src.diagnostics.t5_projected_pilot_10step import (
    SOURCE_STATE_SHA,
    _load_pilot_runtime,
    batch_plan,
    load_pilot_config,
    next_batch_hash,
    shadow_adamw_proposal_for_step,
    validate_checkpoint_chain,
)
from src.diagnostics.t5_reconstructed_official import (
    JsonPromptDataset,
    compute_components,
    forced_logits,
    freeze_teacher,
    load_config,
    load_legacy_model,
    move_batch,
    sha256_file,
)
from src.diagnostics.t5_step812_gradient_geometry import catalog_pair
from src.diagnostics.t5_step813_optimizer_aware_audit import (
    _flatten_gradients,
    _runtime_binding,
    _tensor_hash,
    direct_svd_projection,
    directional_metrics,
    json_native,
)
from src.diagnostics.t5_step813_update_space_stage_b import (
    _rng_state_from_payload,
    materialize_dtype_delta,
    update_space_projection,
)
from src.diagnostics.t5_trajectory_diagnostics import (
    capture_rng,
    restore_rng,
    rng_hashes,
)
from src.diagnostics.t5_zero_training_audit import _data_lineage


SCHEMA = "t5-step817-forget-conflict-audit-v1"
UNIT_SCHEMA = "t5-step817-forget-conflict-unit-v1"
ANALYSIS_SCHEMA = "t5-step817-forget-conflict-analysis-v1"
SCALES = (1.0, 0.5)
EXPECTED_DELTA_HASHES = {
    1.0: "9685151f57b2e5ad3e544749705df0b1dfb075f7c6cb104da178658b917a1451",
    0.5: "460bdbf8f0ba13285f78c58e3e6e700c8405bcd5761703833e0f37ac186f4f1a",
    0.25: "dd306469365017f702c765acd39931554b2e2ae71d57ebbf23fce03732fe5e85",
}
IMPLEMENTATION_FILES = (
    "src/diagnostics/t5_step817_forget_conflict_audit.py",
    "configs/t5_step817_forget_conflict_audit_v1.yaml",
    "scripts/diagnostics/t5_step817_forget_conflict_audit_v1.ps1",
    "docs/t5_step817_forget_conflict_audit_v1.md",
    "src/diagnostics/t5_projected_pilot_10step.py",
    "configs/t5_projected_pilot_10step_v1.yaml",
    "src/diagnostics/t5_step813_update_space_stage_b.py",
    "src/diagnostics/t5_step813_optimizer_aware_audit.py",
    "src/diagnostics/t5_step812_gradient_geometry.py",
    "src/diagnostics/t5_reconstructed_official.py",
    "src/diagnostics/t5_zero_training_audit.py",
    "src/diagnostics/git_provenance.py",
)
FORBIDDEN_CACHE_KEYS = {
    "logits", "full_vocabulary_logits", "gradient", "gradients", "delta",
    "input_ids", "target_ids", "tokens", "raw_sample", "raw_samples",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_hash(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(json_native(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )


def tensor_tree_hash(value: Any) -> str:
    def normalize(item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            return {"dtype": str(item.dtype), "shape": list(item.shape), "sha256": _tensor_hash(item)}
        if isinstance(item, dict):
            return {str(key): normalize(child) for key, child in sorted(item.items(), key=lambda pair: str(pair[0]))}
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        return item
    return canonical_hash(normalize(value))


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def directory_hash(path: Path) -> dict[str, Any]:
    rows = []
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        rows.append([item.relative_to(path).as_posix(), sha256_file(item), item.stat().st_size])
    return {"canonical_sha256": canonical_hash(rows), "files": rows}


def load_audit_config(path: Path, root: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("development_only") is not True:
        raise ValueError("conflict audit config schema/scope mismatch")
    if value.get("test_access_policy") != "forbidden":
        raise ValueError("test access policy must remain forbidden")
    if value.get("bootstrap") != {"cluster": "authoritative_user_id", "seed": 42, "resamples": 2000}:
        raise ValueError("authoritative clustered bootstrap changed")
    if value.get("evaluation", {}).get("scales") != [1.0, 0.5] or value["evaluation"].get("diagnostic_scale") != 0.25:
        raise ValueError("frozen counterfactual scales changed")
    if value.get("minimum_effect", {}).get("derivation") != "ten_times_float32_unit_roundoff_with_cluster_ci":
        raise ValueError("minimum-effect authority changed")
    value["_path"] = str(path.resolve())
    value["_sha256"] = sha256_file(path)
    return value


def _validate_pilot_authority(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    full = _resolve(root, config["pilot_full"])
    analysis = _resolve(root, config["pilot_analysis"])
    checkpoint = _resolve(root, config["source_checkpoint"])
    authority = config["authority"]
    required_full = {"contract.json", "run_manifest.json", "run_state.json", "STOPPED_SAFELY", "checkpoints"}
    required_analysis = {"analysis.json", "manifest.json", "COMPLETED"}
    if not full.is_dir() or {item.name for item in full.iterdir()} != required_full:
        raise ValueError("Pilot Full inventory mismatch")
    if not analysis.is_dir() or {item.name for item in analysis.iterdir()} != required_analysis:
        raise ValueError("Pilot Analyze inventory mismatch")
    source_files = {
        "pilot_contract_sha256": full / "contract.json",
        "pilot_run_manifest_sha256": full / "run_manifest.json",
        "pilot_run_state_sha256": full / "run_state.json",
        "pilot_analysis_sha256": analysis / "analysis.json",
        "step816_state_sha256": checkpoint / "state.pt",
        "step816_manifest_sha256": checkpoint / "manifest.json",
    }
    for key, source in source_files.items():
        if sha256_file(source) != authority[key]:
            raise ValueError(f"authority SHA mismatch: {key}")
    state = _read_json(full / "run_state.json")
    decision = _read_json(analysis / "analysis.json")
    manifest = _read_json(checkpoint / "manifest.json")
    if state.get("status") != "STOPPED_SAFELY" or state.get("last_step") != 816:
        raise ValueError("Pilot must be safely stopped at step816")
    if state.get("accepted_scales") != [1.0, 1.0, 0.5]:
        raise ValueError("Pilot accepted-scale lineage mismatch")
    if decision.get("category") != "P10-C" or decision.get("final_step") != 816:
        raise ValueError("Pilot Analyze must be P10-C at step816")
    if decision.get("accepted_scales") != [1.0, 1.0, 0.5] or decision.get("test_accessed") is not False:
        raise ValueError("Pilot Analyze scale/test contract mismatch")
    chain = validate_checkpoint_chain(full, SOURCE_STATE_SHA, expected_last=816)
    if chain["last_state_sha256"] != authority["step816_state_sha256"]:
        raise ValueError("Pilot checkpoint chain does not end at frozen step816 state")
    if (
        manifest.get("step") != 816
        or manifest.get("state_sha256") != authority["step816_state_sha256"]
        or manifest.get("parent_state_sha256") != authority["step816_parent_state_sha256"]
        or manifest.get("next_optimizer_step") != 817
        or manifest.get("next_batch_hash") != authority["step817_batch_hash"]
        or manifest.get("optimizer_counter") != 816
        or manifest.get("test_accessed") is not False
    ):
        raise ValueError("step816 manifest continuation mismatch")
    payload = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=False)
    counters = [float(item["step"]) for item in payload.get("optimizer_state", {}).get("state", {}).values()]
    if (
        payload.get("state", {}).get("step") != 816
        or payload.get("state", {}).get("next_optimizer_step") != 817
        or payload.get("state", {}).get("next_batch_hash") != authority["step817_batch_hash"]
        or payload.get("state", {}).get("accepted_scales") != [1.0, 1.0, 0.5]
        or not counters
        or any(value != 816.0 for value in counters)
        or payload.get("rng_hash") != rng_hashes(_rng_state_from_payload(payload["rng"]))
        or len(payload.get("adapter_state", {})) != 144
        or sum(value.numel() for value in payload["adapter_state"].values()) != 1_769_472
        or payload.get("test_accessed") is not False
    ):
        raise ValueError("step816 serialized state mismatch")
    trials = state.get("scale_trials", [])
    expected = [(1.0, authority["scale_1_actual_delta_hash"]), (0.5, authority["scale_0_5_actual_delta_hash"]), (0.25, authority["scale_0_25_actual_delta_hash"])]
    if len(trials) != 3:
        raise ValueError("step817 trial inventory mismatch")
    for trial, (scale, digest) in zip(trials, expected):
        if trial.get("scale") != scale or trial.get("evidence", {}).get("directional", {}).get("actual_delta_hash") != digest:
            raise ValueError("step817 recorded candidate hash mismatch")
    if config["authority"]["step817_batch_hash"] != batch_plan()["817"]["batch_hash"]:
        raise ValueError("step817 batch differs from frozen Pilot plan")
    return json_native({
        "full": str(full), "analysis": str(analysis), "checkpoint": str(checkpoint),
        "category": "P10-C", "final_step": 816, "accepted_scales": [1.0, 1.0, 0.5],
        "state_sha256": authority["step816_state_sha256"],
        "manifest_sha256": authority["step816_manifest_sha256"],
        "parent_state_sha256": authority["step816_parent_state_sha256"],
        "optimizer_counters": sorted(set(counters)), "rng_hash": payload["rng_hash"],
        "step817_batch_hash": authority["step817_batch_hash"],
        "candidate_hashes": {str(key): value for key, value in EXPECTED_DELTA_HASHES.items()},
        "test_accessed": False,
    })


def preflight(
    root: Path,
    config_path: Path,
    *,
    git_function: Callable[[Path], dict[str, Any]] = git_provenance,
    implementation_function: Callable[[Path, Iterable[str]], dict[str, Any]] = implementation_provenance,
) -> dict[str, Any]:
    config = load_audit_config(config_path, root)
    source = _validate_pilot_authority(root, config)
    base = load_config(_resolve(root, config["base_config"]), root)
    lineage, indices, _ = _data_lineage(root, base, _resolve(root, config["protocol_root"]))
    paths = base["paths"]
    model_lineage = {
        "original": {"path": str(Path(paths["original"]).resolve()), "sha256": sha256_file(Path(paths["original"]))},
        "augmented": {"path": str(Path(paths["augmented_teacher"]).resolve()), "sha256": sha256_file(Path(paths["augmented_teacher"]))},
        "retrain": {"path": str(Path(paths["retrain_reference"]).resolve()), "sha256": sha256_file(Path(paths["retrain_reference"]))},
    }
    tokenizer = directory_hash(Path(paths["model_dir"]))
    expected = config["lineage_sha256"]
    for name in ("original", "augmented", "retrain"):
        if model_lineage[name]["sha256"] != expected[name]:
            raise ValueError(f"{name} checkpoint lineage changed")
    if tokenizer["canonical_sha256"] != expected["tokenizer_directory"]:
        raise ValueError("tokenizer/model directory lineage changed")
    data_expected = {
        "forget_train": lineage["data"]["forget_train"]["sha256"],
        "retain_train": lineage["data"]["retain_train"]["sha256"],
        "validation": lineage["data"]["overall_validation"]["sha256"],
        "validation_user_sidecar": lineage["validation_sidecar"]["sha256"],
    }
    if data_expected != {key: expected[key] for key in data_expected}:
        raise ValueError("development data lineage changed")
    if len(indices["forget_user_validation"]) != config["evaluation"]["samples"]:
        raise ValueError("Forget development sample count changed")
    return json_native({
        "schema": SCHEMA,
        "mode": "Preflight",
        "development_only": True,
        "config_sha256": config["_sha256"],
        "git": git_function(root),
        "implementation": implementation_function(root, IMPLEMENTATION_FILES),
        "pilot_authority": source,
        "model_lineage": model_lineage,
        "tokenizer_lineage": tokenizer,
        "data_lineage": lineage,
        "forget_validation": {
            "samples": len(indices["forget_user_validation"]),
            "users": config["evaluation"]["users"],
            "indices_sha256": lineage["validation_splits"]["forget_user_validation"]["indices_sha256"],
        },
        "candidate_reconstruction": {
            "step": 817,
            "scales": list(SCALES),
            "diagnostic_scale": 0.25,
            "expected_actual_delta_hashes": {str(key): value for key, value in EXPECTED_DELTA_HASHES.items()},
            "official_optimizer_step_calls": 0,
            "step817_checkpoint_published": False,
        },
        "spaces": ["full_valid_vocabulary", "yes_no_conditional"],
        "minimum_effect": config["minimum_effect"],
        "bootstrap": config["bootstrap"],
        "resource_estimate": config["resource_estimate"],
        "retrain_role": "posthoc_audit_only_not_selection",
        "model_loaded": False,
        "cuda_used": False,
        "test_loader_built": False,
        "test_accessed": False,
    })


def _all_finite(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return False


def _nested_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        result = set(value)
        for item in value.values():
            result.update(_nested_keys(item))
        return result
    if isinstance(value, list):
        result: set[str] = set()
        for item in value:
            result.update(_nested_keys(item))
        return result
    return set()


def _unit_name(scale: float) -> str:
    return f"scale_{str(scale).replace('.', '_')}__forget_development"


def _unit_binding(contract_sha: str, pre: dict[str, Any], scale: float) -> dict[str, Any]:
    return {
        "schema": UNIT_SCHEMA,
        "scale": scale,
        "split": "forget_user_validation",
        "contract_sha256": contract_sha,
        "git_commit": pre["git"]["git_commit"],
        "implementation_sha256": pre["implementation"]["canonical_sha256"],
        "checkpoint_state_sha256": pre["pilot_authority"]["state_sha256"],
        "checkpoint_manifest_sha256": pre["pilot_authority"]["manifest_sha256"],
        "checkpoint_rng_hash": pre["pilot_authority"]["rng_hash"],
        "step817_batch_hash": pre["pilot_authority"]["step817_batch_hash"],
        "validation_sha256": pre["data_lineage"]["data"]["overall_validation"]["sha256"],
        "sample_order_sha256": pre["forget_validation"]["indices_sha256"],
        "user_sidecar_sha256": pre["data_lineage"]["validation_sidecar"]["sha256"],
        "expected_candidate_hash": EXPECTED_DELTA_HASHES[scale],
        "test_accessed": False,
    }


def publish_unit(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any], binding: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite audit unit: {path}")
    if not rows or any(_nested_keys(row) & FORBIDDEN_CACHE_KEYS for row in rows):
        raise ValueError("unit rows are empty or contain forbidden tensors/samples")
    if not _all_finite(rows) or not _all_finite(summary):
        raise FloatingPointError("non-finite unit evidence")
    stage = path.parent / f".{path.name}.{uuid.uuid4().hex[:10]}.stage"
    stage.mkdir(parents=True)
    try:
        row_text = "".join(json.dumps(json_native(row), sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
        atomic_text(stage / "rows.jsonl", row_text)
        atomic_json(stage / "summary.json", summary)
        manifest = {
            **binding,
            "rows": len(rows),
            "row_ids_sha256": canonical_hash([row["sample_id"] for row in rows]),
            "user_order_sha256": canonical_hash([row["user_id"] for row in rows]),
            "rows_sha256": sha256_file(stage / "rows.jsonl"),
            "summary_sha256": sha256_file(stage / "summary.json"),
            "published_atomically": True,
            "full_vocabulary_logits_persisted": False,
            "gradient_vectors_persisted": False,
            "delta_vectors_persisted": False,
            "token_tensors_persisted": False,
            "raw_samples_persisted": False,
            "test_accessed": False,
        }
        atomic_json(stage / "manifest.json", manifest)
        atomic_text(stage / "COMPLETED", "FORGET_CONFLICT_UNIT_COMPLETED\n")
        path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, path)
    except BaseException:
        if stage.exists():
            for child in stage.iterdir():
                child.unlink()
            stage.rmdir()
        raise


def validate_unit(path: Path, binding: dict[str, Any], expected_ids: list[int] | None = None) -> dict[str, Any]:
    required = {"rows.jsonl", "summary.json", "manifest.json", "COMPLETED"}
    if not path.is_dir() or {item.name for item in path.iterdir()} != required:
        raise ValueError("unit cache incomplete or contains extra artifacts")
    if (path / "COMPLETED").read_text(encoding="utf-8") != "FORGET_CONFLICT_UNIT_COMPLETED\n":
        raise ValueError("unit completion marker mismatch")
    manifest = _read_json(path / "manifest.json")
    if any(manifest.get(key) != value for key, value in binding.items()):
        raise ValueError("unit authority binding mismatch")
    if manifest.get("rows_sha256") != sha256_file(path / "rows.jsonl") or manifest.get("summary_sha256") != sha256_file(path / "summary.json"):
        raise ValueError("unit cache SHA mismatch")
    rows = [json.loads(line) for line in (path / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    ids = [row.get("sample_id") for row in rows]
    if len(ids) != len(set(ids)) or ids != sorted(ids):
        raise ValueError("unit rows duplicate or out of order")
    if expected_ids is not None and ids != expected_ids:
        raise ValueError("unit sample order differs from authority")
    if manifest.get("rows") != len(rows) or manifest.get("row_ids_sha256") != canonical_hash(ids):
        raise ValueError("unit row count/order hash mismatch")
    if manifest.get("user_order_sha256") != canonical_hash([row.get("user_id") for row in rows]):
        raise ValueError("unit user-order hash mismatch")
    summary = _read_json(path / "summary.json")
    if not _all_finite(rows) or not _all_finite(summary) or any(_nested_keys(row) & FORBIDDEN_CACHE_KEYS for row in rows):
        raise ValueError("unit contains non-finite or forbidden evidence")
    safety = {
        "published_atomically": True,
        "full_vocabulary_logits_persisted": False,
        "gradient_vectors_persisted": False,
        "delta_vectors_persisted": False,
        "token_tensors_persisted": False,
        "raw_samples_persisted": False,
        "test_accessed": False,
    }
    if any(manifest.get(key) != value for key, value in safety.items()):
        raise ValueError("unit safety contract mismatch")
    return {"rows": rows, "summary": summary, "manifest": manifest}


def _statistic(rows: list[dict[str, Any]], space: str) -> dict[str, float]:
    def mean(key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows]))

    before_retrain_l2 = math.sqrt(max(0.0, mean(f"{space}_retrain_before_l2_mse")))
    candidate_retrain_l2 = math.sqrt(max(0.0, mean(f"{space}_retrain_candidate_l2_mse")))
    before_original_l2 = math.sqrt(max(0.0, mean(f"{space}_original_before_l2_mse")))
    candidate_original_l2 = math.sqrt(max(0.0, mean(f"{space}_original_candidate_l2_mse")))
    return {
        "retrain_l2_rms_before": before_retrain_l2,
        "retrain_l2_rms_candidate": candidate_retrain_l2,
        "retrain_l2_rms_improvement": before_retrain_l2 - candidate_retrain_l2,
        "retrain_jsd_before": mean(f"{space}_retrain_before_jsd"),
        "retrain_jsd_candidate": mean(f"{space}_retrain_candidate_jsd"),
        "retrain_jsd_improvement": mean(f"{space}_retrain_before_jsd") - mean(f"{space}_retrain_candidate_jsd"),
        "retrain_kl_before": mean(f"{space}_retrain_before_kl"),
        "retrain_kl_candidate": mean(f"{space}_retrain_candidate_kl"),
        "retrain_kl_improvement": mean(f"{space}_retrain_before_kl") - mean(f"{space}_retrain_candidate_kl"),
        "retrain_answer_loss_distance_before": mean("retrain_before_answer_loss_distance"),
        "retrain_answer_loss_distance_candidate": mean("retrain_candidate_answer_loss_distance"),
        "retrain_answer_loss_distance_improvement": mean("retrain_before_answer_loss_distance") - mean("retrain_candidate_answer_loss_distance"),
        "retrain_argmax_agreement_before": mean(f"{space}_retrain_before_argmax_agreement"),
        "retrain_argmax_agreement_candidate": mean(f"{space}_retrain_candidate_argmax_agreement"),
        "retrain_argmax_agreement_change": mean(f"{space}_retrain_candidate_argmax_agreement") - mean(f"{space}_retrain_before_argmax_agreement"),
        "retrain_margin_distance_before": mean("retrain_before_margin_distance"),
        "retrain_margin_distance_candidate": mean("retrain_candidate_margin_distance"),
        "retrain_margin_distance_improvement": mean("retrain_before_margin_distance") - mean("retrain_candidate_margin_distance"),
        "original_l2_rms_before": before_original_l2,
        "original_l2_rms_candidate": candidate_original_l2,
        "original_l2_rms_divergence": candidate_original_l2 - before_original_l2,
        "original_jsd_before": mean(f"{space}_original_before_jsd"),
        "original_jsd_candidate": mean(f"{space}_original_candidate_jsd"),
        "original_jsd_divergence": mean(f"{space}_original_candidate_jsd") - mean(f"{space}_original_before_jsd"),
        "original_kl_before": mean(f"{space}_original_before_kl"),
        "original_kl_candidate": mean(f"{space}_original_candidate_kl"),
        "original_kl_divergence": mean(f"{space}_original_candidate_kl") - mean(f"{space}_original_before_kl"),
        "original_argmax_agreement_before": mean(f"{space}_original_before_argmax_agreement"),
        "original_argmax_agreement_candidate": mean(f"{space}_original_candidate_argmax_agreement"),
        "original_argmax_agreement_change": mean(f"{space}_original_candidate_argmax_agreement") - mean(f"{space}_original_before_argmax_agreement"),
        "forced_teacher_ce_before": mean("forced_teacher_ce_before"),
        "forced_teacher_ce_candidate": mean("forced_teacher_ce_candidate"),
        "forced_teacher_ce_improvement": mean("forced_teacher_ce_before") - mean("forced_teacher_ce_candidate"),
    }


def clustered_bootstrap(
    rows: list[dict[str, Any]],
    space: str,
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    users = sorted({int(row["user_id"]) for row in rows})
    by_user = {user: [row for row in rows if int(row["user_id"]) == user] for user in users}
    point = _statistic(rows, space)
    draws = {key: [] for key in point}
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        selected = rng.choice(users, size=len(users), replace=True)
        sampled = [row for user in selected for row in by_user[int(user)]]
        value = _statistic(sampled, space)
        for key in draws:
            draws[key].append(value[key])
    intervals = {
        key: {
            "point": point[key],
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }
        for key, values in draws.items()
    }
    return {
        "space": space,
        "cluster": "authoritative_user_id",
        "seed": seed,
        "resamples": resamples,
        "users": len(users),
        "samples": len(rows),
        "metrics": intervals,
    }


def classify_scale(
    full: dict[str, Any],
    yes_no: dict[str, Any],
    minimum: dict[str, float],
    *,
    collapse: bool,
    main_batch_forget_change: float,
    global_forced_change: float,
    valid: bool = True,
) -> dict[str, str]:
    if not valid or collapse:
        return {"category": "FC-D", "next_action": "stop_invalid_or_conflicted"}
    full_r = full["metrics"]["retrain_jsd_improvement"]
    yes_r = yes_no["metrics"]["retrain_jsd_improvement"]
    epsilon = float(minimum["distribution_distance"])
    full_direction = 1 if full_r["ci95_low"] > epsilon else (-1 if full_r["ci95_high"] < -epsilon else 0)
    yes_direction = 1 if yes_r["ci95_low"] > epsilon else (-1 if yes_r["ci95_high"] < -epsilon else 0)
    if full_direction * yes_direction < 0:
        return {"category": "FC-D", "next_action": "stop_invalid_or_conflicted"}
    forced = full["metrics"]["forced_teacher_ce_improvement"]
    forced_not_worse = forced["ci95_low"] >= -float(minimum["forced_teacher_ce"])
    forced_improved = forced["ci95_low"] > float(minimum["forced_teacher_ce"])
    original = full["metrics"]["original_jsd_divergence"]
    toward_original = original["ci95_high"] < -epsilon
    batch_global_conflict = (
        abs(main_batch_forget_change) > float(minimum["forced_teacher_ce"])
        and abs(global_forced_change) > float(minimum["forced_teacher_ce"])
        and math.copysign(1.0, main_batch_forget_change) != math.copysign(1.0, global_forced_change)
    )
    if full_direction > 0 and yes_direction > 0 and forced_not_worse:
        return {
            "category": "FC-A",
            "next_action": "replace_strict_original_divergence_gate_with_preregistered_retrain_direction_or_equivalence_audit",
        }
    no_meaningful_retrain_gain = full_r["ci95_high"] <= epsilon and yes_r["ci95_high"] <= epsilon
    if forced_improved and no_meaningful_retrain_gain and (toward_original or batch_global_conflict):
        return {
            "category": "FC-B",
            "next_action": "localize_or_recalibrate_forced_teacher_before_more_updates",
        }
    return {
        "category": "FC-C",
        "next_action": "do_not_continue_training_design_larger_or_more_sensitive_frozen_audit",
    }


def classify_audit(scale_results: dict[str, dict[str, Any]]) -> dict[str, str]:
    categories = [scale_results[str(scale)]["classification"]["category"] for scale in SCALES]
    if "FC-D" in categories:
        return {"category": "FC-D", "next_action": "stop_invalid_or_conflicted"}
    if categories == ["FC-A", "FC-A"]:
        return {
            "category": "FC-A",
            "next_action": "replace_strict_original_divergence_gate_with_preregistered_retrain_direction_or_equivalence_audit",
        }
    if categories == ["FC-B", "FC-B"]:
        return {"category": "FC-B", "next_action": "localize_or_recalibrate_forced_teacher_before_more_updates"}
    return {
        "category": "FC-C",
        "next_action": "do_not_continue_training_design_larger_or_more_sensitive_frozen_audit",
    }


def reconstruct_step817_candidates(
    root: Path, config: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    pilot_config = load_pilot_config(_resolve(root, config["pilot_config"]), root)
    checkpoint = _resolve(root, config["source_checkpoint"])
    source_state_before = sha256_file(checkpoint / "state.pt")
    source_manifest_before = sha256_file(checkpoint / "manifest.json")
    outer_rng = capture_rng()
    runtime = None
    try:
        runtime, payload = _load_pilot_runtime(root, pilot_config, checkpoint, device)
        _, parameters = _runtime_binding(runtime)
        official_before = {key: value.detach().cpu().clone() for key, value in get_peft_model_state_dict(runtime["current"]).items()}
        optimizer_before = tensor_tree_hash(payload["optimizer_state"])
        continuation = _rng_state_from_payload(payload["rng"])
        restore_rng(continuation)
        runtime["current"].train()
        pair = catalog_pair(4, len(runtime["forget"]), len(runtime["retain"]), 16, 42)
        if pair["batch_hash"] != config["authority"]["step817_batch_hash"]:
            raise ValueError("reconstructed step817 batch hash mismatch")
        forget = move_batch(_batch(runtime["forget"], pair["forget_indices"]), device)
        retain = move_batch(_batch(runtime["retain"], pair["retain_indices"]), device)
        components = compute_components(
            runtime["current"], runtime["original"], runtime["augmented"], forget, retain, 2.0
        )
        gradients = _flatten_gradients(components, parameters)
        continuation_after_gradient = capture_rng()
        projection = direct_svd_projection(
            gradients["L_forget"], gradients["L_retain_KL"], gradients["L_sup"],
            relative_tolerance=1.0e-10,
        )
        proposal = shadow_adamw_proposal_for_step(
            [parameter.detach().cpu() for parameter in parameters],
            payload["optimizer_state"], projection["safe"], 816,
        )
        if proposal["delta_hash"] != config["authority"]["shadow_delta_hash"]:
            raise ValueError("step817 shadow AdamW delta hash mismatch")
        projected = update_space_projection(proposal["delta"], projection["basis"])
        candidates: dict[float, list[torch.Tensor]] = {}
        evidence: dict[str, Any] = {}
        for scale in (1.0, 0.5, 0.25):
            after, actual = materialize_dtype_delta(
                [parameter.detach().cpu() for parameter in parameters], projected * scale
            )
            actual_hash = _tensor_hash(actual)
            if actual_hash != EXPECTED_DELTA_HASHES[scale]:
                raise ValueError(f"scale {scale} actual delta hash differs from Pilot authority")
            actual_metrics = directional_metrics(
                gradients["L_forget"], gradients["L_retain_KL"], gradients["L_sup"], actual
            )
            evidence[str(scale)] = {
                "actual_delta_hash": actual_hash,
                "actual_directional": actual_metrics,
            }
            if scale in SCALES:
                candidates[scale] = after
            else:
                before_update_projection = directional_metrics(
                    gradients["L_forget"], gradients["L_retain_KL"], gradients["L_sup"], proposal["delta"] * scale
                )
                after_update_projection = directional_metrics(
                    gradients["L_forget"], gradients["L_retain_KL"], gradients["L_sup"], projected * scale
                )
                reprojected = update_space_projection(actual, projection["basis"])
                reprojected_metrics = directional_metrics(
                    gradients["L_forget"], gradients["L_retain_KL"], gradients["L_sup"], reprojected
                )
                evidence[str(scale)].update({
                    "float64_before_update_projection": before_update_projection,
                    "float64_after_update_projection": after_update_projection,
                    "dtype_roundtrip": actual_metrics,
                    "float64_reprojection_after_dtype_roundtrip": reprojected_metrics,
                    "frozen_retain_sup_tolerance": 1.0e-8,
                    "failure_source": "dtype_quantization" if abs(reprojected_metrics["L_sup"]["normalized"] or 0.0) <= 1.0e-8 else "projection_or_mapping",
                    "parameter_mapping_validated": True,
                    "candidate_committable": False,
                })
        official_after = get_peft_model_state_dict(runtime["current"])
        if official_before.keys() != official_after.keys() or any(
            not torch.equal(official_before[key], official_after[key].detach().cpu()) for key in official_before
        ):
            raise RuntimeError("candidate reconstruction modified official student")
        if optimizer_before != tensor_tree_hash(payload["optimizer_state"]):
            raise RuntimeError("candidate reconstruction modified official optimizer state")
        if any(parameter.grad is not None for parameter in parameters):
            raise RuntimeError("candidate reconstruction populated official .grad")
        if sha256_file(checkpoint / "state.pt") != source_state_before or sha256_file(checkpoint / "manifest.json") != source_manifest_before:
            raise RuntimeError("candidate reconstruction changed source checkpoint")
        return {
            "runtime": runtime,
            "candidates": candidates,
            "evidence": json_native(evidence),
            "step817_batch": {
                "batch_hash": pair["batch_hash"], "catalog_index": pair["catalog_index"],
                "forget_epoch": pair["forget_epoch"], "forget_position": pair["forget_position"],
                "retain_epoch": pair["retain_epoch"], "retain_position": pair["retain_position"],
            },
            "gradient_space_projection": {
                key: projection[key] for key in (
                    "rank", "singular_values", "condition_number", "rho", "eta_F",
                    "normalized_residuals", "retain_dots_after", "algorithm",
                )
            },
            "continuation_rng_hash": rng_hashes(continuation_after_gradient),
            "source_state_sha256_before_after": source_state_before,
            "source_manifest_sha256_before_after": source_manifest_before,
            "official_parameters_modified": False,
            "official_optimizer_modified": False,
            "official_optimizer_step_calls": 0,
            "step817_checkpoint_published": False,
        }
    except BaseException:
        if runtime is not None:
            del runtime
        raise
    finally:
        restore_rng(outer_rng)


def _distribution_metrics(p: torch.Tensor, q: torch.Tensor) -> dict[str, torch.Tensor]:
    epsilon = 1.0e-12
    p_safe = p.clamp_min(epsilon)
    q_safe = q.clamp_min(epsilon)
    middle = ((p + q) * 0.5).clamp_min(epsilon)
    return {
        "l2_mse": ((p - q) ** 2).mean(dim=-1),
        "jsd": 0.5 * (
            (p * (p_safe.log() - middle.log())).sum(dim=-1)
            + (q * (q_safe.log() - middle.log())).sum(dim=-1)
        ),
        "kl": (p * (p_safe.log() - q_safe.log())).sum(dim=-1),
        "argmax_agreement": (p.argmax(dim=-1) == q.argmax(dim=-1)).float(),
    }


def _append_metric_rows(
    rows: list[dict[str, Any]],
    sample_ids: list[int],
    user_ids: list[int],
    labels: torch.Tensor,
    logits: dict[str, torch.Tensor],
    valid_vocab_size: int,
    yes_id: int,
    no_id: int,
) -> None:
    first_logits = {name: value[:, 0] for name, value in logits.items()}
    full = {name: torch.softmax(value[:, :valid_vocab_size].float(), dim=-1) for name, value in first_logits.items()}
    yes_no = {
        name: torch.softmax(value[:, [no_id, yes_id]].float(), dim=-1) for name, value in first_logits.items()
    }
    forced = forced_logits(logits["original"], logits["augmented"], 2.0)
    forced_probability = torch.softmax(forced.float(), dim=-1)
    log_before = torch.log_softmax(logits["before"].float(), dim=-1)
    log_candidate = torch.log_softmax(logits["candidate"].float(), dim=-1)
    forced_before = -(forced_probability * log_before).sum(dim=-1).mean(dim=-1)
    forced_candidate = -(forced_probability * log_candidate).sum(dim=-1).mean(dim=-1)
    log_probabilities = {name: torch.log_softmax(value.float(), dim=-1) for name, value in logits.items()}
    valid_labels = labels != -100
    safe_labels = labels.masked_fill(~valid_labels, 0)
    answer_loss = {
        name: ((-value.gather(2, safe_labels[:, :, None]).squeeze(2)) * valid_labels).sum(dim=1) / valid_labels.sum(dim=1).clamp_min(1)
        for name, value in log_probabilities.items()
    }
    margins = {name: value[:, yes_id].float() - value[:, no_id].float() for name, value in first_logits.items()}
    comparisons: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for space_name, distributions in (("full", full), ("yes_no", yes_no)):
        comparisons[space_name] = {}
        for reference in ("retrain", "original"):
            comparisons[space_name][reference] = {
                "before": _distribution_metrics(distributions["before"], distributions[reference]),
                "candidate": _distribution_metrics(distributions["candidate"], distributions[reference]),
            }
    candidate_yes = yes_no["candidate"][:, 1]
    candidate_confidence = yes_no["candidate"].max(dim=-1).values
    for offset, sample_id in enumerate(sample_ids):
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "user_id": user_ids[offset],
            "label": int(labels[offset, 0]),
            "forced_teacher_ce_before": float(forced_before[offset]),
            "forced_teacher_ce_candidate": float(forced_candidate[offset]),
            "retrain_before_answer_loss_distance": float(abs(answer_loss["before"][offset] - answer_loss["retrain"][offset])),
            "retrain_candidate_answer_loss_distance": float(abs(answer_loss["candidate"][offset] - answer_loss["retrain"][offset])),
            "retrain_before_margin_distance": float(abs(margins["before"][offset] - margins["retrain"][offset])),
            "retrain_candidate_margin_distance": float(abs(margins["candidate"][offset] - margins["retrain"][offset])),
            "candidate_yes_probability": float(candidate_yes[offset]),
            "candidate_confidence": float(candidate_confidence[offset]),
            "candidate_positive": int(candidate_yes[offset] >= 0.5),
        }
        for space_name, references in comparisons.items():
            for reference, states in references.items():
                for state_name, metrics in states.items():
                    for metric_name, values in metrics.items():
                        row[f"{space_name}_{reference}_{state_name}_{metric_name}"] = float(values[offset])
        rows.append(row)


def evaluate_candidate_unit(
    runtime: dict[str, Any],
    candidate_parameters: list[torch.Tensor],
    retrain: torch.nn.Module,
    validation: JsonPromptDataset,
    indices: list[int],
    user_by_index: dict[int, int],
    device: torch.device,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate = copy.deepcopy(runtime["current"])
    trainable = [parameter for parameter in candidate.parameters() if parameter.requires_grad]
    if len(trainable) != len(candidate_parameters):
        raise ValueError("candidate trainable parameter mapping mismatch")
    with torch.no_grad():
        for parameter, value in zip(trainable, candidate_parameters):
            if parameter.shape != value.shape or parameter.dtype != value.dtype:
                raise ValueError("candidate parameter shape/dtype mismatch")
            parameter.copy_(value.to(parameter.device))
    candidate.eval()
    runtime["current"].eval()
    runtime["original"].eval()
    runtime["augmented"].eval()
    retrain.eval()
    rows: list[dict[str, Any]] = []
    batch_size = int(config["evaluation"]["batch_size"])
    valid_vocab_size = int(runtime["tokenizer"].vocab_size)
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            batch = move_batch(_batch(validation, selected), device)
            target = batch["target_ids"]
            model_logits = {}
            for name, model in (
                ("before", runtime["current"]), ("candidate", candidate),
                ("original", runtime["original"]), ("augmented", runtime["augmented"]),
                ("retrain", retrain),
            ):
                model_logits[name] = model(input_ids=batch["input_ids"], labels=batch["target_ids"]).logits.float()
            _append_metric_rows(
                rows, selected, [user_by_index[index] for index in selected], target,
                model_logits, valid_vocab_size,
                int(config["evaluation"]["yes_token_id"]), int(config["evaluation"]["no_token_id"]),
            )
            del batch, target, model_logits
    probability = np.asarray([row["candidate_yes_probability"] for row in rows])
    summary = {
        "samples": len(rows),
        "users": len({row["user_id"] for row in rows}),
        "candidate_confidence_mean": float(np.mean([row["candidate_confidence"] for row in rows])),
        "candidate_probability_mean": float(probability.mean()),
        "candidate_probability_std": float(probability.std()),
        "candidate_positive_rate": float(np.mean([row["candidate_positive"] for row in rows])),
        "probability_collapse": bool(probability.std() <= 1.0e-12 or np.mean(probability >= 0.5) in (0.0, 1.0)),
        "full_vocabulary_logits_persisted": False,
        "retrain_used_for_selection": False,
        "test_accessed": False,
    }
    del candidate
    return rows, summary


def build_contract(pre: dict[str, Any], run_name: str) -> dict[str, Any]:
    return json_native({
        "schema": SCHEMA,
        "run_name": run_name,
        "config_sha256": pre["config_sha256"],
        "git": pre["git"],
        "implementation": pre["implementation"],
        "pilot_authority": pre["pilot_authority"],
        "model_lineage": pre["model_lineage"],
        "tokenizer_lineage": pre["tokenizer_lineage"],
        "data_lineage": pre["data_lineage"],
        "minimum_effect": pre["minimum_effect"],
        "bootstrap": pre["bootstrap"],
        "scales_are_frozen_counterfactuals_not_selection": True,
        "optimizer_steps_committed": 0,
        "step817_checkpoint_published": False,
        "test_accessed": False,
    })


def _acquire_lock(run_dir: Path) -> Path:
    lock = run_dir / "RUN.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError("RunName is locked by another invocation") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid()}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    return lock


def _load_user_map(root: Path, pre: dict[str, Any]) -> dict[int, int]:
    path = Path(pre["data_lineage"]["validation_sidecar"]["path"])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    result = {int(row["processed_row_index"]): int(row["authoritative_user_id"]) for row in rows}
    if len(result) != len(rows):
        raise ValueError("validation sidecar has duplicate row indices")
    return result


def validate_units_inventory(run_dir: Path) -> list[str]:
    unit_root = run_dir / "units"
    if not unit_root.exists():
        return []
    expected = {_unit_name(scale) for scale in SCALES}
    names = {item.name for item in unit_root.iterdir()}
    if not names <= expected:
        raise ValueError("unit cache contains extra or partial-stage artifacts")
    return sorted(names)


def _main_batch_change(root: Path, config: dict[str, Any], scale: float) -> float:
    state = _read_json(_resolve(root, config["pilot_full"]) / "run_state.json")
    trial = next(item for item in state["scale_trials"] if item["scale"] == scale)
    return float(trial["evidence"]["paired_replay"]["actual_delta"]["L_forget"])


def _publish_full_terminal(run_dir: Path, contract_sha: str, state: dict[str, Any]) -> None:
    atomic_json(run_dir / "run_state.json", state)
    units = {}
    if set(validate_units_inventory(run_dir)) != {_unit_name(scale) for scale in SCALES}:
        raise ValueError("Full run is missing a required scale unit")
    for scale in SCALES:
        path = run_dir / "units" / _unit_name(scale)
        units[str(scale)] = sha256_file(path / "manifest.json")
    manifest = {
        "schema": SCHEMA,
        "contract_sha256": contract_sha,
        "reconstruction_sha256": sha256_file(run_dir / "reconstruction.json"),
        "run_state_sha256": sha256_file(run_dir / "run_state.json"),
        "unit_manifest_sha256": units,
        "optimizer_steps_committed": 0,
        "official_parameters_modified": False,
        "step817_checkpoint_published": False,
        "published_atomically": True,
        "test_accessed": False,
    }
    atomic_json(run_dir / "full_manifest.json", manifest)
    atomic_text(run_dir / "COMPLETED", "FORGET_CONFLICT_FULL_COMPLETED\n")


def _validate_full(run_dir: Path, pre: dict[str, Any], run_name: str) -> dict[str, Any]:
    required = {"contract.json", "reconstruction.json", "run_state.json", "full_manifest.json", "units", "COMPLETED"}
    if not run_dir.is_dir() or {item.name for item in run_dir.iterdir()} != required:
        raise ValueError("Full run incomplete or contains extra artifacts")
    if (run_dir / "COMPLETED").read_text(encoding="utf-8") != "FORGET_CONFLICT_FULL_COMPLETED\n":
        raise ValueError("Full completion marker mismatch")
    contract = _read_json(run_dir / "contract.json")
    expected_contract = build_contract(pre, run_name)
    if contract != expected_contract:
        raise ValueError("Full contract/HEAD/implementation mismatch")
    contract_sha = sha256_file(run_dir / "contract.json")
    reconstruction = _read_json(run_dir / "reconstruction.json")
    if (
        reconstruction.get("candidate_hashes") != {str(key): value for key, value in EXPECTED_DELTA_HASHES.items()}
        or reconstruction.get("optimizer_steps_committed") != 0
        or reconstruction.get("official_parameters_modified") is not False
        or reconstruction.get("step817_checkpoint_published") is not False
        or reconstruction.get("test_accessed") is not False
    ):
        raise ValueError("candidate reconstruction evidence mismatch")
    expected_ids = sorted(
        _data_lineage(
            run_dir.parents[3],
            load_config(run_dir.parents[3] / "configs/t5_e2urec_diagnostics_v1.yaml", run_dir.parents[3]),
            run_dir.parents[3] / "outputs/t5_e2urec_development_protocol_v1",
        )[1]["forget_user_validation"]
    )
    units = {}
    for scale in SCALES:
        units[str(scale)] = validate_unit(
            run_dir / "units" / _unit_name(scale), _unit_binding(contract_sha, pre, scale), expected_ids
        )
    state = _read_json(run_dir / "run_state.json")
    if state != {
        "status": "COMPLETED", "completed_scales": [1.0, 0.5],
        "optimizer_steps_committed": 0, "step817_checkpoint_published": False,
        "official_parameters_modified": False, "test_accessed": False,
    }:
        raise ValueError("Full run state mismatch")
    manifest = _read_json(run_dir / "full_manifest.json")
    expected_manifest = {
        "schema": SCHEMA,
        "contract_sha256": contract_sha,
        "reconstruction_sha256": sha256_file(run_dir / "reconstruction.json"),
        "run_state_sha256": sha256_file(run_dir / "run_state.json"),
        "unit_manifest_sha256": {str(scale): sha256_file(run_dir / "units" / _unit_name(scale) / "manifest.json") for scale in SCALES},
        "optimizer_steps_committed": 0,
        "official_parameters_modified": False,
        "step817_checkpoint_published": False,
        "published_atomically": True,
        "test_accessed": False,
    }
    if manifest != expected_manifest:
        raise ValueError("Full manifest mismatch")
    return {"contract": contract, "contract_sha256": contract_sha, "reconstruction": reconstruction, "units": units, "state": state}


def validate_resume_inventory(run_dir: Path, expected_contract: dict[str, Any]) -> dict[str, Any]:
    if not run_dir.is_dir() or (run_dir / "COMPLETED").exists():
        raise ValueError("Resume requires a nonterminal existing Full run")
    allowed = {"contract.json", "run_state.json", "reconstruction.json", "units"}
    names = {item.name for item in run_dir.iterdir()}
    if not {"contract.json", "run_state.json"} <= names or names - allowed:
        raise ValueError("Resume run contains missing or unexpected artifacts")
    if _read_json(run_dir / "contract.json") != expected_contract:
        raise ValueError("Resume contract/HEAD/implementation mismatch")
    state = _read_json(run_dir / "run_state.json")
    if state.get("status") != "INTERRUPTED" or state.get("test_accessed") is not False or state.get("optimizer_steps_committed") != 0:
        raise ValueError("Resume requires zero-commit abnormal INTERRUPTED state")
    return state


def _execute_full(root: Path, config_path: Path, run_name: str, *, resume: bool) -> dict[str, Any]:
    pre = preflight(root, config_path)
    require_clean_git(pre["git"], "Forget conflict Resume" if resume else "Forget conflict Full")
    config = load_audit_config(config_path, root)
    run_dir = _resolve(root, Path(config["output_root"]) / "full_runs" / _safe_name(run_name), output=True)
    if resume:
        validate_resume_inventory(run_dir, build_contract(pre, run_name))
    else:
        if run_dir.exists():
            raise FileExistsError("refusing to overwrite Forget conflict Full")
        run_dir.mkdir(parents=True)
        atomic_json(run_dir / "contract.json", build_contract(pre, run_name))
        atomic_json(run_dir / "run_state.json", {"status": "RUNNING", "completed_scales": [], "optimizer_steps_committed": 0, "test_accessed": False})
    lock = _acquire_lock(run_dir)
    runtime = None
    retrain = None
    source_state_before = sha256_file(_resolve(root, config["source_checkpoint"]) / "state.pt")
    source_manifest_before = sha256_file(_resolve(root, config["source_checkpoint"]) / "manifest.json")
    try:
        contract_sha = sha256_file(run_dir / "contract.json")
        base = load_config(_resolve(root, config["base_config"]), root)
        _, indices, _ = _data_lineage(root, base, _resolve(root, config["protocol_root"]))
        forget_ids = sorted(indices["forget_user_validation"])
        user_by_index = _load_user_map(root, pre)
        validate_units_inventory(run_dir)
        completed = []
        for scale in SCALES:
            path = run_dir / "units" / _unit_name(scale)
            if path.exists():
                validate_unit(path, _unit_binding(contract_sha, pre, scale), forget_ids)
                completed.append(scale)
        reconstruction = reconstruct_step817_candidates(root, config, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        runtime = reconstruction.pop("runtime")
        official_runtime_hash = tensor_tree_hash(get_peft_model_state_dict(runtime["current"]))
        if runtime["original"].training or runtime["augmented"].training or any(
            parameter.requires_grad for name in ("original", "augmented") for parameter in runtime[name].parameters()
        ):
            raise RuntimeError("Original/Augmented audit teachers are not frozen eval models")
        candidate_parameters = reconstruction.pop("candidates")
        reconstruction_payload = {
            **reconstruction,
            "candidate_hashes": {str(key): value for key, value in EXPECTED_DELTA_HASHES.items()},
            "optimizer_steps_committed": 0,
            "official_parameters_modified": False,
            "step817_checkpoint_published": False,
            "test_accessed": False,
        }
        if (run_dir / "reconstruction.json").exists():
            if _read_json(run_dir / "reconstruction.json") != reconstruction_payload:
                raise ValueError("Resume reconstruction evidence differs")
        else:
            atomic_json(run_dir / "reconstruction.json", reconstruction_payload)
        validation = JsonPromptDataset(Path(base["paths"]["validation"]), runtime["tokenizer"])
        device = next(runtime["current"].parameters()).device
        retrain = freeze_teacher(load_legacy_model(Path(base["paths"]["retrain_reference"]))).to(device)
        if retrain.training or any(parameter.requires_grad for parameter in retrain.parameters()):
            raise RuntimeError("Retrain audit reference is not frozen eval")
        for scale in SCALES:
            if scale in completed:
                del candidate_parameters[scale]
                continue
            rows, summary = evaluate_candidate_unit(
                runtime, candidate_parameters.pop(scale), retrain, validation, forget_ids,
                user_by_index, device, config,
            )
            summary["scale"] = scale
            summary["candidate_actual_delta_hash"] = EXPECTED_DELTA_HASHES[scale]
            summary["main_batch_forget_loss_change"] = _main_batch_change(root, config, scale)
            summary["full_development_forced_teacher_change"] = float(np.mean([
                row["forced_teacher_ce_candidate"] - row["forced_teacher_ce_before"] for row in rows
            ]))
            summary["main_and_global_forced_direction_agree"] = (
                math.copysign(1.0, summary["main_batch_forget_loss_change"])
                == math.copysign(1.0, summary["full_development_forced_teacher_change"])
            )
            publish_unit(run_dir / "units" / _unit_name(scale), rows, summary, _unit_binding(contract_sha, pre, scale))
            completed.append(scale)
            atomic_json(run_dir / "run_state.json", {"status": "RUNNING", "completed_scales": completed, "optimizer_steps_committed": 0, "test_accessed": False})
            del rows
        if tensor_tree_hash(get_peft_model_state_dict(runtime["current"])) != official_runtime_hash:
            raise RuntimeError("candidate evaluation modified official student parameters")
        if sha256_file(_resolve(root, config["source_checkpoint"]) / "state.pt") != source_state_before or sha256_file(_resolve(root, config["source_checkpoint"]) / "manifest.json") != source_manifest_before:
            raise RuntimeError("formal step816 checkpoint changed")
        for name, key in (("original", "original"), ("augmented", "augmented_teacher"), ("retrain", "retrain_reference")):
            if sha256_file(Path(base["paths"][key])) != pre["model_lineage"][name]["sha256"]:
                raise RuntimeError(f"{name} checkpoint changed during frozen audit")
        terminal = {
            "status": "COMPLETED", "completed_scales": [1.0, 0.5],
            "optimizer_steps_committed": 0, "step817_checkpoint_published": False,
            "official_parameters_modified": False, "test_accessed": False,
        }
        _publish_full_terminal(run_dir, contract_sha, terminal)
        return {"status": "COMPLETED", "run_dir": str(run_dir), **terminal}
    except BaseException:
        if run_dir.exists() and not (run_dir / "COMPLETED").exists():
            prior = _read_json(run_dir / "run_state.json")
            prior["status"] = "INTERRUPTED"
            prior["optimizer_steps_committed"] = 0
            prior["test_accessed"] = False
            atomic_json(run_dir / "run_state.json", prior)
        raise
    finally:
        if retrain is not None:
            del retrain
        if runtime is not None:
            del runtime
        if lock.exists():
            lock.unlink()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def analyze(root: Path, config_path: Path, run_name: str) -> dict[str, Any]:
    pre = preflight(root, config_path)
    require_clean_git(pre["git"], "Forget conflict Analyze")
    config = load_audit_config(config_path, root)
    full_dir = _resolve(root, Path(config["output_root"]) / "full_runs" / _safe_name(run_name), output=True)
    verified = _validate_full(full_dir, pre, run_name)
    final = _resolve(root, Path(config["output_root"]) / "analysis_runs" / _safe_name(run_name), output=True)
    if final.exists():
        raise FileExistsError("refusing to overwrite Forget conflict Analyze")
    scale_results = {}
    for scale in SCALES:
        unit = verified["units"][str(scale)]
        full = clustered_bootstrap(unit["rows"], "full", seed=42, resamples=2000)
        yes_no = clustered_bootstrap(unit["rows"], "yes_no", seed=42, resamples=2000)
        summary = unit["summary"]
        classification = classify_scale(
            full, yes_no, config["minimum_effect"],
            collapse=summary["probability_collapse"],
            main_batch_forget_change=-float(summary["main_batch_forget_loss_change"]),
            global_forced_change=-float(summary["full_development_forced_teacher_change"]),
        )
        scale_results[str(scale)] = {
            "scale": scale,
            "full_valid_vocabulary": full,
            "yes_no_conditional": yes_no,
            "classification": classification,
            "summary": summary,
        }
    overall = classify_audit(scale_results)
    result = {
        "schema": ANALYSIS_SCHEMA,
        "run_name": run_name,
        **overall,
        "scale_results": scale_results,
        "scale_0_25_numeric_diagnostic": verified["reconstruction"]["evidence"]["0.25"],
        "retrain_role": "posthoc_audit_only_not_scale_selection",
        "scales_compared_as_frozen_counterfactuals": [1.0, 0.5],
        "optimizer_steps_committed": 0,
        "step817_checkpoint_published": False,
        "test_accessed": False,
    }
    stage = final.parent / f".{run_name}.{uuid.uuid4().hex[:10]}.stage"
    stage.mkdir(parents=True)
    try:
        atomic_json(stage / "analysis.json", result)
        atomic_json(stage / "manifest.json", {
            "schema": ANALYSIS_SCHEMA,
            "analysis_sha256": sha256_file(stage / "analysis.json"),
            "source_full_manifest_sha256": sha256_file(full_dir / "full_manifest.json"),
            "published_atomically": True,
            "test_accessed": False,
        })
        atomic_text(stage / "COMPLETED", "FORGET_CONFLICT_ANALYSIS_COMPLETED\n")
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, final)
    except BaseException:
        if stage.exists():
            for child in stage.iterdir():
                child.unlink()
            stage.rmdir()
        raise
    return result


def _synthetic_rows(scale: float) -> list[dict[str, Any]]:
    rows = []
    for sample_id in range(12):
        improvement = 0.004 * scale + (sample_id % 3 - 1) * 0.0001
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "user_id": sample_id // 2,
            "label": sample_id % 2,
            "forced_teacher_ce_before": 0.30,
            "forced_teacher_ce_candidate": 0.30 - improvement,
            "retrain_before_answer_loss_distance": 0.10,
            "retrain_candidate_answer_loss_distance": 0.10 - improvement,
            "retrain_before_margin_distance": 0.20,
            "retrain_candidate_margin_distance": 0.20 - improvement,
            "candidate_yes_probability": 0.3 + 0.03 * sample_id,
            "candidate_confidence": 0.7,
            "candidate_positive": int(sample_id >= 7),
        }
        for space in ("full", "yes_no"):
            for reference in ("retrain", "original"):
                for state in ("before", "candidate"):
                    if reference == "retrain":
                        base = 0.04
                        value = base if state == "before" else base - improvement
                    else:
                        base = 0.03
                        value = base if state == "before" else base + improvement
                    row[f"{space}_{reference}_{state}_l2_mse"] = value * value
                    row[f"{space}_{reference}_{state}_jsd"] = value
                    row[f"{space}_{reference}_{state}_kl"] = value * 2
                    row[f"{space}_{reference}_{state}_argmax_agreement"] = float(
                        state == "candidate" if reference == "retrain" else state == "before"
                    )
        rows.append(row)
    return rows


def synthetic_dry_run(root: Path, config_path: Path, run_name: str) -> dict[str, Any]:
    config = load_audit_config(config_path, root)
    final = _resolve(root, Path(config["output_root"]) / "synthetic_runs" / _safe_name(run_name), output=True)
    if final.exists():
        raise FileExistsError("refusing to overwrite SyntheticDryRun")
    final.mkdir(parents=True)
    synthetic_pre = {
        "git": {"git_commit": "synthetic", "git_working_tree_clean": True},
        "implementation": {"canonical_sha256": "synthetic"},
        "pilot_authority": {
            "state_sha256": config["authority"]["step816_state_sha256"],
            "manifest_sha256": config["authority"]["step816_manifest_sha256"],
            "step817_batch_hash": config["authority"]["step817_batch_hash"],
            "rng_hash": {"synthetic": "seed42"},
        },
        "data_lineage": {
            "data": {"overall_validation": {"sha256": config["lineage_sha256"]["validation"]}},
            "validation_sidecar": {"sha256": config["lineage_sha256"]["validation_user_sidecar"]},
        },
        "forget_validation": {"indices_sha256": "synthetic-order"},
    }
    contract = {"schema": SCHEMA, "mode": "SyntheticDryRun", "run_name": run_name, "test_accessed": False}
    atomic_json(final / "contract.json", contract)
    contract_sha = sha256_file(final / "contract.json")
    resume_skipped = []
    for scale in SCALES:
        rows = _synthetic_rows(scale)
        summary = {
            "scale": scale, "samples": len(rows), "users": 6,
            "candidate_probability_std": float(np.std([row["candidate_yes_probability"] for row in rows])),
            "candidate_positive_rate": float(np.mean([row["candidate_positive"] for row in rows])),
            "probability_collapse": False,
            "main_batch_forget_loss_change": -0.004 * scale,
            "full_development_forced_teacher_change": -0.004 * scale,
            "main_and_global_forced_direction_agree": True,
            "retrain_used_for_selection": False, "test_accessed": False,
        }
        binding = _unit_binding(contract_sha, synthetic_pre, scale)
        publish_unit(final / "units" / _unit_name(scale), rows, summary, binding)
        if scale == 1.0:
            atomic_json(final / "run_state.json", {"status": "INTERRUPTED", "completed_scales": [1.0], "optimizer_steps_committed": 0, "test_accessed": False})
            validate_unit(final / "units" / _unit_name(scale), binding, list(range(12)))
            resume_skipped.append(scale)
    classifications = {}
    for scale in SCALES:
        rows = validate_unit(final / "units" / _unit_name(scale), _unit_binding(contract_sha, synthetic_pre, scale), list(range(12)))["rows"]
        full = clustered_bootstrap(rows, "full", seed=42, resamples=100)
        yes_no = clustered_bootstrap(rows, "yes_no", seed=42, resamples=100)
        classifications[str(scale)] = classify_scale(
            full, yes_no, config["minimum_effect"], collapse=False,
            main_batch_forget_change=0.004 * scale, global_forced_change=0.004 * scale,
        )
    result = {
        "schema": SCHEMA, "mode": "SyntheticDryRun", "units": [1.0, 0.5],
        "resume_skipped_complete_units": resume_skipped,
        "candidate_hash_gate_verified": EXPECTED_DELTA_HASHES,
        "bootstrap_deterministic": clustered_bootstrap(_synthetic_rows(1.0), "full", seed=42, resamples=20) == clustered_bootstrap(_synthetic_rows(1.0), "full", seed=42, resamples=20),
        "classifications": classifications,
        "optimizer_steps_committed": 0,
        "official_parameters_modified": False,
        "step817_checkpoint_published": False,
        "full_vocabulary_logits_persisted": False,
        "test_accessed": False,
    }
    atomic_json(final / "synthetic_result.json", result)
    atomic_json(final / "run_state.json", {"status": "SYNTHETIC_COMPLETED", "optimizer_steps_committed": 0, "test_accessed": False})
    atomic_text(final / "COMPLETED", "FORGET_CONFLICT_SYNTHETIC_COMPLETED\n")
    return {**result, "run_dir": str(final)}


def main() -> None:
    parser = argparse.ArgumentParser(description="T5 step817 frozen Forget conflict audit")
    parser.add_argument("--mode", choices=("Preflight", "SyntheticDryRun", "Full", "Resume", "Analyze"), default="Preflight")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-name")
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = args.config.resolve()
    if args.mode == "Preflight":
        result = preflight(root, config)
    else:
        if not args.run_name:
            parser.error(f"{args.mode} requires --run-name")
        if args.mode == "SyntheticDryRun":
            result = synthetic_dry_run(root, config, args.run_name)
        elif args.mode == "Full":
            result = _execute_full(root, config, args.run_name, resume=False)
        elif args.mode == "Resume":
            result = _execute_full(root, config, args.run_name, resume=True)
        else:
            result = analyze(root, config, args.run_name)
    print(json.dumps(json_native(result), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
