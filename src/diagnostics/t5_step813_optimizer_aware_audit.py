from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml
from peft import get_peft_model_state_dict

from src.diagnostics.git_provenance import (
    git_provenance,
    implementation_provenance,
    require_clean_git,
)
from src.diagnostics.t5_full_runner import _batch, _restore_rng_payload
from src.diagnostics.t5_reconstructed_official import (
    compute_components,
    load_config,
    move_batch,
    sha256_file,
)
from src.diagnostics.t5_step812_gradient_geometry import (
    _data_lineage,
    _load_runtime,
    _tensor_state_hash,
    canonical_hash,
    catalog_pair,
    parameter_group,
    project_forget,
    safe_cosine,
)
from src.diagnostics.t5_trajectory_diagnostics import RngState, capture_rng, restore_rng, rng_hashes

SCHEMA = "t5-step813-optimizer-aware-audit-v1"
ANALYSIS_SCHEMA = "t5-step813-optimizer-aware-analysis-v1"
OUTPUT_NAME = "t5_step813_optimizer_aware_audit_v1"
LOSSES = ("L_forget", "L_retain_KL", "L_sup")
EXPECTED_STATE_SHA256 = "214042180cb67296dd561a6908541a967c1ad30815b08cce3e0b1f1e610489c8"
EXPECTED_MANIFEST_SHA256 = "6304ef86fc022b48e4049fb2eaed28f2e992a04941e687c32604512f3d2b9d14"
EXPECTED_BATCH_HASH = "6b3e14ea4a2496a6ee60e984e91b99e1cc4b08f3d71a1ea4bb90d53b37681773"
IMPLEMENTATION_FILES = (
    "src/diagnostics/t5_step813_optimizer_aware_audit.py",
    "configs/t5_step813_optimizer_aware_audit_v1.yaml",
    "scripts/diagnostics/t5_step813_optimizer_aware_audit_v1.ps1",
    "docs/t5_step813_optimizer_aware_audit_v1.md",
    "src/diagnostics/t5_step812_gradient_geometry.py",
    "configs/t5_step812_gradient_geometry_audit_v1.yaml",
    "src/diagnostics/t5_reconstructed_official.py",
    "src/diagnostics/t5_full_runner.py",
    "src/diagnostics/t5_zero_training_audit.py",
    "src/diagnostics/git_provenance.py",
)


def json_native(value: Any) -> Any:
    """Return the strict JSON-native representation or reject unsupported data."""
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
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
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


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


def load_audit_config(path: Path, root: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("development_only") is not True:
        raise ValueError("optimizer-aware audit config schema/scope mismatch")
    if value.get("test_access_policy") != "forbidden":
        raise ValueError("test access policy must be forbidden")
    if value.get("projection") != {
        "algorithm": "direct_float64_retain_matrix_svd",
        "relative_singular_tolerance": 1.0e-10,
    }:
        raise ValueError("projection protocol changed")
    if value.get("classification") != {
        "normalized_retain_violation_tolerance": 1.0e-8,
        "forget_descent_zero_tolerance": 1.0e-12,
        "forget_effectiveness_min": 0.10,
    }:
        raise ValueError("classification protocol changed")
    if value.get("optimizer") != {
        "class": "torch.optim.AdamW",
        "param_groups": 1,
        "lr": 0.001,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "weight_decay": 0.01,
        "decoupled_weight_decay": True,
    }:
        raise ValueError("optimizer protocol changed")
    expected = value["expected"]
    if expected["checkpoint_state_sha256"] != EXPECTED_STATE_SHA256:
        raise ValueError("expected checkpoint state SHA changed")
    if expected["checkpoint_manifest_sha256"] != EXPECTED_MANIFEST_SHA256:
        raise ValueError("expected checkpoint manifest SHA changed")
    if expected["step813_batch_hash"] != EXPECTED_BATCH_HASH:
        raise ValueError("expected step813 batch hash changed")
    value["_path"] = str(path.resolve())
    value["_sha256"] = sha256_file(path)
    return value


def _checkpoint_payload(checkpoint: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path, state_path = checkpoint / "manifest.json", checkpoint / "state.pt"
    if sha256_file(state_path) != EXPECTED_STATE_SHA256:
        raise ValueError("step812 state.pt SHA256 mismatch")
    if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("step812 manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("step") != 812
        or manifest.get("published_atomically") is not True
        or manifest.get("state_sha256") != EXPECTED_STATE_SHA256
    ):
        raise ValueError("step812 manifest contract mismatch")
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema") != "t5-e2urec-full-runner-v1"
        or payload.get("state", {}).get("step") != 812
        or payload.get("state", {}).get("next_optimizer_step") != 813
        or payload.get("state", {}).get("next_batch_hash") != EXPECTED_BATCH_HASH
    ):
        raise ValueError("step812 serialized state/batch anchor mismatch")
    if payload.get("test_accessed") not in (None, False):
        raise ValueError("checkpoint is not test-free")
    return manifest, payload


def validate_checkpoint_rng(payload: dict[str, Any]) -> dict[str, Any]:
    """Recompute serialized RNG hashes without changing or activating runtime RNGs."""
    if set(payload.get("rng", {})) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("checkpoint RNG payload fields mismatch")
    expected = payload.get("rng_hash")
    if not isinstance(expected, dict):
        raise ValueError("checkpoint RNG hash is missing")
    serialized = payload["rng"]
    actual = rng_hashes(RngState(
        python=serialized["python"],
        numpy=serialized["numpy"],
        torch_cpu=serialized["torch_cpu"],
        torch_cuda=serialized["torch_cuda"],
    ))
    if actual != expected:
        raise ValueError("checkpoint serialized RNG hash mismatch")
    return actual


def _tensor_metadata(value: torch.Tensor) -> dict[str, Any]:
    return {"shape": list(value.shape), "dtype": str(value.dtype), "numel": value.numel()}


def validate_optimizer_mapping(
    adapter: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any],
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(adapter, dict) or not adapter or any(not torch.is_tensor(v) for v in adapter.values()):
        raise ValueError("adapter state is missing or invalid")
    groups = optimizer_state.get("param_groups")
    states = optimizer_state.get("state")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(states, dict):
        raise ValueError("optimizer must have one complete param group")
    ids = groups[0].get("params")
    if not isinstance(ids, list) or len(ids) != len(adapter):
        raise ValueError("optimizer param mapping length mismatch")
    if len(set(ids)) != len(ids) or set(ids) != set(states):
        raise ValueError("optimizer param IDs are missing, duplicated, or ambiguous")
    names = list(adapter)
    bindings = []
    for position, (identifier, name) in enumerate(zip(ids, names)):
        parameter = adapter[name]
        state = states[identifier]
        if not isinstance(state, dict) or not {"step", "exp_avg", "exp_avg_sq"} <= set(state):
            raise ValueError(f"optimizer state fields missing at position {position}")
        if not torch.is_tensor(state["step"]) or state["step"].numel() != 1:
            raise ValueError(f"optimizer step counter invalid at position {position}")
        if not math.isfinite(float(state["step"])) or float(state["step"]) != 812.0:
            raise ValueError(f"optimizer step counter mismatch at position {position}")
        for key in ("exp_avg", "exp_avg_sq"):
            tensor = state[key]
            if not torch.is_tensor(tensor) or tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                raise ValueError(f"optimizer {key} shape/dtype mismatch at position {position}")
            if not bool(torch.isfinite(tensor).all()):
                raise FloatingPointError(f"optimizer {key} non-finite at position {position}")
        bindings.append([identifier, name, list(parameter.shape), str(parameter.dtype)])
    group = groups[0]
    required_group = {
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "eps": 1.0e-8,
        "weight_decay": 0.01,
        "amsgrad": False,
        "maximize": False,
    }
    if any(group.get(key) != value for key, value in required_group.items()):
        raise ValueError("optimizer param-group hyperparameter mismatch")
    if group.get("decoupled_weight_decay") is not True:
        raise ValueError("optimizer is not decoupled AdamW")
    report = {
        "binding_authority": (
            "checkpoint writer serialized PEFT adapter OrderedDict and optimizer.state_dict "
            "from the same authoritative trainable parameter order; Full additionally binds "
            "normalized runtime names/shapes/order before any shadow construction"
        ),
        "tensor_count": len(names),
        "parameter_count": sum(value.numel() for value in adapter.values()),
        "adapter_order_sha256": canonical_hash(names),
        "adapter_shape_dtype_sha256": canonical_hash(
            [[name, list(value.shape), str(value.dtype)] for name, value in adapter.items()]
        ),
        "optimizer_param_ids_sha256": canonical_hash(ids),
        "optimizer_mapping_sha256": canonical_hash(bindings),
        "optimizer_state_order_sha256": canonical_hash(list(states)),
        "param_groups": 1,
        "hyperparameters": json_native(
            {key: group.get(key) for key in group if key != "params"}
        ),
        "all_step_counters": 812,
        "runtime_binding_pending_full": True,
    }
    if expected is not None:
        checks = {
            "tensor_count": expected["adapter_tensors"],
            "parameter_count": expected["adapter_parameters"],
            "adapter_order_sha256": expected["adapter_order_sha256"],
            "adapter_shape_dtype_sha256": expected["adapter_shape_dtype_sha256"],
            "optimizer_param_ids_sha256": expected["optimizer_param_ids_sha256"],
            "optimizer_mapping_sha256": expected["optimizer_mapping_sha256"],
        }
        if any(report[key] != value for key, value in checks.items()):
            raise ValueError("optimizer/adapter binding differs from preregistration")
    return json_native(report)


def _tree_sha256(path: Path) -> str:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError("tokenizer definition is empty")
    return canonical_hash({item.relative_to(path).as_posix(): sha256_file(item) for item in files})


def preflight(
    root: Path,
    config_path: Path,
    *,
    git_function=git_provenance,
    implementation_function=implementation_provenance,
) -> dict[str, Any]:
    audit = load_audit_config(config_path, root)
    base_path = _resolve(root, audit["base_config"])
    base = load_config(base_path, root)
    checkpoint = _resolve(root, audit["step812_checkpoint"])
    manifest, payload = _checkpoint_payload(checkpoint)
    checkpoint_rng = validate_checkpoint_rng(payload)
    mapping = validate_optimizer_mapping(
        payload["adapter_state"], payload["optimizer_state"], audit["expected"]
    )
    forget_path, retain_path = Path(base["paths"]["forget"]), Path(base["paths"]["retain"])
    forget_count = len(json.loads(forget_path.read_text(encoding="utf-8")))
    retain_count = len(json.loads(retain_path.read_text(encoding="utf-8")))
    primary = catalog_pair(0, forget_count, retain_count, 16, 42)
    if primary["batch_hash"] != EXPECTED_BATCH_HASH:
        raise ValueError("step813 deterministic batch hash mismatch")
    lineage, _, _ = _data_lineage(root, base, _resolve(root, audit["protocol_root"]))
    if lineage["data"]["forget_train"]["sha256"] != sha256_file(forget_path):
        raise ValueError("Forget lineage mismatch")
    if lineage["data"]["retain_train"]["sha256"] != sha256_file(retain_path):
        raise ValueError("Retain lineage mismatch")
    git = git_function(root)
    implementation = implementation_function(root, IMPLEMENTATION_FILES)
    teachers = {}
    for role, key in (("original", "original"), ("augmented", "augmented_teacher")):
        path = Path(base["paths"][key])
        teachers[role] = {"path": str(path), "sha256": sha256_file(path), "loaded": False}
        if payload.get("compatibility", {}).get("checkpoint_sha256", {}).get(role) != teachers[role]["sha256"]:
            raise ValueError(f"{role} checkpoint lineage differs from step812 compatibility report")
    tokenizer_path = _resolve(root, base["paths"]["model_dir"])
    result = {
        "schema": SCHEMA,
        "mode": "Preflight",
        "development_only": True,
        "config_sha256": audit["_sha256"],
        "base_config_sha256": sha256_file(base_path),
        "checkpoint": {
            "directory": str(checkpoint),
            "state_sha256": manifest["state_sha256"],
            "manifest_sha256": sha256_file(checkpoint / "manifest.json"),
            "step": 812,
            "next_optimizer_step": 813,
            "rng_hash": checkpoint_rng,
            "rng_hash_recomputed": True,
            "rng_fields": sorted(payload["rng"]),
        },
        "step813": {
            "batch_hash": primary["batch_hash"],
            "forget_epoch": primary["forget_epoch"],
            "forget_position": primary["forget_position"],
            "retain_epoch": primary["retain_epoch"],
            "retain_position": primary["retain_position"],
        },
        "optimizer_mapping": mapping,
        "teachers": teachers,
        "retrain": {"loaded": False, "file_read": False},
        "data": {
            "forget": {"path": str(forget_path), "sha256": sha256_file(forget_path), "samples": forget_count},
            "retain": {"path": str(retain_path), "sha256": sha256_file(retain_path), "samples": retain_count},
        },
        "lineage": {
            "manifest": lineage["manifest"],
            "forget_user_manifest": lineage["forget_user_manifest"],
            "partition_checks": lineage["partition_checks"],
            "processed_test_split_read": False,
            "test_accessed": False,
        },
        "tokenizer": {
            "path": str(tokenizer_path),
            "tree_sha256": _tree_sha256(tokenizer_path),
            "loaded": False,
        },
        "git": git,
        "implementation": implementation,
        "model_loaded": False,
        "gradient_computed": False,
        "authoritative_optimizer_constructed": False,
        "authoritative_optimizer_steps_executed": 0,
        "authoritative_parameters_updated": False,
        "shadow_optimizer_constructed": False,
        "shadow_optimizer_steps_executed": 0,
        "shadow_parameters_updated": False,
        "shadow_updates_committed": False,
        "test_loader_built": False,
        "test_accessed": False,
    }
    return json_native(result)


def direct_svd_projection(
    forget: torch.Tensor,
    kl: torch.Tensor,
    sup: torch.Tensor,
    relative_tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    f, k, s = (value.detach().double().reshape(-1) for value in (forget, kl, sup))
    if any(not bool(torch.isfinite(value).all()) for value in (f, k, s)):
        raise FloatingPointError("non-finite projection input")
    retain = torch.stack((k, s), dim=1)
    left, singular, _ = torch.linalg.svd(retain, full_matrices=False)
    largest = float(singular.max()) if singular.numel() else 0.0
    threshold = largest * relative_tolerance
    active = singular > threshold
    basis = left[:, active]
    safe = f - basis @ (basis.T @ f) if int(active.sum()) else f.clone()
    reference = project_forget(f, k, s, relative_tolerance=relative_tolerance)
    if not torch.equal(safe, reference["g_forget_perp"]):
        raise RuntimeError("direct SVD projection differs from step812 authority")
    return {
        "safe": safe,
        "basis": basis,
        "rank": int(active.sum()),
        "singular_values": [float(value) for value in singular],
        "condition_number": reference["condition_number"],
        "rho": reference["rho"],
        "eta_F": reference["eta_F"],
        "normalized_residuals": reference["normalized_residuals"],
        "retain_dots_after": reference["retain_dots_after"],
        "algorithm": "direct_float64_retain_matrix_svd",
    }


def _split_flat(flat: torch.Tensor, parameters: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    result, offset = [], 0
    for parameter in parameters:
        count = parameter.numel()
        result.append(flat[offset : offset + count].reshape(parameter.shape).to(parameter.dtype))
        offset += count
    if offset != flat.numel():
        raise ValueError("flat gradient length does not match parameter order")
    return result


def shadow_adamw_step(
    official_parameters: list[torch.Tensor],
    optimizer_state: dict[str, Any],
    flat_gradient: torch.Tensor,
    *,
    weight_decay_override: float | None = None,
) -> dict[str, Any]:
    if any(getattr(value, "grad", None) is not None for value in official_parameters):
        raise ValueError("authoritative parameters must have grad=None")
    official_hashes = [_tensor_hash(value) for value in official_parameters]
    shadows = [torch.nn.Parameter(value.detach().clone(), requires_grad=True) for value in official_parameters]
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
    if weight_decay_override is not None:
        for item in optimizer.param_groups:
            item["weight_decay"] = weight_decay_override
    before = [value.detach().clone() for value in shadows]
    gradients = _split_flat(flat_gradient, shadows)
    for parameter, gradient in zip(shadows, gradients):
        parameter.grad = gradient.to(parameter.device)
    optimizer.step()
    delta_parts = [(after.detach() - prior).double().cpu().reshape(-1) for after, prior in zip(shadows, before)]
    delta = torch.cat(delta_parts)
    if any(_tensor_hash(value) != digest for value, digest in zip(official_parameters, official_hashes)):
        raise RuntimeError("shadow AdamW changed authoritative parameter")
    if any(getattr(value, "grad", None) is not None for value in official_parameters):
        raise RuntimeError("shadow AdamW populated authoritative grad")
    return {
        "delta": delta,
        "delta_hash": _tensor_hash(delta),
        "delta_norm": float(torch.linalg.vector_norm(delta)),
        "shadow_optimizer_constructed": True,
        "shadow_optimizer_steps_executed": 1,
        "shadow_parameters_updated": bool(float(torch.linalg.vector_norm(delta)) > 0.0),
        "shadow_updates_committed": False,
        "weight_decay": optimizer.param_groups[0]["weight_decay"],
    }


def _tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def directional_metrics(
    forget: torch.Tensor,
    kl: torch.Tensor,
    sup: torch.Tensor,
    delta: torch.Tensor,
) -> dict[str, Any]:
    vectors = {name: value.detach().double().reshape(-1) for name, value in zip(LOSSES, (forget, kl, sup))}
    d = delta.detach().double().reshape(-1)
    if any(value.numel() != d.numel() for value in vectors.values()):
        raise ValueError("directional derivative shape mismatch")
    if any(not bool(torch.isfinite(value).all()) for value in (*vectors.values(), d)):
        raise FloatingPointError("non-finite directional derivative input")
    delta_norm = float(torch.linalg.vector_norm(d))
    result: dict[str, Any] = {"delta_norm": delta_norm}
    for name, gradient in vectors.items():
        dot = float(torch.dot(gradient, d))
        norm = float(torch.linalg.vector_norm(gradient))
        normalized = None if norm == 0.0 or delta_norm == 0.0 else dot / (norm * delta_norm)
        result[name] = {"dot": dot, "gradient_norm": norm, "normalized": normalized}
    return result


def forget_effectiveness(a0: dict[str, Any], a1: dict[str, Any]) -> float | None:
    denominator = -float(a0["L_forget"]["dot"])
    numerator = -float(a1["L_forget"]["dot"])
    if not math.isfinite(denominator) or not math.isfinite(numerator) or denominator <= 0.0:
        return None
    return numerator / denominator


def classify_optimizer_audit(
    a0: dict[str, Any],
    a1: dict[str, Any],
    a3: dict[str, Any],
    *,
    invariants_valid: bool,
    retain_tolerance: float = 1.0e-8,
    forget_zero_tolerance: float = 1.0e-12,
    effectiveness_min: float = 0.10,
) -> dict[str, str]:
    def finite(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, (bool, str)):
            return True
        if isinstance(value, (int, float)):
            return math.isfinite(float(value))
        if isinstance(value, dict):
            return all(finite(item) for item in value.values())
        return False

    effectiveness = forget_effectiveness(a0, a1)
    if not invariants_valid or not all(finite(value) for value in (a0, a1, a3)) or effectiveness is None:
        return {"category": "OA-D", "next_action": "stop_invalid_or_inconclusive"}
    a1_descent = a1["L_forget"]["dot"] < -forget_zero_tolerance
    a3_descent = a3["L_forget"]["dot"] < -forget_zero_tolerance
    retain_safe = all(
        a1[name]["normalized"] is not None and a1[name]["normalized"] <= retain_tolerance
        for name in ("L_retain_KL", "L_sup")
    )
    a3_retain_safe = all(
        a3[name]["normalized"] is not None and a3[name]["normalized"] <= retain_tolerance
        for name in ("L_retain_KL", "L_sup")
    )
    if a1_descent and effectiveness >= effectiveness_min and retain_safe:
        return {"category": "OA-A", "next_action": "proceed_to_reversible_one_step_stage_B"}
    if not a1_descent or effectiveness < effectiveness_min or not a3_descent:
        return {
            "category": "OA-C",
            "next_action": "inspect_optimizer_state_or_use_constrained_optimizer_plan_B",
        }
    if not retain_safe and a3_retain_safe and a3_descent:
        return {
            "category": "OA-B",
            "next_action": "implement_update_space_projection_before_stage_B",
        }
    return {"category": "OA-D", "next_action": "stop_invalid_or_inconclusive"}


def _gradient_geometry(f: torch.Tensor, k: torch.Tensor, s: torch.Tensor, projection: dict[str, Any]) -> dict[str, Any]:
    vectors = {name: value.double() for name, value in zip(LOSSES, (f, k, s))}
    return {
        "norms": {name: float(torch.linalg.vector_norm(value)) for name, value in vectors.items()},
        "dots": {
            "forget_KL": float(torch.dot(vectors["L_forget"], vectors["L_retain_KL"])),
            "forget_sup": float(torch.dot(vectors["L_forget"], vectors["L_sup"])),
            "KL_sup": float(torch.dot(vectors["L_retain_KL"], vectors["L_sup"])),
        },
        "cosines": {
            "forget_KL": safe_cosine(f, k),
            "forget_sup": safe_cosine(f, s),
            "KL_sup": safe_cosine(k, s),
        },
        "projection": {
            key: value
            for key, value in projection.items()
            if key not in {"safe", "basis"}
        },
        "safe_directional": directional_metrics(f, k, s, -projection["safe"]),
    }


def _counterfactual_scalar(name: str, result: dict[str, Any], metrics: dict[str, Any], authority: bool) -> dict[str, Any]:
    return {
        "name": name,
        "authoritative_configuration": authority,
        "directional": metrics,
        "delta_hash": result.get("delta_hash"),
        "delta_norm": result.get("delta_norm", metrics["delta_norm"]),
        "weight_decay": result.get("weight_decay"),
        "shadow_optimizer_constructed": result.get("shadow_optimizer_constructed", False),
        "shadow_optimizer_steps_executed": result.get("shadow_optimizer_steps_executed", 0),
        "shadow_parameters_updated": result.get("shadow_parameters_updated", False),
        "shadow_updates_committed": False,
    }


def build_authority_contract(
    preflight_value: dict[str, Any],
    config_sha256: str,
    run_name: str,
    classification: dict[str, Any],
) -> dict[str, Any]:
    """Build the single JSON-native authority contract used by Full and Analyze."""
    return json_native({
        "schema": SCHEMA,
        "run_name": run_name,
        "config_sha256": config_sha256,
        "git": preflight_value["git"],
        "implementation": preflight_value["implementation"],
        "checkpoint": preflight_value["checkpoint"],
        "optimizer_mapping": preflight_value["optimizer_mapping"],
        "step813": preflight_value["step813"],
        "classification": classification,
        "test_accessed": False,
    })


def grouped_diagnostics(
    names: list[str],
    shapes: list[torch.Size],
    f: torch.Tensor,
    k: torch.Tensor,
    s: torch.Tensor,
    deltas: dict[str, torch.Tensor],
    retain_tolerance: float,
) -> dict[str, Any]:
    slices, offset = [], 0
    groups: dict[str, list[int]] = {"all": list(range(len(names)))}
    for index, (name, shape) in enumerate(zip(names, shapes)):
        count = math.prod(shape)
        slices.append(slice(offset, offset + count))
        offset += count
        metadata = parameter_group(name)
        for kind in ("stack", "projection", "layer", "module"):
            groups.setdefault(f"{kind}:{metadata[kind]}", []).append(index)
    if offset != f.numel():
        raise ValueError("group slices do not cover flattened parameter vector")
    total_f_energy = max(float(torch.dot(f, f)), torch.finfo(torch.float64).tiny)
    total_update_energy = {
        key: max(float(torch.dot(value, value)), torch.finfo(torch.float64).tiny)
        for key, value in deltas.items()
    }
    output = {}
    for group, indices in groups.items():
        select = torch.cat([torch.arange(slices[i].start, slices[i].stop) for i in indices])
        local_f, local_k, local_s = f[select], k[select], s[select]
        local = {key: value[select] for key, value in deltas.items()}
        metrics = {key: directional_metrics(local_f, local_k, local_s, value) for key, value in local.items()}
        output[group] = {
            "parameters": len(indices),
            "forget_gradient_energy_fraction": float(torch.dot(local_f, local_f)) / total_f_energy,
            "update_norm_fraction": {
                key: float(torch.dot(value, value)) / total_update_energy[key]
                for key, value in local.items()
            },
            "counterfactuals": metrics,
            "weight_decay_contribution": directional_metrics(
                local_f, local_k, local_s, local["A1"] - local["A2"]
            ),
            "a0_to_a1_delta_norm_change": metrics["A1"]["delta_norm"] - metrics["A0"]["delta_norm"],
            "retain_violation_A1": any(
                metrics["A1"][name]["normalized"] is not None
                and metrics["A1"][name]["normalized"] > retain_tolerance
                for name in ("L_retain_KL", "L_sup")
            ),
        }
    return output


def _runtime_binding(runtime: dict[str, Any]) -> tuple[list[str], list[torch.nn.Parameter]]:
    named = [(name, value) for name, value in runtime["current"].named_parameters() if value.requires_grad]
    runtime_names = [name.replace(".default", "") for name, _ in named]
    serialized_names = list(runtime["payload"]["adapter_state"])
    if runtime_names != serialized_names:
        raise ValueError("runtime LoRA name/order binding differs from serialized adapter order")
    for (name, value), serialized_name in zip(named, serialized_names):
        stored = runtime["payload"]["adapter_state"][serialized_name]
        if value.shape != stored.shape or value.dtype != stored.dtype:
            raise ValueError(f"runtime LoRA shape/dtype binding mismatch: {name}")
    return [name for name, _ in named], [value for _, value in named]


def _flatten_gradients(
    components: dict[str, torch.Tensor], parameters: list[torch.nn.Parameter]
) -> dict[str, torch.Tensor]:
    gradients = {}
    for index, loss_name in enumerate(LOSSES):
        values = torch.autograd.grad(
            components[loss_name],
            parameters,
            retain_graph=index < len(LOSSES) - 1,
            allow_unused=True,
            create_graph=False,
        )
        gradients[loss_name] = torch.cat(
            [
                (torch.zeros_like(parameter) if value is None else value)
                .detach()
                .double()
                .cpu()
                .reshape(-1)
                for parameter, value in zip(parameters, values)
            ]
        )
    if any(parameter.grad is not None for parameter in parameters):
        raise RuntimeError("torch.autograd.grad populated authoritative .grad")
    return gradients


def _publish_full(root: Path, audit: dict[str, Any], run_name: str, pre: dict[str, Any]) -> dict[str, Any]:
    final = _resolve(root, Path(audit["output_root"]) / "full_runs" / _safe_name(run_name), output=True)
    if final.exists():
        raise FileExistsError(f"refusing to overwrite Full: {final}")
    stage = final.parent / f".{run_name}.{uuid.uuid4().hex[:10]}.stage"
    stage.mkdir(parents=True)
    runtime = None
    outer_rng = capture_rng()
    outer_rng_hash = rng_hashes(outer_rng)
    checkpoint = _resolve(root, audit["step812_checkpoint"])
    checkpoint_hash_before = sha256_file(checkpoint / "state.pt")
    checkpoint_manifest_hash_before = sha256_file(checkpoint / "manifest.json")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        runtime = _load_runtime(
            root,
            root / "configs" / "t5_step812_gradient_geometry_audit_v1.yaml",
            device,
        )
        names, parameters = _runtime_binding(runtime)
        validate_optimizer_mapping(runtime["payload"]["adapter_state"], runtime["payload"]["optimizer_state"], audit["expected"])
        student_before = _tensor_state_hash(runtime["current"])
        teacher_before = {
            "original": _tensor_state_hash(runtime["original"]),
            "augmented": _tensor_state_hash(runtime["augmented"]),
        }
        _restore_rng_payload(runtime["payload"]["rng"])
        rng_used = rng_hashes(capture_rng())
        runtime["current"].train()
        pair = catalog_pair(0, len(runtime["forget"]), len(runtime["retain"]), 16, 42)
        forget = move_batch(_batch(runtime["forget"], pair["forget_indices"]), device)
        retain = move_batch(_batch(runtime["retain"], pair["retain_indices"]), device)
        components = compute_components(
            runtime["current"], runtime["original"], runtime["augmented"], forget, retain, 2.0
        )
        gradients = _flatten_gradients(components, parameters)
        f, k, s = (gradients[name] for name in LOSSES)
        projection = direct_svd_projection(f, k, s, audit["projection"]["relative_singular_tolerance"])
        official_cpu = [value.detach().cpu() for value in parameters]
        state = runtime["payload"]["optimizer_state"]
        a0_raw = shadow_adamw_step(official_cpu, state, f)
        a1_raw = shadow_adamw_step(official_cpu, state, projection["safe"])
        a2_raw = shadow_adamw_step(official_cpu, state, projection["safe"], weight_decay_override=0.0)
        a3_delta = a1_raw["delta"] - projection["basis"] @ (projection["basis"].T @ a1_raw["delta"])
        directional = {
            "A0": directional_metrics(f, k, s, a0_raw["delta"]),
            "A1": directional_metrics(f, k, s, a1_raw["delta"]),
            "A2": directional_metrics(f, k, s, a2_raw["delta"]),
            "A3": directional_metrics(f, k, s, a3_delta),
        }
        effectiveness = forget_effectiveness(directional["A0"], directional["A1"])
        groups = grouped_diagnostics(
            names,
            [value.shape for value in parameters],
            f,
            k,
            s,
            {"A0": a0_raw["delta"], "A1": a1_raw["delta"], "A2": a2_raw["delta"], "A3": a3_delta},
            audit["classification"]["normalized_retain_violation_tolerance"],
        )
        student_after = _tensor_state_hash(runtime["current"])
        teacher_after = {
            "original": _tensor_state_hash(runtime["original"]),
            "augmented": _tensor_state_hash(runtime["augmented"]),
        }
        restore_rng(outer_rng)
        invariants = {
            "authoritative_parameters_unchanged": student_before == student_after,
            "authoritative_parameter_grads_none": all(value.grad is None for value in parameters),
            "teachers_unchanged": teacher_before == teacher_after,
            "checkpoint_unchanged": checkpoint_hash_before == sha256_file(checkpoint / "state.pt"),
            "checkpoint_manifest_unchanged": checkpoint_manifest_hash_before == sha256_file(checkpoint / "manifest.json"),
            "teacher_checkpoint_files_unchanged": all(
                sha256_file(Path(pre["teachers"][role]["path"])) == pre["teachers"][role]["sha256"]
                for role in ("original", "augmented")
            ),
            "outer_rng_unchanged": rng_hashes(capture_rng()) == outer_rng_hash,
            "batch_hash_exact": pair["batch_hash"] == EXPECTED_BATCH_HASH,
        }
        result = {
            "schema": SCHEMA,
            "run_name": run_name,
            "scope": "development-only zero-authoritative-update step813",
            "gradient_geometry": _gradient_geometry(f, k, s, projection),
            "counterfactuals": {
                "A0": _counterfactual_scalar("raw_forget_through_adamw", a0_raw, directional["A0"], True),
                "A1": _counterfactual_scalar("gradient_projected_forget_through_adamw", a1_raw, directional["A1"], True),
                "A2": _counterfactual_scalar("gradient_projected_adam_no_weight_decay", a2_raw, directional["A2"], False),
                "A3": _counterfactual_scalar(
                    "update_space_projection_reference",
                    {"delta_hash": _tensor_hash(a3_delta), "delta_norm": directional["A3"]["delta_norm"]},
                    directional["A3"],
                    False,
                ),
            },
            "forget_effectiveness": effectiveness,
            "groups": groups,
            "classification_inputs": {
                "A0": directional["A0"],
                "A1": directional["A1"],
                "A3": directional["A3"],
                "invariants_valid": all(invariants.values()),
            },
            "gradient_vector_hashes": {name: _tensor_hash(value) for name, value in gradients.items()},
            "parameter_order": {
                "tensors": len(names),
                "parameters": sum(value.numel() for value in parameters),
                "runtime_name_order_sha256": canonical_hash([name.replace(".default", "") for name in names]),
                "runtime_shape_dtype_sha256": canonical_hash(
                    [[name.replace(".default", ""), list(value.shape), str(value.dtype)] for name, value in zip(names, parameters)]
                ),
            },
            "rng": {"step813_rng_hash": rng_used, "outer_rng_restored_on_exit": True},
            "invariants": invariants,
            "authoritative_optimizer_constructed": False,
            "authoritative_optimizer_steps_executed": 0,
            "authoritative_parameters_updated": False,
            "shadow_optimizer_constructed": True,
            "shadow_optimizer_steps_executed": 3,
            "shadow_parameters_updated": all(
                item["shadow_parameters_updated"] for item in (a0_raw, a1_raw, a2_raw)
            ),
            "shadow_updates_committed": False,
            "gradient_vectors_persisted": False,
            "parameter_delta_vectors_persisted": False,
            "optimizer_state_persisted": False,
            "logits_persisted": False,
            "tokens_persisted": False,
            "raw_samples_persisted": False,
            "retrain_loaded": False,
            "test_loader_built": False,
            "test_accessed": False,
        }
        contract = build_authority_contract(
            pre, audit["_sha256"], run_name, audit["classification"]
        )
        provenance = {
            "schema": SCHEMA,
            "git": pre["git"],
            "implementation": pre["implementation"],
            "checkpoint": pre["checkpoint"],
            "teachers": pre["teachers"],
            "data": pre["data"],
            "lineage": pre["lineage"],
            "tokenizer": pre["tokenizer"],
            "retrain_loaded": False,
            "test_accessed": False,
        }
        del gradients, f, k, s, a0_raw, a1_raw, a2_raw, a3_delta, projection, components, forget, retain
        atomic_json(stage / "contract.json", contract)
        atomic_json(stage / "audit.json", result)
        atomic_json(stage / "provenance.json", provenance)
        manifest = {
            "schema": SCHEMA,
            "contract_sha256": sha256_file(stage / "contract.json"),
            "audit_sha256": sha256_file(stage / "audit.json"),
            "provenance_sha256": sha256_file(stage / "provenance.json"),
            "published_atomically": True,
            "authoritative_optimizer_steps_executed": 0,
            "shadow_optimizer_steps_executed": 3,
            "test_accessed": False,
        }
        atomic_json(stage / "manifest.json", manifest)
        atomic_text(stage / "COMPLETED", "FULL_AUDIT_COMPLETED\n")
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, final)
        return {**result, "run_dir": str(final)}
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        restore_rng(outer_rng)
        if rng_hashes(capture_rng()) != rng_hashes(outer_rng):
            raise RuntimeError("Full failed to restore outer RNG")
        if runtime is not None:
            del runtime
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def validate_publication_binding(
    contract: dict[str, Any],
    provenance: dict[str, Any],
    pre: dict[str, Any],
    config_sha256: str,
    run_name: str,
    classification: dict[str, Any],
) -> None:
    expected = build_authority_contract(
        pre, config_sha256, run_name, classification
    )
    if json_native(contract) != contract:
        raise ValueError("Full contract is not JSON-native")
    checks = (
        ("schema", "Full schema contract mismatch"),
        ("run_name", "Full RunName contract mismatch"),
        ("config_sha256", "Full config SHA mismatch"),
        ("git", "Full HEAD contract mismatch"),
        ("implementation", "Full implementation contract mismatch"),
        ("checkpoint", "Full checkpoint contract mismatch"),
        ("optimizer_mapping", "Full optimizer_mapping contract mismatch"),
        ("step813", "Full step813 contract mismatch"),
        ("classification", "Full classification contract mismatch"),
        ("test_accessed", "Full test-access contract mismatch"),
    )
    for field, message in checks:
        if contract.get(field) != expected[field]:
            raise ValueError(message)
    if provenance.get("git") != expected["git"]:
        raise ValueError("Full provenance HEAD mismatch")
    if provenance.get("implementation") != expected["implementation"]:
        raise ValueError("Full provenance implementation mismatch")


def verify_full(root: Path, config_path: Path, run_name: str) -> dict[str, Any]:
    audit = load_audit_config(config_path, root)
    pre = preflight(root, config_path)
    require_clean_git(pre["git"], "Analyze")
    source = _resolve(root, Path(audit["output_root"]) / "full_runs" / _safe_name(run_name), output=True)
    required = {"contract.json", "audit.json", "provenance.json", "manifest.json", "COMPLETED"}
    if not source.is_dir() or {item.name for item in source.iterdir()} != required:
        raise ValueError("Full artifact inventory mismatch")
    if (source / "COMPLETED").read_text(encoding="utf-8") != "FULL_AUDIT_COMPLETED\n":
        raise ValueError("Full completion marker mismatch")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    hashes = {
        "contract_sha256": sha256_file(source / "contract.json"),
        "audit_sha256": sha256_file(source / "audit.json"),
        "provenance_sha256": sha256_file(source / "provenance.json"),
    }
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("published_atomically") is not True
        or manifest.get("test_accessed") is not False
        or manifest.get("authoritative_optimizer_steps_executed") != 0
        or manifest.get("shadow_optimizer_steps_executed") != 3
        or any(manifest.get(key) != value for key, value in hashes.items())
    ):
        raise ValueError("Full manifest/SHA/safety mismatch")
    contract = json.loads((source / "contract.json").read_text(encoding="utf-8"))
    provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    result = json.loads((source / "audit.json").read_text(encoding="utf-8"))
    validate_publication_binding(
        contract,
        provenance,
        pre,
        verified_config_sha(config_path),
        run_name,
        audit["classification"],
    )
    safety = {
        "authoritative_optimizer_steps_executed": 0,
        "authoritative_parameters_updated": False,
        "shadow_optimizer_constructed": True,
        "shadow_optimizer_steps_executed": 3,
        "shadow_updates_committed": False,
        "gradient_vectors_persisted": False,
        "parameter_delta_vectors_persisted": False,
        "optimizer_state_persisted": False,
        "logits_persisted": False,
        "tokens_persisted": False,
        "raw_samples_persisted": False,
        "retrain_loaded": False,
        "test_accessed": False,
    }
    if any(result.get(key) != value for key, value in safety.items()):
        raise ValueError("Full safety fields mismatch")
    return {"source": source, "audit": audit, "preflight": pre, "result": result, "manifest": manifest}


def verified_config_sha(config_path: Path) -> str:
    return sha256_file(config_path.resolve())


def verify_analysis_publication(path: Path, source_manifest_sha256: str) -> dict[str, Any]:
    required = {"analysis.json", "manifest.json", "COMPLETED"}
    if not path.is_dir() or {item.name for item in path.iterdir()} != required:
        raise ValueError("Analyze artifact inventory mismatch")
    if (path / "COMPLETED").read_text(encoding="utf-8") != "ANALYSIS_COMPLETED\n":
        raise ValueError("Analyze completion marker mismatch")
    result = json.loads((path / "analysis.json").read_text(encoding="utf-8"))
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != ANALYSIS_SCHEMA
        or manifest.get("published_atomically") is not True
        or manifest.get("analysis_sha256") != sha256_file(path / "analysis.json")
        or manifest.get("source_manifest_sha256") != source_manifest_sha256
        or manifest.get("test_accessed") is not False
        or result.get("authoritative_optimizer_constructed") is not False
        or result.get("authoritative_optimizer_steps_executed") != 0
        or result.get("authoritative_parameters_updated") is not False
        or result.get("shadow_optimizer_constructed") is not True
        or result.get("shadow_optimizer_steps_executed") != 3
        or result.get("shadow_updates_committed") is not False
        or result.get("test_accessed") is not False
    ):
        raise ValueError("Analyze manifest/SHA/safety mismatch")
    return result


def analyze(root: Path, config_path: Path, run_name: str) -> dict[str, Any]:
    verified = verify_full(root, config_path, run_name)
    audit = verified["audit"]
    inputs = verified["result"]["classification_inputs"]
    classification = classify_optimizer_audit(
        inputs["A0"],
        inputs["A1"],
        inputs["A3"],
        invariants_valid=inputs["invariants_valid"],
        retain_tolerance=audit["classification"]["normalized_retain_violation_tolerance"],
        forget_zero_tolerance=audit["classification"]["forget_descent_zero_tolerance"],
        effectiveness_min=audit["classification"]["forget_effectiveness_min"],
    )
    final = _resolve(root, Path(audit["output_root"]) / "analysis_runs" / _safe_name(run_name), output=True)
    if final.exists():
        raise FileExistsError(f"refusing to overwrite Analyze: {final}")
    stage = final.parent / f".{run_name}.{uuid.uuid4().hex[:10]}.stage"
    stage.mkdir(parents=True)
    result = {
        "schema": ANALYSIS_SCHEMA,
        "run_name": run_name,
        **classification,
        "forget_effectiveness": verified["result"]["forget_effectiveness"],
        "counterfactuals": verified["result"]["counterfactuals"],
        "source_full": str(verified["source"]),
        "source_manifest_sha256": sha256_file(verified["source"] / "manifest.json"),
        "git": verified["preflight"]["git"],
        "implementation": verified["preflight"]["implementation"],
        "authoritative_optimizer_constructed": False,
        "authoritative_optimizer_steps_executed": 0,
        "authoritative_parameters_updated": False,
        "shadow_optimizer_constructed": True,
        "shadow_optimizer_steps_executed": 3,
        "shadow_updates_committed": False,
        "test_accessed": False,
    }
    atomic_json(stage / "analysis.json", result)
    atomic_json(
        stage / "manifest.json",
        {
            "schema": ANALYSIS_SCHEMA,
            "analysis_sha256": sha256_file(stage / "analysis.json"),
            "source_manifest_sha256": result["source_manifest_sha256"],
            "published_atomically": True,
            "test_accessed": False,
        },
    )
    atomic_text(stage / "COMPLETED", "ANALYSIS_COMPLETED\n")
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage, final)
    verified = verify_analysis_publication(final, result["source_manifest_sha256"])
    return {**verified, "analysis_dir": str(final)}


def _toy_optimizer_state(parameter: torch.Tensor, *, weight_decay: float, exp_avg: torch.Tensor | None = None) -> dict[str, Any]:
    average = torch.zeros_like(parameter) if exp_avg is None else exp_avg.clone()
    return {
        "state": {0: {"step": torch.tensor(1.0), "exp_avg": average, "exp_avg_sq": torch.ones_like(parameter)}},
        "param_groups": [
            {
                "lr": 0.01,
                "betas": (0.9, 0.999),
                "eps": 1.0e-8,
                "weight_decay": weight_decay,
                "amsgrad": False,
                "maximize": False,
                "foreach": None,
                "capturable": False,
                "differentiable": False,
                "fused": None,
                "decoupled_weight_decay": True,
                "params": [0],
            }
        ],
    }


def synthetic_dry_run(root: Path, config_path: Path, run_name: str) -> dict[str, Any]:
    audit = load_audit_config(config_path, root)
    final = _resolve(root, Path(audit["output_root"]) / "synthetic_runs" / _safe_name(run_name), output=True)
    if final.exists():
        raise FileExistsError(f"refusing to overwrite SyntheticDryRun: {final}")
    parameter = torch.tensor([0.0, -10.0])
    f = torch.tensor([1.0, 0.0], dtype=torch.float64)
    k = torch.tensor([0.0, 1.0], dtype=torch.float64)
    s = torch.tensor([0.0, 2.0], dtype=torch.float64)
    projection = direct_svd_projection(f, k, s)
    state = _toy_optimizer_state(parameter, weight_decay=0.1)
    a0 = shadow_adamw_step([parameter], state, f)
    a1 = shadow_adamw_step([parameter], state, projection["safe"])
    a2 = shadow_adamw_step([parameter], state, projection["safe"], weight_decay_override=0.0)
    a3_delta = a1["delta"] - projection["basis"] @ (projection["basis"].T @ a1["delta"])
    metrics = {
        "A0": directional_metrics(f, k, s, a0["delta"]),
        "A1": directional_metrics(f, k, s, a1["delta"]),
        "A2": directional_metrics(f, k, s, a2["delta"]),
        "A3": directional_metrics(f, k, s, a3_delta),
    }
    classification = classify_optimizer_audit(
        metrics["A0"], metrics["A1"], metrics["A3"], invariants_valid=True
    )
    result = {
        "schema": SCHEMA,
        "mode": "SyntheticDryRun",
        "case": "weight_decay_breaks_retain_orthogonality",
        "projection": {key: value for key, value in projection.items() if key not in {"safe", "basis"}},
        "directional": metrics,
        "classification": classification,
        "authoritative_optimizer_constructed": False,
        "authoritative_optimizer_steps_executed": 0,
        "authoritative_parameters_updated": False,
        "shadow_optimizer_constructed": True,
        "shadow_optimizer_steps_executed": 3,
        "shadow_parameters_updated": True,
        "shadow_updates_committed": False,
        "model_loaded": False,
        "gradient_vectors_persisted": False,
        "parameter_delta_vectors_persisted": False,
        "optimizer_state_persisted": False,
        "test_accessed": False,
    }
    stage = final.parent / f".{run_name}.{uuid.uuid4().hex[:10]}.stage"
    stage.mkdir(parents=True)
    atomic_json(stage / "synthetic_audit.json", result)
    atomic_json(
        stage / "manifest.json",
        {
            "schema": SCHEMA,
            "artifact_sha256": sha256_file(stage / "synthetic_audit.json"),
            "published_atomically": True,
            "test_accessed": False,
        },
    )
    atomic_text(stage / "COMPLETED", "SYNTHETIC_DRY_RUN_COMPLETED\n")
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage, final)
    return {**result, "run_dir": str(final)}


def main() -> None:
    parser = argparse.ArgumentParser(description="T5 step813 optimizer-aware direction audit")
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
            pre = preflight(root, config)
            require_clean_git(pre["git"], "Full")
            result = _publish_full(root, load_audit_config(config, root), args.run_name, pre)
        else:
            result = analyze(root, config, args.run_name)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
