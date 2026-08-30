from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from transformers import T5Tokenizer

from src.diagnostics.t5_full_runner import (
    _batch,
    _budget_object,
    _load_checkpoint,
    _rng_payload,
    _restore_rng_payload,
    _verify_next_batch,
    evaluate_overall_validation,
    batch_index_hash,
    batch_indices,
    component_loss_values,
)
from src.diagnostics.t5_reconstructed_official import (
    JsonPromptDataset,
    build_current_model,
    compute_components,
    freeze_teacher,
    load_config,
    load_legacy_model,
    make_optimizer,
    move_batch,
    sha256_file,
)
from src.diagnostics.t5_trajectory_diagnostics import (
    capture_rng,
    isolated_component_shadow,
    named_gradient_hashes,
    named_parameter_hashes,
    optimizer_hash,
    restore_rng,
    rng_hashes,
    _teacher_scalars,
)

EXPERIMENT = "t5_e2urec_joint_ablation_v1"
SCHEMA = "t5-e2urec-joint-ablation-v1"
PARENT_STEP = 812
FIRST_STEP = 813
TARGET_STEP = 1200
JOINT_STEPS = 388
FRAMEWORK_VALIDATION_NAME = "framework_validation"
CHECKPOINT_STEPS = (813, 850, 900, 1000, 1200)
EVALUATION_STEPS = (812, 813, 850, 900, 1000, 1200)


@dataclass(frozen=True)
class Branch:
    name: str
    formula: str
    active_components: tuple[str, ...]
    disabled_components: tuple[str, ...]
    change_from_j0: str


BRANCHES = {
    "j0_original_joint_reference": Branch(
        "j0_original_joint_reference",
        "0.6 * (L_sup + L_retain_KL) + 0.4 * L_forget",
        ("L_forget", "L_sup", "L_retain_KL"),
        (),
        "none; exact reconstructed-official joint objective",
    ),
    "j1_supervised_only_remember": Branch(
        "j1_supervised_only_remember",
        "0.6 * L_sup + 0.4 * L_forget",
        ("L_forget", "L_sup"),
        ("L_retain_KL",),
        "disable only L_retain_KL",
    ),
    "j2_kl_only_remember": Branch(
        "j2_kl_only_remember",
        "0.6 * L_retain_KL + 0.4 * L_forget",
        ("L_forget", "L_retain_KL"),
        ("L_sup",),
        "disable only L_sup",
    ),
    "j3_forget_only_control": Branch(
        "j3_forget_only_control",
        "L_forget",
        ("L_forget",),
        ("L_sup", "L_retain_KL"),
        "disable both remembering components; diagnostic control, not a fair single-factor method",
    ),
}


def branch_loss(
    components: dict[str, torch.Tensor], branch_name: str
) -> torch.Tensor:
    if branch_name == "j0_original_joint_reference":
        return 0.6 * (
            components["L_sup"] + components["L_retain_KL"]
        ) + 0.4 * components["L_forget"]
    if branch_name == "j1_supervised_only_remember":
        return 0.6 * components["L_sup"] + 0.4 * components["L_forget"]
    if branch_name == "j2_kl_only_remember":
        return 0.6 * components["L_retain_KL"] + 0.4 * components["L_forget"]
    if branch_name == "j3_forget_only_control":
        return components["L_forget"]
    raise ValueError(f"unknown joint ablation branch: {branch_name}")


def component_status(
    branch_name: str, losses: dict[str, float], shadow: dict[str, Any] | None
) -> dict[str, Any]:
    branch = BRANCHES[branch_name]
    result = {}
    for name in ("L_forget", "L_sup", "L_retain_KL"):
        active = name in branch.active_components
        result[name] = {
            "active": active,
            "loss": losses[name] if active else 0.0,
            "contribution": (
                losses[name]
                * (
                    1.0
                    if branch_name == "j3_forget_only_control"
                    else (0.4 if name == "L_forget" else 0.6)
                )
                if active
                else 0.0
            ),
            "gradient_norm": (
                shadow["component_gradient_norms"][name]
                if active and shadow is not None
                else 0.0
            ),
            "reason": None if active else "component_disabled_by_ablation",
        }
    return result


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _tensor_snapshot(
    values: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in values.items()}


def _trainable_parameters(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in model.named_parameters()
        if value.requires_grad
    }


def _recursive_exact(left: Any, right: Any, path: str = "") -> list[str]:
    errors: list[str] = []
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(
            left.cpu(), right.cpu()
        ):
            errors.append(path)
    elif isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            errors.append(f"{path}.keys")
        else:
            for key in left:
                errors.extend(_recursive_exact(left[key], right[key], f"{path}.{key}"))
    elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            errors.append(f"{path}.length")
        else:
            for index, (a, b) in enumerate(zip(left, right)):
                errors.extend(_recursive_exact(a, b, f"{path}[{index}]"))
    elif left != right:
        errors.append(path)
    return errors


def _tensorwise_compare(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> dict[str, Any]:
    if left.keys() != right.keys():
        return {"exact": False, "key_mismatch": True}
    errors = {}
    hashes = {}
    exact = True
    for key in left:
        a, b = left[key], right[key]
        equal = (
            a.shape == b.shape
            and a.dtype == b.dtype
            and torch.equal(a.cpu(), b.cpu())
        )
        exact &= equal
        errors[key] = (
            0.0
            if equal
            else float((a.double() - b.double()).abs().max())
        )
        hashes[key] = {
            "left": _tensor_hash(a),
            "right": _tensor_hash(b),
            "equal": equal,
        }
    return {
        "exact": exact,
        "key_mismatch": False,
        "max_absolute_error": max(errors.values(), default=0.0),
        "tensor_errors": errors,
        "tensor_hashes": hashes,
    }


def _step_data(
    config: dict[str, Any],
    forget_dataset: JsonPromptDataset,
    retain_dataset: JsonPromptDataset,
    step: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], str]:
    budget = _budget_object(config)
    epoch = (step - 1) // budget.forget_batches
    position = (step - 1) % budget.forget_batches
    forget_indices = batch_indices(
        len(forget_dataset),
        config["training"]["per_device_batch_size"],
        config["training"]["seed"],
        epoch,
        position,
        0,
    )
    retain_indices = batch_indices(
        len(retain_dataset),
        config["training"]["per_device_batch_size"],
        config["training"]["seed"],
        epoch - config["training"]["forget_epoch"],
        position,
        10_000,
    )
    return (
        _batch(forget_dataset, forget_indices),
        _batch(retain_dataset, retain_indices),
        batch_index_hash(forget_indices, retain_indices),
    )


def _parent_checkpoint(config: dict[str, Any]) -> Path:
    path = (
        Path(config["paths"]["output_root"]).resolve()
        / "full_runs"
        / "t5_reconstructed_official_seed42_v2"
        / "checkpoints"
        / "step_00812"
    )
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("step") != PARENT_STEP:
        raise ValueError("joint ablation parent must be step812")
    if manifest["state_sha256"] != sha256_file(path / "state.pt"):
        raise ValueError("step812 parent checkpoint hash mismatch")
    return path


def _source_run(config: dict[str, Any]) -> Path:
    return (
        Path(config["paths"]["output_root"]).resolve()
        / "full_runs"
        / "t5_reconstructed_official_seed42_v2"
    )


def _git_head(project_root: Path) -> str:
    head = (project_root / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        return (project_root / ".git" / head[5:]).read_text(encoding="utf-8").strip()
    return head


def experiment_contract(
    config: dict[str, Any],
    project_root: Path,
    branch_name: str,
) -> dict[str, Any]:
    branch = BRANCHES[branch_name]
    parent = _parent_checkpoint(config)
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "branch": branch.name,
        "formula": branch.formula,
        "active_components": list(branch.active_components),
        "disabled_components": list(branch.disabled_components),
        "change_from_j0": branch.change_from_j0,
        "parent_checkpoint": str(parent),
        "parent_checkpoint_sha256": sha256_file(parent / "state.pt"),
        "parent_step": PARENT_STEP,
        "first_optimizer_step": FIRST_STEP,
        "target_step": TARGET_STEP,
        "joint_optimizer_steps": JOINT_STEPS,
        "source_checkpoint_hashes": {
            role: sha256_file(Path(config["paths"][key]))
            for role, key in (
                ("original", "original"),
                ("augmented", "augmented_teacher"),
                ("retrain", "retrain_reference"),
            )
        },
        "data_hashes": {
            role: sha256_file(Path(config["paths"][role]))
            for role in ("forget", "retain", "validation")
        },
        "optimizer": "AdamW",
        "learning_rate": config["training"]["learning_rate"],
        "batch_size": config["training"]["per_device_batch_size"],
        "gradient_accumulation": config["training"]["gradient_accumulation"],
        "alpha": config["training"]["alpha"],
        "remember_weight": config["training"]["code_weight"],
        "seed": config["training"]["seed"],
        "sampler": config["training"]["sampler"],
        "scheduler": config["training"]["scheduler"],
        "finite_gradient_clipping": config["training"]["finite_gradient_clipping"],
        "lora": config["lora"],
        "attention_implementation": "eager",
        "validation_only": True,
        "test_used": False,
        "git_commit": _git_head(project_root),
    }


def validate_experiment_contract(
    config: dict[str, Any],
    project_root: Path,
    branch_name: str,
    saved: dict[str, Any],
) -> None:
    expected = experiment_contract(config, project_root, branch_name)
    if saved != expected:
        differing = [
            key
            for key in sorted(set(expected) | set(saved))
            if expected.get(key) != saved.get(key)
        ]
        raise ValueError(f"joint ablation Resume contract mismatch: {differing}")


def resolve_ablation_run(
    project_root: Path, branch_name: str, run_name: str, resume: bool
) -> Path:
    if branch_name not in BRANCHES:
        raise ValueError("unknown branch")
    if not run_name or Path(run_name).name != run_name:
        raise ValueError("run name must be one explicit path component")
    path = (
        project_root
        / "outputs"
        / "t5_e2urec_joint_ablation_v1"
        / "full_runs"
        / branch_name
        / run_name
    ).resolve()
    protected = (project_root / "outputs" / "t5_e2urec_diagnostics_v1").resolve()
    if path == protected or protected in path.parents:
        raise ValueError("ablation output overlaps original diagnostics")
    if resume:
        if not path.is_dir():
            raise FileNotFoundError("ablation Resume directory missing")
    else:
        if path.exists() and any(path.iterdir()):
            raise FileExistsError("refusing to overwrite existing ablation run")
        path.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_ablation_checkpoint(
    run_dir: Path,
    step: int,
    current: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any],
    contract: dict[str, Any],
) -> Path:
    root = run_dir / "checkpoints"
    root.mkdir(exist_ok=True)
    destination = root / f"step_{step:05d}"
    if destination.exists():
        raise FileExistsError(f"checkpoint already exists: {destination}")
    temporary = root / f".step_{step:05d}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        payload = {
            "schema": SCHEMA,
            "contract": contract,
            "adapter_state": {
                key: value.detach().cpu()
                for key, value in get_peft_model_state_dict(current).items()
            },
            "optimizer_state": optimizer.state_dict(),
            "state": state,
            "rng": _rng_payload(),
            "rng_hash": rng_hashes(capture_rng()),
            "test_accessed": False,
        }
        torch.save(payload, temporary / "state.pt")
        _atomic_json(
            temporary / "manifest.json",
            {
                "schema": SCHEMA,
                "step": step,
                "state_sha256": sha256_file(temporary / "state.pt"),
                "published_atomically": True,
            },
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _latest_ablation_checkpoint(run_dir: Path) -> Path:
    if (run_dir / "run_state.json").is_file():
        status = json.loads(
            (run_dir / "run_state.json").read_text(encoding="utf-8")
        ).get("status")
        if status == "FAILED":
            raise ValueError("cannot Resume FAILED ablation run")
    candidates = sorted((run_dir / "checkpoints").glob("step_*"))
    if not candidates:
        raise FileNotFoundError("no ablation checkpoint available")
    checkpoint = candidates[-1]
    manifest = json.loads(
        (checkpoint / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest["state_sha256"] != sha256_file(checkpoint / "state.pt"):
        raise ValueError("ablation checkpoint hash mismatch")
    return checkpoint


def _load_ablation_checkpoint(
    checkpoint: Path,
    config: dict[str, Any],
    project_root: Path,
    branch_name: str,
    current: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    payload = torch.load(
        checkpoint / "state.pt", map_location="cpu", weights_only=False
    )
    if payload.get("schema") != SCHEMA:
        raise ValueError("ablation checkpoint schema mismatch")
    validate_experiment_contract(
        config, project_root, branch_name, payload["contract"]
    )
    result = set_peft_model_state_dict(current, payload["adapter_state"])
    if getattr(result, "unexpected_keys", []):
        raise ValueError("ablation adapter unexpected keys")
    optimizer.load_state_dict(payload["optimizer_state"])
    _restore_rng_payload(payload["rng"])
    if rng_hashes(capture_rng()) != payload["rng_hash"]:
        raise ValueError("ablation Resume RNG mismatch")
    return payload["state"]


def _bernoulli_jsd(left: np.ndarray, right: np.ndarray) -> float:
    epsilon = 1e-12
    left = np.clip(left, epsilon, 1 - epsilon)
    right = np.clip(right, epsilon, 1 - epsilon)
    middle = (left + right) / 2
    kl_left = left * np.log(left / middle) + (1 - left) * np.log(
        (1 - left) / (1 - middle)
    )
    kl_right = right * np.log(right / middle) + (1 - right) * np.log(
        (1 - right) / (1 - middle)
    )
    return float(np.mean((kl_left + kl_right) / 2))


def _legacy_symmetric_kl(left: np.ndarray, right: np.ndarray) -> float:
    epsilon = 1e-12
    left = np.clip(left, epsilon, 1)
    right = np.clip(right, epsilon, 1)
    return float(
        np.mean(left * np.log(left / right) + right * np.log(right / left))
    )


def _safe_correlation(function, left: np.ndarray, right: np.ndarray) -> float | None:
    value = function(left, right).statistic
    return float(value) if np.isfinite(value) else None


def development_metrics(
    prediction: dict[str, Any],
    original: dict[str, Any],
    retrain: dict[str, Any],
) -> dict[str, Any]:
    probability = np.asarray(prediction["probabilities"], dtype=float)
    original_probability = np.asarray(original["probabilities"], dtype=float)
    retrain_probability = np.asarray(retrain["probabilities"], dtype=float)
    gold = np.asarray(prediction["gold"], dtype=int)
    if (
        prediction["sample_ids"] != original["sample_ids"]
        or prediction["sample_ids"] != retrain["sample_ids"]
        or prediction["gold"] != original["gold"]
        or prediction["gold"] != retrain["gold"]
    ):
        raise ValueError("development prediction sample/gold mismatch")
    predicted = probability >= 0.5

    def relative(reference: np.ndarray) -> dict[str, Any]:
        return {
            "l2_rms": float(np.sqrt(np.mean((probability - reference) ** 2))),
            "standard_jsd": _bernoulli_jsd(probability, reference),
            "legacy_symmetric_kl": _legacy_symmetric_kl(probability, reference),
            "prediction_agreement": float(
                np.mean(predicted == (reference >= 0.5))
            ),
        }

    change = probability - original_probability
    target = retrain_probability - original_probability
    return {
        "scope": "overall_validation_only",
        "auc": float(roc_auc_score(gold, probability)),
        "accuracy": float(accuracy_score(gold, predicted)),
        "log_loss": float(log_loss(gold, probability, labels=[0, 1])),
        "probability": {
            "mean": float(probability.mean()),
            "std": float(probability.std()),
            "min": float(probability.min()),
            "max": float(probability.max()),
        },
        "positive_rate": float(predicted.mean()),
        "mean_confidence": float(np.maximum(probability, 1 - probability).mean()),
        "relative_original": relative(original_probability),
        "relative_retrain": relative(retrain_probability),
        "retrain_direction": {
            "sign_agreement": float(np.mean(np.sign(change) == np.sign(target))),
            "pearson": _safe_correlation(pearsonr, change, target),
            "spearman": _safe_correlation(spearmanr, change, target),
        },
        "mean_absolute_change_from_original": float(np.abs(change).mean()),
        "samples": len(gold),
        "sample_order_hash": prediction["sample_order_hash"],
        "test_accessed": False,
    }


def _historical_step813(config: dict[str, Any]) -> Path:
    return (
        Path(config["paths"]["output_root"]).resolve()
        / "full_runs"
        / "t5_reconstructed_official_seed42_v2"
        / "checkpoints"
        / "step_00813"
    )


def _arm(
    config: dict[str, Any],
    branch_name: str,
    instrumented: bool,
    output: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(config["paths"]["model_dir"])
    forget_dataset = JsonPromptDataset(Path(config["paths"]["forget"]), tokenizer)
    retain_dataset = JsonPromptDataset(Path(config["paths"]["retain"]), tokenizer)
    current = build_current_model(
        Path(config["paths"]["original"]), config["lora"]
    ).to(device)
    original = freeze_teacher(
        load_legacy_model(Path(config["paths"]["original"]))
    ).to(device)
    augmented = freeze_teacher(
        load_legacy_model(Path(config["paths"]["augmented_teacher"]))
    ).to(device)
    optimizer = make_optimizer(current, config["training"]["learning_rate"])
    parent = _parent_checkpoint(config)
    state, compatibility = _load_checkpoint(parent, config, current, optimizer)
    if state["step"] != PARENT_STEP or state["next_optimizer_step"] != FIRST_STEP:
        raise ValueError("parent does not resume exactly at step813")
    _verify_next_batch(state, config, state["next_batch_hash"])
    provenance = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "branch": branch_name,
        "formula": BRANCHES[branch_name].formula,
        "active_components": list(BRANCHES[branch_name].active_components),
        "disabled_components": list(BRANCHES[branch_name].disabled_components),
        "change_from_j0": BRANCHES[branch_name].change_from_j0,
        "parent_checkpoint": str(parent),
        "parent_checkpoint_sha256": sha256_file(parent / "state.pt"),
        "parent_step": PARENT_STEP,
        "first_optimizer_step": FIRST_STEP,
        "target_step": 814,
        "validation_only": True,
        "test_used": False,
        "instrumented": instrumented,
    }
    _atomic_json(output / "pretrain_provenance.json", provenance)
    records = []
    for step in (813, 814):
        forget_cpu, retain_cpu, order_hash = _step_data(
            config, forget_dataset, retain_dataset, step
        )
        forget = move_batch(forget_cpu, device)
        retain = move_batch(retain_cpu, device)
        current.train()
        pre_rng = capture_rng()
        optimizer.zero_grad()
        components = compute_components(
            current,
            original,
            augmented,
            forget,
            retain,
            config["training"]["alpha"],
        )
        loss = branch_loss(components, branch_name)
        loss.backward()
        shadow = None
        if instrumented:
            shadow = isolated_component_shadow(
                current,
                optimizer,
                pre_rng,
                lambda: compute_components(
                    current,
                    original,
                    augmented,
                    forget,
                    retain,
                    config["training"]["alpha"],
                ),
                phase="joint",
            )
        gradients = _tensor_snapshot(
            {
                name: parameter.grad
                for name, parameter in _trainable_parameters(current).items()
                if parameter.grad is not None
            }
        )
        losses = component_loss_values(components, loss)
        optimizer.step()
        parameters = _tensor_snapshot(_trainable_parameters(current))
        state["step"] = step
        state["executed_optimizer_steps"] = step
        state["epoch"] = 1
        state["epoch_batch_position"] = step - 812
        state["next_optimizer_step"] = step + 1
        state["next_batch_hash"] = _verify_next_batch(state, config, None)
        record = {
            "step": step,
            "sample_ids": {
                "forget": forget_cpu["sample_id"].tolist(),
                "retain": retain_cpu["sample_id"].tolist(),
            },
            "batch_order_hash": order_hash,
            "losses": losses,
            "gradients": gradients,
            "gradient_hashes": {key: _tensor_hash(value) for key, value in gradients.items()},
            "parameters": parameters,
            "parameter_hashes": {key: _tensor_hash(value) for key, value in parameters.items()},
            "optimizer_state": copy.deepcopy(optimizer.state_dict()),
            "optimizer_hash": optimizer_hash(optimizer),
            "rng": capture_rng(),
            "rng_hashes": rng_hashes(capture_rng()),
            "next_optimizer_step": state["next_optimizer_step"],
            "next_batch_hash": state["next_batch_hash"],
            "model_training": current.training,
            "attention_implementation": current.config._attn_implementation_internal,
            "shadow": shadow,
            "test_loader_built": False,
            "test_accessed": False,
        }
        records.append(record)
    final = {
        "adapter_state": _tensor_snapshot(_trainable_parameters(current)),
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "rng": capture_rng(),
        "state": copy.deepcopy(state),
        "compatibility": compatibility,
    }
    del current, original, augmented, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records, final


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"gradients", "parameters", "optimizer_state", "rng"}
    }


def _compare_steps(
    canonical: list[dict[str, Any]], instrumented: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result = []
    for left, right in zip(canonical, instrumented):
        fields = {
            "step": left["step"] == right["step"],
            "sample_ids": left["sample_ids"] == right["sample_ids"],
            "batch_order_hash": left["batch_order_hash"] == right["batch_order_hash"],
            "losses": left["losses"] == right["losses"],
            "gradient_tensors": _tensorwise_compare(left["gradients"], right["gradients"]),
            "parameter_tensors": _tensorwise_compare(left["parameters"], right["parameters"]),
            "optimizer_state": not _recursive_exact(
                left["optimizer_state"], right["optimizer_state"], "optimizer"
            ),
            "optimizer_hash": left["optimizer_hash"] == right["optimizer_hash"],
            "rng": left["rng_hashes"] == right["rng_hashes"],
            "next_optimizer_step": left["next_optimizer_step"]
            == right["next_optimizer_step"],
            "next_batch_hash": left["next_batch_hash"] == right["next_batch_hash"],
            "model_mode": left["model_training"] == right["model_training"],
            "attention": left["attention_implementation"]
            == right["attention_implementation"]
            == "eager",
        }
        exact = all(
            value["exact"] if isinstance(value, dict) else value
            for value in fields.values()
        )
        result.append({"step": left["step"], "fields": fields, "exact": exact})
    return result


def _historical_anchor(
    config: dict[str, Any], canonical_step813: dict[str, Any]
) -> dict[str, Any]:
    checkpoint = _historical_step813(config)
    payload = torch.load(
        checkpoint / "state.pt", map_location="cpu", weights_only=False
    )
    historical_adapter = payload["adapter_state"]
    canonical_adapter = canonical_step813["parameters"]
    # PEFT checkpoint keys omit the runtime ".default" adapter segment.
    normalized = {
        key.replace(".default", ""): value
        for key, value in canonical_adapter.items()
    }
    adapter = _tensorwise_compare(historical_adapter, normalized)
    optimizer_errors = _recursive_exact(
        payload["optimizer_state"],
        canonical_step813["optimizer_state"],
        "optimizer",
    )
    fields = {
        "adapter": adapter,
        "optimizer_state": not optimizer_errors,
        "rng": payload["rng_hash"] == canonical_step813["rng_hashes"],
        "step": payload["state"]["step"] == canonical_step813["step"] == 813,
        "next_optimizer_step": payload["state"]["next_optimizer_step"]
        == canonical_step813["next_optimizer_step"],
        "next_batch_hash": payload["state"]["next_batch_hash"]
        == canonical_step813["next_batch_hash"],
    }
    exact = all(
        value["exact"] if isinstance(value, dict) else value
        for value in fields.values()
    )
    return {
        "checkpoint": str(checkpoint),
        "fields": fields,
        "optimizer_mismatches": optimizer_errors,
        "exact": exact,
    }


def run_paired_validation(config_path: Path, project_root: Path) -> dict[str, Any]:
    config = load_config(config_path, project_root)
    root = project_root / "outputs" / "t5_e2urec_joint_ablation_v1"
    framework = root / FRAMEWORK_VALIDATION_NAME
    canonical_dir = framework / "canonical_step813_814"
    instrumented_dir = framework / "instrumented_j0_step813_814"
    if framework.exists() and any(framework.iterdir()):
        raise FileExistsError(f"refusing to overwrite paired framework validation: {framework}")
    canonical_dir.mkdir(parents=True)
    instrumented_dir.mkdir(parents=True)
    canonical, _ = _arm(
        config, "j0_original_joint_reference", False, canonical_dir
    )
    instrumented, _ = _arm(
        config, "j0_original_joint_reference", True, instrumented_dir
    )
    comparisons = _compare_steps(canonical, instrumented)
    historical = _historical_anchor(config, canonical[0])
    exact = historical["exact"] and all(row["exact"] for row in comparisons)
    payload = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "historical_step813_vs_new_canonical": historical,
        "canonical_vs_instrumented": comparisons,
        "paired_step_positions": 2,
        "canonical_optimizer_steps": 2,
        "instrumented_optimizer_steps": 2,
        "total_physical_optimizer_step_calls": 4,
        "test_loader_built": False,
        "test_accessed": False,
        "exact": exact,
    }
    _atomic_json(
        canonical_dir / "steps.json",
        [_public_record(record) for record in canonical],
    )
    _atomic_json(
        instrumented_dir / "steps.json",
        [_public_record(record) for record in instrumented],
    )
    _atomic_json(framework / "paired_equivalence.json", payload)
    if not exact:
        raise RuntimeError("joint ablation paired equivalence is not exact")
    return payload


def _frozen_references(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = _source_run(config)
    cache = torch.load(
        source / "frozen_validation_predictions.pt",
        map_location="cpu",
        weights_only=False,
    )
    if cache["metadata"].get("test_accessed") is not False:
        raise ValueError("frozen validation cache is not test-free")
    original = cache["predictions"]["original"]
    retrain = cache["predictions"]["retrain"]
    step812 = json.loads(
        (source / "development_step_00812.json").read_text(encoding="utf-8")
    )
    for value in (original, retrain, step812):
        if value["samples"] != 20000 or value["test_accessed"] is not False:
            raise ValueError("invalid frozen development reference")
    if not (
        original["sample_ids"]
        == retrain["sample_ids"]
        == step812["sample_ids"]
        and original["gold"] == retrain["gold"] == step812["gold"]
        and original["sample_order_hash"]
        == retrain["sample_order_hash"]
        == step812["sample_order_hash"]
    ):
        raise ValueError("step812 development reference order mismatch")
    return original, retrain, step812


def run_branch(
    config_path: Path,
    project_root: Path,
    branch_name: str,
    run_name: str,
    resume: bool,
) -> dict[str, Any]:
    config = load_config(config_path, project_root)
    branch = BRANCHES[branch_name]
    run_dir = resolve_ablation_run(
        project_root, branch_name, run_name, resume=resume
    )
    contract = experiment_contract(config, project_root, branch_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(config["paths"]["model_dir"])
    forget_dataset = JsonPromptDataset(Path(config["paths"]["forget"]), tokenizer)
    retain_dataset = JsonPromptDataset(Path(config["paths"]["retain"]), tokenizer)
    validation_dataset = JsonPromptDataset(
        Path(config["paths"]["validation"]), tokenizer
    )
    current = build_current_model(
        Path(config["paths"]["original"]), config["lora"]
    ).to(device)
    original_model = freeze_teacher(
        load_legacy_model(Path(config["paths"]["original"]))
    ).to(device)
    augmented_model = freeze_teacher(
        load_legacy_model(Path(config["paths"]["augmented_teacher"]))
    ).to(device)
    optimizer = make_optimizer(current, config["training"]["learning_rate"])
    if resume:
        checkpoint = _latest_ablation_checkpoint(run_dir)
        state = _load_ablation_checkpoint(
            checkpoint,
            config,
            project_root,
            branch_name,
            current,
            optimizer,
        )
        if state["next_optimizer_step"] != state["step"] + 1:
            raise ValueError("ablation Resume next step mismatch")
        _verify_next_batch(state, config, state["next_batch_hash"])
    else:
        state, _ = _load_checkpoint(
            _parent_checkpoint(config), config, current, optimizer
        )
        if state["step"] != PARENT_STEP:
            raise ValueError("ablation Full must restore step812, never step0")
        state["branch_optimizer_steps"] = 0
        state["branch"] = branch_name
        state["target_step"] = TARGET_STEP
        state["test_accessed"] = False
        _atomic_json(run_dir / "pretrain_provenance.json", contract)
        _atomic_json(run_dir / "contract.json", contract)
        original_reference, retrain_reference, step812 = _frozen_references(config)
        _atomic_json(
            run_dir / "development_step_00812.json",
            {
                "source": str(
                    _source_run(config) / "development_step_00812.json"
                ),
                "source_sha256": sha256_file(
                    _source_run(config) / "development_step_00812.json"
                ),
                "metrics": development_metrics(
                    step812, original_reference, retrain_reference
                ),
                "test_accessed": False,
            },
        )
    original_reference, retrain_reference, _ = _frozen_references(config)
    metrics_path = run_dir / "metrics.jsonl"
    _atomic_json(run_dir / "run_state.json", {"status": "RUNNING", **state})
    diagnostic_steps = set(CHECKPOINT_STEPS)
    try:
        for step in range(state["step"] + 1, TARGET_STEP + 1):
            if step < FIRST_STEP:
                raise ValueError("ablation attempted a pre-joint optimizer step")
            forget_cpu, retain_cpu, order_hash = _step_data(
                config, forget_dataset, retain_dataset, step
            )
            forget = move_batch(forget_cpu, device)
            retain = move_batch(retain_cpu, device)
            current.train()
            before = _tensor_snapshot(_trainable_parameters(current))
            pre_rng = capture_rng()
            optimizer.zero_grad()
            components = compute_components(
                current,
                original_model,
                augmented_model,
                forget,
                retain,
                config["training"]["alpha"],
            )
            loss = branch_loss(components, branch_name)
            loss.backward()
            shadow = None
            if step in diagnostic_steps:
                shadow = isolated_component_shadow(
                    current,
                    optimizer,
                    pre_rng,
                    lambda: compute_components(
                        current,
                        original_model,
                        augmented_model,
                        forget,
                        retain,
                        config["training"]["alpha"],
                    ),
                    phase="joint",
                    active_components=branch.active_components,
                )
            gradient_norm = float(
                torch.sqrt(
                    sum(
                        parameter.grad.detach().double().pow(2).sum()
                        for parameter in _trainable_parameters(current).values()
                        if parameter.grad is not None
                    )
                ).cpu()
            )
            optimizer.step()
            after = _trainable_parameters(current)
            update_norm = float(
                torch.sqrt(
                    sum(
                        (
                            after[name].detach().cpu().double()
                            - before[name].double()
                        )
                        .pow(2)
                        .sum()
                        for name in before
                    )
                )
            )
            state["step"] = step
            state["executed_optimizer_steps"] = step
            state["branch_optimizer_steps"] += 1
            state["epoch"] = 1
            state["epoch_batch_position"] = step - 812
            state["next_optimizer_step"] = step + 1
            state["next_batch_hash"] = _verify_next_batch(state, config, None)
            state["forget_visits"] += len(forget_cpu["sample_id"])
            state["retain_visits"] += len(retain_cpu["sample_id"])
            state["unique_forget"] = sorted(
                set(state["unique_forget"]) | set(forget_cpu["sample_id"].tolist())
            )
            state["unique_retain"] = sorted(
                set(state["unique_retain"]) | set(retain_cpu["sample_id"].tolist())
            )
            raw_losses = component_loss_values(components, loss)
            status = component_status(branch_name, raw_losses, shadow)
            weights = {
                "L_forget": (
                    1.0 if branch_name == "j3_forget_only_control" else 0.4
                ),
                "L_sup": 0.6,
                "L_retain_KL": 0.6,
            }
            weighted_gradient_norms = {
                name: (
                    weights[name] * value["gradient_norm"]
                    if value["active"]
                    else 0.0
                )
                for name, value in status.items()
            }
            record = {
                "step": step,
                "phase": "joint",
                "branch": branch_name,
                "formula": branch.formula,
                "losses": status,
                "total_loss": float(loss.detach().cpu()),
                "weighted_component_gradient_norms": weighted_gradient_norms,
                "component_gradient_cosines": (
                    shadow["component_gradient_cosines"] if shadow else None
                ),
                "total_gradient_norm": gradient_norm,
                "update_norm": update_norm,
                "parameter_norm": float(
                    torch.sqrt(
                        sum(
                            value.detach().double().pow(2).sum()
                            for value in current.parameters()
                        )
                    ).cpu()
                ),
                "lora_norm": float(
                    torch.sqrt(
                        sum(
                            value.detach().double().pow(2).sum()
                            for value in _trainable_parameters(current).values()
                        )
                    ).cpu()
                ),
                "teacher": _teacher_scalars(components),
                "forget_visits": state["forget_visits"],
                "retain_visits": state["retain_visits"],
                "unique_forget": len(state["unique_forget"]),
                "unique_retain": len(state["unique_retain"]),
                "batch_order_hash": order_hash,
                "sample_ids": {
                    "forget": forget_cpu["sample_id"].tolist(),
                    "retain": retain_cpu["sample_id"].tolist(),
                },
                "attention_implementation": "eager",
                "nan_or_inf": False,
                "test_accessed": False,
            }
            _append_jsonl(metrics_path, record)
            if step in EVALUATION_STEPS:
                prediction = evaluate_overall_validation(
                    current,
                    validation_dataset,
                    device,
                    config["training"]["per_device_batch_size"],
                )
                _atomic_json(
                    run_dir / f"development_step_{step:05d}.json",
                    {
                        "metrics": development_metrics(
                            prediction, original_reference, retrain_reference
                        ),
                        "prediction": prediction,
                        "test_accessed": False,
                    },
                )
            if step in CHECKPOINT_STEPS:
                _atomic_ablation_checkpoint(
                    run_dir, step, current, optimizer, copy.deepcopy(state), contract
                )
            _atomic_json(
                run_dir / "run_state.json", {"status": "RUNNING", **state}
            )
        final_checkpoint = run_dir / "checkpoints" / "step_01200"
        completed = (
            state["step"] == TARGET_STEP
            and state["branch_optimizer_steps"] == JOINT_STEPS
            and final_checkpoint.is_dir()
            and (run_dir / "development_step_01200.json").is_file()
            and state["test_accessed"] is False
        )
        result = {
            "status": "COMPLETED" if completed else "FAILED",
            **state,
            "final_checkpoint": str(final_checkpoint),
            "test_loader_built": False,
            "test_accessed": False,
        }
        _atomic_json(run_dir / "run_state.json", result)
        if not completed:
            raise RuntimeError("ablation completion gate failed")
        return result
    except BaseException as error:
        _atomic_json(
            run_dir / "run_state.json",
            {
                "status": "FAILED",
                **state,
                "error": f"{type(error).__name__}: {error}",
                "test_accessed": False,
            },
        )
        raise
    finally:
        del current, original_model, augmented_model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def compare_development(project_root: Path) -> dict[str, Any]:
    root = project_root / "outputs" / "t5_e2urec_joint_ablation_v1" / "full_runs"
    result = {}
    for branch_name in BRANCHES:
        branch_root = root / branch_name
        if not branch_root.is_dir():
            continue
        for run_dir in branch_root.iterdir():
            state_path = run_dir / "run_state.json"
            if not state_path.is_file():
                continue
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("status") != "COMPLETED":
                continue
            result[branch_name] = {
                str(step): json.loads(
                    (run_dir / f"development_step_{step:05d}.json").read_text(
                        encoding="utf-8"
                    )
                )["metrics"]
                for step in EVALUATION_STEPS
            }
    return {
        "experiment": EXPERIMENT,
        "branches": result,
        "validation_only": True,
        "test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="T5 joint objective ablation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--mode",
        choices=("PairedValidation", "Full", "Resume", "CompareDevelopment"),
        required=True,
    )
    parser.add_argument("--branch", choices=tuple(BRANCHES))
    parser.add_argument("--run-name")
    arguments = parser.parse_args()
    if arguments.mode == "PairedValidation":
        payload = run_paired_validation(
            arguments.config.resolve(), arguments.project_root.resolve()
        )
    elif arguments.mode == "CompareDevelopment":
        payload = compare_development(arguments.project_root.resolve())
    else:
        if arguments.branch is None or arguments.run_name is None:
            parser.error("Full/Resume require --branch and --run-name")
        payload = run_branch(
            arguments.config.resolve(),
            arguments.project_root.resolve(),
            arguments.branch,
            arguments.run_name,
            resume=arguments.mode == "Resume",
        )
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
