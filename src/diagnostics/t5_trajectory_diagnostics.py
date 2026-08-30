from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from transformers import T5Tokenizer

from src.diagnostics.t5_reconstructed_official import (
    PROTOCOL_NAME,
    SCHEMA_VERSION,
    batch_order_hash,
    build_current_model,
    compute_components,
    freeze_teacher,
    load_config,
    load_legacy_t5_for_reconstructed_diagnostics,
    load_legacy_model,
    make_loader,
    make_optimizer,
    move_batch,
    prepare_run_directory,
    protocol_hash,
    seed_everything,
    total_loss,
    verify_clean_reconstruction_determinism,
)


@dataclass
class RngState:
    python: object
    numpy: tuple[Any, ...]
    torch_cpu: torch.Tensor
    torch_cuda: list[torch.Tensor]


def capture_rng() -> RngState:
    return RngState(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.get_rng_state().clone(),
        torch_cuda=[state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    )


def restore_rng(state: RngState) -> None:
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.set_rng_state(state.torch_cpu)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state.torch_cuda)


@contextmanager
def development_state_guard(*models: torch.nn.Module):
    rng = capture_rng()
    modes = [_model_mode(model) for model in models]
    try:
        yield
    finally:
        for model, mode in zip(models, modes):
            _restore_model_mode(model, mode)
        restore_rng(rng)


def _hash_bytes(parts: list[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    return _hash_bytes(
        [
            str(value.dtype).encode(),
            str(tuple(value.shape)).encode(),
            value.numpy().tobytes(),
        ]
    )


def rng_hashes(state: RngState) -> dict[str, Any]:
    numpy_array = np.asarray(state.numpy[1]).tobytes()
    return {
        "python": hashlib.sha256(repr(state.python).encode()).hexdigest(),
        "numpy": _hash_bytes(
            [str(state.numpy[0]).encode(), numpy_array, repr(state.numpy[2:]).encode()]
        ),
        "torch_cpu": tensor_hash(state.torch_cpu),
        "torch_cuda": [tensor_hash(value) for value in state.torch_cuda],
    }


def named_parameter_hashes(model: torch.nn.Module) -> dict[str, str]:
    return {name: tensor_hash(parameter) for name, parameter in model.named_parameters()}


def named_gradient_hashes(model: torch.nn.Module) -> dict[str, str | None]:
    return {
        name: None if parameter.grad is None else tensor_hash(parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _recursive_hash(value: Any) -> Any:
    if torch.is_tensor(value):
        return {"tensor": tensor_hash(value)}
    if isinstance(value, dict):
        return {str(key): _recursive_hash(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_recursive_hash(item) for item in value]
    return value


def optimizer_hash(optimizer: torch.optim.Optimizer) -> str:
    normalized = _recursive_hash(optimizer.state_dict())
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def gradient_norm(model: torch.nn.Module) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += parameter.grad.detach().double().pow(2).sum().cpu()
    return math.sqrt(float(total))


def parameter_norm(model: torch.nn.Module, trainable_only: bool = False) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for parameter in model.parameters():
        if not trainable_only or parameter.requires_grad:
            total += parameter.detach().double().pow(2).sum().cpu()
    return math.sqrt(float(total))


def update_norm(before: dict[str, torch.Tensor], model: torch.nn.Module) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for name, parameter in model.named_parameters():
        if name in before:
            total += (
                parameter.detach().cpu().double() - before[name].double()
            ).pow(2).sum()
    return math.sqrt(float(total))


COMPONENT_NAMES = ("L_forget", "L_sup", "L_retain_KL")


def _grad_stats(
    gradients: dict[str, tuple[torch.Tensor | None, ...]],
    active_components: tuple[str, ...],
    inactive_reason: str,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    norms: dict[str, float] = {}
    for label in COMPONENT_NAMES:
        values = gradients.get(label, ())
        square = sum(
            float(value.detach().double().pow(2).sum().cpu())
            for value in values
            if value is not None
        )
        norms[label] = math.sqrt(square)
    cosines: dict[str, dict[str, Any]] = {}
    labels = list(COMPONENT_NAMES)
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            key = f"{left}__{right}"
            if left not in active_components or right not in active_components:
                cosines[key] = {
                    "value": None,
                    "status": "not_applicable",
                    "reason": inactive_reason,
                }
                continue
            dot = 0.0
            for left_value, right_value in zip(gradients[left], gradients[right]):
                if left_value is not None and right_value is not None:
                    dot += float(
                        (left_value.detach().double() * right_value.detach().double())
                        .sum()
                        .cpu()
                    )
            denominator = norms[left] * norms[right]
            cosines[key] = (
                {"value": dot / denominator, "status": "finite", "reason": None}
                if denominator
                else {
                    "value": None,
                    "status": "not_applicable",
                    "reason": "zero_gradient_norm",
                }
            )
    return norms, cosines


def _model_mode(model: torch.nn.Module) -> dict[str, bool]:
    return {name: module.training for name, module in model.named_modules()}


def _restore_model_mode(model: torch.nn.Module, modes: dict[str, bool]) -> None:
    for name, module in model.named_modules():
        module.training = modes[name]


def isolated_component_shadow(
    current: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    pre_forward_rng: RngState,
    component_builder: Callable[[], dict[str, torch.Tensor]],
    phase: str,
    active_components: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if active_components is None:
        if phase == "forget_only":
            active_components = ("L_forget",)
        elif phase == "joint":
            active_components = COMPONENT_NAMES
        else:
            raise ValueError(f"unsupported shadow phase: {phase}")
    if not active_components or any(
        name not in COMPONENT_NAMES for name in active_components
    ):
        raise ValueError(f"invalid active shadow components: {active_components}")
    inactive_reason = (
        "component_inactive_in_forget_only_phase"
        if phase == "forget_only"
        else "component_disabled_by_ablation"
    )
    parameters = [value for value in current.parameters() if value.requires_grad]
    parameter_snapshot = {
        name: value.detach().clone()
        for name, value in current.named_parameters()
        if value.requires_grad
    }
    buffer_snapshot = {
        name: value.detach().clone() for name, value in current.named_buffers()
    }
    gradient_snapshot = {
        name: None if value.grad is None else value.grad.detach().clone()
        for name, value in current.named_parameters()
        if value.requires_grad
    }
    optimizer_snapshot = copy.deepcopy(optimizer.state_dict())
    optimizer_hash_before = optimizer_hash(optimizer)
    modes = _model_mode(current)
    canonical_rng = capture_rng()
    canonical_gradient_hash = named_gradient_hashes(current)
    try:
        component_gradients: dict[str, tuple[torch.Tensor | None, ...]] = {}
        component_losses: dict[str, float] = {}
        for name in active_components:
            restore_rng(pre_forward_rng)
            components = component_builder()
            loss = components[name]
            component_losses[name] = float(loss.detach().cpu())
            component_gradients[name] = torch.autograd.grad(
                loss, parameters, allow_unused=True
            )
        norms, cosines = _grad_stats(
            component_gradients, active_components, inactive_reason
        )
        for name in COMPONENT_NAMES:
            component_losses.setdefault(name, 0.0)
        return {
            "phase": phase,
            "active_components": list(active_components),
            "component_losses": component_losses,
            "component_gradient_norms": norms,
            "component_gradient_cosines": cosines,
            "inactive_component_reason": (
                inactive_reason
                if len(active_components) != len(COMPONENT_NAMES)
                else None
            ),
        }
    finally:
        with torch.no_grad():
            for name, value in current.named_parameters():
                if name in parameter_snapshot:
                    value.copy_(parameter_snapshot[name])
            for name, value in current.named_buffers():
                value.copy_(buffer_snapshot[name])
        for name, value in current.named_parameters():
            if name in gradient_snapshot:
                saved = gradient_snapshot[name]
                value.grad = None if saved is None else saved.clone()
        optimizer.load_state_dict(optimizer_snapshot)
        _restore_model_mode(current, modes)
        restore_rng(canonical_rng)
        if named_gradient_hashes(current) != canonical_gradient_hash:
            raise RuntimeError("shadow replay changed canonical gradients")
        for name, value in current.named_parameters():
            if name in parameter_snapshot and not torch.equal(
                value.detach(), parameter_snapshot[name]
            ):
                raise RuntimeError(f"shadow replay changed parameter {name}")
        for name, value in current.named_buffers():
            if not torch.equal(value.detach(), buffer_snapshot[name]):
                raise RuntimeError(f"shadow replay changed buffer {name}")
        if optimizer_hash(optimizer) != optimizer_hash_before:
            raise RuntimeError("shadow replay changed optimizer state")
        if _model_mode(current) != modes:
            raise RuntimeError("shadow replay changed model mode")
        if rng_hashes(capture_rng()) != rng_hashes(canonical_rng):
            raise RuntimeError("shadow replay failed to restore RNG")


def label_probability(logits: torch.Tensor, yes_id: int = 2163, no_id: int = 465) -> torch.Tensor:
    pair = logits[:, 0, [no_id, yes_id]]
    return torch.softmax(pair, dim=-1)[:, 1]


def probability_summary(probability: np.ndarray, gold: np.ndarray) -> dict[str, float]:
    return {
        "probability_mean": float(np.mean(probability)),
        "probability_std": float(np.std(probability)),
        "probability_min": float(np.min(probability)),
        "probability_max": float(np.max(probability)),
        "positive_rate": float(np.mean(probability >= 0.5)),
        "AUC": float(roc_auc_score(gold, probability)),
        "ACC": float(accuracy_score(gold, probability >= 0.5)),
        "LogLoss": float(log_loss(gold, probability)),
    }


def standard_jsd(left: np.ndarray, right: np.ndarray, epsilon: float = 1e-12) -> float:
    midpoint = 0.5 * (left + right)
    kl_left = np.sum(left * np.log((left + epsilon) / (midpoint + epsilon)), axis=-1)
    kl_right = np.sum(right * np.log((right + epsilon) / (midpoint + epsilon)), axis=-1)
    return float(np.mean(0.5 * (kl_left + kl_right)))


def legacy_symmetric_kl(
    left: np.ndarray, right: np.ndarray, epsilon: float = 1e-12
) -> float:
    forward = np.sum(left * np.log((left + epsilon) / (right + epsilon)), axis=-1)
    reverse = np.sum(right * np.log((right + epsilon) / (left + epsilon)), axis=-1)
    return float(np.mean(0.5 * (forward + reverse)))


def direction_diagnostics(
    original_margin: np.ndarray,
    augmented_margin: np.ndarray,
    forced_margin: np.ndarray,
    retrain_margin: np.ndarray,
) -> dict[str, Any]:
    raw = augmented_margin - original_margin
    forced = forced_margin - original_margin
    retrain = retrain_margin - original_margin

    def compare(candidate: np.ndarray) -> dict[str, float]:
        error = candidate - retrain
        return {
            "sign_agreement": float(np.mean(np.sign(candidate) == np.sign(retrain))),
            "Pearson": float(pearsonr(candidate, retrain).statistic),
            "Spearman": float(spearmanr(candidate, retrain).statistic),
            "RMSE": float(np.sqrt(np.mean(error**2))),
            "MAE": float(np.mean(np.abs(error))),
        }

    quantiles = np.quantile(np.abs(raw), [0.0, 0.25, 0.5, 0.75, 1.0])
    bins = []
    for lower, upper in zip(quantiles[:-1], quantiles[1:]):
        mask = (np.abs(raw) >= lower) & (np.abs(raw) <= upper)
        bins.append(
            {
                "teacher_strength_min": float(lower),
                "teacher_strength_max": float(upper),
                "count": int(mask.sum()),
                "forced_teacher_MAE": float(np.mean(np.abs(forced[mask] - retrain[mask])))
                if mask.any()
                else float("nan"),
            }
        )
    return {
        "direction_signs": {
            "raw": "Augmented - Original",
            "forced": "ForcedTeacher - Original",
            "reference": "Retrain - Original",
        },
        "raw_augmented_minus_original": compare(raw),
        "actual_forced_teacher_direction": compare(forced),
        "teacher_strength_quantiles": bins,
    }


def distribution_proximity(
    candidate: np.ndarray, reference: np.ndarray
) -> dict[str, float]:
    delta = candidate - reference
    return {
        "probability_L2_RMS": float(np.sqrt(np.mean(delta**2))),
        "standard_JSD": standard_jsd(candidate, reference),
        "legacy_symmetric_KL": legacy_symmetric_kl(candidate, reference),
        "prediction_agreement": float(
            np.mean(np.argmax(candidate, axis=-1) == np.argmax(reference, axis=-1))
        ),
    }


def _teacher_scalars(components: dict[str, torch.Tensor]) -> dict[str, float]:
    forced = components["forced_logits"].detach()
    probability = torch.softmax(forced, dim=-1)
    entropy = -(probability * torch.log(probability + 1e-12)).sum(-1).mean()
    yes_no_margin = (forced[..., 2163] - forced[..., 465]).mean()
    raw_strength = (
        components["forget_augmented_logits"] - components["forget_original_logits"]
    ).norm() / math.sqrt(forced.numel())
    forced_strength = (
        forced - components["forget_original_logits"]
    ).norm() / math.sqrt(forced.numel())
    return {
        "forced_teacher_entropy": float(entropy.cpu()),
        "yes_no_margin": float(yes_no_margin.cpu()),
        "raw_teacher_strength_rms": float(raw_strength.cpu()),
        "forced_teacher_strength_rms": float(forced_strength.cpu()),
    }


def _single_real_step(
    config: dict[str, Any],
    retain_batch: dict[str, torch.Tensor],
    forget_batch: dict[str, torch.Tensor],
    instrumented: bool,
    device: torch.device,
) -> dict[str, Any]:
    training = config["training"]
    paths = config["paths"]
    seed_everything(training["seed"])
    current = build_current_model(Path(paths["original"]), config["lora"]).to(device)
    original = freeze_teacher(load_legacy_model(Path(paths["original"]))).to(device)
    augmented = freeze_teacher(load_legacy_model(Path(paths["augmented_teacher"]))).to(device)
    current.train()
    optimizer = make_optimizer(current, training["learning_rate"])
    retain_batch = move_batch(retain_batch, device)
    forget_batch = move_batch(forget_batch, device)
    trainable_before = {
        name: value.detach().cpu().clone()
        for name, value in current.named_parameters()
        if value.requires_grad
    }
    pre_forward_rng = capture_rng()
    optimizer.zero_grad()
    components = compute_components(
        current,
        original,
        augmented,
        forget_batch,
        retain_batch,
        training["alpha"],
    )
    loss = total_loss(components, "joint", training["code_weight"])
    loss.backward()
    canonical_post_backward_rng = capture_rng()
    shadow = None
    if instrumented:
        shadow = isolated_component_shadow(
            current,
            optimizer,
            pre_forward_rng,
            lambda: compute_components(
                current,
                original,
                augmented,
                forget_batch,
                retain_batch,
                training["alpha"],
            ),
            phase="joint",
        )
        if rng_hashes(capture_rng()) != rng_hashes(canonical_post_backward_rng):
            raise RuntimeError("instrumentation changed post-backward RNG")
    gradients = named_gradient_hashes(current)
    total_gradient_norm = gradient_norm(current)
    optimizer.step()
    result = {
        "mode": "instrumented" if instrumented else "canonical",
        "phase": "joint",
        "losses": {
            name: float(components[name].detach().cpu())
            for name in ("L_forget", "L_sup", "L_retain_KL")
        }
        | {"total_loss": float(loss.detach().cpu())},
        "gradient_hashes": gradients,
        "parameter_hashes": named_parameter_hashes(current),
        "optimizer_hash": optimizer_hash(optimizer),
        "rng": rng_hashes(capture_rng()),
        "total_gradient_norm": total_gradient_norm,
        "parameter_norm": parameter_norm(current),
        "lora_norm": parameter_norm(current, trainable_only=True),
        "update_norm": update_norm(trainable_before, current),
        "teacher": _teacher_scalars(components),
        "shadow": shadow,
        "gradient_checkpointing": bool(
            getattr(current, "is_gradient_checkpointing", False)
        ),
        "dropout_training": all(
            module.training
            for module in current.modules()
            if isinstance(module, torch.nn.Dropout)
        ),
        "attention_implementation": current.config._attn_implementation_internal,
    }
    del current, original, augmented, optimizer, components, loss
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _exact_comparison(canonical: dict[str, Any], instrumented: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "losses": canonical["losses"] == instrumented["losses"],
        "gradients_tensorwise": canonical["gradient_hashes"]
        == instrumented["gradient_hashes"],
        "parameters_tensorwise": canonical["parameter_hashes"]
        == instrumented["parameter_hashes"],
        "optimizer_state": canonical["optimizer_hash"]
        == instrumented["optimizer_hash"],
        "rng": canonical["rng"] == instrumented["rng"],
        "gradient_checkpointing": canonical["gradient_checkpointing"]
        == instrumented["gradient_checkpointing"],
        "dropout_state": canonical["dropout_training"]
        == instrumented["dropout_training"],
        "attention_implementation": (
            canonical["attention_implementation"]
            == instrumented["attention_implementation"]
            == "eager"
        ),
    }
    exact = all(fields.values())
    return {
        "fields": fields,
        "loss_max_absolute_error": 0.0 if fields["losses"] else None,
        "gradient_tensorwise_max_absolute_error": (
            0.0 if fields["gradients_tensorwise"] else None
        ),
        "parameter_tensorwise_max_absolute_error": (
            0.0 if fields["parameters_tensorwise"] else None
        ),
        "exact": exact,
    }


def run_paired(
    config_path: Path,
    project_root: Path,
    run_name: str,
) -> dict[str, Any]:
    config = load_config(config_path, project_root)
    output = prepare_run_directory(
        Path(config["paths"]["output_root"]), run_name, dry_run=True
    )
    seed = config["training"]["seed"]
    tokenizer = T5Tokenizer.from_pretrained(
        str((project_root / config["paths"].get("model_dir", "pretrained_models/t5-base")).resolve())
        if not Path(config["paths"].get("model_dir", "")).is_absolute()
        else config["paths"]["model_dir"]
    )
    retain_loader = make_loader(
        Path(config["paths"]["retain"]),
        tokenizer,
        config["training"]["per_device_batch_size"],
        shuffle=True,
        seed=seed,
    )
    forget_loader = make_loader(
        Path(config["paths"]["forget"]),
        tokenizer,
        config["training"]["per_device_batch_size"],
        shuffle=True,
        seed=seed,
    )
    retain_batch = next(iter(retain_loader))
    forget_batch = next(iter(forget_loader))
    order_hash = batch_order_hash(retain_batch, forget_batch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    reconstruction_reports = {}
    for role, path_key in (
        ("original", "original"),
        ("augmented", "augmented_teacher"),
        ("retrain", "retrain_reference"),
    ):
        clean_model, report = load_legacy_t5_for_reconstructed_diagnostics(
            Path(config["paths"][path_key])
        )
        reconstruction_reports[role] = report
        del clean_model
    compatibility = {
        "checkpoints": reconstruction_reports,
        "original_runtime_determinism": verify_clean_reconstruction_determinism(
            Path(config["paths"]["original"]), device=device
        ),
        "all_three_strict_and_tensor_exact": True,
        "weight_reconstruction_exact": True,
        "runtime_reconstruction_deterministic": True,
        "legacy_object_runtime_compatible": False,
        "historical_bitwise_equivalence": False,
    }
    canonical = _single_real_step(
        config, retain_batch, forget_batch, instrumented=False, device=device
    )
    instrumented = _single_real_step(
        config, retain_batch, forget_batch, instrumented=True, device=device
    )
    comparison = _exact_comparison(canonical, instrumented)
    payload = {
        "protocol_name": PROTOCOL_NAME,
        "schema_version": SCHEMA_VERSION,
        "validation": "real_t5_reconstructed_paired_optimizer_step",
        "protocol_sha256": protocol_hash(config),
        "device": str(device),
        "compatibility": compatibility,
        "attention_implementation": "eager",
        "sample_ids": {
            "retain": retain_batch["sample_id"].tolist(),
            "forget": forget_batch["sample_id"].tolist(),
        },
        "sample_id_hash": {
            "canonical": order_hash,
            "instrumented": order_hash,
            "equal": True,
        },
        "batch_order_hash": {
            "canonical": order_hash,
            "instrumented": order_hash,
            "equal": True,
        },
        "canonical": canonical,
        "instrumented": instrumented,
        "comparison": comparison,
        "executed_optimizer_steps": 2,
        "test_loader_built": False,
        "test_accessed": False,
        "elapsed_seconds": time.time() - started,
        "exact": comparison["exact"],
        "failure_reason": None if comparison["exact"] else comparison["fields"],
    }
    destination = output / "paired_t5_equivalence.json"
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not payload["exact"]:
        raise RuntimeError(f"real T5 paired validation diverged: {comparison}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One canonical + one instrumented real-T5 reconstructed paired step"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-name", required=True)
    arguments = parser.parse_args()
    payload = run_paired(
        arguments.config.resolve(), arguments.project_root.resolve(), arguments.run_name
    )
    print(
        json.dumps(
            {
                "exact": payload["exact"],
                "executed_optimizer_steps": payload["executed_optimizer_steps"],
                "test_accessed": payload["test_accessed"],
                "output": str(
                    Path(payload["canonical"]["mode"])
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
