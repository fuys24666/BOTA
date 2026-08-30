from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import T5Tokenizer

from src.diagnostics.t5_reconstructed_official import (
    PROTOCOL_NAME,
    SCHEMA_VERSION,
    JsonPromptDataset,
    batch_order_hash,
    build_current_model,
    compute_components,
    freeze_teacher,
    load_config,
    load_legacy_model,
    make_optimizer,
    move_batch,
    phase_for_step,
    protocol_hash,
    reject_test_path,
    resume_contract,
    seed_everything,
    sha256_file,
    total_loss,
)
from src.diagnostics.t5_trajectory_diagnostics import (
    capture_rng,
    gradient_norm,
    isolated_component_shadow,
    parameter_norm,
    restore_rng,
    rng_hashes,
    update_norm,
    _teacher_scalars,
    development_state_guard,
)

FULL_RUNNER_SCHEMA = "t5-e2urec-full-runner-v1"
FULL_DRY_RUN_NAME = "full_runner_forced_shadow_2step_v1"


def diagnostic_plan(total_steps: int) -> list[int]:
    return sorted(
        {0, 200, 400, 600, 800, 812, 813, 1000, 1200, total_steps}
        | set(range(1400, total_steps + 1, 200))
    )


def evaluation_plan(total_steps: int) -> list[int]:
    return sorted(
        step
        for step in {0, 800, 812, 813, 1000, 1200, 2000, 4000, 8000, 12000, total_steps}
        if step <= total_steps
    )


def checkpoint_steps(total_steps: int) -> list[int]:
    return diagnostic_plan(total_steps)


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_contract(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    return {
        "models": {
            role: {
                "path": str(Path(paths[key]).resolve()),
                "sha256": sha256_file(Path(paths[key])),
            }
            for role, key in (
                ("original", "original"),
                ("augmented", "augmented_teacher"),
                ("retrain", "retrain_reference"),
            )
        },
        "data": {
            role: {
                "path": str(reject_test_path(Path(paths[key])).resolve()),
                "sha256": sha256_file(Path(paths[key])),
                "samples": config["data"][f"{role}_samples"],
            }
            for role, key in (
                ("forget", "forget"),
                ("retain", "retain"),
                ("validation", "validation"),
            )
        },
    }


def full_resume_contract(config: dict[str, Any]) -> dict[str, Any]:
    budget = config["derived_budget"]
    return {
        "base": resume_contract(config),
        "runner_schema": FULL_RUNNER_SCHEMA,
        "protocol": PROTOCOL_NAME,
        "protocol_sha256": protocol_hash(config),
        "files": _file_contract(config),
        "fixed": {
            "learning_rate": config["training"]["learning_rate"],
            "batch_size": config["training"]["per_device_batch_size"],
            "effective_batch_size": config["training"]["effective_batch_size"],
            "epochs": config["training"]["epochs"],
            "alpha": config["training"]["alpha"],
            "remembering_weight": config["training"]["code_weight"],
            "weight_semantics": config["training"]["weight_semantics"],
            "warmup_steps": budget["warmup_steps"],
            "joint_steps": budget["joint_steps"],
            "total_steps": budget["total_steps"],
            "sampler": config["training"]["sampler"],
            "seed": config["training"]["seed"],
            "scheduler": config["training"]["scheduler"],
            "finite_gradient_clipping": config["training"][
                "finite_gradient_clipping"
            ],
            "lora": config["lora"],
            "attention_implementation": "eager",
        },
        "diagnostic_plan": diagnostic_plan(budget["total_steps"]),
        "evaluation_plan": evaluation_plan(budget["total_steps"]),
        "test_accessed": False,
    }


def validate_full_resume_contract(
    config: dict[str, Any], saved: dict[str, Any]
) -> None:
    expected = full_resume_contract(config)
    if saved != expected:
        differing = sorted(
            key
            for key in set(expected) | set(saved)
            if expected.get(key) != saved.get(key)
        )
        raise ValueError(f"full Resume protocol mismatch: {differing}")


def resolve_run_directory(
    config: dict[str, Any], run_name: str, mode: str
) -> Path:
    if not run_name or Path(run_name).name != run_name:
        raise ValueError("Full and Resume require one explicit RunName component")
    root = Path(config["paths"]["output_root"]).resolve()
    dry_run = mode == "FullDryRun"
    if dry_run and run_name != FULL_DRY_RUN_NAME:
        raise ValueError(f"FullDryRun RunName must be {FULL_DRY_RUN_NAME!r}")
    destination = root / ("dry_runs" if dry_run else "full_runs") / run_name
    if root not in destination.resolve().parents:
        raise ValueError("run directory escaped output root")
    if mode in {"Full", "FullDryRun"}:
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"Full refuses non-empty run directory: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
    elif mode == "Resume":
        if not destination.is_dir():
            raise FileNotFoundError(f"Resume run directory missing: {destination}")
    return destination


def epoch_order(sample_count: int, seed: int, epoch: int, stream: int) -> list[int]:
    generator = torch.Generator()
    generator.manual_seed(seed + stream)
    order = None
    for _ in range(epoch + 1):
        order = torch.randperm(sample_count, generator=generator).tolist()
    assert order is not None
    return order


def batch_indices(
    sample_count: int,
    batch_size: int,
    seed: int,
    epoch: int,
    position: int,
    stream: int,
) -> list[int]:
    order = epoch_order(sample_count, seed, epoch, stream)
    start = position * batch_size
    return order[start : start + batch_size]


def batch_index_hash(forget: list[int], retain: list[int] | None) -> str:
    return _json_hash({"forget": forget, "retain": retain or []})


def _batch(dataset: JsonPromptDataset, indices: list[int]) -> dict[str, torch.Tensor]:
    return dataset.collate_fn([dataset[index] for index in indices])


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_checkpoint_publish(
    checkpoint_root: Path, step: int, payload: dict[str, Any]
) -> Path:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_root / f"step_{step:05d}"
    if destination.exists():
        raise FileExistsError(f"checkpoint already exists: {destination}")
    temporary = checkpoint_root / f".step_{step:05d}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        torch.save(payload, temporary / "state.pt")
        manifest = {
            "schema": FULL_RUNNER_SCHEMA,
            "step": step,
            "state_sha256": sha256_file(temporary / "state.pt"),
            "published_atomically": True,
        }
        _atomic_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def latest_checkpoint(run_dir: Path) -> Path:
    run_state = run_dir / "run_state.json"
    if run_state.is_file():
        status = json.loads(run_state.read_text(encoding="utf-8")).get("status")
        if status == "FAILED":
            raise ValueError("Resume forbidden for FAILED run")
    candidates = sorted((run_dir / "checkpoints").glob("step_*"))
    if not candidates:
        raise FileNotFoundError("Resume requires a published checkpoint")
    candidate = candidates[-1]
    manifest_path = candidate / "manifest.json"
    state_path = candidate / "state.pt"
    if not manifest_path.is_file() or not state_path.is_file():
        raise ValueError(f"incomplete checkpoint: {candidate}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != FULL_RUNNER_SCHEMA:
        raise ValueError("checkpoint schema mismatch")
    if manifest.get("state_sha256") != sha256_file(state_path):
        raise ValueError("checkpoint hash mismatch")
    return candidate


def can_write_completed(
    state: dict[str, Any], total_steps: int, final_checkpoint_valid: bool
) -> bool:
    return (
        state.get("executed_optimizer_steps") == total_steps
        and state.get("step") == total_steps
        and final_checkpoint_valid
        and state.get("final_development_evaluation_complete") is True
        and state.get("adapter_reload_verified") is True
        and state.get("metrics_complete") is True
        and state.get("provenance_complete") is True
        and state.get("test_accessed") is False
    )


def _rng_payload() -> dict[str, Any]:
    state = capture_rng()
    return {
        "python": state.python,
        "numpy": state.numpy,
        "torch_cpu": state.torch_cpu,
        "torch_cuda": state.torch_cuda,
    }


def _restore_rng_payload(value: dict[str, Any]) -> None:
    state = capture_rng()
    state.python = value["python"]
    state.numpy = value["numpy"]
    state.torch_cpu = value["torch_cpu"]
    state.torch_cuda = value["torch_cuda"]
    restore_rng(state)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def evaluate_overall_validation(
    model: torch.nn.Module,
    dataset: JsonPromptDataset,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    probabilities: list[float] = []
    gold: list[int] = []
    sample_ids: list[int] = []
    with development_state_guard(model):
        model.eval()
        with torch.no_grad():
            for start in range(0, len(dataset), batch_size):
                indices = list(range(start, min(start + batch_size, len(dataset))))
                batch = move_batch(_batch(dataset, indices), device)
                output = model(
                    input_ids=batch["input_ids"], labels=batch["target_ids"]
                )
                pair = torch.softmax(
                    output.logits[:, 0, [465, 2163]], dim=-1
                )[:, 1]
                target = batch["target_ids"][:, 0]
                probabilities.extend(pair.detach().cpu().tolist())
                gold.extend((target == 2163).long().cpu().tolist())
                sample_ids.extend(indices)
    prediction = [int(value >= 0.5) for value in probabilities]
    return {
        "scope": "overall_validation_only",
        "generalization_claim": "overall_validation_only",
        "samples": len(gold),
        "accuracy": float(accuracy_score(gold, prediction)),
        "roc_auc": (
            float(roc_auc_score(gold, probabilities))
            if len(set(gold)) == 2
            else None
        ),
        "probabilities": probabilities,
        "gold": gold,
        "sample_ids": sample_ids,
        "sample_order_hash": _json_hash(sample_ids),
        "test_accessed": False,
    }


def _prediction_cache_metadata(
    config: dict[str, Any], tokenizer: T5Tokenizer
) -> dict[str, Any]:
    files = _file_contract(config)
    return {
        "schema": FULL_RUNNER_SCHEMA,
        "models": files["models"],
        "validation": files["data"]["validation"],
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "name_or_path": str(tokenizer.name_or_path),
            "vocab_size": tokenizer.vocab_size,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
        "sample_order": "sequential_0_based",
        "contains_full_vocabulary_logits": False,
        "test_accessed": False,
    }


def frozen_prediction_cache(
    run_dir: Path,
    config: dict[str, Any],
    tokenizer: T5Tokenizer,
    validation: JsonPromptDataset,
    device: torch.device,
    original: torch.nn.Module,
    augmented: torch.nn.Module,
) -> dict[str, Any]:
    path = run_dir / "frozen_validation_predictions.pt"
    metadata = _prediction_cache_metadata(config, tokenizer)
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("metadata") != metadata:
            raise ValueError("frozen validation cache provenance mismatch")
        return payload
    retrain = freeze_teacher(
        load_legacy_model(Path(config["paths"]["retrain_reference"]))
    ).to(device)
    payload = {
        "metadata": metadata,
        "predictions": {
            "original": evaluate_overall_validation(
                original, validation, device, config["training"]["per_device_batch_size"]
            ),
            "augmented": evaluate_overall_validation(
                augmented, validation, device, config["training"]["per_device_batch_size"]
            ),
            "retrain": evaluate_overall_validation(
                retrain, validation, device, config["training"]["per_device_batch_size"]
            ),
        },
    }
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    del retrain
    return payload


def component_loss_values(
    components: dict[str, torch.Tensor], total: torch.Tensor
) -> dict[str, float]:
    return {
        key: (
            float(components[key].detach().cpu())
            if key in components
            else 0.0
        )
        for key in ("L_forget", "L_sup", "L_retain_KL")
    } | {"total_loss": float(total.detach().cpu())}


def _checkpoint_payload(
    current: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any],
    contract: dict[str, Any],
    compatibility: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": FULL_RUNNER_SCHEMA,
        "contract": contract,
        "adapter_state": {
            key: value.detach().cpu()
            for key, value in get_peft_model_state_dict(current).items()
        },
        "optimizer_state": optimizer.state_dict(),
        "state": state,
        "rng": _rng_payload(),
        "rng_hash": rng_hashes(capture_rng()),
        "compatibility": compatibility,
        "provenance": {
            "runtime_model": "clean_state_dict_reconstruction_plus_fresh_lora",
            "attention_implementation": "eager",
            "historical_runtime_equivalence_claimed": False,
            "test_accessed": False,
        },
    }


def _load_checkpoint(
    checkpoint: Path,
    config: dict[str, Any],
    current: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = torch.load(
        checkpoint / "state.pt", map_location="cpu", weights_only=False
    )
    if payload.get("schema") != FULL_RUNNER_SCHEMA:
        raise ValueError("Resume checkpoint schema mismatch")
    validate_full_resume_contract(config, payload["contract"])
    result = set_peft_model_state_dict(current, payload["adapter_state"])
    if getattr(result, "unexpected_keys", []):
        raise ValueError(f"adapter reload unexpected keys: {result}")
    reloaded = get_peft_model_state_dict(current)
    if reloaded.keys() != payload["adapter_state"].keys() or any(
        not torch.equal(reloaded[key].cpu(), payload["adapter_state"][key])
        for key in reloaded
    ):
        raise ValueError("adapter reload tensor mismatch")
    optimizer.load_state_dict(payload["optimizer_state"])
    _restore_rng_payload(payload["rng"])
    if rng_hashes(capture_rng()) != payload["rng_hash"]:
        raise ValueError("Resume RNG restoration mismatch")
    return payload["state"], payload["compatibility"]


def _initial_state(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": 0,
        "executed_optimizer_steps": 0,
        "epoch": 0,
        "epoch_batch_position": 0,
        "forget_visits": 0,
        "retain_visits": 0,
        "unique_forget": [],
        "unique_retain": [],
        "batch_hash_chain": "",
        "phase_transitions": [{"step": 1, "phase": "forget_only"}],
        "final_development_evaluation_complete": False,
        "adapter_reload_verified": False,
        "metrics_complete": False,
        "provenance_complete": True,
        "test_accessed": False,
    }


def _verify_next_batch(
    state: dict[str, Any], config: dict[str, Any], expected: str | None
) -> str | None:
    total = config["derived_budget"]["total_steps"]
    next_step = state["step"] + 1
    if next_step > total:
        return None
    batches = config["derived_budget"]["forget_batches"]
    epoch = (next_step - 1) // batches
    position = (next_step - 1) % batches
    forget = batch_indices(
        config["data"]["forget_samples"],
        config["training"]["per_device_batch_size"],
        config["training"]["seed"],
        epoch,
        position,
        0,
    )
    retain = None
    if phase_for_step(next_step, _budget_object(config)) == "joint":
        retain = batch_indices(
            config["data"]["retain_samples"],
            config["training"]["per_device_batch_size"],
            config["training"]["seed"],
            epoch - config["training"]["forget_epoch"],
            position,
            10_000,
        )
    actual = batch_index_hash(forget, retain)
    if expected is not None and actual != expected:
        raise ValueError("Resume batch-order reconstruction mismatch")
    return actual


def _budget_object(config: dict[str, Any]):
    from src.diagnostics.t5_reconstructed_official import Budget

    return Budget(**config["derived_budget"])


def _compatibility_summary(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "weight_reconstruction_exact": True,
        "runtime_reconstruction_deterministic": True,
        "legacy_object_runtime_compatible": False,
        "historical_bitwise_equivalence": False,
        "checkpoint_sha256": {
            role: record["sha256"]
            for role, record in _file_contract(config)["models"].items()
        },
    }


def run_full(
    config_path: Path,
    project_root: Path,
    run_name: str,
    mode: str,
    stop_after: int | None = None,
    _run_dir: Path | None = None,
) -> dict[str, Any]:
    if mode not in {"Full", "Resume", "FullDryRun"}:
        raise ValueError(f"unsupported full runner mode: {mode}")
    config = load_config(config_path, project_root)
    run_dir = _run_dir or resolve_run_directory(config, run_name, mode)
    contract = full_resume_contract(config)
    dry_run = mode == "FullDryRun" or "dry_runs" in run_dir.parts
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(config["paths"]["model_dir"])
    forget_dataset = JsonPromptDataset(Path(config["paths"]["forget"]), tokenizer)
    retain_dataset = JsonPromptDataset(Path(config["paths"]["retain"]), tokenizer)
    validation_dataset = JsonPromptDataset(Path(config["paths"]["validation"]), tokenizer)
    compatibility = _compatibility_summary(config)

    seed_everything(config["training"]["seed"])
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
    state = _initial_state(config)
    resumed_from_step = None
    if mode == "Resume":
        checkpoint = latest_checkpoint(run_dir)
        state, compatibility = _load_checkpoint(
            checkpoint, config, current, optimizer
        )
        resumed_from_step = state["step"]
        state["resume_parent_checkpoint_step"] = resumed_from_step
        state["optimizer_state_restored"] = True
        state["rng_restored"] = True
        state["adapter_state_restored"] = True
        _verify_next_batch(state, config, state.get("next_batch_hash"))

    if not dry_run:
        frozen_prediction_cache(
            run_dir,
            config,
            tokenizer,
            validation_dataset,
            device,
            original,
            augmented,
        )
        if state["step"] == 0:
            evaluation = evaluate_overall_validation(
                current,
                validation_dataset,
                device,
                config["training"]["per_device_batch_size"],
            )
            _atomic_json(run_dir / "development_step_00000.json", evaluation)

    _atomic_json(run_dir / "contract.json", contract)
    _atomic_json(
        run_dir / "run_state.json",
        {"status": "RUNNING", **state, "test_accessed": False},
    )
    target = config["derived_budget"]["total_steps"]
    if stop_after is not None:
        target = min(target, state["step"] + stop_after)
    metrics_path = run_dir / "metrics.jsonl"
    started = time.time()
    budget = _budget_object(config)
    try:
        while state["step"] < target:
            step = state["step"] + 1
            if resumed_from_step is not None and "first_resumed_step" not in state:
                state["first_resumed_step"] = step
            epoch = (step - 1) // budget.forget_batches
            position = (step - 1) % budget.forget_batches
            phase = phase_for_step(step, budget)
            if step == budget.warmup_steps + 1:
                state["phase_transitions"].append({"step": step, "phase": phase})
            forget_indices = batch_indices(
                len(forget_dataset),
                config["training"]["per_device_batch_size"],
                config["training"]["seed"],
                epoch,
                position,
                0,
            )
            retain_indices = None
            if phase == "joint":
                retain_indices = batch_indices(
                    len(retain_dataset),
                    config["training"]["per_device_batch_size"],
                    config["training"]["seed"],
                    epoch - config["training"]["forget_epoch"],
                    position,
                    10_000,
                )
            order_hash = batch_index_hash(forget_indices, retain_indices)
            forget_batch = move_batch(_batch(forget_dataset, forget_indices), device)
            retain_batch = (
                move_batch(_batch(retain_dataset, retain_indices), device)
                if retain_indices is not None
                else None
            )
            current.train()
            before = {
                name: parameter.detach().cpu().clone()
                for name, parameter in current.named_parameters()
                if parameter.requires_grad
            }
            pre_rng = capture_rng()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            step_started = time.time()
            optimizer.zero_grad()
            components = compute_components(
                current,
                original,
                augmented,
                forget_batch,
                retain_batch,
                config["training"]["alpha"],
            )
            loss = total_loss(
                components, phase, config["training"]["code_weight"]
            )
            loss.backward()
            shadow = None
            if dry_run or step in diagnostic_plan(budget.total_steps):
                shadow = isolated_component_shadow(
                    current,
                    optimizer,
                    pre_rng,
                    lambda: compute_components(
                        current,
                        original,
                        augmented,
                        forget_batch,
                        retain_batch,
                        config["training"]["alpha"],
                    ),
                    phase=phase,
                )
            grad_norm = gradient_norm(current)
            nonfinite = not bool(torch.isfinite(loss).item()) or not math.isfinite(
                grad_norm
            )
            if nonfinite:
                raise FloatingPointError(f"NaN/Inf at step {step}")
            optimizer.step()
            duration = time.time() - step_started
            state["step"] = step
            state["executed_optimizer_steps"] += 1
            state["epoch"] = epoch
            state["epoch_batch_position"] = position + 1
            state["forget_visits"] += len(forget_indices)
            state["retain_visits"] += len(retain_indices or [])
            state["unique_forget"] = sorted(
                set(state["unique_forget"]) | set(forget_indices)
            )
            state["unique_retain"] = sorted(
                set(state["unique_retain"]) | set(retain_indices or [])
            )
            state["batch_hash_chain"] = hashlib.sha256(
                (state["batch_hash_chain"] + order_hash).encode()
            ).hexdigest()
            state["next_batch_hash"] = _verify_next_batch(state, config, None)
            state["next_optimizer_step"] = step + 1
            elapsed = time.time() - started
            remaining = budget.total_steps - step
            record = {
                "step": step,
                "epoch": epoch,
                "epoch_batch_position": position,
                "phase": phase,
                "losses": component_loss_values(components, loss),
                "total_loss": float(loss.detach().cpu()),
                "total_gradient_norm": grad_norm,
                "update_norm": update_norm(before, current),
                "parameter_norm": parameter_norm(current),
                "lora_norm": parameter_norm(current, trainable_only=True),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "step_seconds": duration,
                "elapsed_seconds": elapsed,
                "eta_seconds": (elapsed / state["executed_optimizer_steps"]) * remaining,
                "peak_vram_bytes": (
                    torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
                ),
                "forget_visits": state["forget_visits"],
                "retain_visits": state["retain_visits"],
                "unique_forget": len(state["unique_forget"]),
                "unique_retain": len(state["unique_retain"]),
                "batch_sample_id_order_hash": order_hash,
                "nan_or_inf": False,
                "shadow_executed": shadow is not None,
                "shadow": shadow,
                "teacher": (
                    _teacher_scalars(components)
                    if dry_run or step in diagnostic_plan(budget.total_steps)
                    else None
                ),
                "attention_implementation": "eager",
                "test_accessed": False,
            }
            _append_jsonl(metrics_path, record)
            if not dry_run and step in evaluation_plan(budget.total_steps):
                evaluation = evaluate_overall_validation(
                    current,
                    validation_dataset,
                    device,
                    config["training"]["per_device_batch_size"],
                )
                _atomic_json(
                    run_dir / f"development_step_{step:05d}.json", evaluation
                )
            if (
                step in checkpoint_steps(budget.total_steps) or step == target
            ) and step != budget.total_steps:
                payload = _checkpoint_payload(
                    current, optimizer, state, contract, compatibility
                )
                checkpoint = atomic_checkpoint_publish(
                    run_dir / "checkpoints", step, payload
                )
                state["last_checkpoint"] = str(checkpoint)
            _atomic_json(
                run_dir / "run_state.json",
                {"status": "RUNNING", **state, "test_accessed": False},
            )
            del components, loss, forget_batch, retain_batch

        status = "PAUSED" if target < budget.total_steps else "RUNNING"
        if target == budget.total_steps:
            final_evaluation = run_dir / f"development_step_{target:05d}.json"
            state["final_development_evaluation_complete"] = final_evaluation.is_file()
            adapter_state = {
                key: value.detach().cpu().clone()
                for key, value in get_peft_model_state_dict(current).items()
            }
            reload_model = build_current_model(
                Path(config["paths"]["original"]), config["lora"]
            )
            set_peft_model_state_dict(reload_model, adapter_state)
            reloaded_state = get_peft_model_state_dict(reload_model)
            state["adapter_reload_verified"] = (
                reloaded_state.keys() == adapter_state.keys()
                and all(
                    torch.equal(reloaded_state[key].cpu(), adapter_state[key])
                    for key in adapter_state
                )
            )
            del reload_model
            state["metrics_complete"] = (
                sum(1 for _ in metrics_path.open(encoding="utf-8"))
                == budget.total_steps
            )
            final_payload = _checkpoint_payload(
                current, optimizer, state, contract, compatibility
            )
            final_checkpoint = atomic_checkpoint_publish(
                run_dir / "checkpoints", target, final_payload
            )
            if can_write_completed(
                state, budget.total_steps, final_checkpoint.is_dir()
            ):
                status = "COMPLETED"
            else:
                status = "FAILED"
        result = {
            "status": status,
            **state,
            "run_dir": str(run_dir),
            "mode": mode,
            "test_loader_built": False,
            "test_accessed": False,
        }
        _atomic_json(run_dir / "run_state.json", result)
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
        del current, original, augmented, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_full_dry_run(
    config_path: Path, project_root: Path, run_name: str
) -> dict[str, Any]:
    first = run_full(
        config_path, project_root, run_name, "FullDryRun", stop_after=1
    )
    config = load_config(config_path, project_root)
    dry_dir = (
        Path(config["paths"]["output_root"]).resolve()
        / "dry_runs"
        / run_name
    )
    # Resume uses the exact dry-run directory but the same strict Resume path.
    checkpoint = latest_checkpoint(dry_dir)
    state_payload = torch.load(
        checkpoint / "state.pt", map_location="cpu", weights_only=False
    )
    saved_hash = state_payload["state"]["next_batch_hash"]

    # Resume without resolving a formal full_runs path.
    original_output = config["paths"]["output_root"]
    config_copy = json.loads(json.dumps(config))
    del original_output, config_copy
    second = _resume_dry_directory(
        config_path, project_root, run_name, dry_dir, saved_hash
    )
    if first["step"] != 1 or second["step"] != 2:
        raise RuntimeError("FullDryRun steps are not exactly 1 then 2")
    if second.get("resume_parent_checkpoint_step") != 1:
        raise RuntimeError("FullDryRun Resume parent is not step 1")
    if second.get("first_resumed_step") != 2:
        raise RuntimeError("FullDryRun did not begin Resume at step 2")
    step2_checkpoint = latest_checkpoint(dry_dir)
    step2_payload = torch.load(
        step2_checkpoint / "state.pt", map_location="cpu", weights_only=False
    )
    if step2_payload["state"]["step"] != 2:
        raise RuntimeError("reloaded final dry-run checkpoint is not step 2")
    result = {
        "status": "DRY_RUN_COMPLETED",
        "exact": True,
        "first_step": 1,
        "resumed_step": 2,
        "executed_optimizer_steps": 2,
        "optimizer_state_restored": True,
        "rng_restored": True,
        "batch_order_continuous": True,
        "protocol_validated": True,
        "adapter_reload_verified": True,
        "formal_full_directory_written": False,
        "test_accessed": False,
        "run_dir": str(dry_dir),
    }
    _atomic_json(dry_dir / "dry_run_verification.json", result)
    _atomic_json(dry_dir / "run_state.json", result)
    return result


def _resume_dry_directory(
    config_path: Path,
    project_root: Path,
    run_name: str,
    dry_dir: Path,
    expected_next_hash: str,
) -> dict[str, Any]:
    # Temporarily route Resume resolution to the already isolated dry-run root.
    config = load_config(config_path, project_root)
    formal = (
        Path(config["paths"]["output_root"]).resolve() / "full_runs" / run_name
    )
    if formal.exists():
        raise FileExistsError("FullDryRun refuses an existing formal Full directory")
    result = run_full(
        config_path,
        project_root,
        run_name,
        "Resume",
        stop_after=1,
        _run_dir=dry_dir,
    )
    if result.get("step") != 2:
        raise RuntimeError("dry Resume did not reach step 2")
    checkpoint = latest_checkpoint(dry_dir)
    payload = torch.load(
        checkpoint / "state.pt", map_location="cpu", weights_only=False
    )
    validate_full_resume_contract(config, payload["contract"])
    if payload["state"]["batch_hash_chain"] == "":
        raise RuntimeError("batch hash chain was not restored")
    if expected_next_hash is None:
        raise RuntimeError("step-1 checkpoint lacks next-batch hash")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable full T5 diagnostics")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=("Full", "Resume", "FullDryRun"), required=True)
    parser.add_argument("--run-name", required=True)
    arguments = parser.parse_args()
    if arguments.mode == "FullDryRun":
        result = run_full_dry_run(
            arguments.config.resolve(),
            arguments.project_root.resolve(),
            arguments.run_name,
        )
    else:
        result = run_full(
            arguments.config.resolve(),
            arguments.project_root.resolve(),
            arguments.run_name,
            arguments.mode,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
