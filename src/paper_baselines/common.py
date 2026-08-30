from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import psutil
import torch
import yaml

from src.paper_if_a2.artifacts import atomic_torch_save, complete, publish_manifest
from src.paper_if_a2.common import (
    atomic_json,
    atomic_text,
    canonical_hash,
    directory_hash,
    git_snapshot,
    require_formal_preflight,
    safe_run_name,
    sha256_file,
)


MODEL_MANIFEST_SCHEMA = "paper-unlearning-model-manifest-v1"
TRAINING_MODES = ("Preflight", "BudgetAudit", "SyntheticDryRun", "Full", "Resume", "Analyze")
TRAINING_SPLITS = ("train", "forget", "retain")
FINAL_TEST_PARTS = {"test", "finaltest", "final_test"}


class SafeStop(RuntimeError):
    def __init__(self, reason: str, evidence: dict[str, Any] | None = None):
        super().__init__(reason)
        self.reason = reason
        self.evidence = evidence or {}


def reject_final_test_path(path: str | Path) -> None:
    parts = {part.lower() for part in Path(path).parts}
    if parts & FINAL_TEST_PARTS:
        raise ValueError(f"FinalTest path is forbidden for training: {path}")


def load_config(path: Path, schema: str) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError(f"configuration schema must be {schema}")
    if value.get("test_access_policy") != "forbidden":
        raise ValueError("training configuration must forbid FinalTest")
    return value


def validate_source_files(root: Path, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, spec in sources.items():
        path = (root / spec["path"]).resolve()
        reject_final_test_path(path)
        if not path.exists() or not (path.is_file() or path.is_dir()):
            raise FileNotFoundError(path)
        actual = directory_hash(path) if path.is_dir() else sha256_file(path)
        if spec.get("sha256") and actual != spec["sha256"]:
            raise ValueError(f"source SHA mismatch: {name}")
        size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.is_dir() else path.stat().st_size
        result[name] = {"path": str(path), "sha256": actual, "bytes": size, "type": "directory" if path.is_dir() else "file"}
    return result


def new_run_directory(root: Path, output_root: str, run_name: str) -> Path:
    base = (root / output_root).resolve()
    target = (base / safe_run_name(run_name)).resolve()
    if base not in target.parents or target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)
    return target


def existing_run_directory(root: Path, output_root: str, run_name: str) -> Path:
    base = (root / output_root).resolve()
    target = (base / safe_run_name(run_name)).resolve()
    if base not in target.parents or not target.is_dir():
        raise FileNotFoundError(target)
    return target


def atomic_publish_directory(stage: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage, destination)


def publish_stage(run_dir: Path, name: str, files: list[str], extra: dict[str, Any]) -> Path:
    work = run_dir / "work" / name
    destination = run_dir / "stages" / name
    publish_manifest(work, files, {"stage": name, "published_atomically": True, "test_accessed": False, **extra})
    complete(work, f"PAPER_BASELINE_STAGE_{name.upper()}_COMPLETED")
    atomic_publish_directory(work, destination)
    return destination


def verify_stage(path: Path, name: str) -> dict[str, Any]:
    expected = f"PAPER_BASELINE_STAGE_{name.upper()}_COMPLETED\n"
    if (path / "COMPLETED").read_text(encoding="utf-8") != expected:
        raise ValueError(f"invalid stage marker: {name}")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("stage") != name or manifest.get("test_accessed") is not False:
        raise ValueError(f"invalid stage manifest: {name}")
    for relative, expected_sha in manifest["files"].items():
        item = path / relative
        actual = directory_hash(item) if item.is_dir() else sha256_file(item)
        if actual != expected_sha:
            raise ValueError(f"stage artifact SHA mismatch: {name}/{relative}")
    return manifest


def capture_rng() -> dict[str, Any]:
    value = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    return value


def restore_rng(value: dict[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(value["torch_cuda"])


def tensor_tree_hash(value: Any) -> str:
    digest = hashlib.sha256()
    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode()); digest.update(str(tuple(tensor.shape)).encode()); digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item): digest.update(str(key).encode()); visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item: visit(child)
        else: digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def atomic_checkpoint(run_dir: Path, method_id: str, step: int, state: dict[str, Any]) -> Path:
    root = run_dir / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"step_{step:05d}"
    if destination.exists():
        raise FileExistsError(destination)
    stage = root / f".step_{step:05d}.{uuid.uuid4().hex}.tmp"
    stage.mkdir()
    atomic_torch_save(stage / "state.pt", state)
    atomic_json(stage / "manifest.json", {
        "schema": "paper-baseline-checkpoint-v1",
        "method_id": method_id,
        "step": step,
        "state_sha256": sha256_file(stage / "state.pt"),
        "contract_sha256": state["contract_sha256"],
        "published_atomically": True,
        "test_accessed": False,
    })
    atomic_publish_directory(stage, destination)
    return destination


def latest_checkpoint(run_dir: Path, method_id: str, contract_sha256: str) -> tuple[Path, dict[str, Any]]:
    candidates = sorted((run_dir / "checkpoints").glob("step_*")) if (run_dir / "checkpoints").is_dir() else []
    if not candidates:
        raise FileNotFoundError("Resume requires a published checkpoint")
    path = candidates[-1]
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    state_path = path / "state.pt"
    if manifest.get("schema") != "paper-baseline-checkpoint-v1" or manifest.get("method_id") != method_id:
        raise ValueError("checkpoint method/schema mismatch")
    if manifest.get("contract_sha256") != contract_sha256 or manifest.get("state_sha256") != sha256_file(state_path):
        raise ValueError("checkpoint contract/SHA mismatch")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state.get("contract_sha256") != contract_sha256 or state.get("test_accessed") is not False:
        raise ValueError("checkpoint payload mismatch")
    return path, state


def memory_snapshot() -> dict[str, Any]:
    value: dict[str, Any] = {
        "process_cpu_rss": psutil.Process().memory_info().rss,
        "cuda_allocated": 0,
        "cuda_reserved": 0,
        "cuda_peak_allocated": 0,
        "cuda_peak_reserved": 0,
        "dedicated_used": None,
        "dedicated_free": None,
        "shared_memory_risk": False,
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        value.update({
            "cuda_allocated": torch.cuda.memory_allocated(),
            "cuda_reserved": torch.cuda.memory_reserved(),
            "cuda_peak_allocated": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved": torch.cuda.max_memory_reserved(),
            "dedicated_used": total - free,
            "dedicated_free": free,
        })
    return value


def enforce_memory_guard(config: dict[str, Any]) -> dict[str, Any]:
    snapshot = memory_snapshot()
    if not torch.cuda.is_available():
        return {**snapshot, "passed": True, "cleanup_retried": False}
    _, total = torch.cuda.mem_get_info()
    passed = snapshot["cuda_reserved"] <= total * config["max_reserved_fraction"] and snapshot["dedicated_free"] >= config["minimum_dedicated_free_bytes"]
    if not passed:
        gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache(); snapshot = memory_snapshot()
        passed = snapshot["cuda_reserved"] <= total * config["max_reserved_fraction"] and snapshot["dedicated_free"] >= config["minimum_dedicated_free_bytes"]
        snapshot["cleanup_retried"] = True
    else:
        snapshot["cleanup_retried"] = False
    snapshot["passed"] = bool(passed)
    if not passed:
        raise SafeStop("memory_high_watermark", snapshot)
    return snapshot


def cleanup_cuda(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize(); torch.cuda.empty_cache()


def deterministic_order(length: int, seed: int, epoch: int, stream: int = 0) -> list[int]:
    generator = random.Random(seed + 1_000_003 * epoch + 97 * stream)
    order = list(range(length)); generator.shuffle(order); return order


def logical_batches(length: int, batch_size: int, seed: int, epoch: int, stream: int = 0, wrap: bool = True) -> Iterator[list[int]]:
    order = deterministic_order(length, seed, epoch, stream)
    for start in range(0, length, batch_size):
        batch = order[start:start + batch_size]
        if len(batch) < batch_size and wrap:
            batch += deterministic_order(length, seed, epoch + 1, stream)[:batch_size - len(batch)]
        yield batch


def logical_batch_at(length: int, batch_size: int, seed: int, epoch: int, position: int, stream: int = 0) -> list[int]:
    count = (length + batch_size - 1) // batch_size
    if position < 0 or position >= count:
        raise IndexError("logical batch position out of range")
    current = deterministic_order(length, seed, epoch, stream)
    start = position * batch_size
    result = current[start:start + batch_size]
    if len(result) < batch_size:
        result += deterministic_order(length, seed, epoch + 1, stream)[:batch_size - len(result)]
    return result


def batch_hash(indices: Iterable[int]) -> str:
    return canonical_hash(list(indices))


def build_contract(root: Path, config_path: Path, config: dict[str, Any], method_id: str, implementation_files: list[Path]) -> dict[str, Any]:
    sources = validate_source_files(root, config["sources"])
    value = {
        "schema": "paper-baseline-contract-v1",
        "method_id": method_id,
        "config_sha256": sha256_file(config_path),
        "git": git_snapshot(root),
        "implementation": {str(path.relative_to(root)).replace("\\", "/"): sha256_file(path) for path in implementation_files},
        "sources": sources,
        "seed": config["seed"],
        "test_accessed": False,
    }
    if config.get("deletion_experiment") is not None:value["deletion_experiment"] = config["deletion_experiment"]
    return value


def preflight(root: Path, config_path: Path, schema: str, method_id: str, implementation_files: list[Path], formal: bool = False) -> dict[str, Any]:
    config = load_config(config_path, schema)
    contract = build_contract(root, config_path, config, method_id, implementation_files)
    result = {
        "schema": schema,
        "mode": "Preflight",
        "method_id": method_id,
        "contract": contract,
        "model_loaded": False,
        "optimizer_constructed": False,
        "test_loader_built": False,
        "test_accessed": False,
    }
    if formal:
        result["formal"] = require_formal_preflight(root)
    return result


def paper_model_manifest(
    *, method_id: str, display_name: str, run_name: str, model_type: str,
    artifacts: list[Path], root: Path, contract: dict[str, Any], config_sha256: str,
    trainable_parameters: int, total_parameters: int, optimizer_steps: int,
    timing: dict[str, Any], resources: dict[str, Any], completion_marker: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_rows = []
    for path in artifacts:
        artifact_rows.append({"path": str(path.resolve()), "sha256": directory_hash(path) if path.is_dir() else sha256_file(path), "type": "directory" if path.is_dir() else "file"})
    value = {
        "schema": MODEL_MANIFEST_SCHEMA,
        "method_id": method_id,
        "display_name": display_name,
        "run_name": run_name,
        "model_type": model_type,
        "artifacts": artifact_rows,
        "source_original_sha256": contract["sources"]["original"]["sha256"],
        "configuration_sha256": config_sha256,
        "git_head": contract["git"]["head"],
        "implementation_sha256": canonical_hash(contract["implementation"]),
        "training_split": "Development-free training data only",
        "training_data_lineage": contract["sources"],
        "trainable_parameters": int(trainable_parameters),
        "total_parameters": int(total_parameters),
        "optimizer_steps": int(optimizer_steps),
        "wall_time_seconds": timing["end_to_end_wall_seconds"],
        "training_time_seconds": timing["training_seconds"],
        "augmentation_time_seconds": timing.get("augmentation_seconds", 0.0),
        "resources": resources,
        "test_accessed": False,
        "completion_marker": completion_marker,
        **(extra or {}),
    }
    verify_paper_model_manifest(value)
    return value


def verify_paper_model_manifest(value: dict[str, Any], *, verify_artifacts: bool = False) -> dict[str, Any]:
    required = {
        "schema", "method_id", "display_name", "run_name", "model_type", "artifacts",
        "source_original_sha256", "configuration_sha256", "git_head", "implementation_sha256",
        "training_split", "training_data_lineage", "trainable_parameters", "total_parameters",
        "optimizer_steps", "wall_time_seconds", "training_time_seconds", "augmentation_time_seconds",
        "resources", "test_accessed", "completion_marker",
    }
    if value.get("schema") != MODEL_MANIFEST_SCHEMA or required - set(value):
        raise ValueError("paper model manifest schema incomplete")
    if value.get("test_accessed") is not False or not value.get("method_id") or not value.get("run_name"):
        raise ValueError("paper model manifest safety/identity invalid")
    if value["trainable_parameters"] < 0 or value["total_parameters"] <= 0 or value["optimizer_steps"] < 0:
        raise ValueError("paper model manifest counters invalid")
    for field in ("wall_time_seconds", "training_time_seconds", "augmentation_time_seconds"):
        if not isinstance(value[field], (int, float)) or not math.isfinite(value[field]) or value[field] < 0:
            raise ValueError(f"invalid manifest timing: {field}")
    if verify_artifacts:
        for artifact in value["artifacts"]:
            path = Path(artifact["path"])
            actual = directory_hash(path) if artifact["type"] == "directory" else sha256_file(path)
            if actual != artifact["sha256"]:
                raise ValueError("model artifact SHA mismatch")
    return value


def write_model_manifest(run_dir: Path, value: dict[str, Any]) -> None:
    atomic_json(run_dir / "paper_model_manifest.json", value)


def stopped_safely(run_dir: Path, stop: SafeStop) -> dict[str, Any]:
    value = {"status": "STOPPED_SAFELY", "reason": stop.reason, "evidence": stop.evidence, "test_accessed": False}
    atomic_json(run_dir / "run_state.json", value)
    atomic_text(run_dir / "STOPPED_SAFELY", "PAPER_BASELINE_STOPPED_SAFELY\n")
    return value


def analyze_run(run_dir: Path, expected_marker: str) -> dict[str, Any]:
    state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    if state.get("test_accessed") is not False:
        raise ValueError("test invariant failed")
    if state.get("status") == "COMPLETED":
        if (run_dir / "COMPLETED").read_text(encoding="utf-8") != expected_marker + "\n":
            raise ValueError("completion marker invalid")
        manifest = json.loads((run_dir / "paper_model_manifest.json").read_text(encoding="utf-8"))
        verify_paper_model_manifest(manifest, verify_artifacts=True)
    elif state.get("status") != "STOPPED_SAFELY":
        raise ValueError("run is not analyzable")
    return {"status": state["status"], "method_id": state.get("method_id"), "test_accessed": False}


@dataclass(frozen=True)
class StageTimer:
    started: float
    @classmethod
    def start(cls) -> "StageTimer": return cls(time.perf_counter())
    def elapsed(self) -> float: return time.perf_counter() - self.started
