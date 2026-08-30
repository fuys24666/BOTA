from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from peft import set_peft_model_state_dict
from transformers import T5Tokenizer

from src.diagnostics.ml1m_development_protocol import (
    formal_forget_users,
    reconstruct_authoritative_rows,
    reject_processed_test_path,
)
from src.diagnostics.t5_reconstructed_official import (
    JsonPromptDataset,
    build_current_model,
    freeze_teacher,
    load_config,
    load_legacy_model,
    move_batch,
    sha256_file,
)

SCHEMA = "t5-e2urec-zero-training-audit-v2"
CACHE_SCHEMA = "t5-e2urec-zero-training-cache-v2"
OUTPUT_NAME = "t5_e2urec_zero_training_audit_v1"
STEPS = (812, 813, 850, 900, 1000, 1200)
REFERENCE_MODELS = ("original", "augmented", "retrain")
SPLITS = (
    "forget_train",
    "retain_train",
    "forget_user_validation",
    "retain_user_validation",
    "overall_validation",
)
PHYSICAL_SPLITS = ("forget_train", "retain_train", "overall_validation")
DERIVED_VALIDATION_SPLITS = (
    "forget_user_validation",
    "retain_user_validation",
)
EXPECTED_COUNTS = {
    "forget_train": 12_982,
    "retain_train": 47_018,
    "forget_user_validation": 3_336,
    "retain_user_validation": 16_664,
    "overall_validation": 20_000,
}


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:12]}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_audit_config(path: Path, project_root: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        value.get("schema") != SCHEMA
        or value.get("development_only") is not True
        or tuple(value.get("checkpoint_steps", ())) != STEPS
        or value.get("dry_run", {}).get("max_samples_per_split") != 2
        or tuple(value.get("physical_splits", ())) != PHYSICAL_SPLITS
        or tuple(value.get("derived_validation_splits", ()))
        != DERIVED_VALIDATION_SPLITS
        or value.get("cache_schema") != CACHE_SCHEMA
    ):
        raise ValueError("zero-training audit config differs from preregistration")
    value["_path"] = str(path.resolve())
    value["_sha256"] = sha256_file(path)
    value["_root"] = str(project_root.resolve())
    return value


def _resolve(root: Path, path: str | Path) -> Path:
    resolved = (root / Path(path)).resolve()
    if root.resolve() not in resolved.parents and resolved != root.resolve():
        raise ValueError(f"path escapes project root: {resolved}")
    reject_processed_test_path(resolved)
    return resolved


def _checkpoint_record(run: Path, step: int) -> dict[str, Any]:
    directory = run / "checkpoints" / f"step_{step:05d}"
    state, manifest = directory / "state.pt", directory / "manifest.json"
    prediction = run / f"development_step_{step:05d}.json"
    record: dict[str, Any] = {
        "step": step,
        "checkpoint": None,
        "prediction": None,
        "status": "unavailable",
        "reason": [],
    }
    if state.is_file() and manifest.is_file():
        value = json.loads(manifest.read_text(encoding="utf-8"))
        digest = sha256_file(state)
        if value.get("state_sha256") == digest and value.get("step") == step:
            record["checkpoint"] = {
                "directory": str(directory.resolve()),
                "state_path": str(state.resolve()),
                "state_sha256": digest,
                "manifest_path": str(manifest.resolve()),
                "manifest_sha256": sha256_file(manifest),
            }
        else:
            record["reason"].append("checkpoint_manifest_or_sha256_invalid")
    else:
        record["reason"].append("checkpoint_not_published")
    if prediction.is_file():
        value = json.loads(prediction.read_text(encoding="utf-8"))
        if value.get("test_accessed") is False:
            record["prediction"] = {
                "path": str(prediction.resolve()),
                "sha256": sha256_file(prediction),
            }
        else:
            record["reason"].append("prediction_not_test_free")
    else:
        record["reason"].append("prediction_not_published")
    if record["checkpoint"] is not None:
        record["status"] = "available"
    return record


def _data_lineage(
    root: Path, base: dict[str, Any], protocol_root: Path
) -> tuple[dict[str, Any], dict[str, list[int]], dict[str, list[int]]]:
    manifest_path = protocol_root / "development_protocol_manifest.json"
    forget_manifest_path = protocol_root / "forget_user_manifest.json"
    sidecar_path = protocol_root / "validation_user_sidecar.jsonl"
    forget_indices_path = protocol_root / "forget_validation_indices.json"
    retain_indices_path = protocol_root / "retain_validation_indices.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    forget_manifest = json.loads(forget_manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("test_accessed") is not False
        or manifest.get("processed_test_split_read") is not False
        or sha256_file(sidecar_path)
        != manifest["output_sha256"]["validation_user_sidecar.jsonl"]
        or sha256_file(forget_manifest_path)
        != manifest["output_sha256"]["forget_user_manifest.json"]
    ):
        raise ValueError("authoritative development protocol hash/safety mismatch")
    train_rows, validation_rows, replay = reconstruct_authoritative_rows(
        root / "data" / "ml-1m" / "raw_data"
    )
    formal_order, forget_users = formal_forget_users(train_rows)
    if formal_order != forget_manifest["user_ids"]:
        raise ValueError("formal Forget user order differs from manifest")
    forget_train = [
        row for row in train_rows if row.authoritative_user_id in forget_users
    ]
    retain_train = [
        row for row in train_rows if row.authoritative_user_id not in forget_users
    ]
    sidecar = [
        json.loads(line)
        for line in sidecar_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if (
        len(forget_train) != EXPECTED_COUNTS["forget_train"]
        or len(retain_train) != EXPECTED_COUNTS["retain_train"]
        or len(sidecar) != EXPECTED_COUNTS["overall_validation"]
    ):
        raise ValueError("authoritative train/validation lineage count mismatch")
    if any(
        item["authoritative_user_id"] != row.authoritative_user_id
        for item, row in zip(sidecar, validation_rows)
    ):
        raise ValueError("validation sidecar user mapping mismatch")
    forget_indices = json.loads(
        forget_indices_path.read_text(encoding="utf-8")
    )["indices"]
    retain_indices = json.loads(
        retain_indices_path.read_text(encoding="utf-8")
    )["indices"]
    if (
        len(forget_indices) != EXPECTED_COUNTS["forget_user_validation"]
        or len(retain_indices) != EXPECTED_COUNTS["retain_user_validation"]
        or set(forget_indices) & set(retain_indices)
        or sorted(forget_indices + retain_indices) != list(range(20_000))
    ):
        raise ValueError("validation split partition mismatch")
    forget_validation_users = {
        sidecar[index]["authoritative_user_id"] for index in forget_indices
    }
    retain_validation_users = {
        sidecar[index]["authoritative_user_id"] for index in retain_indices
    }
    if (
        forget_validation_users & retain_validation_users
        or not forget_validation_users <= forget_users
        or retain_validation_users & forget_users
    ):
        raise ValueError("Forget/Retain user groups are mixed")
    paths = {
        name: _resolve(root, base["paths"][key])
        for name, key in (
            ("forget_train", "forget"),
            ("retain_train", "retain"),
            ("overall_validation", "validation"),
        )
    }
    for name, path in paths.items():
        expected_key = {
            "forget_train": "processed/forget",
            "retain_train": "processed/retain",
            "overall_validation": "processed/validation",
        }[name]
        if sha256_file(path) != manifest["input_sha256"][expected_key]:
            raise ValueError(f"{name} SHA256 differs from authoritative protocol")
    user_ids = {
        "forget_train": [row.authoritative_user_id for row in forget_train],
        "retain_train": [row.authoritative_user_id for row in retain_train],
        "overall_validation": [
            int(item["authoritative_user_id"]) for item in sidecar
        ],
    }
    raw_rows = {
        "forget_train": [row.raw_source_row for row in forget_train],
        "retain_train": [row.raw_source_row for row in retain_train],
        "overall_validation": [row.raw_source_row for row in validation_rows],
    }
    if set(raw_rows["forget_train"]) & set(raw_rows["retain_train"]):
        raise ValueError("train member partitions overlap")
    if (
        set(raw_rows["forget_train"]) & set(raw_rows["overall_validation"])
        or set(raw_rows["retain_train"]) & set(raw_rows["overall_validation"])
    ):
        raise ValueError("train and validation source rows overlap")
    lineage = {
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
        },
        "forget_user_manifest": {
            "path": str(forget_manifest_path.resolve()),
            "sha256": sha256_file(forget_manifest_path),
            "users": len(forget_users),
        },
        "validation_sidecar": {
            "path": str(sidecar_path.resolve()),
            "sha256": sha256_file(sidecar_path),
            "rows": len(sidecar),
        },
        "data": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "samples": EXPECTED_COUNTS[name],
            }
            for name, path in paths.items()
        },
        "validation_splits": {
            "forget_user_validation": {
                "samples": len(forget_indices),
                "users": len(forget_validation_users),
                "indices_sha256": sha256_file(forget_indices_path),
            },
            "retain_user_validation": {
                "samples": len(retain_indices),
                "users": len(retain_validation_users),
                "indices_sha256": sha256_file(retain_indices_path),
            },
        },
        "matched_user_mia": {
            "forget": {
                "train_users": len(
                    {row.authoritative_user_id for row in forget_train}
                ),
                "validation_users": len(forget_validation_users),
                "intersection_users": len(
                    {row.authoritative_user_id for row in forget_train}
                    & forget_validation_users
                ),
                "excluded_train_only_users": len(
                    {row.authoritative_user_id for row in forget_train}
                    - forget_validation_users
                ),
                "excluded_validation_only_users": len(
                    forget_validation_users
                    - {row.authoritative_user_id for row in forget_train}
                ),
                "matched_member_samples": sum(
                    row.authoritative_user_id in forget_validation_users
                    for row in forget_train
                ),
                "matched_nonmember_samples": len(forget_indices),
            },
            "retain": {
                "train_users": len(
                    {row.authoritative_user_id for row in retain_train}
                ),
                "validation_users": len(retain_validation_users),
                "intersection_users": len(
                    {row.authoritative_user_id for row in retain_train}
                    & retain_validation_users
                ),
                "excluded_train_only_users": len(
                    {row.authoritative_user_id for row in retain_train}
                    - retain_validation_users
                ),
                "excluded_validation_only_users": len(
                    retain_validation_users
                    - {row.authoritative_user_id for row in retain_train}
                ),
                "matched_member_samples": sum(
                    row.authoritative_user_id in retain_validation_users
                    for row in retain_train
                ),
                "matched_nonmember_samples": len(retain_indices),
            },
        },
        "partition_checks": {
            "train_mutually_exclusive": True,
            "validation_mutually_exclusive": True,
            "train_validation_source_rows_disjoint": True,
            "user_manifest_consistent": True,
        },
        "raw_replay": replay,
        "processed_test_split_read": False,
        "test_accessed": False,
    }
    indices = {
        "forget_train": list(range(len(forget_train))),
        "retain_train": list(range(len(retain_train))),
        "forget_user_validation": forget_indices,
        "retain_user_validation": retain_indices,
        "overall_validation": list(range(20_000)),
    }
    user_split = {
        "forget_train": user_ids["forget_train"],
        "retain_train": user_ids["retain_train"],
        "forget_user_validation": [
            user_ids["overall_validation"][index] for index in forget_indices
        ],
        "retain_user_validation": [
            user_ids["overall_validation"][index] for index in retain_indices
        ],
        "overall_validation": user_ids["overall_validation"],
    }
    return lineage, indices, user_split


def preflight(
    project_root: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[int]], dict[str, list[int]]]:
    audit = load_audit_config(config_path, project_root)
    base_path = _resolve(project_root, audit["base_config"])
    base = load_config(base_path, project_root)
    protocol_root = _resolve(project_root, audit["protocol_root"])
    lineage, indices, users = _data_lineage(
        project_root, base, protocol_root
    )
    model_files = {
        role: {
            "path": str(_resolve(project_root, base["paths"][key])),
            "sha256": sha256_file(_resolve(project_root, base["paths"][key])),
        }
        for role, key in (
            ("original", "original"),
            ("augmented", "augmented_teacher"),
            ("retrain", "retrain_reference"),
        )
    }
    runs: dict[str, Any] = {}
    for name, relative in audit["models"].items():
        run = _resolve(project_root, relative)
        state_path, contract_path = run / "run_state.json", run / "contract.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if (
            state.get("status") != "COMPLETED"
            or state.get("test_accessed") is not False
            or contract.get("test_used", contract.get("test_accessed")) is not False
        ):
            raise ValueError(f"{name} is not a completed test-free run")
        runs[name] = {
            "run_path": str(run),
            "run_state_sha256": sha256_file(state_path),
            "contract_sha256": sha256_file(contract_path),
            "status": state["status"],
            "checkpoints": {
                str(step): _checkpoint_record(run, step) for step in STEPS
            },
            "test_accessed": False,
        }
    result = {
        "schema": SCHEMA,
        "mode": "Preflight",
        "config_sha256": audit["_sha256"],
        "models": model_files,
        "runs": runs,
        "lineage": lineage,
        "thresholds_preregistered": audit["success_thresholds"],
        "formal_output_created": False,
        "model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_steps_executed": 0,
        "processed_test_split_read": False,
        "test_loader_built": False,
        "test_accessed": False,
    }
    return result, base, indices, users


def _dataset_bundle(
    base: dict[str, Any], tokenizer: T5Tokenizer
) -> dict[str, JsonPromptDataset]:
    validation = JsonPromptDataset(Path(base["paths"]["validation"]), tokenizer)
    return {
        "forget_train": JsonPromptDataset(Path(base["paths"]["forget"]), tokenizer),
        "retain_train": JsonPromptDataset(Path(base["paths"]["retain"]), tokenizer),
        "forget_user_validation": validation,
        "retain_user_validation": validation,
        "overall_validation": validation,
    }


def _model_specifications(
    preflight_result: dict[str, Any], dry_run: bool
) -> list[dict[str, Any]]:
    specs = [
        {
            "name": role,
            "kind": "reference",
            "model_path": preflight_result["models"][role]["path"],
            "model_sha256": preflight_result["models"][role]["sha256"],
            "checkpoint_step": None,
        }
        for role in REFERENCE_MODELS
    ]
    for family in ("j0", "j2", "j4", "j5"):
        for step in STEPS:
            item = preflight_result["runs"][family]["checkpoints"][str(step)]
            if item["status"] == "available":
                specs.append(
                    {
                        "name": f"{family}_step{step}",
                        "kind": "adapter",
                        "family": family,
                        "checkpoint_step": step,
                        "state_path": item["checkpoint"]["state_path"],
                        "model_sha256": item["checkpoint"]["state_sha256"],
                    }
                )
    if dry_run:
        names = {"original", "retrain", "j0_step1200", "j5_step1200"}
        specs = [spec for spec in specs if spec["name"] in names]
    return specs


def _load_audit_model(
    spec: dict[str, Any], base: dict[str, Any], device: torch.device
) -> torch.nn.Module:
    if spec["kind"] == "reference":
        model = load_legacy_model(Path(spec["model_path"]))
    else:
        payload = torch.load(
            spec["state_path"], map_location="cpu", weights_only=False
        )
        adapter = payload.get("adapter_state")
        if not isinstance(adapter, dict) or not adapter:
            raise ValueError(f"{spec['name']} checkpoint lacks adapter state")
        model = build_current_model(Path(base["paths"]["original"]), base["lora"])
        result = set_peft_model_state_dict(model, adapter)
        if getattr(result, "unexpected_keys", []):
            raise ValueError(f"{spec['name']} adapter unexpected keys")
    model = freeze_teacher(model).to(device)
    if model.config._attn_implementation_internal != "eager":
        raise ValueError("audit runtime must use eager attention")
    return model


def _records_for_unit(
    model: torch.nn.Module,
    dataset: JsonPromptDataset,
    source_indices: list[int],
    user_ids: list[int],
    split: str,
    spec: dict[str, Any],
    data_sha256: str,
    device: torch.device,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    forward_calls = 0
    with torch.inference_mode():
        for start in range(0, len(source_indices), 16):
            local = source_indices[start : start + 16]
            batch = move_batch(
                dataset.collate_fn([dataset[index] for index in local]), device
            )
            output = model(
                input_ids=batch["input_ids"], labels=batch["target_ids"]
            )
            forward_calls += 1
            logits = output.logits.float()
            pair = torch.softmax(logits[:, 0, [465, 2163]], dim=-1)
            targets = batch["target_ids"]
            token_loss = F.cross_entropy(
                logits.transpose(1, 2), targets, reduction="none", ignore_index=-100
            )
            mask = targets.ne(-100)
            losses = (token_loss * mask).sum(1) / mask.sum(1).clamp_min(1)
            for offset, source_index in enumerate(local):
                p_no, p_yes = map(float, pair[offset].cpu())
                gold = int(targets[offset, 0].item() == 2163)
                rows.append(
                    {
                        "canonical_sample_id": f"{split}:{source_index}",
                        "source_index": source_index,
                        "user_id": int(user_ids[start + offset]),
                        "split": split,
                        "gold_yes": gold,
                        "p_yes": p_yes,
                        "p_no": p_no,
                        "predicted_yes": int(p_yes >= 0.5),
                        "answer_sequence_loss": float(losses[offset].cpu()),
                        "confidence": max(p_yes, p_no),
                        "binary_entropy": float(
                            -(p_yes * math.log(max(p_yes, 1e-12))
                              + p_no * math.log(max(p_no, 1e-12)))
                        ),
                        "yes_no_margin": float(
                            (logits[offset, 0, 2163] - logits[offset, 0, 465]).cpu()
                        ),
                        "source_model": spec["name"],
                        "checkpoint_step": spec["checkpoint_step"],
                        "model_checkpoint_sha256": spec["model_sha256"],
                        "data_sha256": data_sha256,
                        "test_accessed": False,
                    }
                )
    return rows, forward_calls


def _cache_paths(root: Path, model: str, split: str) -> tuple[Path, Path]:
    directory = root / "caches" / model
    return directory / f"{split}.jsonl", directory / f"{split}.manifest.json"


def _validate_cache(data_path: Path, manifest_path: Path, expected: dict[str, Any]) -> bool:
    if not data_path.is_file() or not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return all(manifest.get(key) == value for key, value in expected.items()) and (
        manifest.get("data_sha256") == sha256_file(data_path)
        and manifest.get("published") is True
        and manifest.get("test_accessed") is False
    )


def _publish_cache(
    data_path: Path,
    manifest_path: Path,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    if data_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite cache unit")
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    )
    atomic_text(data_path, text)
    manifest = {
        **metadata,
        "data_sha256": sha256_file(data_path),
        "rows": len(rows),
        "unique_sample_ids": len({row["canonical_sample_id"] for row in rows}),
        "sample_order_hash": canonical_hash(
            [row["canonical_sample_id"] for row in rows]
        ),
        "published": True,
        "full_vocabulary_logits_persisted": False,
        "test_accessed": False,
    }
    try:
        atomic_json(manifest_path, manifest)
    except BaseException:
        if data_path.exists():
            data_path.unlink()
        raise


def _load_published_rows(
    data_path: Path, manifest_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not data_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("published cache pair is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("published") is not True
        or manifest.get("data_sha256") != sha256_file(data_path)
        or manifest.get("test_accessed") is not False
    ):
        raise ValueError("parent cache is invalid")
    rows = [
        json.loads(line)
        for line in data_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if (
        len(rows) != manifest.get("rows")
        or len({row["canonical_sample_id"] for row in rows}) != len(rows)
    ):
        raise ValueError("parent cache rows are incomplete or duplicated")
    return rows, manifest


def _derive_validation_cache(
    run_dir: Path,
    model_name: str,
    split: str,
    authoritative_indices: list[int],
    authoritative_index_path: Path,
) -> str:
    if split not in DERIVED_VALIDATION_SPLITS:
        raise ValueError("only logical validation splits may be derived")
    parent_data, parent_manifest_path = _cache_paths(
        run_dir, model_name, "overall_validation"
    )
    parent_rows, _ = _load_published_rows(parent_data, parent_manifest_path)
    parent_sha = sha256_file(parent_data)
    index_set = set(authoritative_indices)
    rows = [row for row in parent_rows if row["source_index"] in index_set]
    for row in rows:
        if row["split"] != "overall_validation":
            raise ValueError("derived parent row has wrong split")
    # Preserve every parent row field byte-for-byte at the semantic value level.
    # The logical split identity belongs to the child manifest, not the prediction.
    derived_rows = [dict(row) for row in rows]
    data_path, manifest_path = _cache_paths(run_dir, model_name, split)
    expected = {
        "schema": CACHE_SCHEMA,
        "source_model": model_name,
        "split": split,
        "derived_from": "overall_validation",
        "parent_cache_path": str(parent_data.resolve()),
        "parent_cache_sha256": parent_sha,
        "authoritative_index_path": str(authoritative_index_path.resolve()),
        "authoritative_index_sha256": sha256_file(authoritative_index_path),
        "expected_rows": len(derived_rows),
        "model_forward_performed": False,
        "tokenization_performed": False,
        "runtime_device": "cpu",
    }
    if _validate_cache(data_path, manifest_path, expected):
        return "skipped"
    if data_path.exists() or manifest_path.exists():
        raise ValueError("partial/invalid derived cache blocks Resume")
    _publish_cache(data_path, manifest_path, derived_rows, expected)
    loaded, manifest = _load_published_rows(data_path, manifest_path)
    if loaded != derived_rows or manifest["parent_cache_sha256"] != parent_sha:
        raise RuntimeError("derived validation cache verification failed")
    return "completed"


def run_inference(
    project_root: Path,
    config_path: Path,
    run_name: str,
    *,
    dry_run: bool,
    resume: bool = False,
) -> dict[str, Any]:
    pre, base, split_indices, split_users = preflight(project_root, config_path)
    audit = load_audit_config(config_path, project_root)
    output_root = _resolve(project_root, audit["output_root"])
    category = "dry_runs" if dry_run else "full_runs"
    run_dir = output_root / category / run_name
    if dry_run and resume:
        raise ValueError("DryRun is not resumed")
    if not resume:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"refusing non-empty run: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
    elif not run_dir.is_dir():
        raise FileNotFoundError("Resume requires an existing Full run")
    state_path = run_dir / "run_state.json"
    contract = {
        "schema": SCHEMA,
        "mode": "DryRun" if dry_run else "Full",
        "run_name": run_name,
        "config_sha256": audit["_sha256"],
        "preflight_sha256": canonical_hash(pre),
        "success_thresholds": audit["success_thresholds"],
        "cache_schema": CACHE_SCHEMA,
        "physical_split_plan": list(PHYSICAL_SPLITS),
        "derived_validation_split_plan": list(DERIVED_VALIDATION_SPLITS),
        "matched_user_protocol_version": audit[
            "matched_user_protocol_version"
        ],
        "mia_primary_score": audit["mia_primary_score"],
        "bootstrap_seed": audit["seed"],
        "bootstrap_resamples": audit["bootstrap_resamples"],
        "max_samples_per_split": 2 if dry_run else None,
        "optimizer_constructed": False,
        "optimizer_steps_executed": 0,
        "test_accessed": False,
    }
    contract_path = run_dir / "contract.json"
    if resume:
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise ValueError("Resume contract mismatch")
    else:
        atomic_json(contract_path, contract)
    atomic_json(
        state_path,
        {
            "status": "RUNNING",
            "mode": contract["mode"],
            "optimizer_steps_executed": 0,
            "test_accessed": False,
        },
    )
    tokenizer = T5Tokenizer.from_pretrained(base["paths"]["model_dir"])
    datasets = _dataset_bundle(base, tokenizer)
    specs = _model_specifications(pre, dry_run)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    immutable_paths = {
        Path(spec["model_path"] if spec["kind"] == "reference" else spec["state_path"])
        for spec in specs
    }
    immutable_paths.update(
        Path(item["path"]) for item in pre["lineage"]["data"].values()
    )
    immutable_before = {
        str(path.resolve()): sha256_file(path)
        for path in sorted(immutable_paths, key=str)
    }
    calls = 0
    completed, skipped = [], []
    try:
        for spec in specs:
            model = _load_audit_model(spec, base, device)
            try:
                for split in PHYSICAL_SPLITS:
                    count = min(2, len(split_indices[split])) if dry_run else len(
                        split_indices[split]
                    )
                    if dry_run and split == "overall_validation":
                        selected = [
                            split_indices["forget_user_validation"][0],
                            split_indices["retain_user_validation"][0],
                        ][:count]
                        selected_users = [
                            split_users["overall_validation"][index]
                            for index in selected
                        ]
                    else:
                        selected = split_indices[split][:count]
                        selected_users = split_users[split][:count]
                    data_path, manifest_path = _cache_paths(
                        run_dir, spec["name"], split
                    )
                    expected = {
                        "schema": CACHE_SCHEMA,
                        "source_model": spec["name"],
                        "split": split,
                        "model_checkpoint_sha256": spec["model_sha256"],
                        "expected_rows": count,
                    }
                    if _validate_cache(data_path, manifest_path, expected):
                        skipped.append(f"{spec['name']}:{split}")
                        continue
                    if data_path.exists() or manifest_path.exists():
                        raise ValueError("partial or invalid cache blocks Resume")
                    rows, unit_calls = _records_for_unit(
                        model,
                        datasets[split],
                        selected,
                        selected_users,
                        split,
                        spec,
                        pre["lineage"]["data"][
                            "overall_validation"
                            if "validation" in split
                            else split
                        ]["sha256"],
                        device,
                    )
                    calls += unit_calls
                    _publish_cache(
                        data_path,
                        manifest_path,
                        rows,
                        {
                            **expected,
                            "expected_rows": count,
                            "forward_calls": unit_calls,
                        },
                    )
                    completed.append(f"{spec['name']}:{split}")
                protocol_root = _resolve(project_root, audit["protocol_root"])
                for split, index_name in (
                    ("forget_user_validation", "forget_validation_indices.json"),
                    ("retain_user_validation", "retain_validation_indices.json"),
                ):
                    status = _derive_validation_cache(
                        run_dir,
                        spec["name"],
                        split,
                        split_indices[split],
                        protocol_root / index_name,
                    )
                    target = completed if status == "completed" else skipped
                    target.append(f"{spec['name']}:{split}:derived")
            finally:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        immutable_after = {
            str(path.resolve()): sha256_file(path)
            for path in sorted(immutable_paths, key=str)
        }
        if immutable_after != immutable_before:
            raise RuntimeError("source model/checkpoint/data hashes changed")
        status = "DRY_RUN_COMPLETED" if dry_run else "INFERENCE_COMPLETED"
        final = {
            "status": status,
            "mode": contract["mode"],
            "models": [spec["name"] for spec in specs],
            "splits": list(SPLITS),
            "physical_splits": list(PHYSICAL_SPLITS),
            "derived_validation_splits": list(DERIVED_VALIDATION_SPLITS),
            "samples_per_split": 2 if dry_run else EXPECTED_COUNTS,
            "cache_units_completed": completed,
            "cache_units_skipped": skipped,
            "unavailable_checkpoints": {
                family: {
                    step: record
                    for step, record in value["checkpoints"].items()
                    if record["status"] == "unavailable"
                }
                for family, value in pre["runs"].items()
            },
            "model_forward_calls": calls,
            "physical_forward_cache_units": len(specs) * len(PHYSICAL_SPLITS),
            "derived_cpu_cache_units": len(specs)
            * len(DERIVED_VALIDATION_SPLITS),
            "immutable_source_sha256_before": immutable_before,
            "immutable_source_sha256_after": immutable_after,
            "immutable_sources_unchanged": True,
            "optimizer_constructed": False,
            "optimizer_steps_executed": 0,
            "full_vocabulary_logits_persisted": False,
            "formal_full_directory_created": not dry_run,
            "test_loader_built": False,
            "test_accessed": False,
        }
        atomic_json(state_path, final)
        return {**final, "run_dir": str(run_dir.resolve())}
    except BaseException as error:
        atomic_json(
            state_path,
            {
                "status": "FAILED",
                "error": f"{type(error).__name__}: {error}",
                "model_forward_calls": calls,
                "optimizer_steps_executed": 0,
                "test_accessed": False,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-training T5 audit runtime")
    parser.add_argument("--mode", choices=("Preflight", "DryRun", "Full", "Resume"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-name")
    args = parser.parse_args()
    root, config = args.project_root.resolve(), args.config.resolve()
    if args.mode == "Preflight":
        result, _, _, _ = preflight(root, config)
    else:
        if not args.run_name:
            raise ValueError(f"{args.mode} requires --run-name")
        result = run_inference(
            root,
            config,
            args.run_name,
            dry_run=args.mode == "DryRun",
            resume=args.mode == "Resume",
        )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
