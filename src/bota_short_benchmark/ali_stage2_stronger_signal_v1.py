"""Development-only Stage 2 stronger-signal analysis from Ali Amendment 01."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import shutil
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import T5Tokenizer

from src.bota_if import p1_trajectory_transport_audit as p1
from src.bota_if.p1_trajectory_transport_audit import StepBudget
from src.bota_short_benchmark import runner
from src.bota_short_benchmark import t012_behavior_ablation as behavior
from src.bota_short_benchmark.ali_stage1_mechanism_v1 import (
    SUMMARY_MARKER as STAGE1_SUMMARY_MARKER,
    run_shared_canonical,
)
from src.bota_short_benchmark.protocol import load_config as load_benchmark_config
from src.bota_short_benchmark.protocol import validate_prepared
from src.diagnostics.ml1m_development_protocol import reconstruct_authoritative_rows
from src.diagnostics.t5_reconstructed_official import JsonPromptDataset
from src.paper_baselines.common import capture_rng, restore_rng, tensor_tree_hash
from src.paper_if_a2.common import (
    atomic_json,
    canonical_hash,
    directory_hash,
    git_snapshot,
    safe_run_name,
    seed_everything,
    sha256_file,
)


SCHEMA = "bota-ali-stage2-stronger-signal-v1"
MARKER = "BOTA_ALI_STAGE2_STRONGER_SIGNAL_V1_COMPLETED"
SEED = 42
REQUEST_SIZES = (2, 4, 8)
STRATA = ("single", "moderate", "high")
SELECTED_ARM = "FS_empirical_fisher"
RUN_REQUIRED = {
    "COMPLETED",
    "contract.json",
    "manifest.json",
    "metrics.csv",
    "metrics.json",
    "per_sample_metrics.jsonl",
    "provenance.json",
    "references",
    "report.md",
    "request_registry.json",
    "requests",
    "run_state.json",
    "timing.json",
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_config(repo: Path, workspace: Path, path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "schema",
        "test_access_policy",
        "parent_amendment",
        "stage1_selection",
        "benchmark_config",
        "output_root",
        "protocol",
        "evaluation",
        "decision",
        "runtime",
    }
    if not isinstance(value, dict) or set(value) != required or value["schema"] != SCHEMA:
        raise ValueError("invalid Stage 2 stronger-signal configuration")
    if value["test_access_policy"] != "forbidden":
        raise ValueError("Stage 2 must forbid FinalTest")

    parent = value["parent_amendment"]
    parent_path = repo / parent["path"]
    if not parent_path.is_file() or sha256_file(parent_path) != parent["sha256"]:
        raise ValueError("Stage 0 Amendment 01 binding changed")
    if _load_json(parent_path).get("amendment_id") != parent["amendment_id"]:
        raise ValueError("Stage 0 Amendment 01 identity changed")

    stage1 = value["stage1_selection"]
    stage1_path = workspace / stage1["path"]
    if not stage1_path.is_file() or sha256_file(stage1_path) != stage1["sha256"]:
        raise ValueError("frozen Stage 1 selection summary changed")
    summary = _load_json(stage1_path)
    if (
        summary.get("status") != "COMPLETED"
        or summary.get("selection", {}).get("selected_arm") != SELECTED_ARM
        or stage1.get("selected_arm") != SELECTED_ARM
        or summary.get("final_test_accessed") is not False
        or summary.get("test_accessed") is not False
    ):
        raise ValueError("Stage 1 mechanism selection is not frozen to FS_empirical_fisher")
    summary_dir = stage1_path.parent
    summary_manifest = _load_json(summary_dir / "manifest.json")
    if (summary_dir / "COMPLETED").read_text(encoding="utf-8") != STAGE1_SUMMARY_MARKER + "\n":
        raise ValueError("invalid Stage 1 summary completion marker")
    for name, expected in summary_manifest.get("files", {}).items():
        if sha256_file(summary_dir / name) != expected:
            raise ValueError(f"Stage 1 summary artifact changed: {name}")

    expected_protocol = {
        "selection_seed": 42,
        "optimizer_steps": 200,
        "batch_size": 16,
        "request_sizes": [2, 4, 8],
        "exposure_strata": {
            "single": {"minimum": 1, "maximum": 1},
            "moderate": {"minimum": 2, "maximum": 5},
            "high": {"minimum": 8, "maximum": None},
        },
        "minimum_development_examples_per_user": 10,
        "exclude_primary_users": False,
        "selection_uses_labels_or_predictions": False,
        "ordering": "ascending_sha256_public_user_association_within_stratum",
        "endpoints": ["Original", "Exact-Masked", "BOTA-FS-EmpiricalFisher"],
        "report_every_request": True,
        "online_optimizer_steps": 0,
    }
    if value["protocol"] != expected_protocol:
        raise ValueError("Stage 2 protocol changed")
    if value["evaluation"] != {
        "split": "Development",
        "final_test": False,
        "inference_batch_size": 4,
        "bootstrap_resamples": 1000,
        "bootstrap_seed": 44200,
    }:
        raise ValueError("Stage 2 evaluation contract changed")
    if value["decision"] != {
        "adequate_effect_du_minimum": 0.001,
        "secondary_only": True,
        "may_replace_primary_scenarios": False,
    }:
        raise ValueError("Stage 2 decision contract changed")
    return value


def _public_user_hash(seed: int, stratum: str, user: int) -> str:
    return hashlib.sha256(f"bota-ali-stage2:{seed}:{stratum}:{user}".encode()).hexdigest()


def build_request_registry(
    train_users: Sequence[int],
    development_users: Sequence[int],
    order: Sequence[int],
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[int]]:
    if len(order) != 3200 or len(set(map(int, order))) != 3200:
        raise ValueError("Stage 2 requires the frozen 3,200-row adaptation order")
    global_counts = Counter(map(int, train_users))
    window_counts = Counter(int(train_users[index]) for index in order)
    development_counts = Counter(map(int, development_users))
    strata = protocol["exposure_strata"]
    pools: dict[str, list[int]] = {}
    for stratum in STRATA:
        bounds = strata[stratum]
        minimum = int(bounds["minimum"])
        maximum = bounds["maximum"]
        pool = [
            int(user)
            for user, exposure in window_counts.items()
            if exposure >= minimum
            and (maximum is None or exposure <= int(maximum))
            and development_counts[int(user)]
            >= int(protocol["minimum_development_examples_per_user"])
        ]
        pool.sort(key=lambda user: _public_user_hash(SEED, stratum, user))
        if len(pool) < max(REQUEST_SIZES):
            raise RuntimeError(f"insufficient eligible users in {stratum}: {len(pool)}")
        pools[stratum] = pool

    private_requests: list[dict[str, Any]] = []
    public_requests: list[dict[str, Any]] = []
    for stratum in STRATA:
        for request_size in REQUEST_SIZES:
            users = pools[stratum][:request_size]
            public_members = [
                {
                    "user_hash": _public_user_hash(SEED, stratum, user),
                    "stratum": stratum,
                    "window_exposure": int(window_counts[user]),
                    "development_samples": int(development_counts[user]),
                    "global_train_frequency": int(global_counts[user]),
                }
                for user in users
            ]
            request_id = f"{stratum}_k{request_size}"
            positions = [
                position
                for position, source_index in enumerate(order)
                if int(train_users[source_index]) in set(users)
            ]
            private = {
                "id": request_id,
                "stratum": stratum,
                "request_size": request_size,
                "users": users,
                "forget_window_positions": positions,
                "deleted_interactions": len(positions),
                "public_members": public_members,
                "request_hash": canonical_hash(public_members),
            }
            private_requests.append(private)
            public_requests.append({key: item for key, item in private.items() if key != "users"})

    union_users = sorted(
        {user for request in private_requests for user in request["users"]},
        key=lambda user: canonical_hash([SEED, user]),
    )
    public_registry = {
        "schema": SCHEMA,
        "selection_seed": SEED,
        "ordering": protocol["ordering"],
        "selection_uses_labels_or_predictions": False,
        "eligibility_uses_development_labels": False,
        "pool_sizes": {stratum: len(pools[stratum]) for stratum in STRATA},
        "requests": public_requests,
        "registry_sha256": canonical_hash(public_requests),
        "union_users": len(union_users),
        "development_only": True,
        "final_test_accessed": False,
        "test_accessed": False,
    }
    return public_registry, private_requests, union_users


def _source_users(workspace: Path, benchmark: dict[str, Any]) -> tuple[list[int], list[int], dict[str, Any]]:
    train_users, train_lineage = p1._train_user_ids_only(
        workspace / benchmark["source"]["raw_data"]
    )
    _, development_rows, development_lineage = reconstruct_authoritative_rows(
        workspace / benchmark["source"]["raw_data"]
    )
    development_users = [int(row.authoritative_user_id) for row in development_rows]
    if len(train_users) != 60000 or len(development_users) != 20000:
        raise RuntimeError("Stage 2 authoritative data lineage count mismatch")
    return train_users, development_users, {
        "train": train_lineage,
        "development": development_lineage,
        "processed_test_split_read": False,
        "final_test_rows_materialized": 0,
        "test_accessed": False,
    }


def preflight(
    repo: Path, workspace: Path, config_path: Path
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], list[int]
]:
    config = load_config(repo, workspace, config_path)
    benchmark_path = repo / config["benchmark_config"]
    benchmark = load_benchmark_config(benchmark_path)
    if int(benchmark["protocol"]["run_seed"]) != SEED:
        raise ValueError("Stage 2 benchmark run seed changed")
    benchmark_name = "bota_short_i02_seed42_v3"
    protocol_dir, contract, base_registry = validate_prepared(
        workspace, benchmark, benchmark_name
    )
    train_users, development_users, lineage = _source_users(workspace, benchmark)
    public_registry, private_requests, union_users = build_request_registry(
        train_users, development_users, base_registry["order"], config["protocol"]
    )
    original_run, original_manifest, original_state = behavior._validate_reference(
        workspace,
        benchmark,
        "Original-Short",
        "bota_short_original_seed42_v3",
        benchmark_name,
        "L8",
    )
    if original_manifest.get("test_accessed") is not False:
        raise ValueError("Stage 2 Original lineage is not Development-only")
    model_dir = Path(runner._base_t5_config(workspace, benchmark)["paths"]["model_dir"])
    if not model_dir.is_absolute():
        model_dir = workspace / model_dir
    required_model_files = ("config.json", "pytorch_model.bin", "spiece.model", "tokenizer_config.json")
    missing = [name for name in required_model_files if not (model_dir / name).is_file()]
    if missing:
        raise ValueError(f"local T5-base directory is incomplete: {missing}")
    authority = {
        "schema": SCHEMA,
        "parent_amendment": config["parent_amendment"],
        "stage1_selection": config["stage1_selection"],
        "benchmark_name": benchmark_name,
        "benchmark_config": str(benchmark_path),
        "benchmark_config_sha256": sha256_file(benchmark_path),
        "prepared_contract_sha256": sha256_file(protocol_dir / "contract.json"),
        "base_registry_sha256": base_registry["registry_sha256"],
        "stronger_signal_registry_sha256": public_registry["registry_sha256"],
        "request_count": len(private_requests),
        "union_users": len(union_users),
        "selected_arm": SELECTED_ARM,
        "original_reference": {
            "run": str(original_run),
            "adapter_sha256": original_manifest["artifact"]["sha256"],
        },
        "source_checkpoint_sha256": sha256_file(
            workspace / benchmark["source"]["original_checkpoint"]
        ),
        "config_sha256": sha256_file(config_path),
        "lineage": lineage,
        "development_metadata_accessed": True,
        "development_predictions_accessed": False,
        "final_test_accessed": False,
        "test_accessed": False,
    }
    return authority, config, benchmark, base_registry, private_requests, union_users


def _artifact_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _same_tensor_map(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> bool:
    return list(left) == list(right) and all(
        torch.equal(left[name].detach().cpu(), right[name].detach().cpu()) for name in right
    )


def _fmt(value: Any, digits: int = 6) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def _classification(prediction: dict[str, Any]) -> dict[str, float | bool]:
    probability = np.asarray(prediction["probability"], dtype=np.float64)
    label = np.asarray(prediction["gold_label"], dtype=np.int64)
    clipped = probability.clip(1e-12, 1 - 1e-12)
    loss = float(np.mean(-(label * np.log(clipped) + (1 - label) * np.log(1 - clipped))))
    return {
        "auc": float(roc_auc_score(label, probability)),
        "accuracy": float(accuracy_score(label, probability >= 0.5)),
        "log_loss": loss,
        "prediction_collapse": bool(np.std(probability) < 1e-6),
    }


def _same_prediction_order(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return bool(
        left["gold_label"] == right["gold_label"]
        and left["sample_order_sha256"] == right["sample_order_sha256"]
    )


def execute(repo: Path, workspace: Path, config_path: Path, run_name: str) -> dict[str, Any]:
    authority, config, benchmark, base_registry, requests, union_users = preflight(
        repo, workspace, config_path
    )
    code_git = git_snapshot(repo)
    if config["runtime"]["code_repository_must_be_clean"] and not code_git["clean"]:
        raise RuntimeError("formal Stage 2 run requires a clean BOTA code repository")
    workspace_git = git_snapshot(workspace)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("formal Stage 2 run requires exactly one CUDA GPU")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    if free / 2**30 < float(config["runtime"]["minimum_free_gib"]):
        raise RuntimeError("insufficient free GPU memory for Stage 2")
    fraction = runner.allocator_fraction_for(
        total / 2**30,
        float(config["runtime"]["allocator_fraction"]),
        float(config["runtime"]["hard_peak_reserved_gib"]),
    )
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    torch.cuda.reset_peak_memory_stats()

    destination = workspace / config["output_root"] / "runs" / safe_run_name(run_name)
    if destination.exists():
        raise FileExistsError(destination)
    stage = destination.parent / ".work" / f"{safe_run_name(run_name)}.{uuid.uuid4().hex}.stage"
    stage.mkdir(parents=True)
    source_checkpoint = workspace / benchmark["source"]["original_checkpoint"]
    source_before = sha256_file(source_checkpoint)
    original_run = Path(authority["original_reference"]["run"])
    original_state = torch.load(
        original_run / "scenarios" / "L8" / "adapter" / "adapter_model.pt",
        map_location="cpu",
        weights_only=True,
    )
    started = time.perf_counter()
    model = None
    original_working_directory = Path.cwd()
    os.chdir(workspace)
    try:
        seed_everything(SEED)
        initialization_started = time.perf_counter()
        model, names, parameters, bases, _, tokenizer, train_data, train_users = runner._model_context(
            workspace, benchmark, base_registry, device, original_state["A"]
        )
        initialization_seconds = time.perf_counter() - initialization_started
        bases_cpu = {name: value.detach().cpu().clone() for name, value in bases.items()}
        if not _same_tensor_map(bases_cpu, original_state["A"]):
            raise RuntimeError("Stage 2 generated fixed-A coordinate changed")
        trajectory_initial_rng = capture_rng()
        canonical_started = time.perf_counter()
        canonical_budget = StepBudget(200)
        canonical, all_states, canonical_trace, optimizer_hash, rng_hash = run_shared_canonical(
            model,
            train_data,
            base_registry["order"],
            train_users,
            union_users,
            parameters,
            names,
            device,
            tokenizer.pad_token_id,
            runner._engine(workspace, benchmark),
            canonical_budget,
        )
        canonical_seconds = time.perf_counter() - canonical_started
        if canonical_budget.calls != 200:
            raise RuntimeError("Stage 2 canonical optimizer-step budget mismatch")
        if any(not torch.equal(canonical[name], original_state["B"][name]) for name in names):
            raise RuntimeError("Stage 2 canonical endpoint differs from frozen Original")
        selected_states = all_states[SELECTED_ARM]
        del all_states, model, parameters
        model = None
        gc.collect()
        torch.cuda.empty_cache()

        original_artifact = runner._save_fixed_ab(
            stage / "references" / "original" / "adapter",
            names,
            canonical,
            bases_cpu,
            "Original-Stage2-200step",
        )
        user_index = {user: slot for slot, user in enumerate(union_users)}
        request_artifacts: dict[str, dict[str, str]] = {}
        bota_publication: dict[str, float] = {}
        for request in requests:
            request_root = stage / "requests" / request["id"]
            slots = [user_index[user] for user in request["users"]]
            candidate = {
                name: canonical[name] + selected_states[module]["theta"][slots].sum(0).float()
                for module, name in enumerate(names)
            }
            publication_started = time.perf_counter()
            artifact = runner._save_fixed_ab(
                request_root / "bota" / "adapter",
                names,
                candidate,
                bases_cpu,
                "BOTA-FS-EmpiricalFisher-Stage2",
            )
            bota_publication[request["id"]] = time.perf_counter() - publication_started
            request_artifacts[request["id"]] = {"bota": artifact["sha256"]}
            del candidate
        del selected_states

        exact_seconds: dict[str, float] = {}
        exact_optimizer_hashes: dict[str, str] = {}
        for request in requests:
            seed_everything(SEED)
            model, exact_names, exact_parameters, exact_bases, _, exact_tokenizer, exact_data, exact_users = runner._model_context(
                workspace, benchmark, base_registry, device, original_state["A"]
            )
            if list(exact_names) != list(names) or not _same_tensor_map(exact_bases, bases_cpu):
                raise RuntimeError("Stage 2 paired Exact-Masked coordinate mismatch")
            restore_rng(trajectory_initial_rng)
            exact_started = time.perf_counter()
            exact_budget = StepBudget(200)
            exact_values, exact_optimizer, exact_trace = runner._train_named(
                model,
                exact_data,
                base_registry["order"],
                exact_users,
                request["users"],
                exact_names,
                exact_parameters,
                device,
                exact_tokenizer.pad_token_id,
                runner._engine(workspace, benchmark),
                exact_budget,
                f"stage2_exact_{request['id']}",
            )
            exact_seconds[request["id"]] = time.perf_counter() - exact_started
            if exact_budget.calls != 200 or [row["batch_hash"] for row in exact_trace] != [
                row["batch_hash"] for row in canonical_trace
            ]:
                raise RuntimeError(f"Stage 2 paired Exact-Masked trace mismatch: {request['id']}")
            exact_optimizer_hashes[request["id"]] = tensor_tree_hash(exact_optimizer.state_dict())
            artifact = runner._save_fixed_ab(
                stage / "requests" / request["id"] / "exact_masked" / "adapter",
                names,
                exact_values,
                bases_cpu,
                "Exact-Masked-Stage2-200step",
            )
            request_artifacts[request["id"]]["exact_masked"] = artifact["sha256"]
            del model, exact_parameters, exact_optimizer, exact_values
            model = None
            gc.collect()
            torch.cuda.empty_cache()

        evaluation_started = time.perf_counter()
        base = runner._base_t5_config(workspace, benchmark)
        model_dir = Path(base["paths"]["model_dir"])
        if not model_dir.is_absolute():
            model_dir = workspace / model_dir
        eval_tokenizer = T5Tokenizer.from_pretrained(model_dir)
        development = JsonPromptDataset(
            workspace / benchmark["source"]["development_json"], eval_tokenizer
        )
        _, development_rows, development_lineage = reconstruct_authoritative_rows(
            workspace / benchmark["source"]["raw_data"]
        )
        development_users = [int(row.authoritative_user_id) for row in development_rows]
        indices = list(range(len(development)))
        original_prediction = behavior._predict_scenario_dir(
            workspace, benchmark, stage / "references" / "original", development, indices, device
        )
        original_utility = _classification(original_prediction)
        metric_rows: list[dict[str, Any]] = []
        safe_samples: list[dict[str, Any]] = []
        for number, request in enumerate(requests):
            request_root = stage / "requests" / request["id"]
            exact_prediction = behavior._predict_scenario_dir(
                workspace, benchmark, request_root / "exact_masked", development, indices, device
            )
            bota_prediction = behavior._predict_scenario_dir(
                workspace, benchmark, request_root / "bota", development, indices, device
            )
            if not _same_prediction_order(original_prediction, exact_prediction) or not _same_prediction_order(
                original_prediction, bota_prediction
            ):
                raise RuntimeError(f"Stage 2 Development prediction order mismatch: {request['id']}")
            selected = [
                index for index, user in enumerate(development_users) if user in set(request["users"])
            ]
            if len(selected) < int(request["request_size"]) * 10:
                raise RuntimeError(f"Stage 2 Development support changed: {request['id']}")
            behavior_row = behavior.behavior_metrics(
                "BOTA-FS-EmpiricalFisher-Stage2",
                bota_prediction,
                exact_prediction,
                original_prediction,
                selected,
                development_users,
                int(config["evaluation"]["bootstrap_resamples"]),
                int(config["evaluation"]["bootstrap_seed"]) + number,
            )
            original_probability = np.asarray(original_prediction["probability"], dtype=np.float64)
            exact_probability = np.asarray(exact_prediction["probability"], dtype=np.float64)
            bota_probability = np.asarray(bota_prediction["probability"], dtype=np.float64)
            chosen = np.asarray(selected, dtype=np.int64)
            du = float(np.mean(np.abs(original_probability[chosen] - exact_probability[chosen])))
            eu = float(np.mean(np.abs(bota_probability[chosen] - exact_probability[chosen])))
            exact_utility = _classification(exact_prediction)
            bota_utility = _classification(bota_prediction)
            row = {
                "request_id": request["id"],
                "stratum": request["stratum"],
                "request_size": request["request_size"],
                "deleted_interactions": request["deleted_interactions"],
                "development_samples": len(selected),
                "DU": du,
                "EU": eu,
                "local_residual": behavior_row["local_residual"],
                "residual_ci_lower": behavior_row["ci_lower"],
                "residual_ci_upper": behavior_row["ci_upper"],
                "toward_exact": behavior_row["toward_exact"],
                "away_from_exact": behavior_row["away_from_exact"],
                "toward_rate": behavior_row["toward_rate"],
                "adequate_effect": bool(du >= float(config["decision"]["adequate_effect_du_minimum"])),
                "original_auc": original_utility["auc"],
                "original_log_loss": original_utility["log_loss"],
                "exact_auc": exact_utility["auc"],
                "exact_log_loss": exact_utility["log_loss"],
                "bota_auc": bota_utility["auc"],
                "bota_log_loss": bota_utility["log_loss"],
                "bota_auc_damage_vs_original": behavior_row["auc_damage_vs_original"],
                "bota_logloss_damage_vs_original": behavior_row["log_loss_damage_vs_original"],
                "prediction_collapse": behavior_row["prediction_collapse"],
                "finite": behavior_row["finite"],
                "bota_artifact_bytes": _artifact_bytes(request_root / "bota" / "adapter"),
                "exact_artifact_bytes": _artifact_bytes(request_root / "exact_masked" / "adapter"),
            }
            metric_rows.append(row)
            safe_samples.extend(
                {
                    "request_id": request["id"],
                    "development_index": index,
                    "user_hash": canonical_hash([SEED, development_users[index]]),
                    "original_probability": float(original_probability[index]),
                    "exact_probability": float(exact_probability[index]),
                    "bota_probability": float(bota_probability[index]),
                }
                for index in selected
            )
        evaluation_seconds = time.perf_counter() - evaluation_started
        peak = torch.cuda.max_memory_reserved() / 2**30
        if peak > float(config["runtime"]["hard_peak_reserved_gib"]):
            raise RuntimeError("Stage 2 GPU hard cap exceeded")
        if sha256_file(source_checkpoint) != source_before:
            raise RuntimeError("Stage 2 source checkpoint changed")

        public_registry, _, _ = build_request_registry(
            train_users, development_users, base_registry["order"], config["protocol"]
        )
        atomic_json(stage / "request_registry.json", public_registry)
        atomic_json(
            stage / "metrics.json",
            {
                "schema": SCHEMA,
                "evidence_class": "secondary_stronger_signal",
                "selected_arm": SELECTED_ARM,
                "adequate_effect_du_minimum": config["decision"]["adequate_effect_du_minimum"],
                "requests": metric_rows,
                "all_constructed_requests_reported": len(metric_rows) == 9,
                "may_replace_primary_scenarios": False,
                "development_only": True,
                "final_test_accessed": False,
                "test_accessed": False,
            },
        )
        with (stage / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
            writer.writeheader()
            writer.writerows(metric_rows)
        with (stage / "per_sample_metrics.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            for row in safe_samples:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        timing = {
            "initialization_seconds": initialization_seconds,
            "shared_canonical_and_bota_bank_seconds": canonical_seconds,
            "bota_adapter_publication_seconds": bota_publication,
            "exact_masked_replay_seconds": exact_seconds,
            "development_evaluation_seconds": evaluation_seconds,
            "end_to_end_seconds": time.perf_counter() - started,
        }
        atomic_json(stage / "timing.json", timing)
        lines = [
            "# Ali Stage 2 stronger-signal secondary analysis",
            "",
            "Development only. All nine outcome-blind requests are reported. This secondary analysis cannot replace the parent L8/L4M4 scenarios.",
            "",
            "| Request | K | Deleted slots | Dev rows | DU | EU | Residual [95% CI] | Toward | AUC | LogLoss | DU adequate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in metric_rows:
            lines.append(
                f"| {row['request_id']} | {row['request_size']} | {row['deleted_interactions']} | "
                f"{row['development_samples']} | {_fmt(row['DU'])} | {_fmt(row['EU'])} | "
                f"{_fmt(row['local_residual'], 4)} [{_fmt(row['residual_ci_lower'], 4)}, {_fmt(row['residual_ci_upper'], 4)}] | "
                f"{row['toward_exact']}/{row['development_samples']} | {row['bota_auc']:.6f} | "
                f"{row['bota_log_loss']:.6f} | {row['adequate_effect']} |"
            )
        lines.extend(
            [
                "",
                "`DU >= 0.001` permits later attack interpretation only for that secondary request; it is not a success criterion for BOTA and does not promote the request to primary status.",
                "",
            ]
        )
        (stage / "report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
        atomic_json(stage / "contract.json", authority)
        atomic_json(
            stage / "provenance.json",
            {
                "schema": SCHEMA,
                "code_repository_git": code_git,
                "experiment_workspace_git": workspace_git,
                "experiment_workspace_dirty_allowed": True,
                "source_checkpoint_sha256": source_before,
                "canonical_trace_sha256": canonical_hash(canonical_trace),
                "canonical_optimizer_state_sha256": optimizer_hash,
                "canonical_rng_sha256": rng_hash,
                "paired_trajectory_initial_rng_sha256": tensor_tree_hash(trajectory_initial_rng),
                "exact_optimizer_state_sha256": exact_optimizer_hashes,
                "request_registry_sha256": public_registry["registry_sha256"],
                "raw_user_ids_persisted": False,
                "selection_used_labels_or_predictions": False,
                "development_lineage": development_lineage,
                "development_accessed": True,
                "final_test_accessed": False,
                "test_loader_built": False,
                "test_accessed": False,
            },
        )
        state = {
            "schema": SCHEMA,
            "status": "COMPLETED",
            "selected_arm": SELECTED_ARM,
            "requests": len(requests),
            "canonical_optimizer_steps": 200,
            "exact_masked_optimizer_steps": 1800,
            "physical_optimizer_step_calls": 2000,
            "online_optimizer_steps": 0,
            "authoritative_optimizer_steps_committed": 0,
            "peak_reserved_gib": peak,
            "all_constructed_requests_reported": len(metric_rows) == 9,
            "test_accessed": False,
        }
        atomic_json(stage / "run_state.json", state)
        (stage / "COMPLETED").write_text(MARKER + "\n", encoding="utf-8", newline="\n")
        files = {
            name: sha256_file(stage / name)
            for name in (
                "COMPLETED",
                "contract.json",
                "metrics.csv",
                "metrics.json",
                "per_sample_metrics.jsonl",
                "provenance.json",
                "report.md",
                "request_registry.json",
                "run_state.json",
                "timing.json",
            )
        }
        atomic_json(
            stage / "manifest.json",
            {
                "schema": SCHEMA,
                "files": files,
                "original_adapter_sha256": original_artifact["sha256"],
                "request_adapter_sha256": request_artifacts,
                "published_atomically": True,
                "test_accessed": False,
            },
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, destination)
        return {
            "status": "COMPLETED",
            "run_dir": str(destination),
            "requests": len(metric_rows),
            "adequate_requests": [row["request_id"] for row in metric_rows if row["adequate_effect"]],
            "selected_arm": SELECTED_ARM,
            "physical_optimizer_step_calls": 2000,
            "test_accessed": False,
        }
    except Exception as error:
        shutil.rmtree(stage, ignore_errors=True)
        raise RuntimeError("Stage 2 stronger-signal run failed before atomic publication") from error
    finally:
        os.chdir(original_working_directory)
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def analyze(config: dict[str, Any], workspace: Path, run_name: str) -> dict[str, Any]:
    run = workspace / config["output_root"] / "runs" / safe_run_name(run_name)
    if not run.is_dir() or {item.name for item in run.iterdir()} != RUN_REQUIRED:
        raise ValueError(f"invalid Stage 2 run layout: {run}")
    if (run / "COMPLETED").read_text(encoding="utf-8") != MARKER + "\n":
        raise ValueError("invalid Stage 2 completion marker")
    manifest = _load_json(run / "manifest.json")
    if manifest.get("schema") != SCHEMA or manifest.get("test_accessed") is not False:
        raise ValueError("invalid Stage 2 manifest")
    for name, expected in manifest.get("files", {}).items():
        if sha256_file(run / name) != expected:
            raise ValueError(f"Stage 2 artifact mismatch: {name}")
    if directory_hash(run / "references" / "original" / "adapter") != manifest.get(
        "original_adapter_sha256"
    ):
        raise ValueError("Stage 2 Original artifact mismatch")
    request_hashes = manifest.get("request_adapter_sha256", {})
    if set(request_hashes) != {f"{stratum}_k{size}" for stratum in STRATA for size in REQUEST_SIZES}:
        raise ValueError("Stage 2 request artifact registry mismatch")
    for request_id, endpoints in request_hashes.items():
        for endpoint, expected in endpoints.items():
            if directory_hash(run / "requests" / request_id / endpoint / "adapter") != expected:
                raise ValueError(f"Stage 2 adapter mismatch: {request_id}/{endpoint}")
    state = _load_json(run / "run_state.json")
    metrics = _load_json(run / "metrics.json")
    registry = _load_json(run / "request_registry.json")
    provenance = _load_json(run / "provenance.json")
    if (
        state.get("status") != "COMPLETED"
        or state.get("physical_optimizer_step_calls") != 2000
        or state.get("all_constructed_requests_reported") is not True
        or len(metrics.get("requests", [])) != 9
        or metrics.get("all_constructed_requests_reported") is not True
        or registry.get("selection_uses_labels_or_predictions") is not False
        or provenance.get("selection_used_labels_or_predictions") is not False
        or provenance.get("raw_user_ids_persisted") is not False
        or provenance.get("test_loader_built") is not False
        or any(value.get("test_accessed") is not False for value in (state, metrics, registry, provenance))
    ):
        raise ValueError("Stage 2 evidence-state mismatch")
    return {
        "status": "COMPLETED",
        "run_dir": str(run),
        "selected_arm": state["selected_arm"],
        "requests": metrics["requests"],
        "adequate_requests": [
            row["request_id"] for row in metrics["requests"] if row["adequate_effect"]
        ],
        "test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/ali_stage2_stronger_signal_v1.yaml")
    )
    parser.add_argument(
        "--mode", choices=("Preflight", "SyntheticDryRun", "Full", "Analyze"), default="Preflight"
    )
    parser.add_argument("--run-name", default="ali_stage2_stronger_signal_seed42_v1")
    args = parser.parse_args()
    repo = args.repo.resolve()
    workspace = args.workspace.resolve()
    config_path = args.config if args.config.is_absolute() else repo / args.config
    config_path = config_path.resolve()
    config = load_config(repo, workspace, config_path)
    if args.mode == "Preflight":
        authority, _, _, _, requests, union_users = preflight(repo, workspace, config_path)
        result = {
            "schema": SCHEMA,
            "status": "READY",
            "authority": authority,
            "request_ids": [request["id"] for request in requests],
            "union_users": len(union_users),
            "test_accessed": False,
        }
    elif args.mode == "SyntheticDryRun":
        result = {
            "schema": SCHEMA,
            "status": "COMPLETED",
            "request_ids": [f"{stratum}_k{size}" for stratum in STRATA for size in REQUEST_SIZES],
            "selected_arm": SELECTED_ARM,
            "real_model_loaded": False,
            "test_accessed": False,
        }
    elif args.mode == "Analyze":
        result = analyze(config, workspace, args.run_name)
    else:
        result = execute(repo, workspace, config_path, args.run_name)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
