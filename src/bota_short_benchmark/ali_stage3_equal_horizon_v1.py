"""Development-only Stage 3 fixed-long-horizon equal-billing experiment."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import statistics
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import T5Tokenizer

from src.bota_if import p2b_full_module_adamw_transport_audit as p2b
from src.bota_if.p1_trajectory_transport_audit import StepBudget
from src.bota_short_benchmark import paper_v2, runner
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


SCHEMA = "bota-ali-stage3-equal-horizon-v1"
MARKER = "BOTA_ALI_STAGE3_EQUAL_HORIZON_V1_COMPLETED"
SUMMARY_SCHEMA = "bota-ali-stage3-equal-horizon-summary-v1"
SUMMARY_MARKER = "BOTA_ALI_STAGE3_EQUAL_HORIZON_SUMMARY_V1_COMPLETED"
SEEDS = (41, 42, 43)
SCENARIOS = ("L8", "L4M4")
SELECTED_ARM = "FS_empirical_fisher"
STEPS = 1200
EPOCHS = 6
STEPS_PER_EPOCH = 200
RUN_REQUIRED = {
    "COMPLETED",
    "adapters",
    "contract.json",
    "manifest.json",
    "metrics.csv",
    "metrics.json",
    "per_sample_metrics.jsonl",
    "provenance.json",
    "report.md",
    "run_state.json",
    "timing.json",
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _same_tensor_map(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return list(left) == list(right) and all(
        torch.equal(left[name].detach().cpu(), right[name].detach().cpu()) for name in right
    )


def _classification(prediction: dict[str, Any]) -> dict[str, float | bool]:
    probability = np.asarray(prediction["probability"], dtype=np.float64)
    label = np.asarray(prediction["gold_label"], dtype=np.int64)
    clipped = probability.clip(1e-12, 1 - 1e-12)
    return {
        "auc": float(roc_auc_score(label, probability)),
        "accuracy": float(accuracy_score(label, probability >= 0.5)),
        "log_loss": float(np.mean(-(label * np.log(clipped) + (1 - label) * np.log(1 - clipped)))),
        "prediction_collapse": bool(np.std(probability) < 1e-6),
    }


def _same_prediction_order(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return bool(
        left["gold_label"] == right["gold_label"]
        and left["sample_order_sha256"] == right["sample_order_sha256"]
    )


def _fmt(value: Any, digits: int = 6) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def _verify_file_manifest(directory: Path, marker: str) -> dict[str, Any]:
    if not directory.is_dir() or (directory / "COMPLETED").read_text(encoding="utf-8") != marker + "\n":
        raise ValueError(f"invalid completed artifact: {directory}")
    manifest = _load_json(directory / "manifest.json")
    for name, expected in manifest.get("files", {}).items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"artifact hash changed: {directory / name}")
    return manifest


def load_config(repo: Path, workspace: Path, path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "schema", "test_access_policy", "parent_amendment", "stage1_selection",
        "stage2_completion", "benchmark_configs", "p5_sources", "output_root",
        "protocol", "evaluation", "reporting", "runtime",
    }
    if not isinstance(value, dict) or set(value) != required or value["schema"] != SCHEMA:
        raise ValueError("invalid Stage 3 equal-horizon configuration")
    if value["test_access_policy"] != "forbidden":
        raise ValueError("Stage 3 must forbid FinalTest")

    amendment = value["parent_amendment"]
    amendment_path = repo / amendment["path"]
    if not amendment_path.is_file() or sha256_file(amendment_path) != amendment["sha256"]:
        raise ValueError("Stage 0 Amendment 02 binding changed")
    amendment_value = _load_json(amendment_path)
    amended = amendment_value.get("amended_equal_billing_long_horizon", {})
    if (
        amendment_value.get("amendment_id") != amendment["amendment_id"]
        or amendment_value.get("status") != "FROZEN_BEFORE_STAGE3_EXECUTION"
        or amended.get("fixed_optimizer_steps") != STEPS
        or amended.get("finaltest_access") is not False
    ):
        raise ValueError("Stage 0 Amendment 02 is not a valid frozen parent")

    stage1 = value["stage1_selection"]
    stage1_path = workspace / stage1["path"]
    if not stage1_path.is_file() or sha256_file(stage1_path) != stage1["sha256"]:
        raise ValueError("frozen Stage 1 selection summary changed")
    stage1_summary = _load_json(stage1_path)
    if (
        stage1_summary.get("selection", {}).get("selected_arm") != SELECTED_ARM
        or stage1.get("selected_arm") != SELECTED_ARM
        or stage1_summary.get("test_accessed") is not False
    ):
        raise ValueError("Stage 1 arm is not frozen to FS_empirical_fisher")
    _verify_file_manifest(stage1_path.parent, STAGE1_SUMMARY_MARKER)

    stage2 = value["stage2_completion"]
    stage2_path = workspace / stage2["path"]
    if not stage2_path.is_file() or sha256_file(stage2_path) != stage2["sha256"]:
        raise ValueError("Stage 2 completion binding changed")
    stage2_manifest = _load_json(stage2_path)
    if stage2_manifest.get("schema") != "bota-ali-stage2-stronger-signal-v1" or stage2_manifest.get("test_accessed") is not False:
        raise ValueError("Stage 2 is not a valid Development-only predecessor")

    expected_protocol = {
        "seeds": list(SEEDS),
        "scenarios": list(SCENARIOS),
        "optimizer_steps": STEPS,
        "epochs": EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "batch_size": 16,
        "schedule": "concatenate_existing_p5_epoch_orders",
        "endpoints": ["Original-Tlong", "Exact-Masked-Tlong", "BOTA-FS-EmpiricalFisher-Tlong"],
        "reference": "Exact-Masked-Tlong",
        "full_batch_denominator_preserved": True,
        "online_optimizer_steps": 0,
    }
    if value["protocol"] != expected_protocol:
        raise ValueError("Stage 3 protocol changed")
    if value["evaluation"] != {
        "split": "Development", "final_test": False, "inference_batch_size": 4,
        "bootstrap_resamples": 1000, "bootstrap_seed_base": 44300,
    }:
        raise ValueError("Stage 3 evaluation contract changed")
    if set(map(int, value["benchmark_configs"])) != set(SEEDS) or set(map(int, value["p5_sources"])) != set(SEEDS):
        raise ValueError("Stage 3 seed registry changed")
    return value


def build_long_schedule(base_order: Sequence[int], seed: int) -> list[int]:
    if len(base_order) != 3200 or len(set(map(int, base_order))) != 3200:
        raise ValueError("Stage 3 requires the frozen 3,200-row adaptation order")
    result: list[int] = []
    for epoch in range(1, EPOCHS + 1):
        result.extend(paper_v2._epoch_order(base_order, seed, epoch))
    if len(result) != STEPS * 16:
        raise RuntimeError("Stage 3 long schedule is not exactly 1,200 full batches")
    return result


def _validate_p5_source(workspace: Path, source: dict[str, Any], *, full_control: bool) -> dict[str, Any]:
    directory = workspace / source["path"]
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != source["manifest_sha256"]:
        raise ValueError(f"P5 source manifest changed: {directory}")
    manifest = _load_json(manifest_path)
    state = _load_json(directory / "run_state.json")
    if manifest.get("test_accessed") is not False or state.get("test_accessed") is not False:
        raise ValueError("P5 source is not Development-only")
    scenario_rows = {}
    for scenario in SCENARIOS:
        row = _load_json(directory / "scenarios" / scenario / "scenario_manifest.json")
        if row.get("test_accessed") is not False:
            raise ValueError("P5 scenario source accessed FinalTest")
        scenario_rows[scenario] = {
            "best_epoch": row.get("best_epoch"),
            "stopping_epoch": row.get("stopping_epoch"),
            "optimizer_steps": row.get("optimizer_steps"),
            "scenario_manifest_sha256": sha256_file(directory / "scenarios" / scenario / "scenario_manifest.json"),
        }
    if full_control:
        expected = {"best_epoch": 1, "stopping_epoch": 6, "optimizer_steps": STEPS}
        for scenario, row in scenario_rows.items():
            if {key: row[key] for key in expected} != expected:
                raise ValueError(f"FullControl-P5 horizon record changed: {scenario}")
    return {
        "path": str(directory),
        "manifest_sha256": source["manifest_sha256"],
        "run_state_sha256": sha256_file(directory / "run_state.json"),
        "scenarios": scenario_rows,
        "used_as_stage3_denominator": False,
    }


def preflight(repo: Path, workspace: Path, config_path: Path, seed: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if seed not in SEEDS:
        raise ValueError(f"unsupported Stage 3 seed: {seed}")
    config = load_config(repo, workspace, config_path)
    benchmark_entry = config["benchmark_configs"][seed]
    benchmark_path = repo / benchmark_entry["path"]
    if sha256_file(benchmark_path) != benchmark_entry["sha256"]:
        raise ValueError(f"benchmark config changed for seed {seed}")
    benchmark = load_benchmark_config(benchmark_path)
    if int(benchmark["protocol"]["run_seed"]) != seed:
        raise ValueError("benchmark run seed mismatch")
    benchmark_name = f"bota_short_i02_seed{seed}_v3"
    protocol_dir, _, registry = validate_prepared(workspace, benchmark, benchmark_name)
    schedule = build_long_schedule(registry["order"], seed)

    original_run, original_manifest, _ = behavior._validate_reference(
        workspace, benchmark, "Original-Short", f"bota_short_original_seed{seed}_v3",
        benchmark_name, "L8",
    )
    if original_manifest.get("test_accessed") is not False:
        raise ValueError("source Original is not Development-only")
    p5_entry = config["p5_sources"][seed]
    full_control = _validate_p5_source(workspace, p5_entry["full_control"], full_control=True)
    retain_retrain = _validate_p5_source(workspace, p5_entry["retain_retrain"], full_control=False)
    model_dir = Path(runner._base_t5_config(workspace, benchmark)["paths"]["model_dir"])
    if not model_dir.is_absolute():
        model_dir = workspace / model_dir
    missing = [name for name in ("config.json", "pytorch_model.bin", "spiece.model", "tokenizer_config.json") if not (model_dir / name).is_file()]
    if missing:
        raise ValueError(f"local T5-base directory is incomplete: {missing}")
    authority = {
        "schema": SCHEMA,
        "parent_amendment": config["parent_amendment"],
        "stage1_selection": config["stage1_selection"],
        "stage2_completion": config["stage2_completion"],
        "seed": seed,
        "benchmark_name": benchmark_name,
        "benchmark_config": str(benchmark_path),
        "benchmark_config_sha256": sha256_file(benchmark_path),
        "prepared_contract_sha256": sha256_file(protocol_dir / "contract.json"),
        "base_registry_sha256": registry["registry_sha256"],
        "long_schedule_sha256": canonical_hash(schedule),
        "long_schedule_optimizer_steps": len(schedule) // 16,
        "full_control_p5_horizon_source": full_control,
        "retain_retrain_p5_lineage_only": retain_retrain,
        "retain_retrain_loaded": False,
        "original_fixed_a_source": {"run": str(original_run), "adapter_sha256": original_manifest["artifact"]["sha256"]},
        "source_checkpoint_sha256": sha256_file(workspace / benchmark["source"]["original_checkpoint"]),
        "config_sha256": sha256_file(config_path),
        "development_predictions_accessed": False,
        "final_test_accessed": False,
        "test_accessed": False,
    }
    return authority, config, benchmark, registry


def _scenario(registry: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    return next(row for row in registry["scenarios"] if row["id"] == scenario_id)


def default_run_name(seed: int) -> str:
    return f"ali_stage3_equal_horizon_seed{seed}_v1"


def execute(repo: Path, workspace: Path, config_path: Path, seed: int, run_name: str) -> dict[str, Any]:
    authority, config, benchmark, registry = preflight(repo, workspace, config_path, seed)
    code_git = git_snapshot(repo)
    if config["runtime"]["code_repository_must_be_clean"] and not code_git["clean"]:
        raise RuntimeError("formal Stage 3 run requires a clean BOTA code repository")
    workspace_git = git_snapshot(workspace)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("formal Stage 3 run requires exactly one CUDA GPU")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    if free / 2**30 < float(config["runtime"]["minimum_free_gib"]):
        raise RuntimeError("insufficient free GPU memory for Stage 3")
    fraction = runner.allocator_fraction_for(total / 2**30, float(config["runtime"]["allocator_fraction"]), float(config["runtime"]["hard_peak_reserved_gib"]))
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    torch.cuda.reset_peak_memory_stats()

    destination = workspace / config["output_root"] / "runs" / safe_run_name(run_name)
    if destination.exists():
        raise FileExistsError(destination)
    stage = destination.parent / ".work" / f"{safe_run_name(run_name)}.{uuid.uuid4().hex}.stage"
    stage.mkdir(parents=True)
    source_checkpoint = workspace / benchmark["source"]["original_checkpoint"]
    source_before = sha256_file(source_checkpoint)
    original_short = Path(authority["original_fixed_a_source"]["run"])
    original_state = torch.load(original_short / "scenarios" / "L8" / "adapter" / "adapter_model.pt", map_location="cpu", weights_only=True)
    long_order = build_long_schedule(registry["order"], seed)
    scenarios = [_scenario(registry, name) for name in SCENARIOS]
    union_users = sorted({int(user) for scenario in scenarios for user in scenario["users"]})
    started = time.perf_counter()
    model = None
    original_working_directory = Path.cwd()
    os.chdir(workspace)
    try:
        seed_everything(seed)
        initialization_started = time.perf_counter()
        model, names, parameters, bases, _, tokenizer, train_data, train_users = runner._model_context(
            workspace, benchmark, registry, device, original_state["A"]
        )
        initialization_seconds = time.perf_counter() - initialization_started
        bases_cpu = {name: value.detach().cpu().clone() for name, value in bases.items()}
        if not _same_tensor_map(bases_cpu, original_state["A"]):
            raise RuntimeError("Stage 3 generated fixed-A coordinate changed")
        trajectory_initial_rng = capture_rng()
        canonical_started = time.perf_counter()
        canonical_budget = StepBudget(STEPS)
        canonical, all_states, canonical_trace, optimizer_hash, rng_hash = run_shared_canonical(
            model, train_data, long_order, train_users, union_users, parameters, names,
            device, tokenizer.pad_token_id, runner._engine(workspace, benchmark), canonical_budget,
            arms=(SELECTED_ARM,),
        )
        canonical_seconds = time.perf_counter() - canonical_started
        if canonical_budget.calls != STEPS or [row["batch_hash"] for row in canonical_trace] != [canonical_hash(long_order[start:start + 16]) for start in range(0, len(long_order), 16)]:
            raise RuntimeError("Stage 3 canonical schedule or budget mismatch")
        selected_states = all_states[SELECTED_ARM]
        del all_states, model, parameters
        model = None
        gc.collect()
        torch.cuda.empty_cache()

        original_artifact = runner._save_fixed_ab(stage / "adapters" / "original" / "adapter", names, canonical, bases_cpu, "Original-Tlong")
        user_index = {user: slot for slot, user in enumerate(union_users)}
        bota_values: dict[str, dict[str, torch.Tensor]] = {}
        bota_hashes: dict[str, str] = {}
        bota_publication_seconds: dict[str, float] = {}
        for scenario in scenarios:
            slots = [user_index[int(user)] for user in scenario["users"]]
            candidate = {name: canonical[name] + selected_states[module]["theta"][slots].sum(0).float() for module, name in enumerate(names)}
            bota_values[scenario["id"]] = candidate
            publication_started = time.perf_counter()
            artifact = runner._save_fixed_ab(stage / "adapters" / scenario["id"] / "bota" / "adapter", names, candidate, bases_cpu, "BOTA-FS-EmpiricalFisher-Tlong")
            bota_publication_seconds[scenario["id"]] = time.perf_counter() - publication_started
            bota_hashes[scenario["id"]] = artifact["sha256"]
        del selected_states

        exact_values_by_scenario: dict[str, dict[str, torch.Tensor]] = {}
        exact_hashes: dict[str, str] = {}
        exact_seconds: dict[str, float] = {}
        exact_optimizer_hashes: dict[str, str] = {}
        for scenario in scenarios:
            seed_everything(seed)
            model, exact_names, exact_parameters, exact_bases, _, exact_tokenizer, exact_data, exact_users = runner._model_context(
                workspace, benchmark, registry, device, original_state["A"]
            )
            if list(exact_names) != list(names) or not _same_tensor_map(exact_bases, bases_cpu):
                raise RuntimeError("Stage 3 paired Exact-Masked coordinate mismatch")
            restore_rng(trajectory_initial_rng)
            exact_started = time.perf_counter()
            exact_budget = StepBudget(STEPS)
            exact_values, exact_optimizer, exact_trace = runner._train_named(
                model, exact_data, long_order, exact_users, scenario["users"], exact_names,
                exact_parameters, device, exact_tokenizer.pad_token_id,
                runner._engine(workspace, benchmark), exact_budget, f"stage3_exact_{scenario['id']}",
            )
            exact_seconds[scenario["id"]] = time.perf_counter() - exact_started
            if exact_budget.calls != STEPS or [row["batch_hash"] for row in exact_trace] != [row["batch_hash"] for row in canonical_trace]:
                raise RuntimeError(f"Stage 3 paired schedule mismatch: {scenario['id']}")
            exact_optimizer_hashes[scenario["id"]] = tensor_tree_hash(exact_optimizer.state_dict())
            exact_values_by_scenario[scenario["id"]] = exact_values
            artifact = runner._save_fixed_ab(stage / "adapters" / scenario["id"] / "exact_masked" / "adapter", names, exact_values, bases_cpu, "Exact-Masked-Tlong")
            exact_hashes[scenario["id"]] = artifact["sha256"]
            del model, exact_parameters, exact_optimizer
            model = None
            gc.collect()
            torch.cuda.empty_cache()

        evaluation_started = time.perf_counter()
        base = runner._base_t5_config(workspace, benchmark)
        model_dir = Path(base["paths"]["model_dir"])
        if not model_dir.is_absolute():
            model_dir = workspace / model_dir
        eval_tokenizer = T5Tokenizer.from_pretrained(model_dir)
        development = JsonPromptDataset(workspace / benchmark["source"]["development_json"], eval_tokenizer)
        _, development_rows, development_lineage = reconstruct_authoritative_rows(workspace / benchmark["source"]["raw_data"])
        development_users = [int(row.authoritative_user_id) for row in development_rows]
        indices = list(range(len(development)))
        original_prediction = behavior._predict_scenario_dir(workspace, benchmark, stage / "adapters" / "original", development, indices, device)
        original_utility = _classification(original_prediction)
        metric_rows: list[dict[str, Any]] = []
        safe_samples: list[dict[str, Any]] = []
        for number, scenario in enumerate(scenarios):
            exact_prediction = behavior._predict_scenario_dir(workspace, benchmark, stage / "adapters" / scenario["id"] / "exact_masked", development, indices, device)
            bota_prediction = behavior._predict_scenario_dir(workspace, benchmark, stage / "adapters" / scenario["id"] / "bota", development, indices, device)
            if not _same_prediction_order(original_prediction, exact_prediction) or not _same_prediction_order(original_prediction, bota_prediction):
                raise RuntimeError(f"Stage 3 Development prediction order mismatch: {scenario['id']}")
            selected = [index for index, user in enumerate(development_users) if user in set(scenario["users"])]
            behavior_row = behavior.behavior_metrics(
                "BOTA-FS-EmpiricalFisher-Tlong", bota_prediction, exact_prediction,
                original_prediction, selected, development_users,
                int(config["evaluation"]["bootstrap_resamples"]),
                int(config["evaluation"]["bootstrap_seed_base"]) + seed * 10 + number,
            )
            original_probability = np.asarray(original_prediction["probability"], dtype=np.float64)
            exact_probability = np.asarray(exact_prediction["probability"], dtype=np.float64)
            bota_probability = np.asarray(bota_prediction["probability"], dtype=np.float64)
            chosen = np.asarray(selected, dtype=np.int64)
            du = float(np.mean(np.abs(original_probability[chosen] - exact_probability[chosen])))
            eu = float(np.mean(np.abs(bota_probability[chosen] - exact_probability[chosen])))
            actual_delta = torch.cat([(exact_values_by_scenario[scenario["id"]][name] - canonical[name]).reshape(-1) for name in names])
            predicted_delta = torch.cat([(bota_values[scenario["id"]][name] - canonical[name]).reshape(-1) for name in names])
            parameter = p2b.vector_metrics(actual_delta, predicted_delta)
            exact_utility = _classification(exact_prediction)
            bota_utility = _classification(bota_prediction)
            row = {
                "seed": seed,
                "scenario": scenario["id"],
                "optimizer_steps_per_endpoint": STEPS,
                "deleted_interactions": int(scenario["deleted_interactions"]),
                "development_samples": len(selected),
                "DU": du,
                "EU": eu,
                "local_residual": behavior_row["local_residual"],
                "residual_ci_lower": behavior_row["ci_lower"],
                "residual_ci_upper": behavior_row["ci_upper"],
                "toward_exact": behavior_row["toward_exact"],
                "away_from_exact": behavior_row["away_from_exact"],
                "toward_rate": behavior_row["toward_rate"],
                "delta_cosine": parameter["cosine"],
                "relative_l2": parameter["relative_l2_error"],
                "norm_ratio": parameter["norm_ratio"],
                "original_auc": original_utility["auc"],
                "original_log_loss": original_utility["log_loss"],
                "exact_auc": exact_utility["auc"],
                "exact_log_loss": exact_utility["log_loss"],
                "bota_auc": bota_utility["auc"],
                "bota_log_loss": bota_utility["log_loss"],
                "prediction_collapse": behavior_row["prediction_collapse"],
                "finite": behavior_row["finite"],
                "original_and_bank_wall_time_seconds": canonical_seconds,
                "exact_masked_wall_time_seconds": exact_seconds[scenario["id"]],
                "bota_publication_wall_time_seconds": bota_publication_seconds[scenario["id"]],
                "bota_online_optimizer_steps": 0,
                "moved_toward_exact_point_estimate": bool(behavior_row["local_residual"] < 1.0),
            }
            metric_rows.append(row)
            safe_samples.extend({
                "seed": seed,
                "scenario": scenario["id"],
                "development_index": index,
                "user_hash": canonical_hash([seed, development_users[index]]),
                "original_probability": float(original_probability[index]),
                "exact_probability": float(exact_probability[index]),
                "bota_probability": float(bota_probability[index]),
            } for index in selected)
        evaluation_seconds = time.perf_counter() - evaluation_started
        peak = torch.cuda.max_memory_reserved() / 2**30
        if peak > float(config["runtime"]["hard_peak_reserved_gib"]):
            raise RuntimeError("Stage 3 GPU hard cap exceeded")
        if sha256_file(source_checkpoint) != source_before:
            raise RuntimeError("Stage 3 source checkpoint changed")

        atomic_json(stage / "metrics.json", {
            "schema": SCHEMA, "evidence_class": "secondary_reference_sensitivity",
            "selected_arm": SELECTED_ARM, "reference": "Exact-Masked-Tlong",
            "fixed_optimizer_steps": STEPS, "cells": metric_rows,
            "all_seed_scenario_cells_in_this_run_reported": len(metric_rows) == 2,
            "retain_retrain_used_as_denominator": False, "development_only": True,
            "final_test_accessed": False, "test_accessed": False,
        })
        with (stage / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
            writer.writeheader(); writer.writerows(metric_rows)
        with (stage / "per_sample_metrics.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in safe_samples:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        timing = {
            "initialization_seconds": initialization_seconds,
            "original_and_shared_bota_bank_seconds": canonical_seconds,
            "exact_masked_replay_seconds": exact_seconds,
            "bota_adapter_publication_seconds": bota_publication_seconds,
            "development_evaluation_seconds": evaluation_seconds,
            "end_to_end_seconds": time.perf_counter() - started,
        }
        atomic_json(stage / "timing.json", timing)
        lines = [
            f"# Ali Stage 3 fixed-long-horizon equal-billing — seed {seed}", "",
            "Development only; 1,200 optimizer steps per endpoint. Exact-Masked-Tlong is the local counterfactual reference. Retain-Retrain-P5 is lineage-only and is not the denominator.", "",
            "| Scenario | DU | EU | Residual [95% CI] | Cosine | Relative L2 | Original AUC | Exact AUC | BOTA AUC | BOTA LogLoss | Toward exact |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in metric_rows:
            lines.append(
                f"| {row['scenario']} | {_fmt(row['DU'])} | {_fmt(row['EU'])} | {_fmt(row['local_residual'], 4)} [{_fmt(row['residual_ci_lower'], 4)}, {_fmt(row['residual_ci_upper'], 4)}] | {_fmt(row['delta_cosine'], 4)} | {_fmt(row['relative_l2'], 4)} | {_fmt(row['original_auc'])} | {_fmt(row['exact_auc'])} | {_fmt(row['bota_auc'])} | {_fmt(row['bota_log_loss'])} | {row['moved_toward_exact_point_estimate']} |"
            )
        lines.extend(["", "A point residual below 1 means movement toward the matched Exact-Masked-Tlong endpoint. This secondary result does not establish equivalence to independently converged retain-only retraining.", ""])
        (stage / "report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
        atomic_json(stage / "contract.json", authority)
        atomic_json(stage / "provenance.json", {
            "schema": SCHEMA, "code_repository_git": code_git,
            "experiment_workspace_git": workspace_git, "experiment_workspace_dirty_allowed": True,
            "source_checkpoint_sha256": source_before,
            "long_schedule_sha256": canonical_hash(long_order),
            "canonical_trace_sha256": canonical_hash(canonical_trace),
            "canonical_optimizer_state_sha256": optimizer_hash,
            "canonical_rng_sha256": rng_hash,
            "paired_trajectory_initial_rng_sha256": tensor_tree_hash(trajectory_initial_rng),
            "exact_optimizer_state_sha256": exact_optimizer_hashes,
            "development_lineage": development_lineage, "development_accessed": True,
            "retain_retrain_loaded": False, "final_test_accessed": False,
            "test_loader_built": False, "test_accessed": False,
        })
        atomic_json(stage / "run_state.json", {
            "schema": SCHEMA, "status": "COMPLETED", "seed": seed,
            "scenarios": list(SCENARIOS), "selected_arm": SELECTED_ARM,
            "optimizer_steps_per_endpoint": STEPS, "canonical_optimizer_steps": STEPS,
            "exact_masked_optimizer_steps": STEPS * len(SCENARIOS),
            "physical_optimizer_step_calls": STEPS * (1 + len(SCENARIOS)),
            "online_optimizer_steps": 0, "authoritative_optimizer_steps_committed": 0,
            "peak_reserved_gib": peak, "retain_retrain_loaded": False, "test_accessed": False,
        })
        (stage / "COMPLETED").write_text(MARKER + "\n", encoding="utf-8", newline="\n")
        files = {name: sha256_file(stage / name) for name in ("COMPLETED", "contract.json", "metrics.csv", "metrics.json", "per_sample_metrics.jsonl", "provenance.json", "report.md", "run_state.json", "timing.json")}
        atomic_json(stage / "manifest.json", {
            "schema": SCHEMA, "files": files,
            "adapter_sha256": {"original": original_artifact["sha256"], "exact_masked": exact_hashes, "bota": bota_hashes},
            "published_atomically": True, "test_accessed": False,
        })
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, destination)
        return {"status": "COMPLETED", "run_dir": str(destination), "seed": seed, "cells": len(metric_rows), "physical_optimizer_step_calls": STEPS * 3, "test_accessed": False}
    except Exception as error:
        shutil.rmtree(stage, ignore_errors=True)
        raise RuntimeError("Stage 3 equal-horizon run failed before atomic publication") from error
    finally:
        os.chdir(original_working_directory)
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def analyze(config: dict[str, Any], workspace: Path, run_name: str, seed: int) -> dict[str, Any]:
    run = workspace / config["output_root"] / "runs" / safe_run_name(run_name)
    if not run.is_dir() or {item.name for item in run.iterdir()} != RUN_REQUIRED:
        raise ValueError("invalid Stage 3 run layout")
    manifest = _verify_file_manifest(run, MARKER)
    if manifest.get("schema") != SCHEMA or manifest.get("test_accessed") is not False:
        raise ValueError("invalid Stage 3 manifest")
    for group, registry in manifest.get("adapter_sha256", {}).items():
        if group == "original":
            continue
        for scenario, expected in registry.items():
            directory = run / "adapters" / scenario / ("exact_masked" if group == "exact_masked" else "bota") / "adapter"
            if directory_hash(directory) != expected:
                raise ValueError(f"Stage 3 adapter changed: {group}/{scenario}")
    if directory_hash(run / "adapters" / "original" / "adapter") != manifest["adapter_sha256"]["original"]:
        raise ValueError("Stage 3 Original-Tlong adapter changed")
    state = _load_json(run / "run_state.json")
    metrics = _load_json(run / "metrics.json")
    provenance = _load_json(run / "provenance.json")
    if (
        state.get("seed") != seed or state.get("physical_optimizer_step_calls") != 3600
        or state.get("retain_retrain_loaded") is not False or state.get("test_accessed") is not False
        or metrics.get("retain_retrain_used_as_denominator") is not False
        or len(metrics.get("cells", [])) != 2
        or provenance.get("test_loader_built") is not False
        or provenance.get("test_accessed") is not False
    ):
        raise ValueError("Stage 3 safety or accounting invariant failed")
    return {"status": "VALID", "run_dir": str(run), "seed": seed, "cells": metrics["cells"], "test_accessed": False}


def _mean(values: Sequence[float]) -> float:
    return float(statistics.mean(values))


def _stdev(values: Sequence[float]) -> float:
    return float(statistics.stdev(values))


def summarize(repo: Path, workspace: Path, config_path: Path, summary_name: str) -> dict[str, Any]:
    config = load_config(repo, workspace, config_path)
    cells: list[dict[str, Any]] = []
    run_hashes: dict[str, str] = {}
    for seed in SEEDS:
        run_name = default_run_name(seed)
        result = analyze(config, workspace, run_name, seed)
        cells.extend(result["cells"])
        run = Path(result["run_dir"])
        run_hashes[str(seed)] = sha256_file(run / "manifest.json")
    if len(cells) != 6 or {(int(row["seed"]), row["scenario"]) for row in cells} != {(seed, scenario) for seed in SEEDS for scenario in SCENARIOS}:
        raise ValueError("Stage 3 summary requires all six seed-by-scenario cells")
    metrics = ("DU", "EU", "local_residual", "delta_cosine", "relative_l2", "original_auc", "exact_auc", "bota_auc", "bota_log_loss")
    by_scenario: dict[str, dict[str, float]] = {}
    for scenario in SCENARIOS:
        rows = [row for row in cells if row["scenario"] == scenario]
        block: dict[str, float] = {}
        for metric in metrics:
            values = [float(row[metric]) for row in rows]
            block[f"{metric}_mean"] = _mean(values)
            block[f"{metric}_sample_sd"] = _stdev(values)
        block["cells_moved_toward_exact"] = float(sum(bool(row["moved_toward_exact_point_estimate"]) for row in rows))
        by_scenario[scenario] = block
    payload = {
        "schema": SUMMARY_SCHEMA, "status": "COMPLETED",
        "evidence_class": "secondary_reference_sensitivity", "fixed_optimizer_steps": STEPS,
        "selected_arm": SELECTED_ARM, "reference": "Exact-Masked-Tlong",
        "cells": cells, "summary_by_scenario": by_scenario,
        "all_six_cells_reported": True, "significance_claim_permitted": False,
        "retain_retrain_used_as_denominator": False,
        "interpretation": "Movement toward Exact-Masked-Tlong supports only long-horizon local approximation; it does not establish equivalence to independently converged retain-only retraining.",
        "final_test_accessed": False, "test_accessed": False,
    }
    destination = workspace / config["output_root"] / "summaries" / safe_run_name(summary_name)
    if destination.exists():
        raise FileExistsError(destination)
    stage = destination.parent / ".work" / f"{safe_run_name(summary_name)}.{uuid.uuid4().hex}.stage"
    stage.mkdir(parents=True)
    atomic_json(stage / "summary.json", payload)
    lines = ["# Ali Stage 3 fixed-long-horizon equal-billing summary", "", "Development-only secondary sensitivity; all six cells are retained.", "", "| Scenario | DU | EU | Residual | Cosine | Relative L2 | Cells toward Exact |", "|---|---:|---:|---:|---:|---:|---:|"]
    for scenario in SCENARIOS:
        row = by_scenario[scenario]
        lines.append(f"| {scenario} | {row['DU_mean']:.6f} ± {row['DU_sample_sd']:.6f} | {row['EU_mean']:.6f} ± {row['EU_sample_sd']:.6f} | {row['local_residual_mean']:.4f} ± {row['local_residual_sample_sd']:.4f} | {row['delta_cosine_mean']:.4f} ± {row['delta_cosine_sample_sd']:.4f} | {row['relative_l2_mean']:.4f} ± {row['relative_l2_sample_sd']:.4f} | {int(row['cells_moved_toward_exact'])}/3 |")
    lines.extend(["", "Retain-Retrain-P5 was not used as the denominator. No significance claim is made from three seeds.", ""])
    (stage / "summary.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    (stage / "COMPLETED").write_text(SUMMARY_MARKER + "\n", encoding="utf-8", newline="\n")
    atomic_json(stage / "manifest.json", {"schema": SUMMARY_SCHEMA, "files": {name: sha256_file(stage / name) for name in ("COMPLETED", "summary.json", "summary.md")}, "source_run_manifest_sha256": run_hashes, "published_atomically": True, "test_accessed": False})
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage, destination)
    return {"status": "COMPLETED", "summary_dir": str(destination), "cells": 6, "test_accessed": False}


def synthetic(repo: Path, workspace: Path, config_path: Path, seed: int, run_name: str) -> dict[str, Any]:
    authority, config, _, registry = preflight(repo, workspace, config_path, seed)
    schedule = build_long_schedule(registry["order"], seed)
    destination = workspace / config["output_root"] / "synthetic" / safe_run_name(run_name)
    if destination.exists():
        raise FileExistsError(destination)
    stage = destination.parent / ".work" / f"{safe_run_name(run_name)}.{uuid.uuid4().hex}.stage"
    stage.mkdir(parents=True)
    atomic_json(stage / "audit.json", {"schema": SCHEMA, "seed": seed, "schedule_rows": len(schedule), "optimizer_steps": len(schedule) // 16, "schedule_sha256": canonical_hash(schedule), "retain_retrain_loaded": False, "final_test_accessed": False, "test_accessed": False})
    atomic_json(stage / "contract.json", authority)
    (stage / "COMPLETED").write_text(MARKER + "\n", encoding="utf-8", newline="\n")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage, destination)
    return {"status": "COMPLETED", "run_dir": str(destination), "synthetic": True, "optimizer_steps": 1200, "test_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/ali_stage3_equal_horizon_v1.yaml"))
    parser.add_argument("--mode", choices=("Preflight", "SyntheticDryRun", "Full", "Analyze", "Summarize"), default="Preflight")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=41)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--summary-name", default="ali_stage3_equal_horizon_summary_v1")
    args = parser.parse_args()
    repo = args.repo.resolve(); workspace = args.workspace.resolve()
    config_path = args.config if args.config.is_absolute() else repo / args.config
    run_name = args.run_name or default_run_name(args.seed)
    if args.mode == "Preflight":
        result = preflight(repo, workspace, config_path, args.seed)[0]
    elif args.mode == "SyntheticDryRun":
        result = synthetic(repo, workspace, config_path, args.seed, run_name)
    elif args.mode == "Full":
        result = execute(repo, workspace, config_path, args.seed, run_name)
    elif args.mode == "Analyze":
        config = load_config(repo, workspace, config_path)
        result = analyze(config, workspace, run_name, args.seed)
    else:
        result = summarize(repo, workspace, config_path, args.summary_name)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
