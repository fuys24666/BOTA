"""Stage 9: fixed-configuration cross-seed Development confirmation.

Seed 42 selected the Stage 7/8 settings. Seeds 41 and 43 are held-out
confirmation seeds. FinalTest is forbidden throughout this module.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import T5Tokenizer

from src.diagnostics.t5_reconstructed_official import JsonPromptDataset
from src.paper_if_a2.common import atomic_json, directory_hash, git_snapshot, safe_run_name, sha256_file

from . import ali_stage7_baseline_sweep_v1 as stage7
from . import ali_stage8_bota_response_sweep_v1 as stage8
from . import runner


SCHEMA = "bota-ali-stage9-crossseed-development-v1"
MARKER = "BOTA_ALI_STAGE9_CROSSSEED_DEVELOPMENT_V1_COMPLETED"
SUMMARY_MARKER = "BOTA_ALI_STAGE9_CROSSSEED_DEVELOPMENT_V1_SUMMARY_COMPLETED"
METHODS = ("BOTA", "IFRU", "E2URec", "NegGrad", "PCGrad", "RecEraser")
HELD_OUT_SEEDS = (41, 43)
REPORTED_SEEDS = (41, 42, 43)
SCENARIOS = ("L8", "L4M4")
RUN_REQUIRED = {"COMPLETED", "candidate_states", "metrics.csv", "metrics.json", "contract.json", "manifest.json", "report.md", "run_state.json"}
SUMMARY_REQUIRED = {"COMPLETED", "metrics.csv", "metrics.json", "decision.json", "contract.json", "manifest.json", "report.md"}
SELECTED_IDS = {
    "BOTA": "bota_r1",
    "IFRU": "ifru_p512_d0.1",
    "E2URec": "e2urec_w0.2_s200",
    "NegGrad": "neggrad_w0.05_s25",
    "PCGrad": "pcgrad_lr0.0005_s25",
    "RecEraser": "receraser_lr0.0005_e3",
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _assert_no_test_access(value: Any, location: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"test_accessed", "finaltest_accessed", "final_test_accessed"} and item is not False:
                raise ValueError(f"test-access flag is not false at {location}.{key}")
            _assert_no_test_access(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_test_access(item, f"{location}[{index}]")


def load_config(repo: Path, workspace: Path, path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"schema", "test_access_policy", "parent_amendment", "stage7_audit", "stage8_audit", "benchmarks", "development_partition", "protocol", "selected_configurations", "sisa", "selection", "decision", "output_root", "runtime"}
    if not isinstance(value, dict) or set(value) != required or value["schema"] != SCHEMA or value["test_access_policy"] != "forbidden":
        raise ValueError("invalid Stage 9 configuration surface")
    amendment_binding = value["parent_amendment"]
    amendment_path = repo / amendment_binding["path"]
    amendment = _json(amendment_path)
    if sha256_file(amendment_path) != amendment_binding["sha256"] or amendment.get("amendment_id") != amendment_binding["amendment_id"] or amendment.get("status") != "FROZEN_BEFORE_STAGE9_EXECUTION":
        raise ValueError("Amendment 08 identity/status changed")
    for field, expected_status in (("stage7_audit", "FROZEN_COMPLETE_STAGE7A"), ("stage8_audit", "FROZEN_COMPLETE_STAGE8")):
        binding = value[field]; audit_path = repo / binding["path"]
        if sha256_file(audit_path) != binding["sha256"] or _json(audit_path).get("status") != expected_status:
            raise ValueError(f"{field} identity/status changed")
    if set(map(int, value["benchmarks"])) != set(REPORTED_SEEDS):
        raise ValueError("Stage 9 benchmark seeds changed")
    for seed in REPORTED_SEEDS:
        binding = value["benchmarks"][seed]
        if int(binding["seed"]) != seed:
            raise ValueError("Stage 9 seed binding mismatch")
        for field in ("config", "paper_config"):
            item = binding[field]
            if sha256_file(repo / item["path"]) != item["sha256"]:
                raise ValueError(f"Stage 9 seed {seed} {field} changed")
        frozen = workspace / binding["frozen_development_evaluation"]["path"]
        item = binding["frozen_development_evaluation"]
        if sha256_file(frozen / "manifest.json") != item["manifest_sha256"] or sha256_file(frozen / "per_sample_metrics.jsonl") != item["per_sample_sha256"]:
            raise ValueError(f"Stage 9 seed {seed} Development authority changed")
    protocol = value["protocol"]
    if protocol != {"selection_seed": 42, "held_out_confirmation_seeds": list(HELD_OUT_SEEDS), "reported_seeds": list(REPORTED_SEEDS), "scenarios": list(SCENARIOS), "methods": list(METHODS), "split": "Development/Dev-Select", "final_test": False}:
        raise ValueError("Stage 9 protocol changed")
    expected_selected = {
        "BOTA": {"candidate_id": "bota_r1", "response_scale": 1.0, "online_optimizer_steps": 0},
        "IFRU": {"candidate_id": "ifru_p512_d0.1", "relative_damping_ratio": 0.1, "curvature_panel_samples": 512, "cg_max_iterations": 40},
        "E2URec": {"candidate_id": "e2urec_w0.2_s200", "teacher_steps": 1200, "teacher_learning_rate": 0.00005, "student_learning_rate": 0.001, "forget_weight": 0.2, "checkpoint": 200},
        "NegGrad": {"candidate_id": "neggrad_w0.05_s25", "learning_rate": 0.001, "forget_weight": 0.05, "checkpoint": 25},
        "PCGrad": {"candidate_id": "pcgrad_lr0.0005_s25", "learning_rate": 0.0005, "checkpoint": 25},
        "RecEraser": {"candidate_id": "receraser_lr0.0005_e3", "shards": 4, "feature_dimensions": 32, "learning_rate": 0.0005, "epochs_per_slice": 3, "attention_steps": 200, "attention_learning_rate": 0.1},
    }
    if value["selected_configurations"] != expected_selected or value["sisa"] != {"stage7_status": "NO_ELIGIBLE_CANDIDATE", "rerun": False}:
        raise ValueError("frozen Stage 7/8 identities changed")
    stage7_audit = _json(repo / value["stage7_audit"]["path"])
    for method in METHODS[1:]:
        if stage7_audit["selections"][method]["selected_candidate"] != SELECTED_IDS[method]:
            raise ValueError(f"Stage 7 selected identity changed for {method}")
    if stage7_audit["selections"]["SISA"]["status"] != "NO_ELIGIBLE_CANDIDATE":
        raise ValueError("Stage 7 SISA no-eligible decision changed")
    stage8_audit = _json(repo / value["stage8_audit"]["path"])
    if stage8_audit["selection"]["selected_candidate"] != "bota_r1" or stage8_audit["selection"]["response_scale"] != 1.0:
        raise ValueError("Stage 8 BOTA selection changed")
    if value["selection"] != {"auc_damage_max": 0.005, "logloss_damage_max": 0.01}:
        raise ValueError("Stage 9 utility guardrails changed")
    expected_decision = {"scope_seeds": list(HELD_OUT_SEEDS), "require_bota_utility_every_cell": True, "require_bota_residual_below_one_every_cell": True, "require_bota_mean_eu_below_each_baseline": True, "failure_status": "NOT_STABLE_NO_FINALTEST_AUTHORIZATION", "success_status": "STABLE_CONFIRMED"}
    if value["decision"] != expected_decision:
        raise ValueError("Stage 9 decision rule changed")
    return value


def _seed_config(config: dict[str, Any], seed: int) -> dict[str, Any]:
    selected = config["selected_configurations"]
    grids = {
        "IFRU": {"relative_damping_ratio": [selected["IFRU"]["relative_damping_ratio"]], "curvature_panel_samples": [selected["IFRU"]["curvature_panel_samples"]], "cg_max_iterations": selected["IFRU"]["cg_max_iterations"]},
        "E2URec": {"teacher_steps": selected["E2URec"]["teacher_steps"], "teacher_learning_rate": selected["E2URec"]["teacher_learning_rate"], "student_learning_rate": selected["E2URec"]["student_learning_rate"], "forget_weight": [selected["E2URec"]["forget_weight"]], "checkpoints": [selected["E2URec"]["checkpoint"]]},
        "NegGrad": {"learning_rate": selected["NegGrad"]["learning_rate"], "forget_weight": [selected["NegGrad"]["forget_weight"]], "checkpoints": [selected["NegGrad"]["checkpoint"]]},
        "PCGrad": {"learning_rate": [selected["PCGrad"]["learning_rate"]], "checkpoints": [selected["PCGrad"]["checkpoint"]]},
        "RecEraser": {"shards": selected["RecEraser"]["shards"], "feature_dimensions": selected["RecEraser"]["feature_dimensions"], "learning_rate": [selected["RecEraser"]["learning_rate"]], "epochs_per_slice": [selected["RecEraser"]["epochs_per_slice"]], "attention_steps": selected["RecEraser"]["attention_steps"], "attention_learning_rate": selected["RecEraser"]["attention_learning_rate"]},
    }
    return {
        "benchmark": config["benchmarks"][seed],
        "development_partition": config["development_partition"],
        "grids": grids,
        "selection": {"scenarios": list(SCENARIOS), "auc_damage_max": config["selection"]["auc_damage_max"], "logloss_damage_max": config["selection"]["logloss_damage_max"], "primary": "equal_scenario_mean_EU", "tie_break": ["compute_count", "candidate_id"], "no_eligible_policy": "NO_ELIGIBLE_CANDIDATE"},
        "output_root": f"{config['output_root']}/seed{seed}",
        "runtime": config["runtime"],
    }


def _run_name(seed: int, method: str) -> str:
    return f"ali_stage9_{method.lower()}_seed{seed}_v1"


def _run_dir(workspace: Path, config: dict[str, Any], seed: int, method: str) -> Path:
    return workspace / config["output_root"] / f"seed{seed}" / "runs" / _run_name(seed, method)


def _evaluate_bota(repo: Path, workspace: Path, runner_config: dict[str, Any], benchmark: dict[str, Any], registry: dict[str, Any], destination: Path) -> list[dict[str, Any]]:
    device = torch.device("cuda:0")
    _, users, _, select = stage7._development_partition(repo, workspace, runner_config, benchmark)
    tokenizer = T5Tokenizer.from_pretrained(runner._base_t5_config(workspace, benchmark)["paths"]["model_dir"])
    development = JsonPromptDataset(workspace / benchmark["source"]["development_json"], tokenizer)
    references = stage7._reference_runs(workspace, runner_config)
    source = references["BOTA-T2-Short"]
    rows: list[dict[str, Any]] = []
    for scenario in registry["scenarios"]:
        sid = scenario["id"]; request = set(map(int, scenario["users"])); local = [position for position, index in enumerate(select) if users[index] in request]
        original = stage7._predict_reference(workspace, benchmark, references["Original-Short-200step"], "Original-Short", sid, development, select, device)
        exact = stage7._predict_reference(workspace, benchmark, references["Exact-Masked-Reference-200step"], "Retrain-Short", sid, development, select, device)
        begun = time.perf_counter(); prediction = stage7._predict_reference(workspace, benchmark, source, "BOTA-T2-Short", sid, development, select, device); elapsed = time.perf_counter() - begun
        row = stage7._metric("bota_r1", "BOTA", sid, prediction, original, exact, local, 200, elapsed, {"response_scale": 1.0, "frozen_formal_endpoint": True}, runner_config)
        row.update({"optimizer_steps": 200, "online_optimizer_steps": 0, "hvp_calls": 0, "source_model_run": str(source)})
        rows.append(row)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(destination / "reference.json", {"candidate_id": "bota_r1", "source_model_run": str(source), "source_directory_sha256": directory_hash(source), "test_accessed": False})
    return rows


def _execute_rows(repo: Path, workspace: Path, runner_config: dict[str, Any], benchmark: dict[str, Any], registry: dict[str, Any], seed: int, method: str, destination: Path) -> list[dict[str, Any]]:
    if method == "BOTA":
        return _evaluate_bota(repo, workspace, runner_config, benchmark, registry, destination / "candidate_states" / "bota_r1")
    if method in {"NegGrad", "PCGrad"}:
        rows = stage7._gradient_sweep(repo, workspace, runner_config, benchmark, registry, method, destination)
    elif method == "IFRU":
        rows = stage7._ifru_sweep(repo, workspace, runner_config, benchmark, registry, destination)
    elif method == "E2URec":
        rows = stage7._e2urec_sweep(repo, workspace, runner_config, benchmark, registry, destination)
    else:
        rows, _ = stage7._partition_sweep(repo, workspace, runner_config, benchmark, registry, method, destination)
    if len(rows) != 2 or {row["scenario"] for row in rows} != set(SCENARIOS) or {row["candidate_id"] for row in rows} != {SELECTED_IDS[method]}:
        raise RuntimeError(f"frozen Stage 9 candidate mismatch for seed {seed}/{method}")
    return rows


def preflight(repo: Path, workspace: Path, config_path: Path, seed: int, method: str) -> dict[str, Any]:
    config = load_config(repo, workspace, config_path); runner_config = _seed_config(config, seed); benchmark, registry = stage7._authority(repo, workspace, runner_config); _, users, fit, select = stage7._development_partition(repo, workspace, runner_config, benchmark)
    support = {scenario["id"]: {"fit": sum(users[index] in set(scenario["users"]) for index in fit), "select": sum(users[index] in set(scenario["users"]) for index in select)} for scenario in registry["scenarios"]}
    return {"schema": SCHEMA, "mode": "Preflight", "seed": seed, "method": method, "candidate_id": SELECTED_IDS[method], "scenarios": list(SCENARIOS), "development_fit": len(fit), "development_select": len(select), "request_support": support, "finaltest_accessed": False, "test_accessed": False}


def synthetic(config: dict[str, Any], seed: int, method: str) -> dict[str, Any]:
    return {"schema": SCHEMA, "mode": "SyntheticDryRun", "seed": seed, "method": method, "candidate_id": config["selected_configurations"][method]["candidate_id"], "candidate_cells": 2, "model_loaded": False, "optimizer_constructed": False, "finaltest_accessed": False, "test_accessed": False}


def execute(repo: Path, workspace: Path, config_path: Path, seed: int, method: str) -> dict[str, Any]:
    config = load_config(repo, workspace, config_path); runner_config = _seed_config(config, seed); benchmark, registry = stage7._authority(repo, workspace, runner_config); git = git_snapshot(repo)
    if seed not in HELD_OUT_SEEDS:
        raise ValueError("Stage 9 Full execution is restricted to held-out seeds 41 and 43")
    if config["runtime"]["repository_must_be_clean"] and not git["clean"]:
        raise RuntimeError("commit Amendment 08 and Stage 9 code before a formal Full run")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA GPU required")
    free, _ = torch.cuda.mem_get_info(0)
    if free / 2**30 < config["runtime"]["minimum_free_gib"]:
        raise RuntimeError("insufficient dedicated GPU memory")
    destination = _run_dir(workspace, config, seed, method)
    if destination.exists():
        raise FileExistsError(destination)
    stage = destination.parent / ".work" / f"{_run_name(seed, method)}.{uuid.uuid4().hex}.stage"; stage.mkdir(parents=True); begun = time.perf_counter(); previous = Path.cwd(); os.chdir(workspace)
    try:
        rows = _execute_rows(repo, workspace, runner_config, benchmark, registry, seed, method, stage)
        for row in rows:
            row["seed"] = seed; row["confirmation_role"] = "held_out"; row["frozen_configuration"] = True
        atomic_json(stage / "metrics.json", {"schema": SCHEMA, "seed": seed, "method": method, "candidate_id": SELECTED_IDS[method], "rows": rows, "finaltest_accessed": False, "test_accessed": False})
        flat = [{**{key: value for key, value in row.items() if key != "hyperparameters"}, "hyperparameters": json.dumps(row["hyperparameters"], sort_keys=True)} for row in rows]
        with (stage / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
        (stage / "report.md").write_text(f"# Stage 9 cross-seed Development confirmation\n\nSeed: `{seed}`; method: `{method}`; frozen candidate: `{SELECTED_IDS[method]}`.\n\nFinalTest was not opened.\n", encoding="utf-8", newline="\n")
        atomic_json(stage / "run_state.json", {"schema": SCHEMA, "status": "COMPLETED", "seed": seed, "method": method, "candidate_id": SELECTED_IDS[method], "candidate_cells": len(rows), "wall_time_seconds": time.perf_counter() - begun, "maximum_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30, "finaltest_accessed": False, "test_accessed": False})
        atomic_json(stage / "contract.json", {"schema": SCHEMA, "config_sha256": sha256_file(config_path), "amendment_sha256": sha256_file(repo / config["parent_amendment"]["path"]), "stage7_audit_sha256": sha256_file(repo / config["stage7_audit"]["path"]), "stage8_audit_sha256": sha256_file(repo / config["stage8_audit"]["path"]), "git": git, "seed": seed, "method": method, "candidate_id": SELECTED_IDS[method], "split": "Development/Dev-Select", "finaltest_accessed": False, "test_accessed": False})
        (stage / "COMPLETED").write_text(MARKER + "\n", encoding="utf-8", newline="\n")
        state_hashes = {str(path.parent.relative_to(stage / "candidate_states")).replace("\\", "/"): directory_hash(path.parent) for path in sorted((stage / "candidate_states").rglob("adapter_model.pt"))}
        state_hashes.update({str(path.parent.relative_to(stage / "candidate_states")).replace("\\", "/"): directory_hash(path.parent) for path in sorted((stage / "candidate_states").rglob("component.json"))})
        state_hashes.update({str(path.parent.relative_to(stage / "candidate_states")).replace("\\", "/"): directory_hash(path.parent) for path in sorted((stage / "candidate_states").rglob("reference.json"))})
        atomic_json(stage / "manifest.json", {"schema": SCHEMA, "files": {name: sha256_file(stage / name) for name in RUN_REQUIRED - {"manifest.json", "candidate_states"}}, "candidate_states": state_hashes, "published_atomically": True, "finaltest_accessed": False, "test_accessed": False})
        destination.parent.mkdir(parents=True, exist_ok=True); os.replace(stage, destination)
        return {"status": "COMPLETED", "seed": seed, "method": method, "candidate_id": SELECTED_IDS[method], "run_dir": str(destination), "test_accessed": False}
    except Exception as error:
        if stage.exists(): shutil.rmtree(stage, ignore_errors=True)
        raise RuntimeError("Stage 9 run failed before atomic publication") from error
    finally:
        os.chdir(previous); gc.collect(); torch.cuda.empty_cache()


def analyze(workspace: Path, config: dict[str, Any], seed: int, method: str) -> dict[str, Any]:
    run = _run_dir(workspace, config, seed, method)
    if not run.is_dir() or {path.name for path in run.iterdir()} != RUN_REQUIRED or (run / "COMPLETED").read_text(encoding="utf-8") != MARKER + "\n":
        raise ValueError("invalid Stage 9 run")
    manifest = _json(run / "manifest.json")
    for name, expected in manifest["files"].items():
        if sha256_file(run / name) != expected: raise ValueError(f"Stage 9 artifact changed: {name}")
    for relative, expected in manifest["candidate_states"].items():
        if directory_hash(run / "candidate_states" / relative) != expected: raise ValueError(f"Stage 9 candidate state changed: {relative}")
    metrics = _json(run / "metrics.json"); _assert_no_test_access(metrics)
    if metrics["seed"] != seed or metrics["method"] != method or metrics["candidate_id"] != SELECTED_IDS[method] or len(metrics["rows"]) != 2:
        raise ValueError("Stage 9 run identity/completeness mismatch")
    return {"status": "COMPLETED", "seed": seed, "method": method, "candidate_id": SELECTED_IDS[method], "run_dir": str(run), "test_accessed": False}


def _selection_seed_rows(repo: Path, workspace: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    stage7_config_path = repo / "configs/ali_stage7_baseline_sweep_v1.yaml"; stage7_config = stage7.load_config(repo, workspace, stage7_config_path)
    stage8_config_path = repo / "configs/ali_stage8_bota_response_sweep_v1.yaml"; stage8_config = stage8.load_config(repo, stage8_config_path)
    stage8.analyze(workspace, stage8_config, "ali_stage8_bota_response_seed42_v1")
    bota_metrics = _json(workspace / stage8_config["output_root"] / "runs" / "ali_stage8_bota_response_seed42_v1" / "candidate_metrics.json")
    rows = [dict(row) for row in bota_metrics["rows"] if row["candidate_id"] == "bota_r1"]
    for method in METHODS[1:]:
        run_name = f"ali_stage7a_{method.lower()}_seed42_v1"; stage7.analyze(workspace, stage7_config, run_name)
        metrics = _json(workspace / stage7_config["output_root"] / "runs" / run_name / "candidate_metrics.json")
        rows.extend(dict(row) for row in metrics["rows"] if row["candidate_id"] == SELECTED_IDS[method])
    if len(rows) != len(METHODS) * len(SCENARIOS):
        raise ValueError("selection-seed source matrix incomplete")
    for row in rows:
        row["seed"] = 42; row["confirmation_role"] = "selection"; row["frozen_configuration"] = True
    return rows


def summarize(repo: Path, workspace: Path, config_path: Path) -> dict[str, Any]:
    config = load_config(repo, workspace, config_path); destination = workspace / config["output_root"] / "summary"
    if destination.exists():
        if {path.name for path in destination.iterdir()} != SUMMARY_REQUIRED or (destination / "COMPLETED").read_text(encoding="utf-8") != SUMMARY_MARKER + "\n": raise ValueError("invalid existing Stage 9 summary")
        manifest = _json(destination / "manifest.json")
        for name, expected in manifest["files"].items():
            if sha256_file(destination / name) != expected: raise ValueError(f"Stage 9 summary changed: {name}")
        decision = _json(destination / "decision.json")
        return {"status": decision["status"], "summary_dir": str(destination), "finaltest_authorized": decision["finaltest_authorized"], "test_accessed": False}
    rows = _selection_seed_rows(repo, workspace, config)
    source_hashes: dict[str, str] = {}
    for seed in HELD_OUT_SEEDS:
        for method in METHODS:
            analyze(workspace, config, seed, method); run = _run_dir(workspace, config, seed, method); source_hashes[f"{seed}/{method}"] = sha256_file(run / "manifest.json"); rows.extend(_json(run / "metrics.json")["rows"])
    if len(rows) != len(REPORTED_SEEDS) * len(METHODS) * len(SCENARIOS):
        raise RuntimeError("Stage 9 three-seed matrix incomplete")
    held_out = [row for row in rows if row["seed"] in HELD_OUT_SEEDS]
    bota = [row for row in held_out if row["method"] == "BOTA"]
    utility_pass = all(row["utility_eligible"] and not row["prediction_collapse"] for row in bota)
    residual_pass = all(row["local_residual"] is not None and row["local_residual"] < 1.0 for row in bota)
    bota_mean = float(np.mean([row["E_U"] for row in bota]))
    comparisons = {}
    for method in METHODS[1:]:
        method_mean = float(np.mean([row["E_U"] for row in held_out if row["method"] == method]))
        comparisons[method] = {"bota_mean_E_U": bota_mean, "baseline_mean_E_U": method_mean, "bota_lower": bota_mean < method_mean, "difference_baseline_minus_bota": method_mean - bota_mean}
    comparative_pass = all(value["bota_lower"] for value in comparisons.values())
    stable = utility_pass and residual_pass and comparative_pass
    aggregates = {}
    for method in METHODS:
        cells = [row for row in rows if row["method"] == method]
        aggregates[method] = {"three_seed_six_cell_mean_E_U": float(np.mean([row["E_U"] for row in cells])), "three_seed_six_cell_mean_residual": float(np.mean([row["local_residual"] for row in cells])), "utility_eligible_cells": sum(bool(row["utility_eligible"] and not row["prediction_collapse"]) for row in cells), "cell_count": len(cells)}
    decision = {"schema": SCHEMA, "status": config["decision"]["success_status"] if stable else config["decision"]["failure_status"], "decision_scope_seeds": list(HELD_OUT_SEEDS), "selection_seed_excluded_from_decision": 42, "bota_utility_every_cell": utility_pass, "bota_residual_below_one_every_cell": residual_pass, "bota_mean_eu_below_each_baseline": comparative_pass, "held_out_bota_mean_E_U": bota_mean, "comparisons": comparisons, "finaltest_authorized": stable, "finaltest_preregistered": False, "next_step": "Create and commit a separate FinalTest preregistration before any FinalTest access." if stable else "Do not create or run a new FinalTest evaluation under this protocol.", "finaltest_accessed": False, "test_accessed": False}
    stage = destination.parent / ".work" / f"summary.{uuid.uuid4().hex}.stage"; stage.mkdir(parents=True)
    try:
        atomic_json(stage / "metrics.json", {"schema": SCHEMA, "rows": rows, "aggregates": aggregates, "sisa_stage7_status": "NO_ELIGIBLE_CANDIDATE", "finaltest_accessed": False, "test_accessed": False})
        flat = [{**{key: value for key, value in row.items() if key != "hyperparameters"}, "hyperparameters": json.dumps(row["hyperparameters"], sort_keys=True)} for row in rows]
        with (stage / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
        atomic_json(stage / "decision.json", decision)
        atomic_json(stage / "contract.json", {"schema": SCHEMA, "config_sha256": sha256_file(config_path), "amendment_sha256": sha256_file(repo / config["parent_amendment"]["path"]), "stage7_audit_sha256": sha256_file(repo / config["stage7_audit"]["path"]), "stage8_audit_sha256": sha256_file(repo / config["stage8_audit"]["path"]), "source_manifest_sha256": source_hashes, "selection_seed_source": "frozen Stage 7/8 artifacts", "split": "Development/Dev-Select", "finaltest_accessed": False, "test_accessed": False})
        report = ["# Stage 9 cross-seed Development confirmation", "", f"Decision: `{decision['status']}`.", "", f"FinalTest authorized for separate preregistration: `{'yes' if stable else 'no'}`.", "", "Seed 42 selected the settings and is reported but excluded from the binary confirmation decision.", "", "| Method | Three-seed mean E_U | Mean residual | Eligible cells |", "|---|---:|---:|---:|"]
        for method in METHODS:
            entry = aggregates[method]; report.append(f"| {method} | {entry['three_seed_six_cell_mean_E_U']:.9f} | {entry['three_seed_six_cell_mean_residual']:.6f} | {entry['utility_eligible_cells']} / {entry['cell_count']} |")
        report.extend(["", "SISA is not promoted: Stage 7 reported `NO_ELIGIBLE_CANDIDATE`.", "", "FinalTest was not opened.", ""])
        (stage / "report.md").write_text("\n".join(report), encoding="utf-8", newline="\n")
        (stage / "COMPLETED").write_text(SUMMARY_MARKER + "\n", encoding="utf-8", newline="\n")
        atomic_json(stage / "manifest.json", {"schema": SCHEMA, "files": {name: sha256_file(stage / name) for name in SUMMARY_REQUIRED - {"manifest.json"}}, "published_atomically": True, "finaltest_accessed": False, "test_accessed": False})
        destination.parent.mkdir(parents=True, exist_ok=True); os.replace(stage, destination)
    finally:
        if stage.exists(): shutil.rmtree(stage, ignore_errors=True)
    return {"status": decision["status"], "summary_dir": str(destination), "finaltest_authorized": stable, "test_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--repo", type=Path, default=Path.cwd()); parser.add_argument("--workspace", type=Path, required=True); parser.add_argument("--config", type=Path, default=Path("configs/ali_stage9_crossseed_development_v1.yaml")); parser.add_argument("--mode", choices=["Preflight", "SyntheticDryRun", "Full", "Analyze", "Summarize"], default="Preflight"); parser.add_argument("--seed", type=int, choices=HELD_OUT_SEEDS, default=41); parser.add_argument("--method", choices=METHODS, default="BOTA"); args = parser.parse_args(); repo = args.repo.resolve(); workspace = args.workspace.resolve(); config_path = (repo / args.config).resolve() if not args.config.is_absolute() else args.config.resolve(); config = load_config(repo, workspace, config_path)
    if args.mode == "Preflight": result = preflight(repo, workspace, config_path, args.seed, args.method)
    elif args.mode == "SyntheticDryRun": result = synthetic(config, args.seed, args.method)
    elif args.mode == "Full": result = execute(repo, workspace, config_path, args.seed, args.method)
    elif args.mode == "Analyze": result = analyze(workspace, config, args.seed, args.method)
    else: result = summarize(repo, workspace, config_path)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
