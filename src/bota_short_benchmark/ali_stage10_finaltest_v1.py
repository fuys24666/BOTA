"""One separately preregistered, post-confirmation FinalTest re-evaluation.

Preflight and SyntheticDryRun do not materialize test records. Full requires
an explicit confirmation, a clean committed repository, frozen model hashes
and a permanent one-shot access reservation. No training is performed.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import T5Tokenizer

from src.diagnostics.t5_reconstructed_official import JsonPromptDataset
from src.paper_baselines import partitioned
from src.paper_if_a2.common import atomic_json, atomic_text, canonical_hash, directory_hash, git_snapshot, sha256_file
from . import ali_stage9_crossseed_development_v1 as stage9
from . import ali_stage9_result_audit_v1 as audit9
from . import evaluation
from . import multiseed_finaltest_v3 as lineage
from . import paper_evaluation_v2 as paper_eval
from . import runner


SCHEMA = "bota-ali-stage10-finaltest-v1"
MARKER = "BOTA_ALI_STAGE10_FINALTEST_V1_COMPLETED"
OUTPUT_ROOT = "outputs/ali_stage10_finaltest_v1"
RUN_NAME = "ali_stage10_finaltest_v1"
METHODS = audit9.ENDPOINT_METHODS
SEEDS = stage9.REPORTED_SEEDS
SCENARIOS = stage9.SCENARIOS
RUN_REQUIRED = {"COMPLETED", "metrics.json", "metrics.csv", "summary.csv", "decision.json", "predictions.jsonl", "contract.json", "run_state.json", "report.md", "manifest.json"}
_TEST_CAPABILITY = object()
EXPECTED_EVALUATION = {"inference_batch_size": 4, "bootstrap_unit": "authoritative_requested_user", "bootstrap_resamples": 2000, "confidence": 0.95, "auc_damage_max": 0.005, "logloss_damage_max": 0.01, "probability_collapse_std": 0.000001, "aggregation": "per_scenario_three_seed_mean_and_sample_std_plus_six_cell_mean_EU", "primary": "requested_user_mean_absolute_yes_probability_error_to_exact_masked"}
SUMMARY_FIELDS = ("E_U", "D_U", "local_exact_masked_residual", "toward_exact_rate", "overall_auc", "overall_acc", "overall_log_loss", "yes_no_jsd_to_exact", "probability_l2_rms_to_exact", "mean_answer_loss", "retain_retrain_absolute_error", "p5_retrain_residual")


class _AuthorizedDataset(JsonPromptDataset):
    def __init__(self, path: Path, tokenizer, capability: object):
        if capability is not _TEST_CAPABILITY: raise PermissionError("Stage 10 FinalTest access capability required")
        self.records = json.loads(path.read_text(encoding="utf-8"))
        self.tokenizer = tokenizer


def _path(workspace: Path, relative: str) -> Path:
    value = (workspace / relative).resolve()
    if workspace.resolve() not in value.parents: raise ValueError("artifact path must stay inside the frozen workspace")
    return value


def load_config(repo: Path, workspace: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    required = {"schema", "test_access_policy", "parent_amendment", "stage9_audit", "stage9_config", "protocol", "evaluation", "access", "output_root", "runtime"}
    if not isinstance(config, dict) or set(config) != required or config["schema"] != SCHEMA or config["test_access_policy"] != "separately_preregistered_one_shot_reevaluation": raise ValueError("invalid Stage 10 configuration surface")
    for field in ("parent_amendment", "stage9_audit", "stage9_config"):
        binding = config[field]
        if sha256_file(repo / binding["path"]) != binding["sha256"]: raise ValueError(f"Stage 10 {field} hash changed")
    amendment = stage9._json(repo / config["parent_amendment"]["path"])
    if amendment["amendment_id"] != config["parent_amendment"]["amendment_id"] or amendment["status"] != "FROZEN_BEFORE_STAGE10_FINALTEST_ACCESS": raise ValueError("Amendment 09 identity/status changed")
    audit = stage9._json(repo / config["stage9_audit"]["path"])
    if audit["status"] != "FROZEN_COMPLETE_STAGE9" or audit["decision"]["status"] != "STABLE_CONFIRMED" or not audit["decision"]["finaltest_authorized"]: raise ValueError("Stage 9 did not authorize separate FinalTest preregistration")
    stage9._assert_no_test_access(audit)
    recomputed = audit9.stability(audit["rows"])
    if not all(recomputed[key] for key in ("bota_utility_every_cell", "bota_residual_below_one_every_cell", "bota_mean_eu_below_each_baseline")): raise ValueError("Stage 9 confirmation failed independent recomputation")
    config9 = stage9.load_config(repo, workspace, repo / config["stage9_config"]["path"])
    expected_protocol = {"seeds": list(SEEDS), "scenarios": list(SCENARIOS), "methods": list(METHODS), "split": "FinalTest", "rows": 20000, "optimizer_updates": 0, "historical_finaltest_exposure": True, "request_replacement": False}
    if config["protocol"] != expected_protocol or config["evaluation"] != EXPECTED_EVALUATION: raise ValueError("Stage 10 fixed protocol/metrics changed")
    if config["access"] != {"confirm_switch_required": True, "one_fixed_cohort": True, "automatic_retry": False, "preflight_test_materialization": False, "partial_publication": False} or config["output_root"] != OUTPUT_ROOT: raise ValueError("Stage 10 one-shot access policy changed")
    if config["runtime"] != {"required_cuda_devices": 1, "minimum_free_gib": 10.0, "hard_peak_reserved_gib": 14.0, "repository_must_be_clean": True}: raise ValueError("Stage 10 runtime contract changed")
    if amendment["frozen_confirmation_boundary"]["sha256"] != config["stage9_audit"]["sha256"] or amendment["stage10_finaltest"]["methods"] != list(METHODS): raise ValueError("Amendment 09 and executable protocol disagree")
    return config, audit, config9


def _adapter_signature(path: Path) -> str:
    state = torch.load(path / "adapter_model.pt", map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not isinstance(state.get("A"), dict) or not isinstance(state.get("B"), dict): raise ValueError("invalid frozen fixed-A/B adapter")
    if len(state["A"]) != 72 or len(state["B"]) != 72: raise ValueError("unexpected fixed-A/B module count")
    signature = []
    for name, b in sorted(state["B"].items()):
        a = state["A"].get(name[:-2]) if name.endswith(".B") else None
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor) or a.ndim != 2 or b.ndim != 2 or a.shape[0] != 16 or b.shape[1] != 16 or not torch.isfinite(a).all() or not torch.isfinite(b).all(): raise ValueError(f"invalid frozen coordinate: {name}")
        signature.append((name, list(a.shape), list(b.shape)))
    return canonical_hash(signature)


def _cohort_binding(audit: dict[str, Any]) -> str:
    # Neither run name nor git commit can create another access to this cohort.
    return canonical_hash({"schema": SCHEMA, "endpoints": audit["endpoint_inventory"], "dependencies": audit["shared_model_dependencies"], "seeds": SEEDS, "scenarios": SCENARIOS, "lineage": "ML-1M-authoritative-chronological-last-20000-v1"})


def preflight(repo: Path, workspace: Path, config_path: Path) -> dict[str, Any]:
    config, audit, config9 = load_config(repo, workspace, config_path)
    for name, dependency in audit["shared_model_dependencies"].items():
        path = _path(workspace, dependency["path"])
        actual = directory_hash(path) if path.is_dir() else sha256_file(path)
        if actual != dependency["sha256"]: raise ValueError(f"shared model dependency changed: {name}")
    expected = {(seed, sid, method) for seed in SEEDS for sid in SCENARIOS for method in METHODS}
    entries = audit["endpoint_inventory"]
    if len(entries) != len(expected) or {(entry["seed"], entry["scenario"], entry["method"]) for entry in entries} != expected: raise ValueError("frozen endpoint matrix incomplete or duplicated")
    signatures = set()
    for entry in entries:
        path = _path(workspace, entry["path"])
        if not path.is_dir() or directory_hash(path) != entry["directory_sha256"]: raise ValueError(f"frozen endpoint changed: {entry['seed']}/{entry['scenario']}/{entry['method']}")
        if entry["kind"] == "fixed_ab": signatures.add(_adapter_signature(path))
        elif entry["kind"] == "receraser_ensemble": partitioned.analyze_run(path, partitioned.MARKERS["receraser"])
        else: raise ValueError("unknown endpoint kind")
    if len(signatures) != 1: raise ValueError("fixed-A/B architecture signatures disagree")
    for seed in SEEDS:
        benchmark, registry = stage9.stage7._authority(repo, workspace, stage9._seed_config(config9, seed))
        del benchmark
        for entry in entries:
            if entry["seed"] == seed:
                scenario = next(row for row in registry["scenarios"] if row["id"] == entry["scenario"])
                if entry["users"] != list(map(int, scenario["users"])) or entry["registry_sha256"] != registry["registry_sha256"]: raise ValueError("frozen request or registry changed")
    return {"schema": SCHEMA, "mode": "Preflight", "endpoint_cells": len(entries), "all_model_hashes_verified": True, "cpu_adapter_signatures_verified": True, "cohort_binding_sha256": _cohort_binding(audit), "historical_finaltest_exposure": True, "final_test_rows_materialized": 0, "test_loader_built": False, "model_loaded_on_gpu": False, "optimizer_constructed": False, "test_accessed": False}


def _predict_endpoint(workspace: Path, benchmark: dict[str, Any], entry: dict[str, Any], dataset, device, indices=range(20000)):
    path = _path(workspace, entry["path"])
    if directory_hash(path) != entry["directory_sha256"]: raise ValueError("frozen endpoint changed after admission preflight")
    if entry["kind"] == "receraser_ensemble":
        return evaluation._ensemble_predictions(workspace, benchmark, {"component_run": str(path), "method_id": "RecEraser-Adapter-Short"}, dataset, indices, device, 4)
    state = torch.load(path / "adapter_model.pt", map_location="cpu", weights_only=True)
    model = evaluation.load_legacy_model(workspace / benchmark["source"]["original_checkpoint"])
    try:
        names, parameters = evaluation.install_fixed_ab_coordinate(model, state["A"], 32)
        with torch.no_grad():
            for name, parameter in zip(names, parameters): parameter.copy_(state["B"][name].to(parameter))
        model = model.to(device).eval()
        return evaluation._single_predictions(model, dataset, indices, device, 4)
    finally:
        del model, state; gc.collect(); torch.cuda.empty_cache()


def _metric(method: str, predictions: dict[str, dict[str, Any]], selected: list[int], users: list[int], config: dict[str, Any], bootstrap_seed: int) -> dict[str, Any]:
    reference = predictions["Exact-Masked"]
    if any(value["gold_label"] != reference["gold_label"] or value["sample_order_sha256"] != reference["sample_order_sha256"] for value in predictions.values()): raise ValueError("FinalTest prediction label/order mismatch")
    p = np.asarray(predictions[method]["probability"], dtype=np.float64); e = np.asarray(reference["probability"], dtype=np.float64); o = np.asarray(predictions["Original"]["probability"], dtype=np.float64); r = np.asarray(predictions["Retain-Retrain-P5"]["probability"], dtype=np.float64); chosen = np.asarray(selected, dtype=np.int64)
    if not selected or not all(np.isfinite(np.asarray(value["probability"])).all() and np.isfinite(np.asarray(value["answer_loss"])).all() for value in predictions.values()): raise ValueError("empty request support or nonfinite predictions")
    names = {"Exact-Masked-Reference-200step": predictions["Exact-Masked"], "Original-Short-200step": predictions["Original"], "Retain-Retrain-P5-Short": predictions["Retain-Retrain-P5"], "FullControl-P5-Short": predictions["FullControl-P5"]}
    row = paper_eval._metric_row(method, predictions[method], names, selected, users, config["evaluation"]["bootstrap_resamples"], bootstrap_seed)
    # Overall utility uses exactly the same Original control as Development.
    y = np.asarray(reference["gold_label"], dtype=np.int64); clipped = o.clip(1e-12, 1 - 1e-12)
    original_auc = float(stage9.stage7.roc_auc_score(y, o)); original_ll = float(np.mean(-(y * np.log(clipped) + (1-y) * np.log(1-clipped))))
    auc_damage = original_auc - row["overall_auc"]; ll_damage = row["overall_log_loss"] - original_ll
    row.update({"method": method, "E_U": float(np.mean(np.abs(p[chosen]-e[chosen]))), "D_U": float(np.mean(np.abs(o[chosen]-e[chosen]))), "toward_exact_rate": row["local_toward"] / len(chosen), "yes_no_jsd_to_exact": float(np.mean(evaluation.bernoulli_jsd(p[chosen], e[chosen]))), "probability_l2_rms_to_exact": float(np.sqrt(np.mean((p[chosen]-e[chosen])**2))), "mean_answer_loss": float(np.mean(predictions[method]["answer_loss"])), "retain_retrain_absolute_error": float(np.mean(np.abs(p[chosen]-r[chosen]))), "auc_damage": auc_damage, "logloss_damage": ll_damage, "utility_eligible": auc_damage <= config["evaluation"]["auc_damage_max"] and ll_damage <= config["evaluation"]["logloss_damage_max"], "prediction_collapse": bool(np.std(p) < config["evaluation"]["probability_collapse_std"]), "test_accessed": True})
    return row


def aggregate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = {(seed, scenario, method) for seed in SEEDS for scenario in SCENARIOS for method in METHODS}
    if len(rows) != len(expected) or {(row["seed"], row["scenario"], row["method"]) for row in rows} != expected: raise ValueError("FinalTest matrix incomplete or duplicated")
    summary = []
    for scenario in SCENARIOS:
        for method in METHODS:
            cells = [row for row in rows if row["scenario"] == scenario and row["method"] == method]
            item = {"scenario": scenario, "method": method, "seeds": 3}
            for field in SUMMARY_FIELDS:
                values = [row[field] for row in cells]
                valid = all(value is not None and math.isfinite(value) for value in values)
                item[field + "_mean"] = float(np.mean(values)) if valid else None
                item[field + "_std"] = float(np.std(values, ddof=1)) if valid else None
            summary.append(item)
    bota = [row for row in rows if row["method"] == "BOTA"]; bota_mean = float(np.mean([row["E_U"] for row in bota]))
    comparisons = {method: {"bota_mean_E_U": bota_mean, "baseline_mean_E_U": float(np.mean([row["E_U"] for row in rows if row["method"] == method]))} for method in stage9.METHODS[1:]}
    for value in comparisons.values(): value["bota_lower"] = value["bota_mean_E_U"] < value["baseline_mean_E_U"]
    utility = all(row["utility_eligible"] and not row["prediction_collapse"] for row in bota)
    residual = all(row["local_exact_masked_residual"] is not None and row["local_exact_masked_residual"] < 1 for row in bota)
    comparative = all(value["bota_lower"] for value in comparisons.values())
    decision = {"schema": SCHEMA, "status": "FINALTEST_LOCAL_COMPARISON_CONFIRMED" if utility and residual and comparative else "FINALTEST_LOCAL_COMPARISON_NOT_CONFIRMED", "bota_utility_all_six_cells": utility, "bota_residual_below_one_all_six_cells": residual, "bota_mean_eu_below_each_baseline": comparative, "bota_six_cell_mean_E_U": bota_mean, "comparisons": comparisons, "historical_finaltest_exposure": True, "sisa_status": "NO_ELIGIBLE_CANDIDATE", "retuning_or_retry_authorized": False, "test_accessed": True}
    return summary, decision


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows: writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def _reserve(ledger_root: Path, binding: str) -> Path:
    ledger_root.mkdir(parents=True, exist_ok=True); reservation = ledger_root / binding
    try: reservation.mkdir()
    except FileExistsError as error: raise RuntimeError("this fixed FinalTest cohort was already reserved/accessed; automatic retry is forbidden") from error
    return reservation


def synthetic(config: dict[str, Any]) -> dict[str, Any]:
    probabilities = np.linspace(.15, .85, 32); users = [index // 4 for index in range(32)]; labels = [index % 2 for index in range(32)]
    predictions = {}
    for method in METHODS:
        offset = {"Original": 0, "Exact-Masked": .01, "FullControl-P5": .02, "Retain-Retrain-P5": .03, "BOTA": .009}.get(method, .04)
        predictions[method] = {"probability": (probabilities+offset).tolist(), "gold_label": labels, "answer_loss": [1.] * 32, "sample_order_sha256": canonical_hash(list(range(32)))}
    metrics = {method: _metric(method, predictions, list(range(32)), users, config, 4100) for method in METHODS}
    rows = [{**metrics[method], "seed": seed, "scenario": scenario} for seed in SEEDS for scenario in SCENARIOS for method in METHODS]
    summary, decision = aggregate(rows)
    return {"schema": SCHEMA, "mode": "SyntheticDryRun", "synthetic_cells": len(rows), "summary_rows": len(summary), "synthetic_decision": decision["status"], "actual_test_accessed": False, "test_loader_built": False, "optimizer_constructed": False}


def execute(repo: Path, workspace: Path, config_path: Path, confirm: bool) -> dict[str, Any]:
    if not confirm: raise PermissionError("Stage 10 Full requires -ConfirmFinalTest")
    config, audit, config9 = load_config(repo, workspace, config_path); authority = preflight(repo, workspace, config_path); git = git_snapshot(repo); admitted_config_hash = sha256_file(config_path)
    if not git["clean"]: raise RuntimeError("commit Amendment 09 and Stage 10 code before FinalTest access")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1: raise RuntimeError("exactly one CUDA GPU required")
    free, _ = torch.cuda.mem_get_info(0)
    if free / 2**30 < config["runtime"]["minimum_free_gib"]: raise RuntimeError("insufficient dedicated GPU memory")
    torch.cuda.reset_peak_memory_stats(0)
    output = workspace / OUTPUT_ROOT; destination = output / "evaluations" / RUN_NAME
    if destination.exists(): raise FileExistsError("completed FinalTest run exists; use Analyze")
    binding = authority["cohort_binding_sha256"]; reservation = _reserve(output / "access_ledger", binding)
    atomic_json(reservation / "access_started.json", {"schema": SCHEMA, "binding_sha256": binding, "git": git, "config_sha256": sha256_file(config_path), "amendment_sha256": config["parent_amendment"]["sha256"], "status": "FINALTEST_ACCESS_STARTED", "prediction_calls_attempted": 0, "prediction_calls_completed": 0, "test_accessed": True})
    stage = output / ".work" / uuid.uuid4().hex; stage.mkdir(parents=True); begun = time.perf_counter(); previous = Path.cwd(); os.chdir(workspace); attempted = completed = 0; rows = []; device = torch.device("cuda:0")
    try:
        final_rows = lineage._final_lineage(workspace / "data/ml-1m/raw_data"); final_path = lineage._validate_final_file(workspace, final_rows); final_users = [int(row.authoritative_user_id) for row in final_rows]
        atomic_json(reservation / "raw_replay_validated.json", {"schema": SCHEMA, "rows": len(final_rows), "processed_test_sha256": sha256_file(final_path), "raw_replay_exact": True, "test_accessed": True})
        with (stage / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for seed in SEEDS:
                benchmark, _ = stage9.stage7._authority(repo, workspace, stage9._seed_config(config9, seed)); base = runner._base_t5_config(workspace, benchmark)
                tokenizer = T5Tokenizer.from_pretrained(_path(workspace, audit["shared_model_dependencies"]["pretrained_model_directory"]["path"]), local_files_only=True)
                dataset = _AuthorizedDataset(final_path, tokenizer, _TEST_CAPABILITY)
                if len(dataset) != 20000: raise ValueError("FinalTest sample count mismatch")
                for number, scenario in enumerate(SCENARIOS):
                    entries = {entry["method"]: entry for entry in audit["endpoint_inventory"] if entry["seed"] == seed and entry["scenario"] == scenario}
                    request = set(entries["BOTA"]["users"]); selected = [index for index, user in enumerate(final_users) if user in request]
                    if not selected: raise ValueError(f"zero FinalTest support for seed {seed}/{scenario}; request replacement forbidden")
                    support = Counter(final_users[index] for index in selected); missing = sorted(request - set(support)); predictions = {}; inference_times = {}
                    for method in METHODS:
                        attempted += 1
                        atomic_json(reservation / "progress.json", {"seed": seed, "scenario": scenario, "method": method, "prediction_calls_attempted": attempted, "prediction_calls_completed": completed, "test_accessed": True})
                        torch.cuda.synchronize(); prediction_begun = time.perf_counter()
                        predictions[method] = _predict_endpoint(workspace, benchmark, entries[method], dataset, device)
                        torch.cuda.synchronize(); inference_times[method] = time.perf_counter()-prediction_begun; completed += 1
                        if torch.cuda.max_memory_reserved()/2**30 > config["runtime"]["hard_peak_reserved_gib"]: raise RuntimeError("Stage 10 GPU memory contract exceeded")
                        atomic_json(reservation / "progress.json", {"seed": seed, "scenario": scenario, "method": method, "prediction_calls_attempted": attempted, "prediction_calls_completed": completed, "test_accessed": True})
                        print(f"[stage10:inference] seed={seed} scenario={scenario} method={method} completed={completed}/60", flush=True)
                    for method in METHODS:
                        metric = _metric(method, predictions, selected, final_users, config, seed*100+number)
                        rows.append({"seed": seed, "scenario": scenario, "split": "FinalTest", "selected_samples": len(selected), "selected_users_with_support": len(support), "missing_user_support": missing, "endpoint_directory_sha256": entries[method]["directory_sha256"], "inference_wall_time_seconds": inference_times[method], "optimizer_updates": 0, **metric})
                        for index in selected:
                            handle.write(json.dumps({"seed": seed, "scenario": scenario, "method": method, "final_test_index": index, "user_hash": canonical_hash([seed, final_users[index]]), "gold_label": predictions[method]["gold_label"][index], "probability": predictions[method]["probability"][index], "answer_loss": predictions[method]["answer_loss"][index]}, sort_keys=True) + "\n")
                    handle.flush(); del predictions; gc.collect(); torch.cuda.empty_cache()
        if sha256_file(config_path) != admitted_config_hash or git_snapshot(repo) != git: raise RuntimeError("repository or preregistration changed during FinalTest execution")
        summary, decision = aggregate(rows)
        atomic_json(stage / "metrics.json", {"schema": SCHEMA, "rows": rows, "summary": summary, "historical_finaltest_exposure": True, "test_accessed": True}); _write_csv(stage / "metrics.csv", rows); _write_csv(stage / "summary.csv", summary); atomic_json(stage / "decision.json", decision)
        atomic_json(stage / "contract.json", {"schema": SCHEMA, "config_sha256": sha256_file(config_path), "amendment_sha256": config["parent_amendment"]["sha256"], "stage9_audit_sha256": config["stage9_audit"]["sha256"], "cohort_binding_sha256": binding, "git": git, "endpoint_inventory": audit["endpoint_inventory"], "historical_finaltest_exposure": True, "optimizer_updates": 0, "test_accessed": True})
        atomic_json(stage / "run_state.json", {"schema": SCHEMA, "status": "COMPLETED", "cells": len(rows), "prediction_calls_attempted": attempted, "prediction_calls_completed": completed, "wall_time_seconds": time.perf_counter()-begun, "maximum_peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30, "optimizer_updates": 0, "test_accessed": True})
        report = ["# Stage 10 post-confirmation FinalTest re-evaluation", "", f"Result: `{decision['status']}`.", "", "Historical test exposure: this is a post-Development-tuning re-evaluation on the previously used FinalTest cohort, not an untouched or independently collected new holdout.", "", "All 60 cells are reported. No training or selection was performed. SISA remains `NO_ELIGIBLE_CANDIDATE`.", "", "| Scenario | Method | E_U mean | E_U std | Local residual mean | Retain-P5 residual mean | AUC mean |", "|---|---|---:|---:|---:|---:|---:|"]
        def formatted(value): return "--" if value is None else f"{value:.9f}"
        for row in summary: report.append(f"| {row['scenario']} | {row['method']} | {formatted(row['E_U_mean'])} | {formatted(row['E_U_std'])} | {formatted(row['local_exact_masked_residual_mean'])} | {formatted(row['p5_retrain_residual_mean'])} | {formatted(row['overall_auc_mean'])} |")
        report.extend(["", "Retain-Retrain-P5 is a separately stopped converged-endpoint sensitivity, with FullControl-P5 as its control. Exact-Masked is the matched 200-step target. Earlier long-horizon, attack and sequential limitations are unchanged.", ""])
        atomic_text(stage / "report.md", "\n".join(report)); atomic_text(stage / "COMPLETED", MARKER+"\n")
        atomic_json(stage / "manifest.json", {"schema": SCHEMA, "files": {name: sha256_file(stage / name) for name in sorted(RUN_REQUIRED - {"manifest.json"})}, "published_atomically": True, "test_accessed": True})
        destination.parent.mkdir(parents=True, exist_ok=True); os.replace(stage, destination)
        atomic_json(reservation / "access_completed.json", {"schema": SCHEMA, "status": "FINALTEST_ACCESS_COMPLETED", "binding_sha256": binding, "run_dir": str(destination), "manifest_sha256": sha256_file(destination / "manifest.json"), "prediction_calls_completed": completed, "test_accessed": True})
        return {"status": "COMPLETED", "comparison": decision["status"], "run_dir": str(destination), "cells": len(rows), "test_accessed": True}
    except Exception as error:
        atomic_json(reservation / "access_failed_no_retry.json", {"schema": SCHEMA, "status": "FINALTEST_ACCESS_FAILED_NO_RETRY", "binding_sha256": binding, "reason": type(error).__name__, "message": str(error), "prediction_calls_attempted": attempted, "prediction_calls_completed": completed, "metric_cells_computed": len(rows), "partial_stage_path": str(stage), "test_accessed": True})
        raise RuntimeError("Stage 10 failed; preserve the ledger and do not retry automatically") from error
    finally:
        os.chdir(previous); gc.collect(); torch.cuda.empty_cache()


def analyze(repo: Path, workspace: Path, config_path: Path) -> dict[str, Any]:
    config, audit, _ = load_config(repo, workspace, config_path); run = workspace / OUTPUT_ROOT / "evaluations" / RUN_NAME
    if not run.is_dir() or {path.name for path in run.iterdir()} != RUN_REQUIRED or (run / "COMPLETED").read_text(encoding="utf-8") != MARKER+"\n": raise ValueError("invalid Stage 10 run")
    manifest = stage9._json(run / "manifest.json")
    for name, expected in manifest["files"].items():
        if sha256_file(run / name) != expected: raise ValueError(f"Stage 10 artifact changed: {name}")
    contract = stage9._json(run / "contract.json")
    if contract["config_sha256"] != sha256_file(config_path) or contract["cohort_binding_sha256"] != _cohort_binding(audit) or contract["stage9_audit_sha256"] != config["stage9_audit"]["sha256"]: raise ValueError("Stage 10 contract changed")
    metrics = stage9._json(run / "metrics.json"); summary, decision = aggregate(metrics["rows"])
    if summary != metrics["summary"] or decision != stage9._json(run / "decision.json"): raise ValueError("Stage 10 summary/decision mismatch")
    return {"status": "COMPLETED", "comparison": decision["status"], "run_dir": str(run), "cells": len(metrics["rows"]), "test_accessed_during_analyze": False, "historical_run_test_accessed": True}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--repo", type=Path, default=Path.cwd()); parser.add_argument("--workspace", type=Path, required=True); parser.add_argument("--config", type=Path, default=Path("configs/ali_stage10_finaltest_v1.yaml")); parser.add_argument("--mode", choices=["Preflight", "SyntheticDryRun", "Full", "Analyze"], default="Preflight"); parser.add_argument("--confirm-final-test", action="store_true"); args = parser.parse_args()
    repo = args.repo.resolve(); workspace = args.workspace.resolve(); config_path = (repo / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    if args.mode == "Preflight": result = preflight(repo, workspace, config_path)
    elif args.mode == "SyntheticDryRun": result = synthetic(load_config(repo, workspace, config_path)[0])
    elif args.mode == "Full": result = execute(repo, workspace, config_path, args.confirm_final_test)
    else: result = analyze(repo, workspace, config_path)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
