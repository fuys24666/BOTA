"""Two-reference Development evaluation for the formal short-window paper study."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import T5Tokenizer

from src.diagnostics.ml1m_development_protocol import reconstruct_authoritative_rows
from src.diagnostics.t5_reconstructed_official import JsonPromptDataset
from src.paper_if_a2.common import atomic_json, canonical_hash, git_snapshot, safe_run_name, sha256_file

from . import evaluation as v1_eval
from . import paper_v2
from . import runner as v1_runner
from .protocol import validate_prepared
from .timing import SCENARIO_CHOICES, TIMING_SCHEMA, select_scenarios

SCHEMA = "bota-short-paper-evaluation-v2"
MARKER = "BOTA_SHORT_PAPER_EVALUATION_V2_COMPLETED"
ORDER = [
    "Original-Short-200step",
    "Exact-Masked-Reference-200step",
    "FullControl-P5-Short",
    "Retain-Retrain-P5-Short",
    "BOTA-T2-Short",
    "IFRU-Short-LoRA",
    "NegGrad-Mixed-Short-BOnly",
    "PCGrad-Short-BOnly",
    "SISA-Short-T5",
    "RecEraser-Adapter-Short",
]
OLD_IDS = {
    "Original-Short-200step": "Original-Short",
    "Exact-Masked-Reference-200step": "Retrain-Short",
    "BOTA-T2-Short": "BOTA-T2-Short",
    "IFRU-Short-LoRA": "IFRU-Short-LoRA",
    "SISA-Short-T5": "SISA-Short-T5",
    "RecEraser-Adapter-Short": "RecEraser-Adapter-Short",
}
NEW_IDS = {value: value for value in paper_v2.METHODS.values()}


def _new_run(root: Path, config: dict[str, Any], method_id: str, run_name: str, benchmark_name: str) -> Path:
    run = root / config["output_root"] / "models" / method_id / safe_run_name(run_name); required = {"COMPLETED", "contract.json", "manifest.json", "run_state.json", "scenarios"}
    if not run.is_dir() or {path.name for path in run.iterdir()} != required or (run / "COMPLETED").read_text(encoding="utf-8") != paper_v2.MARKER + "\n": raise ValueError(f"invalid {method_id} run")
    state = json.loads((run / "run_state.json").read_text(encoding="utf-8")); manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8")); contract = json.loads((run / "contract.json").read_text(encoding="utf-8"))
    if state.get("method_id") != method_id or state.get("benchmark_name") != benchmark_name or state.get("test_accessed") is not False or sha256_file(run / "run_state.json") != manifest.get("run_state_sha256") or contract.get("test_accessed") is not False: raise ValueError(f"{method_id} binding mismatch")
    return run


def resolve(root: Path, config: dict[str, Any], benchmark_name: str, names: dict[str, str]):
    v1_config = v1_runner.load_config(root / config["source"]["v1_config"]); _, _, registry = validate_prepared(root, v1_config, benchmark_name)
    if set(names) != set(ORDER) or any(not value for value in names.values()): raise ValueError("all ten frozen short-paper models are required")
    runs = {}
    for display, run_name in names.items():
        if display in OLD_IDS: runs[display] = v1_eval._validate_method(root, v1_config, OLD_IDS[display], run_name, benchmark_name)
        else: runs[display] = _new_run(root, config, display, run_name, benchmark_name)
    return v1_config, registry, runs


def _predict(root, config, v1_config, run, display, scenario, dataset, indices, device):
    if display in OLD_IDS: return v1_eval._predict(root, v1_config, run, OLD_IDS[display], scenario, dataset, indices, device)
    scenario_run = run / "scenarios" / scenario; manifest = json.loads((scenario_run / "scenario_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("method_id") != display or manifest.get("test_accessed") is not False: raise ValueError("new scenario manifest mismatch")
    model = v1_eval._load_fixed(root, v1_config, scenario_run, device)
    try: return v1_eval._single_predictions(model, dataset, indices, device, config["evaluation"]["inference_batch_size"])
    finally: v1_eval._release(model)


def _phase_timing(run: Path, scenario: str) -> dict[str, Any]:
    path = run / "scenarios" / scenario / "phase_timing.json"
    if not path.is_file(): raise ValueError(f"missing phase timing: {scenario}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != TIMING_SCHEMA or value.get("scenario") != scenario: raise ValueError("phase timing binding mismatch")
    for key in ("initialization_seconds", "offline_construction_seconds", "online_compute_seconds", "end_to_end_seconds", "online_total_seconds"):
        if not isinstance(value.get(key), (int, float)) or not math.isfinite(value[key]) or value[key] < 0: raise ValueError(f"invalid phase timing: {key}")
    publication = value.get("adapter_publication_seconds")
    if publication is not None and (not isinstance(publication, (int, float)) or not math.isfinite(publication) or publication < 0): raise ValueError("invalid adapter publication timing")
    return value


def _residual(candidate: np.ndarray, reference: np.ndarray, control: np.ndarray, selected: np.ndarray) -> dict[str, Any]:
    distance = np.abs(candidate[selected] - reference[selected]); baseline = np.abs(control[selected] - reference[selected]); denominator = float(np.mean(baseline)); point = float(np.mean(distance) / denominator) if denominator else None
    return {"point": point, "distance": float(np.mean(distance)), "denominator": denominator, "toward": int(np.sum(distance < baseline - 1e-15)), "away": int(np.sum(distance > baseline + 1e-15)), "equal": int(np.sum(np.abs(distance - baseline) <= 1e-15))}


def _cluster_bootstrap(candidate: np.ndarray, reference: np.ndarray, control: np.ndarray, selected: Sequence[int], users: Sequence[int], resamples: int, seed: int) -> dict[str, float | None]:
    clusters = sorted(set(int(users[index]) for index in selected)); mapping = {user: np.asarray([index for index in selected if int(users[index]) == user], dtype=np.int64) for user in clusters}; rng = np.random.default_rng(seed); values = []
    for _ in range(resamples):
        sampled = rng.choice(clusters, size=len(clusters), replace=True); indices = np.concatenate([mapping[int(user)] for user in sampled]); value = _residual(candidate, reference, control, indices)["point"]
        if value is not None and math.isfinite(value): values.append(value)
    if not values: return {"ci_lower": None, "ci_upper": None}
    return {"ci_lower": float(np.quantile(values, .025)), "ci_upper": float(np.quantile(values, .975))}


def _metric_row(display: str, prediction: dict[str, Any], predictions: dict[str, dict[str, Any]], selected: Sequence[int], users: Sequence[int], resamples: int, scenario_seed: int) -> dict[str, Any]:
    p = np.asarray(prediction["probability"], dtype=np.float64); y = np.asarray(prediction["gold_label"], dtype=np.int64); chosen = np.asarray(selected, dtype=np.int64)
    exact = np.asarray(predictions["Exact-Masked-Reference-200step"]["probability"], dtype=np.float64); original = np.asarray(predictions["Original-Short-200step"]["probability"], dtype=np.float64); retain = np.asarray(predictions["Retain-Retrain-P5-Short"]["probability"], dtype=np.float64); control = np.asarray(predictions["FullControl-P5-Short"]["probability"], dtype=np.float64)
    local = _residual(p, exact, original, chosen); converged = _residual(p, retain, control, chosen); local.update(_cluster_bootstrap(p, exact, original, selected, users, resamples, scenario_seed)); converged.update(_cluster_bootstrap(p, retain, control, selected, users, resamples, scenario_seed + 10_000))
    clipped = p.clip(1e-12, 1 - 1e-12)
    return {"method_id": display, "overall_auc": float(roc_auc_score(y, p)), "overall_acc": float(accuracy_score(y, p >= .5)), "overall_log_loss": float(np.mean(-(y * np.log(clipped) + (1-y) * np.log(1-clipped)))), "local_exact_masked_residual": local["point"], "local_ci_lower": local["ci_lower"], "local_ci_upper": local["ci_upper"], "local_toward": local["toward"], "local_away": local["away"], "p5_retrain_residual": converged["point"], "p5_ci_lower": converged["ci_lower"], "p5_ci_upper": converged["ci_upper"], "p5_toward": converged["toward"], "p5_away": converged["away"], "prediction_collapse": bool(np.std(p) < 1e-6), "finite": bool(np.isfinite(p).all())}


def execute(root: Path, config_path: Path, benchmark_name: str, run_name: str, names: dict[str, str], scenario_selection: str = "All") -> dict[str, Any]:
    config = paper_v2.load_config(config_path); run_seed = int(config["protocol"]["seed"]); v1_config, registry, runs = resolve(root, config, benchmark_name, names); git = git_snapshot(root)
    if not git["clean"]: raise RuntimeError("formal paper evaluation requires clean Git")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1: raise RuntimeError("exactly one CUDA GPU required")
    destination = root / config["output_root"] / "evaluations" / safe_run_name(run_name)
    if destination.exists(): raise FileExistsError(destination)
    selected_scenarios = select_scenarios(registry["scenarios"], scenario_selection)
    for display, run in runs.items():
        state = json.loads((run / "run_state.json").read_text(encoding="utf-8"))
        for scenario in selected_scenarios:
            if scenario["id"] not in state.get("scenarios", []) or not (run / "scenarios" / scenario["id"]).is_dir(): raise ValueError(f"{display} does not contain requested scenario {scenario['id']}")
    stage = destination.parent / ".work" / f"{safe_run_name(run_name)}.{uuid.uuid4().hex}.stage"; stage.mkdir(parents=True); device = torch.device("cuda:0"); base = v1_runner._base_t5_config(root, v1_config); tokenizer = T5Tokenizer.from_pretrained(base["paths"]["model_dir"]); dataset = JsonPromptDataset(root / v1_config["source"]["development_json"], tokenizer)
    if "development_user_ids" in v1_config["source"]: dev_users = json.loads((root / v1_config["source"]["development_user_ids"]).read_text(encoding="utf-8"))
    else:
        _, development, _ = reconstruct_authoritative_rows(root / v1_config["source"]["raw_data"]); dev_users = [int(row.authoritative_user_id) for row in development]
    if len(dataset) != registry["development_samples"] or len(dev_users) != registry["development_samples"]: raise RuntimeError("Development count mismatch")
    rows = []; efficiency_rows = []; safe_samples = []; shared_predictions: dict[str, dict[str, Any]] = {}; started = time.perf_counter()
    try:
        for scenario_number, scenario in enumerate(selected_scenarios):
            request = set(map(int, scenario["users"])); selected = [index for index, user in enumerate(dev_users) if user in request]
            minimum_support = 10 * int(config["protocol"]["deleted_samples"])
            if len(selected) < minimum_support: raise RuntimeError("insufficient Development support")
            predictions = {}
            for display in ORDER:
                if display in {"Original-Short-200step", "FullControl-P5-Short"} and display in shared_predictions: predictions[display] = shared_predictions[display]
                else:
                    predictions[display] = _predict(root, config, v1_config, runs[display], display, scenario["id"], dataset, list(range(len(dataset))), device)
                    if display in {"Original-Short-200step", "FullControl-P5-Short"}: shared_predictions[display] = predictions[display]
            authority = predictions["Exact-Masked-Reference-200step"]
            if any(value["gold_label"] != authority["gold_label"] or value["sample_order_sha256"] != authority["sample_order_sha256"] for value in predictions.values()): raise RuntimeError("Development label/order mismatch")
            for display in ORDER:
                row = _metric_row(display, predictions[display], predictions, selected, dev_users, config["evaluation"]["bootstrap_resamples"], run_seed * 100 + scenario_number); rows.append({"run_seed": run_seed, "scenario": scenario["id"], "composition": scenario["composition"], **row})
                timing = _phase_timing(runs[display], scenario["id"]); efficiency_rows.append({"scenario": scenario["id"], "method_id": display, "initialization_seconds": timing["initialization_seconds"], "offline_construction_seconds": timing["offline_construction_seconds"], "online_compute_seconds": timing["online_compute_seconds"], "adapter_publication_seconds": timing["adapter_publication_seconds"], "online_total_seconds": timing["online_total_seconds"], "end_to_end_seconds": timing["end_to_end_seconds"], "publication_included_in_online_compute": timing["publication_included_in_online_compute"]})
                p = predictions[display]["probability"]
                safe_samples.extend({"run_seed": run_seed, "scenario": scenario["id"], "method_id": display, "user_hash": canonical_hash([run_seed, int(dev_users[index])]), "development_index": index, "probability": float(p[index])} for index in selected)
        atomic_json(stage / "metrics.json", {"schema": SCHEMA, "benchmark_name": benchmark_name, "two_reference_policy": {"local": "Original-Short-200step -> Exact-Masked-Reference-200step", "converged": "FullControl-P5-Short -> Retain-Retrain-P5-Short"}, "metrics": rows, "test_accessed": False})
        with (stage / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = sorted({key for row in rows for key in row}); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        with (stage / "efficiency.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = list(efficiency_rows[0]); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(efficiency_rows)
        with (stage / "per_sample_metrics.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in safe_samples: handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        report = ["# BOTA short-window paper evaluation v2", "", "Development only. Local and P5 convergence residuals use different, explicitly named controls and are never pooled.", ""]
        for scenario in [row["id"] for row in selected_scenarios]:
            report.extend([f"## {scenario}", "", "| Method | AUC | LogLoss | Local residual [95% CI] | P5 residual [95% CI] |", "|---|---:|---:|---:|---:|"])
            for row in [value for value in rows if value["scenario"] == scenario]: report.append(f"| {row['method_id']} | {row['overall_auc']:.6f} | {row['overall_log_loss']:.6f} | {row['local_exact_masked_residual']:.4f} [{row['local_ci_lower']:.4f}, {row['local_ci_upper']:.4f}] | {row['p5_retrain_residual']:.4f} [{row['p5_ci_lower']:.4f}, {row['p5_ci_upper']:.4f}] |")
            report.append("")
        report.extend(["## Efficiency timing", "", "Offline construction, online compute, and adapter publication are reported separately in `efficiency.csv`. For methods whose component runner does not expose publication separately, that field is blank and the inclusion flag is true.", ""])
        (stage / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8", newline="\n"); state = {"schema": SCHEMA, "status": "COMPLETED", "benchmark_name": benchmark_name, "run_seed": run_seed, "models": ORDER, "scenarios": [row["id"] for row in selected_scenarios], "scenario_selection": scenario_selection, "wall_time_seconds": time.perf_counter() - started, "test_accessed": False}; atomic_json(stage / "run_state.json", state); atomic_json(stage / "provenance.json", {"schema": SCHEMA, "git": git, "run_seed": run_seed, "registry_sha256": registry["registry_sha256"], "model_runs": {key: str(value.resolve()) for key, value in runs.items()}, "scenario_selection": scenario_selection, "split": "Development", "final_test_accessed": False, "test_accessed": False}); (stage / "COMPLETED").write_text(MARKER + "\n", encoding="utf-8", newline="\n"); atomic_json(stage / "manifest.json", {"schema": SCHEMA, "files": {name: sha256_file(stage / name) for name in ("metrics.json", "metrics.csv", "efficiency.csv", "per_sample_metrics.jsonl", "report.md", "run_state.json", "provenance.json", "COMPLETED")}, "published_atomically": True, "test_accessed": False}); destination.parent.mkdir(parents=True, exist_ok=True); os.replace(stage, destination); return {"status": "COMPLETED", "run_dir": str(destination), "models": ORDER, "scenarios": state["scenarios"], "test_accessed": False}
    finally:
        if stage.exists(): shutil.rmtree(stage)
        gc.collect(); torch.cuda.empty_cache()


def analyze(root: Path, config_path: Path, run_name: str) -> dict[str, Any]:
    config = paper_v2.load_config(config_path); run = root / config["output_root"] / "evaluations" / safe_run_name(run_name); required = {"COMPLETED", "efficiency.csv", "manifest.json", "metrics.csv", "metrics.json", "per_sample_metrics.jsonl", "provenance.json", "report.md", "run_state.json"}
    if not run.is_dir() or {path.name for path in run.iterdir()} != required or (run / "COMPLETED").read_text(encoding="utf-8") != MARKER + "\n": raise ValueError("invalid paper evaluation")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if sha256_file(run / name) != expected: raise ValueError(f"evaluation artifact mismatch: {name}")
    state = json.loads((run / "run_state.json").read_text(encoding="utf-8")); return {"status": "COMPLETED", "run_dir": str(run), "models": state["models"], "scenarios": state["scenarios"], "test_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path.cwd()); parser.add_argument("--config", type=Path, default=Path("configs/bota_short_paper_v2.yaml")); parser.add_argument("--mode", choices=["Preflight", "SyntheticDryRun", "Full", "Analyze"], default="Preflight"); parser.add_argument("--benchmark-name", default=""); parser.add_argument("--run-name", default=""); parser.add_argument("--scenario", choices=SCENARIO_CHOICES, default="All")
    for key in ("original", "exact-masked", "full-control-p5", "retain-p5", "bota", "ifru", "neggrad", "pcgrad", "sisa", "receraser"): parser.add_argument(f"--{key}-run-name", default="")
    args = parser.parse_args(); root = args.root.resolve(); cp = (root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve(); names = dict(zip(ORDER, (args.original_run_name, args.exact_masked_run_name, args.full_control_p5_run_name, args.retain_p5_run_name, args.bota_run_name, args.ifru_run_name, args.neggrad_run_name, args.pcgrad_run_name, args.sisa_run_name, args.receraser_run_name)))
    if args.mode == "SyntheticDryRun": result = {"schema": SCHEMA, "models": ORDER, "real_model_loaded": False, "test_accessed": False}
    elif args.mode == "Preflight": result = {"schema": SCHEMA, "models": ORDER, "benchmark_name": args.benchmark_name, "scenario_selection": args.scenario, "two_reference_policy": True, "model_loaded": False, "test_accessed": False}
    elif not args.run_name: parser.error(f"{args.mode} requires RunName")
    elif args.mode == "Analyze": result = analyze(root, cp, args.run_name)
    else: result = execute(root, cp, args.benchmark_name, args.run_name, names, args.scenario)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
