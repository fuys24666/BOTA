"""Request-conditioned E2URec baseline for the frozen BOTA short benchmark.

This adapts E2URec's teacher/student objective to the same fixed-A, trainable-B
LoRA coordinate and the same registered L8/L4M4 request as the 200-step study.
It starts from the already trained Original-Short endpoint.  FinalTest is never
loaded here; model selection and reporting are Development-only.
"""
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
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from src.if_a2_optimization.group_a_gradient_audit import GIB, masked_batch
from src.paper_e2urec_fair_pair_v2.losses import forget_loss_light, joint_losses_light
from src.paper_if_a2.common import atomic_json, canonical_hash, git_snapshot, safe_run_name, seed_everything, sha256_file

from . import evaluation as short_eval
from . import paper_evaluation_v2 as paper_eval
from . import paper_v2
from . import runner
from .protocol import validate_prepared
from .timing import select_scenarios, timing_record

SCHEMA = "bota-short-e2urec-v1"
MARKER = "BOTA_SHORT_E2UREC_V1_COMPLETED"
METHOD_ID = "E2URec-Short-FixedAB"
SEEDS = (41, 42, 43)
SCENARIOS = ("L8", "L4M4")
TEACHER_STEPS = 1200
STUDENT_STEPS = 1000
EFFECTIVE_BATCH = 16
MICROBATCH = 4
SUPPLEMENTAL_MARKER = "BOTA_SHORT_E2UREC_SUPPLEMENTAL_FINALTEST_V1_COMPLETED"


def _paths(root: Path, seed: int) -> tuple[Path, Path, str, str]:
    core = root / f"configs/bota_short_benchmark_seed{seed}_v3.yaml"
    paper = root / f"configs/bota_short_paper_seed{seed}_v3.yaml"
    return core, paper, f"bota_short_i02_seed{seed}_v3", f"bota_short_original_seed{seed}_v3"


def _authority(root: Path, seed: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    core_path, paper_path, benchmark, original_name = _paths(root, seed)
    v1 = runner.load_config(core_path); v1["_config_path"] = str(core_path.resolve())
    protocol, _, registry = validate_prepared(root, v1, benchmark)
    paper = paper_v2.load_config(paper_path)
    if int(v1["protocol"].get("run_seed", -1)) != seed or int(paper["protocol"]["seed"]) != seed:
        raise ValueError("seed/config binding mismatch")
    return v1, paper, registry, original_name


def _freeze(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False); parameter.grad = None
    model.eval()


def _wrapped(indices: Sequence[int], count: int, seed: int, step: int, stream: int) -> list[int]:
    if not indices:
        raise ValueError("empty deterministic batch source")
    generator = torch.Generator().manual_seed(seed + stream + 1_000_003 * (step // max(1, math.ceil(len(indices) / count))))
    order = torch.randperm(len(indices), generator=generator).tolist()
    start = (step * count) % len(indices); result = []
    while len(result) < count:
        take = min(count - len(result), len(indices) - start)
        result.extend(int(indices[order[index]]) for index in range(start, start + take)); start = 0
        if len(result) < count:
            order = torch.randperm(len(indices), generator=generator).tolist()
    return result


def _batch(dataset, indices: Sequence[int], device: torch.device, pad: int):
    return runner.move_batch(masked_batch(dataset, list(indices), pad), device)


def _train_teacher(model, parameters, dataset, forget: list[int], device, pad: int, seed: int) -> tuple[list[dict[str, Any]], float]:
    optimizer = torch.optim.AdamW(parameters, lr=5e-5, betas=(.9, .999), eps=1e-8, weight_decay=.01)
    history = []; started = time.perf_counter(); model.train()
    for step in range(TEACHER_STEPS):
        logical = _wrapped(forget, EFFECTIVE_BATCH, seed, step, 70_000); optimizer.zero_grad(set_to_none=True); total = 0.
        for offset in range(0, EFFECTIVE_BATCH, MICROBATCH):
            value = _batch(dataset, logical[offset:offset + MICROBATCH], device, pad)
            loss = runner.p1._sample_losses(model, value).mean() / (EFFECTIVE_BATCH / MICROBATCH)
            loss.backward(); total += float(loss.detach().cpu()); del value, loss
        optimizer.step()
        if (step + 1) % 50 == 0 or step == 0:
            history.append({"step": step + 1, "loss": total, "logical_batch_hash": canonical_hash(logical)})
    elapsed = time.perf_counter() - started; del optimizer
    return history, elapsed


def _train_student(current, current_parameters, original, augmented, dataset, forget: list[int], retain: list[int], device, pad: int, seed: int) -> tuple[list[dict[str, Any]], float]:
    optimizer = torch.optim.AdamW(current_parameters, lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=.01)
    warmup = math.ceil(len(forget) / EFFECTIVE_BATCH); history = []; started = time.perf_counter(); current.train()
    for step in range(STUDENT_STEPS):
        fidx = _wrapped(forget, EFFECTIVE_BATCH, seed, step, 90_000); ridx = _wrapped(retain, EFFECTIVE_BATCH, seed, step, 110_000)
        optimizer.zero_grad(set_to_none=True); values = {"forget": 0., "retain_sup": 0., "retain_kl": 0., "total": 0.}
        for offset in range(0, EFFECTIVE_BATCH, MICROBATCH):
            fb = _batch(dataset, fidx[offset:offset + MICROBATCH], device, pad)
            if step < warmup:
                loss = forget_loss_light(current, original, augmented, fb, 2.0); values["forget"] += float(loss.detach().cpu()) / 4
            else:
                rb = _batch(dataset, ridx[offset:offset + MICROBATCH], device, pad)
                components = joint_losses_light(current, original, augmented, fb, rb, 2.0)
                loss = .6 * (components["L_sup"] + components["L_retain_KL"]) + .4 * components["L_forget"]
                values["forget"] += float(components["L_forget"].detach().cpu()) / 4
                values["retain_sup"] += float(components["L_sup"].detach().cpu()) / 4
                values["retain_kl"] += float(components["L_retain_KL"].detach().cpu()) / 4
                del rb, components
            (loss / 4).backward(); values["total"] += float(loss.detach().cpu()) / 4; del fb, loss
        optimizer.step()
        if (step + 1) % 25 == 0 or step == 0:
            history.append({"step": step + 1, "phase": "forget_warmup" if step < warmup else "joint", "losses": values, "forget_batch_hash": canonical_hash(fidx), "retain_batch_hash": canonical_hash(ridx)})
    elapsed = time.perf_counter() - started; del optimizer
    return history, elapsed


def _one(root: Path, seed: int, scenario: dict[str, Any], stage: Path, original_name: str, v1: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    seed_everything(seed); device = torch.device("cuda:0"); scenario_id = scenario["id"]; started = time.perf_counter()
    current = original = augmented = None
    try:
        current, names, current_parameters, bases, tokenizer, dataset, _, source_hash = paper_v2._load_original_endpoint(root, v1, registry, original_name, scenario_id, device)
        original, original_names, original_parameters, _, _, _, _, source_hash_2 = paper_v2._load_original_endpoint(root, v1, registry, original_name, scenario_id, device)
        augmented, augmented_names, augmented_parameters, _, _, _, _, source_hash_3 = paper_v2._load_original_endpoint(root, v1, registry, original_name, scenario_id, device)
        if names != original_names or names != augmented_names or len(names) != 72 or len(set((source_hash, source_hash_2, source_hash_3))) != 1:
            raise RuntimeError("Original-Short coordinate binding mismatch")
        _freeze(original)
        forget = list(map(int, scenario["forget_train_indices"])); forget_set = set(forget); retain = [int(index) for index in registry["order"] if int(index) not in forget_set]
        deleted_interactions = int(scenario["deleted_interactions"])
        if len(forget) != deleted_interactions or len(retain) != 3200 - deleted_interactions:
            raise RuntimeError("registered short request cardinality mismatch")
        teacher_history, teacher_seconds = _train_teacher(augmented, augmented_parameters, dataset, forget, device, tokenizer.pad_token_id, seed)
        _freeze(augmented)
        student_history, student_seconds = _train_student(current, current_parameters, original, augmented, dataset, forget, retain, device, tokenizer.pad_token_id, seed)
        values = runner._copy_parameters(names, current_parameters); scenario_dir = stage / "scenarios" / scenario_id; scenario_dir.mkdir(parents=True)
        publication_started = time.perf_counter(); artifact = runner._save_fixed_ab(scenario_dir / "adapter", names, values, bases, METHOD_ID); publication_seconds = time.perf_counter() - publication_started
        atomic_json(scenario_dir / "teacher_history.json", teacher_history); atomic_json(scenario_dir / "student_history.json", student_history)
        phase = timing_record(scenario=scenario_id, initialization_seconds=0., offline_construction_seconds=teacher_seconds, online_compute_seconds=student_seconds, adapter_publication_seconds=publication_seconds, end_to_end_seconds=time.perf_counter() - started, details={"teacher_training_seconds": teacher_seconds, "student_training_seconds": student_seconds})
        atomic_json(scenario_dir / "phase_timing.json", phase)
        warmup_steps = math.ceil(deleted_interactions / EFFECTIVE_BATCH)
        manifest = {"schema": SCHEMA, "method_id": METHOD_ID, "scenario_id": scenario_id, "request_hash": scenario["request_hash"], "model_type": "if_a2_fixed_ab", "artifact": artifact, "source_original_run_manifest_sha256": source_hash, "teacher_optimizer_steps": TEACHER_STEPS, "student_optimizer_steps": STUDENT_STEPS, "forget_warmup_steps": warmup_steps, "joint_steps": STUDENT_STEPS - warmup_steps, "deleted_interactions": deleted_interactions, "coordinate": "fixed_A_trainable_B", "phase_timing": phase, "test_accessed": False}
        atomic_json(scenario_dir / "scenario_manifest.json", manifest); return manifest
    finally:
        del current, original, augmented; gc.collect(); torch.cuda.empty_cache()


def preflight(root: Path, seed: int, scenario_selection: str) -> dict[str, Any]:
    v1, _, registry, original_name = _authority(root, seed); selected = select_scenarios(registry["scenarios"], scenario_selection)
    original_root = root / v1["output_root"] / "models" / runner.METHODS["Original"] / original_name
    if not original_root.is_dir(): raise FileNotFoundError(original_root)
    return {"schema": SCHEMA, "seed": seed, "benchmark_name": f"bota_short_i02_seed{seed}_v3", "scenarios": [row["id"] for row in selected], "source_original_run": str(original_root.resolve()), "teacher_steps_per_scenario": TEACHER_STEPS, "student_steps_per_scenario": STUDENT_STEPS, "model_loaded": False, "test_accessed": False}


def execute(root: Path, seed: int, scenario_selection: str, run_name: str) -> dict[str, Any]:
    authority = preflight(root, seed, scenario_selection); git = git_snapshot(root)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1: raise RuntimeError("exactly one CUDA GPU required")
    v1, _, registry, original_name = _authority(root, seed); selected = select_scenarios(registry["scenarios"], scenario_selection)
    destination = root / f"outputs/bota_short_multiseed_v3/seed{seed}/e2urec/models/{METHOD_ID}" / safe_run_name(run_name)
    if destination.exists(): raise FileExistsError(destination)
    stage = destination.parent / ".work" / f"{safe_run_name(run_name)}.{uuid.uuid4().hex}.stage"; stage.mkdir(parents=True); torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    try:
        manifests = [_one(root, seed, scenario, stage, original_name, v1, registry) for scenario in selected]
        state = {"schema": SCHEMA, "status": "COMPLETED", "method_id": METHOD_ID, "run_name": run_name, "benchmark_name": authority["benchmark_name"], "run_seed": seed, "scenarios": [row["id"] for row in selected], "teacher_optimizer_steps": TEACHER_STEPS * len(selected), "student_optimizer_steps": STUDENT_STEPS * len(selected), "wall_time_seconds": time.perf_counter() - started, "peak_gpu_reserved": torch.cuda.max_memory_reserved(), "test_accessed": False}
        if state["peak_gpu_reserved"] / GIB > 14.: raise RuntimeError("GPU hard cap exceeded")
        atomic_json(stage / "run_state.json", state); atomic_json(stage / "contract.json", {"schema": SCHEMA, "authority": authority, "registry_sha256": registry["registry_sha256"], "git": git, "implementation_sha256": sha256_file(Path(__file__)), "test_accessed": False}); (stage / "COMPLETED").write_text(MARKER + "\n", encoding="utf-8", newline="\n")
        atomic_json(stage / "manifest.json", {"schema": SCHEMA, "run_state_sha256": sha256_file(stage / "run_state.json"), "contract_sha256": sha256_file(stage / "contract.json"), "scenario_manifests": {row["scenario_id"]: sha256_file(stage / "scenarios" / row["scenario_id"] / "scenario_manifest.json") for row in manifests}, "published_atomically": True, "test_accessed": False})
        destination.parent.mkdir(parents=True, exist_ok=True); os.replace(stage, destination); return {"status": "COMPLETED", "run_dir": str(destination), "seed": seed, "scenarios": state["scenarios"], "test_accessed": False}
    finally:
        if stage.exists(): shutil.rmtree(stage)


def _model_run(root: Path, seed: int, run_name: str) -> Path:
    run = root / f"outputs/bota_short_multiseed_v3/seed{seed}/e2urec/models/{METHOD_ID}" / safe_run_name(run_name)
    required = {"COMPLETED", "contract.json", "manifest.json", "run_state.json", "scenarios"}
    if not run.is_dir() or {item.name for item in run.iterdir()} != required or (run / "COMPLETED").read_text(encoding="utf-8") != MARKER + "\n": raise ValueError(f"invalid E2URec run: {run}")
    return run


def evaluate_development(root: Path, run_name: str) -> dict[str, Any]:
    destination = root / "outputs/bota_short_multiseed_v3/e2urec/evaluations" / safe_run_name(run_name)
    if destination.exists(): raise FileExistsError(destination)
    git = git_snapshot(root)
    stage = destination.parent / ".work" / f"{safe_run_name(run_name)}.{uuid.uuid4().hex}.stage"; stage.mkdir(parents=True); rows = []; device = torch.device("cuda:0")
    try:
        for seed in SEEDS:
            v1, _, registry, _ = _authority(root, seed); model_run = _model_run(root, seed, f"bota_short_e2urec_seed{seed}_v1")
            dev_run = root / f"outputs/bota_short_multiseed_v3/seed{seed}/paper/evaluations/bota_short_development_seed{seed}_v3"
            frozen = [json.loads(line) for line in (dev_run / "per_sample_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            base = runner._base_t5_config(root, v1); tokenizer = runner.T5Tokenizer.from_pretrained(base["paths"]["model_dir"]); dataset = runner.JsonPromptDataset(root / v1["source"]["development_json"], tokenizer)
            if "development_user_ids" in v1["source"]: users = json.loads((root / v1["source"]["development_user_ids"]).read_text(encoding="utf-8"))
            else:
                from src.diagnostics.ml1m_development_protocol import reconstruct_authoritative_rows
                _, development, _ = reconstruct_authoritative_rows(root / v1["source"]["raw_data"]); users = [int(row.authoritative_user_id) for row in development]
            for number, scenario in enumerate(registry["scenarios"]):
                sid = scenario["id"]; scenario_run = model_run / "scenarios" / sid; model = short_eval._load_fixed(root, v1, scenario_run, device)
                try: prediction = short_eval._single_predictions(model, dataset, list(range(len(dataset))), device, 4)
                finally: short_eval._release(model)
                selected = [index for index, user in enumerate(users) if int(user) in set(map(int, scenario["users"]))]
                cached = {(row["method_id"], int(row["development_index"])): float(row["probability"]) for row in frozen if row["scenario"] == sid}
                p = np.asarray(prediction["probability"], dtype=np.float64); y = np.asarray(prediction["gold_label"], dtype=np.int64); chosen = np.asarray(selected, dtype=np.int64)
                exact = np.asarray([cached[("Exact-Masked-Reference-200step", index)] for index in selected]); original = np.asarray([cached[("Original-Short-200step", index)] for index in selected]); retain = np.asarray([cached[("Retain-Retrain-P5-Short", index)] for index in selected]); control = np.asarray([cached[("FullControl-P5-Short", index)] for index in selected])
                local_den = float(np.mean(np.abs(original - exact))); p5_den = float(np.mean(np.abs(control - retain))); clipped = p.clip(1e-12, 1 - 1e-12)
                rows.append({"seed": seed, "scenario": sid, "method_id": "E2URec", "overall_auc": float(roc_auc_score(y, p)), "overall_acc": float(accuracy_score(y, p >= .5)), "overall_log_loss": float(np.mean(-(y*np.log(clipped)+(1-y)*np.log(1-clipped)))), "local_exact_masked_residual": float(np.mean(np.abs(p[chosen] - exact)) / local_den), "p5_retrain_residual": float(np.mean(np.abs(p[chosen] - retain)) / p5_den), "selected_samples": len(selected), "test_accessed": False})
        summary = []
        for sid in SCENARIOS:
            group = [row for row in rows if row["scenario"] == sid]
            value = {"scenario": sid, "method_id": "E2URec", "seeds": len(group)}
            for key in ("overall_auc", "overall_acc", "overall_log_loss", "local_exact_masked_residual", "p5_retrain_residual"):
                sample = np.asarray([row[key] for row in group]); value[key + "_mean"] = float(sample.mean()); value[key + "_std"] = float(sample.std(ddof=1))
            summary.append(value)
        for name, values in (("metrics_by_seed.csv", rows), ("metrics_summary.csv", summary)):
            with (stage / name).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
        atomic_json(stage / "metrics.json", {"schema": SCHEMA, "split": "Development", "rows": rows, "summary": summary, "test_accessed": False}); atomic_json(stage / "run_state.json", {"schema": SCHEMA, "status": "COMPLETED", "split": "Development", "seeds": list(SEEDS), "scenarios": list(SCENARIOS), "test_accessed": False}); (stage / "COMPLETED").write_text(MARKER + "\n", encoding="utf-8", newline="\n")
        atomic_json(stage / "manifest.json", {"schema": SCHEMA, "files": {name: sha256_file(stage / name) for name in ("metrics_by_seed.csv", "metrics_summary.csv", "metrics.json", "run_state.json", "COMPLETED")}, "test_accessed": False}); destination.parent.mkdir(parents=True, exist_ok=True); os.replace(stage, destination)
        return {"status": "COMPLETED", "run_dir": str(destination), "summary": summary, "test_accessed": False}
    finally:
        if stage.exists(): shutil.rmtree(stage)
        gc.collect(); torch.cuda.empty_cache()


def _completed_development(root: Path, run_name: str) -> Path:
    run = root / "outputs/bota_short_multiseed_v3/e2urec/evaluations" / safe_run_name(run_name)
    required = {"COMPLETED", "manifest.json", "metrics.json", "metrics_by_seed.csv", "metrics_summary.csv", "run_state.json"}
    if not run.is_dir() or {item.name for item in run.iterdir()} != required or (run / "COMPLETED").read_text(encoding="utf-8") != MARKER + "\n":
        raise ValueError("frozen E2URec Development authority is invalid")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if any(sha256_file(run / name) != expected for name, expected in manifest["files"].items()):
        raise ValueError("E2URec Development artifact hash mismatch")
    return run


def _completed_primary_finaltest(root: Path, run_name: str) -> Path:
    run = root / "outputs/bota_short_multiseed_finaltest_v3/evaluations" / safe_run_name(run_name)
    required = {"COMPLETED", "manifest.json", "metrics_by_seed.csv", "metrics_by_seed.json", "metrics_summary.csv", "per_sample_metrics.jsonl", "provenance.json", "report.md", "run_state.json"}
    if not run.is_dir() or {item.name for item in run.iterdir()} != required:
        raise ValueError("primary frozen FinalTest authority is invalid")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if any(sha256_file(run / name) != expected for name, expected in manifest["files"].items()):
        raise ValueError("primary FinalTest artifact hash mismatch")
    return run


def supplemental_preflight(root: Path, development_run_name: str, primary_finaltest_run_name: str) -> dict[str, Any]:
    development = _completed_development(root, development_run_name); primary = _completed_primary_finaltest(root, primary_finaltest_run_name); models = {}
    for seed in SEEDS:
        run = _model_run(root, seed, f"bota_short_e2urec_seed{seed}_v1")
        models[str(seed)] = {"run": str(run.resolve()), "manifest_sha256": sha256_file(run / "manifest.json"), "scenarios": list(SCENARIOS)}
    authority = {"schema": SCHEMA, "split": "FinalTest", "scope": "E2URec_baseline_only_supplement", "models": models, "development_run": str(development.resolve()), "development_manifest_sha256": sha256_file(development / "manifest.json"), "primary_finaltest_run": str(primary.resolve()), "primary_finaltest_manifest_sha256": sha256_file(primary / "manifest.json"), "reference_methods_reused": ["Original-Short-200step", "Exact-Masked-Reference-200step", "FullControl-P5-Short", "Retain-Retrain-P5-Short"], "e2urec_model_selection_after_finaltest": False, "other_model_training_or_inference": False, "implementation_sha256": sha256_file(Path(__file__))}
    access_binding = canonical_hash({"schema": SCHEMA, "scope": authority["scope"], "models": {seed: value["manifest_sha256"] for seed, value in models.items()}, "development_manifest_sha256": authority["development_manifest_sha256"], "primary_finaltest_manifest_sha256": authority["primary_finaltest_manifest_sha256"]})
    return {"schema": SCHEMA, "status": "ELIGIBLE_FOR_BASELINE_ONLY_SUPPLEMENTAL_FINALTEST", "authority": authority, "authority_sha256": canonical_hash(authority), "access_binding_sha256": access_binding, "final_test_rows_materialized": 0, "e2urec_prediction_calls": 0, "other_method_prediction_calls": 0, "test_accessed": False}


def evaluate_supplemental_finaltest(root: Path, run_name: str, development_run_name: str, primary_finaltest_run_name: str, confirm: bool) -> dict[str, Any]:
    if not confirm:
        raise RuntimeError("supplemental FinalTest requires --confirm-supplemental-final-test")
    pre = supplemental_preflight(root, development_run_name, primary_finaltest_run_name); authority = pre["authority"]; binding = pre["access_binding_sha256"]
    output_root = root / "outputs/bota_short_multiseed_v3/e2urec/supplemental_finaltest"; ledger = output_root / "access_ledger" / binding; destination = output_root / "evaluations" / safe_run_name(run_name)
    if destination.exists(): raise FileExistsError(destination)
    if ledger.exists(): raise RuntimeError("this frozen E2URec supplemental FinalTest has already been reserved")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1: raise RuntimeError("exactly one CUDA GPU required")
    ledger.mkdir(parents=True); atomic_json(ledger / "access_started.json", {"schema": SCHEMA, "status": "E2UREC_BASELINE_ONLY_SUPPLEMENTAL_FINALTEST_STARTED", "binding_sha256": binding, "authority": authority, "test_accessed": True})
    stage = destination.parent / ".work" / f"{safe_run_name(run_name)}.{uuid.uuid4().hex}.stage"; stage.mkdir(parents=True); rows = []; safe_samples = []; prediction_calls = 0; metric_rows = 0; started = time.perf_counter(); device = torch.device("cuda:0")
    try:
        from . import multiseed_finaltest_v3 as primary_module
        final_lineage = primary_module._final_lineage(root / "data/ml-1m/raw_data"); final_path = primary_module._validate_final_file(root, final_lineage); final_users = [int(row.authoritative_user_id) for row in final_lineage]
        if len(final_users) != primary_module.FINAL_ROWS: raise RuntimeError("supplemental FinalTest row count mismatch")
        primary = Path(authority["primary_finaltest_run"]); frozen_rows = [json.loads(line) for line in (primary / "per_sample_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        for seed in SEEDS:
            v1, _, registry, _ = _authority(root, seed); model_run = _model_run(root, seed, f"bota_short_e2urec_seed{seed}_v1"); base = runner._base_t5_config(root, v1); tokenizer = runner.T5Tokenizer.from_pretrained(base["paths"]["model_dir"])
            dataset = primary_module._AuthorizedFinalTestDataset(final_path, tokenizer, primary_module._FINALTEST_DATASET_CAPABILITY)
            for number, scenario in enumerate(registry["scenarios"]):
                sid = scenario["id"]; selected = [index for index, user in enumerate(final_users) if int(user) in set(map(int, scenario["users"]))]; support = Counter(final_users[index] for index in selected)
                if not selected: raise RuntimeError(f"FinalTest has no support for seed{seed}/{sid}")
                cached = {(row["method_id"], int(row["final_test_index"])): float(row["probability"]) for row in frozen_rows if int(row["seed"]) == seed and row["scenario"] == sid}
                required_methods = authority["reference_methods_reused"]
                if any((method, index) not in cached for method in required_methods for index in selected): raise RuntimeError("primary FinalTest reference probabilities are incomplete")
                model = short_eval._load_fixed(root, v1, model_run / "scenarios" / sid, device)
                try: prediction = short_eval._single_predictions(model, dataset, list(range(len(dataset))), device, 4); prediction_calls += 1
                finally: short_eval._release(model)
                p = np.asarray(prediction["probability"], dtype=np.float64); y = np.asarray(prediction["gold_label"], dtype=np.int64); chosen = np.asarray(selected, dtype=np.int64)
                reference_arrays = {}
                for method in required_methods:
                    value = np.full(len(dataset), np.nan, dtype=np.float64)
                    for index in selected: value[index] = cached[(method, index)]
                    reference_arrays[method] = value
                exact = reference_arrays["Exact-Masked-Reference-200step"]; original = reference_arrays["Original-Short-200step"]; retain = reference_arrays["Retain-Retrain-P5-Short"]; control = reference_arrays["FullControl-P5-Short"]
                local = paper_eval._residual(p, exact, original, chosen); local.update(paper_eval._cluster_bootstrap(p, exact, original, selected, final_users, 1000, seed * 100 + number))
                p5 = paper_eval._residual(p, retain, control, chosen); p5.update(paper_eval._cluster_bootstrap(p, retain, control, selected, final_users, 1000, seed * 100 + number + 10_000))
                clipped = p.clip(1e-12, 1-1e-12); row = {"seed": seed, "split": "SupplementalFinalTest", "scenario": sid, "composition": json.dumps(scenario["composition"], sort_keys=True), "method_id": "E2URec", "overall_auc": float(roc_auc_score(y,p)), "overall_acc": float(accuracy_score(y,p>=.5)), "overall_log_loss": float(np.mean(-(y*np.log(clipped)+(1-y)*np.log(1-clipped)))), "local_exact_masked_residual": local["point"], "local_ci_lower": local["ci_lower"], "local_ci_upper": local["ci_upper"], "local_toward": local["toward"], "local_away": local["away"], "p5_retrain_residual": p5["point"], "p5_ci_lower": p5["ci_lower"], "p5_ci_upper": p5["ci_upper"], "p5_toward": p5["toward"], "p5_away": p5["away"], "selected_samples": len(selected), "selected_users_with_support": len(support), "prediction_collapse": bool(np.std(p)<1e-6), "finite": bool(np.isfinite(p).all())}; rows.append(row); metric_rows += 1
                safe_samples.extend({"seed": seed, "scenario": sid, "method_id": "E2URec", "user_hash": canonical_hash([seed, int(final_users[index])]), "final_test_index": index, "probability": float(p[index])} for index in selected)
                del prediction, p, y, reference_arrays; gc.collect(); torch.cuda.empty_cache()
        summary = []
        for sid in SCENARIOS:
            group = [row for row in rows if row["scenario"] == sid]; value = {"scenario": sid, "method_id": "E2URec", "seeds": len(group), "split": "SupplementalFinalTest"}
            for key in ("overall_auc", "overall_acc", "overall_log_loss", "local_exact_masked_residual", "p5_retrain_residual"):
                sample = np.asarray([row[key] for row in group], dtype=np.float64); value[key + "_mean"] = float(sample.mean()); value[key + "_std"] = float(sample.std(ddof=1))
            summary.append(value)
        for name, values in (("metrics_by_seed.csv", rows), ("metrics_summary.csv", summary)):
            with (stage / name).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
        atomic_json(stage / "metrics.json", {"schema": SCHEMA, "split": "SupplementalFinalTest", "rows": rows, "summary": summary, "primary_reference_predictions_reused": True, "other_method_prediction_calls": 0, "test_accessed": True})
        with (stage / "per_sample_metrics.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in safe_samples: handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        atomic_json(stage / "provenance.json", {"schema": SCHEMA, "authority": authority, "binding_sha256": binding, "git": git_snapshot(root), "e2urec_prediction_calls": prediction_calls, "other_method_prediction_calls": 0, "primary_reference_predictions_reused": True, "development_selection_frozen": True, "test_accessed": True})
        atomic_json(stage / "run_state.json", {"schema": SCHEMA, "status": "COMPLETED", "split": "SupplementalFinalTest", "seeds": list(SEEDS), "scenarios": list(SCENARIOS), "e2urec_prediction_calls": prediction_calls, "other_method_prediction_calls": 0, "metric_rows": metric_rows, "wall_time_seconds": time.perf_counter()-started, "test_accessed": True}); (stage / "COMPLETED").write_text(SUPPLEMENTAL_MARKER + "\n", encoding="utf-8", newline="\n")
        files = ("metrics_by_seed.csv", "metrics_summary.csv", "metrics.json", "per_sample_metrics.jsonl", "provenance.json", "run_state.json", "COMPLETED"); atomic_json(stage / "manifest.json", {"schema": SCHEMA, "files": {name: sha256_file(stage/name) for name in files}, "published_atomically": True, "test_accessed": True})
        destination.parent.mkdir(parents=True, exist_ok=True); os.replace(stage, destination); atomic_json(ledger / "access_completed.json", {"schema": SCHEMA, "status": "E2UREC_BASELINE_ONLY_SUPPLEMENTAL_FINALTEST_COMPLETED", "binding_sha256": binding, "run": str(destination.resolve()), "run_manifest_sha256": sha256_file(destination/"manifest.json"), "e2urec_prediction_calls": prediction_calls, "other_method_prediction_calls": 0, "metric_rows": metric_rows, "test_accessed": True}); return {"status": "COMPLETED", "run_dir": str(destination), "summary": summary, "e2urec_prediction_calls": prediction_calls, "other_method_prediction_calls": 0, "test_accessed": True}
    except BaseException as error:
        atomic_json(ledger / "access_failed_no_retry.json", {"schema": SCHEMA, "status": "E2UREC_BASELINE_ONLY_SUPPLEMENTAL_FINALTEST_FAILED_NO_RETRY", "binding_sha256": binding, "reason": type(error).__name__, "message": str(error), "e2urec_prediction_calls": prediction_calls, "other_method_prediction_calls": 0, "metric_rows": metric_rows, "test_accessed": True}); raise
    finally:
        if stage.exists(): shutil.rmtree(stage)
        gc.collect(); torch.cuda.empty_cache()


def analyze_supplemental(root: Path, run_name: str) -> dict[str, Any]:
    run = root / "outputs/bota_short_multiseed_v3/e2urec/supplemental_finaltest/evaluations" / safe_run_name(run_name)
    if not run.is_dir() or (run/"COMPLETED").read_text(encoding="utf-8") != SUPPLEMENTAL_MARKER + "\n": raise ValueError("invalid supplemental FinalTest run")
    manifest = json.loads((run/"manifest.json").read_text(encoding="utf-8"))
    if any(sha256_file(run/name) != expected for name,expected in manifest["files"].items()): raise ValueError("supplemental FinalTest artifact mismatch")
    return {"status": "COMPLETED", "run_dir": str(run), "summary": json.loads((run/"metrics.json").read_text(encoding="utf-8"))["summary"], "test_accessed": True}


def analyze(root: Path, run_name: str, seed: int | None) -> dict[str, Any]:
    if seed is None:
        run = root / "outputs/bota_short_multiseed_v3/e2urec/evaluations" / safe_run_name(run_name)
        return {"status": "COMPLETED", "run_dir": str(run), "summary": json.loads((run / "metrics.json").read_text())["summary"], "test_accessed": False}
    run = _model_run(root, seed, run_name); return json.loads((run / "run_state.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path.cwd()); parser.add_argument("--mode", choices=("Preflight", "Full", "EvaluateDevelopment", "SupplementalFinalTestPreflight", "EvaluateSupplementalFinalTest", "Analyze", "AnalyzeSupplementalFinalTest"), default="Preflight"); parser.add_argument("--seed", type=int, choices=SEEDS); parser.add_argument("--scenario", choices=("All",) + SCENARIOS, default="All"); parser.add_argument("--run-name", default=""); parser.add_argument("--development-run-name", default="bota_short_e2urec_multiseed_development_v1"); parser.add_argument("--primary-finaltest-run-name", default="bota_short_multiseed_finaltest_ml1m_seed41_43_v3_recovery1"); parser.add_argument("--confirm-supplemental-final-test", action="store_true")
    args = parser.parse_args(); root = args.root.resolve()
    if args.mode in {"Preflight", "Full"} and args.seed is None: parser.error("Seed is required")
    if args.mode in {"Full", "EvaluateDevelopment", "EvaluateSupplementalFinalTest", "Analyze", "AnalyzeSupplementalFinalTest"} and not args.run_name: parser.error("RunName is required")
    if args.mode == "Preflight": result = preflight(root, args.seed, args.scenario)
    elif args.mode == "Full": result = execute(root, args.seed, args.scenario, args.run_name)
    elif args.mode == "EvaluateDevelopment": result = evaluate_development(root, args.run_name)
    elif args.mode == "SupplementalFinalTestPreflight": result = supplemental_preflight(root, args.development_run_name, args.primary_finaltest_run_name)
    elif args.mode == "EvaluateSupplementalFinalTest": result = evaluate_supplemental_finaltest(root, args.run_name, args.development_run_name, args.primary_finaltest_run_name, args.confirm_supplemental_final_test)
    elif args.mode == "AnalyzeSupplementalFinalTest": result = analyze_supplemental(root, args.run_name)
    else: result = analyze(root, args.run_name, args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
