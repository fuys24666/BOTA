"""Development-only 200-step T0/T1/T2 BOTA transport ablation."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import T5Tokenizer

from src.bota_if import p1_trajectory_transport_audit as p1
from src.bota_if import p2b_full_module_adamw_transport_audit as p2b
from src.bota_if.p1_trajectory_transport_audit import StepBudget
from src.bota_short_benchmark import evaluation as short_eval
from src.bota_short_benchmark import runner
from src.bota_short_benchmark.paper_evaluation_v2 import _cluster_bootstrap, _residual
from src.bota_short_benchmark.protocol import load_config as load_benchmark_config
from src.bota_short_benchmark.protocol import validate_prepared
from src.diagnostics.ml1m_development_protocol import reconstruct_authoritative_rows
from src.diagnostics.t5_reconstructed_official import JsonPromptDataset, move_batch
from src.if_a2_optimization.group_a_gradient_audit import masked_batch
from src.paper_baselines.common import capture_rng, restore_rng, tensor_tree_hash
from src.paper_if_a2.common import atomic_json, canonical_hash, directory_hash, git_snapshot, safe_run_name, sha256_file

SCHEMA = "bota-short-t012-behavior-ablation-v1"
MARKER = "BOTA_SHORT_T012_BEHAVIOR_ABLATION_V1_COMPLETED"
VARIANTS = ("T0_SGD", "T1_AdamW_frozen_v", "T2_AdamW_full_state")
DISPLAY = {
    "T0_SGD": "BOTA-T0-SGD-Transport",
    "T1_AdamW_frozen_v": "BOTA-T1-AdamW-FrozenV",
    "T2_AdamW_full_state": "BOTA-T2-AdamW-FullState",
}


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"schema", "test_access_policy", "output_root", "benchmark_config", "protocol", "evaluation", "runtime"}
    if not isinstance(value, dict) or set(value) != required or value["schema"] != SCHEMA:
        raise ValueError("invalid T0/T1/T2 ablation config")
    if value["test_access_policy"] != "forbidden":
        raise ValueError("test access policy changed")
    protocol = value["protocol"]
    if protocol != {
        "optimizer_steps": 200,
        "batch_size": 16,
        "variants": list(VARIANTS),
        "scenarios": ["L8", "L4M4"],
        "paired_reference_optimizer_steps": 200,
        "physical_optimizer_step_calls": 400,
        "authoritative_optimizer_steps_committed": 0,
    }:
        raise ValueError("frozen T0/T1/T2 protocol changed")
    evaluation = value["evaluation"]
    if evaluation["split"] != "Development" or evaluation["final_test"] is not False:
        raise ValueError("Development-only evaluation changed")
    return value


def _scenario(registry: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    rows = [row for row in registry["scenarios"] if row["id"] == scenario_id]
    if len(rows) != 1 or scenario_id not in {"L8", "L4M4"}:
        raise ValueError("T0/T1/T2 ablation supports only L8 or L4M4")
    return rows[0]


def _validate_reference(root: Path, benchmark_config: dict[str, Any], method_id: str, run_name: str, benchmark_name: str, scenario_id: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    run = short_eval._validate_method(root, benchmark_config, method_id, run_name, benchmark_name)
    scenario_dir = run / "scenarios" / scenario_id
    manifest_path = scenario_dir / "scenario_manifest.json"
    adapter_path = scenario_dir / "adapter"
    if not manifest_path.is_file() or not adapter_path.is_dir():
        raise ValueError(f"{method_id} is missing scenario {scenario_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("method_id") != method_id or manifest.get("scenario_id") != scenario_id or manifest.get("optimizer_steps") != 200 or manifest.get("test_accessed") is not False:
        raise ValueError(f"{method_id} scenario binding mismatch")
    artifact = manifest.get("artifact", {})
    if artifact.get("sha256") != directory_hash(adapter_path) or artifact.get("reload_exact") is not True:
        raise ValueError(f"{method_id} adapter integrity mismatch")
    state = torch.load(adapter_path / "adapter_model.pt", map_location="cpu", weights_only=True)
    if state.get("schema") != "bota-short-fixed-ab-v1" or state.get("rank") != 16 or state.get("alpha") != 32:
        raise ValueError(f"{method_id} adapter schema mismatch")
    if not state.get("A") or set(state.get("B", {})) != {name + ".B" for name in state["A"]}:
        raise ValueError(f"{method_id} LoRA tensor schema mismatch")
    return run, manifest, state


def _same_coordinate(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return list(left["A"]) == list(right["A"]) and all(torch.equal(left["A"][name], right["A"][name]) for name in left["A"])


def _bernoulli_jsd(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    p = np.stack([1-left, left], axis=1).clip(1e-12, 1.0)
    q = np.stack([1-right, right], axis=1).clip(1e-12, 1.0)
    midpoint = (p + q) / 2
    return .5 * np.sum(p * np.log(p / midpoint), axis=1) + .5 * np.sum(q * np.log(q / midpoint), axis=1)


def _fmt(value: Any, digits: int = 6) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def behavior_metrics(method_id: str, prediction: dict[str, Any], exact: dict[str, Any], original: dict[str, Any], selected: Sequence[int], users: Sequence[int], resamples: int, seed: int) -> dict[str, Any]:
    p = np.asarray(prediction["probability"], dtype=np.float64)
    r = np.asarray(exact["probability"], dtype=np.float64)
    o = np.asarray(original["probability"], dtype=np.float64)
    y = np.asarray(prediction["gold_label"], dtype=np.int64)
    chosen = np.asarray(selected, dtype=np.int64)
    if not all(np.isfinite(value).all() for value in (p, r, o)):
        raise ValueError("non-finite prediction")
    local = _residual(p, r, o, chosen)
    local.update(_cluster_bootstrap(p, r, o, selected, users, resamples, seed))
    clipped = p.clip(1e-12, 1 - 1e-12)
    original_p = o.clip(1e-12, 1 - 1e-12)
    log_loss = float(np.mean(-(y*np.log(clipped) + (1-y)*np.log(1-clipped))))
    original_log_loss = float(np.mean(-(y*np.log(original_p) + (1-y)*np.log(1-original_p))))
    return {
        "method_id": method_id,
        "yes_no_jsd_to_exact": float(np.mean(_bernoulli_jsd(p[chosen], r[chosen]))),
        "probability_l2_rms_to_exact": float(np.sqrt(np.mean((p[chosen]-r[chosen])**2))),
        "local_residual": local["point"],
        "ci_lower": local["ci_lower"],
        "ci_upper": local["ci_upper"],
        "toward_exact": local["toward"],
        "away_from_exact": local["away"],
        "equivalent": local["equal"],
        "toward_rate": float(local["toward"] / len(chosen)),
        "overall_auc": float(roc_auc_score(y, p)),
        "overall_acc": float(accuracy_score(y, p >= .5)),
        "overall_log_loss": log_loss,
        "auc_damage_vs_original": float(roc_auc_score(y, o) - roc_auc_score(y, p)),
        "log_loss_damage_vs_original": log_loss - original_log_loss,
        "prediction_collapse": bool(np.std(p) < 1e-6),
        "finite": True,
    }


def _release(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_canonical_formal(model: Any, dataset: Any, indices: Sequence[int], user_ids: Sequence[int], selected_users: Sequence[int], parameters: Sequence[torch.Tensor], names: Sequence[str], device: torch.device, pad: int, config: dict[str, Any], budget: StepBudget):
    """Propagate transports while preserving the formal batch-backward update."""
    optimizer = p1._optimizer(parameters, config)
    states = p2b.new_full_states(parameters, len(selected_users))
    traces = []
    batch_size = config["schedule"]["batch_size"]
    for step, start in enumerate(range(0, len(indices), batch_size), 1):
        chosen = list(indices[start:start + batch_size])
        batch_users = [int(user_ids[index]) for index in chosen]
        batch = move_batch(masked_batch(dataset, chosen, pad), device)
        losses = p1._sample_losses(model, batch)
        per_module_samples = [[] for _ in parameters]
        for sample in range(len(chosen)):
            gradients = torch.autograd.grad(losses[sample], parameters, retain_graph=True)
            for module_index, gradient in enumerate(gradients):
                per_module_samples[module_index].append(gradient.detach())
        optimizer.zero_grad(set_to_none=True)
        formal_loss = torch.sum(losses) / len(chosen)
        formal_loss.backward()
        coefficient_rows = [torch.stack(rows) for rows in per_module_samples]
        sources = []
        for coefficients, parameter in zip(coefficient_rows, parameters):
            rows = []
            for target in selected_users:
                slots = [index for index, user in enumerate(batch_users) if user == target]
                rows.append(-coefficients[slots].sum(0) / len(chosen) if slots else torch.zeros_like(parameter))
            sources.append(torch.stack(rows))
        p2b.advance_full_transports(states, coefficient_rows, sources, parameters, optimizer, step, config)
        budget.step(optimizer, "canonical_reference")
        traces.append({"step": step, "loss": float(formal_loss.detach()), "batch_hash": canonical_hash(chosen), "selected_user_slot_counts": [batch_users.count(user) for user in selected_users]})
        del batch, losses, formal_loss, per_module_samples, coefficient_rows, sources
    cpu_states = {variant: [{key: value.detach().cpu() for key, value in module.items()} for module in modules] for variant, modules in states.items()}
    canonical = {name: parameter.detach().cpu().clone() for name, parameter in zip(names, parameters)}
    return canonical, cpu_states, traces, tensor_tree_hash(optimizer.state_dict()), tensor_tree_hash(capture_rng())


def _predict_reference(root: Path, benchmark_config: dict[str, Any], run: Path, scenario_id: str, dataset: Any, indices: list[int], device: torch.device) -> dict[str, Any]:
    model = short_eval._load_fixed(root, benchmark_config, run / "scenarios" / scenario_id, device)
    try:
        return short_eval._single_predictions(model, dataset, indices, device, benchmark_config["evaluation"]["inference_batch_size"])
    finally:
        _release(model)


def _predict_scenario_dir(root: Path, benchmark_config: dict[str, Any], scenario_dir: Path, dataset: Any, indices: list[int], device: torch.device) -> dict[str, Any]:
    model = short_eval._load_fixed(root, benchmark_config, scenario_dir, device)
    try:
        return short_eval._single_predictions(model, dataset, indices, device, benchmark_config["evaluation"]["inference_batch_size"])
    finally:
        _release(model)


def _verify_manifest(run: Path) -> dict[str, Any]:
    required = {"COMPLETED", "adapters", "contract.json", "manifest.json", "metrics.csv", "metrics.json", "per_sample_metrics.jsonl", "provenance.json", "references", "report.md", "run_state.json", "timing.json"}
    if not run.is_dir() or {path.name for path in run.iterdir()} != required:
        raise ValueError("invalid T0/T1/T2 run layout")
    if (run / "COMPLETED").read_text(encoding="utf-8") != MARKER + "\n":
        raise ValueError("invalid T0/T1/T2 completion marker")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("test_accessed") is not False:
        raise ValueError("invalid T0/T1/T2 manifest")
    for name, expected in manifest.get("files", {}).items():
        if sha256_file(run / name) != expected:
            raise ValueError(f"T0/T1/T2 artifact mismatch: {name}")
    for variant, expected in manifest.get("adapter_sha256", {}).items():
        if variant not in VARIANTS or directory_hash(run / "adapters" / variant) != expected:
            raise ValueError(f"T0/T1/T2 adapter mismatch: {variant}")
    expected_references = {"canonical", "exact_masked"}
    if set(manifest.get("reference_sha256", {})) != expected_references:
        raise ValueError("T0/T1/T2 reference registry mismatch")
    for name, expected in manifest["reference_sha256"].items():
        if directory_hash(run / "references" / name / "adapter") != expected:
            raise ValueError(f"T0/T1/T2 paired reference mismatch: {name}")
    state = json.loads((run / "run_state.json").read_text(encoding="utf-8"))
    if state.get("status") != "COMPLETED" or state.get("physical_optimizer_step_calls") != 400 or state.get("canonical_optimizer_steps") != 200 or state.get("masked_reference_optimizer_steps") != 200 or state.get("authoritative_optimizer_steps_committed") != 0 or state.get("test_accessed") is not False:
        raise ValueError("T0/T1/T2 run-state mismatch")
    contract = json.loads((run / "contract.json").read_text(encoding="utf-8"))
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    if contract.get("test_accessed") is not False or provenance.get("test_accessed") is not False or provenance.get("test_loader_built") is not False:
        raise ValueError("T0/T1/T2 safety provenance mismatch")
    return state


def preflight(root: Path, config_path: Path, benchmark_name: str, scenario_id: str, original_run_name: str, exact_run_name: str) -> dict[str, Any]:
    config = load_config(config_path)
    benchmark_path = root / config["benchmark_config"]
    benchmark_config = load_benchmark_config(benchmark_path)
    _, contract, registry = validate_prepared(root, benchmark_config, benchmark_name)
    scenario = _scenario(registry, scenario_id)
    original_run, original_manifest, original_state = _validate_reference(root, benchmark_config, "Original-Short", original_run_name, benchmark_name, scenario_id)
    exact_run, exact_manifest, exact_state = _validate_reference(root, benchmark_config, "Retrain-Short", exact_run_name, benchmark_name, scenario_id)
    if original_manifest.get("request_hash") != scenario["request_hash"] or exact_manifest.get("request_hash") != scenario["request_hash"]:
        raise ValueError("request binding mismatch")
    if not _same_coordinate(original_state, exact_state):
        raise ValueError("Original/Exact fixed-A coordinates differ")
    return {
        "schema": SCHEMA,
        "benchmark_name": benchmark_name,
        "scenario": scenario_id,
        "request_hash": scenario["request_hash"],
        "registry_sha256": registry["registry_sha256"],
        "prepared_contract_sha256": sha256_file((root / benchmark_config["output_root"] / "protocols" / safe_run_name(benchmark_name) / "contract.json")),
        "benchmark_config_sha256": sha256_file(benchmark_path),
        "ablation_config_sha256": sha256_file(config_path),
        "original": {"run": str(original_run), "scenario_manifest_sha256": sha256_file(original_run / "scenarios" / scenario_id / "scenario_manifest.json"), "adapter_sha256": original_manifest["artifact"]["sha256"]},
        "exact_masked": {"run": str(exact_run), "scenario_manifest_sha256": sha256_file(exact_run / "scenarios" / scenario_id / "scenario_manifest.json"), "adapter_sha256": exact_manifest["artifact"]["sha256"]},
        "trajectory_optimizer_steps": contract["optimizer_steps"],
        "physical_optimizer_step_calls": 400,
        "variants": list(VARIANTS),
        "test_accessed": False,
    }


def synthetic(root: Path, config_path: Path, run_name: str) -> dict[str, Any]:
    config = load_config(config_path)
    destination = root / config["output_root"] / "synthetic" / safe_run_name(run_name)
    if destination.exists():
        raise FileExistsError(destination)
    stage = destination.parent / ".work" / f"{safe_run_name(run_name)}.{uuid.uuid4().hex}.stage"
    stage.mkdir(parents=True)
    atomic_json(stage / "metrics.json", {"schema": SCHEMA, "variants": list(VARIANTS), "physical_optimizer_step_calls": 0, "authoritative_optimizer_steps_committed": 0, "test_accessed": False})
    (stage / "COMPLETED").write_text(MARKER + "\n", encoding="utf-8", newline="\n")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage, destination)
    return {"status": "COMPLETED", "run_dir": str(destination), "synthetic": True, "test_accessed": False}


def execute(root: Path, config_path: Path, benchmark_name: str, scenario_id: str, original_run_name: str, exact_run_name: str, run_name: str) -> dict[str, Any]:
    config = load_config(config_path)
    pre = preflight(root, config_path, benchmark_name, scenario_id, original_run_name, exact_run_name)
    git = git_snapshot(root)
    if not git["clean"]:
        raise RuntimeError("formal T0/T1/T2 ablation requires clean Git")
    if not torch.cuda.is_available() or torch.cuda.device_count() != config["runtime"]["required_cuda_devices"]:
        raise RuntimeError("exactly one CUDA GPU required")
    benchmark_config = load_benchmark_config(root / config["benchmark_config"])
    _, _, registry = validate_prepared(root, benchmark_config, benchmark_name)
    scenario = _scenario(registry, scenario_id)
    original_run = Path(pre["original"]["run"])
    exact_run = Path(pre["exact_masked"]["run"])
    original_state = torch.load(original_run / "scenarios" / scenario_id / "adapter/adapter_model.pt", map_location="cpu", weights_only=True)
    exact_state = torch.load(exact_run / "scenarios" / scenario_id / "adapter/adapter_model.pt", map_location="cpu", weights_only=True)
    destination = root / config["output_root"] / "runs" / safe_run_name(run_name)
    if destination.exists():
        raise FileExistsError(destination)
    stage = destination.parent / ".work" / f"{safe_run_name(run_name)}.{uuid.uuid4().hex}.stage"
    stage.mkdir(parents=True)
    # Some Windows/PyTorch builds reject torch.device("cuda:0") in the
    # reset API even though ordinary tensor placement accepts it. Select the
    # sole formal device explicitly and use the current-device overload.
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    source_sha_before = sha256_file(root / benchmark_config["source"]["original_checkpoint"])
    started = time.perf_counter()
    model = None
    try:
        initialization_started = time.perf_counter()
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        paired_initial_rng = capture_rng()
        model, names, parameters, bases, _, tokenizer, train_dataset, train_users = runner._model_context(root, benchmark_config, registry, device, original_state["A"])
        initialization_seconds = time.perf_counter() - initialization_started
        generated_coordinate = {name: value.detach().cpu() for name, value in bases.items()}
        if list(generated_coordinate) != list(original_state["A"]) or any(not torch.equal(generated_coordinate[name], original_state["A"][name]) for name in generated_coordinate):
            raise RuntimeError("generated fixed-A coordinate differs from formal Original")
        transport_started = time.perf_counter()
        budget = StepBudget(200)
        canonical, states, trace, optimizer_hash, rng_hash = run_canonical_formal(model, train_dataset, registry["order"], train_users, scenario["users"], parameters, names, device, tokenizer.pad_token_id, runner._engine(root, benchmark_config), budget)
        transport_seconds = time.perf_counter() - transport_started
        if budget.calls != 200 or budget.by_arm != {"canonical_reference": 200}:
            raise RuntimeError("T0/T1/T2 physical optimizer-step budget mismatch")
        historical_anchor = {"exact": all(torch.equal(canonical[name], original_state["B"][name]) for name in names)}
        historical_anchor["canonical_sha256"] = tensor_tree_hash(canonical)
        historical_anchor["historical_original_sha256"] = tensor_tree_hash(original_state["B"])
        del model; model = None; gc.collect(); torch.cuda.empty_cache()

        restore_rng(paired_initial_rng)
        masked_started = time.perf_counter()
        model, masked_names, masked_parameters, masked_bases, _, _, _, _ = runner._model_context(root, benchmark_config, registry, device, original_state["A"])
        if list(masked_names) != list(names) or any(not torch.equal(masked_bases[key], bases[key]) for key in bases):
            raise RuntimeError("paired masked coordinate mismatch")
        masked_budget = StepBudget(200)
        exact_values, exact_optimizer, exact_trace = runner._train_named(model, train_dataset, registry["order"], train_users, scenario["users"], masked_names, masked_parameters, device, tokenizer.pad_token_id, runner._engine(root, benchmark_config), masked_budget, "masked_reference")
        masked_seconds = time.perf_counter() - masked_started
        if masked_budget.calls != 200 or masked_budget.by_arm != {"masked_reference": 200}:
            raise RuntimeError("paired masked optimizer-step budget mismatch")
        if [row["batch_hash"] for row in exact_trace] != [row["batch_hash"] for row in trace]:
            raise RuntimeError("paired canonical/masked batch order mismatch")
        exact_state = {**exact_state, "B": exact_values}
        exact_optimizer_hash = tensor_tree_hash(exact_optimizer.state_dict())
        del model, exact_optimizer; model = None; gc.collect(); torch.cuda.empty_cache()

        reference_sha = {
            "canonical": runner._save_fixed_ab(stage / "references" / "canonical" / "adapter", names, canonical, bases, "Paired-Canonical-200step")["sha256"],
            "exact_masked": runner._save_fixed_ab(stage / "references" / "exact_masked" / "adapter", names, exact_values, bases, "Paired-Exact-Masked-200step")["sha256"],
        }
        user_index = {user: slot for slot, user in enumerate(scenario["users"])}
        adapter_sha: dict[str, str] = {}
        online_timing: dict[str, Any] = {}
        parameter_transport: dict[str, Any] = {}
        actual_delta = torch.cat([(exact_state["B"][name].float() - canonical[name].float()).reshape(-1) for name in names])
        for variant in VARIANTS:
            compose_started = time.perf_counter()
            candidate = {name: canonical[name].clone() for name in names}
            slots = [user_index[user] for user in scenario["users"]]
            for module, name in enumerate(names):
                candidate[name].add_(states[variant][module]["theta"][slots].sum(0).float())
            compose_seconds = time.perf_counter() - compose_started
            predicted_delta = torch.cat([(candidate[name] - canonical[name]).reshape(-1) for name in names])
            parameter_transport[variant] = p2b.vector_metrics(actual_delta, predicted_delta)
            publish_started = time.perf_counter()
            artifact = runner._save_fixed_ab(stage / "adapters" / variant, names, candidate, bases, DISPLAY[variant])
            publish_seconds = time.perf_counter() - publish_started
            adapter_sha[variant] = artifact["sha256"]
            online_timing[variant] = {"composition_seconds": compose_seconds, "adapter_publication_seconds": publish_seconds, "online_total_seconds": compose_seconds + publish_seconds}
            del candidate
        del states

        evaluation_started = time.perf_counter()
        base = runner._base_t5_config(root, benchmark_config)
        tokenizer = T5Tokenizer.from_pretrained(base["paths"]["model_dir"])
        development = JsonPromptDataset(root / benchmark_config["source"]["development_json"], tokenizer)
        _, development_rows, _ = reconstruct_authoritative_rows(root / benchmark_config["source"]["raw_data"])
        development_users = [int(row.authoritative_user_id) for row in development_rows]
        indices = list(range(len(development)))
        selected = [index for index, user in enumerate(development_users) if user in set(scenario["users"])]
        if len(development) != 20000 or len(selected) < 80:
            raise RuntimeError("Development support mismatch")
        original_prediction = _predict_scenario_dir(root, benchmark_config, stage / "references" / "canonical", development, indices, device)
        exact_prediction = _predict_scenario_dir(root, benchmark_config, stage / "references" / "exact_masked", development, indices, device)
        if original_prediction["gold_label"] != exact_prediction["gold_label"] or original_prediction["sample_order_sha256"] != exact_prediction["sample_order_sha256"]:
            raise RuntimeError("Original/Exact Development order mismatch")
        metrics = []
        safe_samples = []
        for number, variant in enumerate(VARIANTS):
            scenario_run = stage / "_prediction" / "scenarios" / variant
            scenario_run.mkdir(parents=True)
            shutil.copytree(stage / "adapters" / variant, scenario_run / "adapter")
            prediction = _predict_reference(root, benchmark_config, stage / "_prediction", variant, development, indices, device)
            if prediction["gold_label"] != exact_prediction["gold_label"] or prediction["sample_order_sha256"] != exact_prediction["sample_order_sha256"]:
                raise RuntimeError(f"{variant} Development order mismatch")
            metrics.append(behavior_metrics(DISPLAY[variant], prediction, exact_prediction, original_prediction, selected, development_users, config["evaluation"]["bootstrap_resamples"], config["evaluation"]["bootstrap_seed"] + number))
            p = prediction["probability"]
            safe_samples.extend({"variant": variant, "development_index": index, "user_hash": canonical_hash([42, development_users[index]]), "probability": float(p[index]), "exact_probability": float(exact_prediction["probability"][index]), "original_probability": float(original_prediction["probability"][index])} for index in selected)
        shutil.rmtree(stage / "_prediction")
        evaluation_seconds = time.perf_counter() - evaluation_started
        peak = torch.cuda.max_memory_reserved() / (1024**3)
        if peak > config["runtime"]["hard_peak_reserved_gib"]:
            raise RuntimeError("GPU hard cap exceeded")
        if sha256_file(root / benchmark_config["source"]["original_checkpoint"]) != source_sha_before:
            raise RuntimeError("source checkpoint changed")

        atomic_json(stage / "metrics.json", {"schema": SCHEMA, "benchmark_name": benchmark_name, "scenario": scenario_id, "reference": "Exact-Masked-Reference-200step", "control": "Original-Short-200step", "parameter_transport": parameter_transport, "metrics": metrics, "test_accessed": False})
        with (stage / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
        with (stage / "per_sample_metrics.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in safe_samples:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        timing = {"initialization_seconds": initialization_seconds, "shared_offline_trajectory_seconds": transport_seconds, "paired_masked_reference_seconds": masked_seconds, "per_variant_online": online_timing, "development_evaluation_seconds": evaluation_seconds, "end_to_end_seconds": time.perf_counter()-started}
        atomic_json(stage / "timing.json", timing)
        report = [f"# 200-step T0/T1/T2 behavior ablation — {scenario_id}", "", "Development only. All variants share one canonical 200-step trajectory; Exact-Masked is the local counterfactual reference.", "", "| Variant | Delta cosine | Norm ratio | Relative L2 | Y/N JSD | L2 RMS | Local residual [95% CI] | Toward | AUC | LogLoss |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for variant, row in zip(VARIANTS, metrics):
            parameter = parameter_transport[variant]
            report.append(f"| {DISPLAY[variant]} | {_fmt(parameter['cosine'])} | {_fmt(parameter['norm_ratio'])} | {_fmt(parameter['relative_l2_error'])} | {_fmt(row['yes_no_jsd_to_exact'])} | {_fmt(row['probability_l2_rms_to_exact'])} | {_fmt(row['local_residual'], 4)} [{_fmt(row['ci_lower'], 4)}, {_fmt(row['ci_upper'], 4)}] | {row['toward_exact']}/{len(selected)} | {_fmt(row['overall_auc'])} | {_fmt(row['overall_log_loss'])} |")
        report.extend(["", "No optimizer step is executed online; adapter composition and publication timings are reported separately in `timing.json`.", ""])
        (stage / "report.md").write_text("\n".join(report), encoding="utf-8", newline="\n")
        atomic_json(stage / "contract.json", pre)
        atomic_json(stage / "provenance.json", {"schema": SCHEMA, "git": git, "source_checkpoint_sha256": source_sha_before, "canonical_trace_sha256": canonical_hash(trace), "canonical_optimizer_state_sha256": optimizer_hash, "canonical_rng_sha256": rng_hash, "masked_optimizer_state_sha256": exact_optimizer_hash, "paired_initial_rng_sha256": tensor_tree_hash(paired_initial_rng), "historical_anchor": historical_anchor, "historical_runs_used_for_lineage_only": True, "test_loader_built": False, "test_accessed": False})
        state = {"schema": SCHEMA, "status": "COMPLETED", "benchmark_name": benchmark_name, "scenario": scenario_id, "variants": list(VARIANTS), "physical_optimizer_step_calls": budget.calls + masked_budget.calls, "canonical_optimizer_steps": budget.calls, "masked_reference_optimizer_steps": masked_budget.calls, "physical_optimizer_steps_by_arm": {**budget.by_arm, **masked_budget.by_arm}, "authoritative_optimizer_steps_committed": 0, "candidate_optimizer_steps": 0, "peak_reserved_gib": peak, "test_loader_built": False, "test_accessed": False}
        atomic_json(stage / "run_state.json", state)
        (stage / "COMPLETED").write_text(MARKER + "\n", encoding="utf-8", newline="\n")
        files = {name: sha256_file(stage / name) for name in ("COMPLETED", "contract.json", "metrics.csv", "metrics.json", "per_sample_metrics.jsonl", "provenance.json", "report.md", "run_state.json", "timing.json")}
        atomic_json(stage / "manifest.json", {"schema": SCHEMA, "files": files, "adapter_sha256": adapter_sha, "reference_sha256": reference_sha, "published_atomically": True, "test_accessed": False})
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, destination)
        return {"status": "COMPLETED", "run_dir": str(destination), "scenario": scenario_id, "trajectory_optimizer_steps": 200, "physical_optimizer_step_calls": 400, "authoritative_optimizer_steps_committed": 0, "test_accessed": False}
    except Exception as error:
        if model is not None:
            _release(model)
        shutil.rmtree(stage, ignore_errors=True)
        raise RuntimeError("T0/T1/T2 ablation failed before atomic publication") from error


def analyze(root: Path, config_path: Path, run_name: str) -> dict[str, Any]:
    config = load_config(config_path)
    run = root / config["output_root"] / "runs" / safe_run_name(run_name)
    state = _verify_manifest(run)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    rows = metrics.get("metrics")
    if not isinstance(rows, list) or [row.get("method_id") for row in rows] != [DISPLAY[name] for name in VARIANTS]:
        raise ValueError("T0/T1/T2 metric order mismatch")
    return {"status": "COMPLETED", "run_dir": str(run), "scenario": state["scenario"], "physical_optimizer_step_calls": state["physical_optimizer_step_calls"], "authoritative_optimizer_steps_committed": 0, "metrics": rows, "test_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/bota_short_t012_behavior_ablation_v1.yaml"))
    parser.add_argument("--mode", choices=("Preflight", "SyntheticDryRun", "Full", "Analyze"), default="Preflight")
    parser.add_argument("--benchmark-name", default="")
    parser.add_argument("--scenario", choices=("L8", "L4M4"), default="L8")
    parser.add_argument("--original-run-name", default="")
    parser.add_argument("--exact-masked-run-name", default="")
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()
    root = args.root.resolve(); config_path = (root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    if args.mode == "SyntheticDryRun":
        result = synthetic(root, config_path, args.run_name)
    elif args.mode == "Analyze":
        result = analyze(root, config_path, args.run_name)
    elif args.mode == "Preflight":
        result = preflight(root, config_path, args.benchmark_name, args.scenario, args.original_run_name, args.exact_masked_run_name)
    else:
        result = execute(root, config_path, args.benchmark_name, args.scenario, args.original_run_name, args.exact_masked_run_name, args.run_name)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
