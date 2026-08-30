"""Repeated zero-step BOTA online latency benchmark.

The historical formal BOTA run did not persist its per-user trajectory bank.
``BuildBank`` therefore reconstructs that bank once and accepts it only when
both registered scenario compositions reproduce the historical adapters with
exact tensor equality. ``Full`` is deliberately separate: it reads the frozen
bank and performs no training, backward pass, HVP, CG, or bank construction.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import shutil
import statistics
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import psutil
import torch
import yaml

from src.bota_if import p1_trajectory_transport_audit as p1
from src.bota_if import p2b_full_module_adamw_transport_audit as p2b
from src.bota_if.p1_trajectory_transport_audit import StepBudget
from src.paper_if_a2.artifacts import atomic_torch_save
from src.paper_if_a2.common import atomic_json, canonical_hash, directory_hash, git_snapshot, safe_run_name, seed_everything, sha256_file

from . import runner
from .protocol import validate_prepared

SCHEMA = "bota-short-online-latency-v1"
BANK_MARKER = "BOTA_SHORT_ONLINE_LATENCY_BANK_V1_COMPLETED"
RUN_MARKER = "BOTA_SHORT_ONLINE_LATENCY_V1_COMPLETED"
BANK_REQUIRED = {"COMPLETED", "bank.pt", "contract.json", "manifest.json", "provenance.json", "summary.json"}
RUN_REQUIRED = {"COMPLETED", "contract.json", "latency_samples.csv", "manifest.json", "provenance.json", "report.md", "run_state.json", "summary.csv", "summary.json"}


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"schema", "test_access_policy", "output_root", "source", "protocol", "privacy", "runtime"} or value["schema"] != SCHEMA:
        raise ValueError("invalid BOTA online-latency config")
    if value["test_access_policy"] != "forbidden":
        raise ValueError("online latency benchmark must forbid FinalTest")
    expected_protocol = {
        "seed": 42,
        "scenarios": ["L8", "L4M4"],
        "trajectory_optimizer_steps": 200,
        "transport": "T2_AdamW_full_state",
        "warmup_iterations": 20,
        "measurement_iterations": 1000,
        "percentiles": [50, 95],
        "percentile_method": "linear",
        "bank_dtype": "float32",
        "publication_semantics": "fsync_files_then_atomic_directory_rename",
        "reload_validation": "exact_tensor_equality_every_iteration",
        "scenario_order": "alternating_first_scenario",
    }
    if value["protocol"] != expected_protocol:
        raise ValueError("online latency protocol changed")
    if value["privacy"] != {"persist_raw_user_ids": False, "persist_prompt_or_token_data": False, "persist_latency_scalars": True}:
        raise ValueError("online latency privacy policy changed")
    return value


def _benchmark_authority(root: Path, config: dict[str, Any]) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    benchmark_path = root / config["source"]["benchmark_config"]
    benchmark = runner.load_config(benchmark_path)
    protocol, _, registry = validate_prepared(root, benchmark, config["source"]["benchmark_name"])
    source_run = root / benchmark["output_root"] / "models" / runner.METHODS["BOTA"] / config["source"]["bota_run_name"]
    required = {"COMPLETED", "contract.json", "manifest.json", "run_state.json", "scenarios"}
    if not source_run.is_dir() or {path.name for path in source_run.iterdir()} != required or (source_run / "COMPLETED").read_text(encoding="utf-8") != runner.MARKER + "\n":
        raise ValueError("invalid frozen BOTA source run")
    state = json.loads((source_run / "run_state.json").read_text(encoding="utf-8"))
    manifest = json.loads((source_run / "manifest.json").read_text(encoding="utf-8"))
    if state.get("status") != "COMPLETED" or state.get("method_id") != "BOTA-T2-Short" or state.get("test_accessed") is not False:
        raise ValueError("BOTA source state mismatch")
    if state.get("registry_sha256") != registry.get("registry_sha256") or state.get("scenarios") != config["protocol"]["scenarios"]:
        raise ValueError("BOTA source registry/scenario mismatch")
    if manifest.get("run_state_sha256") != sha256_file(source_run / "run_state.json") or manifest.get("contract_sha256") != sha256_file(source_run / "contract.json"):
        raise ValueError("BOTA source manifest mismatch")
    for scenario in config["protocol"]["scenarios"]:
        scenario_dir = source_run / "scenarios" / scenario
        if manifest.get("scenario_manifests", {}).get(scenario) != sha256_file(scenario_dir / "scenario_manifest.json"):
            raise ValueError("BOTA source scenario manifest mismatch")
        scenario_manifest = json.loads((scenario_dir / "scenario_manifest.json").read_text(encoding="utf-8"))
        if scenario_manifest.get("request_hash") != next(row["request_hash"] for row in registry["scenarios"] if row["id"] == scenario):
            raise ValueError("BOTA source request mismatch")
        if directory_hash(scenario_dir / "adapter") != scenario_manifest["artifact"]["sha256"]:
            raise ValueError("BOTA source adapter SHA mismatch")
    return benchmark_path, benchmark, source_run, registry


def _bank_dir(root: Path, config: dict[str, Any], bank_name: str) -> Path:
    return root / config["output_root"] / "banks" / safe_run_name(bank_name)


def _run_dir(root: Path, config: dict[str, Any], run_name: str) -> Path:
    return root / config["output_root"] / "runs" / safe_run_name(run_name)


def _verify_manifest(directory: Path, marker: str, required: set[str]) -> dict[str, Any]:
    if not directory.is_dir() or {path.name for path in directory.iterdir()} != required or (directory / "COMPLETED").read_text(encoding="utf-8") != marker + "\n":
        raise ValueError("invalid online-latency artifact")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest.get("files", {}).items():
        path = directory / name
        actual = directory_hash(path) if path.is_dir() else sha256_file(path)
        if actual != expected:
            raise ValueError(f"online-latency artifact SHA mismatch: {name}")
    return manifest


def _compose(bank: dict[str, Any], request_keys: Sequence[str]) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
    index = {key: slot for slot, key in enumerate(bank["user_keys"])}
    try:
        slots = torch.tensor([index[key] for key in request_keys], dtype=torch.long)
    except KeyError as error:
        raise ValueError("request user absent from frozen bank") from error
    selected = [value.index_select(0, slots) for value in bank["vectors"]]
    candidate = {name: bank["canonical"][name] + values.sum(0) for name, values in zip(bank["names"], selected)}
    return selected, candidate


def _materialize(bank: dict[str, Any], candidate: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "A": {name: value.clone() for name, value in bank["bases"].items()},
        "B": {name: candidate[name].clone() for name in bank["names"]},
        "schema": "bota-short-fixed-ab-v1",
        "rank": 16,
        "alpha": 32,
        "method_id": "BOTA-T2-Short",
    }


def _assert_adapter_exact(left: dict[str, Any], right: dict[str, Any], names: Sequence[str]) -> None:
    if left.get("schema") != right.get("schema") or left.get("rank") != right.get("rank") or left.get("alpha") != right.get("alpha") or left.get("method_id") != right.get("method_id"):
        raise RuntimeError("adapter metadata mismatch")
    if set(left["B"]) != set(names) or set(right["B"]) != set(names):
        raise RuntimeError("adapter B tensor names mismatch")
    if set(left["A"]) != set(right["A"]):
        raise RuntimeError("adapter A tensor names mismatch")
    if any(not torch.equal(left["B"][name], right["B"][name]) for name in names):
        raise RuntimeError("adapter B tensor mismatch")
    if any(not torch.equal(left["A"][name], right["A"][name]) for name in left["A"]):
        raise RuntimeError("adapter A tensor mismatch")


def _publish_reload(work: Path, iteration_key: str, state: dict[str, Any], names: Sequence[str]) -> tuple[float, float]:
    stage = work / f"{iteration_key}.stage"
    target = work / f"{iteration_key}.published"
    if stage.exists() or target.exists():
        raise FileExistsError("iteration publication path already exists")
    publication_started = time.perf_counter_ns()
    stage.mkdir(parents=True)
    atomic_torch_save(stage / "adapter_model.pt", state)
    atomic_json(stage / "adapter_config.json", {"format": "paper-fixed-A-B-LoRA-v1", "target_modules": ["q", "v"], "rank": 16, "alpha": 32, "method_id": "BOTA-T2-Short"})
    os.replace(stage, target)
    publication_ms = (time.perf_counter_ns() - publication_started) / 1e6
    reload_started = time.perf_counter_ns()
    loaded = torch.load(target / "adapter_model.pt", map_location="cpu", weights_only=True)
    _assert_adapter_exact(state, loaded, names)
    reload_ms = (time.perf_counter_ns() - reload_started) / 1e6
    shutil.rmtree(target)
    return publication_ms, reload_ms


def _source_adapters(source_run: Path, scenarios: Sequence[str]) -> dict[str, dict[str, Any]]:
    return {scenario: torch.load(source_run / "scenarios" / scenario / "adapter" / "adapter_model.pt", map_location="cpu", weights_only=True) for scenario in scenarios}


def build_bank(root: Path, config_path: Path, bank_name: str) -> dict[str, Any]:
    config = load_config(config_path)
    benchmark_path, benchmark, source_run, registry = _benchmark_authority(root, config)
    git = git_snapshot(root)
    if not git["clean"]:
        raise RuntimeError("formal bank recovery requires clean Git")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA GPU required for bank recovery")
    destination = _bank_dir(root, config, bank_name)
    if destination.exists():
        raise FileExistsError(destination)
    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    if free / (1024 ** 3) < config["runtime"]["minimum_free_gib_for_bank_build"]:
        raise RuntimeError("insufficient dedicated GPU memory for bank recovery")
    fraction = runner.allocator_fraction_for(total / (1024 ** 3), config["runtime"]["allocator_fraction"], config["runtime"]["hard_peak_reserved_gib"])
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    scenarios = [next(row for row in registry["scenarios"] if row["id"] == scenario) for scenario in config["protocol"]["scenarios"]]
    selected_users = sorted({int(user) for scenario in scenarios for user in scenario["users"]})
    user_keys = [canonical_hash([config["protocol"]["seed"], user]) for user in selected_users]
    user_index = {user: slot for slot, user in enumerate(selected_users)}
    request_keys = {scenario["id"]: [user_keys[user_index[int(user)]] for user in scenario["users"]] for scenario in scenarios}
    source_before = sha256_file(source_run / "manifest.json")
    stage = destination.parent / ".work" / f"{safe_run_name(bank_name)}.{uuid.uuid4().hex}.stage"
    stage.mkdir(parents=True)
    model = optimizer = None
    started = time.perf_counter()
    try:
        seed_everything(config["protocol"]["seed"])
        model, names, parameters, bases, _, tokenizer, dataset, users = runner._model_context(root, benchmark, registry, device)
        budget = StepBudget(config["protocol"]["trajectory_optimizer_steps"])
        canonical, states, trace, optimizer_hash, rng_hash = p2b.run_canonical_full(model, dataset, registry["order"], users, selected_users, parameters, names, device, tokenizer.pad_token_id, runner._engine(root, benchmark), budget)
        vectors = [module["theta"].float().contiguous() for module in states[config["protocol"]["transport"]]]
        bank = {
            "schema": SCHEMA,
            "names": list(names),
            "canonical": {name: canonical[name].float().contiguous() for name in names},
            # Fixed-A module keys intentionally omit the trainable ``.B``
            # suffix. Preserve the authority's independent A-key namespace.
            # The formal fixed-A authority is persisted in float64. It is a
            # coordinate definition, not part of the float32 user-vector
            # bank, so changing its dtype breaks exact adapter recovery.
            "bases": {name: value.detach().cpu().contiguous() for name, value in bases.items()},
            "user_keys": user_keys,
            "vectors": vectors,
            "scenario_request_keys": request_keys,
            "scenario_request_hashes": {scenario["id"]: scenario["request_hash"] for scenario in scenarios},
            "transport": config["protocol"]["transport"],
            "trajectory_optimizer_steps": budget.calls,
            "test_accessed": False,
        }
        authorities = _source_adapters(source_run, config["protocol"]["scenarios"])
        for scenario in config["protocol"]["scenarios"]:
            _, candidate = _compose(bank, request_keys[scenario])
            _assert_adapter_exact(_materialize(bank, candidate), authorities[scenario], names)
        atomic_torch_save(stage / "bank.pt", bank)
        elapsed = time.perf_counter() - started
        bank_bytes = (stage / "bank.pt").stat().st_size
        vector_bytes = sum(value.numel() * value.element_size() for value in vectors)
        summary = {
            "schema": SCHEMA,
            "status": "COMPLETED",
            "bank_name": safe_run_name(bank_name),
            "registered_users": len(user_keys),
            "module_count": len(names),
            "bank_file_bytes": bank_bytes,
            "user_vector_bytes_total": vector_bytes,
            "user_vector_bytes_per_registered_user": vector_bytes // len(user_keys),
            "full_1025_user_vector_bytes_estimate": (vector_bytes // len(user_keys)) * 1025,
            "build_seconds": elapsed,
            "source_adapters_reproduced_exactly": True,
            "optimizer_steps": budget.calls,
            "test_accessed": False,
        }
        atomic_json(stage / "summary.json", summary)
        atomic_json(stage / "contract.json", {"schema": SCHEMA, "bank_name": safe_run_name(bank_name), "benchmark_config": str(benchmark_path.relative_to(root)), "benchmark_config_sha256": sha256_file(benchmark_path), "benchmark_name": config["source"]["benchmark_name"], "source_run": str(source_run.relative_to(root)), "source_run_manifest_sha256": source_before, "registry_sha256": registry["registry_sha256"], "scenario_request_hashes": bank["scenario_request_hashes"], "git": git, "test_accessed": False})
        atomic_json(stage / "provenance.json", {"schema": SCHEMA, "purpose": "one_time_recovery_of_historical_unpersisted_registered_request_bank", "historical_source_unchanged": sha256_file(source_run / "manifest.json") == source_before, "canonical_trace_sha256": canonical_hash(trace), "optimizer_state_sha256": optimizer_hash, "rng_sha256": rng_hash, "raw_user_ids_persisted": False, "prompt_or_token_data_persisted": False, "source_adapters_reproduced_exactly": True, "test_accessed": False})
        (stage / "COMPLETED").write_text(BANK_MARKER + "\n", encoding="utf-8", newline="\n")
        files = ("bank.pt", "contract.json", "provenance.json", "summary.json", "COMPLETED")
        atomic_json(stage / "manifest.json", {"schema": SCHEMA, "files": {name: sha256_file(stage / name) for name in files}, "published_atomically": True, "test_accessed": False})
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, destination)
        return {"status": "COMPLETED", "bank_dir": str(destination), **summary}
    finally:
        if model is not None:
            del model
        if optimizer is not None:
            del optimizer
        if stage.exists():
            shutil.rmtree(stage)
        gc.collect()
        torch.cuda.empty_cache()


def _quantile(values: Sequence[float], percentile: int) -> float:
    if not values or percentile not in {50, 95}:
        raise ValueError("invalid latency quantile request")
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile, method="linear"))


def _summarize(rows: list[dict[str, Any]], scenarios: Sequence[str]) -> list[dict[str, Any]]:
    result = []
    phases = ("lookup_ms", "composition_ms", "materialization_ms", "publication_ms", "reload_ms", "online_total_ms")
    for scenario in scenarios:
        chosen = [row for row in rows if row["scenario"] == scenario]
        for phase in phases:
            values = [float(row[phase]) for row in chosen]
            result.append({"scenario": scenario, "phase": phase.removesuffix("_ms"), "unit": "ms", "iterations": len(values), "p50": _quantile(values, 50), "p95": _quantile(values, 95), "mean": statistics.fmean(values), "standard_deviation": statistics.stdev(values), "minimum": min(values), "maximum": max(values)})
    return result


def execute_latency(root: Path, config_path: Path, bank_name: str, run_name: str) -> dict[str, Any]:
    config = load_config(config_path)
    _, _, source_run, registry = _benchmark_authority(root, config)
    git = git_snapshot(root)
    if not git["clean"]:
        raise RuntimeError("formal online latency benchmark requires clean Git")
    bank_dir = _bank_dir(root, config, bank_name)
    _verify_manifest(bank_dir, BANK_MARKER, BANK_REQUIRED)
    bank_sha_before = sha256_file(bank_dir / "bank.pt")
    bank = torch.load(bank_dir / "bank.pt", map_location="cpu", weights_only=True)
    if bank.get("schema") != SCHEMA or bank.get("test_accessed") is not False or bank.get("trajectory_optimizer_steps") != 200:
        raise ValueError("frozen bank binding mismatch")
    if list(bank.get("scenario_request_keys", {})) != config["protocol"]["scenarios"]:
        raise ValueError("frozen bank scenario order mismatch")
    source_authorities = _source_adapters(source_run, config["protocol"]["scenarios"])
    for scenario in config["protocol"]["scenarios"]:
        _, candidate = _compose(bank, bank["scenario_request_keys"][scenario])
        _assert_adapter_exact(_materialize(bank, candidate), source_authorities[scenario], bank["names"])
    del source_authorities, candidate
    bank_index = {key: slot for slot, key in enumerate(bank["user_keys"])}
    destination = _run_dir(root, config, run_name)
    if destination.exists():
        raise FileExistsError(destination)
    work = destination.parent / ".work" / f"{safe_run_name(run_name)}.{uuid.uuid4().hex}"
    publication_work = work / "publication"
    result_stage = work / "result"
    publication_work.mkdir(parents=True)
    result_stage.mkdir()
    warmups = config["protocol"]["warmup_iterations"]
    iterations = config["protocol"]["measurement_iterations"]
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    gc_enabled = gc.isenabled()
    try:
        # Warm-up touches every tensor and exercises the same fsync/publish/reload path.
        for number in range(warmups):
            for scenario in config["protocol"]["scenarios"]:
                _, candidate = _compose(bank, bank["scenario_request_keys"][scenario])
                state = _materialize(bank, candidate)
                _publish_reload(publication_work, f"warmup-{number}-{scenario}", state, bank["names"])
        gc.collect()
        gc.disable()
        for number in range(iterations):
            order = list(config["protocol"]["scenarios"])
            if number % 2:
                order.reverse()
            for scenario in order:
                total_started = time.perf_counter_ns()
                phase = time.perf_counter_ns()
                slots = torch.tensor([bank_index[key] for key in bank["scenario_request_keys"][scenario]], dtype=torch.long)
                selected = [value.index_select(0, slots) for value in bank["vectors"]]
                lookup_ms = (time.perf_counter_ns() - phase) / 1e6
                phase = time.perf_counter_ns()
                candidate = {name: bank["canonical"][name] + values.sum(0) for name, values in zip(bank["names"], selected)}
                composition_ms = (time.perf_counter_ns() - phase) / 1e6
                phase = time.perf_counter_ns()
                state = _materialize(bank, candidate)
                materialization_ms = (time.perf_counter_ns() - phase) / 1e6
                publication_ms, reload_ms = _publish_reload(publication_work, f"measure-{number}-{scenario}", state, bank["names"])
                total_ms = (time.perf_counter_ns() - total_started) / 1e6
                rows.append({"iteration": number, "scenario": scenario, "lookup_ms": lookup_ms, "composition_ms": composition_ms, "materialization_ms": materialization_ms, "publication_ms": publication_ms, "reload_ms": reload_ms, "online_total_ms": total_ms, "accounted_phase_ms": lookup_ms + composition_ms + materialization_ms + publication_ms + reload_ms, "reload_exact": True})
        if gc_enabled:
            gc.enable()
        summary_rows = _summarize(rows, config["protocol"]["scenarios"])
        with (result_stage / "latency_samples.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        with (result_stage / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0])); writer.writeheader(); writer.writerows(summary_rows)
        summary = {"schema": SCHEMA, "status": "COMPLETED", "run_name": safe_run_name(run_name), "bank_name": safe_run_name(bank_name), "warmup_iterations_per_scenario": warmups, "measurement_iterations_per_scenario": iterations, "total_measured_requests": len(rows), "percentile_method": "linear", "summaries": summary_rows, "optimizer_steps": 0, "backward_calls": 0, "hvp_calls": 0, "cg_calls": 0, "bank_rebuilt_during_latency_run": False, "reload_exact_all_iterations": all(row["reload_exact"] for row in rows), "wall_time_seconds": time.perf_counter() - started, "cpu_rss_bytes": psutil.Process().memory_info().rss, "test_accessed": False}
        atomic_json(result_stage / "summary.json", summary)
        report = ["# BOTA repeated online latency v1", "", f"Frozen registered-request bank: `{safe_run_name(bank_name)}`. Warm-up: {warmups} per scenario; measured repetitions: {iterations} per scenario.", "", "| Scenario | Phase | P50 (ms) | P95 (ms) | Mean (ms) |", "|---|---|---:|---:|---:|"]
        for row in summary_rows:
            report.append(f"| {row['scenario']} | {row['phase']} | {row['p50']:.3f} | {row['p95']:.3f} | {row['mean']:.3f} |")
        report.extend(["", "Online total includes lookup, vector composition, Adapter materialization, fsync-backed file publication, atomic directory rename, and exact reload validation. It excludes the one-time trajectory-bank construction. No optimizer, backward pass, HVP, CG, Development, or FinalTest is used.", ""])
        (result_stage / "report.md").write_text("\n".join(report), encoding="utf-8", newline="\n")
        atomic_json(result_stage / "contract.json", {"schema": SCHEMA, "run_name": safe_run_name(run_name), "bank_name": safe_run_name(bank_name), "bank_manifest_sha256": sha256_file(bank_dir / "manifest.json"), "bank_file_sha256": bank_sha_before, "source_run_manifest_sha256": sha256_file(source_run / "manifest.json"), "registry_sha256": registry["registry_sha256"], "protocol": config["protocol"], "git": git, "test_accessed": False})
        atomic_json(result_stage / "provenance.json", {"schema": SCHEMA, "bank_read_only": True, "bank_rebuilt_during_latency_run": False, "source_adapters_validated_before_timing": True, "source_adapters_reproduced_exactly": True, "raw_user_ids_persisted": False, "prompt_or_token_data_read": False, "development_accessed": False, "final_test_accessed": False, "optimizer_steps": 0, "backward_calls": 0, "hvp_calls": 0, "cg_calls": 0, "publication_semantics": config["protocol"]["publication_semantics"], "reload_validation": config["protocol"]["reload_validation"], "test_accessed": False})
        atomic_json(result_stage / "run_state.json", summary)
        (result_stage / "COMPLETED").write_text(RUN_MARKER + "\n", encoding="utf-8", newline="\n")
        files = ("contract.json", "latency_samples.csv", "provenance.json", "report.md", "run_state.json", "summary.csv", "summary.json", "COMPLETED")
        atomic_json(result_stage / "manifest.json", {"schema": SCHEMA, "files": {name: sha256_file(result_stage / name) for name in files}, "published_atomically": True, "test_accessed": False})
        if sha256_file(bank_dir / "bank.pt") != bank_sha_before:
            raise RuntimeError("frozen bank changed during latency run")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(result_stage, destination)
        return {"status": "COMPLETED", "run_dir": str(destination), "iterations_per_scenario": iterations, "summaries": summary_rows, "test_accessed": False}
    finally:
        if gc_enabled and not gc.isenabled():
            gc.enable()
        if work.exists():
            shutil.rmtree(work)


def analyze_bank(root: Path, config: dict[str, Any], bank_name: str) -> dict[str, Any]:
    directory = _bank_dir(root, config, bank_name)
    _verify_manifest(directory, BANK_MARKER, BANK_REQUIRED)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    return {"status": "COMPLETED", "bank_dir": str(directory), **summary}


def analyze_run(root: Path, config: dict[str, Any], run_name: str) -> dict[str, Any]:
    directory = _run_dir(root, config, run_name)
    _verify_manifest(directory, RUN_MARKER, RUN_REQUIRED)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    return {"status": "COMPLETED", "run_dir": str(directory), **summary}


def preflight(root: Path, config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    benchmark_path, _, source_run, registry = _benchmark_authority(root, config)
    return {"schema": SCHEMA, "status": "READY", "benchmark_config": str(benchmark_path), "source_run": str(source_run), "registry_sha256": registry["registry_sha256"], "scenarios": config["protocol"]["scenarios"], "warmups": config["protocol"]["warmup_iterations"], "iterations": config["protocol"]["measurement_iterations"], "model_loaded": False, "bank_loaded": False, "development_accessed": False, "test_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/bota_short_online_latency_v1.yaml"))
    parser.add_argument("--mode", choices=["Preflight", "SyntheticDryRun", "BuildBank", "Full", "AnalyzeBank", "Analyze"], default="Preflight")
    parser.add_argument("--bank-name", default="")
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()
    root = args.root.resolve(); config_path = (root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve(); config = load_config(config_path)
    if args.mode == "Preflight":
        result = preflight(root, config_path)
    elif args.mode == "SyntheticDryRun":
        result = {"schema": SCHEMA, "status": "COMPLETED", "warmups": config["protocol"]["warmup_iterations"], "iterations": config["protocol"]["measurement_iterations"], "real_model_loaded": False, "real_bank_loaded": False, "optimizer_steps": 0, "backward_calls": 0, "test_accessed": False}
    elif args.mode in {"BuildBank", "AnalyzeBank"} and not args.bank_name:
        parser.error(f"{args.mode} requires BankName")
    elif args.mode == "BuildBank":
        result = build_bank(root, config_path, args.bank_name)
    elif args.mode == "AnalyzeBank":
        result = analyze_bank(root, config, args.bank_name)
    elif not args.bank_name or not args.run_name:
        parser.error(f"{args.mode} requires BankName and RunName")
    elif args.mode == "Full":
        result = execute_latency(root, config_path, args.bank_name, args.run_name)
    else:
        result = analyze_run(root, config, args.run_name)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
