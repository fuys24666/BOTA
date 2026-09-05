"""E2URec baseline on the frozen Amazon new-user K2 request."""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import torch

from src.bota_short_benchmark import e2urec_short_v1 as core
from src.bota_short_benchmark import runner
from src.bota_short_benchmark.protocol import load_config, validate_prepared
from src.paper_if_a2.common import atomic_json, git_snapshot, safe_run_name, sha256_file

SCHEMA = "bota-amazon-newuser-e2urec-v4"
MARKER = "BOTA_AMAZON_NEWUSER_E2UREC_V4_COMPLETED"
METHOD_ID = core.METHOD_ID


def _authority(root: Path, config_path: Path, benchmark_name: str, original_run_name: str):
    config = load_config(config_path); _, _, registry = validate_prepared(root, config, benchmark_name)
    selected = [row for row in registry["scenarios"] if row["id"] == "K2"]
    if len(selected) != 1 or selected[0]["deleted_interactions"] != 4 or selected[0]["requested_users"] != 2:
        raise ValueError("Amazon E2URec requires the frozen four-interaction K2 request")
    original = root / config["output_root"] / "models" / runner.METHODS["Original"] / safe_run_name(original_run_name)
    runner.analyze(root, config_path, original_run_name, "Original")
    return config, registry, selected[0], original


def preflight(root: Path, config_path: Path, benchmark_name: str, original_run_name: str):
    _, _, scenario, original = _authority(root, config_path, benchmark_name, original_run_name)
    return {"schema": SCHEMA, "benchmark_name": benchmark_name, "scenario": scenario["id"], "deleted_interactions": 4, "source_original_run": str(original.resolve()), "teacher_steps": core.TEACHER_STEPS, "student_steps": core.STUDENT_STEPS, "model_loaded": False, "test_accessed": False}


def execute(root: Path, config_path: Path, benchmark_name: str, original_run_name: str, run_name: str):
    authority = preflight(root, config_path, benchmark_name, original_run_name)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1: raise RuntimeError("exactly one CUDA GPU required")
    config, registry, scenario, _ = _authority(root, config_path, benchmark_name, original_run_name)
    destination = root / "outputs/amz_v4/e2urec_k2/models" / METHOD_ID / safe_run_name(run_name)
    if destination.exists(): raise FileExistsError(destination)
    stage = destination.parent / ".work" / f"{safe_run_name(run_name)}.{uuid.uuid4().hex}.stage"; stage.mkdir(parents=True)
    torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    try:
        manifest = core._one(root, 42, scenario, stage, original_run_name, config, registry)
        state = {"schema": SCHEMA, "status": "COMPLETED", "method_id": METHOD_ID, "run_name": run_name, "benchmark_name": benchmark_name, "run_seed": 42, "scenarios": ["K2"], "teacher_optimizer_steps": core.TEACHER_STEPS, "student_optimizer_steps": core.STUDENT_STEPS, "wall_time_seconds": time.perf_counter() - started, "peak_gpu_reserved": torch.cuda.max_memory_reserved(), "test_accessed": False}
        atomic_json(stage / "run_state.json", state); atomic_json(stage / "contract.json", {"schema": SCHEMA, "authority": authority, "registry_sha256": registry["registry_sha256"], "git": git_snapshot(root), "implementation_sha256": sha256_file(Path(__file__)), "test_accessed": False}); (stage / "COMPLETED").write_text(MARKER + "\n", encoding="utf-8", newline="\n")
        atomic_json(stage / "manifest.json", {"schema": SCHEMA, "run_state_sha256": sha256_file(stage / "run_state.json"), "contract_sha256": sha256_file(stage / "contract.json"), "scenario_manifests": {"K2": sha256_file(stage / "scenarios/K2/scenario_manifest.json")}, "published_atomically": True, "test_accessed": False})
        destination.parent.mkdir(parents=True, exist_ok=True); os.replace(stage, destination)
        return {"status": "COMPLETED", "run_dir": str(destination), "scenarios": ["K2"], "test_accessed": False}
    finally:
        if stage.exists(): shutil.rmtree(stage)
        gc.collect(); torch.cuda.empty_cache()


def analyze(root: Path, run_name: str):
    run = root / "outputs/amz_v4/e2urec_k2/models" / METHOD_ID / safe_run_name(run_name); required = {"COMPLETED", "contract.json", "manifest.json", "run_state.json", "scenarios"}
    if not run.is_dir() or {path.name for path in run.iterdir()} != required or (run / "COMPLETED").read_text(encoding="utf-8") != MARKER + "\n": raise ValueError("invalid Amazon E2URec run")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8")); state = json.loads((run / "run_state.json").read_text(encoding="utf-8"))
    if sha256_file(run / "run_state.json") != manifest["run_state_sha256"] or state["test_accessed"] is not False: raise ValueError("Amazon E2URec integrity mismatch")
    return {"status": "COMPLETED", "run_dir": str(run), "scenarios": state["scenarios"], "test_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path.cwd()); parser.add_argument("--config", type=Path, default=Path("configs/bota_short_amazon_movies_newuser_k2k4_v4.yaml")); parser.add_argument("--mode", choices=["Preflight", "SyntheticDryRun", "Full", "Analyze"], default="Preflight"); parser.add_argument("--benchmark-name", default="amz_new_k2k4_s42_v4"); parser.add_argument("--original-run-name", default="amz_new_orig_s42_v4"); parser.add_argument("--run-name", default="amz_new_e2urec_s42_v4")
    args = parser.parse_args(); root = args.root.resolve(); config = (root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    if args.mode == "SyntheticDryRun": result = {"schema": SCHEMA, "method_id": METHOD_ID, "real_model_loaded": False, "test_accessed": False}
    elif args.mode == "Preflight": result = preflight(root, config, args.benchmark_name, args.original_run_name)
    elif args.mode == "Analyze": result = analyze(root, args.run_name)
    else: result = execute(root, config, args.benchmark_name, args.original_run_name, args.run_name)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()

