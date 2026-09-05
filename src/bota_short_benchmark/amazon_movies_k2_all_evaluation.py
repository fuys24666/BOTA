"""Nine-method Development comparison for the frozen Amazon new-user K2."""
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

import torch
from transformers import T5Tokenizer

from src.bota_short_benchmark import amazon_movies_e2urec as e2
from src.bota_short_benchmark import evaluation as core_eval
from src.bota_short_benchmark import paper_evaluation_v2, paper_v2, runner
from src.bota_short_benchmark.protocol import load_config, validate_prepared
from src.diagnostics.t5_reconstructed_official import JsonPromptDataset
from src.paper_if_a2.common import atomic_json, safe_run_name, sha256_file

SCHEMA = "bota-amazon-newuser-k2-all-evaluation-v4"
MARKER = "BOTA_AMAZON_NEWUSER_K2_ALL_EVALUATION_V4_COMPLETED"
ORDER = ["Original-Short", "Retrain-Short", "BOTA-T2-Short", "IFRU-Short-LoRA", "E2URec-Short-FixedAB", "NegGrad-Mixed-Short-BOnly", "PCGrad-Short-BOnly", "SISA-Short-T5", "RecEraser-Adapter-Short"]
CORE = set(ORDER) & {"Original-Short", "Retrain-Short", "BOTA-T2-Short", "IFRU-Short-LoRA", "SISA-Short-T5", "RecEraser-Adapter-Short"}
PAPER = {"NegGrad-Mixed-Short-BOnly", "PCGrad-Short-BOnly"}


def _runs(root: Path, config: dict, paper_config: dict, benchmark: str, names: dict[str, str]):
    if set(names) != set(ORDER) or any(not name for name in names.values()): raise ValueError("all nine Amazon K2 methods are required")
    result = {}
    for method in CORE: result[method] = core_eval._validate_method(root, config, method, names[method], benchmark)
    for method in PAPER: result[method] = paper_evaluation_v2._new_run(root, paper_config, method, names[method], benchmark)
    result[e2.METHOD_ID] = root / "outputs/amz_v4/e2urec_k2/models" / e2.METHOD_ID / safe_run_name(names[e2.METHOD_ID]); e2.analyze(root, names[e2.METHOD_ID])
    return result


def execute(root: Path, config_path: Path, paper_path: Path, benchmark: str, run_name: str, names: dict[str, str]):
    config = load_config(config_path); paper_config = paper_v2.load_config(paper_path); _, _, registry = validate_prepared(root, config, benchmark); runs = _runs(root, config, paper_config, benchmark, names)
    scenario = next(row for row in registry["scenarios"] if row["id"] == "K2"); destination = root / "outputs/amz_v4/k2_all_evaluation" / safe_run_name(run_name)
    if destination.exists(): raise FileExistsError(destination)
    stage = destination.parent / ".work" / f"{safe_run_name(run_name)}.{uuid.uuid4().hex}.stage"; stage.mkdir(parents=True); device = torch.device("cuda:0")
    tokenizer = T5Tokenizer.from_pretrained(runner._base_t5_config(root, config)["paths"]["model_dir"]); dataset = JsonPromptDataset(root / config["source"]["development_json"], tokenizer); dev_users = json.loads((root / config["source"]["development_user_ids"]).read_text(encoding="utf-8")); indices = list(range(len(dataset))); requested = set(map(int, scenario["users"])); selected = [index for index, user in enumerate(dev_users) if int(user) in requested]
    if len(selected) != 10: raise RuntimeError("Amazon K2 evaluation requires exactly ten Development samples")
    predictions = {}; rows = []; samples = []; started = time.perf_counter()
    try:
        for method in ORDER:
            if method in CORE: predictions[method] = core_eval._predict(root, config, runs[method], method, "K2", dataset, indices, device)
            else:
                predictions[method] = core_eval._predict(root, config, runs[method], method, "K2", dataset, indices, device)
        reference = predictions["Retrain-Short"]; original = predictions["Original-Short"]
        for method in ORDER:
            row, per_sample = core_eval._metrics(method, predictions[method], reference, original, selected); rows.append({"scenario": "K2", "composition": scenario["composition"], **row}); samples.extend({"scenario": "K2", **value} for value in per_sample)
        atomic_json(stage / "metrics.json", {"schema": SCHEMA, "benchmark_name": benchmark, "metrics": rows, "test_accessed": False})
        with (stage / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = sorted({key for row in rows for key in row}); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        with (stage / "per_sample_metrics.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in samples: handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        report = ["# Amazon Movies and TV new-user K2 baseline comparison", "", "Development-only dataset-extension experiment; no FinalTest or MIA was accessed.", "", "| Method | AUC | ACC | LogLoss | MAE to Exact | Residual | Toward/Away |", "|---|---:|---:|---:|---:|---:|---:|"]
        for row in rows: report.append(f"| {row['method_id']} | {row['overall_auc']:.6f} | {row['overall_acc']:.6f} | {row['overall_log_loss']:.6f} | {row['probability_mae_to_retrain']:.6f} | {row['residual_ratio_to_retrain']:.6f} | {row['toward_retrain']}/{row['away_from_retrain']} |")
        (stage / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8", newline="\n"); state = {"schema": SCHEMA, "status": "COMPLETED", "benchmark_name": benchmark, "models": ORDER, "scenarios": ["K2"], "wall_time_seconds": time.perf_counter() - started, "test_accessed": False}; atomic_json(stage / "run_state.json", state); atomic_json(stage / "provenance.json", {"schema": SCHEMA, "registry_sha256": registry["registry_sha256"], "model_runs": {method: str(path.resolve()) for method, path in runs.items()}, "split": "Development", "test_accessed": False}); (stage / "COMPLETED").write_text(MARKER + "\n", encoding="utf-8", newline="\n")
        files = ("metrics.json", "metrics.csv", "per_sample_metrics.jsonl", "report.md", "run_state.json", "provenance.json", "COMPLETED"); atomic_json(stage / "manifest.json", {"schema": SCHEMA, "files": {name: sha256_file(stage / name) for name in files}, "published_atomically": True, "test_accessed": False}); destination.parent.mkdir(parents=True, exist_ok=True); os.replace(stage, destination)
        return {"status": "COMPLETED", "run_dir": str(destination), "models": ORDER, "scenario": "K2", "test_accessed": False}
    finally:
        if stage.exists(): shutil.rmtree(stage)
        gc.collect(); torch.cuda.empty_cache()


def analyze(root: Path, run_name: str):
    run = root / "outputs/amz_v4/k2_all_evaluation" / safe_run_name(run_name); required = {"COMPLETED", "manifest.json", "metrics.csv", "metrics.json", "per_sample_metrics.jsonl", "provenance.json", "report.md", "run_state.json"}
    if not run.is_dir() or {path.name for path in run.iterdir()} != required or (run / "COMPLETED").read_text(encoding="utf-8") != MARKER + "\n": raise ValueError("invalid Amazon all-baseline evaluation")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        if sha256_file(run / name) != digest: raise ValueError(f"Amazon evaluation artifact mismatch: {name}")
    state = json.loads((run / "run_state.json").read_text(encoding="utf-8")); return {"status": "COMPLETED", "run_dir": str(run), "models": state["models"], "test_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path.cwd()); parser.add_argument("--config", type=Path, default=Path("configs/bota_short_amazon_movies_newuser_k2k4_v4.yaml")); parser.add_argument("--paper-config", type=Path, default=Path("configs/bota_short_paper_amazon_newuser_k2_v4.yaml")); parser.add_argument("--mode", choices=["Preflight", "SyntheticDryRun", "Full", "Analyze"], default="Preflight"); parser.add_argument("--benchmark-name", default="amz_new_k2k4_s42_v4"); parser.add_argument("--run-name", default="amz_new_k2_all_eval_s42_v4")
    for key in ("original", "exact", "bota", "ifru", "e2urec", "neggrad", "pcgrad", "sisa", "receraser"): parser.add_argument(f"--{key}-run-name", default="")
    args = parser.parse_args(); root = args.root.resolve(); config = (root / args.config).resolve(); paper = (root / args.paper_config).resolve(); names = dict(zip(ORDER, (args.original_run_name, args.exact_run_name, args.bota_run_name, args.ifru_run_name, args.e2urec_run_name, args.neggrad_run_name, args.pcgrad_run_name, args.sisa_run_name, args.receraser_run_name)))
    if args.mode == "SyntheticDryRun": result = {"schema": SCHEMA, "models": ORDER, "real_model_loaded": False, "test_accessed": False}
    elif args.mode == "Preflight": result = {"schema": SCHEMA, "benchmark_name": args.benchmark_name, "models": ORDER, "split": "Development", "model_loaded": False, "test_accessed": False}
    elif args.mode == "Analyze": result = analyze(root, args.run_name)
    else: result = execute(root, config, paper, args.benchmark_name, args.run_name, names)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()

