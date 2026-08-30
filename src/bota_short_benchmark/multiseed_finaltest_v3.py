"""One-shot, three-seed FinalTest evaluation for the ML-1M short benchmark."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import shutil
import subprocess
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import T5Tokenizer

from src.diagnostics.ml1m_development_protocol import load_raw_metadata, render_record, RawLineage
from src.diagnostics.t5_reconstructed_official import JsonPromptDataset
from src.paper_if_a2.common import atomic_json, canonical_hash, git_snapshot, safe_run_name, sha256_file

from . import paper_evaluation_v2 as dev_eval
from . import paper_v2
from . import runner
from .timing import select_scenarios

SCHEMA = "bota-short-multiseed-finaltest-v3"
MARKER = "BOTA_SHORT_MULTISEED_FINALTEST_V3_COMPLETED"
SEEDS = (41, 42, 43)
SCENARIOS = ("L8", "L4M4")
FINAL_ROWS = 20_000
RUN_NAMES = {
    "Original-Short-200step": "bota_short_original_seed{seed}_v3",
    "Exact-Masked-Reference-200step": "bota_short_exact_masked_seed{seed}_v3",
    "FullControl-P5-Short": "bota_short_full_control_p5_seed{seed}_v3",
    "Retain-Retrain-P5-Short": "bota_short_retain_p5_seed{seed}_v3",
    "BOTA-T2-Short": "bota_short_bota_seed{seed}_v3",
    "IFRU-Short-LoRA": "bota_short_ifru_seed{seed}_v3",
    "NegGrad-Mixed-Short-BOnly": "bota_short_neggrad_seed{seed}_v3",
    "PCGrad-Short-BOnly": "bota_short_pcgrad_seed{seed}_v3",
    "SISA-Short-T5": "bota_short_sisa_seed{seed}_v3",
    "RecEraser-Adapter-Short": "bota_short_receraser_seed{seed}_v3",
}
_FINALTEST_DATASET_CAPABILITY = object()


class _AuthorizedFinalTestDataset(JsonPromptDataset):
    """JsonPromptDataset variant gated by this module's one-shot access capability."""

    def __init__(self, path: Path, tokenizer: T5Tokenizer, capability: object):
        if capability is not _FINALTEST_DATASET_CAPABILITY:
            raise PermissionError("authorized FinalTest dataset capability required")
        self.records = json.loads(path.read_text(encoding="utf-8"))
        self.tokenizer = tokenizer


def _paths(root: Path, seed: int) -> tuple[Path, Path, str]:
    core = root / f"configs/bota_short_benchmark_seed{seed}_v3.yaml"
    paper = root / f"configs/bota_short_paper_seed{seed}_v3.yaml"
    return core, paper, f"bota_short_i02_seed{seed}_v3"


def _final_lineage(raw_dir: Path) -> list[RawLineage]:
    users, movies = load_raw_metadata(raw_dir)
    ratings: list[tuple[int, int, int, int, int]] = []
    for raw_row, line in enumerate((raw_dir / "ratings.dat").read_text(encoding="utf-8").splitlines()):
        user, movie, rating, timestamp = map(int, line.split("::"))
        if user in users and movie in movies:
            ratings.append((timestamp, user, movie, rating, raw_row))
    frame = pd.DataFrame(ratings, columns=["timestamp", "user", "movie", "rating", "raw_row"])
    frame.sort_values(["timestamp", "user", "movie"], kind="stable", inplace=True)
    counts = frame.groupby("user").size(); filtered_count = len(frame) - sum(min(5, int(value)) for value in counts)
    begin = filtered_count - FINAL_ROWS; histories = {user: ([], []) for user in users}; filtered_index = 0; rows: list[RawLineage] = []
    for value in frame.itertuples(index=False):
        user = int(value.user); history_ids, history_ratings = histories[user]
        if len(history_ids) >= 5:
            if filtered_index >= begin:
                rows.append(RawLineage("final_test", len(rows), user, "ratings.dat", int(value.raw_row), int(value.movie), int(value.rating), 1 if int(value.rating) > 3 else 0, int(value.timestamp), history_ids.copy(), history_ratings.copy()))
            filtered_index += 1
        history_ids.append(int(value.movie)); history_ratings.append(int(value.rating))
    if filtered_index != filtered_count or len(rows) != FINAL_ROWS:
        raise RuntimeError("ML-1M FinalTest lineage mismatch")
    return rows


def _validate_final_file(root: Path, rows: list[RawLineage]) -> Path:
    path = root / "data/ml-1m/proc_data/data/test/test_10_simple.json"
    existing = json.loads(path.read_text(encoding="utf-8")); users, movies = load_raw_metadata(root / "data/ml-1m/raw_data")
    replay = [render_record(row, users, movies) for row in rows]
    if existing != replay:
        raise RuntimeError("processed FinalTest does not match authoritative raw replay")
    return path


def _development_run(root: Path, config: dict[str, Any], name: str, expected_runs: dict[str, Path]) -> Path:
    run = root / config["output_root"] / "evaluations" / safe_run_name(name)
    dev_eval.analyze(root, Path(config["_config_path"]), name)
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("split") != "Development" or provenance.get("test_accessed") is not False:
        raise ValueError("invalid frozen Development authority")
    actual = {key: str(value.resolve()) for key, value in expected_runs.items()}
    if provenance.get("model_runs") != actual or provenance.get("scenario_selection") not in {"All", None}:
        raise ValueError("Development/model combination mismatch")
    return run


def preflight(root: Path, development_names: dict[int, str]) -> dict[str, Any]:
    seeds = {}
    for seed in SEEDS:
        _, paper_path, benchmark = _paths(root, seed); config = paper_v2.load_config(paper_path); config["_config_path"] = str(paper_path.resolve())
        names = {key: pattern.format(seed=seed) for key, pattern in RUN_NAMES.items()}; _, registry, runs = dev_eval.resolve(root, config, benchmark, names)
        development = _development_run(root, config, development_names[seed], runs)
        seeds[str(seed)] = {"benchmark_name": benchmark, "registry_sha256": registry["registry_sha256"], "development_run": str(development.resolve()), "model_runs": {key: str(path.resolve()) for key, path in runs.items()}}
    return {"schema": SCHEMA, "seeds": seeds, "scenarios": list(SCENARIOS), "models": list(dev_eval.ORDER), "final_test_rows_materialized": 0, "test_loader_built": False, "test_accessed": False}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _combination_binding(authority: dict[str, Any]) -> str:
    return canonical_hash({"schema": SCHEMA, "seeds": authority["seeds"], "scenarios": list(SCENARIOS)})


def _assert_combination_never_accessed(ledger_root: Path, authority: dict[str, Any], binding: str) -> None:
    if not ledger_root.exists():
        return
    for directory in ledger_root.iterdir():
        started_path = directory / "access_started.json"
        if not directory.is_dir() or not started_path.is_file():
            continue
        started = json.loads(started_path.read_text(encoding="utf-8"))
        legacy = canonical_hash({"schema": SCHEMA, "git_head": started.get("git_head"), "seeds": authority["seeds"], "scenarios": list(SCENARIOS)})
        if directory.name in {binding, legacy}:
            raise RuntimeError("this frozen three-seed model combination has already reserved/accessed FinalTest")


def _historical_finaltest_source(root: Path, git_head: str) -> str:
    value = subprocess.run(["git", "show", f"{git_head}:src/bota_short_benchmark/multiseed_finaltest_v3.py"], cwd=root, capture_output=True, text=True, check=False)
    if value.returncode != 0 or not value.stdout:
        raise RuntimeError("failed FinalTest implementation is unavailable from Git")
    return value.stdout


def _zero_inference_recovery_evidence(root: Path, authority: dict[str, Any], failed_binding: str, failed_run_name: str) -> dict[str, Any]:
    if len(failed_binding) != 64 or any(character not in "0123456789abcdef" for character in failed_binding):
        raise ValueError("FailedBinding must be one lowercase SHA256")
    failed_run_name = safe_run_name(failed_run_name); output_root = root / "outputs/bota_short_multiseed_finaltest_v3"; failed = output_root / "access_ledger" / failed_binding
    if not failed.is_dir() or not (failed / "access_started.json").is_file() or not (failed / "access_failed.json").is_file():
        raise ValueError("failed FinalTest ledger is incomplete")
    if (failed / "access_completed.json").exists() or (output_root / "evaluations" / failed_run_name).exists():
        raise ValueError("failed FinalTest produced a completed evaluation")
    started = json.loads((failed / "access_started.json").read_text(encoding="utf-8")); failure = json.loads((failed / "access_failed.json").read_text(encoding="utf-8")); old_head = started.get("git_head")
    legacy = canonical_hash({"schema": SCHEMA, "git_head": old_head, "seeds": authority["seeds"], "scenarios": list(SCENARIOS)})
    if started.get("binding_sha256") != failed_binding or failure.get("binding_sha256") != failed_binding or legacy != failed_binding:
        raise ValueError("failed ledger does not bind the frozen model combination")
    expected_message = "test paths are forbidden in reconstructed diagnostics"
    if started.get("status") != "FINALTEST_ACCESS_STARTED" or failure.get("status") != "FINALTEST_ACCESS_FAILED_NO_RETRY" or failure.get("reason") != "ValueError" or expected_message not in str(failure.get("message")):
        raise ValueError("failure is not the registered zero-inference loader incident")
    if int(failure.get("prediction_calls", 0)) != 0 or int(failure.get("metrics_rows_written", 0)) != 0:
        raise ValueError("failed access already produced predictions or metrics")
    source = _historical_finaltest_source(root, str(old_head)); body = source.split("def execute", 1)[1].split("def analyze", 1)[0]
    loader_position = body.find("dataset = JsonPromptDataset(final_path, tokenizer)"); prediction_position = body.find("dev_eval._predict(")
    if loader_position < 0 or prediction_position < 0 or loader_position >= prediction_position:
        raise ValueError("historical control flow cannot prove a zero-inference failure")
    recovery = failed / "zero_inference_recovery"
    if recovery.exists():
        raise RuntimeError("zero-inference infrastructure recovery was already reserved")
    return {"schema": "bota-short-zero-inference-recovery-v1", "failed_binding_sha256": failed_binding, "combination_binding_sha256": _combination_binding(authority), "failed_run_name": failed_run_name, "failed_git_head": old_head, "access_started_sha256": sha256_file(failed / "access_started.json"), "access_failed_sha256": sha256_file(failed / "access_failed.json"), "historical_implementation_sha256": canonical_hash(source), "proof_type": "exception_boundary_plus_historical_control_flow", "loader_position_before_prediction": True, "prediction_calls_before_failure": 0, "metrics_rows_before_failure": 0, "model_or_algorithm_change_authorized": False, "single_recovery_authorized": True, "test_accessed": True}


def recovery_preflight(root: Path, development_names: dict[int, str], failed_binding: str, failed_run_name: str) -> dict[str, Any]:
    authority = preflight(root, development_names); git = git_snapshot(root)
    if not git["clean"]:
        raise RuntimeError("recovery preflight requires clean Git")
    evidence = _zero_inference_recovery_evidence(root, authority, failed_binding, failed_run_name)
    return {"schema": evidence["schema"], "status": "ELIGIBLE_FOR_SINGLE_ZERO_INFERENCE_RECOVERY", "authority": authority, "recovery_evidence": evidence, "current_git": git, "final_test_rows_materialized_during_preflight": 0, "test_loader_built": False, "test_accessed_during_preflight": False}


def execute(root: Path, run_name: str, development_names: dict[int, str], confirm: bool, *, recovery_failed_binding: str = "", failed_run_name: str = "") -> dict[str, Any]:
    if not confirm:
        raise RuntimeError("FinalTest requires -ConfirmFinalTest")
    authority = preflight(root, development_names); git = git_snapshot(root)
    if not git["clean"]:
        raise RuntimeError("formal FinalTest requires clean Git")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA GPU required")
    binding = _combination_binding(authority)
    output_root = root / "outputs/bota_short_multiseed_finaltest_v3"; ledger_root = output_root / "access_ledger"; ledger_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / "evaluations" / safe_run_name(run_name)
    if destination.exists(): raise FileExistsError(destination)
    recovery_evidence = None
    if recovery_failed_binding:
        recovery_evidence = _zero_inference_recovery_evidence(root, authority, recovery_failed_binding, failed_run_name)
        reservation = ledger_root / recovery_failed_binding / "zero_inference_recovery"
    else:
        _assert_combination_never_accessed(ledger_root, authority, binding); reservation = ledger_root / binding
    try: reservation.mkdir()
    except FileExistsError as error: raise RuntimeError("this FinalTest access or recovery was already reserved") from error
    started_name = "recovery_started.json" if recovery_evidence else "access_started.json"; completed_name = "recovery_completed.json" if recovery_evidence else "access_completed.json"; failed_name = "recovery_failed_no_retry.json" if recovery_evidence else "access_failed.json"
    atomic_json(reservation / started_name, {"schema": SCHEMA, "binding_sha256": binding, "git_head": git["head"], "status": "ZERO_INFERENCE_RECOVERY_STARTED" if recovery_evidence else "FINALTEST_ACCESS_STARTED", "recovery_evidence": recovery_evidence, "test_accessed": True})
    stage = destination.parent / ".work" / f"{safe_run_name(run_name)}.{uuid.uuid4().hex}.stage"; stage.mkdir(parents=True); started = time.perf_counter(); device = torch.device("cuda:0"); prediction_calls = 0; metric_rows_written = 0; raw_replay_exact = False
    try:
        final_rows = _final_lineage(root / "data/ml-1m/raw_data"); final_path = _validate_final_file(root, final_rows); final_users = [int(row.authoritative_user_id) for row in final_rows]; raw_replay_exact = True; atomic_json(reservation / "raw_replay_validated.json", {"schema": SCHEMA, "binding_sha256": binding, "rows": len(final_rows), "prediction_calls": 0, "metrics_rows_written": 0, "raw_replay_exact": True, "test_accessed": True})
        rows: list[dict[str, Any]] = []; safe_samples: list[dict[str, Any]] = []
        for seed in SEEDS:
            core_path, paper_path, benchmark = _paths(root, seed); del core_path
            config = paper_v2.load_config(paper_path); names = {key: pattern.format(seed=seed) for key, pattern in RUN_NAMES.items()}; v1_config, registry, runs = dev_eval.resolve(root, config, benchmark, names)
            scenarios = select_scenarios(registry["scenarios"], "All"); base = runner._base_t5_config(root, v1_config); tokenizer = T5Tokenizer.from_pretrained(base["paths"]["model_dir"]); dataset = _AuthorizedFinalTestDataset(final_path, tokenizer, _FINALTEST_DATASET_CAPABILITY); shared: dict[str, dict[str, Any]] = {}
            if len(dataset) != FINAL_ROWS: raise RuntimeError("FinalTest sample count mismatch")
            for scenario_number, scenario in enumerate(scenarios):
                selected = [index for index, user in enumerate(final_users) if user in set(map(int, scenario["users"]))]
                support = Counter(final_users[index] for index in selected)
                if not selected: raise RuntimeError(f"FinalTest has no support for seed{seed}/{scenario['id']}")
                predictions: dict[str, dict[str, Any]] = {}
                for display in dev_eval.ORDER:
                    if display in {"Original-Short-200step", "FullControl-P5-Short"} and display in shared: predictions[display] = shared[display]
                    else:
                        predictions[display] = dev_eval._predict(root, config, v1_config, runs[display], display, scenario["id"], dataset, list(range(FINAL_ROWS)), device); prediction_calls += 1
                        if display in {"Original-Short-200step", "FullControl-P5-Short"}: shared[display] = predictions[display]
                reference = predictions["Exact-Masked-Reference-200step"]
                if any(value["gold_label"] != reference["gold_label"] or value["sample_order_sha256"] != reference["sample_order_sha256"] for value in predictions.values()): raise RuntimeError("FinalTest label/order mismatch")
                for display in dev_eval.ORDER:
                    metric = dev_eval._metric_row(display, predictions[display], predictions, selected, final_users, config["evaluation"]["bootstrap_resamples"], seed * 100 + scenario_number)
                    rows.append({"seed": seed, "split": "FinalTest", "scenario": scenario["id"], "composition": json.dumps(scenario["composition"], sort_keys=True), "selected_samples": len(selected), "selected_users_with_support": len(support), **metric}); metric_rows_written += 1
                    probability = predictions[display]["probability"]
                    safe_samples.extend({"seed": seed, "scenario": scenario["id"], "method_id": display, "user_hash": canonical_hash([seed, int(final_users[index])]), "final_test_index": index, "probability": float(probability[index])} for index in selected)
                del predictions; gc.collect(); torch.cuda.empty_cache()
        summary = []
        numeric = ("overall_auc", "overall_acc", "overall_log_loss", "local_exact_masked_residual", "p5_retrain_residual")
        for scenario in SCENARIOS:
            for method in dev_eval.ORDER:
                group = [row for row in rows if row["scenario"] == scenario and row["method_id"] == method]
                value: dict[str, Any] = {"scenario": scenario, "method_id": method, "seeds": len(group)}
                for key in numeric:
                    samples = np.asarray([row[key] for row in group if row[key] is not None], dtype=np.float64); value[f"{key}_mean"] = float(samples.mean()) if len(samples) else None; value[f"{key}_std"] = float(samples.std(ddof=1)) if len(samples) > 1 else 0.0 if len(samples) else None
                summary.append(value)
        atomic_json(stage / "metrics_by_seed.json", {"schema": SCHEMA, "split": "FinalTest", "rows": rows, "test_accessed": True}); _write_csv(stage / "metrics_by_seed.csv", rows); _write_csv(stage / "metrics_summary.csv", summary)
        with (stage / "per_sample_metrics.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in safe_samples: handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        report = ["# BOTA ML-1M three-seed FinalTest v3", "", "One-shot evaluation of the frozen seed41/42/43 Development-selected combinations. No model selection or training was performed on FinalTest.", ""]
        if recovery_evidence:
            report.extend(["## Audited infrastructure recovery", "", "This evaluation is the single authorized recovery from a registered zero-inference loader failure. The original failed ledger is preserved; no model prediction or metric was produced before that failure, and no model, algorithm, request, or Development selection was changed.", ""])
        for scenario in SCENARIOS:
            report.extend([f"## {scenario}", "", "| Method | Local residual mean±sd | AUC mean±sd | LogLoss mean±sd |", "|---|---:|---:|---:|"])
            for row in [value for value in summary if value["scenario"] == scenario]: report.append(f"| {row['method_id']} | {row['local_exact_masked_residual_mean']:.4f}±{row['local_exact_masked_residual_std']:.4f} | {row['overall_auc_mean']:.6f}±{row['overall_auc_std']:.6f} | {row['overall_log_loss_mean']:.6f}±{row['overall_log_loss_std']:.6f} |")
            report.append("")
        (stage / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8", newline="\n")
        provenance = {"schema": SCHEMA, "split": "FinalTest", "binding_sha256": binding, "git": git, "authority": authority, "recovery": recovery_evidence, "final_test": {"path": str(final_path.resolve()), "sha256": sha256_file(final_path), "rows": FINAL_ROWS, "raw_replay_exact": True}, "optimizer_steps": 0, "backward_calls": 0, "model_selection_performed": False, "test_accessed": True}; atomic_json(stage / "provenance.json", provenance)
        state = {"schema": SCHEMA, "status": "COMPLETED", "run_name": run_name, "split": "FinalTest", "seeds": list(SEEDS), "scenarios": list(SCENARIOS), "models": list(dev_eval.ORDER), "recovery_mode": recovery_evidence is not None, "failed_binding_recovered": recovery_evidence["failed_binding_sha256"] if recovery_evidence else None, "wall_time_seconds": time.perf_counter() - started, "test_accessed": True}; atomic_json(stage / "run_state.json", state); (stage / "COMPLETED").write_text(MARKER + "\n", encoding="utf-8", newline="\n")
        files = ("metrics_by_seed.json", "metrics_by_seed.csv", "metrics_summary.csv", "per_sample_metrics.jsonl", "report.md", "provenance.json", "run_state.json", "COMPLETED"); atomic_json(stage / "manifest.json", {"schema": SCHEMA, "files": {name: sha256_file(stage / name) for name in files}, "published_atomically": True, "test_accessed": True})
        destination.parent.mkdir(parents=True, exist_ok=True); os.replace(stage, destination); atomic_json(reservation / completed_name, {"schema": SCHEMA, "binding_sha256": binding, "evaluation": str(destination.resolve()), "manifest_sha256": sha256_file(destination / "manifest.json"), "status": "ZERO_INFERENCE_RECOVERY_COMPLETED" if recovery_evidence else "COMPLETED", "test_accessed": True})
        return {"status": "COMPLETED", "run_dir": str(destination), "split": "FinalTest", "seeds": list(SEEDS), "scenarios": list(SCENARIOS), "models": list(dev_eval.ORDER), "recovery_mode": recovery_evidence is not None, "test_accessed": True}
    except BaseException as error:
        atomic_json(reservation / failed_name, {"schema": SCHEMA, "binding_sha256": binding, "status": "ZERO_INFERENCE_RECOVERY_FAILED_NO_RETRY" if recovery_evidence else "FINALTEST_ACCESS_FAILED_NO_RETRY", "reason": type(error).__name__, "message": str(error), "raw_replay_exact": raw_replay_exact,"prediction_calls": prediction_calls,"metrics_rows_written": metric_rows_written,"test_accessed": True}); raise
    finally:
        if stage.exists(): shutil.rmtree(stage)
        gc.collect(); torch.cuda.empty_cache()


def analyze(root: Path, run_name: str) -> dict[str, Any]:
    run = root / "outputs/bota_short_multiseed_finaltest_v3/evaluations" / safe_run_name(run_name); required = {"COMPLETED", "manifest.json", "metrics_by_seed.csv", "metrics_by_seed.json", "metrics_summary.csv", "per_sample_metrics.jsonl", "provenance.json", "report.md", "run_state.json"}
    if not run.is_dir() or {path.name for path in run.iterdir()} != required or (run / "COMPLETED").read_text(encoding="utf-8") != MARKER + "\n": raise ValueError("invalid multiseed FinalTest run")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if sha256_file(run / name) != expected: raise ValueError(f"FinalTest artifact mismatch: {name}")
    return {"status": "COMPLETED", "run_dir": str(run), "split": "FinalTest", "seeds": list(SEEDS), "scenarios": list(SCENARIOS), "models": list(dev_eval.ORDER), "test_accessed": True}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path.cwd()); parser.add_argument("--mode", choices=["Preflight", "RecoveryPreflight", "Full", "RecoverZeroInference", "Analyze"], default="Preflight"); parser.add_argument("--run-name", default=""); parser.add_argument("--development-run-seed41", default="bota_short_development_seed41_v3"); parser.add_argument("--development-run-seed42", default="bota_short_development_seed42_v3"); parser.add_argument("--development-run-seed43", default="bota_short_development_seed43_v3"); parser.add_argument("--failed-binding", default=""); parser.add_argument("--failed-run-name", default=""); parser.add_argument("--confirm-final-test", action="store_true"); args = parser.parse_args(); root = args.root.resolve(); names = {41: args.development_run_seed41, 42: args.development_run_seed42, 43: args.development_run_seed43}
    if args.mode in {"Full", "RecoverZeroInference", "Analyze"} and not args.run_name: parser.error(f"{args.mode} requires RunName")
    if args.mode in {"RecoveryPreflight", "RecoverZeroInference"} and (not args.failed_binding or not args.failed_run_name): parser.error(f"{args.mode} requires FailedBinding and FailedRunName")
    if args.mode == "Preflight": result = preflight(root, names)
    elif args.mode == "RecoveryPreflight": result = recovery_preflight(root, names, args.failed_binding, args.failed_run_name)
    elif args.mode == "Analyze": result = analyze(root, args.run_name)
    elif args.mode == "RecoverZeroInference": result = execute(root, args.run_name, names, args.confirm_final_test, recovery_failed_binding=args.failed_binding, failed_run_name=args.failed_run_name)
    else: result = execute(root, args.run_name, names, args.confirm_final_test)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
