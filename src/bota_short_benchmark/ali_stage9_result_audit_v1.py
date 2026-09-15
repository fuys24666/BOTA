"""Freeze Stage 9 Development results and the selected endpoint inventory."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from src.paper_if_a2.common import atomic_json, atomic_text, canonical_hash, directory_hash, sha256_file
from . import ali_stage9_crossseed_development_v1 as stage9
from . import runner


SCHEMA = "bota-ali-stage9-result-audit-v1"
REFERENCE_IDS = {
    "Original": "Original-Short-200step",
    "Exact-Masked": "Exact-Masked-Reference-200step",
    "FullControl-P5": "FullControl-P5-Short",
    "Retain-Retrain-P5": "Retain-Retrain-P5-Short",
}
ENDPOINT_METHODS = tuple(REFERENCE_IDS) + stage9.METHODS


def stability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {(seed, scenario, method) for seed in stage9.REPORTED_SEEDS for scenario in stage9.SCENARIOS for method in stage9.METHODS}
    actual = [(row["seed"], row["scenario"], row["method"]) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Stage 9 matrix is incomplete or contains duplicates")
    held = [row for row in rows if row["seed"] in stage9.HELD_OUT_SEEDS]
    bota = [row for row in held if row["method"] == "BOTA"]
    bota_mean = float(np.mean([row["E_U"] for row in bota]))
    return {
        "bota_utility_every_cell": all(row["utility_eligible"] and not row["prediction_collapse"] for row in bota),
        "bota_residual_below_one_every_cell": all(row["local_residual"] is not None and row["local_residual"] < 1 for row in bota),
        "bota_mean_eu_below_each_baseline": all(bota_mean < float(np.mean([row["E_U"] for row in held if row["method"] == method])) for method in stage9.METHODS[1:]),
        "held_out_bota_mean_E_U": bota_mean,
    }


def _relative(path: Path, workspace: Path) -> str:
    return str(path.resolve().relative_to(workspace.resolve())).replace("\\", "/")


def endpoint_inventory(repo: Path, workspace: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    endpoints = []; dependencies: dict[str, Any] = {}; cached: dict[Path, str] = {}

    def tree(path: Path) -> str:
        resolved = path.resolve()
        if not resolved.is_dir(): raise FileNotFoundError(resolved)
        if resolved not in cached: cached[resolved] = directory_hash(resolved)
        return cached[resolved]

    for seed in stage9.REPORTED_SEEDS:
        seed_config = stage9._seed_config(config, seed)
        benchmark, registry = stage9.stage7._authority(repo, workspace, seed_config)
        references = stage9.stage7._reference_runs(workspace, seed_config)
        if seed == 41:
            base = runner._base_t5_config(workspace, benchmark)
            checkpoint = workspace / benchmark["source"]["original_checkpoint"]
            model_dir = Path(base["paths"]["model_dir"])
            if not model_dir.is_absolute(): model_dir = workspace / model_dir
            dependencies = {"original_checkpoint": {"path": _relative(checkpoint, workspace), "sha256": sha256_file(checkpoint)}, "pretrained_model_directory": {"path": _relative(model_dir, workspace), "sha256": tree(model_dir)}}
        for scenario in registry["scenarios"]:
            sid = scenario["id"]
            for method in ENDPOINT_METHODS:
                kind = "fixed_ab"
                if method in REFERENCE_IDS:
                    path = references[REFERENCE_IDS[method]] / "scenarios" / sid / "adapter"
                elif method == "BOTA":
                    if seed == 42:
                        path = workspace / "outputs/ali_stage8_bota_response_sweep_v1/runs/ali_stage8_bota_response_seed42_v1/adapters/bota_r1" / sid / "adapter"
                    else:
                        path = references["BOTA-T2-Short"] / "scenarios" / sid / "adapter"
                else:
                    if seed == 42:
                        run = workspace / "outputs/ali_stage7_baseline_sweep_v1/runs" / f"ali_stage7a_{method.lower()}_seed42_v1"
                    else:
                        run = stage9._run_dir(workspace, config, seed, method)
                    state = run / "candidate_states" / stage9.SELECTED_IDS[method] / sid
                    if method == "RecEraser":
                        pointer = stage9._json(state / "component.json")
                        path = Path(pointer["component_run"]); kind = "receraser_ensemble"
                        if sha256_file(path / "paper_model_manifest.json") != pointer["paper_model_manifest_sha256"]:
                            raise ValueError("selected RecEraser manifest changed")
                    else:
                        path = state if method == "E2URec" else state / "adapter"
                if kind == "fixed_ab" and not (path / "adapter_model.pt").is_file(): raise FileNotFoundError(path)
                endpoints.append({"seed": seed, "scenario": sid, "method": method, "kind": kind, "path": _relative(path, workspace), "directory_sha256": tree(path), "users": list(map(int, scenario["users"])), "registry_sha256": registry["registry_sha256"]})
    return endpoints, dependencies


def build(repo: Path, workspace: Path) -> tuple[dict[str, Any], str]:
    config_path = repo / "configs/ali_stage9_crossseed_development_v1.yaml"
    config = stage9.load_config(repo, workspace, config_path)
    summary = workspace / config["output_root"] / "summary"
    if not summary.is_dir(): raise ValueError("Stage 9 summary must exist before freezing")
    stage9.summarize(repo, workspace, config_path)
    source_rows = stage9._selection_seed_rows(repo, workspace, config); sources = {}
    for seed in stage9.HELD_OUT_SEEDS:
        for method in stage9.METHODS:
            stage9.analyze(workspace, config, seed, method)
            run = stage9._run_dir(workspace, config, seed, method)
            contract = stage9._json(run / "contract.json")
            if contract["config_sha256"] != sha256_file(config_path) or contract["amendment_sha256"] != config["parent_amendment"]["sha256"]:
                raise ValueError("Stage 9 run configuration binding changed")
            stage9._assert_no_test_access(contract)
            metrics = stage9._json(run / "metrics.json")
            source_rows.extend(metrics["rows"])
            sources[f"{seed}/{method}"] = {"path": _relative(run, workspace), "manifest_sha256": sha256_file(run / "manifest.json"), "metrics_sha256": sha256_file(run / "metrics.json"), "run_code_commit": contract["git"]["head"]}
    metrics = stage9._json(summary / "metrics.json"); decision = stage9._json(summary / "decision.json")
    stage9._assert_no_test_access(metrics); stage9._assert_no_test_access(decision)
    if canonical_hash(metrics["rows"]) != canonical_hash(source_rows): raise ValueError("Stage 9 summary differs from its frozen source rows")
    recomputed = stability(source_rows)
    for key, value in recomputed.items():
        if decision[key] != value: raise ValueError(f"Stage 9 decision mismatch: {key}")
    stable = all(recomputed[key] for key in ("bota_utility_every_cell", "bota_residual_below_one_every_cell", "bota_mean_eu_below_each_baseline"))
    if stable != decision["finaltest_authorized"] or decision["status"] != ("STABLE_CONFIRMED" if stable else "NOT_STABLE_NO_FINALTEST_AUTHORIZATION"):
        raise ValueError("Stage 9 status mismatch")
    endpoints, dependencies = endpoint_inventory(repo, workspace, config)
    audit = {
        "schema": SCHEMA, "status": "FROZEN_COMPLETE_STAGE9", "frozen_date": "2026-09-15",
        "analysis_split": "Development/Dev-Select", "test_accessed": False, "finaltest_accessed": False,
        "configuration": {"path": "configs/ali_stage9_crossseed_development_v1.yaml", "sha256": sha256_file(config_path), "amendment_sha256": config["parent_amendment"]["sha256"], "stage7_audit_sha256": config["stage7_audit"]["sha256"], "stage8_audit_sha256": config["stage8_audit"]["sha256"]},
        "completeness": {"reported_seeds": list(stage9.REPORTED_SEEDS), "decision_seeds": list(stage9.HELD_OUT_SEEDS), "scenarios": list(stage9.SCENARIOS), "methods": list(stage9.METHODS), "candidate_cells": len(source_rows), "all_source_manifests_validated": True},
        "decision": decision, "aggregates": metrics["aggregates"], "rows": source_rows,
        "source_runs": sources, "summary": {"path": _relative(summary, workspace), "files": {name: sha256_file(summary / name) for name in sorted(stage9.SUMMARY_REQUIRED)}},
        "endpoint_inventory": endpoints, "shared_model_dependencies": dependencies,
        "sisa_status": "NO_ELIGIBLE_CANDIDATE",
        "interpretation": ["Seed 42 selected configurations and is excluded from the confirmation decision.", "The selected endpoint tensors are frozen before a new FinalTest preregistration.", "Stage 9 confirms only the 200-step local matched-counterfactual scope, not long-horizon, attack or sequential validity.", "FinalTest authorization requires a separately committed preregistration; no test data were accessed by this audit."],
    }
    lines = ["# Ali Stage 9 frozen result audit", "", f"Decision: `{decision['status']}`. All 36 Development cells and 60 reference/selected endpoint cells are frozen.", "", "FinalTest was not accessed. Seed 42 is excluded from the confirmation decision.", "", "| Method | Three-seed mean E_U | Mean residual | Eligible cells |", "|---|---:|---:|---:|"]
    for method in stage9.METHODS:
        row = metrics["aggregates"][method]; lines.append(f"| {method} | {row['three_seed_six_cell_mean_E_U']:.9f} | {row['three_seed_six_cell_mean_residual']:.6f} | {row['utility_eligible_cells']} / {row['cell_count']} |")
    lines.extend(["", "SISA remains `NO_ELIGIBLE_CANDIDATE`. All adverse earlier-stage scope limitations remain unchanged.", ""])
    return audit, "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--repo", type=Path, default=Path.cwd()); parser.add_argument("--workspace", type=Path, required=True); parser.add_argument("--write", action="store_true"); args = parser.parse_args()
    repo = args.repo.resolve(); workspace = args.workspace.resolve(); audit, markdown = build(repo, workspace)
    if args.write:
        atomic_json(repo / "docs/audits/ali_stage9_result_audit_v1.json", audit)
        atomic_text(repo / "docs/audits/ALI_STAGE9_RESULT_AUDIT.md", markdown)
    print(__import__("json").dumps({"status": audit["status"], "confirmation": audit["decision"]["status"], "candidate_cells": audit["completeness"]["candidate_cells"], "endpoint_cells": len(audit["endpoint_inventory"]), "test_accessed": False}, indent=2))


if __name__ == "__main__": main()
