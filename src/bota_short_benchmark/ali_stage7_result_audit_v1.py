"""Freeze the complete Stage 7A baseline Development sweep inventory."""
from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

from src.paper_if_a2.common import atomic_json, sha256_file

from . import ali_stage7_baseline_sweep_v1 as stage7


SCHEMA = "bota-ali-stage7-result-audit-v1"
RUN_NAMES = {method: f"ali_stage7a_{method.lower()}_seed42_v1" for method in stage7.METHODS}


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


def build(repo: Path, workspace: Path, config_path: Path) -> tuple[dict[str, Any], str]:
    config = stage7.load_config(repo, workspace, config_path)
    expected_config_hash = sha256_file(config_path)
    amendment_path = repo / config["parent_amendment"]["path"]
    expected_amendment_hash = sha256_file(amendment_path)
    candidates: list[dict[str, Any]] = []
    selections: dict[str, Any] = {}
    source_hashes: dict[str, dict[str, str]] = {}
    total_cells = 0
    for method in stage7.METHODS:
        run_name = RUN_NAMES[method]
        stage7.analyze(workspace, config, run_name)
        run = workspace / config["output_root"] / "runs" / run_name
        contract = _json(run / "contract.json")
        state = _json(run / "run_state.json")
        selection = _json(run / "selection.json")
        metrics = _json(run / "candidate_metrics.json")
        for value, label in ((contract, "contract"), (state, "run_state"), (selection, "selection"), (metrics, "metrics")):
            _assert_no_test_access(value, f"{method}.{label}")
        if contract["config_sha256"] != expected_config_hash or contract["amendment_sha256"] != expected_amendment_hash:
            raise ValueError(f"{method} is not bound to the frozen Stage 7 configuration")
        if state["method"] != method or selection["method"] != method or metrics["method"] != method:
            raise ValueError(f"{method} method binding mismatch")
        rows = metrics["rows"]
        expected_ids = stage7._candidate_ids(config, method)
        if len(rows) != 2 * len(expected_ids) or state["candidate_cells"] != len(rows):
            raise ValueError(f"{method} candidate-cell count mismatch")
        by_id: dict[str, list[dict[str, Any]]] = {candidate: [] for candidate in expected_ids}
        for row in rows:
            if row["candidate_id"] not in by_id or row["scenario"] not in stage7.SCENARIOS:
                raise ValueError(f"unexpected {method} candidate row")
            by_id[row["candidate_id"]].append(row)
        selected_id = selection["selected"]["candidate_id"] if selection["selected"] else None
        for summary in selection["candidates"]:
            candidate_id = summary["candidate_id"]
            scenario_rows = sorted(by_id[candidate_id], key=lambda row: stage7.SCENARIOS.index(row["scenario"]))
            if [row["scenario"] for row in scenario_rows] != list(stage7.SCENARIOS):
                raise ValueError(f"{method}/{candidate_id} scenario completeness mismatch")
            candidates.append({
                "method": method,
                "candidate_id": candidate_id,
                "eligible": bool(summary["eligible"]),
                "selected": candidate_id == selected_id,
                "equal_scenario_mean_EU": float(summary["equal_scenario_mean_EU"]),
                "compute_count": int(summary["compute_count"]),
                "failure_reasons": list(summary["failure_reasons"]),
                "scenarios": [{
                    "scenario": row["scenario"],
                    "E_U": float(row["E_U"]),
                    "D_U": float(row["D_U"]),
                    "local_residual": float(row["local_residual"]),
                    "toward_exact_rate": float(row["toward_exact_rate"]),
                    "overall_auc": float(row["overall_auc"]),
                    "overall_log_loss": float(row["overall_log_loss"]),
                    "auc_damage": float(row["auc_damage"]),
                    "logloss_damage": float(row["logloss_damage"]),
                    "utility_eligible": bool(row["utility_eligible"]),
                    "prediction_collapse": bool(row["prediction_collapse"]),
                    "hyperparameters": row["hyperparameters"],
                } for row in scenario_rows],
            })
        selections[method] = {
            "status": selection["status"],
            "selected_candidate": selected_id,
            "equal_scenario_mean_EU": None if selection["selected"] is None else float(selection["selected"]["equal_scenario_mean_EU"]),
            "eligible_candidates": sum(bool(row["eligible"]) for row in selection["candidates"]),
            "candidate_count": len(selection["candidates"]),
            "run_code_commit": contract["git"]["head"],
        }
        source_hashes[method] = {name: sha256_file(run / name) for name in (
            "manifest.json", "contract.json", "run_state.json", "selection.json", "candidate_metrics.json", "candidate_metrics.csv"
        )}
        total_cells += len(rows)
    audit = {
        "schema": SCHEMA,
        "status": "FROZEN_COMPLETE_STAGE7A",
        "frozen_date": "2026-09-14",
        "experiment": "Development-only joint L8/L4M4 baseline hyperparameter sweep",
        "analysis_split": "MovieLens-1M Development/Dev-Select",
        "screen_seed": 42,
        "finaltest_accessed": False,
        "test_accessed": False,
        "configuration": {
            "amendment_path": str(amendment_path.relative_to(repo)).replace("\\", "/"),
            "amendment_sha256": expected_amendment_hash,
            "config_path": str(config_path.relative_to(repo)).replace("\\", "/"),
            "config_sha256": expected_config_hash,
            "dev_fit_index_sha256": config["development_partition"]["fit_index_sha256"],
            "dev_select_index_sha256": config["development_partition"]["select_index_sha256"],
        },
        "completeness": {
            "methods": list(stage7.METHODS),
            "candidate_count": len(candidates),
            "candidate_cells": total_cells,
            "scenarios": list(stage7.SCENARIOS),
            "all_run_manifests_validated": True,
            "all_candidates_reported": True,
        },
        "selections": selections,
        "candidates": candidates,
        "source_sha256": source_hashes,
        "integrity_rules": [
            "No Stage 7A candidate or selection used FinalTest.",
            "One configuration is selected jointly across L8 and L4M4 for each method.",
            "SISA has no eligible candidate and no failed candidate may be silently promoted.",
            "The selected configurations remain Development-only until a separately frozen confirmation protocol is executed.",
            "No BOTA setting was selected from these baseline outcomes.",
        ],
    }
    lines = [
        "# Ali Stage 7A unified baseline candidate audit", "",
        "Frozen Development-only inventory. FinalTest was not accessed.", "",
        "## Selection summary", "",
        "| Method | Status | Selected candidate | Mean $E_U$ | Eligible / total | Run commit |",
        "|---|---|---|---:|---:|---|",
    ]
    for method in stage7.METHODS:
        row = selections[method]
        mean = "--" if row["equal_scenario_mean_EU"] is None else f"{row['equal_scenario_mean_EU']:.9f}"
        selected = row["selected_candidate"] or "--"
        lines.append(f"| {method} | {row['status']} | `{selected}` | {mean} | {row['eligible_candidates']} / {row['candidate_count']} | `{row['run_code_commit'][:12]}` |")
    lines.extend(["", "## Complete candidate inventory", "", "| Method | Candidate | Mean $E_U$ | Eligible | Selected | Compute | Failure reasons |", "|---|---|---:|---|---|---:|---|"])
    for row in candidates:
        failures = "; ".join(row["failure_reasons"]) or "--"
        lines.append(f"| {row['method']} | `{row['candidate_id']}` | {row['equal_scenario_mean_EU']:.9f} | {'yes' if row['eligible'] else 'no'} | {'yes' if row['selected'] else 'no'} | {row['compute_count']} | {failures} |")
    lines.extend(["", "The JSON companion freezes per-scenario metrics, hyperparameters and hashes for every candidate.", ""])
    return audit, "\n".join(lines)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/ali_stage7_baseline_sweep_v1.yaml"))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve(); workspace = args.workspace.resolve()
    config_path = (repo / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    audit, markdown = build(repo, workspace, config_path)
    if args.write:
        json_path = repo / "docs/audits/ali_stage7_result_audit_v1.json"
        markdown_path = repo / "docs/audits/ALI_STAGE7_CANDIDATE_AUDIT.md"
        atomic_json(json_path, audit); _atomic_text(markdown_path, markdown)
        result = {"status": audit["status"], "json": str(json_path), "markdown": str(markdown_path), "candidate_count": audit["completeness"]["candidate_count"], "candidate_cells": audit["completeness"]["candidate_cells"], "test_accessed": False}
    else:
        result = {"status": audit["status"], "candidate_count": audit["completeness"]["candidate_count"], "candidate_cells": audit["completeness"]["candidate_cells"], "test_accessed": False}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
