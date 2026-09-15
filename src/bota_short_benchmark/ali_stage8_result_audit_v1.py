"""Freeze the complete Stage 8 BOTA Development response-strength sweep."""
from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

from src.paper_if_a2.common import atomic_json, sha256_file

from . import ali_stage8_bota_response_sweep_v1 as stage8


SCHEMA = "bota-ali-stage8-result-audit-v1"
RUN_NAME = "ali_stage8_bota_response_seed42_v1"


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
    config = stage8.load_config(repo, config_path)
    stage8.analyze(workspace, config, RUN_NAME)
    run = workspace / config["output_root"] / "runs" / RUN_NAME
    contract = _json(run / "contract.json")
    state = _json(run / "run_state.json")
    selection = _json(run / "selection.json")
    metrics = _json(run / "candidate_metrics.json")
    manifest = _json(run / "manifest.json")
    for value, label in ((contract, "contract"), (state, "run_state"), (selection, "selection"), (metrics, "metrics"), (manifest, "manifest")):
        _assert_no_test_access(value, label)

    config_hash = sha256_file(config_path)
    amendment_path = repo / config["parent_amendment"]["path"]
    baseline_audit_path = repo / config["baseline_audit"]["path"]
    if contract["config_sha256"] != config_hash:
        raise ValueError("Stage 8 run is not bound to the frozen configuration")
    if contract["amendment_sha256"] != sha256_file(amendment_path):
        raise ValueError("Stage 8 run is not bound to Amendment 07")
    if contract["baseline_audit_sha256"] != sha256_file(baseline_audit_path):
        raise ValueError("Stage 8 run is not bound to the frozen Stage 7 audit")

    expected_candidates = [stage8._candidate_id(scale) for scale in stage8.SCALES]
    rows = metrics["rows"]
    if len(rows) != len(expected_candidates) * len(stage8.SCENARIOS) or state["candidate_cells"] != len(rows):
        raise ValueError("Stage 8 candidate-cell count mismatch")
    by_id: dict[str, list[dict[str, Any]]] = {candidate: [] for candidate in expected_candidates}
    for row in rows:
        if row["candidate_id"] not in by_id or row["scenario"] not in stage8.SCENARIOS:
            raise ValueError("unexpected Stage 8 candidate row")
        by_id[row["candidate_id"]].append(row)
    if [entry["candidate_id"] for entry in selection["candidates"]] != expected_candidates:
        raise ValueError("Stage 8 candidate order or identity changed")
    if selection["status"] != "SELECTED" or selection["selected"]["candidate_id"] != "bota_r1":
        raise ValueError("Stage 8 frozen selection is not bota_r1")

    candidates: list[dict[str, Any]] = []
    for summary in selection["candidates"]:
        candidate_id = summary["candidate_id"]
        scenario_rows = sorted(by_id[candidate_id], key=lambda row: stage8.SCENARIOS.index(row["scenario"]))
        if [row["scenario"] for row in scenario_rows] != list(stage8.SCENARIOS):
            raise ValueError(f"scenario completeness mismatch for {candidate_id}")
        candidates.append({
            "candidate_id": candidate_id,
            "response_scale": float(summary["response_scale"]),
            "eligible": bool(summary["eligible"]),
            "selected": candidate_id == selection["selected"]["candidate_id"],
            "equal_scenario_mean_EU": float(summary["equal_scenario_mean_EU"]),
            "failure_reasons": list(summary["failure_reasons"]),
            "scenarios": [{
                "scenario": row["scenario"],
                "E_U": float(row["E_U"]),
                "D_U": float(row["D_U"]),
                "local_residual": float(row["local_residual"]),
                "toward_exact_rate": float(row["toward_exact_rate"]),
                "overall_auc": float(row["overall_auc"]),
                "overall_acc": float(row["overall_acc"]),
                "overall_log_loss": float(row["overall_log_loss"]),
                "auc_damage": float(row["auc_damage"]),
                "logloss_damage": float(row["logloss_damage"]),
                "utility_eligible": bool(row["utility_eligible"]),
                "prediction_collapse": bool(row["prediction_collapse"]),
            } for row in scenario_rows],
        })

    reproduction: dict[str, Any] = {}
    for scenario in stage8.SCENARIOS:
        entry = metrics["formal_reproduction"][scenario]
        if entry["passed"] is not True:
            raise ValueError(f"formal scale-one reproduction failed for {scenario}")
        reproduction[scenario] = entry

    audit = {
        "schema": SCHEMA,
        "status": "FROZEN_COMPLETE_STAGE8",
        "frozen_date": "2026-09-15",
        "experiment": "Development-only joint L8/L4M4 BOTA response-strength sweep",
        "analysis_split": "MovieLens-1M Development/Dev-Select",
        "screen_seed": 42,
        "finaltest_accessed": False,
        "test_accessed": False,
        "configuration": {
            "amendment_path": str(amendment_path.relative_to(repo)).replace("\\", "/"),
            "amendment_sha256": sha256_file(amendment_path),
            "baseline_audit_path": str(baseline_audit_path.relative_to(repo)).replace("\\", "/"),
            "baseline_audit_sha256": sha256_file(baseline_audit_path),
            "config_path": str(config_path.relative_to(repo)).replace("\\", "/"),
            "config_sha256": config_hash,
            "dev_fit_index_sha256": config["development_partition"]["fit_index_sha256"],
            "dev_select_index_sha256": config["development_partition"]["select_index_sha256"],
        },
        "completeness": {
            "candidate_count": len(candidates),
            "candidate_cells": len(rows),
            "scenarios": list(stage8.SCENARIOS),
            "response_scales": list(stage8.SCALES),
            "all_run_manifests_validated": True,
            "all_candidates_reported": True,
        },
        "selection": {
            "status": selection["status"],
            "selected_candidate": selection["selected"]["candidate_id"],
            "response_scale": float(selection["effective_response_scale"]),
            "equal_scenario_mean_EU": float(selection["selected"]["equal_scenario_mean_EU"]),
            "eligible_candidates": sum(bool(row["eligible"]) for row in selection["candidates"]),
            "candidate_count": len(selection["candidates"]),
            "run_code_commit": contract["git"]["head"],
        },
        "formal_scale_one_reproduction": reproduction,
        "candidates": candidates,
        "source_sha256": {name: sha256_file(run / name) for name in (
            "manifest.json", "contract.json", "run_state.json", "selection.json", "candidate_metrics.json", "candidate_metrics.csv"
        )},
        "adapter_sha256": manifest["adapters"],
        "integrity_rules": [
            "No Stage 8 candidate or selection used FinalTest.",
            "One response scale was selected jointly across L8 and L4M4.",
            "All five preregistered response scales and both scenarios are reported.",
            "The selected scale-one endpoint passed the non-selection formal-reproduction gate in both scenarios.",
            "The Development selection is frozen before any cross-seed confirmation run.",
        ],
    }

    lines = [
        "# Ali Stage 8 BOTA response-strength audit", "",
        "Frozen Development-only inventory. FinalTest was not accessed.", "",
        "## Selection", "",
        f"Selected `bota_r1` (response scale 1.0), with equal-scenario mean $E_U={selection['selected']['equal_scenario_mean_EU']:.9f}$.", "",
        "| Candidate | Response scale | Mean $E_U$ | Eligible | Selected |", "|---|---:|---:|---|---|",
    ]
    for row in candidates:
        lines.append(f"| `{row['candidate_id']}` | {row['response_scale']:.2f} | {row['equal_scenario_mean_EU']:.9f} | {'yes' if row['eligible'] else 'no'} | {'yes' if row['selected'] else 'no'} |")
    lines.extend(["", "## Formal scale-one reproduction", "", "| Scenario | Cosine | Relative $L_2$ error | Maximum absolute error | Passed |", "|---|---:|---:|---:|---|"])
    for scenario in stage8.SCENARIOS:
        entry = reproduction[scenario]
        values = entry["metrics"]
        lines.append(f"| {scenario} | {values['cosine']:.9f} | {values['relative_l2_error']:.9f} | {values['maximum_absolute_error']:.9f} | yes |")
    lines.extend(["", "The JSON companion freezes all per-scenario metrics, artifact hashes and source bindings.", ""])
    return audit, "\n".join(lines)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/ali_stage8_bota_response_sweep_v1.yaml"))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve(); workspace = args.workspace.resolve()
    config_path = (repo / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    audit, markdown = build(repo, workspace, config_path)
    if args.write:
        json_path = repo / "docs/audits/ali_stage8_result_audit_v1.json"
        markdown_path = repo / "docs/audits/ALI_STAGE8_RESULT_AUDIT.md"
        atomic_json(json_path, audit); _atomic_text(markdown_path, markdown)
        result = {"status": audit["status"], "json": str(json_path), "markdown": str(markdown_path), "candidate_count": audit["completeness"]["candidate_count"], "candidate_cells": audit["completeness"]["candidate_cells"], "test_accessed": False}
    else:
        result = {"status": audit["status"], "candidate_count": audit["completeness"]["candidate_count"], "candidate_cells": audit["completeness"]["candidate_cells"], "test_accessed": False}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
