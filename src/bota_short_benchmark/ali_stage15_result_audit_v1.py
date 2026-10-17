"""Seal completed Stage 15 without rewriting its source or experimental record."""
import argparse
import json
from pathlib import Path

from src.paper_if_a2.common import atomic_json, atomic_text, canonical_hash, sha256_file
from . import ali_stage15_ranking_v1 as s15

SCHEMA = "bota-ali-stage15-result-audit-v1"


def build(repo, workspace):
    root = workspace / s15.ROOT
    s15.verify_bundle(root / "protocol")
    s15.verify_bundle(root / "summary")
    contract = s15.read(root / "protocol/contract.json")
    plan = s15.read(root / "protocol/candidates.json")
    s15.verify_inputs(contract, workspace)
    config = s15.load_config(repo, repo / "configs/ali_stage15_ranking_v1.json")
    _, entries, dependencies = s15.authority(repo, workspace, config)
    if entries != contract["endpoints"] or dependencies != contract["dependencies"]:
        raise ValueError("Stage 15 endpoint inventory changed")
    summary = s15.summarize(root, plan, contract)
    if summary["status"] != "COMPLETED" or summary["missing"] or len(summary["rows"]) != 198:
        raise ValueError("Stage 15 is incomplete")
    paths = [root / "protocol", root / "summary"] + sorted((root / "cells").iterdir())
    if len(paths) != 68:
        raise ValueError("Stage 15 cell inventory mismatch")
    for path in paths:
        s15.verify_bundle(path)
    coverage = []
    wall = 0.
    for entry in entries:
        cell = root / "cells" / f"s{entry['seed']}_{entry['scenario']}_{entry['method']}"
        metrics = s15.read(cell / "metrics.json")
        wall += metrics["wall_seconds"]
        if entry["method"] == "BOTA64":
            coverage.append({"seed": entry["seed"], "scenario": entry["scenario"],
                "requested": len(entry["users"]), "covered": len(metrics["covered_requested_users"]),
                "missing": metrics["uncovered_requested_users"]})
    return {"schema": SCHEMA, "status": "FROZEN_COMPLETE_STAGE15", "test_accessed": False,
        "source_config_sha256": sha256_file(repo / "configs/ali_stage15_ranking_v1.json"),
        "root": s15.ROOT, "bundles": {p.relative_to(root).as_posix(): sha256_file(p / "manifest.json") for p in paths},
        "candidate_sha256": canonical_hash(plan), "contract_sha256": canonical_hash(contract),
        "endpoint_cells": 66, "all_development_users": len(plan["coverage"]), "eligible_users": len(plan["users"]),
        "candidate_pairs": plan["candidate_count"], "request_coverage": coverage,
        "wall_seconds_sum": wall, "lambda_frozen": 64, "hyperparameter_reselection": False,
        "three_seed_results": summary["three_seed"], "comparisons": summary["comparisons"],
        "interpretation": "Descriptive sampled-ranking utility only. No equivalence or privacy claim. No automatic FinalTest authorization. Prior Development exposure retained."}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, default=Path.cwd())
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--write", action="store_true")
    a = p.parse_args(); repo = a.repo.resolve(); workspace = a.workspace.resolve()
    audit = build(repo, workspace)
    path = repo / "docs/audits/ali_stage15_result_audit_v1.json"
    if a.write:
        if path.exists() and s15.read(path) != audit:
            raise ValueError("refusing to overwrite a different sealed audit")
        if not path.exists():
            atomic_json(path, audit)
            atomic_text(repo / "docs/audits/ALI_STAGE15_RESULT_AUDIT.md",
                "# Stage 15 sealed result audit\n\n66/66 endpoints, 484 eligible users, 53,809 candidate pairs.\n\n"
                "Lambda=64 fixed; no training or FinalTest access. Raw bundles, candidate order and all results are hash-bound in the adjacent JSON.\n\n"
                "Sampled ranking utility is broadly preserved; do not infer full-catalogue performance, statistical equivalence, or privacy forgetting.\n")
    print(json.dumps({"status": audit["status"], "endpoint_cells": audit["endpoint_cells"],
        "written": a.write, "test_accessed": False}, indent=2))


if __name__ == "__main__": main()
