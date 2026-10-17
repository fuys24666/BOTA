"""Recover six frozen Amazon RecEraser endpoints, then score with encoder masks.

No search, no new requests, no FinalTest, no changes to Stage20/21/25.
Reconstruction requires explicit authorization and exact archived artifact hashes.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import torch
import yaml

from src.paper_if_a2.common import (
    atomic_json, atomic_text, canonical_hash, directory_hash, seed_everything, sha256_file,
)
from . import ali_stage20_amazon_baselines_v1 as s20
from . import ali_stage21_amazon_seed43_kernel_v1 as s21
from . import ali_stage25_historical_mask_audit_v1 as s25
from . import ali_stage25_scoring as scoring

SCHEMA = "bota-ali-stage27-amazon-receraser-rescore-v1"
ROOT = "outputs/ali_stage27_amazon_receraser_rescore_v1"
SCRIPT = "scripts/bota_if/run_ali_stage27_amazon_receraser_rescore_v1.ps1"
DOC = "docs/ali_stage27_amazon_receraser_rescore_v1.md"
UNITS = {
    42: ("ali_stage20_amazon_baselines_v1", {"A1": "a5c09354fb73", "A2": "2399a89eabcb", "A3": "3aa47b7ceb5b"}),
    43: ("ali_stage21_amazon_seed43_v1", {"A1": "a5a57fadbc8b", "A2": "15d9c87a0b77", "A3": "b92d908aa807"}),
}
read = s20.read


def digest(path):
    return directory_hash(path) if path.is_dir() else sha256_file(path)


def verify(path, contract=None):
    manifest = read(path / "manifest.json")
    actual = {p.relative_to(path).as_posix() for p in path.rglob("*")
              if p.is_file() and p.name != "manifest.json"}
    if (manifest["schema"] != SCHEMA or actual != set(manifest["files"])
            or (path / "COMPLETED").read_text(encoding="utf-8") != SCHEMA + "\n"):
        raise ValueError(f"invalid Stage27 bundle: {path}")
    for name, expected in manifest["files"].items():
        if sha256_file(path / name) != expected:
            raise ValueError(f"changed artifact: {path / name}")
    result = read(path / "result.json")
    if contract is not None and result["contract"] != contract:
        raise ValueError("Stage27 contract changed")
    return result


def task(root, key, contract, work):
    dest = root / key
    if dest.exists():
        return verify(dest, contract)
    stage = root / ".work" / uuid.uuid4().hex[:10]
    stage.mkdir(parents=True)
    started = time.perf_counter()
    try:
        result = {**work(stage), "contract": contract, "test_accessed": False,
                  "selection_performed": False, "wall_seconds": time.perf_counter() - started}
        atomic_json(stage / "result.json", result)
        atomic_text(stage / "COMPLETED", SCHEMA + "\n")
        atomic_json(stage / "manifest.json", {"schema": SCHEMA, "files": {
            p.relative_to(stage).as_posix(): sha256_file(p)
            for p in sorted(stage.rglob("*")) if p.is_file()}})
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.rename(stage, dest)
        return result
    except BaseException as error:
        atomic_json(stage / "FAILED.json", {"error": repr(error), "key": key, "contract": contract})
        raise


def relocated_config(old, workspace, root, key):
    """Only relocate artifacts. Training, partition, seed and input paths stay frozen."""
    new = copy.deepcopy(old)
    new["output_root"] = (root / "models" / key).relative_to(workspace).as_posix()
    new["synthetic_root"] = (root / "synthetic").relative_to(workspace).as_posix()
    return new


def validate_config(config, seed):
    p = config["protocol"]
    if (config["schema"] != "paper-receraser-adapter-v1" or config["seed"] != seed
            or p["partition_seed"] != seed or p["shards"] != 4
            or p["epochs_per_slice"] != 1 or p["learning_rate"] != .002
            or config["test_access_policy"] != "forbidden"):
        raise ValueError("archived selected configuration differs from frozen scope")


def prepare(repo, workspace):
    print("[Stage27] checking frozen inputs, six endpoints and corrected references; no inference", flush=True)
    parent_root = workspace / s25.ROOT
    parent = s25.verify(parent_root / "protocol")["plan"]
    parent_contract = canonical_hash(parent)
    hashes = {}

    def bind(path):
        path = path.resolve()
        if str(path) not in hashes:
            hashes[str(path)] = digest(path)

    # Bind code without modifying any historical hash-bound source.
    for path in sorted((repo / "src").rglob("*.py")):
        bind(path)
    bind(repo / SCRIPT)
    bind(repo / DOC)
    bind(parent_root / "protocol")
    cohorts, jobs, entries, seconds = {}, [], [], []
    for seed, (folder, units) in UNITS.items():
        cohort_id = f"Amazon_comparison_s{seed}"
        cohort = copy.deepcopy(parent["cohorts"][cohort_id])
        if cohort["scope"] != "Development" or cohort["lane"] != "binary":
            raise ValueError("only frozen Amazon Development binary comparison is authorized")
        if [r["id"] for r in cohort["requests"]] != ["A1", "A2", "A3"]:
            raise ValueError("request cohort changed")
        cohorts[cohort_id] = cohort
        bind(Path(cohort["data_path"]))
        group = [j for j in parent["jobs"] if j["cohort"] == cohort_id]
        if len({(j["request"], j["method"]) for j in group}) != len(group):
            raise ValueError("duplicate parent comparison jobs")
        for job in group:
            result = s25.verify(parent_root / "cells" / job["id"], parent_contract)
            bind(parent_root / "cells" / job["id"])
            if job["method"] in ("Original", "Exact-Masked", "BOTA", "BOTA-source-only") and result["status"] != "OK":
                raise ValueError("required corrected reference is unavailable")
            if job["method"] == "Original":
                seconds.append(result["wall_seconds"])
            jobs.append(copy.deepcopy(job))
        for rid, unit_id in units.items():
            unit = workspace / "outputs" / folder / "units" / unit_id
            (s20 if seed == 42 else s21).verify(unit)
            old_result = read(unit / "result.json")
            identity = old_result["identity"]
            if (identity["method"] != "RecEraser" or identity["phase"] != "comparison"
                    or identity["request"] != rid or identity["spec"] != {"epochs_per_slice": 1, "learning_rate": .002}):
                raise ValueError("unexpected archived endpoint identity")
            bind(unit)
            config = yaml.safe_load((unit / "config.yaml").read_text(encoding="utf-8"))
            validate_config(config, seed)
            for source in config["sources"].values():
                path = workspace / source["path"]
                bind(path)
                if hashes[str(path.resolve())] != source["sha256"]:
                    raise ValueError(f"archived model/data input changed: {path}")
            bind(workspace / config["tokenizer"])
            authority = config["development_subset_authority"]
            for field in ("full", "indices"):
                path = workspace / authority[field + "_path"]
                bind(path)
                if hashes[str(path.resolve())] != authority[field + "_sha256"]:
                    raise ValueError("Development identity authority changed")
            # Identity sidecars are consumed by the unchanged partition engine.
            bind((workspace / config["sources"]["train"]["path"]).parent.parent)
            job = next(j for j in group if j["request"] == rid and j["method"] == "RecEraser")
            if (job["old_metrics"]["model_sha256"] != sha256_file(unit / "paper_model_manifest.json")
                    or job["provenance"]["calibration_status"] != "SELECTED"):
                raise ValueError("archived model is not the selected comparator")
            manifest = read(unit / "paper_model_manifest.json")
            engine_files = [repo / "src/paper_baselines" / name for name in
                            ("partitioned.py", "common.py", "runtime.py", "receraser.py")]
            implementation = {p.relative_to(repo).as_posix(): sha256_file(p) for p in engine_files}
            if canonical_hash(implementation) != manifest["implementation_sha256"]:
                raise ValueError("historical partition engine changed; diagnose before reconstruction")
            artifacts = manifest["artifacts"]
            model_artifact = next(a for a in artifacts if Path(a["path"]).name == "models")
            aggregation_artifact = next(a for a in artifacts if Path(a["path"]).name == "aggregation.json")
            if aggregation_artifact["sha256"] != sha256_file(unit / "aggregation.json"):
                raise ValueError("archived aggregation checksum changed")
            entries.append({"key": f"s{seed}_{rid}", "seed": seed, "request": rid, "unit": str(unit),
                            "cohort": cohort_id, "job_id": job["id"], "config": config,
                            "old_run": str((workspace / model_artifact["path"]).parent),
                            "expected_models_sha256": model_artifact["sha256"],
                            "expected_aggregation_sha256": aggregation_artifact["sha256"],
                            "optimizer_steps": manifest["optimizer_steps"],
                            "historical_wall_seconds": old_result["rows"][0]["wall_time_seconds"]})
    return {"schema": SCHEMA, "cohorts": cohorts, "jobs": jobs, "entries": entries, "hashes": hashes,
            "parent_root": str(parent_root), "parent_contract": parent_contract, "config": parent["config"],
            "runtime": {**s25.s24.s23.c17.runtime(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
            "historical_single_model_score_seconds": sum(seconds) / len(seconds),
            "selection_performed": False, "test_accessed": False,
            "reconstruction_policy": "explicit_opt_in_and_exact_archived_models_and_aggregation_hashes"}


def check_plan(plan):
    runtime = {**s25.s24.s23.c17.runtime(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    if plan["runtime"] != runtime:
        raise ValueError("runtime changed since Stage27 freeze")
    for path, expected in plan["hashes"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"frozen input/source changed: {path}")


def model_match(run, entry):
    """A new configuration-identical run is NOT automatically the same old model."""
    if not (run / "COMPLETED").is_file():
        return False
    s20.partitioned.analyze_run(run, s20.partitioned.MARKERS["receraser"])
    if digest(run / "models") != entry["expected_models_sha256"]:
        raise ValueError(f"reconstruction tensor artifact hash mismatch: {run}; no scores published")
    if sha256_file(run / "aggregation.json") != entry["expected_aggregation_sha256"]:
        raise ValueError(f"frozen aggregation hash mismatch: {run}; no refitted weights accepted")
    if read(run / "partition.json") != read(Path(entry["unit"]) / "partition.json"):
        raise ValueError("partition or data lineage differs from archived endpoint")
    if read(run / "paper_model_manifest.json")["optimizer_steps"] != entry["optimizer_steps"]:
        raise ValueError("reconstruction budget changed")
    return True


@contextmanager
def reconstruction_checkpoints(run, archived_manifest):
    """Old RNG checkpoint semantics, with visible shard progress (no math changes)."""
    engine = s20.partitioned
    manifest_writer = engine.paper_model_manifest

    def model_manifest(**kwargs):
        # If all shards were committed before interruption, the historical engine
        # skips its loop and leaves model_counts at zero. Repair metadata only.
        if kwargs["trainable_parameters"] == 0 and kwargs["total_parameters"] == 0:
            kwargs["trainable_parameters"] = archived_manifest["trainable_parameters"]
            kwargs["total_parameters"] = archived_manifest["total_parameters"]
        return manifest_writer(**kwargs)

    with s20.partition_resume_rng(run):
        writer = engine.atomic_json

        def write(path, value):
            writer(path, value)
            if Path(path) == run / "run_state.json":
                print(f"[Stage27] {run.parent.name} shards={value.get('next_shard', 0)}/4 "
                      f"optimizer_steps={value.get('optimizer_steps', 0)} status={value.get('status')}", flush=True)

        engine.atomic_json = write
        engine.paper_model_manifest = model_manifest
        try:
            yield
        finally:
            engine.atomic_json = writer
            engine.paper_model_manifest = manifest_writer


def quarantine_uncommitted_shards(root, run):
    """Preserve a shard published before its durable RNG/state checkpoint, then replay it."""
    if not run.exists() or (run / "COMPLETED").exists():
        return
    state = read(run / "run_state.json") if (run / "run_state.json").exists() else {"next_shard": 0}
    next_shard = int(state["next_shard"])
    if next_shard and not (run / f"rng_{next_shard}.pt").exists():
        raise ValueError("missing durable reconstruction RNG state; do not silently restart")
    for i in range(next_shard, 4):
        shard = run / "models" / f"shard_{i:02d}"
        if not shard.exists():
            continue
        if (root / "models").resolve() not in shard.resolve().parents:
            raise ValueError("recovery path escaped Stage27 models")
        dest = root / "recovery" / f"{run.parent.name}_{i}_{uuid.uuid4().hex[:8]}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.rename(shard, dest)
        print(f"[Stage27] preserved uncommitted shard at {dest}", flush=True)


def ensure_model(root, repo, workspace, entry, rebuild):
    old_run = Path(entry["old_run"])
    if model_match(old_run, entry):
        return old_run, 0
    run = root / "models" / entry["key"] / "run"
    if model_match(run, entry):
        return run, 0
    if not rebuild:
        raise PermissionError("RecEraser weights are absent. Explicit -RebuildMissing is required; no training started.")
    if (run / "STOPPED_SAFELY").exists():
        raise RuntimeError(f"memory guard stopped this model; inspect before resuming: {run}")
    quarantine_uncommitted_shards(root, run)
    config = relocated_config(entry["config"], workspace, root, entry["key"])
    config_path = root / "configs" / (entry["key"] + ".yaml")
    text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    if config_path.exists() and config_path.read_text(encoding="utf-8") != text:
        raise ValueError("reconstruction configuration changed")
    if not config_path.exists():
        atomic_text(config_path, text)
    print(f"[Stage27] reconstructing {entry['key']} at frozen lr=.002, epochs=1, shards=4; "
          f"expected steps={entry['optimizer_steps']}", flush=True)
    seed_everything(entry["seed"])
    previous = Path.cwd()
    try:
        os.chdir(workspace)
        with reconstruction_checkpoints(run, read(Path(entry["unit"]) / "paper_model_manifest.json")):
            result = s20.partitioned.execute(workspace, config_path, "run", "receraser", run.exists(),
                                             partition_seed_override=entry["seed"], formal=False, implementation_root=repo)
    finally:
        os.chdir(previous)
    if result["status"] != "COMPLETED" or not model_match(run, entry):
        raise RuntimeError("reconstruction not complete and verified")
    return run, entry["optimizer_steps"]


def selected_entries(plan, seed):
    return [e for e in plan["entries"] if seed == "All" or e["seed"] == int(seed)]


def preflight(root, plan, seed):
    entries = selected_entries(plan, seed)
    rows = []
    for entry in entries:
        run = root / "models" / entry["key"] / "run"
        retained = model_match(Path(entry["old_run"]), entry) or model_match(run, entry)
        done = root / "cells" / entry["key"]
        if done.exists():
            result = verify(done, canonical_hash(plan))
            status = "SCORED" if result["status"] == "OK" else result["status"]
        else:
            status = "SCORE_ONLY" if retained else "REBUILD_REQUIRED"
        rows.append({"key": entry["key"], "status": status, "optimizer_steps_if_rebuild": entry["optimizer_steps"]})
    build = sum(e["historical_wall_seconds"] for e, row in zip(entries, rows) if row["status"] == "REBUILD_REQUIRED")
    score = sum(row["status"] != "SCORED" for row in rows) * 4 * plan["historical_single_model_score_seconds"]
    return {"status": "PREFLIGHT_PASS", "inventory": rows, "model_execution": False,
            "historical_rebuild_plus_old_scoring_minutes": build / 60,
            "estimated_new_scoring_minutes": score / 60,
            "planning_minutes_including_checks": [round((build + score) / 60 * 1.1), round((build + score) / 60 * 1.8)],
            "note": "Estimate from historical runs; exact-identity failure stops rather than replacing the baseline."}


def summarize(root, plan):
    contract = canonical_hash(plan)
    pending = [e["key"] for e in plan["entries"] if not (root / "cells" / e["key"]).exists()]
    if pending:
        return {"status": "INCOMPLETE", "pending": pending}
    results = {e["job_id"]: verify(root / "cells" / e["key"], contract) for e in plan["entries"]}
    if any(r["status"] != "OK" for r in results.values()):
        raise RuntimeError("summary blocked by scoring invariance failure")
    all_rows = []
    for cohort_id, cohort in plan["cohorts"].items():
        jobs = [j for j in plan["jobs"] if j["cohort"] == cohort_id]
        current = {j["id"]: results[j["id"]] if j["method"] == "RecEraser" else
                   s25.verify(Path(plan["parent_root"]) / "cells" / j["id"], plan["parent_contract"])
                   for j in jobs}
        # Parent metric routine preserves eligibility and unavailable solver outcomes.
        rows = s25.metric_rows(cohort, jobs, current)
        for row in rows:
            row["scoring_source"] = "Stage27" if row["method"] == "RecEraser" else "Stage25"
            if row["method"] == "RecEraser":
                entry = next(e for e in plan["entries"] if e["cohort"] == cohort_id and e["request"] == row["request"])
                row.update(candidate_id="receraser_lr0.002_e1", model_identity="EXACT_ARCHIVED_ARTIFACT_HASH_MATCH",
                           construction_optimizer_steps=entry["optimizer_steps"], compute_count=entry["optimizer_steps"],
                           hyperparameters=next(j["old_metrics"]["hyperparameters"] for j in jobs
                                                if j["request"] == row["request"] and j["method"] == "RecEraser"),
                           scoring_optimizer_updates=0)
        all_rows.extend(rows)

    def work(stage):
        rec = [r for r in all_rows if r["method"] == "RecEraser"]
        s25.s24.write_csv(stage / "amazon_all_methods.csv", all_rows)
        s25.s24.write_csv(stage / "receraser_metrics.csv", rec)
        lines = ["# Stage27 Amazon RecEraser masked rescoring", "",
                 "Six frozen selected endpoints; no search or new requests. Corrected Stage25 comparators reused.",
                 "Restored/reconstructed model and aggregation hashes must exactly match archived artifacts.",
                 "Utility failures remain in the report. PCGrad/SISA calibration failures and IFRU solver failures are not successes.",
                 "This supplements the reused Development comparison, not an independent FinalTest evaluation.", "",
                 "| Seed | Request | E_U | Residual | AUC | Log loss | Utility eligible |",
                 "|---|---|---:|---:|---:|---:|---|"]
        for r in rec:
            residual = "NA" if r["local_residual"] is None else f"{r['local_residual']:.6g}"
            lines.append(f"| {r['seed']} | {r['request']} | {r['E_U']:.6g} | {residual} | "
                         f"{r['overall_auc']:.6g} | {r['overall_log_loss']:.6g} | {r['utility_eligible']} |")
        atomic_text(stage / "REPORT.md", "\n".join(lines) + "\n")
        return {"status": "COMPLETED", "rows": all_rows, "receraser_rows": len(rec), "optimizer_updates": 0}
    result = task(root, "summary", contract, work)
    return {"status": result["status"], "receraser_rows": result["receraser_rows"],
            "all_comparison_rows": len(result["rows"]), "report": str(root / "summary" / "REPORT.md")}


def execute(root, repo, workspace, plan, seed, rebuild):
    contract = canonical_hash(plan)
    entries = selected_entries(plan, seed)
    # Fail before starting any model if missing weights have not been authorized.
    if not rebuild:
        status = preflight(root, plan, seed)
        if any(r["status"] == "REBUILD_REQUIRED" for r in status["inventory"]):
            raise PermissionError("missing selected weights; rerun with -RebuildMissing only if reconstruction is authorized")
    for entry in entries:
        cell = root / "cells" / entry["key"]
        if cell.exists():
            if verify(cell, contract)["status"] != "OK":
                raise RuntimeError("completed diagnostic failure requires investigation, not an automatic retry")
            print(f"[Stage27] reuse scored {entry['key']}", flush=True)
            continue
        run, construction_steps = ensure_model(root, repo, workspace, entry, rebuild)
        cohort = plan["cohorts"][entry["cohort"]]
        job = copy.deepcopy(next(j for j in plan["jobs"] if j["id"] == entry["job_id"]))
        job.update(path=str(run), kind="receraser_ensemble", status="READY")
        data = s25.dataset(workspace, cohort, False)

        def work(stage):
            def progress(shard, done, total):
                atomic_json(stage / "progress.json", {"key": entry["key"], "shard": shard, "done": done, "total": total})
                print(f"[Stage27] score {entry['key']} shard={shard} {done}/{total}", flush=True)
            scored = scoring.score_endpoint(workspace, cohort, job, data, plan["config"], progress, read)
            if not model_match(run, entry):
                raise RuntimeError("model changed during scoring")
            return {**scored, "job": job, "optimizer_updates": 0,
                    "construction_steps_in_this_invocation": construction_steps,
                    "endpoint_construction_optimizer_steps": entry["optimizer_steps"],
                    "identity": "EXACT_ARCHIVED_ARTIFACT_HASH_MATCH"}
        try:
            result = task(root, "cells/" + entry["key"], contract, work)
        finally:
            del data
        if result["status"] != "OK":
            raise RuntimeError("batch invariance failed; diagnostics preserved and summary blocked")
    check_plan(plan)
    return summarize(root, plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--workspace", type=Path, default=Path("C:/python/E2URec-main"))
    parser.add_argument("--mode", choices=["Preflight", "Freeze", "Full", "Analyze"], default="Preflight")
    parser.add_argument("--seed", choices=["All", "42", "43"], default="All")
    parser.add_argument("--rebuild-missing", action="store_true")
    args = parser.parse_args()
    repo, workspace = args.repo.resolve(), args.workspace.resolve()
    root = workspace / ROOT
    if (root / "protocol").exists():
        plan = verify(root / "protocol")["plan"]
        check_plan(plan)
    else:
        plan = prepare(repo, workspace)
    if args.mode == "Preflight":
        result = preflight(root, plan, args.seed)
    else:
        with s20.lock(root):
            task(root, "protocol", canonical_hash(plan), lambda stage: {"status": "FROZEN", "plan": plan, "optimizer_updates": 0})
            if args.mode == "Freeze":
                result = {"status": "FROZEN", "contract": canonical_hash(plan)}
            elif args.mode == "Analyze":
                result = summarize(root, plan)
            else:
                result = execute(root, repo, workspace, plan, args.seed, args.rebuild_missing)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
