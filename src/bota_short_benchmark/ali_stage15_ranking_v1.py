"""Frozen lambda=64, Development-only sampled ranking evaluation (no training).

Candidate construction is outcome-blind with respect to model scores. See the
versioned config/amendment for the deliberately limited sampled-ranking scope.
FinalTest loaders are neither imported nor called. Existing artifacts are read-only.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import T5Tokenizer

from src.diagnostics import ml1m_development_protocol as lineage
from src.diagnostics.t5_reconstructed_official import JsonPromptDataset
from src.paper_if_a2.common import atomic_json, atomic_text, canonical_hash, directory_hash, sha256_file
from . import ali_stage9_crossseed_development_v1 as s9
from . import ali_stage14_lambda_search_v1 as s14
from . import evaluation, runner

SCHEMA = "bota-ali-stage15-ranking-v1"
ROOT = "outputs/ali_stage15_ranking_v1"
SEEDS = (41, 42, 43)
SCENARIOS = ("L8", "L4M4")
METHODS = ("Original", "Exact-Masked", "BOTA64", "BOTA1", "FullControl-P5",
           "Retain-Retrain-P5", "IFRU", "E2URec", "NegGrad", "PCGrad", "RecEraser")
KS = (5, 10, 20)
PROTOCOL = {
    "split": "Development/Dev-Select", "response_scale": 64,
    "seeds": list(SEEDS), "scenarios": list(SCENARIOS), "methods": list(METHODS),
    "candidate_mode": "all_eligible_DevSelect_positives_plus_100_unobserved",
    "negative_count": 100, "sampling_namespace": "ali-stage15-ranking-v1",
    "catalogue": "items_observed_before_Development",
    "history": "last_10_before_first_Development_event_per_user",
    "exclude": "all_user_observations_through_Development_end",
    "positive": "DevSelect_rating_ge_4_not_in_preDevelopment_history_in_catalogue",
    "no_positive": "exclude_from_macro_and_report", "insufficient_negatives": "use_all_and_report",
    "ks": list(KS), "primary": "sampled_NDCG@10", "secondary": "NDCG@5/20_and_Recall@5/10/20",
    "tie_break": "ascending_movie_id", "aggregation": "macro_user_then_seed_mean_sample_sd",
    "score": "first_token_PYes_over_PYes_plus_PNo;ensemble_mix_token_mass_first",
    "inference_batch_size": 4, "optimizer_updates": 0, "hyperparameter_selection": False,
    "final_test": False,
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_path(root, relative):
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or any("test" in p.lower() for p in path.relative_to(root.resolve()).parts):
        raise ValueError(f"out-of-scope or test path: {relative}")
    return path


def load_config(repo, path):
    config = read(path)
    if (config.get("schema") != SCHEMA or config.get("protocol") != PROTOCOL
            or config.get("test_access_policy") != "forbidden" or config.get("output_root") != ROOT):
        raise ValueError("Stage 15 protocol changed; use a new amendment/version")
    for name, expected in config["bindings"].items():
        if sha256_file(repo / name) != expected:
            raise ValueError(f"frozen binding changed: {name}")
    amendment = read(repo / "configs/ali_stage0_amendment14_v1.json")
    if amendment["protocol"] != PROTOCOL or amendment["status"] != "SPECIFIED_BEFORE_STAGE15_EXECUTION":
        raise ValueError("amendment and executable rules disagree")
    return config


def build_candidates(train, development, select, catalogue, negative_count=100):
    """Pure deterministic candidate construction; no model or test predictions."""
    if len(set(select)) != len(select) or any(i < 0 or i >= len(development) for i in select):
        raise ValueError("invalid Development indices")
    groups = defaultdict(list)
    training_seen = defaultdict(set)
    for row in train:
        training_seen[row.authoritative_user_id].update([row.movie_id, *row.history_ids])
    for i, row in enumerate(development):
        groups[row.authoritative_user_id].append((i, row))
    chosen = set(select)
    result, coverage = [], []
    for user, entries in sorted(groups.items()):
        entries.sort(key=lambda pair: (pair[1].timestamp, pair[1].raw_source_row))
        anchor = entries[0][1]
        seen = set(anchor.history_ids) | training_seen[user]
        observed = seen | {row.movie_id for _, row in entries}
        item_labels = {}
        for i, row in entries:
            if row.movie_id in item_labels and item_labels[row.movie_id] != row.label:
                raise ValueError("conflicting duplicate Development user/item labels")
            item_labels[row.movie_id] = row.label
        raw_positive = {row.movie_id for i, row in entries if i in chosen and row.label == 1}
        positive = sorted((raw_positive - seen) & catalogue)
        eligible_pool = catalogue - observed
        negatives = sorted(eligible_pool, key=lambda item: (
            canonical_hash([PROTOCOL["sampling_namespace"], user, item]), item))[:negative_count]
        coverage.append({"user": user, "development_rows": len(entries),
                         "select_rows": sum(i in chosen for i, _ in entries),
                         "raw_positive_items": len(raw_positive), "eligible_positive_items": len(positive),
                         "positive_seen_excluded": len(raw_positive & seen),
                         "positive_out_of_catalogue": len(raw_positive - catalogue),
                         "negative_pool": len(eligible_pool), "negative_count": len(negatives),
                         "status": "ELIGIBLE" if positive else "NO_ELIGIBLE_POSITIVE"})
        if not positive:
            continue
        if not negatives:
            raise ValueError(f"user {user} has no unseen comparison items; protocol review required")
        items = sorted(positive + negatives)
        if len(items) != len(set(items)) or set(items) & seen:
            raise ValueError("candidate duplicates or training/history leakage")
        result.append({"user": user, "items": items, "positive_items": positive,
                       "history_ids": anchor.history_ids[-10:], "history_ratings": anchor.history_ratings[-10:],
                       "anchor_index": anchor.processed_row_index, "anchor_timestamp": anchor.timestamp})
    if not result:
        raise ValueError("no eligible ranking users")
    return {"users": result, "coverage": coverage, "candidate_count": sum(len(r["items"]) for r in result),
            "catalogue_items": len(catalogue), "test_accessed": False}


def user_metrics(items, positive_items, scores):
    if len(items) != len(scores) or len(set(items)) != len(items) or not positive_items:
        raise ValueError("invalid ranking input")
    if not set(positive_items) <= set(items) or len(set(positive_items)) != len(positive_items):
        raise ValueError("invalid relevance labels")
    if not all(math.isfinite(float(x)) and 0 <= x <= 1 for x in scores):
        raise ValueError("nonfinite or invalid candidate score")
    order = sorted(range(len(items)), key=lambda i: (-float(scores[i]), items[i]))
    positive = set(positive_items)
    relevance = [int(items[i] in positive) for i in order]
    result = {}
    for k in KS:
        dcg = sum(y / math.log2(rank + 2) for rank, y in enumerate(relevance[:k]))
        ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(positive))))
        result[f"ndcg@{k}"] = dcg / ideal
        result[f"recall@{k}"] = sum(relevance[:k]) / len(positive)
    return result


def metrics_for(plan, scores, requested):
    if len(scores) != plan["candidate_count"]:
        raise ValueError("candidate score count mismatch")
    rows, offset = [], 0
    for user in plan["users"]:
        count = len(user["items"])
        row = {"user": user["user"], "candidates": count, "positives": len(user["positive_items"]),
               **user_metrics(user["items"], user["positive_items"], scores[offset:offset+count])}
        rows.append(row)
        offset += count
    aggregates = []
    for group in ("all", "requested", "retained"):
        selected = [row for row in rows if group == "all" or (row["user"] in requested) == (group == "requested")]
        aggregate = {"group": group, "users": len(selected)}
        for key in (f"{metric}@{k}" for metric in ("ndcg", "recall") for k in KS):
            aggregate[key] = float(np.mean([r[key] for r in selected])) if selected else None
        aggregates.append(aggregate)
    covered = {r["user"] for r in rows}
    return {"per_user": rows, "aggregates": aggregates, "requested_users": sorted(requested),
            "covered_requested_users": sorted(requested & covered),
            "uncovered_requested_users": sorted(requested - covered)}


def authority(repo, workspace, config):
    root14 = workspace / "outputs/ali_stage14_lambda_search_v1"
    expected = {"protocol", "coarse", "selection", "summary", "cells/fine/s42_L8", "cells/fine/s42_L4M4"} | {
        f"cells/{'coarse' if seed == 42 else 'confirm'}/s{seed}_{sid}" for seed in SEEDS for sid in SCENARIOS}
    if set(config["stage14_manifests"]) != expected:
        raise ValueError("incomplete frozen Stage 14 inventory")
    for relative, expected_hash in config["stage14_manifests"].items():
        path = root14 / relative
        if sha256_file(path / "manifest.json") != expected_hash:
            raise ValueError(f"Stage 14 changed: {relative}")
        s14.verify_bundle(path)
    if (read(root14 / "selection/decision.json")["selected"]["response_scale"] != 64
            or read(root14 / "summary/summary.json")["confirmation"]["status"] != "CROSS_SEED_DEVELOPMENT_CONFIRMED"):
        raise ValueError("requires frozen confirmed lambda=64")
    config9 = s9.load_config(repo, workspace, repo / "configs/ali_stage9_crossseed_development_v1.yaml")
    audit = read(repo / "docs/audits/ali_stage9_result_audit_v1.json")
    s9._assert_no_test_access(audit)
    entries = []
    for old in audit["endpoint_inventory"]:
        entry = dict(old)
        if entry["method"] == "BOTA":
            entry["method"] = "BOTA1"
        path = safe_path(workspace, entry["path"])
        if directory_hash(path) != entry["directory_sha256"]:
            raise ValueError(f"frozen endpoint changed: {path}")
        entries.append(entry)
    for seed in SEEDS:
        for sid in SCENARIOS:
            template = next(e for e in entries if e["seed"] == seed and e["scenario"] == sid and e["method"] == "BOTA1")
            path = root14 / "cells" / ("coarse" if seed == 42 else "confirm") / f"s{seed}_{sid}" / "adapters/bota_r64/adapter"
            entries.append({**template, "method": "BOTA64", "path": path.relative_to(workspace).as_posix(),
                            "directory_sha256": directory_hash(path)})
    expected_matrix = {(s, sid, m) for s in SEEDS for sid in SCENARIOS for m in METHODS}
    if len(entries) != len(expected_matrix) or {(e["seed"], e["scenario"], e["method"]) for e in entries} != expected_matrix:
        raise ValueError("endpoint matrix incomplete or duplicated")
    for dep in audit["shared_model_dependencies"].values():
        path = safe_path(workspace, dep["path"])
        if (directory_hash(path) if path.is_dir() else sha256_file(path)) != dep["sha256"]:
            raise ValueError("shared model dependency changed")
    return config9, entries, audit["shared_model_dependencies"]


def prepare(repo, workspace, config_path):
    config = load_config(repo, config_path)
    config9, entries, dependencies = authority(repo, workspace, config)
    benchmarks = {}
    source_paths = set()
    development_paths = set()
    for seed in SEEDS:
        benchmark, registry = s9.stage7._authority(repo, workspace, s9._seed_config(config9, seed))
        benchmarks[seed] = benchmark
        for entry in (e for e in entries if e["seed"] == seed):
            scenario = next(r for r in registry["scenarios"] if r["id"] == entry["scenario"])
            if entry["users"] != scenario["users"] or entry["registry_sha256"] != registry["registry_sha256"]:
                raise ValueError("frozen request changed")
        development_paths.add(safe_path(workspace, benchmark["source"]["development_json"]))
        base = runner._base_t5_config(workspace, benchmark)
        model_dir = Path(base["paths"]["model_dir"])
        if not model_dir.is_absolute(): model_dir = workspace / model_dir
        if model_dir.resolve() != safe_path(workspace, dependencies["pretrained_model_directory"]["path"]):
            raise ValueError("tokenizer/model directory differs from frozen dependency")
        source_paths.add(safe_path(workspace, benchmark["source"]["base_config"]))
    benchmark = benchmarks[42]
    raw = safe_path(workspace, benchmark["source"]["raw_data"])
    train, development, replay = lineage.reconstruct_authoritative_rows(raw)
    users, movies = lineage.load_raw_metadata(raw)
    rendered = [lineage.render_record(row, users, movies) for row in development]
    for path in development_paths:
        lineage.compare_exact(rendered, read(path), "Development")
    source_paths.update(development_paths)
    split = config9["development_partition"]
    select = [i for i, row in enumerate(development) if int(canonical_hash(
        [split["namespace"], 42, row.authoritative_user_id, row.label, i]), 16) % 2 == split["select_parity"]]
    if len(select) != split["select_count"] or canonical_hash(select) != split["select_index_sha256"]:
        raise ValueError("Dev-Select partition mismatch")
    catalogue = {item for row in train for item in [row.movie_id, *row.history_ids]}
    # Include histories for users whose pre-Development observations predate the
    # retained 60k training slice. Never use Development targets to enlarge it.
    first = {}
    for row in development:
        first.setdefault(row.authoritative_user_id, row)
    for row in first.values():
        catalogue.update(row.history_ids)
    if not catalogue <= set(movies):
        raise ValueError("catalogue metadata incomplete")
    plan = build_candidates(train, development, select, catalogue)
    plan["protocol"] = PROTOCOL
    plan["lineage"] = replay
    plan["select_index_sha256"] = canonical_hash(select)
    source_paths.update(raw / name for name in lineage.RAW_FILES)
    sources = {str(p): sha256_file(p) for p in sorted(source_paths)}
    code = list((repo / "src").rglob("*.py")) + list((repo / "data_preprocess").glob("*.py"))
    code += [config_path, repo / "scripts/bota_if/run_ali_stage15_ranking_v1.ps1"]
    code += [repo / name for name in config["bindings"]]
    contract = {"schema": SCHEMA, "protocol": PROTOCOL, "config_sha256": sha256_file(config_path),
                "candidate_sha256": canonical_hash(plan), "endpoints": entries, "dependencies": dependencies,
                "input_sha256": sources, "source_sha256": {str(p): sha256_file(p) for p in sorted(set(code))},
                "test_accessed": False}
    return config, benchmarks, plan, contract, users, movies


def verify_inputs(contract, workspace):
    for section in ("input_sha256", "source_sha256"):
        for name, expected in contract[section].items():
            if sha256_file(Path(name)) != expected:
                raise ValueError(f"input/code changed during run: {name}")
    for dep in contract["dependencies"].values():
        path = safe_path(workspace, dep["path"])
        if (directory_hash(path) if path.is_dir() else sha256_file(path)) != dep["sha256"]:
            raise ValueError("shared model dependency changed during run")


def publish(stage, destination):
    if destination.exists():
        raise FileExistsError(destination)
    atomic_text(stage / "COMPLETED", SCHEMA + "\n")
    atomic_json(stage / "manifest.json", {"schema": SCHEMA, "test_accessed": False,
        "files": {p.name: sha256_file(p) for p in sorted(stage.iterdir()) if p.is_file()}})
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.rename(stage, destination)


def verify_bundle(path):
    manifest = read(path / "manifest.json")
    if manifest["schema"] != SCHEMA or (path / "COMPLETED").read_text(encoding="utf-8") != SCHEMA + "\n":
        raise ValueError("invalid Stage 15 bundle")
    if {p.name for p in path.iterdir()} != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("Stage 15 bundle inventory mismatch")
    for name, expected in manifest["files"].items():
        if sha256_file(path / name) != expected:
            raise ValueError(f"Stage 15 artifact changed: {name}")
    s9._assert_no_test_access(manifest)


def stage_path(root):
    path = root / ".work" / uuid.uuid4().hex[:10]
    path.mkdir(parents=True)
    return path


def freeze(root, plan, contract):
    target = root / "protocol"
    if target.exists():
        verify_bundle(target)
        if read(target / "contract.json") != contract or read(target / "candidates.json") != plan:
            raise ValueError("frozen Stage 15 changed; no overwrite/reselection")
    else:
        stage = stage_path(root)
        atomic_json(stage / "candidates.json", plan)
        atomic_json(stage / "contract.json", contract)
        publish(stage, target)


class RankingDataset(JsonPromptDataset):
    """All candidates for a user have the same pre-Development context.

    Constant dummy output ensures relevance is never passed to the model. The
    scorer consumes only the first decoder position. A constant dummy target
    preserves the legacy decoder length/numerical path without leaking relevance.
    """
    def __init__(self, plan, tokenizer, users, movies):
        self.tokenizer = tokenizer
        self.records = []
        self.encoded = []
        for group in plan["users"]:
            for item in group["items"]:
                row = lineage.RawLineage("validation", 0, group["user"], "ratings.dat", 0,
                    item, 0, 0, group["anchor_timestamp"], group["history_ids"], group["history_ratings"])
                self.records.append({"input": lineage.render_record(row, users, movies)["input"], "output": "Yes."})

    def __getitem__(self, index):
        # Cache tokenization across all frozen endpoints. Never silently truncate
        # away the candidate title/question; reject an incompatible prompt.
        if not self.encoded:
            self.encoded = [None] * len(self.records)
        if self.encoded[index] is None:
            ids = self.tokenizer.encode(self.records[index]["input"], truncation=False)
            if len(ids) > 512:
                raise ValueError(f"ranking prompt exceeds 512 tokens at candidate {index}; no silent truncation")
            self.encoded[index] = torch.tensor(ids, dtype=torch.long)
        return {"sample_id": index, "input_ids": self.encoded[index],
                "target_ids": torch.tensor(self.tokenizer.encode("Yes."), dtype=torch.long)}

    def validate_tokens(self):
        for i in range(len(self)):
            self[i]
        return {"max_input_tokens": max(len(v) for v in self.encoded), "prompts": len(self)}


def score_model(model, dataset, device, progress, component=0):
    masses = []
    with torch.inference_mode():
        for start in range(0, len(dataset), PROTOCOL["inference_batch_size"]):
            end = min(start + PROTOCOL["inference_batch_size"], len(dataset))
            batch = dataset.collate_fn([dataset[i] for i in range(start, end)])
            inputs = batch["input_ids"].to(device)
            logits = model(input_ids=inputs, labels=batch["target_ids"].to(device)).logits[:, 0]
            # Same legacy no-attention-mask convention as existing binary eval.
            pair = torch.softmax(logits, dim=-1)[:, [465, 2163]].double().cpu().numpy()
            if not np.isfinite(pair).all() or (pair.sum(1) <= 0).any():
                raise RuntimeError("invalid Yes/No token masses")
            masses.append(pair)
            if start == 0 or end == len(dataset) or start % 2000 == 0:
                atomic_json(progress, {"component": component, "candidates_done": end,
                                      "candidates_total": len(dataset), "test_accessed": False})
                print(f"[Stage15] component={component} candidates={end}/{len(dataset)}", flush=True)
    return np.concatenate(masses)


def predict(workspace, benchmark, entry, dataset, progress):
    path = safe_path(workspace, entry["path"])
    if directory_hash(path) != entry["directory_sha256"]:
        raise ValueError("endpoint changed before inference")
    device = torch.device("cuda:0")
    if entry["kind"] == "fixed_ab":
        components = [(None, 1.)]
    elif entry["kind"] == "receraser_ensemble":
        weights = read(path / "aggregation.json")["weights"]
        if not weights or any(not math.isfinite(w) or w < 0 for w in weights) or not math.isclose(sum(weights), 1., abs_tol=1e-5):
            raise ValueError("invalid ensemble weights")
        components = list(enumerate(weights))
    else:
        raise ValueError("unknown endpoint format")
    total = np.zeros((len(dataset), 2), dtype=np.float64)
    for shard, weight in components:
        model = None
        try:
            if shard is None:
                state = torch.load(path / "adapter_model.pt", map_location="cpu", weights_only=True)
                model = evaluation.load_legacy_model(workspace / benchmark["source"]["original_checkpoint"])
                names, parameters = evaluation.install_fixed_ab_coordinate(model, state["A"], 32)
                with torch.no_grad():
                    for name, parameter in zip(names, parameters):
                        parameter.copy_(state["B"][name].to(parameter))
                del state, parameters
            else:
                model = evaluation.clean_lora_model(workspace / benchmark["source"]["original_checkpoint"],
                    {"r": 16, "lora_alpha": 32, "lora_dropout": .05, "target_modules": ["q", "v"]}, torch.device("cpu"))
                state = torch.load(path / f"models/shard_{shard:02d}/final_state.pt", map_location="cpu", weights_only=True)
                evaluation.load_trainable_state(model, state, "lora_adapter")
                del state
            model = model.to(device).eval()
            total += weight * score_model(model, dataset, device, progress, shard or 0)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
    if directory_hash(path) != entry["directory_sha256"]:
        raise ValueError("endpoint changed during inference")
    return (total[:, 1] / total.sum(1)).tolist()


def run_cell(root, workspace, benchmark, plan, contract, entry, dataset):
    destination = root / "cells" / f"s{entry['seed']}_{entry['scenario']}_{entry['method']}"
    identity = {"contract_sha256": canonical_hash(contract), "candidate_sha256": canonical_hash(plan), "endpoint": entry}
    if destination.exists():
        verify_bundle(destination)
        if read(destination / "identity.json") != identity:
            raise ValueError("completed cell identity mismatch")
        print(f"[Stage15] verified reuse {destination.name}", flush=True)
        return
    stage = stage_path(root)
    atomic_json(stage / "identity.json", identity)
    begun = time.perf_counter()
    print(f"[Stage15] start {destination.name} work={stage}", flush=True)
    try:
        scores = predict(workspace, benchmark, entry, dataset, stage / "progress.json")
        metrics = metrics_for(plan, scores, set(entry["users"]))
        atomic_json(stage / "scores.json", {"candidate_sha256": canonical_hash(plan), "scores": scores, "test_accessed": False})
        atomic_json(stage / "metrics.json", {**metrics, "seed": entry["seed"], "scenario": entry["scenario"],
            "method": entry["method"], "wall_seconds": time.perf_counter()-begun, "optimizer_updates": 0, "test_accessed": False})
        verify_inputs(contract, workspace)
        publish(stage, destination)
    except Exception as error:
        atomic_json(stage / "failure.json", {"type": type(error).__name__, "message": str(error), "test_accessed": False})
        raise RuntimeError(f"Stage 15 cell failed; diagnostics kept at {stage}. Completed cells remain reusable.") from error


def summarize(root, plan, contract):
    rows, missing = [], []
    for entry in contract["endpoints"]:
        name = f"s{entry['seed']}_{entry['scenario']}_{entry['method']}"
        path = root / "cells" / name
        if not path.exists():
            missing.append(name)
            continue
        verify_bundle(path)
        identity = read(path / "identity.json")
        if identity != {"contract_sha256": canonical_hash(contract), "candidate_sha256": canonical_hash(plan), "endpoint": entry}:
            raise ValueError("summary cell binding mismatch")
        scores = read(path / "scores.json")
        if scores["candidate_sha256"] != canonical_hash(plan):
            raise ValueError("score candidate order mismatch")
        recomputed = metrics_for(plan, scores["scores"], set(entry["users"]))
        saved = read(path / "metrics.json")
        if any(saved[key] != value for key, value in recomputed.items()):
            raise ValueError("ranking metric recomputation mismatch")
        for group in saved["aggregates"]:
            rows.append({"seed": entry["seed"], "scenario": entry["scenario"], "method": entry["method"], **group})
    # Complete three-seed comparisons only. Partial rows remain visible, never
    # mislabelled as full results or used to select a new response scale.
    aggregates = []
    for sid in SCENARIOS:
        for method in METHODS:
            for group in ("all", "requested", "retained"):
                cells = [r for r in rows if r["scenario"] == sid and r["method"] == method and r["group"] == group]
                if {r["seed"] for r in cells} != set(SEEDS):
                    continue
                item = {"scenario": sid, "method": method, "group": group, "seeds": 3, "users": cells[0]["users"]}
                for key in (f"{m}@{k}" for m in ("ndcg", "recall") for k in KS):
                    values = [r[key] for r in cells]
                    item[key] = {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1))} if all(v is not None for v in values) else None
                aggregates.append(item)
    comparisons = []
    for row in rows:
        original = next((r for r in rows if r["seed"] == row["seed"] and r["scenario"] == row["scenario"] and r["group"] == row["group"] and r["method"] == "Original"), None)
        exact = next((r for r in rows if r["seed"] == row["seed"] and r["scenario"] == row["scenario"] and r["group"] == row["group"] and r["method"] == "Exact-Masked"), None)
        if original and exact and row["ndcg@10"] is not None:
            comparisons.append({"seed": row["seed"], "scenario": row["scenario"], "group": row["group"], "method": row["method"],
                "ndcg10_delta_vs_original": row["ndcg@10"]-original["ndcg@10"],
                "ndcg10_delta_vs_exact": row["ndcg@10"]-exact["ndcg@10"]})
    result = {"status": "PARTIAL" if missing else "COMPLETED", "missing": missing, "rows": rows,
              "three_seed": aggregates, "comparisons": comparisons, "lambda_frozen": 64,
              "hyperparameter_reselection": False, "finaltest_authorized": False, "test_accessed": False}
    if not missing:
        target = root / "summary"
        if target.exists():
            verify_bundle(target)
            if read(target / "summary.json") != result:
                raise ValueError("summary changed")
        else:
            stage = stage_path(root)
            atomic_json(stage / "summary.json", result)
            lines = ["# Stage 15: frozen sampled Development ranking", "",
                "Lambda=64 frozen; no training, selection or FinalTest access. SISA has no eligible frozen checkpoint.",
                "All eligible Dev-Select positives + 100 deterministic unobserved candidates; not full-catalogue NDCG.",
                "Unobserved relevance is assumed zero, not known negative. Same pre-Development user history for all candidates.",
                "Development was previously exposed; this is a supplementary descriptive utility check, not an untouched holdout.",
                "Seed SD is descriptive. Ranking preservation is not evidence of privacy forgetting.", "",
                "| Request | Method | Group | Users | NDCG@10 mean | SD | Recall@10 mean |",
                "|---|---|---|---:|---:|---:|---:|"]
            for r in aggregates:
                n, rc = r["ndcg@10"], r["recall@10"]
                lines.append(f"| {r['scenario']} | {r['method']} | {r['group']} | {r['users']} | " +
                    (f"{n['mean']:.6f} | {n['sample_sd']:.6f} | {rc['mean']:.6f} |" if n else "NA | NA | NA |"))
            atomic_text(stage / "report.md", "\n".join(lines) + "\n")
            publish(stage, target)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--mode", choices=("Preflight", "SyntheticDryRun", "Full", "Analyze"), default="Preflight")
    parser.add_argument("--seed", choices=("All", "41", "42", "43"), default="All")
    parser.add_argument("--scenario", choices=("All", *SCENARIOS), default="All")
    parser.add_argument("--method", choices=("All", *METHODS), default="All")
    args = parser.parse_args()
    repo, workspace = args.repo.resolve(), args.workspace.resolve()
    config_path = repo / "configs/ali_stage15_ranking_v1.json"
    if args.mode == "SyntheticDryRun":
        load_config(repo, config_path)
        assert user_metrics([1, 2], [1], [.9, .1])["ndcg@5"] == 1
        print(json.dumps({"status": "SYNTHETIC_PASS", "test_accessed": False, "model_loaded": False}))
        return
    root = workspace / ROOT
    with s14.execution_lock(root):
        config, benchmarks, plan, contract, users, movies = prepare(repo, workspace, config_path)
        print(f"[Stage15] candidates={plan['candidate_count']} eligible_users={len(plan['users'])} endpoint_cells={len(contract['endpoints'])}", flush=True)
        if args.mode == "Preflight":
            covered = {r["user"] for r in plan["users"]}
            request_coverage = {f"s{e['seed']}_{e['scenario']}": {
                "requested": len(e["users"]), "covered": len(set(e["users"]) & covered),
                "missing_users": sorted(set(e["users"]) - covered)}
                for e in contract["endpoints"] if e["method"] == "BOTA64"}
            result = {"status": "PREFLIGHT_PASS", "eligible_users": len(plan["users"]),
                      "all_development_users": len(plan["coverage"]), "candidate_count": plan["candidate_count"],
                      "endpoint_cells": len(contract["endpoints"]), "candidate_sha256": canonical_hash(plan),
                      "request_coverage": request_coverage,
                      "lambda_frozen": 64, "model_loaded": False, "test_accessed": False}
        else:
            if args.mode == "Analyze" and not (root / "protocol").exists():
                raise FileNotFoundError("Stage 15 Full has not frozen its protocol")
            freeze(root, plan, contract)
            if args.mode == "Full":
                if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                    raise RuntimeError("requires exactly one CUDA GPU")
                free, total = torch.cuda.mem_get_info()
                if free / 2**30 < 10:
                    raise RuntimeError("requires at least 10 GiB free GPU memory")
                torch.cuda.set_per_process_memory_fraction(min(.88, 14 / (total / 2**30)))
                tokenizer_path = safe_path(workspace, contract["dependencies"]["pretrained_model_directory"]["path"])
                tokenizer = T5Tokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
                if tokenizer.encode("Yes.")[0] != 2163 or tokenizer.encode("No.")[0] != 465:
                    raise ValueError("Yes/No token identity changed")
                dataset = RankingDataset(plan, tokenizer, users, movies)
                print(f"[Stage15] token audit: {dataset.validate_tokens()}", flush=True)
                previous = Path.cwd()
                try:
                    os.chdir(workspace)
                    for entry in sorted(contract["endpoints"], key=lambda e: (e["seed"], e["scenario"], METHODS.index(e["method"]))):
                        if args.seed != "All" and entry["seed"] != int(args.seed): continue
                        if args.scenario != "All" and entry["scenario"] != args.scenario: continue
                        if args.method != "All" and entry["method"] != args.method: continue
                        run_cell(root, workspace, benchmarks[entry["seed"]], plan, contract, entry, dataset)
                finally:
                    os.chdir(previous)
            summary = summarize(root, plan, contract)
            result = {"status": summary["status"], "missing_cells": len(summary["missing"]), "lambda_frozen": 64,
                      "output_root": str(root), "optimizer_updates": 0, "test_accessed": False}
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
