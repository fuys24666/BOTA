"""Build a disjoint new-user adaptation cohort for the Amazon screen."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import shutil
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from src.bota_short_benchmark.amazon_movies_small_prepare import (
    _digest, _metadata_titles, _record, _sampled_user,
)
from src.paper_if_a2.common import atomic_json, canonical_hash, safe_run_name, sha256_file

SCHEMA = "bota-amazon-movies-tv-newuser-preparation-v4"
MARKER = "BOTA_AMAZON_MOVIES_TV_NEWUSER_V4_PREPARED"


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != SCHEMA or value.get("test_access_policy") != "forbidden":
        raise ValueError("invalid Amazon new-user preparation config")
    protocol = value.get("protocol", {})
    expected = {
        "seed": 42, "hash_sample_modulus": 1000, "hash_sample_buckets": 50,
        "selected_users": 1024, "historical_users": 256, "adaptation_users": 768,
        "minimum_explicit_interactions": 25, "history_length": 10,
        "development_targets_per_user": 5, "maximum_train_targets_per_user": 20,
        "positive_rating_minimum": 4,
        "chronology": "timestamp_then_parent_asin_then_source_row_hash",
        "cohort_selection": "eligible_users_sha256_ordered_first_256_historical_next_768_new",
    }
    if protocol != expected or value.get("scientific_scope", {}).get("historical_and_adaptation_users_disjoint") is not True:
        raise ValueError("Amazon new-user protocol changed")
    return value


def _cohort_digest(users: list[str], role: str) -> str:
    return canonical_hash([hashlib.sha256(f"amazon-newuser-v4:{role}:{user}".encode()).hexdigest() for user in users])


def prepare(root: Path, config_path: Path, dataset_name: str) -> dict[str, Any]:
    config = load_config(config_path); protocol = config["protocol"]; source_config = config["source"]
    destination = root / config["output_root"] / safe_run_name(dataset_name)
    if destination.exists(): raise FileExistsError(destination)
    reviews = root / source_config["reviews"]; metadata = root / source_config["metadata"]
    historical_manifest = root / source_config["historical_prepared_manifest"]
    if not reviews.is_file() or not metadata.is_file() or not historical_manifest.is_file():
        raise FileNotFoundError("Amazon reviews, metadata, or historical cohort artifact is missing")
    if sha256_file(metadata) != source_config["metadata_sha256"] or sha256_file(historical_manifest) != source_config["historical_prepared_manifest_sha256"]:
        raise ValueError("Amazon metadata or historical cohort binding changed")
    titles, metadata_rows = _metadata_titles(metadata)
    keep = protocol["history_length"] + protocol["development_targets_per_user"] + protocol["maximum_train_targets_per_user"]
    heaps: dict[str, list[tuple[int, str, str, int]]] = defaultdict(list)
    source_rows = valid_rows = sampled_rows = 0; started = time.perf_counter()
    with reviews.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            source_rows += 1
            row = json.loads(line); user = str(row.get("user_id") or ""); parent = str(row.get("parent_asin") or row.get("asin") or "")
            rating = row.get("rating"); timestamp = row.get("timestamp")
            if not user or parent not in titles or not isinstance(rating, (int, float)) or float(rating) not in {1., 2., 3., 4., 5.} or not isinstance(timestamp, int): continue
            valid_rows += 1
            if not _sampled_user(user, protocol["hash_sample_modulus"], protocol["hash_sample_buckets"]): continue
            sampled_rows += 1; row_hash = _digest("row", f"{line_number}:{user}:{parent}:{timestamp}")
            entry = (timestamp, row_hash, titles[parent], int(rating)); heap = heaps[user]
            if len(heap) < keep: heapq.heappush(heap, entry)
            elif entry > heap[0]: heapq.heapreplace(heap, entry)
    eligible = sorted((user for user, rows in heaps.items() if len(rows) >= protocol["minimum_explicit_interactions"]), key=lambda user: _digest("select", user))
    if len(eligible) < protocol["selected_users"]: raise RuntimeError(f"need 1024 eligible users; found {len(eligible)}")
    selected = eligible[:protocol["selected_users"]]; historical = selected[:protocol["historical_users"]]; adaptation = selected[protocol["historical_users"]:]
    if len(adaptation) != protocol["adaptation_users"] or set(historical) & set(adaptation): raise RuntimeError("historical/adaptation cohort split failed")
    pseudonyms = {user: index for index, user in enumerate(sorted(adaptation, key=lambda user: _digest("pseudonym-v4", user)), 1)}
    train: list[dict[str, str]] = []; development: list[dict[str, str]] = []; train_users: list[int] = []; development_users: list[int] = []
    train_ids: list[str] = []; development_ids: list[str] = []; per_user: list[int] = []; labels: Counter[str] = Counter()
    for user in adaptation:
        rows = sorted(heaps[user]); development_start = len(rows) - protocol["development_targets_per_user"]
        train_start = max(protocol["history_length"], development_start - protocol["maximum_train_targets_per_user"])
        train_targets = list(range(train_start, development_start)); development_targets = list(range(development_start, len(rows)))
        pseudo = pseudonyms[user]; token = _digest("user-v4", user); per_user.append(len(train_targets))
        for target in train_targets:
            value = _record(rows, target, protocol["positive_rating_minimum"]); train.append(value); train_users.append(pseudo); train_ids.append(_digest("adapt-train-v4", f"{token}:{rows[target][1]}")); labels["train_" + value["output"].lower().strip(".")] += 1
        for target in development_targets:
            value = _record(rows, target, protocol["positive_rating_minimum"]); development.append(value); development_users.append(pseudo); development_ids.append(_digest("adapt-dev-v4", f"{token}:{rows[target][1]}")); labels["development_" + value["output"].lower().strip(".")] += 1
    if len(train) < 3200 or set(train_ids) & set(development_ids): raise RuntimeError("insufficient or overlapping adaptation rows")
    def reorder(values, users, ids, role):
        order = sorted(range(len(values)), key=lambda index: _digest(role, ids[index])); return [values[i] for i in order], [users[i] for i in order], [ids[i] for i in order]
    train, train_users, train_ids = reorder(train, train_users, train_ids, "adapt-train-order-v4")
    development, development_users, development_ids = reorder(development, development_users, development_ids, "adapt-dev-order-v4")
    stage = destination.parent / ".work" / f"{destination.name}.{uuid.uuid4().hex}.stage"; stage.mkdir(parents=True)
    try:
        (stage / "train.json").write_text(json.dumps(train, ensure_ascii=False, separators=(",", ":")), encoding="utf-8", newline="\n")
        (stage / "development.json").write_text(json.dumps(development, ensure_ascii=False, separators=(",", ":")), encoding="utf-8", newline="\n")
        atomic_json(stage / "train_user_ids.json", train_users); atomic_json(stage / "development_user_ids.json", development_users)
        lineage = {"schema": SCHEMA, "dataset_name": destination.name, "selected_users": len(selected), "historical_users": len(historical), "adaptation_users": len(adaptation), "historical_adaptation_disjoint": True, "historical_cohort_digest": _cohort_digest(historical, "historical"), "adaptation_cohort_digest": _cohort_digest(adaptation, "adaptation"), "historical_prepared_manifest_sha256": sha256_file(historical_manifest), "train_samples": len(train), "development_samples": len(development), "train_user_frequency": {"minimum": min(per_user), "maximum": max(per_user), "mean": sum(per_user) / len(per_user)}, "label_counts": dict(labels), "source": {"review_rows": source_rows, "valid_rows": valid_rows, "sampled_rows": sampled_rows, "metadata_rows": metadata_rows}, "train_sample_order_sha256": canonical_hash(train_ids), "development_sample_order_sha256": canonical_hash(development_ids), "raw_user_ids_persisted": False, "final_test_created": False, "test_accessed": False, "preparation_wall_seconds": time.perf_counter() - started}
        atomic_json(stage / "lineage.json", lineage); (stage / "PREPARED").write_text(MARKER + "\n", encoding="utf-8", newline="\n")
        files = ("train.json", "development.json", "train_user_ids.json", "development_user_ids.json", "lineage.json", "PREPARED")
        atomic_json(stage / "manifest.json", {"schema": SCHEMA, "files": {name: sha256_file(stage / name) for name in files}, "published_atomically": True, "test_accessed": False})
        destination.parent.mkdir(parents=True, exist_ok=True); os.replace(stage, destination)
    finally:
        if stage.exists(): shutil.rmtree(stage)
    return {"status": "PREPARED", "dataset_dir": str(destination), "train_samples": len(train), "development_samples": len(development), "historical_users": len(historical), "adaptation_users": len(adaptation), "test_accessed": False}


def analyze(root: Path, config_path: Path, dataset_name: str) -> dict[str, Any]:
    config = load_config(config_path); directory = root / config["output_root"] / safe_run_name(dataset_name)
    required = {"PREPARED", "development.json", "development_user_ids.json", "lineage.json", "manifest.json", "train.json", "train_user_ids.json"}
    if not directory.is_dir() or {path.name for path in directory.iterdir()} != required or (directory / "PREPARED").read_text(encoding="utf-8") != MARKER + "\n": raise ValueError("invalid Amazon new-user dataset")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        if sha256_file(directory / name) != digest: raise ValueError(f"artifact mismatch: {name}")
    lineage = json.loads((directory / "lineage.json").read_text(encoding="utf-8"))
    return {"status": "PREPARED", "dataset_dir": str(directory), **{key: lineage[key] for key in ("historical_users", "adaptation_users", "historical_adaptation_disjoint", "train_samples", "development_samples", "train_user_frequency", "label_counts", "preparation_wall_seconds")}, "test_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path.cwd()); parser.add_argument("--config", type=Path, default=Path("configs/bota_amazon_movies_newuser_prepare_v4.yaml")); parser.add_argument("--mode", choices=["Preflight", "SyntheticDryRun", "Prepare", "Analyze"], default="Preflight"); parser.add_argument("--dataset-name", default="amazon_movies_tv_newusers_seed42_v4")
    args = parser.parse_args(); root = args.root.resolve(); path = (root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve(); config = load_config(path)
    if args.mode == "Preflight":
        result = {"schema": SCHEMA, "reviews_exist": (root / config["source"]["reviews"]).is_file(), "metadata_exists": (root / config["source"]["metadata"]).is_file(), "historical_artifact_exists": (root / config["source"]["historical_prepared_manifest"]).is_file(), "model_loaded": False, "test_accessed": False}
    elif args.mode == "SyntheticDryRun": result = {"schema": SCHEMA, "real_data_read": False, "model_loaded": False, "test_accessed": False}
    elif args.mode == "Prepare": result = prepare(root, path, args.dataset_name)
    else: result = analyze(root, path, args.dataset_name)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()

