"""Prepare a deterministic Development-only Amazon Movies and TV screen.

The downloaded review file is streamed once.  A hash-based 5% user sample is
formed before ratings are inspected, and only the most recent interactions
needed by the small screen are retained in memory. Official item metadata is
joined by parent_asin so prompts use real movie and television titles.
"""
from __future__ import annotations

import argparse
import gzip
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

from src.paper_if_a2.common import atomic_json, canonical_hash, safe_run_name, sha256_file

SCHEMA = "bota-amazon-movies-tv-small-preparation-v1"
MARKER = "BOTA_AMAZON_MOVIES_TV_SMALL_V1_PREPARED"


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"schema", "test_access_policy", "output_root", "source", "protocol", "privacy", "scientific_scope"}
    if not isinstance(value, dict) or set(value) != required or value["schema"] != SCHEMA or value["test_access_policy"] != "forbidden":
        raise ValueError("invalid Amazon Movies and TV preparation config")
    expected = {
        "seed": 42, "hash_sample_modulus": 1000, "hash_sample_buckets": 50,
        "selected_users": 256, "minimum_explicit_interactions": 25,
        "history_length": 10, "development_targets_per_user": 5,
        "maximum_train_targets_per_user": 20, "positive_rating_minimum": 4,
        "chronology": "timestamp_then_parent_asin_then_source_row_hash",
        "selection": "eligible_users_sha256_ordered",
    }
    if value["protocol"] != expected:
        raise ValueError("Amazon Movies and TV protocol changed")
    if value["privacy"] != {
        "persist_raw_user_ids": False, "persist_raw_item_ids": False,
        "persist_review_text": False, "persist_review_title": False,
        "persist_prompts": True,
    }:
        raise ValueError("Amazon Movies and TV privacy protocol changed")
    if value["scientific_scope"] != {
        "exploratory_small_scale_only": True,
        "real_item_titles_from_official_metadata": True,
        "development_only": True, "final_test_created": False,
        "test_accessed": False,
    }:
        raise ValueError("Amazon Movies and TV scientific scope changed")
    if set(value["source"]) != {"reviews", "metadata", "metadata_sha256"}:
        raise ValueError("Amazon Movies and TV source definition changed")
    for name in ("reviews", "metadata"):
        source = Path(value["source"][name])
        if any(part.lower() in {"test", "final_test", "finaltest"} for part in source.parts):
            raise ValueError("test-like source path forbidden")
    return value


def _digest(role: str, value: str, seed: int = 42) -> str:
    return hashlib.sha256(f"amazon-movies-tv:{seed}:{role}:{value}".encode("utf-8")).hexdigest()


def _sampled_user(user: str, modulus: int, buckets: int) -> bool:
    value = int.from_bytes(hashlib.blake2b(user.encode("utf-8"), digest_size=8).digest(), "big")
    return value % modulus < buckets


def _item_token(parent_asin: str) -> str:
    return "movie-or-tv-item-" + _digest("item", parent_asin)[:10]


def _metadata_titles(path: Path) -> tuple[dict[str, str], int]:
    titles: dict[str, str] = {}; rows = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid metadata JSON at line {line_number}") from error
            parent = str(row.get("parent_asin") or ""); title = " ".join(str(row.get("title") or "").split())
            if parent and title:
                titles[parent] = title[:240]
    if not titles:
        raise RuntimeError("Amazon Movies and TV metadata contains no titled items")
    return titles, rows


def _prompt(history: list[tuple[str, int]], target: str) -> str:
    rendered = "\n".join(
        f"{index}. {item} ({rating} {'star' if rating == 1 else 'stars'})"
        for index, (item, rating) in enumerate(history, 1)
    )
    return (
        "The user rated the following movie or television items in chronological order:\n"
        f"{rendered}\n"
        f"Based on this history, deduce whether the user will like ***{target}***.\n"
        "Higher ratings indicate stronger preference. You should ONLY tell me yes or no."
    )


def _record(rows: list[tuple[int, str, str, int]], target: int, positive_minimum: int) -> dict[str, str]:
    _, _, item, rating = rows[target]
    history = [(previous_item, score) for _, _, previous_item, score in rows[target - 10:target]]
    if len(history) != 10:
        raise RuntimeError("Amazon Movies and TV history length mismatch")
    return {"input": _prompt(history, item), "output": "Yes." if rating >= positive_minimum else "No."}


def prepare(root: Path, config_path: Path, dataset_name: str) -> dict[str, Any]:
    config = load_config(config_path); protocol = config["protocol"]
    dataset_name = safe_run_name(dataset_name); destination = root / config["output_root"] / dataset_name
    if destination.exists():
        raise FileExistsError(destination)
    source = root / config["source"]["reviews"]; metadata = root / config["source"]["metadata"]
    if not source.is_file() or not metadata.is_file():
        raise FileNotFoundError("Amazon Movies and TV reviews or metadata is missing")
    if sha256_file(metadata) != config["source"]["metadata_sha256"]:
        raise ValueError("Amazon Movies and TV metadata SHA mismatch")
    titles, metadata_rows = _metadata_titles(metadata)
    keep = protocol["history_length"] + protocol["development_targets_per_user"] + protocol["maximum_train_targets_per_user"]
    # Per-user min-heaps retain the latest `keep` valid interactions while the
    # multi-gigabyte JSONL file is read exactly once.
    heaps: dict[str, list[tuple[int, str, str, int]]] = defaultdict(list)
    source_rows = valid_rows = sampled_rows = 0; rating_counts: Counter[int] = Counter(); started = time.perf_counter()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            source_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid review JSON at line {line_number}") from error
            user = str(row.get("user_id") or ""); parent = str(row.get("parent_asin") or row.get("asin") or "")
            rating = row.get("rating"); timestamp = row.get("timestamp")
            if not user or parent not in titles or not isinstance(rating, (int, float)) or float(rating) not in {1., 2., 3., 4., 5.} or not isinstance(timestamp, int):
                continue
            valid_rows += 1
            if not _sampled_user(user, protocol["hash_sample_modulus"], protocol["hash_sample_buckets"]):
                continue
            sampled_rows += 1; score = int(rating); rating_counts[score] += 1
            row_hash = _digest("row", f"{line_number}:{user}:{parent}:{timestamp}")
            entry = (timestamp, row_hash, titles[parent], score); heap = heaps[user]
            if len(heap) < keep:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)
    eligible = [user for user, rows in heaps.items() if len(rows) >= protocol["minimum_explicit_interactions"]]
    eligible.sort(key=lambda user: _digest("select", user))
    if len(eligible) < protocol["selected_users"]:
        raise RuntimeError(f"only {len(eligible)} eligible users in deterministic 5% sample; need {protocol['selected_users']}")
    selected = eligible[:protocol["selected_users"]]
    pseudonyms = {user: index for index, user in enumerate(sorted(selected, key=lambda user: _digest("pseudonym", user, 43)), 1)}
    train: list[dict[str, str]] = []; development: list[dict[str, str]] = []
    train_users: list[int] = []; development_users: list[int] = []
    train_ids: list[str] = []; development_ids: list[str] = []; per_user_train: list[int] = []
    labels: Counter[str] = Counter()
    for user in selected:
        rows = sorted(heaps[user]); development_start = len(rows) - protocol["development_targets_per_user"]
        train_start = max(protocol["history_length"], development_start - protocol["maximum_train_targets_per_user"])
        train_targets = list(range(train_start, development_start)); development_targets = list(range(development_start, len(rows)))
        if not train_targets or len(development_targets) != protocol["development_targets_per_user"]:
            raise RuntimeError("invalid Amazon Movies and TV split")
        pseudo = pseudonyms[user]; per_user_train.append(len(train_targets)); user_token = _digest("user", user)
        for target in train_targets:
            value = _record(rows, target, protocol["positive_rating_minimum"]); train.append(value); train_users.append(pseudo)
            train_ids.append(_digest("train", f"{user_token}:{rows[target][1]}")); labels["train_yes" if value["output"] == "Yes." else "train_no"] += 1
        for target in development_targets:
            value = _record(rows, target, protocol["positive_rating_minimum"]); development.append(value); development_users.append(pseudo)
            development_ids.append(_digest("development", f"{user_token}:{rows[target][1]}")); labels["development_yes" if value["output"] == "Yes." else "development_no"] += 1
    if len(train) < 3200 or set(train_ids) & set(development_ids):
        raise RuntimeError(f"Amazon Movies and TV small screen needs at least 3,200 disjoint training examples; got {len(train)}")
    def reorder(values: list[Any], users: list[int], ids: list[str], role: str) -> tuple[list[Any], list[int], list[str]]:
        order = sorted(range(len(values)), key=lambda index: _digest(f"{role}-order", ids[index]))
        return [values[i] for i in order], [users[i] for i in order], [ids[i] for i in order]
    train, train_users, train_ids = reorder(train, train_users, train_ids, "train")
    development, development_users, development_ids = reorder(development, development_users, development_ids, "development")
    stage = destination.parent / ".work" / f"{dataset_name}.{uuid.uuid4().hex}.stage"; stage.mkdir(parents=True)
    try:
        (stage / "train.json").write_text(json.dumps(train, ensure_ascii=False, separators=(",", ":")), encoding="utf-8", newline="\n")
        (stage / "development.json").write_text(json.dumps(development, ensure_ascii=False, separators=(",", ":")), encoding="utf-8", newline="\n")
        atomic_json(stage / "train_user_ids.json", train_users); atomic_json(stage / "development_user_ids.json", development_users)
        lineage = {
            "schema": SCHEMA, "dataset_name": dataset_name,
            "source": {"reviews": config["source"]["reviews"], "bytes": source.stat().st_size, "rows": source_rows, "metadata": config["source"]["metadata"], "metadata_bytes": metadata.stat().st_size, "metadata_rows": metadata_rows, "metadata_sha256": config["source"]["metadata_sha256"], "metadata_titles": len(titles)},
            "sampling": {"user_hash_fraction": protocol["hash_sample_buckets"] / protocol["hash_sample_modulus"], "sampled_valid_rows": sampled_rows, "sampled_users": len(heaps), "eligible_users": len(eligible)},
            "selected_users": len(selected), "train_samples": len(train), "development_samples": len(development),
            "train_user_frequency": {"minimum": min(per_user_train), "maximum": max(per_user_train), "mean": sum(per_user_train) / len(per_user_train)},
            "label_counts": dict(labels), "sampled_rating_counts": dict(rating_counts),
            "train_sample_order_sha256": canonical_hash(train_ids), "development_sample_order_sha256": canonical_hash(development_ids),
            "anonymous_item_tokens": False, "real_item_titles": True, "raw_user_ids_persisted": False, "raw_item_ids_persisted": False,
            "review_text_persisted": False, "review_title_persisted": False,
            "preparation_wall_seconds": time.perf_counter() - started, "final_test_created": False, "test_accessed": False,
        }
        atomic_json(stage / "lineage.json", lineage); (stage / "PREPARED").write_text(MARKER + "\n", encoding="utf-8", newline="\n")
        files = ("train.json", "development.json", "train_user_ids.json", "development_user_ids.json", "lineage.json", "PREPARED")
        atomic_json(stage / "manifest.json", {"schema": SCHEMA, "files": {name: sha256_file(stage / name) for name in files}, "published_atomically": True, "test_accessed": False})
        destination.parent.mkdir(parents=True, exist_ok=True); os.replace(stage, destination)
    finally:
        if stage.exists(): shutil.rmtree(stage)
    return {"status": "PREPARED", "dataset_dir": str(destination), "train_samples": len(train), "development_samples": len(development), "selected_users": len(selected), "eligible_users": len(eligible), "label_counts": dict(labels), "preparation_wall_seconds": lineage["preparation_wall_seconds"], "test_accessed": False}


def analyze(root: Path, config_path: Path, dataset_name: str) -> dict[str, Any]:
    config = load_config(config_path); directory = root / config["output_root"] / safe_run_name(dataset_name)
    required = {"PREPARED", "development.json", "development_user_ids.json", "lineage.json", "manifest.json", "train.json", "train_user_ids.json"}
    if not directory.is_dir() or {path.name for path in directory.iterdir()} != required or (directory / "PREPARED").read_text(encoding="utf-8") != MARKER + "\n":
        raise ValueError("invalid Amazon Movies and TV prepared dataset")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"Amazon Movies and TV artifact mismatch: {name}")
    lineage = json.loads((directory / "lineage.json").read_text(encoding="utf-8"))
    return {"status": "PREPARED", "dataset_dir": str(directory), **{key: lineage[key] for key in ("selected_users", "train_samples", "development_samples", "train_user_frequency", "label_counts", "sampling", "preparation_wall_seconds")}, "test_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path.cwd()); parser.add_argument("--config", type=Path, default=Path("configs/bota_amazon_movies_small_prepare_v1.yaml")); parser.add_argument("--mode", choices=["Preflight", "SyntheticDryRun", "Prepare", "Analyze"], default="Preflight"); parser.add_argument("--dataset-name", default="amazon_movies_tv_titles_seed42_v2")
    args = parser.parse_args(); root = args.root.resolve(); config_path = (root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve(); config = load_config(config_path); source = root / config["source"]["reviews"]; metadata = root / config["source"]["metadata"]
    if args.mode == "Preflight":
        result = {"schema": SCHEMA, "source_exists": source.is_file(), "source_bytes": source.stat().st_size if source.is_file() else None, "metadata_exists": metadata.is_file(), "metadata_bytes": metadata.stat().st_size if metadata.is_file() else None, "model_loaded": False, "test_accessed": False}
    elif args.mode == "SyntheticDryRun":
        result = {"schema": SCHEMA, "real_data_read": False, "model_loaded": False, "test_accessed": False}
    elif args.mode == "Prepare": result = prepare(root, config_path, args.dataset_name)
    else: result = analyze(root, config_path, args.dataset_name)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()

