"""Deterministic, Development-only Goodreads Comics preparation for BOTA."""
from __future__ import annotations

import argparse
import email.utils
import gzip
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from src.paper_if_a2.common import atomic_json, canonical_hash, safe_run_name, sha256_file

SCHEMA = "bota-goodreads-comics-preparation-v1"
MARKER = "BOTA_GOODREADS_COMICS_V1_PREPARED"


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"schema", "test_access_policy", "output_root", "source", "protocol", "privacy"}
    if not isinstance(value, dict) or set(value) != required or value["schema"] != SCHEMA:
        raise ValueError("invalid Goodreads preparation config")
    if value["test_access_policy"] != "forbidden":
        raise ValueError("Goodreads preparation must be Development-only")
    expected = {
        "seed": 42, "selected_users": 2000, "minimum_explicit_interactions": 21,
        "history_length": 10, "development_targets_per_user": 10,
        "maximum_train_targets_per_user": 40, "positive_rating_minimum": 4,
        "chronology": "date_updated_then_review_id", "selection": "sha256_seeded_user_sample",
    }
    if value["protocol"] != expected:
        raise ValueError("Goodreads protocol changed")
    if value["privacy"] != {"persist_raw_user_ids": False, "persist_review_text": False, "persist_prompts": True}:
        raise ValueError("Goodreads privacy protocol changed")
    for name in ("books", "interactions"):
        source = Path(value["source"][name])
        if any(part.lower() in {"test", "final_test", "finaltest"} for part in source.parts):
            raise ValueError("test-like source path forbidden")
    return value


def _iter_gzip_json(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                yield line_number, json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValueError(f"invalid gzip JSON at {path}:{line_number}") from error


def _timestamp(value: str) -> int:
    parsed = email.utils.parsedate_to_datetime(value)
    return int(parsed.timestamp())


def _user_digest(seed: int, user: str) -> str:
    return hashlib.sha256(f"goodreads-comics:{seed}:{user}".encode()).hexdigest()


def _prompt(history: list[tuple[str, int]], title: str) -> str:
    rendered = [f"{index}. {name} ({rating} {'star' if rating == 1 else 'stars'})" for index, (name, rating) in enumerate(history)]
    return (
        "The user read the following books in chronological order and rated them:\n"
        f"{rendered}\n"
        f"Based on this reading history, deduce whether the user will like the book ***{title}***.\n"
        "Higher ratings indicate stronger preference. You should ONLY tell me yes or no."
    )


def _record(interactions: list[tuple[int, str, str, int]], target: int, titles: dict[str, str]) -> dict[str, str]:
    _, review_id, book_id, rating = interactions[target]
    history = [(titles[book], score) for _, _, book, score in interactions[target - 10:target]]
    if len(history) != 10:
        raise RuntimeError("Goodreads history length mismatch")
    return {"input": _prompt(history, titles[book_id]), "output": "Yes." if rating >= 4 else "No."}


def prepare(root: Path, config_path: Path, dataset_name: str) -> dict[str, Any]:
    config = load_config(config_path); protocol = config["protocol"]; dataset_name = safe_run_name(dataset_name)
    destination = root / config["output_root"] / dataset_name
    if destination.exists():
        raise FileExistsError(destination)
    books_path = root / config["source"]["books"]; interactions_path = root / config["source"]["interactions"]
    if not books_path.is_file() or not interactions_path.is_file():
        raise FileNotFoundError("Goodreads Comics source files are incomplete")

    titles: dict[str, str] = {}
    for _, row in _iter_gzip_json(books_path):
        book_id = str(row.get("book_id", "")); title = str(row.get("title_without_series") or row.get("title") or "").strip()
        if book_id and title:
            titles[book_id] = " ".join(title.split())
    if not titles:
        raise RuntimeError("Goodreads book metadata is empty")

    counts: Counter[str] = Counter(); explicit_rows = 0
    for _, row in _iter_gzip_json(interactions_path):
        rating = int(row.get("rating") or 0); user = str(row.get("user_id", "")); book = str(row.get("book_id", ""))
        if rating in {1, 2, 3, 4, 5} and user and book in titles:
            counts[user] += 1; explicit_rows += 1
    eligible = [user for user, count in counts.items() if count >= protocol["minimum_explicit_interactions"]]
    eligible.sort(key=lambda user: _user_digest(protocol["seed"], user))
    selected = eligible[:protocol["selected_users"]]
    if len(selected) != protocol["selected_users"]:
        raise RuntimeError("insufficient eligible Goodreads users")
    selected_set = set(selected)

    interactions: dict[str, list[tuple[int, str, str, int]]] = defaultdict(list)
    seen_reviews: set[str] = set()
    for line_number, row in _iter_gzip_json(interactions_path):
        user = str(row.get("user_id", "")); rating = int(row.get("rating") or 0); book = str(row.get("book_id", ""))
        if user not in selected_set or rating not in {1, 2, 3, 4, 5} or book not in titles:
            continue
        review = str(row.get("review_id") or f"line-{line_number}")
        review_key = f"{user}:{review}"
        if review_key in seen_reviews:
            raise RuntimeError("duplicate Goodreads review id")
        seen_reviews.add(review_key)
        interactions[user].append((_timestamp(str(row["date_updated"])), review, book, rating))

    pseudonyms = {user: index for index, user in enumerate(sorted(selected, key=lambda user: _user_digest(protocol["seed"] + 1, user)), 1)}
    train: list[dict[str, str]] = []; development: list[dict[str, str]] = []
    train_users: list[int] = []; development_users: list[int] = []; train_sample_ids = []; development_sample_ids = []
    train_counts: dict[int, int] = {}; label_counts = {"train_yes": 0, "train_no": 0, "development_yes": 0, "development_no": 0}
    for user in selected:
        rows = sorted(interactions[user], key=lambda row: (row[0], row[1]))
        if len(rows) < protocol["minimum_explicit_interactions"]:
            raise RuntimeError("selected user count changed between passes")
        development_start = len(rows) - protocol["development_targets_per_user"]
        train_start = max(protocol["history_length"], development_start - protocol["maximum_train_targets_per_user"])
        train_targets = list(range(train_start, development_start)); development_targets = list(range(development_start, len(rows)))
        if not train_targets or len(development_targets) != 10:
            raise RuntimeError("invalid per-user chronological split")
        pseudo = pseudonyms[user]; train_counts[pseudo] = len(train_targets)
        for target in train_targets:
            value = _record(rows, target, titles); train.append(value); train_users.append(pseudo)
            label_counts["train_yes" if value["output"] == "Yes." else "train_no"] += 1
            train_sample_ids.append(hashlib.sha256(f"train:{_user_digest(42,user)}:{rows[target][1]}".encode()).hexdigest())
        for target in development_targets:
            value = _record(rows, target, titles); development.append(value); development_users.append(pseudo)
            label_counts["development_yes" if value["output"] == "Yes." else "development_no"] += 1
            development_sample_ids.append(hashlib.sha256(f"development:{_user_digest(42,user)}:{rows[target][1]}".encode()).hexdigest())
    if len(development) != 20_000 or len(train) < 3_200 or set(train_sample_ids) & set(development_sample_ids):
        raise RuntimeError("Goodreads split size or disjointness gate failed")
    # Stable outcome-blind ordering keeps users interleaved while preserving their internal chronology.
    train_order = sorted(range(len(train)), key=lambda index: hashlib.sha256(f"train-order:{protocol['seed']}:{train_sample_ids[index]}".encode()).hexdigest())
    dev_order = sorted(range(len(development)), key=lambda index: hashlib.sha256(f"dev-order:{protocol['seed']}:{development_sample_ids[index]}".encode()).hexdigest())
    train = [train[index] for index in train_order]; train_users = [train_users[index] for index in train_order]; train_sample_ids = [train_sample_ids[index] for index in train_order]
    development = [development[index] for index in dev_order]; development_users = [development_users[index] for index in dev_order]; development_sample_ids = [development_sample_ids[index] for index in dev_order]

    stage = destination.parent / ".work" / f"{dataset_name}.{uuid.uuid4().hex}.stage"; stage.mkdir(parents=True)
    try:
        (stage / "train.json").write_text(json.dumps(train, ensure_ascii=False, separators=(",", ":")), encoding="utf-8", newline="\n")
        (stage / "development.json").write_text(json.dumps(development, ensure_ascii=False, separators=(",", ":")), encoding="utf-8", newline="\n")
        atomic_json(stage / "train_user_ids.json", train_users); atomic_json(stage / "development_user_ids.json", development_users)
        lineage = {
            "schema": SCHEMA, "dataset_name": dataset_name, "seed": protocol["seed"],
            "source": {"books": {"path": config["source"]["books"], "sha256": sha256_file(books_path)}, "interactions": {"path": config["source"]["interactions"], "sha256": sha256_file(interactions_path)}},
            "explicit_source_rows": explicit_rows, "metadata_books": len(titles), "eligible_users": len(eligible), "selected_users": len(selected),
            "train_samples": len(train), "development_samples": len(development), "train_user_frequency": {"minimum": min(train_counts.values()), "maximum": max(train_counts.values()), "mean": sum(train_counts.values()) / len(train_counts)},
            "label_counts": label_counts, "train_sample_order_sha256": canonical_hash(train_sample_ids), "development_sample_order_sha256": canonical_hash(development_sample_ids),
            "raw_user_ids_persisted": False, "review_text_persisted": False, "test_accessed": False,
        }
        atomic_json(stage / "lineage.json", lineage); (stage / "PREPARED").write_text(MARKER + "\n", encoding="utf-8", newline="\n")
        files = ("train.json", "development.json", "train_user_ids.json", "development_user_ids.json", "lineage.json", "PREPARED")
        atomic_json(stage / "manifest.json", {"schema": SCHEMA, "files": {name: sha256_file(stage / name) for name in files}, "published_atomically": True, "test_accessed": False})
        destination.parent.mkdir(parents=True, exist_ok=True); os.replace(stage, destination)
    finally:
        if stage.exists(): shutil.rmtree(stage)
    return {"status": "PREPARED", "dataset_dir": str(destination), "train_samples": len(train), "development_samples": len(development), "selected_users": len(selected), "label_counts": label_counts, "test_accessed": False}


def analyze(root: Path, config_path: Path, dataset_name: str) -> dict[str, Any]:
    config = load_config(config_path); directory = root / config["output_root"] / safe_run_name(dataset_name)
    required = {"PREPARED", "development.json", "development_user_ids.json", "lineage.json", "manifest.json", "train.json", "train_user_ids.json"}
    if not directory.is_dir() or {path.name for path in directory.iterdir()} != required or (directory / "PREPARED").read_text(encoding="utf-8") != MARKER + "\n":
        raise ValueError("invalid Goodreads prepared dataset")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if sha256_file(directory / name) != expected: raise ValueError(f"Goodreads artifact mismatch: {name}")
    lineage = json.loads((directory / "lineage.json").read_text(encoding="utf-8"))
    return {"status": "PREPARED", "dataset_dir": str(directory), "train_samples": lineage["train_samples"], "development_samples": lineage["development_samples"], "selected_users": lineage["selected_users"], "label_counts": lineage["label_counts"], "test_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path.cwd()); parser.add_argument("--config", type=Path, default=Path("configs/goodreads_comics_bota_v1.yaml")); parser.add_argument("--mode", choices=["Preflight", "SyntheticDryRun", "Prepare", "Analyze"], default="Preflight"); parser.add_argument("--dataset-name", default="")
    args = parser.parse_args(); root = args.root.resolve(); config_path = (root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve(); config = load_config(config_path)
    if not args.dataset_name: parser.error("DatasetName is required")
    if args.mode == "Preflight": result = {"schema": SCHEMA, "dataset_name": args.dataset_name, "sources_exist": all((root / config["source"][name]).is_file() for name in ("books", "interactions")), "model_loaded": False, "test_accessed": False}
    elif args.mode == "SyntheticDryRun": result = {"schema": SCHEMA, "real_data_read": False, "model_loaded": False, "test_accessed": False}
    elif args.mode == "Prepare": result = prepare(root, config_path, args.dataset_name)
    else: result = analyze(root, config_path, args.dataset_name)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
