"""Frozen request registry and materialization for the short benchmark."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
import torch

from src.bota_if import p1_trajectory_transport_audit as p1
from src.diagnostics.ml1m_development_protocol import reconstruct_authoritative_rows
from src.paper_if_a2.common import atomic_json, canonical_hash, directory_hash, safe_run_name, sha256_file
from src.paper_baselines.partitioned import balanced_similarity_partition, deletion_plan, sisa_partition, text_features

SCHEMA = "bota-short-benchmark-v1"
MARKER = "BOTA_SHORT_BENCHMARK_V1_PREPARED"


def source_sha256(path: Path) -> str:
    return directory_hash(path) if path.is_dir() else sha256_file(path)


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"schema", "test_access_policy", "output_root", "source", "protocol", "coordinate", "optimizer", "methods", "evaluation", "runtime"}
    if not isinstance(value, dict) or set(value) != required or value["schema"] != SCHEMA or value["test_access_policy"] != "forbidden":
        raise ValueError("invalid BOTA short benchmark config")
    protocol = value["protocol"]
    if protocol["optimizer_steps"] != 200 or protocol["batch_size"] != 16 or protocol["interactions"] != 3200:
        raise ValueError("short-window budget changed")
    legacy = {"L8": {"low": 8, "middle": 0, "high": 0}, "L4M4": {"low": 4, "middle": 4, "high": 0}, "L3M3H2": {"low": 3, "middle": 3, "high": 2}}
    cardinality = {
        "K2": {"low": 2, "middle": 0, "high": 0},
        "K4": {"low": 4, "middle": 0, "high": 0},
    }
    actual = {row["id"]: row["composition"] for row in protocol["scenarios"]}
    registered_short_subset = set(actual) == {"L8", "L4M4"} and all(actual[key] == legacy[key] for key in actual)
    if actual != legacy and not registered_short_subset and not (len(actual) == 1 and next(iter(actual.items())) in cardinality.items()):
        raise ValueError("request registry definition changed")
    if protocol["request_users"] != sum(next(iter(actual.values())).values()) or protocol["selection_uses_labels_or_predictions"] is not False:
        raise ValueError("request registry definition changed")
    if value["evaluation"] != {"split": "Development", "final_test": False, "inference_batch_size": 4, "bootstrap_resamples": 1000}:
        raise ValueError("Development-only evaluation changed")
    source = value["source"]
    sidecars = {"train_user_ids", "development_user_ids", "raw_lineage_manifest"}
    if sidecars & set(source) and not sidecars <= set(source):
        raise ValueError("dataset sidecar source is incomplete")
    value["_config_path"] = str(path.resolve())
    value["_config_sha256"] = sha256_file(path)
    return value


def _source_names(config: dict[str, Any]) -> tuple[str, ...]:
    base = ("train_json", "development_json", "original_checkpoint", "base_config", "trajectory_config")
    if "train_user_ids" in config["source"]:
        return base + ("train_user_ids", "development_user_ids", "raw_lineage_manifest")
    return base


def _user_sidecars(root: Path, config: dict[str, Any]) -> tuple[list[int], list[int], dict[str, Any]] | None:
    source = config["source"]
    if "train_user_ids" not in source:
        return None
    train = json.loads((root / source["train_user_ids"]).read_text(encoding="utf-8")); development = json.loads((root / source["development_user_ids"]).read_text(encoding="utf-8"))
    if not isinstance(train, list) or not isinstance(development, list) or not all(isinstance(value, int) and value > 0 for value in train + development):
        raise ValueError("invalid dataset user sidecars")
    return train, development, {"raw_unsplit_source_read": False, "train_rows": len(train), "development_rows_materialized": 0, "final_test_rows_materialized": 0, "train_user_order_sha256": canonical_hash(train), "sidecar_sha256": sha256_file(root / source["train_user_ids"])}


def _public_member(user: int, role: str, global_counts: Counter[int], window_counts: Counter[int], development_counts: Counter[int], seed: int) -> dict[str, Any]:
    return {"role": role, "user_hash": hashlib.sha256(f"bota-short:{seed}:{role}:{user}".encode()).hexdigest(), "global_train_frequency": int(global_counts[user]), "window_exposure": int(window_counts[user]), "development_samples": int(development_counts[user])}


def freeze_registry(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    source = config["source"]; protocol = config["protocol"]
    sidecars = _user_sidecars(root, config)
    if sidecars is None:
        user_ids, replay = p1._train_user_ids_only(root / source["raw_data"])
        _, development, _ = reconstruct_authoritative_rows(root / source["raw_data"]); development_user_ids = [int(row.authoritative_user_id) for row in development]
    else:
        user_ids, development_user_ids, replay = sidecars
    generator = torch.Generator(device="cpu"); generator.manual_seed(protocol["seed"]); order = torch.randperm(len(user_ids), generator=generator).tolist()[:protocol["interactions"]]
    if len(order) != 3200 or len(set(order)) != 3200:
        raise RuntimeError("short-window order is not a 3200-row permutation prefix")
    global_counts = Counter(map(int, user_ids)); window_counts = Counter(int(user_ids[index]) for index in order); development_counts = Counter(development_user_ids)
    ordered_users = sorted(global_counts, key=lambda user: (global_counts[user], hashlib.sha256(f"p3c1-tercile:{user}".encode()).hexdigest()))
    cut1, cut2 = len(ordered_users) // 3, 2 * len(ordered_users) // 3; terciles = {"low": ordered_users[:cut1], "middle": ordered_users[cut1:cut2], "high": ordered_users[cut2:]}
    pools: dict[str, list[int]] = {}
    for role in ("low", "middle", "high"):
        pool = [int(user) for user in terciles[role] if window_counts[int(user)] == 1 and development_counts[int(user)] >= protocol["minimum_development_samples_per_user"]]
        pool.sort(key=lambda user: hashlib.sha256(f"bota-short-pool:{protocol['seed']}:{role}:{user}".encode()).hexdigest())
        pools[role] = pool
    required = {role: max(int(row["composition"][role]) for row in protocol["scenarios"]) for role in ("low", "middle", "high")}
    if any(len(pools[role]) < required[role] for role in required):
        raise RuntimeError("insufficient outcome-blind users for frozen scenarios")
    scenarios = []
    for row in protocol["scenarios"]:
        selected: list[tuple[str, int]] = []
        for role in ("low", "middle", "high"):
            selected.extend((role, user) for user in pools[role][:int(row["composition"][role])])
        users = [user for _, user in selected]; slots = [position for position, index in enumerate(order) if int(user_ids[index]) in set(users)]
        train_indices = [order[position] for position in slots]
        requested = int(protocol["request_users"])
        if len(users) != requested or len(slots) != requested or len(set(users)) != requested:
            raise RuntimeError(f"scenario {row['id']} does not contain the frozen number of one-exposure users")
        public = [_public_member(user, role, global_counts, window_counts, development_counts, protocol["seed"]) for role, user in selected]
        scenarios.append({"id": row["id"], "composition": row["composition"], "users": users, "forget_window_positions": slots, "forget_train_indices": train_indices, "public_members": public, "request_hash": canonical_hash(public), "deleted_interactions": requested, "window_interactions": 3200, "actual_window_ratio": requested / 3200})
    public_scenarios = [{key: value for key, value in row.items() if key != "users"} for row in scenarios]
    return {"schema": SCHEMA, "seed": protocol["seed"], "order": order, "order_sha256": canonical_hash(order), "batch_order_sha256": canonical_hash([order[start:start + 16] for start in range(0, 3200, 16)]), "scenarios": scenarios, "public_scenarios": public_scenarios, "registry_sha256": canonical_hash(public_scenarios), "global_train_samples": len(user_ids), "development_samples": len(development_user_ids), "train_lineage": replay, "selection_uses_labels_or_predictions": False, "test_accessed": False}


def protocol_dir(root: Path, config: dict[str, Any], benchmark_name: str) -> Path:
    return root / config["output_root"] / "protocols" / safe_run_name(benchmark_name)


def prepare(root: Path, config_path: Path, benchmark_name: str) -> dict[str, Any]:
    config = load_config(config_path); benchmark_name = safe_run_name(benchmark_name); destination = protocol_dir(root, config, benchmark_name)
    if destination.exists(): raise FileExistsError(destination)
    registry = freeze_registry(root, config); source = config["source"]; protocol = config["protocol"]
    partition_seed = int(protocol["seed"]); run_seed = int(protocol.get("run_seed", protocol["seed"]))
    for name in _source_names(config):
        if not (root / source[name]).exists(): raise FileNotFoundError(root / source[name])
    stage = destination.parent / ".work" / f"{safe_run_name(benchmark_name)}.{uuid.uuid4().hex}.stage"; stage.mkdir(parents=True)
    private = {**registry}; public = private.pop("public_scenarios"); private["scenarios"] = [{key: value for key, value in row.items() if key != "public_members"} for row in registry["scenarios"]]
    full_rows = json.loads((root / source["train_json"]).read_text(encoding="utf-8")); window_rows = [full_rows[index] for index in registry["order"]]
    if len(full_rows) != registry["global_train_samples"] or len(window_rows) != 3200: raise RuntimeError("training JSON/order mismatch")
    data_root = stage / "data"; data_root.mkdir(); (data_root / "train_window.json").write_text(json.dumps(window_rows, ensure_ascii=False, separators=(",", ":")), encoding="utf-8", newline="\n")
    for scenario in registry["scenarios"]:
        scenario_root = data_root / scenario["id"]; scenario_root.mkdir(); forget_rows = [window_rows[position] for position in scenario["forget_window_positions"]]
        (scenario_root / "forget.json").write_text(json.dumps(forget_rows, ensure_ascii=False, separators=(",", ":")), encoding="utf-8", newline="\n")
        forget_positions = list(map(int, scenario["forget_window_positions"])); forget_set = set(forget_positions)
        sisa = sisa_partition(window_rows, 4, 4, partition_seed); sisa_plan = deletion_plan(sisa, forget_positions)
        sisa_initial = [[len(sisa[shard][slice_id]) for slice_id in range(4)] for shard in range(4)]; sisa_remaining = [[len([index for index in sisa[shard][slice_id] if index not in forget_set]) for slice_id in range(4)] for shard in range(4)]
        features = text_features(window_rows, 32); assignment = balanced_similarity_partition(features, 4, partition_seed); rec = {shard: {0: []} for shard in range(4)}
        for index, shard in enumerate(assignment): rec[shard][0].append(index)
        rec_plan = deletion_plan(rec, forget_positions); rec_initial = [len(rec[shard][0]) for shard in range(4)]; rec_remaining = [len([index for index in rec[shard][0] if index not in forget_set]) for shard in range(4)]
        relative = str(destination.relative_to(root)).replace("\\", "/"); common_sources = {"original": {"path": source["original_checkpoint"], "sha256": source_sha256(root / source["original_checkpoint"])}, "train": {"path": f"{relative}/data/train_window.json", "sha256": sha256_file(data_root / "train_window.json")}, "forget": {"path": f"{relative}/data/{scenario['id']}/forget.json", "sha256": sha256_file(scenario_root / "forget.json")}, "development": {"path": source["development_json"], "sha256": sha256_file(root / source["development_json"] )}}
        # paper_baselines.partitioned_budget deliberately reports the conservative
        # all-local-model post-delete budget; execute() separately records the
        # affected-shard incremental work that was actually performed.
        sisa_steps = sum((count + 15) // 16 for row in sisa_initial for count in row) + sum((count + 15) // 16 for row in sisa_remaining for count in row)
        rec_steps = sum((count + 15) // 16 for count in rec_initial) + sum((count + 15) // 16 for count in rec_remaining)
        deleted = int(scenario["deleted_interactions"])
        sisa_config = {"schema": "paper-sisa-v1", "display_name": "SISA-Short-T5", "seed": run_seed, "output_root": f"{config['output_root']}/component_runs/{benchmark_name}/{scenario['id']}/sisa", "synthetic_root": f"{config['output_root']}/synthetic/sisa", "test_access_policy": "forbidden", "tokenizer": "pretrained_models/t5-base", "sources": common_sources, "data": {"train_samples": 3200, "forget_samples": deleted}, "protocol": {"sharded": True, "isolated": True, "sliced": True, "shards": 4, "slices": 4, "partition_seed": partition_seed, "partition_unit": "interaction", "partition_rule": "sha256_seeded_balanced_independent_assignment", "aggregation": "mean_yes_no_probability", "epochs_per_slice": 1, "optimizer": "AdamW", "learning_rate": .0005, "physical_microbatch": 4, "accumulation_steps": 4, "effective_batch": 16}, "memory_guard": {"max_reserved_fraction": .90, "minimum_dedicated_free_bytes": 536870912}, "adaptation_disclosure": "True SISA control flow in a frozen 200-step T5 benchmark; full T5 shard models are trained sequentially.", "budget": {"source_paper": "Machine Unlearning", "official_repository": "https://github.com/cleverhans-lab/machine-unlearning", "official_epoch_or_step_budget": "one frozen epoch per slice", "official_checkpoint_used_for_reported_result": "final affected-slice replay state per shard", "initial_partition_counts": sisa_initial, "post_delete_partition_counts": sisa_remaining, "total_expected_optimizer_steps": sisa_steps, "estimated_runtime": "frozen 3200-row short benchmark"}}
        rec_config = {"schema": "paper-receraser-adapter-v1", "display_name": "RecEraser-Adapter-Short", "seed": run_seed, "output_root": f"{config['output_root']}/component_runs/{benchmark_name}/{scenario['id']}/receraser", "synthetic_root": f"{config['output_root']}/synthetic/receraser", "test_access_policy": "forbidden", "tokenizer": "pretrained_models/t5-base", "sources": common_sources, "data": {"train_samples": 3200, "forget_samples": deleted}, "protocol": {"method_name": "RecEraser-Adapter", "shards": 4, "partition_seed": partition_seed, "partition": "balanced_text_similarity", "feature_dimensions": 32, "aggregation": "learned_development_probability_attention", "attention_steps": 200, "attention_learning_rate": .1, "evaluation_batch_size": 4, "epochs_per_slice": 1, "optimizer": "AdamW", "learning_rate": .001, "physical_microbatch": 4, "accumulation_steps": 4, "effective_batch": 16}, "lora": {"r": 16, "lora_alpha": 32, "lora_dropout": .05, "target_modules": ["q", "v"]}, "memory_guard": {"max_reserved_fraction": .90, "minimum_dedicated_free_bytes": 536870912}, "adaptation_disclosure": "Adapter-based T5 adaptation preserving RecEraser partition/local update/aggregation in a frozen 200-step benchmark.", "budget": {"source_paper": "Recommendation Unlearning", "official_repository": "https://github.com/chenchongthu/Recommendation-Unlearning", "official_epoch_or_step_budget": "one frozen epoch per local model", "official_checkpoint_used_for_reported_result": "final affected-local-model state", "initial_partition_counts": rec_initial, "post_delete_partition_counts": rec_remaining, "total_expected_optimizer_steps": rec_steps, "estimated_runtime": "frozen 3200-row short benchmark"}}
        (scenario_root / "sisa.yaml").write_text(yaml.safe_dump(sisa_config, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n"); (scenario_root / "receraser.yaml").write_text(yaml.safe_dump(rec_config, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
    atomic_json(stage / "request_registry.private.json", private); atomic_json(stage / "request_registry.json", {"schema": SCHEMA, "scenarios": public, "registry_sha256": registry["registry_sha256"], "order_sha256": registry["order_sha256"], "batch_order_sha256": registry["batch_order_sha256"], "selection_uses_labels_or_predictions": False, "test_accessed": False})
    contract = {"schema": SCHEMA, "benchmark_name": safe_run_name(benchmark_name), "config_sha256": sha256_file(config_path), "sources": {name: {"path": source[name], "sha256": source_sha256(root / source[name])} for name in _source_names(config)}, "registry_sha256": registry["registry_sha256"], "request_seed": int(protocol["seed"]), "run_seed": int(protocol.get("run_seed", protocol["seed"])), "optimizer_steps": 200, "batch_size": 16, "global_train_samples": registry["global_train_samples"], "development_samples": registry["development_samples"], "dataset": source.get("dataset", "ml-1m"), "final_test_forbidden": True, "test_accessed": False}
    atomic_json(stage / "contract.json", contract); (stage / "PREPARED").write_text(MARKER + "\n", encoding="utf-8", newline="\n"); atomic_json(stage / "manifest.json", {"schema": SCHEMA, "files": {name: sha256_file(stage / name) for name in ("request_registry.private.json", "request_registry.json", "contract.json", "PREPARED")}, "data_tree_sha256": canonical_hash([(str(path.relative_to(data_root)).replace("\\", "/"), sha256_file(path)) for path in sorted(data_root.rglob("*")) if path.is_file()]), "published_atomically": True, "test_accessed": False})
    destination.parent.mkdir(parents=True, exist_ok=True); os.replace(stage, destination)
    return {"status": "PREPARED", "protocol_dir": str(destination), "benchmark_name": benchmark_name, "scenarios": [row["id"] for row in registry["scenarios"]], "registry_sha256": registry["registry_sha256"], "test_accessed": False}


def validate_prepared(root: Path, config: dict[str, Any], benchmark_name: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    directory = protocol_dir(root, config, benchmark_name); required = {"PREPARED", "contract.json", "data", "manifest.json", "request_registry.json", "request_registry.private.json"}
    if not directory.is_dir() or {path.name for path in directory.iterdir()} != required or (directory / "PREPARED").read_text(encoding="utf-8") != MARKER + "\n": raise ValueError("invalid prepared short benchmark")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if sha256_file(directory / name) != expected: raise ValueError(f"prepared artifact mismatch: {name}")
    actual_data_hash = canonical_hash([(str(path.relative_to(directory / "data")).replace("\\", "/"), sha256_file(path)) for path in sorted((directory / "data").rglob("*")) if path.is_file()])
    if actual_data_hash != manifest.get("data_tree_sha256"): raise ValueError("prepared data tree mismatch")
    contract = json.loads((directory / "contract.json").read_text(encoding="utf-8")); registry = json.loads((directory / "request_registry.private.json").read_text(encoding="utf-8"))
    if contract["config_sha256"] != config["_config_sha256"] or contract["registry_sha256"] != registry["registry_sha256"] or contract["test_accessed"] is not False: raise ValueError("prepared contract mismatch")
    return directory, contract, registry
