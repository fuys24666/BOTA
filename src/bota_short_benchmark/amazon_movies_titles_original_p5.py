"""Train the Development-selected P5 Amazon Movies and TV Original."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from src.bota_short_benchmark import recommendation_original as core

SCHEMA = "bota-amazon-movies-tv-titles-original-p5-v3"
MARKER = "BOTA_AMAZON_MOVIES_TV_TITLES_ORIGINAL_P5_V3_COMPLETED"
_CORE_SCHEMA, _CORE_MARKER, _CORE_LOAD_CONFIG = core.SCHEMA, core.MARKER, core.load_config


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != SCHEMA or value.get("test_access_policy") != "forbidden":
        raise ValueError("invalid Amazon Movies and TV P5 Original config")
    coordinate = {"target_modules": ["q", "v"], "module_count": 72, "rank": 16, "alpha": 32, "trainable": "B_only", "initial_B": "zero", "fixed_a_seed": 42}
    training = {"seed": 42, "optimizer": "AdamW", "learning_rate": .001, "betas": [.9, .999], "eps": 1e-8, "weight_decay": .01, "effective_batch_size": 16, "physical_microbatch": 4, "gradient_accumulation": 4, "maximum_epochs": 100, "early_stopping_metric": "development_sample_mean_answer_loss", "patience": 5, "min_delta": 0., "development_batch_size": 4, "checkpoint_every_epochs": 1}
    scope = {"exploratory_small_scale_only": True, "real_item_titles_from_official_metadata": True, "development_only_selection": True, "final_test_access": False, "recommendation_original_not_unlearning_model": True, "formal_dataset_claim": False}
    if value.get("coordinate") != coordinate or value.get("training") != training or value.get("scientific_scope") != scope:
        raise ValueError("Amazon Movies and TV P5 Original protocol changed")
    if value.get("source", {}).get("train_samples") != 4415 or value.get("source", {}).get("development_samples") != 1280 or value.get("publication", {}).get("merge_adapter_into_t5") is not True:
        raise ValueError("Amazon Movies and TV P5 Original source/publication changed")
    return value


def _activate() -> None: core.SCHEMA, core.MARKER, core.load_config = SCHEMA, MARKER, load_config
def _restore() -> None: core.SCHEMA, core.MARKER, core.load_config = _CORE_SCHEMA, _CORE_MARKER, _CORE_LOAD_CONFIG


def preflight(root: Path, config_path: Path, run_name: str):
    _activate()
    try: return core.preflight(root, config_path, run_name)
    finally: _restore()


def execute(root: Path, config_path: Path, run_name: str, resume: bool):
    _activate()
    try: return core.execute(root, config_path, run_name, resume)
    finally: _restore()


def analyze(root: Path, config_path: Path, run_name: str):
    _activate()
    try: return core.analyze(root, config_path, run_name)
    finally: _restore()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path.cwd()); parser.add_argument("--config", type=Path, default=Path("configs/bota_amazon_movies_titles_original_p5_v3.yaml")); parser.add_argument("--mode", choices=["Preflight", "SyntheticDryRun", "Full", "Resume", "Analyze"], default="Preflight"); parser.add_argument("--run-name", default="amazon_movies_tv_titles_original_p5_seed42_v3")
    args = parser.parse_args(); root = args.root.resolve(); path = (root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    if args.mode == "Preflight": result = preflight(root, path, args.run_name)
    elif args.mode == "SyntheticDryRun": result = {"schema": SCHEMA, "real_model_loaded": False, "optimizer_constructed": False, "real_data_read": False, "test_accessed": False}
    elif args.mode == "Analyze": result = analyze(root, path, args.run_name)
    else: result = execute(root, path, args.run_name, resume=args.mode == "Resume")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()

