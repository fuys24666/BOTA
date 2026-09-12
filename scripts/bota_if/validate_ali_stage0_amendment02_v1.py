from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(repo: Path) -> dict[str, object]:
    path = repo / "configs/ali_stage0_amendment02_v1.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    parent = value["parent_amendment"]
    parent_path = repo / parent["path"]
    if sha256_file(parent_path) != parent["sha256"]:
        raise ValueError("Amendment 01 hash changed")
    if value.get("status") != "FROZEN_BEFORE_STAGE3_EXECUTION":
        raise ValueError("Amendment 02 is not frozen")
    protocol = value["amended_equal_billing_long_horizon"]
    required_protocol = {
        "analysis_split": "Development",
        "finaltest_access": False,
        "seeds": [41, 42, 43],
        "scenarios": ["L8", "L4M4"],
        "fixed_optimizer_steps": 1200,
        "primary_reference": "Exact-Masked-T_long",
        "required_metrics": ["DU", "EU", "local_residual", "delta_cosine", "relative_l2", "AUC", "log_loss", "wall_time"],
    }
    if any(protocol.get(key) != expected for key, expected in required_protocol.items()):
        raise ValueError("Amendment 02 fixed protocol changed")
    observations = value["pre_execution_observation"]
    if observations.get("development_outcomes_from_stage3_inspected") is not False:
        raise ValueError("Amendment 02 was not frozen before Stage 3 outcomes")
    for seed in ("41", "42", "43"):
        row = observations["seeds"][seed]
        if row != {
            "best_epoch": 1,
            "stopping_epoch": 6,
            "steps_per_epoch": 200,
            "recorded_total_optimizer_steps": 1200,
        }:
            raise ValueError(f"Stage 3 source-horizon audit changed for seed {seed}")
    return {"status": "PASS", "path": str(path), "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    result = validate(args.repo.resolve())
    print("Amendment 02 validation: PASS")
    print(f"- Amendment: {result['path']}")
    print("- Amendment 01 remains hash-bound and unchanged.")
    print("- Stage 3 is frozen to 1,200 equal-billed steps on Development only.")


if __name__ == "__main__":
    main()
