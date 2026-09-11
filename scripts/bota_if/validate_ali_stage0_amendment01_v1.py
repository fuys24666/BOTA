"""Validate Stage 0 Amendment 01 without reading experiment outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "bota-ali-stage0-amendment-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def validate(repo: Path, amendment_path: Path) -> list[str]:
    errors: list[str] = []
    amendment = load_json(amendment_path)

    if amendment.get("schema") != SCHEMA:
        errors.append(f"Unexpected amendment schema: {amendment.get('schema')!r}")
    if amendment.get("status") != "FROZEN_BEFORE_AMENDED_EXPERIMENTS":
        errors.append("Amendment is not frozen before amended experiments.")

    parent = amendment.get("parent_preregistration", {})
    parent_path = repo / str(parent.get("path", ""))
    if not parent_path.is_file():
        errors.append(f"Parent preregistration is missing: {parent_path}")
    elif sha256_file(parent_path) != parent.get("sha256"):
        errors.append("Parent preregistration hash changed after Amendment 01.")
    else:
        parent_value = load_json(parent_path)
        if parent_value.get("preregistration_id") != parent.get("preregistration_id"):
            errors.append("Parent preregistration ID does not match the amendment binding.")

    for relative, expected in amendment.get("bound_stage0_records", {}).items():
        path = repo / relative
        if not path.is_file():
            errors.append(f"Bound Stage 0 record is missing: {relative}")
        elif sha256_file(path) != expected:
            errors.append(f"Bound Stage 0 record changed: {relative}")

    unchanged = amendment.get("unchanged_primary_contracts", {})
    for key in (
        "attack_verification",
        "sequential_verification",
        "scale_verification",
        "finaltest_access_remains_forbidden",
        "adverse_and_null_results_remain_reportable",
    ):
        if unchanged.get(key) is not True:
            errors.append(f"Required unchanged-contract safeguard is absent: {key}")

    for section in (
        "mechanism_stabilization",
        "stronger_signal_secondary_analysis",
        "equal_billing_long_horizon",
    ):
        value = amendment.get(section, {})
        if value.get("analysis_split") != "Development":
            errors.append(f"{section} is not Development-only.")
        if value.get("finaltest_access") is not False:
            errors.append(f"{section} does not explicitly forbid FinalTest.")

    mechanism = amendment.get("mechanism_stabilization", {})
    if mechanism.get("seeds") != [41, 42, 43]:
        errors.append("Mechanism stabilization must include seeds 41, 42 and 43.")
    if set(mechanism.get("scenarios", [])) != {"L8", "L4M4"}:
        errors.append("Mechanism stabilization must include L8 and L4M4.")
    if set(mechanism.get("arms", {})) != {
        "FM_source_only",
        "FS_source_only",
        "FS_empirical_fisher",
    }:
        errors.append("Mechanism stabilization arm registry is incomplete.")

    stronger = amendment.get("stronger_signal_secondary_analysis", {})
    if stronger.get("request_sizes") != [2, 4, 8]:
        errors.append("Stronger-signal request sizes changed.")
    if stronger.get("adequate_effect_rule", {}).get("minimum_absolute_probability_effect") != 0.001:
        errors.append("Stronger-signal DU threshold changed.")

    equal = amendment.get("equal_billing_long_horizon", {})
    if equal.get("primary_reference_for_this_secondary_experiment") != "Exact-Masked-T_star":
        errors.append("Fixed-horizon matched reference changed.")

    expected_order = [
        "mechanism_stabilization",
        "stronger_signal_secondary_analysis",
        "equal_billing_long_horizon",
        "parent attack_verification",
        "parent sequential_verification",
        "parent scale_verification",
    ]
    if amendment.get("execution_order") != expected_order:
        errors.append("Execution order changed.")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--amendment",
        type=Path,
        default=Path("configs/ali_stage0_amendment01_v1.json"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    amendment = args.amendment
    if not amendment.is_absolute():
        amendment = repo / amendment

    errors = validate(repo, amendment)
    if errors:
        print("Amendment 01 validation: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Amendment 01 validation: PASS")
    print(f"- Amendment: {amendment}")
    print("- Parent preregistration and Stage 0 audit hashes are unchanged.")
    print("- Development-only, FinalTest-forbidden and full-reporting safeguards are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
