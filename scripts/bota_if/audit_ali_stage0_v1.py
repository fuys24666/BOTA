"""Audit frozen ML-1M artifacts before the Ali-feedback experiments.

This script is deliberately read-only with respect to experiment artifacts. It
checks the Development-only three-seed benchmark, writes a compact audit report,
and never traverses the supplemental FinalTest directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


SCHEMA = "bota-ali-stage0-artifact-audit-v1"
SCENARIOS = ("L8", "L4M4")
SEEDS = (41, 42, 43)


@dataclass
class Check:
    level: str
    code: str
    message: str
    path: str | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def directory_hash(path: Path) -> str:
    rows = [
        (item.relative_to(path).as_posix(), sha256_file(item))
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]
    return canonical_hash(rows)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def nested_true(value: Any, keys: set[str]) -> list[str]:
    findings: list[str] = []

    def visit(node: Any, location: str) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                child_location = f"{location}.{key}" if location else key
                if key.lower() in keys and child is True:
                    findings.append(child_location)
                visit(child, child_location)
        elif isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, f"{location}[{index}]")

    visit(value, "")
    return findings


def run_paths(root: Path, seed: int) -> dict[str, Path]:
    prefix = root / f"seed{seed}"
    core_models = prefix / "core" / "models"
    paper_models = prefix / "paper" / "models"
    e2urec_models = prefix / "e2urec" / "models"
    return {
        "Original": core_models / "Original-Short" / f"bota_short_original_seed{seed}_v3",
        "Exact-Masked": core_models / "Retrain-Short" / f"bota_short_exact_masked_seed{seed}_v3",
        "BOTA": core_models / "BOTA-T2-Short" / f"bota_short_bota_seed{seed}_v3",
        "IFRU": core_models / "IFRU-Short-LoRA" / f"bota_short_ifru_seed{seed}_v3",
        "E2URec": e2urec_models / "E2URec-Short-FixedAB" / f"bota_short_e2urec_seed{seed}_v1",
        "NegGrad": paper_models / "NegGrad-Mixed-Short-BOnly" / f"bota_short_neggrad_seed{seed}_v3",
        "PCGrad": paper_models / "PCGrad-Short-BOnly" / f"bota_short_pcgrad_seed{seed}_v3",
        "SISA": core_models / "SISA-Short-T5" / f"bota_short_sisa_seed{seed}_v3",
        "RecEraser": core_models / "RecEraser-Adapter-Short" / f"bota_short_receraser_seed{seed}_v3",
        "Retain-Retrain-P5": paper_models / "Retain-Retrain-P5-Short" / f"bota_short_retain_p5_seed{seed}_v3",
        "FullControl-P5": paper_models / "FullControl-P5-Short" / f"bota_short_full_control_p5_seed{seed}_v3",
    }


def check_json_file(path: Path, checks: list[Check]) -> Any | None:
    if not path.is_file():
        checks.append(Check("ERROR", "MISSING_JSON", "Required JSON file is missing.", str(path)))
        return None
    try:
        value = load_json(path)
    except Exception as error:  # audit must report rather than hide parse failures
        checks.append(Check("ERROR", "INVALID_JSON", f"JSON parse failed: {error}", str(path)))
        return None
    forbidden = nested_true(value, {"test_accessed", "final_test", "finaltest"})
    if forbidden:
        checks.append(Check("ERROR", "FINALTEST_TRUE", f"Forbidden true flag(s): {forbidden}", str(path)))
    return value


def verify_hash(path: Path, expected: str | None, checks: list[Check], code: str) -> None:
    if not expected or not path.is_file():
        return
    actual = sha256_file(path)
    if actual != expected:
        checks.append(Check("ERROR", code, f"Expected {expected}, found {actual}.", str(path)))


def verify_file_manifest(root: Path, manifest: Any, checks: list[Check]) -> None:
    if not isinstance(manifest, dict):
        return
    for relative, expected in manifest.get("files", {}).items():
        target = root / relative
        if not target.is_file():
            checks.append(Check("ERROR", "MANIFEST_FILE_MISSING", "Manifest-listed file is missing.", str(target)))
        else:
            verify_hash(target, expected, checks, "MANIFEST_FILE_HASH")


def audit(repo: Path, experiment_root: Path, prereg: Path) -> dict[str, Any]:
    checks: list[Check] = []
    inventory: list[dict[str, Any]] = []
    preregistration = check_json_file(prereg, checks)
    if not isinstance(preregistration, dict) or preregistration.get("schema") != "bota-ali-stage0-preregistration-v1":
        checks.append(Check("ERROR", "BAD_PREREGISTRATION", "Preregistration schema is absent or invalid.", str(prereg)))
        preregistration = {}
    if preregistration.get("scope", {}).get("test_access_policy") != "forbidden":
        checks.append(Check("ERROR", "TEST_POLICY", "Preregistration must forbid FinalTest.", str(prereg)))
    if "finaltest" in str(experiment_root).lower() or "final_test" in str(experiment_root).lower():
        checks.append(Check("ERROR", "FINALTEST_ROOT", "Experiment root itself names FinalTest.", str(experiment_root)))

    for relative, expected in preregistration.get("frozen_source_files", {}).items():
        source = repo / relative
        if not source.is_file():
            checks.append(Check("ERROR", "MISSING_FROZEN_SOURCE", "Frozen source file is missing.", str(source)))
            continue
        actual = sha256_file(source)
        if actual != expected:
            checks.append(Check("ERROR", "FROZEN_SOURCE_CHANGED", f"Expected {expected}, found {actual}.", str(source)))

    registry_hashes: set[str] = set()
    request_hashes: dict[tuple[int, str], set[str]] = {}
    source_hashes: dict[str, set[str]] = {}
    mia_support: list[dict[str, int | str]] = []

    workspace = experiment_root.parents[1]
    raw_data = workspace / "data" / "ml-1m" / "raw_data"
    for name in ("movies.dat", "ratings.dat", "users.dat"):
        if not (raw_data / name).is_file():
            checks.append(Check("ERROR", "MISSING_RAW_DATA", f"Required ML-1M source '{name}' is missing.", str(raw_data / name)))
    model_dir = workspace / "pretrained_models" / "t5-base"
    for name in ("config.json", "pytorch_model.bin", "spiece.model", "tokenizer_config.json"):
        if not (model_dir / name).is_file():
            checks.append(Check("ERROR", "MISSING_PRETRAINED_MODEL", f"Required T5-base file '{name}' is missing.", str(model_dir / name)))

    for seed in SEEDS:
        protocol = experiment_root / f"seed{seed}" / "core" / "protocols" / f"bota_short_i02_seed{seed}_v3"
        contract = check_json_file(protocol / "contract.json", checks)
        protocol_manifest = check_json_file(protocol / "manifest.json", checks)
        verify_file_manifest(protocol, protocol_manifest, checks)
        if isinstance(protocol_manifest, dict) and (protocol / "data").is_dir():
            expected_data_hash = protocol_manifest.get("data_tree_sha256")
            if expected_data_hash and directory_hash(protocol / "data") != expected_data_hash:
                checks.append(Check("ERROR", "PROTOCOL_DATA_HASH", "Prepared protocol data tree does not match its manifest.", str(protocol / "data")))
        public_registry = check_json_file(protocol / "request_registry.json", checks)
        check_json_file(protocol / "request_registry.private.json", checks)
        if isinstance(public_registry, dict):
            scenario_rows = {row.get("id"): row for row in public_registry.get("scenarios", [])}
            for scenario in SCENARIOS:
                row = scenario_rows.get(scenario)
                members = row.get("public_members", []) if isinstance(row, dict) else []
                if len(members) != 8:
                    checks.append(Check("ERROR", "REGISTRY_MEMBER_COUNT", f"{scenario} expected 8 public members, found {len(members)}.", str(protocol / "request_registry.json")))
                if any(member.get("development_samples", 0) < 10 for member in members):
                    checks.append(Check("ERROR", "MIA_DEVELOPMENT_SUPPORT", f"{scenario} has a requested user with fewer than 10 Development samples.", str(protocol / "request_registry.json")))
                if any(member.get("window_exposure") != 1 for member in members):
                    checks.append(Check("ERROR", "WINDOW_EXPOSURE", f"{scenario} does not preserve the registered one-exposure protocol.", str(protocol / "request_registry.json")))
                if members:
                    counts = [int(member.get("development_samples", 0)) for member in members]
                    mia_support.append({"seed": seed, "scenario": scenario, "users": len(members), "development_samples": sum(counts), "minimum_per_user": min(counts), "maximum_per_user": max(counts)})
        train_window = protocol / "data" / "train_window.json"
        if not train_window.is_file():
            checks.append(Check("ERROR", "MISSING_TRAIN_WINDOW", "Frozen 3,200-row window is missing.", str(train_window)))
        else:
            rows = load_json(train_window)
            if not isinstance(rows, list) or len(rows) != 3200:
                checks.append(Check("ERROR", "TRAIN_WINDOW_COUNT", f"Expected 3200 rows, found {len(rows) if isinstance(rows, list) else 'non-list'}.", str(train_window)))
        for scenario in SCENARIOS:
            forget = protocol / "data" / scenario / "forget.json"
            if not forget.is_file():
                checks.append(Check("ERROR", "MISSING_FORGET_SET", "Scenario forget set is missing.", str(forget)))
            else:
                rows = load_json(forget)
                if not isinstance(rows, list) or len(rows) != 8:
                    checks.append(Check("ERROR", "FORGET_COUNT", f"Expected 8 rows, found {len(rows) if isinstance(rows, list) else 'non-list'}.", str(forget)))

        if isinstance(contract, dict):
            if contract.get("request_seed") != 42 or contract.get("run_seed") != seed:
                checks.append(Check("ERROR", "SEED_BINDING", "Request/run seed binding is inconsistent.", str(protocol / "contract.json")))
            if contract.get("test_accessed") is not False or contract.get("final_test_forbidden") is not True:
                checks.append(Check("ERROR", "PROTOCOL_TEST_POLICY", "Protocol contract does not explicitly forbid FinalTest.", str(protocol / "contract.json")))
            registry_hash = contract.get("registry_sha256")
            if registry_hash:
                registry_hashes.add(registry_hash)
            frozen_config = repo / "configs" / f"bota_short_benchmark_seed{seed}_v3.yaml"
            if frozen_config.is_file() and contract.get("config_sha256") != sha256_file(frozen_config):
                checks.append(Check("ERROR", "PROTOCOL_CONFIG_MISMATCH", "Protocol was not produced from the frozen public configuration.", str(protocol / "contract.json")))
            for source_name, spec in contract.get("sources", {}).items():
                expected = spec.get("sha256")
                if expected:
                    source_hashes.setdefault(source_name, set()).add(expected)
                source_path = Path(spec.get("path", ""))
                resolved = source_path if source_path.is_absolute() else workspace / source_path
                if not resolved.exists():
                    checks.append(Check("ERROR", "MISSING_PROTOCOL_SOURCE", f"Protocol source '{source_name}' is missing.", str(resolved)))

        for method, path in run_paths(experiment_root, seed).items():
            row: dict[str, Any] = {"seed": seed, "method": method, "path": str(path), "scenarios": {}}
            if not path.is_dir():
                checks.append(Check("ERROR", "MISSING_METHOD_RUN", f"{method} run is missing.", str(path)))
                inventory.append(row)
                continue
            if not (path / "COMPLETED").is_file():
                checks.append(Check("ERROR", "MISSING_COMPLETION_MARKER", f"{method} lacks COMPLETED.", str(path)))
            state = check_json_file(path / "run_state.json", checks)
            run_manifest = check_json_file(path / "manifest.json", checks)
            if isinstance(run_manifest, dict):
                verify_hash(path / "contract.json", run_manifest.get("contract_sha256"), checks, "RUN_CONTRACT_HASH")
                verify_hash(path / "run_state.json", run_manifest.get("run_state_sha256"), checks, "RUN_STATE_HASH")
                for scenario, expected in run_manifest.get("scenario_manifests", {}).items():
                    verify_hash(path / "scenarios" / scenario / "scenario_manifest.json", expected, checks, "SCENARIO_MANIFEST_HASH")
                for scenario, expected in run_manifest.get("phase_timings", {}).items():
                    verify_hash(path / "scenarios" / scenario / "phase_timing.json", expected, checks, "PHASE_TIMING_HASH")
            if not isinstance(state, dict) or state.get("status") != "COMPLETED":
                checks.append(Check("ERROR", "RUN_NOT_COMPLETED", f"{method} run_state is not COMPLETED.", str(path / "run_state.json")))
            for scenario in SCENARIOS:
                scenario_root = path / "scenarios" / scenario
                scenario_manifest = check_json_file(scenario_root / "scenario_manifest.json", checks)
                artifact_count = 0
                artifact_bytes = 0
                if method in {"SISA", "RecEraser"}:
                    component_text = scenario_manifest.get("component_run") if isinstance(scenario_manifest, dict) else None
                    component = Path(component_text) if component_text else None
                    if component is None or not component.is_dir():
                        checks.append(Check("ERROR", "MISSING_COMPONENT_RUN", f"{method} component run is missing.", component_text))
                    else:
                        check_json_file(component / "paper_model_manifest.json", checks)
                        expected_name = "final_state.pt"
                        artifacts = list(component.glob(f"models/shard_*/{expected_name}"))
                        artifact_count = len(artifacts)
                        artifact_bytes = sum(item.stat().st_size for item in artifacts)
                        if artifact_count != 4:
                            checks.append(Check("ERROR", "SHARD_COUNT", f"{method} expected 4 final shard states, found {artifact_count}.", str(component)))
                else:
                    adapter = scenario_root / "adapter"
                    artifacts = [adapter / "adapter_model.pt", adapter / "adapter_config.json"]
                    missing = [str(item) for item in artifacts if not item.is_file()]
                    artifact_count = sum(item.is_file() for item in artifacts)
                    artifact_bytes = sum(item.stat().st_size for item in artifacts if item.is_file())
                    if missing:
                        checks.append(Check("ERROR", "MISSING_ADAPTER", f"{method} adapter files are missing: {missing}", str(adapter)))
                    if isinstance(scenario_manifest, dict):
                        recorded_artifact = scenario_manifest.get("artifact", {})
                        recorded_path = Path(recorded_artifact.get("path", "")) if recorded_artifact else None
                        if recorded_path and not recorded_path.exists():
                            actual_hash = directory_hash(adapter) if adapter.is_dir() else None
                            if actual_hash == recorded_artifact.get("sha256"):
                                checks.append(Check("WARNING", "STALE_STAGING_PATH", f"{method} manifest points to the pre-publication .work path; the final adapter exists and matches the recorded hash.", str(scenario_root / "scenario_manifest.json")))
                            else:
                                checks.append(Check("ERROR", "BROKEN_ARTIFACT_POINTER", f"{method} manifest artifact path is stale and the final adapter hash does not match.", str(scenario_root / "scenario_manifest.json")))
                if isinstance(scenario_manifest, dict):
                    if scenario_manifest.get("test_accessed") is not False:
                        checks.append(Check("ERROR", "SCENARIO_TEST_POLICY", f"{method}/{scenario} does not record test_accessed=false.", str(scenario_root / "scenario_manifest.json")))
                    request_hash = scenario_manifest.get("request_hash")
                    if request_hash:
                        request_hashes.setdefault((seed, scenario), set()).add(request_hash)
                row["scenarios"][scenario] = {"artifact_count": artifact_count, "artifact_bytes": artifact_bytes}
            inventory.append(row)

        evaluation = experiment_root / f"seed{seed}" / "paper" / "evaluations" / f"bota_short_development_seed{seed}_v3"
        for name in ("COMPLETED", "metrics.json", "metrics.csv", "per_sample_metrics.jsonl", "provenance.json", "manifest.json"):
            if not (evaluation / name).is_file():
                checks.append(Check("ERROR", "MISSING_DEVELOPMENT_EVALUATION", f"Development evaluation file '{name}' is missing.", str(evaluation / name)))
        check_json_file(evaluation / "metrics.json", checks)
        check_json_file(evaluation / "provenance.json", checks)
        evaluation_manifest = check_json_file(evaluation / "manifest.json", checks)
        verify_file_manifest(evaluation, evaluation_manifest, checks)

    if len(registry_hashes) != 1:
        checks.append(Check("ERROR", "REGISTRY_HASH_MISMATCH", f"Expected one shared request registry hash, found {sorted(registry_hashes)}."))
    for key, hashes in request_hashes.items():
        if len(hashes) != 1:
            checks.append(Check("ERROR", "REQUEST_HASH_MISMATCH", f"Methods disagree on request hash for seed/scenario {key}: {sorted(hashes)}."))
    for source_name, hashes in source_hashes.items():
        if source_name in {"train_json", "development_json", "original_checkpoint"} and len(hashes) != 1:
            checks.append(Check("ERROR", "SOURCE_HASH_MISMATCH", f"Seeds disagree on source '{source_name}': {sorted(hashes)}."))

    errors = sum(item.level == "ERROR" for item in checks)
    warnings = sum(item.level == "WARNING" for item in checks)
    try:
        git_head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        git_status = subprocess.check_output(["git", "-C", str(repo), "status", "--short"], text=True).splitlines()
    except Exception:
        git_head, git_status = None, ["git status unavailable"]

    def portable(path_text: str | None) -> str | None:
        if not path_text:
            return path_text
        path = Path(path_text)
        for base, label in ((experiment_root, "<EXPERIMENT_ROOT>"), (repo, "<REPO_ROOT>"), (workspace, "<WORKSPACE_ROOT>")):
            try:
                return f"{label}/{path.resolve().relative_to(base.resolve()).as_posix()}"
            except (OSError, ValueError):
                pass
        return path_text

    for item in checks:
        item.path = portable(item.path)
    for row in inventory:
        row["path"] = portable(row["path"])
    return {
        "schema": SCHEMA,
        "preregistration_id": preregistration.get("preregistration_id"),
        "repo_root": "<REPO_ROOT>",
        "experiment_root": "<EXPERIMENT_ROOT>",
        "git_head": git_head,
        "git_status_at_audit": git_status,
        "final_test_traversed": False,
        "summary": {"errors": errors, "warnings": warnings, "ready_for_stage1": errors == 0},
        "shared_registry_hashes": sorted(registry_hashes),
        "mia_support": mia_support,
        "checks": [asdict(item) for item in checks],
        "inventory": inventory,
    }


def markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Ali feedback Stage 0 artifact audit",
        "",
        f"- Preregistration: `{report.get('preregistration_id')}`",
        f"- Git HEAD: `{report.get('git_head')}`",
        f"- Experiment root: `{report.get('experiment_root')}`",
        "- Scope: ML-1M Development only; FinalTest was not traversed.",
        f"- Result: **{'READY' if summary['ready_for_stage1'] else 'NOT READY'}** ({summary['errors']} errors, {summary['warnings']} warnings)",
        "",
        "## Artifact matrix",
        "",
        "| Seed | Method | L8 artifacts | L4M4 artifacts | Status |",
        "|---:|---|---:|---:|---|",
    ]
    error_paths = {item.get("path") for item in report["checks"] if item["level"] == "ERROR"}
    for row in report["inventory"]:
        status = "ERROR" if row["path"] in error_paths else "present"
        lines.append(
            f"| {row['seed']} | {row['method']} | {row['scenarios'].get('L8', {}).get('artifact_count', 0)} | "
            f"{row['scenarios'].get('L4M4', {}).get('artifact_count', 0)} | {status} |"
        )
    lines.extend(["", "## Findings", ""])
    if not report["checks"]:
        lines.append("No errors or warnings.")
    else:
        for item in report["checks"]:
            location = f" (`{item['path']}`)" if item.get("path") else ""
            lines.append(f"- **{item['level']} {item['code']}**: {item['message']}{location}")
    lines.extend(["", "## Matched-user Development support", "", "| Seed | Scenario | Users | Development samples | Minimum/user | Maximum/user |", "|---:|---|---:|---:|---:|---:|"])
    for row in report["mia_support"]:
        lines.append(f"| {row['seed']} | {row['scenario']} | {row['users']} | {row['development_samples']} | {row['minimum_per_user']} | {row['maximum_per_user']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "`READY` means the frozen Development artifacts needed to begin attack verification are present and internally bound to the registered requests. It does not mean that MIA, sequential deletion or scalable storage has already been demonstrated.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    prereg = (args.preregistration or repo / "configs" / "ali_stage0_preregistration_v1.json").resolve()
    report = audit(repo, args.experiment_root.resolve(), prereg)
    json_output = args.json_output or repo / "docs" / "audits" / "ali_stage0_artifact_audit_v1.json"
    markdown_output = args.markdown_output or repo / "docs" / "audits" / "ALI_STAGE0_ARTIFACT_AUDIT.md"
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_output.write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], sort_keys=True))
    raise SystemExit(0 if report["summary"]["ready_for_stage1"] else 1)


if __name__ == "__main__":
    main()
