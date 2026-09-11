from __future__ import annotations

import json
from pathlib import Path

from scripts.bota_if.audit_ali_stage0_v1 import nested_true, run_paths


ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "configs" / "ali_stage0_preregistration_v1.json"


def test_preregistration_is_development_only_and_complete() -> None:
    value = json.loads(PREREG.read_text(encoding="utf-8"))
    assert value["schema"] == "bota-ali-stage0-preregistration-v1"
    assert value["status"] == "FROZEN_BEFORE_NEW_EXPERIMENTS"
    assert value["scope"]["test_access_policy"] == "forbidden"
    assert value["scope"]["final_test_may_be_accessed"] is False
    assert value["scope"]["seeds"] == [41, 42, 43]
    assert set(value["scope"]["scenarios"]) == {"L8", "L4M4"}
    assert set(value["method_registry"]) == {
        "Original",
        "Exact-Masked",
        "Retain-Retrain-P5",
        "FullControl-P5",
        "BOTA",
        "IFRU",
        "E2URec",
        "NegGrad",
        "PCGrad",
        "SISA",
        "RecEraser",
    }


def test_frozen_source_hashes_match_repository() -> None:
    import hashlib

    value = json.loads(PREREG.read_text(encoding="utf-8"))
    for relative, expected in value["frozen_source_files"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected, relative


def test_primary_analysis_cannot_select_models() -> None:
    value = json.loads(PREREG.read_text(encoding="utf-8"))
    attack = value["attack_verification"]
    assert attack["model_selection_from_attack_results"] is False
    assert attack["primary_score"] == "negative_answer_loss"
    assert attack["bootstrap"] == {
        "unit": "authoritative user",
        "resamples": 2000,
        "confidence": 0.95,
        "seed": 42,
    }


def test_audit_detects_nested_finaltest_flags() -> None:
    assert nested_true({"outer": [{"test_accessed": False}, {"final_test": True}]}, {"test_accessed", "final_test"}) == [
        "outer[1].final_test"
    ]


def test_run_registry_covers_all_preregistered_methods() -> None:
    value = json.loads(PREREG.read_text(encoding="utf-8"))
    paths = run_paths(Path("experiment-root"), 42)
    assert set(paths) == set(value["method_registry"])
    assert all("seed42" in str(path) for path in paths.values())
