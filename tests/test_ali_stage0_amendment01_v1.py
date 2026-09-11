from __future__ import annotations

import json
from pathlib import Path

from scripts.bota_if.validate_ali_stage0_amendment01_v1 import validate


ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / "configs" / "ali_stage0_amendment01_v1.json"


def test_amendment_validates_against_unchanged_stage0_records() -> None:
    assert validate(ROOT, AMENDMENT) == []


def test_amendment_is_additive_and_development_only() -> None:
    value = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    assert value["schema"] == "bota-ali-stage0-amendment-v1"
    assert value["parent_preregistration"]["path"] == "configs/ali_stage0_preregistration_v1.json"
    assert value["unchanged_primary_contracts"]["original_primary_scenarios"] == ["L8", "L4M4"]
    for section in (
        "mechanism_stabilization",
        "stronger_signal_secondary_analysis",
        "equal_billing_long_horizon",
    ):
        assert value[section]["analysis_split"] == "Development"
        assert value[section]["finaltest_access"] is False


def test_known_results_are_disclosed_as_retrospective_context() -> None:
    value = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    known = value["known_before_freeze"]
    assert "retrospective" in known["policy"]
    assert known["seed42_transport_ablation"]["L4M4"]["FM"]["local_residual"] == 0.6629
    assert known["seed42_transport_ablation"]["L4M4"]["FS"]["local_residual"] == 0.7064
    assert known["seed42_response_correction"]["absolute_residual_improvement"] == 0.0179


def test_mechanism_selection_precedes_attack() -> None:
    value = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    order = value["execution_order"]
    assert order.index("mechanism_stabilization") < order.index("parent attack_verification")
    rules = value["mechanism_stabilization"]["selection_rules"]
    assert "before any new attack result" in rules["freeze_before_attack"]


def test_secondary_analyses_cannot_replace_primary_scenarios() -> None:
    value = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    stronger = value["stronger_signal_secondary_analysis"]
    assert stronger["evidence_class"].startswith("secondary")
    assert stronger["adequate_effect_rule"]["minimum_absolute_probability_effect"] == 0.001
    assert value["equal_billing_long_horizon"]["primary_reference_for_this_secondary_experiment"] == "Exact-Masked-T_star"
