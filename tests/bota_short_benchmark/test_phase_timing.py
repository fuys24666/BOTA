from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.bota_short_benchmark.timing import TIMING_SCHEMA, select_scenarios, timing_record


ROOT = Path(__file__).resolve().parents[2]
ROWS = [{"id": "L8"}, {"id": "L4M4"}, {"id": "L3M3H2"}]


def test_scenario_selection_is_explicit_and_order_preserving():
    assert [row["id"] for row in select_scenarios(ROWS, "All")] == ["L8", "L4M4", "L3M3H2"]
    assert select_scenarios(ROWS, "L8") == [{"id": "L8"}]
    assert select_scenarios(ROWS, "L4M4") == [{"id": "L4M4"}]
    with pytest.raises(ValueError): select_scenarios(ROWS, "unknown")


def test_phase_timing_separates_offline_online_and_publication():
    value = timing_record(scenario="L8", initialization_seconds=1, offline_construction_seconds=20, online_compute_seconds=2, adapter_publication_seconds=3, end_to_end_seconds=26, details={"cg_seconds": 1.5})
    assert value["schema"] == TIMING_SCHEMA
    assert value["online_total_seconds"] == 5
    assert value["offline_construction_seconds"] == 20
    assert value["details"]["cg_seconds"] == 1.5


@pytest.mark.parametrize("value", [-1, math.nan, math.inf])
def test_phase_timing_rejects_invalid_values(value):
    with pytest.raises(ValueError): timing_record(scenario="L8", initialization_seconds=0, offline_construction_seconds=value, online_compute_seconds=0, adapter_publication_seconds=0, end_to_end_seconds=0)


def test_bota_and_ifru_publish_required_phase_breakdown():
    source = (ROOT / "src/bota_short_benchmark/runner.py").read_text(encoding="utf-8")
    for key in ("trajectory_transport_seconds", "online_vector_composition_seconds", "forget_gradient_seconds", "lambda_estimation_seconds", "cg_seconds", "candidate_reconstruction_seconds"):
        assert key in source
    assert 'adapter_publication_seconds=publication_seconds' in source


def test_all_formal_wrappers_forward_single_scenario():
    wrappers = [
        "run_short_original_v1.ps1", "run_short_retrain_v1.ps1", "run_short_bota_v1.ps1",
        "run_short_ifru_v1.ps1", "run_short_sisa_v1.ps1", "run_short_receraser_v1.ps1",
        "run_short_full_control_p5_v2.ps1", "run_short_retain_p5_v2.ps1",
        "run_short_neggrad_v2.ps1", "run_short_pcgrad_v2.ps1", "run_short_paper_evaluation_v2.ps1",
    ]
    for name in wrappers:
        text = (ROOT / "scripts/bota_if" / name).read_text(encoding="utf-8")
        assert 'ValidateSet("All","L8","L4M4","L3M3H2")' in text
        assert '"--scenario",$Scenario' in text or '--scenario $Scenario' in text


def test_evaluation_publishes_efficiency_table_and_validates_timing():
    source = (ROOT / "src/bota_short_benchmark/paper_evaluation_v2.py").read_text(encoding="utf-8")
    assert '"efficiency.csv"' in source
    assert "missing phase timing" in source
    assert "does not contain requested scenario" in source


def test_single_scenario_neggrad_and_pcgrad_do_not_hardcode_l8_source():
    source = (ROOT / "src/bota_short_benchmark/paper_v2.py").read_text(encoding="utf-8")
    assert '"scenarios/L8/adapter/adapter_model.pt"' not in source
    assert 'scenario["id"], device' in source
