from __future__ import annotations

import json
from pathlib import Path

import yaml
import torch

from src.bota_short_benchmark.ali_stage3_equal_horizon_v1 import (
    EPOCHS,
    STEPS,
    build_long_schedule,
)
from src.bota_short_benchmark.ali_stage1_mechanism_v1 import ARMS
from src.bota_if.p2b_full_module_adamw_transport_audit import vector_metrics


ROOT = Path(__file__).resolve().parents[1]


def test_amendment02_discloses_the_pre_execution_horizon_change() -> None:
    value = json.loads((ROOT / "configs/ali_stage0_amendment02_v1.json").read_text(encoding="utf-8"))
    assert value["status"] == "FROZEN_BEFORE_STAGE3_EXECUTION"
    assert value["pre_execution_observation"]["development_outcomes_from_stage3_inspected"] is False
    assert value["pre_execution_observation"]["stage3_endpoints_constructed"] is False
    assert value["amended_equal_billing_long_horizon"]["fixed_optimizer_steps"] == 1200
    assert value["amended_equal_billing_long_horizon"]["finaltest_access"] is False


def test_stage3_config_is_equal_billed_and_development_only() -> None:
    value = yaml.safe_load((ROOT / "configs/ali_stage3_equal_horizon_v1.yaml").read_text(encoding="utf-8"))
    assert value["test_access_policy"] == "forbidden"
    assert value["stage1_selection"]["selected_arm"] == "FS_empirical_fisher"
    assert value["protocol"]["optimizer_steps"] == STEPS
    assert value["protocol"]["epochs"] == EPOCHS
    assert value["protocol"]["reference"] == "Exact-Masked-Tlong"
    assert value["protocol"]["full_batch_denominator_preserved"] is True
    assert value["evaluation"]["split"] == "Development"
    assert value["evaluation"]["final_test"] is False
    assert value["reporting"]["report_all_six_cells"] is True
    assert value["reporting"]["retain_retrain_is_denominator"] is False


def test_long_schedule_is_deterministic_six_epoch_permutation() -> None:
    base = list(range(3200))
    left = build_long_schedule(base, 41)
    right = build_long_schedule(base, 41)
    assert left == right
    assert len(left) == 19200
    assert len(left) // 16 == STEPS
    for epoch in range(EPOCHS):
        assert sorted(left[epoch * 3200 : (epoch + 1) * 3200]) == base
    assert left != build_long_schedule(base, 42)


def test_launcher_validates_amendment_and_runs_all_seeds_sequentially() -> None:
    text = (ROOT / "scripts/bota_if/run_ali_stage3_equal_horizon_v1.ps1").read_text(encoding="utf-8")
    assert "validate_ali_stage0_amendment02_v1.py" in text
    assert '$Seeds=if($Seed -eq "All"){@(41,42,43)}' in text
    assert "ali_stage3_equal_horizon_v1" in text
    assert 'if($Mode -eq "Full" -and $Seed -eq "All")' in text


def test_parameter_metric_source_keys_are_the_audited_vector_metrics() -> None:
    metrics = vector_metrics(torch.tensor([1.0, 0.0]), torch.tensor([0.8, 0.2]))
    assert metrics["cosine"] is not None
    assert metrics["relative_l2_error"] is not None
    source = (ROOT / "src/bota_short_benchmark/ali_stage3_equal_horizon_v1.py").read_text(encoding="utf-8")
    assert '"delta_cosine": parameter["cosine"]' in source
    assert '"relative_l2": parameter["relative_l2_error"]' in source
    assert "arms=(SELECTED_ARM,)" in source
    assert "FS_empirical_fisher" in ARMS
