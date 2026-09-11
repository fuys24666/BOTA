from __future__ import annotations

from pathlib import Path

from src.bota_short_benchmark.ali_stage1_mechanism_v1 import (
    ARMS,
    SCENARIOS,
    decide,
    default_run_name,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "ali_stage1_mechanism_v1.yaml"


def _row(residual: float, relative_l2: float, auc_damage: float = 0.0, log_damage: float = 0.0):
    return {
        "local_residual_mean": residual,
        "relative_l2_mean": relative_l2,
        "overall_auc_damage_mean": auc_damage,
        "overall_logloss_damage_mean": log_damage,
    }


def test_config_binds_amendment_and_forbids_finaltest() -> None:
    value = load_config(ROOT, CONFIG)
    assert value["evaluation"]["split"] == "Development"
    assert value["evaluation"]["final_test"] is False
    assert value["protocol"]["arms"] == list(ARMS)
    assert value["formal_reproduction"] == {
        "exact_tensor_equality_required": False,
        "cosine_minimum": 0.9999,
        "relative_l2_maximum": 0.01,
        "maximum_absolute_error": 0.001,
        "scientific_selection_uses_this_gate": False,
    }


def test_default_run_names_are_unique() -> None:
    names = {default_run_name(seed, scenario) for seed in (41, 42, 43) for scenario in SCENARIOS}
    assert len(names) == 6


def test_source_only_fs_is_selected_when_fisher_gain_is_below_margin() -> None:
    value = load_config(ROOT, CONFIG)
    summary = {
        scenario: {
            "FM_source_only": _row(0.50, 0.28),
            "FS_source_only": _row(0.40, 0.25),
            "FS_empirical_fisher": _row(0.39, 0.249),
        }
        for scenario in SCENARIOS
    }
    result = decide(summary, value["decision"])
    assert result["selected_arm"] == "FS_source_only"
    assert result["fisher_removed"] is True
    assert result["fm_promoted"] is False


def test_fm_requires_lower_residual_in_both_scenarios() -> None:
    value = load_config(ROOT, CONFIG)
    summary = {
        "L8": {
            "FM_source_only": _row(0.35, 0.26),
            "FS_source_only": _row(0.40, 0.25),
            "FS_empirical_fisher": _row(0.39, 0.249),
        },
        "L4M4": {
            "FM_source_only": _row(0.41, 0.26),
            "FS_source_only": _row(0.40, 0.25),
            "FS_empirical_fisher": _row(0.39, 0.249),
        },
    }
    result = decide(summary, value["decision"])
    assert result["selected_arm"] == "FS_source_only"
    assert result["fm_promoted"] is False


def test_fm_can_be_promoted_only_under_frozen_rule() -> None:
    value = load_config(ROOT, CONFIG)
    summary = {
        scenario: {
            "FM_source_only": _row(0.35, 0.26),
            "FS_source_only": _row(0.40, 0.25),
            "FS_empirical_fisher": _row(0.39, 0.249),
        }
        for scenario in SCENARIOS
    }
    result = decide(summary, value["decision"])
    assert result["selected_arm"] == "FM_source_only"
    assert result["fisher_removed"] is True
    assert result["fm_promoted"] is True
