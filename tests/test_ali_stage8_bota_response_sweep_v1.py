from pathlib import Path

from src.bota_short_benchmark.ali_stage8_bota_response_sweep_v1 import SCALES, _candidate_id, _select, load_config


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return load_config(ROOT, ROOT / "configs/ali_stage8_bota_response_sweep_v1.yaml")


def test_frozen_response_grid_and_candidate_ids():
    assert SCALES == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert [_candidate_id(scale) for scale in SCALES] == ["bota_r0", "bota_r0.25", "bota_r0.5", "bota_r0.75", "bota_r1"]


def test_selection_is_joint_and_uses_frozen_tie_break():
    config = _config(); rows = []
    for scale in SCALES:
        for scenario in ("L8", "L4M4"):
            rows.append({"candidate_id": _candidate_id(scale), "scenario": scenario, "E_U": 0.1 if scale in (0.5,1.0) else 1.0, "utility_eligible": True, "prediction_collapse": False})
    result = _select(rows, config)
    assert result["selected"]["candidate_id"] == "bota_r1"
    assert result["selection_uses_finaltest"] is False


def test_no_eligible_calibration_retains_frozen_effective_scale():
    config = _config(); rows = []
    for scale in SCALES:
        for scenario in ("L8", "L4M4"):
            rows.append({"candidate_id": _candidate_id(scale), "scenario": scenario, "E_U": 0.0, "utility_eligible": False, "prediction_collapse": False})
    result = _select(rows, config)
    assert result["status"] == "NO_ELIGIBLE_CALIBRATION"
    assert result["selected"] is None
    assert result["effective_response_scale"] == 1.0
