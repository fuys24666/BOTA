import pytest
from src.bota_short_benchmark.ali_stage9_result_audit_v1 import stability
from src.bota_short_benchmark import ali_stage9_crossseed_development_v1 as stage9


def _rows():
    return [{"seed": seed, "scenario": scenario, "method": method, "E_U": .1 if method == "BOTA" else .2, "local_residual": .5, "utility_eligible": True, "prediction_collapse": False} for seed in stage9.REPORTED_SEEDS for scenario in stage9.SCENARIOS for method in stage9.METHODS]


def test_confirmation_recompute_excludes_selection_seed():
    rows = _rows()
    for row in rows:
        if row["seed"] == 42 and row["method"] == "BOTA": row.update(E_U=10, local_residual=10, utility_eligible=False)
    result = stability(rows)
    assert result["bota_utility_every_cell"]
    assert result["bota_mean_eu_below_each_baseline"]
    assert result["held_out_bota_mean_E_U"] == .1


def test_confirmation_recompute_rejects_duplicates():
    rows = _rows(); rows[-1] = dict(rows[0])
    with pytest.raises(ValueError, match="duplicates"): stability(rows)
